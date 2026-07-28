"""
exposure.py — Time Calculator: per-cell mask/grid clipping and the
dose/stage exposure-time estimate shared by all three workflow modes
(dose-time test, first exposure, second alignment).

Split out of ``tools/ebeam/calculator.py`` (mechanical refactor: moved
as-is, no behaviour change, no numbers changed). Self-contained like the
rest of the EBL calculator (see ``tools/ebeam/AGENTS.md``): imports
nothing outside ``tools/ebeam`` itself.
"""
from __future__ import annotations

import math

import numpy as np
import streamlit as st
import plotly.graph_objects as go

try:
    import gdstk
except ImportError:  # pragma: no cover
    gdstk = None

from tools.ebeam.gdsii.limits import tr
from tools.ebeam.plotting import _use_coverage_raster, _coverage_heatmap_trace

def _round_up_even(value: float) -> int:
    """Smallest even integer >= value, with a floor of 2."""
    n = max(1, math.ceil(value))
    return n if n % 2 == 0 else n + 1


def _show_outside_pattern_notice(
        message: str = "pattern outside of exposure grids detected") -> None:
    """Centered red pill warning."""
    st.markdown(
        '<div style="display: flex; justify-content: center; '
        'margin: 0.5em 0;">'
        '<span style="background-color: #d62728; color: white; '
        'padding: 6px 14px; border-radius: 999px; '
        'font-size: 0.95em; font-weight: 500;">'
        f'{message}'
        '</span></div>',
        unsafe_allow_html=True,
    )


# ─── Time Calculator helpers ─────────────────────────────────────────────────

def _polygon_clip_per_cell_mm(polys_mm: list, cells: list):
    """For each cell rect (xmin, ymin, xmax, ymax), clip `polys_mm` to
    the cell and return ``(areas, clipped_polys)`` where:

    - ``areas``: list of mm² per cell.
    - ``clipped_polys``: flat list of ``(xs, ys)`` clipped polygons in
      mm (across all cells), suitable for plotting only the portion of
      the mask that actually lands inside a grid.

    Uses a numpy bbox prefilter so we only run ``gdstk.boolean`` on
    polygons whose bounding box overlaps the cell.
    """
    n_cells = len(cells)
    if not polys_mm or not n_cells or gdstk is None:
        return [0.0] * n_cells, []

    # Pre-build polygon objects and per-polygon bbox arrays.
    gpolys = []
    bxmin_l, bxmax_l, bymin_l, bymax_l = [], [], [], []
    for xs, ys in polys_mm:
        if len(xs) < 3:
            continue
        gpolys.append(gdstk.Polygon(list(zip(xs, ys))))
        bxmin_l.append(min(xs)); bxmax_l.append(max(xs))
        bymin_l.append(min(ys)); bymax_l.append(max(ys))
    if not gpolys:
        return [0.0] * n_cells, []
    bxmin = np.asarray(bxmin_l); bxmax = np.asarray(bxmax_l)
    bymin = np.asarray(bymin_l); bymax = np.asarray(bymax_l)

    areas: list = []
    clipped_all: list = []
    # gdstk works in user units; choose a precision well below the
    # finest dimension we care about (10 nm = 1e-5 mm here).
    precision = 1e-7
    for xmin, ymin, xmax, ymax in cells:
        overlap = ((bxmax >= xmin) & (bxmin <= xmax)
                   & (bymax >= ymin) & (bymin <= ymax))
        idxs = np.flatnonzero(overlap)
        if idxs.size == 0:
            areas.append(0.0)
            continue
        candidates = [gpolys[i] for i in idxs]
        cell_rect = gdstk.rectangle((xmin, ymin), (xmax, ymax))
        try:
            inter = gdstk.boolean(
                candidates, [cell_rect], "and", precision=precision,
            )
        except Exception:
            areas.append(0.0)
            continue
        cell_area = 0.0
        for p in inter:
            cell_area += p.area()
            pts = p.points
            clipped_all.append(
                (pts[:, 0].tolist(), pts[:, 1].tolist())
            )
        areas.append(float(cell_area))
    return areas, clipped_all


def _time_input_fingerprint(prefix, polys_mm, cells, chip_size_mm, dotmap,
                            mark_cells) -> tuple:
    """Cheap signature of everything a Time Calculator result depends on.

    Used to invalidate ``{prefix}_time_result`` when the mask, the grid or
    the dose settings change.  Deliberately avoids hashing every polygon
    vertex — the mask store digest already identifies the loaded file, and
    the counts/extents catch the in-page geometry edits.
    """
    store = st.session_state.get("_ebc_gds_store") or {}
    dose_keys = (f"{prefix}_dose_us", f"{prefix}_dose_init_us",
                 f"{prefix}_dose_step_us", f"{prefix}_dose_ramp",
                 f"{prefix}_stage_s")
    return (
        store.get("digest"),
        st.session_state.get("ebc_gds_cell"),
        st.session_state.get("ebc_gds_layer"),
        len(polys_mm or ()), len(cells or ()),
        len(mark_cells or ()),
        round(float(chip_size_mm), 9), int(dotmap),
        tuple(round(float(v), 9) for v in (cells[0] if cells else ())),
        tuple(round(float(v), 9) for v in (cells[-1] if cells else ())),
        tuple(str(st.session_state.get(k)) for k in dose_keys),
    )


def _format_hms(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS.sss."""
    if seconds < 0 or not math.isfinite(seconds):
        return "—"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _render_time_calculator(prefix: str, polys_mm: list, cells: list,
                            chip_size_mm: float, dotmap: int,
                            dose_ramp: bool = False,
                            mark_cells: list | None = None,
                            mark_labels: list | None = None,
                            mark_positions: list | None = None,
                            extra_help_under_total: str | None = None,
                            disabled: bool = False,
                            disabled_reason: str | None = None,
                            precomputed_cell_areas: list | None = None,
                            coverage_layer=None, coverage_scale: float = 1.0,
                            coverage_ox: float = 0.0, coverage_oy: float = 0.0,
                            coverage_color: str = "#2ca02c") -> None:
    """Render the per-mode Time Calculator section.

    coverage_layer : optional ``_InstancedLayer`` whose low-res coverage
        raster is overlaid on the result plot (placed at
        ``coverage_ox/oy`` with ``coverage_scale``) — the same view shown
        in the GDS viewer and workflow plots.

    precomputed_cell_areas : optional list of mm² per cell (aligned to
        ``cells``). When supplied, the expensive per-cell gdstk clip is
        skipped — used for very dense layers where building millions of
        polygon objects would be infeasible. The mask-clipped overlay in
        the result plot is omitted in that case.

    polys_mm : mask polygons in global mm coordinates.
    cells    : list of (xmin, ymin, xmax, ymax) grid cells in mm.
    chip_size_mm : grid cell size in mm.
    dotmap   : EBL dotmap (dots per chip-size dimension).
    mark_cells : optional list of extra cells (for alignment marks in
        Second Alignment). Treated as additional active grids — each
        adds one stage movement, no exposure. Rendered in the same
        orange style as exposure grids; a parallel ``mark_labels`` is
        shown as hoverable circle markers on top.
    mark_labels : optional list of names parallel to ``mark_cells``
        for the hover tooltip on the circle markers.
    extra_help_under_total : optional caption rendered below the
        estimated time (e.g. \"+1-2 h for finding marks\").
    dose_ramp : if True (Dose Time Testing), replace the single dose
        input with `initial dose` and `incremental dose` inputs. Each
        grid gets dose = init + grid_index × step, in iteration order.
    """
    st.subheader(tr("Time Calculator", "曝光時間計算"))

    _TC_DEFAULTS = {
        f"{prefix}_dose_us": 2.0,
        f"{prefix}_dose_init_us": 2.0,
        f"{prefix}_dose_step_us": 0.2,
        f"{prefix}_stage_s": 15.0,
    }
    for _k, _v in _TC_DEFAULTS.items():
        st.session_state.setdefault(_k, _v)

    if dose_ramp:
        c_init, c_step, c_stage, c_btn = st.columns([2, 2, 2, 1])
        with c_init:
            st.number_input(
                tr("Initial dose (μs / dot)", "初始劑量 (μs / dot)"),
                min_value=0.0, step=0.01,
                format="%.3f", key=f"{prefix}_dose_init_us",
            )
        with c_step:
            st.number_input(
                tr("Incremental dose (μs / grid)", "增量劑量 (μs / grid)"),
                min_value=0.0, step=0.01,
                format="%.3f", key=f"{prefix}_dose_step_us",
            )
    else:
        c_dose, c_stage, c_btn = st.columns([2, 2, 1])
        with c_dose:
            st.number_input(
                tr("Dose time (μs / dot)", "劑量時間 (μs / dot)"),
                min_value=0.0, step=0.01,
                format="%.3f", key=f"{prefix}_dose_us",
            )
    with c_stage:
        st.number_input(
            tr("Stage movement time (s / grid)", "載台移動時間 (s / grid)"),
            min_value=0.0, step=0.1,
            format="%.2f", key=f"{prefix}_stage_s",
        )
    with c_btn:
        st.markdown("&nbsp;")  # vertical spacing to line up with inputs
        clicked = st.button(
            tr("Calculate time", "計算時間"), key=f"{prefix}_calc",
            type="primary", width="stretch",
            disabled=disabled,
            help=disabled_reason if disabled else None,
        )

    if disabled and disabled_reason:
        st.caption(f":red[⚠ {disabled_reason}]")

    if clicked and not disabled:
        if precomputed_cell_areas is not None:
            cell_areas = list(precomputed_cell_areas)
            clipped_polys = []
        else:
            with st.spinner(tr("Computing polygon area inside each grid…",
                               "正在計算各網格內的多邊形面積…")):
                cell_areas, clipped_polys = _polygon_clip_per_cell_mm(
                    polys_mm, cells)
        res_mm = chip_size_mm / dotmap if dotmap > 0 else 0.0
        res_area = res_mm * res_mm if res_mm > 0 else 1.0
        filled_per_cell = [a / res_area for a in cell_areas]
        filled_total = sum(filled_per_cell)
        active = sum(1 for a in cell_areas if a > 0)
        n_mark_cells = len(mark_cells) if mark_cells else 0
        stage_s = float(st.session_state[f"{prefix}_stage_s"])
        # Mark cells (e.g. SA alignment marks) add a stage movement
        # each — the beam has to travel there to find the mark — but
        # no exposure happens, so they're excluded from filled_total.
        stage_us = (active + n_mark_cells) * stage_s * 1e6

        if dose_ramp:
            init_us = float(st.session_state[f"{prefix}_dose_init_us"])
            step_us = float(st.session_state[f"{prefix}_dose_step_us"])
            # Per-grid dose follows the cell iteration order. Every
            # grid carries the same mask (DT replicates the pattern),
            # so the difference between grids is purely the dose ramp.
            per_grid_dose = [
                init_us + k * step_us for k in range(len(cell_areas))
            ]
            exposure_us = sum(
                f * d for f, d in zip(filled_per_cell, per_grid_dose)
            )
            active_doses = [
                per_grid_dose[k] for k, a in enumerate(cell_areas)
                if a > 0
            ]
            dose_min = min(active_doses) if active_doses else init_us
            dose_max = max(active_doses) if active_doses else init_us
            dose_us = None
        else:
            dose_us = float(st.session_state[f"{prefix}_dose_us"])
            per_grid_dose = None
            exposure_us = filled_total * dose_us
            dose_min = dose_max = dose_us
            init_us = step_us = None

        total_us = exposure_us + stage_us
        st.session_state[f"{prefix}_time_fp"] = _time_input_fingerprint(
            prefix, polys_mm, cells, chip_size_mm, dotmap, mark_cells)
        st.session_state[f"{prefix}_time_result"] = {
            "dose_ramp": dose_ramp,
            "cell_areas": cell_areas,
            "filled_per_cell": filled_per_cell,
            "cells": list(cells),
            "polys_mm": list(polys_mm),
            "clipped_polys": clipped_polys,
            "mark_cells": list(mark_cells) if mark_cells else [],
            "mark_labels": list(mark_labels) if mark_labels else [],
            "mark_positions": (
                list(mark_positions) if mark_positions else []),
            "n_mark_cells": n_mark_cells,
            "extra_help_under_total": extra_help_under_total,
            "chip_size_mm": chip_size_mm,
            "dotmap": dotmap,
            "res_mm": res_mm,
            "filled_total": filled_total,
            "active": active,
            "n_cells": len(cells),
            "dose_us": dose_us,
            "dose_init_us": init_us,
            "dose_step_us": step_us,
            "per_grid_dose": per_grid_dose,
            "dose_min": dose_min,
            "dose_max": dose_max,
            "stage_s": stage_s,
            "exposure_us": exposure_us,
            "stage_us": stage_us,
            "total_us": total_us,
        }

    # Drop a result computed for different inputs.  Nothing used to
    # invalidate it: uploading a different mask, removing the mask, or
    # changing the chip size / dotmap / selected cells left the previous
    # run's breakdown and total on screen looking current, so a dose/time
    # setting could be taken from the wrong mask.
    _fp_now = _time_input_fingerprint(prefix, polys_mm, cells, chip_size_mm,
                                      dotmap, mark_cells)
    if st.session_state.get(f"{prefix}_time_fp") != _fp_now:
        st.session_state.pop(f"{prefix}_time_result", None)
        st.session_state.pop(f"{prefix}_time_fp", None)

    result = st.session_state.get(f"{prefix}_time_result")
    if not result:
        return

    col_vals, col_plot = st.columns(2)

    with col_vals:
        st.markdown(f"**{tr('Breakdown', '明細')}**")
        st.write(
            f"{tr('Resolution', '解析度')}: **{result['res_mm'] * 1e6:.3f} nm** "
            f"({result['res_mm'] * 1000:g} μm) — "
            f"{tr('chip size / dotmap', '晶片尺寸 / dotmap')} = "
            f"{result['chip_size_mm'] * 1000:g} μm / {result['dotmap']}"
        )
        st.write(
            f"{tr('Filled resolution boxes', '已填入解析度方格數')}: "
            f"**{result['filled_total']:,.0f}**"
        )
        _n_marks = result.get("n_mark_cells", 0)
        # if _n_marks:
        #     st.write(
        #         f"Active grids: **{result['active'] + _n_marks}** "
        #         f"({result['active']} exposure + {_n_marks} alignment "
        #         "marks; empty exposure grids skipped)"
        #     )
        # else:
        st.write(
            f"{tr('Active grids', '有效網格數')}: "
            f"**{result['active'] + _n_marks} / "
            f"{result['n_cells'] + _n_marks}** "
            f"({tr('empty grids are skipped', '空白網格已略過')})"
        )
        if result.get("dose_ramp"):
            st.write(
                f"{tr('Dose ramp', '劑量遞增')}: "
                f"**{result['dose_init_us']:.3f} μs** "
                f"+ {result['dose_step_us']:.3f} μs/grid "
                f"→ {tr('active range', '有效範圍')} "
                f"**{result['dose_min']:.3f} – "
                f"{result['dose_max']:.3f} μs**"
            )
            st.write(
                f"{tr('Exposure', '曝光')}: "
                f"{tr('Σ(filled × per-grid dose)', 'Σ(填入量 × 各網格劑量)')} "
                f"= **{result['exposure_us'] / 1e6:,.3f} s**"
            )
        else:
            st.write(
                f"{tr('Exposure', '曝光')}: {result['filled_total']:,.0f} × "
                f"{result['dose_us']:.3f} μs "
                f"= **{result['exposure_us'] / 1e6:,.3f} s**"
            )
        st.write(
            f"{tr('Stage movement', '載台移動')}: "
            f"{result['active'] + _n_marks} × "
            f"{result['stage_s']:.2f} s "
            f"= **{result['stage_us'] / 1e6:,.3f} s**"
        )
        st.markdown(
            f"### {tr('Estimated Time', '預估時間')}: "
            f"`{_format_hms(result['total_us'] / 1e6)}`",
            help=(f"{result['total_us'] / 1e6:,.3f} "
                  + tr("seconds (hh:mm:ss.sss = hours:minutes:seconds)",
                       "秒（hh:mm:ss.sss = 時:分:秒）"))
        )
        if result.get("extra_help_under_total"):
            st.caption(result["extra_help_under_total"])

    with col_plot:
        fig = go.Figure()

        # Active grid cells (empty exposure cells hidden). Mark cells
        # are folded into the same trace because they are additional
        # active grids (stage moves there); they get a separate hover
        # marker below.
        ne_xs, ne_ys = [], []
        for cell, area in zip(result["cells"], result["cell_areas"]):
            if area <= 0:
                continue
            x0, y0, x1, y1 = cell
            ne_xs += [x0, x1, x1, x0, x0, None]
            ne_ys += [y0, y0, y1, y1, y0, None]
        _mark_cells_list = result.get("mark_cells", []) or []
        for cell in _mark_cells_list:
            x0, y0, x1, y1 = cell
            ne_xs += [x0, x1, x1, x0, x0, None]
            ne_ys += [y0, y0, y1, y1, y0, None]
        _total_active = result["active"] + result.get("n_mark_cells", 0)
        if ne_xs:
            fig.add_trace(go.Scatter(
                x=ne_xs, y=ne_ys, mode="lines",
                fill="toself",
                line=dict(color="#ff7f0e", width=1),
                fillcolor="rgba(255,127,14,0.20)",
                name=f"Active grids ({_total_active})",
                hoverinfo="skip",
            ))

        # Mask: for any layer too big (or too streamed) to draw exactly,
        # overlay the low-res coverage raster — the same view as the viewer
        # / workflow. Otherwise draw the mask clipped to the grids (the
        # portion that lands inside a cell, so what's plotted matches what
        # was counted in the area total).
        if coverage_layer is not None and _use_coverage_raster(
                coverage_layer):
            _cov = _coverage_heatmap_trace(
                coverage_layer, coverage_scale, coverage_ox, coverage_oy,
                coverage_color)
            if _cov is not None:
                fig.add_trace(_cov)
        else:
            mxs, mys = [], []
            for xs, ys in result.get("clipped_polys", []):
                if not xs:
                    continue
                mxs += list(xs) + [xs[0], None]
                mys += list(ys) + [ys[0], None]
            if mxs:
                fig.add_trace(go.Scatter(
                    x=mxs, y=mys, mode="lines", fill="toself",
                    line=dict(color="#2ca02c", width=0.5),
                    fillcolor="rgba(44,160,44,0.4)",
                    name=tr("Mask", "遮罩"),
                    hoverinfo="skip",
                ))

        # Alignment-mark circle markers — drawn from ``mark_positions``
        # directly so all selected marks are always shown, regardless
        # of whether they share a cell with a pattern (the previous
        # gate on ``mark_cells`` skipped marks that landed in pattern
        # cells, which is the common case).
        _mark_pos = result.get("mark_positions", []) or []
        if _mark_pos:
            _mark_cx = [p[0] for p in _mark_pos]
            _mark_cy = [p[1] for p in _mark_pos]
            _mark_lbls = result.get("mark_labels", []) or [
                f"M{i + 1}" for i in range(len(_mark_pos))
            ]
            fig.add_trace(go.Scatter(
                x=_mark_cx, y=_mark_cy,
                mode="markers+text",
                marker=dict(size=14, color="#9467bd",
                            symbol="circle-open",
                            line=dict(width=2)),
                text=_mark_lbls,
                textposition="top right",
                hovertemplate=("%{text}<br>"
                               "(%{x:.4f}, %{y:.4f}) mm"
                               "<extra></extra>"),
                name="Alignment marks",
            ))

        fig.update_layout(
            xaxis=dict(title="x (mm)"),
            yaxis=dict(title="y (mm)",
                       scaleanchor="x", scaleratio=1),
            margin=dict(l=40, r=20, t=20, b=40),
            height=500,
            showlegend=True,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, width="stretch",
                        key=f"{prefix}_tc_chart")

