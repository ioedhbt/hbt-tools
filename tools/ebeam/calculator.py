"""
ebeam_calculator.py — E-beam lithography position / dose calculator.

Helps the user map chip-corner positions in the e-beam holder, set the
left-computer origin, and run per-mode workflows (dose-time test,
first exposure, second alignment).

This is the page script Streamlit runs. The GDSII parsing/streaming
pipeline, plotting/coverage-raster helpers and the Time Calculator live in
``tools/ebeam/gdsii/``, ``tools/ebeam/plotting.py`` and
``tools/ebeam/exposure.py`` respectively — split out of what used to be one
6200-line file. See ``tools/ebeam/AGENTS.md`` for the one rule that survives
the split unchanged: this page (and everything under ``tools/ebeam/``)
imports nothing from the rest of the repo.

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.
"""
from __future__ import annotations

__version__ = "1.5"

import sys as _sys
from pathlib import Path as _Path

# Bootstrap sys.path so `streamlit run tools/ebeam/calculator.py` finds the
# `tools.ebeam.*` submodules below both through the portal (IOED_Tool_Web.py
# puts the repo root on sys.path already) and standalone
# (launch_ebl_calculator.py does not). Streamlit always executes this file
# as a script (module name "__main__"), so relative imports don't work here
# regardless. Walk up to the directory holding `tools/` rather than counting
# parents, so it survives this file being moved. Same pattern as
# tools/rf/ssm/agent_api.py.
_REPO_ROOT = next(
    (parent for parent in _Path(__file__).resolve().parents
     if (parent / "tools" / "__init__.py").exists()),
    _Path(__file__).resolve().parents[2])
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import gc
import numpy as np
import streamlit as st
import plotly.graph_objects as go

try:
    import gdstk
except ImportError:  # pragma: no cover
    gdstk = None

from tools.ebeam.gdsii.limits import _mask_budget_mb, _limits_for, _POLY_LIMIT
from tools.ebeam.gdsii.parser import _InstancedLayer
from tools.ebeam.gdsii.stream import (
    _compress_upload, _store_ratio, _load_gds, _load_gds_layers,
    _stream_scan, _stream_window,
)
from tools.ebeam.plotting import (
    _PALETTE, _StreamLayer, _layer_bbox_mm, _placed_bbox_mm, _layer_trace,
    _cell_areas_binned, _rasterize_coverage, _stream_coverage_grid,
    _unit_pattern_traces, _decimated_centers, _instances_in_window,
    _nan_xy_from_flat, _mask_overlay_traces, _dense_layer_note,
    _MAX_REGION_POLYS,
)
from tools.ebeam.exposure import (
    _render_time_calculator, _polygon_clip_per_cell_mm,
    _show_outside_pattern_notice, _round_up_even,
)


# ─── Standalone helpers (inlined so this file needs no repo modules) ──────────
# Static UI strings the tool needs. Kept here (instead of an external i18n
# registry) so ``ebeam_calculator.py`` is fully self-contained and can be run
# on its own via ``streamlit run ebeam_calculator.py``.
_EBL_TITLE = "EBL Calculator"
_EBL_DESC = "Compute JEOL ELS-7000 chip positions and exposure workflow."
_EBL_CORNER_GUIDE = "ℹ️ How corners are labeled"
_EBL_CORNER_NOTE = (
    "Holder frame: x increases →, y increases ↑. In Rectangular mode you "
    "edit one diagonal pair (BL–TR or BR–TL); the other pair is computed."
)


def _is_zh() -> bool:
    """Whether the shared portal UI language is 中文 (Traditional Chinese)."""
    return st.session_state.get("ui_lang") == "中文"


def tr(en: str, zh: str) -> str:
    """Return ``zh`` when the portal UI language is 中文, else ``en``.

    Inline mirror of ``tools.i18n.tr()`` — kept local (no repo import) so
    this file stays fully self-contained per the module docstring above and
    keeps working when copied out on its own via ``launch_ebl_calculator.py``.
    Reads the same ``st.session_state["ui_lang"]`` key the portal's language
    toggle writes, so this page follows it automatically when embedded, and
    defaults to English when run standalone (no toggle present).
    """
    return zh if _is_zh() else en


def segmented_radio(label, options, *, index: int = 0, key=None,
                    horizontal: bool = True, format_func=None, help=None,
                    label_visibility: str = "visible",
                    disabled: bool = False):
    """Drop-in replacement for :func:`st.radio` that renders as a segmented
    button group (``st.segmented_control``) instead of radio circles.

    Returns the *selected option value* (not its label), matching ``st.radio``.
    A selection is always guaranteed: ``st.segmented_control`` can otherwise
    return ``None`` when the user clicks the active chip to deselect it, so we
    fall back to the default value in that case. Falls back to ``st.radio`` on
    Streamlit builds without ``segmented_control`` (added in 1.40).
    """
    options = list(options)
    seg = getattr(st, "segmented_control", None)
    if seg is None or not options:
        return st.radio(label, options, index=index, key=key,
                        horizontal=horizontal,
                        format_func=(format_func or str),
                        help=help, label_visibility=label_visibility,
                        disabled=disabled)

    idx = index if 0 <= index < len(options) else 0

    kwargs = dict(options=options, key=key, help=help,
                  label_visibility=label_visibility, disabled=disabled,
                  selection_mode="single")
    if format_func is not None:
        kwargs["format_func"] = format_func

    if key is not None:
        # Seed / repair the value BEFORE the widget exists (allowed). Covers
        # first render (no value yet) and a stale value whose option list has
        # changed; both resolve to the default option.
        if st.session_state.get(key) not in options:
            st.session_state[key] = options[idx]
        # `key` already carries the value → must NOT also pass `default`.
    else:
        kwargs["default"] = options[idx]

    picked = seg(label, **kwargs)
    # Deselect (user clicked the active chip) → segmented_control returns None.
    if picked is None:
        picked = options[idx]
    return picked


# ─── Page header ─────────────────────────────────────────────────────────────

st.set_page_config(page_title=_EBL_TITLE, layout="wide", page_icon="🧮")

# Gray fill for secondary buttons so they read as buttons.  Inline copy of the
# portal-wide block in IOED_Tool_Web.py (this file stays repo-import-free so
# launch_ebl_calculator.py can run it standalone) — keep the two in sync with
# tools/ui_theme.py (master copy for the portal).
st.markdown(
    """
    <style>
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-secondaryFormSubmit"] {
        background-color: #E9EDF3;
    }
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
        background-color: #DDE3EB;
    }
    button[data-testid="stBaseButton-secondary"]:active,
    button[data-testid="stBaseButton-secondaryFormSubmit"]:active {
        background-color: #D1D8E2;
    }
    button[data-testid="stBaseButton-secondary"]:disabled,
    button[data-testid="stBaseButton-secondaryFormSubmit"]:disabled {
        background-color: #F1F3F7;
    }
    /* Help-hint circle — inline copy of .hbt-help from tools/ui_theme.py */
    .hbt-help {
        display: inline-flex; align-items: center; justify-content: center;
        width: 1.05em; height: 1.05em; margin-left: 0.35em;
        border: 1px solid #94A3B8; border-radius: 50%;
        color: #64748B; font-size: 0.72em; font-weight: 700;
        line-height: 1; cursor: help; vertical-align: 15%;
        user-select: none;
    }
    .hbt-help:hover { color: #1F2933; border-color: #64748B; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(tr(_EBL_TITLE, "電子束微影計算機"))
st.caption(tr(_EBL_DESC, "計算 JEOL ELS-7000 晶片位置與曝光流程。"))

# ─── Session-state defaults ──────────────────────────────────────────────────
# Corner defaults:
#   bottom-left  = (104, 114)
#   top-right    = (116, 126)
#   top-left     = (bl.x, tr.y) = (104, 126)
#   bottom-right = (tr.x, bl.y) = (116, 114)

_DEFAULTS = {
    "ebc_bl_x": 104.0, "ebc_bl_y": 114.4,
    "ebc_tr_x": 116.0, "ebc_tr_y": 126.4,
    "ebc_tl_x": 104.0, "ebc_tl_y": 126.4,
    "ebc_br_x": 116.0, "ebc_br_y": 114.4,
    "ebc_origin_x": 10.0, "ebc_origin_y": 10.0,
    "ebc_chip_size": 600, "ebc_dotmap": 60000,
    "ebc_resolution": 0.01,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


@st.cache_data(show_spinner=False)
def _chip_corner_guide_png(zh: bool = False) -> bytes:
    """Static reference sketch explaining the chip-corner labelling + the two
    diagonals that the "Editable diagonal" radio chooses between.  Drawn with
    matplotlib (Agg, no pyplot global state) and cached per language — the
    geometry never changes, only the legend text (``zh`` is part of the
    cache key so switching the portal language redraws it)."""
    import io
    from matplotlib.figure import Figure

    _diag_word = "對角線" if zh else "diagonal"

    BLUE, ORANGE = "#1f77b4", "#ff7f0e"
    x0, x1, y0, y1 = 0.30, 0.82, 0.30, 0.82
    fig = Figure(figsize=(4.2, 4.0), dpi=130)
    ax = fig.add_subplot(111)

    # Chip body.
    ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color=BLUE, alpha=0.10, zorder=1)
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color=BLUE, lw=2.0, zorder=3)
    # The two diagonals the radio toggles between.
    ax.plot([x0, x1], [y0, y1], ls="--", color=BLUE,   lw=1.6, zorder=2,
            label=f"BL–TR {_diag_word}")
    ax.plot([x1, x0], [y0, y1], ls="--", color=ORANGE, lw=1.6, zorder=2,
            label=f"BR–TL {_diag_word}")

    corners = {
        "BL": (x0, y0, -0.02, -0.02, "right", "top"),
        "BR": (x1, y0, +0.02, -0.02, "left",  "top"),
        "TR": (x1, y1, +0.02, +0.02, "left",  "bottom"),
        "TL": (x0, y1, -0.02, +0.02, "right", "bottom"),
    }
    for name, (cx, cy, dx, dy, ha, va) in corners.items():
        ax.plot(cx, cy, "o", color=BLUE, ms=8, zorder=4)
        ax.annotate(name, (cx + dx, cy + dy), ha=ha, va=va,
                    fontsize=10, fontweight="bold", color=BLUE)

    # Holder coordinate axes.
    ax.annotate("", xy=(0.97, 0.10), xytext=(0.16, 0.10),
                arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.text(0.97, 0.055, "x (mm)", ha="right", va="top", fontsize=8, color="0.4")
    ax.annotate("", xy=(0.10, 0.97), xytext=(0.10, 0.16),
                arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.text(0.055, 0.97, "y (mm)", rotation=90, ha="right", va="top",
            fontsize=8, color="0.4")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=2, fontsize=7.5, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _colored_num(value, default: float, fmt: str = "%.3f",
                  eps: float = 1e-6) -> str:
    """Format ``value`` as Markdown colored text for Setup Instructions:
    green when it still matches ``default`` (the calculator's own
    textbook default), blue when it has moved away from that default —
    the signal that this number differs from the generic instructions and
    needs a second look before typing it into the JEOL software."""
    color = "green" if abs(float(value) - float(default)) <= eps else "blue"
    return f":{color}[{fmt % float(value)}]"


def render_page() -> None:
    """Render the EBL Calculator page body: Sections 1-4 (chip
    position, left-computer setup, GDS mask viewer, mode-selector
    workflows). Wrapped in a function (was straight-line module-level
    script) so the page is importable/testable without executing on
    import; called once at the bottom of this file.
    """
    # ─── Section 1: Chip Position in the E-beam Holder ───────────────────────────
    with st.container(border=True):
        st.header(tr("Chip Position in the E-beam Holder", "夾具中的晶片位置"),
                  help=tr("Check the positions of the chip corners - especially "
                          "the bottom-left and top-right.",
                          "請確認晶片角落位置 — 特別是左下角與右上角。"),
                  anchor=False)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader(tr("Corner Positions", "角落位置"))

            # with st.expander(tr(_EBL_CORNER_GUIDE, "ℹ️ 角落標示說明"), expanded=False):
            #     st.image(_chip_corner_guide_png(zh=_is_zh()), width="stretch")
            #     st.caption(tr(_EBL_CORNER_NOTE,
            #                   "夾具座標：x 向右增加，y 向上增加。矩形模式下只需編輯一組"
            #                   "對角（BL–TR 或 BR–TL），另一組會自動計算。"))

            _shape_opts = [tr("Rectangular", "矩形"), tr("Custom", "自訂")]

            col_left_in, col_right_in = st.columns([1, 1])
            with col_left_in:  
                shape_mode = segmented_radio(
                    tr("Shape", "形狀"), _shape_opts,
                    key="ebc_shape_mode",
                )
                is_rect = (shape_mode == _shape_opts[0])
            with col_right_in:
                # In rectangular mode the user picks which diagonal pair drives the
                # rectangle; the other pair is computed from it and shown disabled.
                bltr_active = True  # default: BL/TR is the editable pair
                if is_rect:
                    # Corner codes (BL/TR/BR/TL) stay verbatim in both languages, so
                    # this option list needs no format_func / index indirection.
                    diag = segmented_radio(
                        tr("Editable diagonal", "可編輯對角"), ["BL / TR", "BR / TL"],
                        key="ebc_diag_mode",
                    )
                    bltr_active = (diag == "BL / TR")

                    if bltr_active:
                        st.session_state["ebc_tl_x"] = st.session_state["ebc_bl_x"]
                        st.session_state["ebc_tl_y"] = st.session_state["ebc_tr_y"]
                        st.session_state["ebc_br_x"] = st.session_state["ebc_tr_x"]
                        st.session_state["ebc_br_y"] = st.session_state["ebc_bl_y"]
                    else:
                        st.session_state["ebc_bl_x"] = st.session_state["ebc_tl_x"]
                        st.session_state["ebc_bl_y"] = st.session_state["ebc_br_y"]
                        st.session_state["ebc_tr_x"] = st.session_state["ebc_br_x"]
                        st.session_state["ebc_tr_y"] = st.session_state["ebc_tl_y"]

                disabled_bl = is_rect and not bltr_active
                disabled_tr = is_rect and not bltr_active
                disabled_tl = is_rect and bltr_active
                disabled_br = is_rect and bltr_active

            # Constraint floors: the EBL stage can't physically place the
            # chip below x = 90 mm or y = 110 mm. Beyond those, enforce
            # that right-side corners sit right of left-side corners and
            # top-side corners sit above bottom-side corners. We clamp
            # session state in place *before* the number_input widgets
            # render so they show the corrected value (Streamlit reads
            # widget values from session_state at render time).
            _FLOOR_X, _FLOOR_Y = 90.0, 110.0
            _CORNER_STEP = 0.0005
            for _k in ("ebc_bl_x", "ebc_tl_x", "ebc_br_x", "ebc_tr_x"):
                if float(st.session_state[_k]) < _FLOOR_X:
                    st.session_state[_k] = _FLOOR_X
            for _k in ("ebc_bl_y", "ebc_tl_y", "ebc_br_y", "ebc_tr_y"):
                if float(st.session_state[_k]) < _FLOOR_Y:
                    st.session_state[_k] = _FLOOR_Y
            _left_x_max = max(float(st.session_state["ebc_bl_x"]),
                              float(st.session_state["ebc_tl_x"]))
            for _k in ("ebc_br_x", "ebc_tr_x"):
                if float(st.session_state[_k]) <= _left_x_max:
                    st.session_state[_k] = _left_x_max + _CORNER_STEP
            _bot_y_max = max(float(st.session_state["ebc_bl_y"]),
                             float(st.session_state["ebc_br_y"]))
            for _k in ("ebc_tl_y", "ebc_tr_y"):
                if float(st.session_state[_k]) <= _bot_y_max:
                    st.session_state[_k] = _bot_y_max + _CORNER_STEP

            def _corner_inputs(label: str, kx: str, ky: str, disabled: bool):
                st.markdown(f"**{label}**")
                cx, cy = st.columns(2)
                cx.number_input(
                    "x", key=kx, format="%.3f", step=0.0005,
                    min_value=_FLOOR_X, disabled=disabled,
                )
                cy.number_input(
                    "y", key=ky, format="%.3f", step=0.0005,
                    min_value=_FLOOR_Y, disabled=disabled,
                )

            # 2×2 grid: top row = top-left, top-right; bottom row = bottom-left, bottom-right
            row_top = st.columns(2)
            with row_top[0]:
                with st.container(border=True):
                    _corner_inputs(f"{tr('Top Left', '左上')} (TL)",
                                   "ebc_tl_x", "ebc_tl_y", disabled=disabled_tl)
            with row_top[1]:
                with st.container(border=True):
                    _corner_inputs(f"{tr('Top Right', '右上')} (TR)",
                                   "ebc_tr_x", "ebc_tr_y", disabled=disabled_tr)

            row_bot = st.columns(2)
            with row_bot[0]:
                with st.container(border=True):
                    _corner_inputs(f"{tr('Bottom Left', '左下')} (BL)",
                                   "ebc_bl_x", "ebc_bl_y", disabled=disabled_bl)
            with row_bot[1]:
                with st.container(border=True):
                    _corner_inputs(f"{tr('Bottom Right', '右下')} (BR)",
                                   "ebc_br_x", "ebc_br_y", disabled=disabled_br)

        with col_right:
            st.subheader(tr("Chip Position", "晶片位置"))

            bl = (st.session_state["ebc_bl_x"], st.session_state["ebc_bl_y"])
            br = (st.session_state["ebc_br_x"], st.session_state["ebc_br_y"])
            c_tr = (st.session_state["ebc_tr_x"], st.session_state["ebc_tr_y"])
            tl = (st.session_state["ebc_tl_x"], st.session_state["ebc_tl_y"])

            # Loop back to bl to close the polygon for fill='toself'.
            poly_x = [bl[0], br[0], c_tr[0], tl[0], bl[0]]
            poly_y = [bl[1], br[1], c_tr[1], tl[1], bl[1]]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=poly_x, y=poly_y,
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.25)",
                hoverinfo="skip",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[bl[0], br[0], c_tr[0], tl[0]],
                y=[bl[1], br[1], c_tr[1], tl[1]],
                mode="markers+text",
                marker=dict(size=10, color="#1f77b4"),
                text=["BL", "BR", "TR", "TL"],
                textposition=["bottom center", "bottom center",
                            "top center", "top center"],
                hovertemplate="%{text}: (%{x:.3f}, %{y:.3f})<extra></extra>",
                showlegend=False,
            ))

            xs = [bl[0], br[0], c_tr[0], tl[0]]
            ys = [bl[1], br[1], c_tr[1], tl[1]]
            pad_x = max(1.0, (max(xs) - min(xs)) * 0.15)
            pad_y = max(1.0, (max(ys) - min(ys)) * 0.15)

            fig.update_layout(
                xaxis=dict(title="x (mm)",
                        range=[min(xs) - pad_x, max(xs) + pad_x]),
                yaxis=dict(title="y (mm)",
                        range=[min(ys) - pad_y, max(ys) + pad_y],
                        scaleanchor="x", scaleratio=1),
                margin=dict(l=40, r=20, t=20, b=40),
                height=400,
            )
            st.plotly_chart(fig, width="stretch")


    # ─── Section 2: Left Computer Setup ──────────────────────────────────────────
    with st.container(border=True):
        st.header(tr("Left Computer Setup", "左側電腦設定"))

        # Live values behind the placeholders in Setup Instructions below.
        # Read straight from session_state (same read-ahead pattern already
        # used for ebc_sa_o_mark1 further down this file) since Job 1/2's
        # own widgets render after this expander. The two global keys are
        # guaranteed present by the _DEFAULTS.setdefault() loop above;
        # per-mode keys only exist once that mode's block has run at least
        # once this session, so those fall back to that mode's own
        # documented default until then.
        _si_origin_x = float(st.session_state["ebc_origin_x"])
        _si_origin_y = float(st.session_state["ebc_origin_y"])
        _si_chip_size = int(st.session_state["ebc_chip_size"])
        _si_dotmap = int(st.session_state["ebc_dotmap"])
        _si_dxdy = _si_chip_size / 1000.0  # μm -> mm; FE/SA tile edge-to-edge
        _si_dt_cel_x = float(st.session_state.get("ebc_dt_cel_x", 9.7))
        _si_dt_cel_y = float(st.session_state.get("ebc_dt_cel_y", 9.7))
        _si_dt_dx = float(st.session_state.get("ebc_dt_dx", 0.6))
        _si_dt_dy = float(st.session_state.get("ebc_dt_dy", 0.6))
        _si_dt_nx = int(st.session_state.get("ebc_dt_nx", 5))
        _si_dt_ny = int(st.session_state.get("ebc_dt_ny", 5))
        _si_dt_shift_x = float(st.session_state.get("ebc_dt_shift_x", 98.8))
        _si_dt_shift_y = float(st.session_state.get("ebc_dt_shift_y", 109.2))
        _si_fe_cel_x = float(st.session_state.get("ebc_fe_cel_x", 9.7))
        _si_fe_cel_y = float(st.session_state.get("ebc_fe_cel_y", 9.7))
        _si_fe_nx = int(st.session_state.get("ebc_fe_nx", 20))
        _si_fe_ny = int(st.session_state.get("ebc_fe_ny", 20))
        _si_fe_shift_x = float(st.session_state.get("ebc_fe_shift_x", 94.3))
        _si_fe_shift_y = float(st.session_state.get("ebc_fe_shift_y", 104.7))
        _si_sa_cel_x = float(st.session_state.get("ebc_sa_sap_cel_x", 9.7))
        _si_sa_cel_y = float(st.session_state.get("ebc_sa_sap_cel_y", 9.7))
        _si_sa_nx = int(st.session_state.get("ebc_sa_sap_nx", 20))
        _si_sa_ny = int(st.session_state.get("ebc_sa_sap_ny", 20))
        _si_sa_shift_x = float(st.session_state.get("ebc_sa_o_shift_x", -9.7))
        _si_sa_shift_y = float(st.session_state.get("ebc_sa_o_shift_y", -9.7))
        # Dose Time Testing's Time Calculator (prefix "ebc_dt", dose_ramp=True
        # in _render_time_calculator) exposes these two — defaults match
        # exposure.py's _TC_DEFAULTS.
        _si_dt_dose_init = float(st.session_state.get("ebc_dt_dose_init_us", 2.0))
        _si_dt_dose_step = float(st.session_state.get("ebc_dt_dose_step_us", 0.2))

        with st.expander(tr("Setup Instructions", "設定說明"), expanded=False):
            st.caption(tr("Make sure you already have the `.cel` file. In `job1`: ",
                          "請確認已備妥 `.cel` 檔案。在 `job1` 中："))
            st.caption(tr("1. Type `pc`.", "1. 輸入 `pc`。"))
            st.caption(tr(
                "2. Select folder where the `.cel` file is, usually in `Desktop/IOED/hbt/your_folder`.",
                "2. 選擇 `.cel` 檔案所在資料夾，通常位於 `Desktop/IOED/hbt/your_folder`。"))
            st.caption(tr("3. Type the chip name, the same name as the `.cel` file.",
                          "3. 輸入晶片名稱，須與 `.cel` 檔案名稱相同。"))
            st.caption(tr(
                f"4. Set chip origin, {_colored_num(_si_origin_x, 10.0)}, "
                f"{_colored_num(_si_origin_y, 10.0)}. Using `0, 0` is difficult to see.",
                f"4. 設定 chip origin， {_colored_num(_si_origin_x, 10.0)}, "
                f"{_colored_num(_si_origin_y, 10.0)}。使用 `0, 0` 較難以辨識。"))
            st.caption(tr("5. Click `Ax: chip dot` (white), it will be changed to `Ax: stage (mm)` (green).",
                          "5. 點擊 `Ax: chip dot`（白色），會變為 `Ax: stage (mm)`（綠色）。"))
            st.caption(tr("6. Type `0.0001g` to set grid spacing to 100 nm.",
                          "6. 輸入 `0.0001g` 將網格間距設為 100 nm。"))
            with st.expander(tr("Dose Time Testing", "劑量時間測試"), expanded=False):
                st.caption(tr("7. Click esc button to open menu, File -> Load CEL. Enter cel name.", "7. 點擊 esc 按鈕打開選單，然後點擊 File -> Load CEL，輸入 cel 名稱。"))
                st.caption(tr(
                    f"8. Origin: {_colored_num(_si_dt_cel_x, 9.7)}, "
                    f"{_colored_num(_si_dt_cel_y, 9.7)}.",
                    f"8. 原點：{_colored_num(_si_dt_cel_x, 9.7)}, "
                    f"{_colored_num(_si_dt_cel_y, 9.7)}。"))
                # st.caption(f"**{tr('Values used:', '使用數值：')}**")
                # st.caption(
                #     f"{tr('Job 1', 'Job 1')} — "
                #     f"{tr('chip origin x, y', '晶片原點 x, y')}: "
                #     f"{_colored_num(_si_origin_x, 10.0)}, {_colored_num(_si_origin_y, 10.0)}; "
                #     f"{tr('cel origin x, y', 'cel 原點 x, y')}: "
                #     f"{_colored_num(_si_dt_cel_x, 9.7)}, {_colored_num(_si_dt_cel_y, 9.7)}")
                # st.caption(
                #     f"{tr('Job 2', 'Job 2')} — "
                #     f"{tr('chip size, dotmap', '晶片尺寸、點陣圖')}: "
                #     f"{_colored_num(_si_chip_size, 600, '%.0f')}, "
                #     f"{_colored_num(_si_dotmap, 60000, '%.0f')}")

            with st.expander(tr("First Exposure", "首次曝光"), expanded=False):
                st.caption(tr("7. Type `mc` to create grid points.",
                              "7. 輸入 `mc` 建立網格點。"))
                st.caption(tr(
                    "8. Click the square grid (click `i` to zoom in, and `o` to zoom out, then click the screen with the mouse pointer if needed).",
                    "8. 點擊方形網格（可按 `i` 放大、`o` 縮小，需要時再用滑鼠點擊畫面）。"))
                st.caption(tr("9. Are you sure? -> `Y`, All `cel_name`? -> `N`.",
                              "9. 確定嗎？-> `Y`，全部 `cel_name`？-> `N`。"))
                st.caption(tr(
                    f"10. dx, dy: {_colored_num(_si_dxdy, 0.6)}, "
                    f"{_colored_num(_si_dxdy, 0.6)}. This is the grid distance from each other.",
                    f"10. dx, dy：{_colored_num(_si_dxdy, 0.6)}, "
                    f"{_colored_num(_si_dxdy, 0.6)}。這是網格之間的間距。"))
                st.caption(tr(
                    f"11. Nx, Ny -> {_colored_num(_si_fe_nx, 20, '%.0f')}, "
                    f"{_colored_num(_si_fe_ny, 20, '%.0f')}, depending on the size of the pattern.",
                    f"11. Nx, Ny -> {_colored_num(_si_fe_nx, 20, '%.0f')}, "
                    f"{_colored_num(_si_fe_ny, 20, '%.0f')}，依圖案大小而定。"))
                st.caption(tr("12. X direction? `Y` -> Auto reverse? `N`",
                              "12. X 方向？`Y` -> 自動反轉？`N`"))
                st.caption(tr("13. Click esc button to open menu, click File -> Load CEL. Enter cel name.",
                              "13. 點擊 esc 按鈕打開選單，然後點擊 File -> Load CEL，輸入 cel 名稱。"))
                st.caption(tr(
                    f"14. Origin: {_colored_num(_si_fe_cel_x, 9.7)}, "
                    f"{_colored_num(_si_fe_cel_y, 9.7)}.",
                    f"14. 原點：{_colored_num(_si_fe_cel_x, 9.7)}, "
                    f"{_colored_num(_si_fe_cel_y, 9.7)}。"))
                # st.caption(f"**{tr('Values used:', '使用數值：')}**")
                # st.caption(
                #     f"{tr('Job 1', 'Job 1')} — "
                #     f"{tr('chip origin x, y', '晶片原點 x, y')}: "
                #     f"{_colored_num(_si_origin_x, 10.0)}, {_colored_num(_si_origin_y, 10.0)}; "
                #     f"{tr('cel origin x, y', 'cel 原點 x, y')}: "
                #     f"{_colored_num(_si_fe_cel_x, 9.7)}, {_colored_num(_si_fe_cel_y, 9.7)}; "
                #     f"Nx, Ny: {_colored_num(_si_fe_nx, 20, '%.0f')}, "
                #     f"{_colored_num(_si_fe_ny, 20, '%.0f')}")
                # st.caption(
                #     f"{tr('Job 2', 'Job 2')} — "
                #     f"{tr('chip size, dotmap', '晶片尺寸、點陣圖')}: "
                #     f"{_colored_num(_si_chip_size, 600, '%.0f')}, "
                #     f"{_colored_num(_si_dotmap, 60000, '%.0f')}")

            with st.expander(tr("Second Alignment", "二次對準"), expanded=False):
                st.caption(tr("7. Type `mc` to create grid points.",
                              "7. 輸入 `mc` 建立網格點。"))
                st.caption(tr(
                    "8. Click the square grid (click `i` to zoom in, and `o` to zoom out, then click the screen with the mouse pointer if needed).",
                    "8. 點擊方形網格（可按 `i` 放大、`o` 縮小，需要時再用滑鼠點擊畫面）。"))
                st.caption(tr("9. Are you sure? -> `Y`, All `cel_name`? -> `N`.",
                              "9. 確定嗎？-> `Y`，全部 `cel_name`？-> `N`。"))
                st.caption(tr(
                    f"10. dx, dy: {_colored_num(_si_dxdy, 0.6)}, "
                    f"{_colored_num(_si_dxdy, 0.6)}. This is the grid distance from each other.",
                    f"10. dx, dy：{_colored_num(_si_dxdy, 0.6)}, "
                    f"{_colored_num(_si_dxdy, 0.6)}。這是網格之間的間距。"))
                st.caption(tr(
                    f"11. Nx, Ny -> {_colored_num(_si_sa_nx, 20, '%.0f')}, "
                    f"{_colored_num(_si_sa_ny, 20, '%.0f')}, depending on the size of the pattern.",
                    f"11. Nx, Ny -> {_colored_num(_si_sa_nx, 20, '%.0f')}, "
                    f"{_colored_num(_si_sa_ny, 20, '%.0f')}，依圖案大小而定。"))
                st.caption(tr("12. X direction? `Y` -> Auto reverse? `N`",
                              "12. X 方向？`Y` -> 自動反轉？`N`"))
                st.caption(tr("13. Click esc button to open menu, then click File -> Load CEL. Enter cel name.",
                              "13. 點擊 esc 按鈕打開選單，然後點擊 File -> Load CEL，輸入 cel 名稱。"))
                st.caption(tr(
                    f"14. Origin: {_colored_num(_si_sa_cel_x, 9.7)}, "
                    f"{_colored_num(_si_sa_cel_y, 9.7)}.",
                    f"14. 原點：{_colored_num(_si_sa_cel_x, 9.7)}, "
                    f"{_colored_num(_si_sa_cel_y, 9.7)}。"))
                st.caption(tr(
                    "15. Click Menu -> Chip -> Reg-2 Mark (R2). Input the positions for the 2 marks.",
                    "15. 點擊 Menu -> Chip -> Reg-2 Mark (R2)，輸入 2 個標記的位置。"))
                # st.caption(f"**{tr('Values used:', '使用數值：')}**")
                # st.caption(
                #     f"{tr('Job 1', 'Job 1')} — "
                #     f"{tr('chip origin x, y', '晶片原點 x, y')}: "
                #     f"{_colored_num(_si_origin_x, 10.0)}, {_colored_num(_si_origin_y, 10.0)}; "
                #     f"{tr('cel origin x, y', 'cel 原點 x, y')}: "
                #     f"{_colored_num(_si_sa_cel_x, 9.7)}, {_colored_num(_si_sa_cel_y, 9.7)}; "
                #     f"Nx, Ny: {_colored_num(_si_sa_nx, 20, '%.0f')}, "
                #     f"{_colored_num(_si_sa_ny, 20, '%.0f')}")
                # st.caption(
                #     f"{tr('Job 2', 'Job 2')} — "
                #     f"{tr('chip size, dotmap', '晶片尺寸、點陣圖')}: "
                #     f"{_colored_num(_si_chip_size, 600, '%.0f')}, "
                #     f"{_colored_num(_si_dotmap, 60000, '%.0f')}")
                # st.caption(
                #     f"{tr('Job 3', 'Job 3')} — "
                #     f"{tr('shift x, y', '位移 x, y')}: "

            st.caption(tr(
                "Click esc button to open menu, then click File -> save -> press enter. Type the file `.con` name, the same as the `.cel` file.",
                "點擊 esc 按鈕打開選單，然後點擊 File -> save -> 按 Enter。輸入 `.con` 檔名，須與 `.cel` 檔案名稱相同。"))
            st.caption(tr(
                "If successful, the grids will be green, your folder should have `.ccc, .cbc, .con` files.",
                "若成功，網格會變為綠色，資料夾中應會有 `.ccc, .cbc, .con` 檔案。"))
            st.caption(tr("Then, in job3:","然後, 在 job3:"))
            with st.expander(tr("Dose Time Testing", "劑量時間測試"), expanded=False):
                st.caption(tr("1. Type `x` (matrix schedule). Then input these values.", "1. 輸入 `x`（matrix schedule）。然後加入這些數值。"))
                st.caption(tr(
                    f"2. Init shift x, y, dose: {_colored_num(_si_dt_shift_x, 98.8)}, {_colored_num(_si_dt_shift_y, 109.2)}, {_colored_num(_si_dt_dose_init, 2.0)}",
                    f"2. Init shift x, y, dose: {_colored_num(_si_dt_shift_x, 98.8)}, {_colored_num(_si_dt_shift_y, 109.2)}, {_colored_num(_si_dt_dose_init, 2.0)}"))
                st.caption(tr(f"3. Init shift modx, mody: `0,0`", "3. Init shift modx, mody: `0,0`"))
                st.caption(tr(
                    f"4. Incremental x, y, dose: {_colored_num(_si_dt_dx, 0.6)}, {_colored_num(_si_dt_dy, 0.6)}, {_colored_num(_si_dt_dose_step, 0.2)}",
                    f"4. Incremental x, y, dose: {_colored_num(_si_dt_dx, 0.6)}, {_colored_num(_si_dt_dy, 0.6)}, {_colored_num(_si_dt_dose_step, 0.2)}"))
                st.caption(tr(f"5. Incremental modx, mody: `0,0`", "5. Incremental modx, mody: `0,0`"))
                st.caption(tr(f"6. Nx, Ny: {_colored_num(_si_dt_nx, 5, '%.0f')}, {_colored_num(_si_dt_ny, 5, '%.0f')}", f"6. Nx, Ny: {_colored_num(_si_dt_nx, 5, '%.0f')}, {_colored_num(_si_dt_ny, 5, '%.0f')}"))
                st.caption(tr("7. Focus shift value: `0`", "7. Focus shift value: `0`"))
                st.caption(tr("8. X Direction? (Y): `y`. Auto Reverse (Y): `n`", "8. X Direction? (Y): `y`. Auto Reverse (Y): `n`"))
                st.caption(tr("9. Schedule file name: (same as your file name). Then click enter.", "9. Schedule file name: (你的檔案名稱)。然後按 Enter。"))
                st.caption(tr("10. Modify the schedule file as needed with `i`. You can change the dose time on each grid, or move the grid position.", "10. 根據需要使用 `i` 修改 schedule 檔案。可以變更每個 grid 上的 dose time，或移動 grid 的位置。"))
                st.caption(tr("11. Click `e` (Exposure Execution) to check your pattern.","11. 點擊 `e` (Exposure Execution) 檢查您的模式。"))
                st.caption(tr("12. In the window, click `esc` button to open menu, then click `Disp Pat` (or click `f`) to display the pattern. Make sure they are positioned correctly. Close the window after finished checking.", "12. 在視窗中，按 Esc 鍵開啟選單，然後點選`Disp Pat`（按`f`）顯示圖案。確保圖案位置正確。檢查完畢後關閉視窗。"))
                st.caption(tr("13. Set up the exposure condition by clicking `c`. Go to `Z Move with Height Sensor`, click enter and fill the z-height from the laser. Usually the position of the center of the chip is `110,120`.","13. 點選`c`設定曝光條件。進入`Z Move with Height Sensor`選項，點選回車鍵並輸入雷射測量所得的Z軸高度值。晶片中心位置通常為`110,120`。"))
                st.caption(tr("14. Click `h` to toggle the height sensor off. Click `f` to do field correction.","14. 點擊 `h` 關閉高度感應器。點擊 `f` 進行 field correction。"))

                
            with st.expander(tr("First Exposure", "首次曝光"), expanded=False):
                st.caption(tr(
                    f"1. Type `i`. Put in your file name. Click the right arrow button → in your keyboard, type in the Position Shift (DX, DY): {_colored_num(_si_fe_shift_x, 94.3)}, {_colored_num(_si_fe_shift_y, 104.7)}. Click the right arrow button → again, then input your dose shift. Click esc when finished.",
                    f"1. 輸入 `i`。輸入檔案名稱。按鍵盤右箭頭 →，輸入 Position Shift (DX, DY): {_colored_num(_si_fe_shift_x, 94.3)}, {_colored_num(_si_fe_shift_y, 104.7)}. 按鍵盤右箭頭 → 再次，然後輸入 dose shift。按 esc 完成。"))
                st.caption(tr("2. Click `c` to set the exposure conditions. Go to `Z Move with Height Sensor`, click Enter and fill the z-height from the laser. Usually the position of the center of the chip is `110,120`.","13. 點選`c`設定曝光條件。進入`Z Move with Height Sensor`選項，按Enter並輸入雷射測量所得的Z軸高度值。晶片中心位置通常為`110,120`。"))
                st.caption(tr("3. Click `f` to do field correction.","14. 點擊 `f` 進行 field correction。"))
            with st.expander(tr("Second Alignment", "二次對準"), expanded=False):
                st.caption(tr(
                    f"1. Type `i`. Put in your file name. Click the right arrow button → in your keyboard, type in the Position Shift (DX, DY): {_colored_num(_si_sa_shift_x, -9.7)}, {_colored_num(_si_sa_shift_y, -9.7)}. Click the right arrow button → again, then input your dose shift. Click esc when finished.",
                    f"1. 輸入 `i`。輸入檔案名稱。按鍵盤右箭頭 →，輸入 Position Shift (DX, DY): {_colored_num(_si_sa_shift_x, -9.7)}, {_colored_num(_si_sa_shift_y, -9.7)}. 按鍵盤右箭頭 → 再次，然後輸入 dose shift。按 esc 完成。"))
                st.caption(tr("2. Click `c` to set the exposure conditions. Go to `Z Move with Height Sensor`, click Enter and fill the z-height from the laser. Usually the position of the center of the chip is `110,120`. Set `Registration Control` to ON.","13. 點選`c`設定曝光條件。進入`Z Move with Height Sensor`選項，按Enter並輸入雷射測量所得的Z軸高度值。晶片中心位置通常為`110,120`。將 `Registration Control` 設為 ON。"))
                st.caption(tr("3. Click `f` to do field correction.","14. 點擊 `f` 進行 field correction。"))
            st.caption(tr("After field correction is finished, click `e` to go to exposure execution. In the window, click `esc` button to open menu, then click `Disp Pat` (or click `f`) to display the pattern. Click `esc` button, and find `Exposure` to start exposure. Exposure start?: `y`.", "Field correction完成後，點擊 `e` 進行 exposure execution。在視窗中，按 Esc 鍵開啟選單，然後點選`Disp Pat`（按`f`）顯示圖案。按 Esc 鍵，然後找到`Exposure`開始曝光。Exposure start?: `y`."))
        c_oxy, c_csdm = st.columns(2)
        with c_oxy:
            with st.container(border=True):
                st.markdown(f"**{tr('In Job 1', 'Job 1 中')}**")
                c_ox, c_oy = st.columns(2)
                c_ox.number_input(tr("Chip Origin x (mm)", "晶片原點 x (mm)"),
                            key="ebc_origin_x",
                            format="%.3f", step=0.0005, min_value=0.0)
                c_oy.number_input(tr("Chip Origin y (mm)", "晶片原點 y (mm)"),
                            key="ebc_origin_y",
                            format="%.3f", step=0.0005, min_value=0.0)
        with c_csdm:
            with st.container(border=True):
                st.markdown(f"**{tr('In Job 2', 'Job 2 中')}**")
                c_cs, c_dm, c_res = st.columns(3)
                c_cs.selectbox(
                    tr("Chip Size (μm)", "晶片尺寸 (μm)"),
                    [75, 150, 300, 600, 1200],
                    key="ebc_chip_size",
                )
                c_dm.selectbox(
                    tr("Dotmap", "點陣圖"),
                    [20000, 60000, 240000],
                    key="ebc_dotmap",
                )
                # Resolution = chip size (μm) / dotmap. Stored in μm under
                # ebc_resolution; also shown in nm for readability.
                _chip_um = int(st.session_state["ebc_chip_size"])
                _dotmap = int(st.session_state["ebc_dotmap"])
                st.session_state["ebc_resolution"] = _chip_um / _dotmap
                _res_um = st.session_state["ebc_resolution"]
                c_res.markdown(f"**{tr('Resolution', '解析度')}**")
                c_res.markdown(
                    f"{_chip_um} μm / {_dotmap} {tr('dots', '點')} "
                    f"= {_res_um * 1000:g} nm")


    # ─── Section 3: GDS Mask Viewer ──────────────────────────────────────────────
    with st.container(border=True):
        st.header(tr("GDS Mask", "GDS 遮罩"))



        # Module-level outputs consumed by workflow modes:
        _gds_selected_polys: list = []  # list of (xs, ys) in GDS user units (e.g. µm)
        _gds_unit_to_mm: float = 1e-3   # multiplier from GDS user units to mm
        _gds_selected_token = None      # opaque ID of current (cell, layer) selection
        _gds_cells_data: dict = {}      # full {cell_name: {(l,d): polys}} from upload
        _gds_selected_cell_name = None  # currently selected top-cell name
        _gds_stream_summary = None      # set when the mask was streamed, not loaded

        if gdstk is None:
            st.warning(tr(
                "`gdstk` is not installed. Run `pip install gdstk` to enable "
                "GDS viewing.",
                "尚未安裝 `gdstk`。請執行 `pip install gdstk` 以啟用 GDS 檢視功能。"))
        else:
            cells_data: dict = {}
            parse_error: str | None = None

            col_upload, col_select = st.columns([1, 1])

            st.session_state.setdefault("ebc_gds_uploader_key", 0)

            with col_upload:
                gds_upload = st.file_uploader(
                    tr("Upload .gds file", "上傳 .gds 檔案"),
                    type=["gds"],
                    key=f"ebc_gds_upload_{st.session_state['ebc_gds_uploader_key']}",
                )
                if gds_upload is not None:
                    # Compress into the block store right away, then bump the
                    # key so the widget resets on rerun. The reset is what
                    # actually drops Streamlit's own raw copy from session
                    # state — skip it and the store is pure overhead on top of
                    # that copy, and stage 1 nets negative instead of positive.
                    st.session_state["_ebc_gds_store"] = _compress_upload(gds_upload)
                    st.session_state["ebc_gds_uploader_key"] += 1
                    gc.collect()
                    st.rerun()

                # How big a mask this machine can take *right now*. On a
                # workstation that is far more than on the deployed container,
                # and the parser's budgets follow this number.
                _budget_mb, _budget_note, _ = _mask_budget_mb()
                st.caption(tr(
                    f"Memory available for one mask: **{_budget_mb / 1024:.1f} GB** "
                    f"({_budget_note}). Bigger masks are refused with a message, "
                    "never a crash.",
                    f"單一遮罩可用記憶體：**{_budget_mb / 1024:.1f} GB**"
                    f"（{_budget_note}）。超出者會顯示訊息拒絕，不會當機。"))

                _gds_store = st.session_state.get("_ebc_gds_store")
                if _gds_store is not None:
                    # The uploader was just reset (see above), so its own
                    # filename display is gone — show the loaded state here.
                    _raw_mb = _gds_store["nbytes"] / 1e6
                    _packed_mb = sum(len(b) for b in _gds_store["blocks"]) / 1e6
                    _ratio = _store_ratio(_gds_store)
                    st.markdown(tr(
                        f"Loaded **{_gds_store['name']}** — {_raw_mb:,.0f} MB "
                        f"stored as {_packed_mb:,.0f} MB ({_ratio:.1f}x)",
                        f"已載入 **{_gds_store['name']}** — {_raw_mb:,.0f} MB "
                        f"壓縮為 {_packed_mb:,.0f} MB（{_ratio:.1f}x）"))
                    if st.button(tr("Remove mask", "移除遮罩"),
                                 key="ebc_gds_remove"):
                        st.session_state.pop("_ebc_gds_store", None)
                        _load_gds.clear()
                        gc.collect()
                        st.rerun()

                    try:
                        unit_m, cells_data = _load_gds_layers(_gds_store)
                        _gds_unit_to_mm = unit_m * 1000.0
                        _gds_cells_data = cells_data
                    except Exception as e:
                        # A budget refusal is not the end of the road: the file
                        # is too big to hold, but not too big to *measure*.
                        # Stream it instead — one 4 MB block at a time, keeping
                        # only the reductions — so the user still gets the
                        # overview and an exact time estimate rather than an
                        # error. Anything else (a corrupt file, a mid-parse
                        # MemoryError already translated by _load_gds_layers)
                        # still surfaces as a message.
                        try:
                            summary = _stream_scan(_gds_store)
                        except Exception:
                            summary = None
                        if summary is None or not summary.layers:
                            parse_error = str(e)
                        else:
                            _gds_stream_summary = summary
                            _gds_unit_to_mm = summary.unit_meters * 1000.0
                            cells_data = {
                                summary.top_name or "TOP": {
                                    key: _StreamLayer(summary, key, _gds_store)
                                    for key in sorted(summary.layers)
                                }
                            }
                            _gds_cells_data = cells_data
                            st.warning(tr(
                                f"This mask needs more memory than is free right "
                                f"now, so it was streamed instead of loaded: "
                                f"{e}",
                                f"此遮罩所需記憶體超過目前可用量，已改用串流掃描："
                                f"{e}"))

            with col_select:
                has_data = bool(cells_data)
                cell_names = list(cells_data.keys()) if has_data else []

                selected_cell = st.selectbox(
                    tr("Top Cell", "頂層元件 (Top Cell)"),
                    cell_names if has_data else ["—"],
                    key="ebc_gds_cell",
                    disabled=not has_data,
                )
                _gds_selected_cell_name = selected_cell if has_data else None

                by_layer = (
                    cells_data.get(selected_cell, {}) if has_data else {}
                )
                layer_keys = sorted(by_layer.keys())
                # NOTE: this label doubles as the dict key used for the lookup
                # below (``layer_options[selected_label]``) and is what
                # Streamlit persists under ``key="ebc_gds_layer"`` across
                # reruns, so it is intentionally kept English-only — swapping
                # it on a language toggle would desync the stored selection
                # from the freshly-rebuilt options dict. See `format_func`
                # elsewhere in this file for the pattern that avoids this.
                layer_options = {
                    f"L{l}/D{d}  ({len(by_layer[(l, d)]):,} polys)": (l, d)
                    for (l, d) in layer_keys
                }
                labels = list(layer_options.keys())

                selected_label = st.selectbox(
                    tr("Layer to expose", "要曝光的圖層"),
                    labels if labels else ["—"],
                    key="ebc_gds_layer",
                    disabled=not labels,
                )

            if parse_error:
                st.error(f"{tr('Failed to parse GDS', 'GDS 解析失敗')}: {parse_error}")
            elif _gds_store is not None and not cells_data:
                st.info(tr("No top-level cells found in this GDS.",
                           "此 GDS 檔案中找不到頂層元件。"))
            elif has_data and not by_layer:
                st.info(tr("Selected cell has no polygons.",
                           "所選元件不含任何多邊形。"))
            elif labels:
                layer_key = layer_options[selected_label]
                polys = by_layer[layer_key]
                _gds_selected_polys = polys
                _gds_selected_token = (selected_cell, layer_key)
                n_polys = len(polys)
                # Viewer plots are labelled in µm, so convert from GDS user
                # units explicitly — files where 1 user unit ≠ 1 µm (the PSO
                # masks use 1 m) otherwise display meter values under a µm
                # axis label.
                _gds_unit_to_um = _gds_unit_to_mm * 1000.0

                if isinstance(polys, _StreamLayer):
                    # Streamed layer: there is no geometry in memory to draw, so
                    # show the coverage raster the scan already produced plus the
                    # measured totals. Falling through to either branch below
                    # would try to iterate polygons that do not exist.
                    st.caption(_dense_layer_note(polys))
                    _s_bb = _layer_bbox_mm(polys, _gds_unit_to_mm)
                    m1, m2, m3 = st.columns(3)
                    m1.metric(tr("Placements", "放置次數"),
                              f"{polys.instance_count():,}")
                    m2.metric(tr("Polygons", "多邊形"), f"{len(polys):,}")
                    m3.metric(tr("Pattern area", "圖案面積"),
                              f"{polys.total_area_units2() * _gds_unit_to_mm ** 2:,.3f} mm²")
                    sfig = go.Figure()
                    _sras = _rasterize_coverage(polys, _gds_unit_to_mm, 0.0, 0.0,
                                                _PALETTE[0])
                    if _sras is not None:
                        _rgba, _rx0, _rdx, _ry0, _rdy = _sras
                        sfig.add_trace(go.Image(z=_rgba, x0=_rx0, dx=_rdx,
                                                y0=_ry0, dy=_rdy,
                                                hoverinfo="skip"))
                    # An image-only figure is not selectable, so scatter the
                    # lit raster pixels (capped) purely to make box-select fire;
                    # the region itself comes from the box geometry, not these
                    # points — same trick as the instanced branch above.
                    _sgrid = _stream_coverage_grid(polys, _gds_unit_to_mm, 0.0, 0.0)
                    if _sgrid is not None:
                        _scnt, _sx0, _sdx, _sy0, _sdy = _sgrid
                        _rows, _cols = np.nonzero(_scnt)
                        if _rows.size > 5000:
                            _pick = np.linspace(0, _rows.size - 1, 5000).astype(int)
                            _rows, _cols = _rows[_pick], _cols[_pick]
                        sfig.add_trace(go.Scattergl(
                            x=_sx0 + _cols * _sdx, y=_sy0 + _rows * _sdy,
                            mode="markers",
                            marker=dict(size=3, color=_PALETTE[0], opacity=0.35),
                            hoverinfo="skip", showlegend=False,
                        ))
                    sfig.update_layout(
                        xaxis=dict(title="x (mm)"),
                        yaxis=dict(title="y (mm)", scaleanchor="x", scaleratio=1),
                        margin=dict(l=40, r=20, t=20, b=40),
                        height=520, showlegend=False, dragmode="select",
                        plot_bgcolor="white",
                    )
                    _sevt = st.plotly_chart(
                        sfig, width="stretch", key="ebc_gds_stream_fp",
                        on_select="rerun", selection_mode="box",
                    )
                    if _s_bb is not None:
                        st.caption(tr(
                            f"Extent {(_s_bb[2] - _s_bb[0]):.3f} × "
                            f"{(_s_bb[3] - _s_bb[1]):.3f} mm. Drag a box to read "
                            "that region back off the compressed file at full "
                            "detail; double-click to clear.",
                            f"範圍 {(_s_bb[2] - _s_bb[0]):.3f} × "
                            f"{(_s_bb[3] - _s_bb[1]):.3f} mm。拖曳方框可從壓縮檔中"
                            "重新讀取該區域的完整細節；點兩下可清除。"))

                    # Box-select → re-read just that window. Only the blocks whose
                    # recorded bbox overlaps it get inflated, so inspecting a
                    # region of a multi-hundred-MB mask touches a few MB.
                    _sbox = None
                    try:
                        _sboxes = _sevt["selection"]["box"]
                        if _sboxes:
                            _sbox = _sboxes[-1]
                    except (KeyError, TypeError, IndexError):
                        _sbox = None
                    if _sbox is not None and _gds_unit_to_mm:
                        _sxr = sorted(_sbox["x"]); _syr = sorted(_sbox["y"])
                        # The plot is in mm; _stream_window works in user units.
                        _u = 1.0 / _gds_unit_to_mm
                        _sres = _stream_window(
                            polys.store, polys.summary,
                            _sxr[0] * _u, _sxr[1] * _u,
                            _syr[0] * _u, _syr[1] * _u, _MAX_REGION_POLYS)
                        if _sres is None:
                            st.info(tr("No patterns in the selected region.",
                                       "所選區域中沒有圖案。"))
                        elif isinstance(_sres[0], str):      # ("over", n_polys)
                            st.warning(tr(
                                f"Selected region holds {_sres[1]:,} polygons "
                                f"(> {_MAX_REGION_POLYS:,}). Select a smaller "
                                "region to inspect at full detail.",
                                f"所選區域含有 {_sres[1]:,} 個多邊形"
                                f"（> {_MAX_REGION_POLYS:,}）。請選擇較小的區域以檢視"
                                "完整細節。"))
                        else:
                            _wcx, _wcy, _wst = _sres
                            _wn = int(_wst.size - 1)
                            st.markdown("**" + tr(
                                f"Selected region (full detail — {_wn:,} polygons)",
                                f"所選區域（完整細節 — {_wn:,} 個多邊形）") + "**")
                            _wxs, _wys = _nan_xy_from_flat(
                                _wcx, _wcy, _wst, _gds_unit_to_mm, 0.0, 0.0)
                            wfig = go.Figure(go.Scatter(
                                x=_wxs, y=_wys, mode="lines", fill="toself",
                                line=dict(color=_PALETTE[0], width=0.8),
                                fillcolor=_hex_to_rgba(_PALETTE[0], 0.4),
                                hoverinfo="skip",
                            ))
                            wfig.update_layout(
                                xaxis=dict(title="x (mm)",
                                           range=[_sxr[0], _sxr[1]]),
                                yaxis=dict(title="y (mm)",
                                           range=[_syr[0], _syr[1]],
                                           scaleanchor="x", scaleratio=1),
                                margin=dict(l=40, r=20, t=20, b=40),
                                height=520, showlegend=False,
                                plot_bgcolor="white",
                            )
                            st.plotly_chart(wfig, width="stretch",
                                            key="ebc_gds_stream_win")
                elif isinstance(polys, _InstancedLayer):
                    # Repetitive layer: show the unit pattern zoomed (so the
                    # repeated shape is actually visible) beside the full array
                    # footprint with a decimated sample of real patterns.
                    st.caption(_dense_layer_note(polys))
                    col_unit, col_full = st.columns([4, 6])
                    with col_unit:
                        st.markdown(f"**{tr('Unit pattern (zoomed)', '單元圖案（放大）')}**")
                        ufig = go.Figure()
                        for _t in _unit_pattern_traces(polys, _PALETTE[2],
                                                       scale=_gds_unit_to_um):
                            ufig.add_trace(_t)
                        ufig.update_layout(
                            xaxis=dict(title="x (µm)"),
                            yaxis=dict(title="y (µm)",
                                       scaleanchor="x", scaleratio=1),
                            margin=dict(l=40, r=20, t=20, b=40),
                            height=550, showlegend=True, plot_bgcolor="white",
                        )
                        st.plotly_chart(ufig, width="stretch")
                    with col_full:
                        st.markdown(
                            f"**{tr('Array footprint', '陣列覆蓋範圍')}** "
                            + tr("(low-res overview) — drag a box to inspect "
                                 "that region below at full detail",
                                 "（低解析度總覽）— 拖曳方框可在下方檢視該區域的完整"
                                 "細節"))
                        fig = go.Figure()
                        _ras = _rasterize_coverage(
                            polys, _gds_unit_to_um, 0.0, 0.0, _PALETTE[0])
                        if _ras is not None:
                            _rgba, _rx0, _rdx, _ry0, _rdy = _ras
                            fig.add_trace(go.Image(
                                z=_rgba, x0=_rx0, dx=_rdx, y0=_ry0, dy=_rdy,
                                hoverinfo="skip",
                            ))
                        # A light, SELECTABLE scatter of decimated instance
                        # centers so box-select fires reliably on every drag
                        # (an image-only figure isn't selectable, which froze
                        # the selection). The detail region is taken from the
                        # box geometry, not these points.
                        _scx, _scy = _decimated_centers(
                            polys, _gds_unit_to_um, 0.0, 0.0)
                        fig.add_trace(go.Scattergl(
                            x=_scx, y=_scy, mode="markers",
                            marker=dict(size=3, color=_PALETTE[0], opacity=0.45),
                            hoverinfo="skip", showlegend=False,
                        ))
                        fig.update_xaxes(title="x (µm)", constrain="domain")
                        fig.update_yaxes(title="y (µm)", autorange=True,
                                         scaleanchor="x", scaleratio=1)
                        fig.update_layout(
                            margin=dict(l=40, r=20, t=20, b=40),
                            height=550, showlegend=False, dragmode="select",
                            plot_bgcolor="white",
                        )
                        _evt = st.plotly_chart(
                            fig, width="stretch", key="ebc_gds_fp",
                            on_select="rerun", selection_mode="box",
                        )
                        st.caption(tr("Double-click the plot to clear the selection.",
                                      "在圖上點兩下可清除選取範圍。"))

                    # Box-select → redraw that region with every polygon. Read the
                    # box geometry from the latest selection event.
                    _box = None
                    try:
                        _boxes = _evt["selection"]["box"]
                        if _boxes:
                            _box = _boxes[-1]
                    except (KeyError, TypeError, IndexError):
                        _box = None
                    if _box is not None:
                        _xr = sorted(_box["x"]); _yr = sorted(_box["y"])
                        _res = _instances_in_window(
                            polys, _gds_unit_to_um, 0.0, 0.0,
                            _xr[0], _xr[1], _yr[0], _yr[1])
                        if _res is None:
                            st.info(tr("No patterns in the selected region.",
                                       "所選區域中沒有圖案。"))
                        elif isinstance(_res[0], str):   # ("over", n_polys)
                            st.warning(tr(
                                f"Selected region holds {_res[1]:,} polygons "
                                f"(> {_MAX_REGION_POLYS:,}). Select a smaller "
                                "region to inspect at full detail.",
                                f"所選區域含有 {_res[1]:,} 個多邊形"
                                f"（> {_MAX_REGION_POLYS:,}）。請選擇較小的區域以檢視"
                                "完整細節。"
                            ))
                        else:
                            _cx, _cy, _starts = _res
                            _n_sel = int(_starts.size - 1)
                            st.markdown(
                                "**" + tr(
                                    f"Selected region (full detail — {_n_sel:,} polygons)",
                                    f"所選區域（完整細節 — {_n_sel:,} 個多邊形）")
                                + "**")
                            _xs, _ys = _nan_xy_from_flat(
                                _cx, _cy, _starts, 1.0, 0.0, 0.0)
                            rfig = go.Figure(go.Scatter(
                                x=_xs, y=_ys, mode="lines", fill="toself",
                                line=dict(color=_PALETTE[0], width=0.8),
                                fillcolor=_hex_to_rgba(_PALETTE[0], 0.4),
                                hoverinfo="skip",
                            ))
                            rfig.update_layout(
                                xaxis=dict(title="x (µm)",
                                           range=[_xr[0], _xr[1]]),
                                yaxis=dict(title="y (µm)", range=[_yr[0], _yr[1]],
                                           scaleanchor="x", scaleratio=1),
                                margin=dict(l=40, r=20, t=20, b=40),
                                height=600, showlegend=False, plot_bgcolor="white",
                            )
                            st.plotly_chart(rfig, width="stretch",
                                            key="ebc_gds_region")
                else:
                    fig = go.Figure()
                    _axis_unit = "µm"
                    _legend = True
                    _bg = None
                    if n_polys > _POLY_LIMIT:
                        # Dense, non-repetitive layer. Drawing every polygon
                        # would stall the browser — but showing *nothing* until
                        # the user opts in is worse, because the one thing that
                        # always fits is a low-res raster. So: raster by
                        # default, full detail behind the checkbox.
                        st.caption(_dense_layer_note(polys))
                        _fbb = _layer_bbox_mm(polys, _gds_unit_to_mm)
                        m1, m2, m3 = st.columns(3)
                        m1.metric(tr("Polygons", "多邊形"), f"{n_polys:,}")
                        # Vertices, not exposed area: `poly_areas()` builds
                        # several vertex-sized temporaries, and this branch is
                        # reached precisely when memory is scarce. The Time
                        # Calculator below reports the area anyway.
                        m2.metric(tr("Vertices", "頂點"),
                                  f"{int(polys.cx.size):,}")
                        m3.metric(tr("Extent", "範圍"),
                                  "—" if _fbb is None else
                                  f"{(_fbb[2] - _fbb[0]):.2f} × "
                                  f"{(_fbb[3] - _fbb[1]):.2f} mm")
                        render = st.checkbox(
                            tr("Render every polygon (may stall the browser)",
                               "繪製每一個多邊形（可能使瀏覽器停頓）"),
                            key="ebc_gds_force")
                        if render:
                            fig.add_trace(_layer_trace(
                                selected_label, polys, _PALETTE[0],
                                scale=_gds_unit_to_um))
                        else:
                            _fras = _rasterize_coverage(
                                polys, _gds_unit_to_mm, 0.0, 0.0, _PALETTE[0])
                            if _fras is not None:
                                _frgba, _frx0, _frdx, _fry0, _frdy = _fras
                                fig.add_trace(go.Image(
                                    z=_frgba, x0=_frx0, dx=_frdx,
                                    y0=_fry0, dy=_frdy, hoverinfo="skip"))
                            _axis_unit = "mm"
                            _legend = False
                            _bg = "white"      # matches the raster's own empty
                                               # pixels, in either theme
                    else:
                        fig.add_trace(_layer_trace(
                            selected_label, polys, _PALETTE[0],
                            scale=_gds_unit_to_um))
                    fig.update_layout(
                        xaxis=dict(title=f"x ({_axis_unit})"),
                        yaxis=dict(title=f"y ({_axis_unit})",
                                scaleanchor="x", scaleratio=1),
                        margin=dict(l=40, r=20, t=20, b=40),
                        height=550,
                        showlegend=_legend,
                        legend=dict(itemsizing="constant"),
                    )
                    if _bg:
                        fig.update_layout(plot_bgcolor=_bg)
                    st.plotly_chart(fig, width="stretch")


    # ─── Section 4: Mode selector ────────────────────────────────────────────────
    with st.container(border=True):
        st.header(tr("Workflow", "曝光流程"))

        # Option text is compared against below (mode == _MODE_OPTS[...]), so the
        # comparisons use the same tr()-built list rather than re-translating the
        # literal — keeps display and logic in sync when the language toggles.
        _MODE_OPTS = [
            tr("Choose mode:", "選擇模式："),
            tr("Dose Time Testing", "劑量時間測試"),
            tr("First Exposure", "首次曝光"),
            tr("Second Alignment", "二次對準"),
        ]
        mode = segmented_radio(
            tr("Mode", "模式"),
            _MODE_OPTS,
            key="ebc_mode",
        )

        # On every workflow-mode change, invalidate the per-mode shift tokens
        # so the newly-active mode re-runs its auto-recenter calculation. Each
        # mode already stores its shift under its own key (ebc_dt_shift_x /
        # ebc_fe_shift_x / ebc_sa_o_shift_x), so resetting only the token
        # triggers a fresh recompute without poking another mode's state.
        _prev_mode = st.session_state.get("_ebc_prev_mode")
        if _prev_mode != mode:
            for _k in ("ebc_dt_shift_token", "ebc_fe_shift_token",
                    "ebc_sa_o_shift_token"):
                st.session_state.pop(_k, None)
            st.session_state["_ebc_prev_mode"] = mode

        if mode == _MODE_OPTS[1]:  # "Dose Time Testing"

            _DT_DEFAULTS = {
                "ebc_dt_cel_x": 9.7, "ebc_dt_cel_y": 9.7,
                "ebc_dt_dx": 0.6, "ebc_dt_dy": 0.6,
                "ebc_dt_nx": 5, "ebc_dt_ny": 5,
                "ebc_dt_shift_x": 94.0, "ebc_dt_shift_y": 104.0,
            }
            for _k, _v in _DT_DEFAULTS.items():
                st.session_state.setdefault(_k, _v)

            # ebc_chip_size is stored in μm (selectbox); convert to mm for the
            # rest of the math, which works in mm.
            chip_size = float(st.session_state["ebc_chip_size"]) / 1000.0
            origin_x = float(st.session_state["ebc_origin_x"])
            origin_y = float(st.session_state["ebc_origin_y"])
            half = chip_size * 0.5
            chip_cx = (st.session_state["ebc_bl_x"]
                    + st.session_state["ebc_tr_x"]) * 0.5
            chip_cy = (st.session_state["ebc_bl_y"]
                    + st.session_state["ebc_tr_y"]) * 0.5

            # Anchor cel to (origin − half) whenever chip size or origin
            # changes. The CEL file's (0,0) lives at the chip-area's
            # bottom-left corner; that point shifts when chip size changes,
            # so the stored 9.7-default that matched a 600 μm chip would
            # drift the mask off the writable area for any other size and
            # the per-grid clip would carve out the wrong region. Token
            # ensures user edits between size changes still stick.
            _dt_cel_token = (chip_size, origin_x, origin_y)
            if st.session_state.get("_ebc_dt_cel_token") != _dt_cel_token:
                st.session_state["ebc_dt_cel_x"] = float(origin_x - half)
                st.session_state["ebc_dt_cel_y"] = float(origin_y - half)
                st.session_state["_ebc_dt_cel_token"] = _dt_cel_token

            # Anchor dx/dy to chip_size on chip-size change so grids tile
            # edge-to-edge by default (otherwise the 0.6-mm default leaves
            # large gaps for small chips or overlap for large ones).
            _dt_dxy_token = (chip_size,)
            if st.session_state.get("_ebc_dt_dxy_token") != _dt_dxy_token:
                st.session_state["ebc_dt_dx"] = float(chip_size)
                st.session_state["ebc_dt_dy"] = float(chip_size)
                st.session_state["_ebc_dt_dxy_token"] = _dt_dxy_token

            # Auto-recenter shift so the grid pattern is centered on the chip.
            # Pattern bbox extends from grid (0,0) to grid (Nx-1, Ny-1), so its
            # center along x equals
            #   origin − half + ((Nx − 1)·dx + chip_size)/2 + shift
            # and similarly for y. Set this equal to the chip center and solve.
            nx_v = int(st.session_state["ebc_dt_nx"])
            ny_v = int(st.session_state["ebc_dt_ny"])
            dx_v = float(st.session_state["ebc_dt_dx"])
            dy_v = float(st.session_state["ebc_dt_dy"])
            shift_token = (nx_v, ny_v, dx_v, dy_v,
                        chip_size, origin_x, origin_y, chip_cx, chip_cy)
            if st.session_state.get("ebc_dt_shift_token") != shift_token:
                st.session_state["ebc_dt_shift_x"] = float(
                    chip_cx - origin_x + half
                    - ((nx_v - 1) * dx_v + chip_size) * 0.5
                )
                st.session_state["ebc_dt_shift_y"] = float(
                    chip_cy - origin_y + half
                    - ((ny_v - 1) * dy_v + chip_size) * 0.5
                )
                st.session_state["ebc_dt_shift_token"] = shift_token

            p1, p2, p3, p4 = st.columns(4)
            with p1:
                with st.container(border=True):
                    st.markdown(f"**{tr('Cel Origin (mm) in job1', 'job1 中的 Cel 原點 (mm)')}**")
                    p11, p12 = st.columns(2)
                    cel_x = p11.number_input("x", format="%.3f",
                                            step=0.1, min_value=0.0, key="ebc_dt_cel_x")
                    cel_y = p12.number_input("y", format="%.3f",
                                            step=0.1, min_value=0.0, key="ebc_dt_cel_y")
            with p2:
                with st.container(border=True):
                    st.markdown(f"**{tr('Increment (mm) in job3', 'job3 中的增量 (mm)')}**")
                    p11, p12 = st.columns(2)
                    dx = p11.number_input("dx", format="%.3f",
                                        min_value=chip_size,
                                        step=0.0005, key="ebc_dt_dx")
                    dy = p12.number_input("dy", format="%.3f",
                                        min_value=chip_size,
                                        step=0.0005, key="ebc_dt_dy")
            with p3:
                with st.container(border=True):
                    st.markdown(f"**{tr('Grid Count in job3', 'job3 中的網格數量')}**")
                    p11, p12 = st.columns(2)
                    Nx = p11.number_input("Nx", min_value=1,
                                            step=1, key="ebc_dt_nx")
                    Ny = p12.number_input("Ny", min_value=1,
                                            step=1, key="ebc_dt_ny")
            with p4:
                with st.container(border=True):
                    st.markdown(f"**{tr('Initial Shift (mm) in job3', 'job3 中的初始位移 (mm)')}**")
                    p11, p12 = st.columns(2)
                    shift_x = p11.number_input("x", format="%.3f",
                                            step=0.0005, key="ebc_dt_shift_x")
                    shift_y = p12.number_input("y", format="%.3f",
                                            step=0.0005, key="ebc_dt_shift_y")

            Nx_i, Ny_i = int(Nx), int(Ny)
            grid_xs: list = []
            grid_ys: list = []
            _dt_cells: list = []
            for i in range(Nx_i):
                for j in range(Ny_i):
                    gx = origin_x - half + i * dx + shift_x
                    gy = origin_y - half + j * dy + shift_y
                    grid_xs += [gx, gx + chip_size, gx + chip_size, gx, gx, None]
                    grid_ys += [gy, gy, gy + chip_size, gy + chip_size, gy, None]
                    _dt_cells.append((gx, gy, gx + chip_size, gy + chip_size))

            # Mask cell overlay (right plot + Time Calculator):
            # 1. Place the mask once at (cel_x, cel_y) — the single-grid
            #    position with no shift.
            # 2. Clip it to that single grid so any polygon spilling outside
            #    is trimmed *before* duplication.
            # 3. Duplicate the already-trimmed shapes to each (i, j) cell.
            # The left plot ("Single Grid (no shift)") builds its own
            # un-clipped placement separately and is intentionally unaffected.
            _dt_polys_mm: list = []
            mask_xs: list = []
            mask_ys: list = []
            _dt_precomp = None
            _dt_huge = bool(_gds_selected_polys) and len(_gds_selected_polys) > _POLY_LIMIT
            single_box = (origin_x - half, origin_y - half,
                          origin_x + half, origin_y + half)
            if _gds_selected_polys and not _dt_huge:
                scale = _gds_unit_to_mm
                single_polys: list = []
                for xs, ys in _gds_selected_polys:
                    pxs = [cel_x + p * scale for p in xs]
                    pys = [cel_y + p * scale for p in ys]
                    single_polys.append((pxs, pys))
                _, _dt_clipped_single = _polygon_clip_per_cell_mm(
                    single_polys, [single_box])
                for i in range(Nx_i):
                    for j in range(Ny_i):
                        offset_x = i * dx + shift_x
                        offset_y = j * dy + shift_y
                        for cxs, cys in _dt_clipped_single:
                            if not cxs:
                                continue
                            txs = [x + offset_x for x in cxs]
                            tys = [y + offset_y for y in cys]
                            _dt_polys_mm.append((txs, tys))
                            mask_xs += txs + [txs[0], None]
                            mask_ys += tys + [tys[0], None]
            elif _dt_huge:
                # Every grid carries the same mask (DT replicates it), so the
                # per-grid filled area is the single-grid mask area; compute
                # it once (binned into the single grid box) and broadcast.
                _single_area = _cell_areas_binned(
                    _gds_selected_polys, _gds_unit_to_mm, cel_x, cel_y,
                    origin_x - half, origin_y - half, chip_size, 1, 1)[0]
                _dt_precomp = [_single_area] * len(_dt_cells)

            bl = (st.session_state["ebc_bl_x"], st.session_state["ebc_bl_y"])
            br = (st.session_state["ebc_br_x"], st.session_state["ebc_br_y"])
            c_tr = (st.session_state["ebc_tr_x"], st.session_state["ebc_tr_y"])
            tl = (st.session_state["ebc_tl_x"], st.session_state["ebc_tl_y"])

            # If the mask bbox extends beyond the single-grid box, parts
            # of every duplicated tile get trimmed by the per-grid clip —
            # surface a centered notice so the user knows mask content is
            # being dropped.
            if _gds_selected_polys:
                _bbox = _layer_bbox_mm(_gds_selected_polys, _gds_unit_to_mm)
                if _bbox is not None:
                    _mx_min, _my_min, _mx_max, _my_max = _bbox
                    _mx_min += cel_x; _mx_max += cel_x
                    _my_min += cel_y; _my_max += cel_y
                    _eps = 1e-9
                    if (_mx_min < (origin_x - half) - _eps
                            or _mx_max > (origin_x + half) + _eps
                            or _my_min < (origin_y - half) - _eps
                            or _my_max > (origin_y + half) + _eps):
                        _show_outside_pattern_notice()

            plot_left, plot_right = st.columns(2)

            # ── Left plot: single grid at the setup origin (no shift) ─────────
            with plot_left:
                st.markdown(f"**{tr('Single Grid (no shift)', '單一網格（無位移）')}**")
                gx0 = origin_x - half
                gy0 = origin_y - half
                single_grid_x = [
                    gx0, gx0 + chip_size, gx0 + chip_size, gx0, gx0,
                ]
                single_grid_y = [
                    gy0, gy0, gy0 + chip_size, gy0 + chip_size, gy0,
                ]

                single_mask_xs: list = []
                single_mask_ys: list = []
                if _gds_selected_polys and not _dt_huge:
                    scale = _gds_unit_to_mm
                    for xs, ys in _gds_selected_polys:
                        pxs = [cel_x + p * scale for p in xs]
                        pys = [cel_y + p * scale for p in ys]
                        single_mask_xs += pxs + [pxs[0], None]
                        single_mask_ys += pys + [pys[0], None]

                fig_single = go.Figure()
                fig_single.add_trace(go.Scatter(
                    x=single_grid_x, y=single_grid_y,
                    mode="lines",
                    line=dict(color="#ff7f0e", width=1),
                    fill="toself",
                    fillcolor="rgba(255,127,14,0.20)",
                    name="Grid",
                    hoverinfo="skip",
                ))
                if single_mask_xs:
                    fig_single.add_trace(go.Scatter(
                        x=single_mask_xs, y=single_mask_ys,
                        mode="lines",
                        line=dict(color="#2ca02c", width=0.5),
                        fill="toself",
                        fillcolor="rgba(44,160,44,0.5)",
                        name=tr("Mask", "遮罩"),
                        hoverinfo="skip",
                    ))
                elif _dt_huge:
                    # Too dense to draw exactly — show where the pattern
                    # actually is, not just the rectangle it fits inside.
                    for _t in _mask_overlay_traces(
                            _gds_selected_polys, _gds_unit_to_mm, cel_x, cel_y,
                            "#2ca02c", tr("Mask", "遮罩")):
                        fig_single.add_trace(_t)

                # Range is fixed to the grid (+ chip-size padding) so that
                # moving cel origin or large mask polygons can't blow up the
                # plot scale and shrink the grid to a dot.
                s_xs = single_grid_x
                s_ys = single_grid_y
                s_pad_x = chip_size * 0.5
                s_pad_y = chip_size * 0.5
                fig_single.update_layout(
                    xaxis=dict(title="x (mm)",
                            range=[min(s_xs) - s_pad_x, max(s_xs) + s_pad_x]),
                    yaxis=dict(title="y (mm)",
                            range=[min(s_ys) - s_pad_y, max(s_ys) + s_pad_y],
                            scaleanchor="x", scaleratio=1),
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=550,
                    showlegend=True,
                )
                st.plotly_chart(fig_single, width="stretch")

            # ── Right plot: full Nx×Ny pattern overlaid on chip (with shift) ──
            with plot_right:
                st.markdown(f"**{tr('Chip Position with Grids (after shift)', '晶片位置與網格（位移後）')}**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[bl[0], br[0], c_tr[0], tl[0], bl[0]],
                    y=[bl[1], br[1], c_tr[1], tl[1], bl[1]],
                    mode="lines",
                    line=dict(color="#1f77b4", width=2),
                    fill="toself",
                    fillcolor="rgba(31,119,180,0.15)",
                    name="Chip",
                    hoverinfo="skip",
                ))
                if grid_xs:
                    fig.add_trace(go.Scatter(
                        x=grid_xs, y=grid_ys,
                        mode="lines",
                        fill="toself",
                        line=dict(color="#ff7f0e", width=1),
                        fillcolor="rgba(255,127,14,0.20)",
                        name=f"Grid ({Nx_i}×{Ny_i})",
                        hoverinfo="skip",
                    ))
                if mask_xs:
                    fig.add_trace(go.Scatter(
                        x=mask_xs, y=mask_ys,
                        mode="lines",
                        fill="toself",
                        line=dict(color="#2ca02c", width=0.5),
                        fillcolor="rgba(44,160,44,0.5)",
                        name=tr("Mask", "遮罩"),
                        hoverinfo="skip",
                    ))
                elif _dt_huge:
                    # Mask too dense to draw — replicate its bounding box,
                    # clipped to the single grid, into every tile.
                    _bb = _placed_bbox_mm(_gds_selected_polys, _gds_unit_to_mm,
                                          cel_x, cel_y)
                    if _bb is not None:
                        bx0 = max(_bb[0], single_box[0]); by0 = max(_bb[1], single_box[1])
                        bx1 = min(_bb[2], single_box[2]); by1 = min(_bb[3], single_box[3])
                        _bxs: list = []; _bys: list = []
                        if bx1 > bx0 and by1 > by0:
                            for i in range(Nx_i):
                                for j in range(Ny_i):
                                    ox = i * dx + shift_x; oy = j * dy + shift_y
                                    _bxs += [bx0 + ox, bx1 + ox, bx1 + ox,
                                             bx0 + ox, bx0 + ox, None]
                                    _bys += [by0 + oy, by0 + oy, by1 + oy,
                                             by1 + oy, by0 + oy, None]
                        if _bxs:
                            fig.add_trace(go.Scatter(
                                x=_bxs, y=_bys, mode="lines", fill="toself",
                                line=dict(color="#2ca02c", width=1, dash="dash"),
                                fillcolor="rgba(44,160,44,0.12)",
                                name=f"Mask bbox ({len(_gds_selected_polys):,} polys)",
                                hoverinfo="skip",
                            ))

                # Chip corner dots (hover to read coords)
                fig.add_trace(go.Scatter(
                    x=[bl[0], br[0], c_tr[0], tl[0]],
                    y=[bl[1], br[1], c_tr[1], tl[1]],
                    mode="markers",
                    marker=dict(size=8, color="#1f77b4", symbol="circle"),
                    hovertemplate=("Chip corner<br>"
                                "(%{x:.3f}, %{y:.3f}) mm<extra></extra>"),
                    name="Chip corners",
                    showlegend=False,
                ))

                # Outer corners of the overall grid pattern
                gx_pts = [v for v in grid_xs if v is not None]
                gy_pts = [v for v in grid_ys if v is not None]
                if gx_pts and gy_pts:
                    g_xmin, g_xmax = min(gx_pts), max(gx_pts)
                    g_ymin, g_ymax = min(gy_pts), max(gy_pts)
                    fig.add_trace(go.Scatter(
                        x=[g_xmin, g_xmax, g_xmax, g_xmin],
                        y=[g_ymin, g_ymin, g_ymax, g_ymax],
                        mode="markers",
                        marker=dict(size=8, color="#ff7f0e",
                                    symbol="square"),
                        hovertemplate=("Grid corner<br>"
                                    "(%{x:.3f}, %{y:.3f}) mm<extra></extra>"),
                        name="Grid corners",
                        showlegend=False,
                    ))

                all_x = [bl[0], br[0], c_tr[0], tl[0]] + [v for v in grid_xs if v is not None]
                all_y = [bl[1], br[1], c_tr[1], tl[1]] + [v for v in grid_ys if v is not None]
                pad_x = max(1.0, (max(all_x) - min(all_x)) * 0.10)
                pad_y = max(1.0, (max(all_y) - min(all_y)) * 0.10)
                fig.update_layout(
                    xaxis=dict(title="x (mm)",
                            range=[min(all_x) - pad_x, max(all_x) + pad_x]),
                    yaxis=dict(title="y (mm)",
                            range=[min(all_y) - pad_y, max(all_y) + pad_y],
                            scaleanchor="x", scaleratio=1),
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=550,
                    showlegend=True,
                )
                st.plotly_chart(fig, width="stretch")

            if _dt_huge:
                if isinstance(_gds_selected_polys, _InstancedLayer):
                    st.caption(":blue[" + tr(
                        f"Repetition detected: a "
                        f"{_gds_selected_polys.base_poly_count():,}-polygon unit "
                        f"pattern tiled {_gds_selected_polys.instance_count():,}× "
                        f"= {len(_gds_selected_polys):,} polygons. Showing the "
                        "mask bounding box per grid; the time estimate uses the "
                        "single-grid mask area replicated per grid.",
                        f"偵測到重複圖案：由 "
                        f"{_gds_selected_polys.base_poly_count():,} 個多邊形組成的"
                        f"單元圖案重複貼附 {_gds_selected_polys.instance_count():,} "
                        f"次 = {len(_gds_selected_polys):,} 個多邊形。此處顯示每個"
                        "網格的遮罩邊界框；時間估計採用單一網格的遮罩面積並依網格"
                        "複製。"
                    ) + "]")
                else:
                    st.caption(":orange[" + tr(
                        f"Layer has {len(_gds_selected_polys):,} polygons "
                        f"(> {_POLY_LIMIT:,}); showing the mask bounding box. The "
                        "time estimate uses the single-grid mask area (sum of "
                        "polygon areas inside the grid), replicated per grid.",
                        f"此圖層有 {len(_gds_selected_polys):,} 個多邊形"
                        f"（超過 {_POLY_LIMIT:,}）；顯示遮罩邊界框。時間估計採用"
                        "單一網格的遮罩面積（網格內多邊形面積總和）並依網格複製。"
                    ) + "]")

            _render_time_calculator(
                "ebc_dt", _dt_polys_mm, _dt_cells,
                chip_size_mm=chip_size,
                dotmap=int(st.session_state["ebc_dotmap"]),
                dose_ramp=True,
                precomputed_cell_areas=_dt_precomp,
            )

        elif mode == _MODE_OPTS[2]:  # "First Exposure"
            _FE_DEFAULTS = {
                "ebc_fe_cel_x": 9.7, "ebc_fe_cel_y": 9.7,
                "ebc_fe_nx": 20, "ebc_fe_ny": 20,
                "ebc_fe_shift_x": 94.0, "ebc_fe_shift_y": 104.0,
            }
            for _k, _v in _FE_DEFAULTS.items():
                st.session_state.setdefault(_k, _v)

            # ebc_chip_size is stored in μm (selectbox); convert to mm.
            chip_size_v = float(st.session_state["ebc_chip_size"]) / 1000.0
            origin_x_v = float(st.session_state["ebc_origin_x"])
            origin_y_v = float(st.session_state["ebc_origin_y"])
            half_v = chip_size_v * 0.5

            # Anchor cel to (origin − half) on chip-size / origin change so
            # the mask's (0,0) sits at the grid's bottom-left corner. See
            # the same block in Dose Time Testing for the rationale.
            _fe_cel_token = (chip_size_v, origin_x_v, origin_y_v)
            if st.session_state.get("_ebc_fe_cel_token") != _fe_cel_token:
                st.session_state["ebc_fe_cel_x"] = float(origin_x_v - half_v)
                st.session_state["ebc_fe_cel_y"] = float(origin_y_v - half_v)
                st.session_state["_ebc_fe_cel_token"] = _fe_cel_token

            # ─── Auto-fit Nx/Ny from mask layer size ─────────────────────────
            # The mask's GDS (0,0) lands at cel_origin in plot coords, so the
            # mask spans (cel + min, cel + max). The grid pattern starts at
            # origin − half. Required Nx is the grid count needed to cover
            # the mask's far edge:
            #   Nx ≥ ceil((cel + max_x − (origin − half)) / chip_size)
            # rounded up to the next even integer. (Assumes the mask layer's
            # bbox starts at min ≥ 0; if it goes negative, the mask extends
            # left of the grid origin and the user must adjust cel.)
            if _gds_selected_polys:
                bbox = _layer_bbox_mm(_gds_selected_polys, _gds_unit_to_mm)
                if bbox is not None:
                    min_x, min_y, max_x, max_y = bbox
                    cel_x_v = float(st.session_state["ebc_fe_cel_x"])
                    cel_y_v = float(st.session_state["ebc_fe_cel_y"])
                    right_x = (cel_x_v + max_x) - (origin_x_v - half_v)
                    right_y = (cel_y_v + max_y) - (origin_y_v - half_v)
                    Nx_req = _round_up_even(right_x / chip_size_v)
                    Ny_req = _round_up_even(right_y / chip_size_v)
                    layer_w = max_x - min_x
                    layer_h = max_y - min_y

                    st.caption(
                        f"{tr('Mask layer size', '遮罩圖層尺寸')}: "
                        f"{layer_w:.3f} × {layer_h:.3f} mm"
                        f"  →  {tr('required grids', '所需網格數')}: "
                        f"{Nx_req} × {Ny_req} "
                        f"({tr('chip-size tile', '晶片尺寸區塊')} = "
                        f"{chip_size_v:.3f} mm, cel = "
                        f"{cel_x_v:.3f}, {cel_y_v:.3f})"
                    )

                    nxny_token = (_gds_selected_token, chip_size_v,
                                cel_x_v, cel_y_v)
                    if st.session_state.get("ebc_fe_nxny_token") != nxny_token:
                        st.session_state["ebc_fe_nx"] = int(Nx_req)
                        st.session_state["ebc_fe_ny"] = int(Ny_req)
                        st.session_state["ebc_fe_nxny_token"] = nxny_token

            # ─── Auto-recenter shift on chip ────────────────────────────────
            # Re-runs whenever Nx, Ny, chip size, origin, or chip corners
            # change — so manually nudging Nx will refresh the shift to keep
            # the grid pattern centered on the chip.
            nx_now = int(st.session_state.get("ebc_fe_nx", 20))
            ny_now = int(st.session_state.get("ebc_fe_ny", 20))
            chip_cx_v = (st.session_state["ebc_bl_x"]
                        + st.session_state["ebc_tr_x"]) * 0.5
            chip_cy_v = (st.session_state["ebc_bl_y"]
                        + st.session_state["ebc_tr_y"]) * 0.5

            shift_token = (
                nx_now, ny_now, chip_size_v, origin_x_v, origin_y_v,
                chip_cx_v, chip_cy_v,
            )
            if st.session_state.get("ebc_fe_shift_token") != shift_token:
                # Pattern center (after shift)
                #   = origin − half + Nx·chip_size/2 + shift
                # Set this equal to the chip center and solve for shift.
                st.session_state["ebc_fe_shift_x"] = float(
                    chip_cx_v - origin_x_v + half_v
                    - nx_now * chip_size_v * 0.5
                )
                st.session_state["ebc_fe_shift_y"] = float(
                    chip_cy_v - origin_y_v + half_v
                    - ny_now * chip_size_v * 0.5
                )
                st.session_state["ebc_fe_shift_token"] = shift_token

            p1, p2, p3 = st.columns(3)
            with p1:
                with st.container(border=True):
                    st.markdown(f"**{tr('Cel Origin (mm) in job1', 'job1 中的 Cel 原點 (mm)')}**")
                    p11, p12 = st.columns(2)
                    with p11:
                        cel_x = st.number_input("x", format="%.3f", step=0.1, min_value=0.0,
                                                key="ebc_fe_cel_x")
                    with p12:
                        cel_y = st.number_input("y", format="%.3f", step=0.1, min_value=0.0,
                                                key="ebc_fe_cel_y")
            with p2:
                with st.container(border=True):
                    st.markdown(f"**{tr('Grid Count in job1', 'job1 中的網格數量')}**")
                    p21, p22 = st.columns(2)
                    with p21:
                        Nx = st.number_input("Nx", min_value=1, step=1,
                                            key="ebc_fe_nx")
                    with p22:
                        Ny = st.number_input("Ny", min_value=1, step=1,
                                        key="ebc_fe_ny")
            with p3:
                with st.container(border=True):
                    st.markdown(f"**{tr('Shift (mm) in job3', 'job3 中的位移 (mm)')}**")
                    p31, p32 = st.columns(2)
                    with p31:
                        shift_x = st.number_input("x", format="%.3f", step=0.0005,
                                                key="ebc_fe_shift_x")
                    with p32:
                        shift_y = st.number_input("y", format="%.3f", step=0.0005,
                                            key="ebc_fe_shift_y")

            Nx_i, Ny_i = int(Nx), int(Ny)

            # Grid pattern — Nx × Ny tiles of chip_size, packed edge-to-edge,
            # representing the e-beam writable area (no mask duplication).
            grid_xs: list = []
            grid_ys: list = []
            _fe_cells: list = []
            for i in range(Nx_i):
                for j in range(Ny_i):
                    gx = origin_x_v - half_v + i * chip_size_v + shift_x
                    gy = origin_y_v - half_v + j * chip_size_v + shift_y
                    grid_xs += [gx, gx + chip_size_v, gx + chip_size_v,
                                gx, gx, None]
                    grid_ys += [gy, gy, gy + chip_size_v,
                                gy + chip_size_v, gy, None]
                    _fe_cells.append(
                        (gx, gy, gx + chip_size_v, gy + chip_size_v))

            # Single mask placement: GDS (0,0) lands at (cel + shift).
            # For dense layers (> _POLY_LIMIT polygons) the per-polygon
            # overlay and gdstk clip are infeasible, so we draw the mask
            # bounding box and feed the Time Calculator vectorized per-cell
            # areas instead.
            mask_xs: list = []
            mask_ys: list = []
            _fe_polys_mm: list = []
            _fe_precomp = None
            _fe_huge = bool(_gds_selected_polys) and len(_gds_selected_polys) > _POLY_LIMIT
            if _gds_selected_polys and not _fe_huge:
                scale = _gds_unit_to_mm
                for xs, ys in _gds_selected_polys:
                    pxs = [cel_x + shift_x + p * scale for p in xs]
                    pys = [cel_y + shift_y + p * scale for p in ys]
                    mask_xs += pxs + [pxs[0], None]
                    mask_ys += pys + [pys[0], None]
                    _fe_polys_mm.append((pxs, pys))
            elif _fe_huge:
                _fe_precomp = _cell_areas_binned(
                    _gds_selected_polys, _gds_unit_to_mm,
                    cel_x + shift_x, cel_y + shift_y,
                    origin_x_v - half_v + shift_x,
                    origin_y_v - half_v + shift_y,
                    chip_size_v, Nx_i, Ny_i,
                )

            bl = (st.session_state["ebc_bl_x"], st.session_state["ebc_bl_y"])
            br = (st.session_state["ebc_br_x"], st.session_state["ebc_br_y"])
            c_tr = (st.session_state["ebc_tr_x"], st.session_state["ebc_tr_y"])
            tl = (st.session_state["ebc_tl_x"], st.session_state["ebc_tl_y"])

            # Centered notice when the mask spills outside the Nx×Ny grid
            # bounding box; those areas will be dropped by the Time
            # Calculator's per-cell clip.
            if _gds_selected_polys:
                _bbox = _layer_bbox_mm(_gds_selected_polys, _gds_unit_to_mm)
                if _bbox is not None:
                    _mx_min, _my_min, _mx_max, _my_max = _bbox
                    _mx_min += cel_x + shift_x; _mx_max += cel_x + shift_x
                    _my_min += cel_y + shift_y; _my_max += cel_y + shift_y
                    _gx_min = origin_x_v - half_v + shift_x
                    _gy_min = origin_y_v - half_v + shift_y
                    _gx_max = _gx_min + Nx_i * chip_size_v
                    _gy_max = _gy_min + Ny_i * chip_size_v
                    _eps = 1e-9
                    if (_mx_min < _gx_min - _eps
                            or _mx_max > _gx_max + _eps
                            or _my_min < _gy_min - _eps
                            or _my_max > _gy_max + _eps):
                        _show_outside_pattern_notice()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[bl[0], br[0], c_tr[0], tl[0], bl[0]],
                y=[bl[1], br[1], c_tr[1], tl[1], bl[1]],
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                fill="toself",
                fillcolor="rgba(31,119,180,0.15)",
                name="Chip",
                hoverinfo="skip",
            ))
            if grid_xs:
                fig.add_trace(go.Scatter(
                    x=grid_xs, y=grid_ys,
                    mode="lines",
                    fill="toself",
                    line=dict(color="#ff7f0e", width=1),
                    fillcolor="rgba(255,127,14,0.15)",
                    name=f"Grid ({Nx_i}×{Ny_i})",
                    hoverinfo="skip",
                ))
            if mask_xs:
                fig.add_trace(go.Scatter(
                    x=mask_xs, y=mask_ys,
                    mode="lines",
                    fill="toself",
                    line=dict(color="#2ca02c", width=0.5),
                    fillcolor="rgba(44,160,44,0.5)",
                    name=tr("Mask", "遮罩"),
                    hoverinfo="skip",
                ))
            elif _fe_huge:
                for _t in _mask_overlay_traces(
                        _gds_selected_polys, _gds_unit_to_mm,
                        cel_x + shift_x, cel_y + shift_y, "#2ca02c",
                        tr("Mask", "遮罩")):
                    fig.add_trace(_t)

            # Chip corner dots (hover to read coords)
            fig.add_trace(go.Scatter(
                x=[bl[0], br[0], c_tr[0], tl[0]],
                y=[bl[1], br[1], c_tr[1], tl[1]],
                mode="markers",
                marker=dict(size=8, color="#1f77b4", symbol="circle"),
                hovertemplate=("Chip corner<br>"
                            "(%{x:.3f}, %{y:.3f}) mm<extra></extra>"),
                name="Chip corners",
                showlegend=False,
            ))

            # Outer corners of the overall grid pattern
            gx_pts = [v for v in grid_xs if v is not None]
            gy_pts = [v for v in grid_ys if v is not None]
            if gx_pts and gy_pts:
                g_xmin, g_xmax = min(gx_pts), max(gx_pts)
                g_ymin, g_ymax = min(gy_pts), max(gy_pts)
                fig.add_trace(go.Scatter(
                    x=[g_xmin, g_xmax, g_xmax, g_xmin],
                    y=[g_ymin, g_ymin, g_ymax, g_ymax],
                    mode="markers",
                    marker=dict(size=8, color="#ff7f0e", symbol="square"),
                    hovertemplate=("Grid corner<br>"
                                "(%{x:.3f}, %{y:.3f}) mm<extra></extra>"),
                    name="Grid corners",
                    showlegend=False,
                ))

            all_x = ([bl[0], br[0], c_tr[0], tl[0]]
                    + [v for v in grid_xs if v is not None])
            all_y = ([bl[1], br[1], c_tr[1], tl[1]]
                    + [v for v in grid_ys if v is not None])
            pad_x = max(1.0, (max(all_x) - min(all_x)) * 0.10)
            pad_y = max(1.0, (max(all_y) - min(all_y)) * 0.10)
            fig.update_layout(
                xaxis=dict(title="x (mm)",
                        range=[min(all_x) - pad_x, max(all_x) + pad_x]),
                yaxis=dict(title="y (mm)",
                        range=[min(all_y) - pad_y, max(all_y) + pad_y],
                        scaleanchor="x", scaleratio=1),
                margin=dict(l=40, r=20, t=20, b=40),
                height=600,
                showlegend=True,
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, width="stretch")

            if _fe_huge:
                st.caption(_dense_layer_note(_gds_selected_polys))

            _render_time_calculator(
                "ebc_fe", _fe_polys_mm, _fe_cells,
                chip_size_mm=chip_size_v,
                dotmap=int(st.session_state["ebc_dotmap"]),
                precomputed_cell_areas=_fe_precomp,
                coverage_layer=_gds_selected_polys if _fe_huge else None,
                coverage_scale=_gds_unit_to_mm,
                coverage_ox=cel_x + shift_x, coverage_oy=cel_y + shift_y,
            )

        elif mode == _MODE_OPTS[3]:  # "Second Alignment"
            # Cross-mark position presets. Coordinates are chip-relative (mm).
            # Add new presets here; the radio below lists all keys.
            # NOTE: the HBT_RF_v3 values below are a placeholder 4×6 grid —
            # replace with the actual cross positions from the mask.
            _CROSS_PRESETS = {
                # in mm
                "HBT_RF_v3": {
                    "A1": (0.3695, 10.1195), "A2": (0.730, 10.12),
                    "A3": (0.0595, 9.8095), "A4": (0.060, 9.45),
                    "A5": (2.975, 9.35), "A6": (2.975, 8.95),
                    "B1": (9.6095,  10.1195), "B2": (9.25,  10.12),
                    "B3": (9.9195,  9.8095), "B4": (9.92,  9.45),
                    "B5": (7.005,  9.35), "B6": (7.005,  8.95),
                    "C1": (9.6095,  0.0595), "C2": (9.25,  0.06),
                    "C3": (9.92,  0.73), "C4": (9.9195,  0.3695),
                    "C5": (7.005,  1.39), "C6": (7.005,  1.0),
                    "D1": (0.3695,  0.0595), "D2": (0.730,  0.06),
                    "D3": (0.0595,  0.3695), "D4": (0.06,  0.73),
                    "D5": (2.96,  1.39), "D6": (2.96,  1.0),
                },
                "Custom": {},
            }

            preset_name = segmented_radio(
                tr("Cross-position preset", "十字標記位置預設"),
                list(_CROSS_PRESETS.keys()),
                key="ebc_sa_preset",
                # Preset keys (e.g. "HBT_RF_v3", "Custom") are compared against
                # below (preset_name == "Custom") and used as dict keys, so they
                # must stay English; only the displayed label is localized.
                format_func=lambda k: tr("Custom", "自訂") if k == "Custom" else k,
            )

            if preset_name == "Custom":
                # Custom preset: collect 2 alignment-mark positions from the
                # user and use them as `crosses` for the rest of the flow.
                # Canonical storage is always mm with 1 nm precision; a unit
                # radio lets the user enter values in mm or μm.
                _CUSTOM_DEFAULTS = {
                    "ebc_sa_custom_m1_x": 0.0, "ebc_sa_custom_m1_y": 0.0,
                    "ebc_sa_custom_m2_x": 6.0, "ebc_sa_custom_m2_y": 6.0,
                    "ebc_sa_custom_unit": "mm",
                }
                for _k, _v in _CUSTOM_DEFAULTS.items():
                    st.session_state.setdefault(_k, _v)

                unit = segmented_radio(
                    tr("Input unit", "輸入單位"), ["mm", "μm"],
                    key="ebc_sa_custom_unit",
                )
                to_disp = 1000.0 if unit == "μm" else 1.0   # mm → display unit
                # 1 nm precision: 1e-6 mm or 1e-3 μm.
                fmt = "%.3f" if unit == "μm" else "%.6f"
                step = 0.001 if unit == "μm" else 0.000001

                # Re-seed the widget value from the canonical mm key whenever
                # the unit toggles, so switching modes shows the same physical
                # value in the new unit instead of a stale display number.
                prev_unit = st.session_state.get("_ebc_sa_custom_prev_unit", unit)
                unit_changed = (prev_unit != unit)
                st.session_state["_ebc_sa_custom_prev_unit"] = unit

                def _mark_input(label: str, mm_key: str) -> float:
                    widget_key = f"{mm_key}_disp"
                    if unit_changed or widget_key not in st.session_state:
                        st.session_state[widget_key] = (
                            st.session_state[mm_key] * to_disp
                        )
                    val = st.number_input(
                        label, format=fmt, step=step, key=widget_key,
                    )
                    # Persist canonical mm value (1 nm precision).
                    st.session_state[mm_key] = round(val / to_disp, 6)
                    return st.session_state[mm_key]

                cm1, cm2 = st.columns(2)
                with cm1:
                    with st.container(border=True):
                        st.markdown(f"**{tr('Mark M1', '標記 M1')} ({unit})**")
                        m1cx, m1cy = st.columns(2)
                        with m1cx:
                            m1_x = _mark_input("x", "ebc_sa_custom_m1_x")
                        with m1cy:
                            m1_y = _mark_input("y", "ebc_sa_custom_m1_y")
                with cm2:
                    with st.container(border=True):
                        st.markdown(f"**{tr('Mark M2', '標記 M2')} ({unit})**")
                        m2cx, m2cy = st.columns(2)
                        with m2cx:
                            m2_x = _mark_input("x", "ebc_sa_custom_m2_x")
                        with m2cy:
                            m2_y = _mark_input("y", "ebc_sa_custom_m2_y")

                crosses = {"M1": (m1_x, m1_y), "M2": (m2_x, m2_y)}
            else:
                crosses = _CROSS_PRESETS[preset_name]
                # Preset-specific default marks for the Overlayed selectors.
                # Only applied when the keys aren't already set, so user
                # picks persist across reruns.
                if preset_name == "HBT_RF_v3":
                    st.session_state.setdefault("ebc_sa_o_mark1", "A1")
                    st.session_state.setdefault("ebc_sa_o_mark2", "C1")

            if not crosses:
                st.info(tr("No cross positions defined for this preset.",
                           "此預設未定義十字標記位置。"))
            else:
                st.caption(tr(
                    f"{len(crosses)} cross positions loaded (chip-relative, mm).",
                    f"已載入 {len(crosses)} 個十字標記位置（相對晶片，mm）。"))
                rows = [{"Cross": k, "x (mm)": v[0], "y (mm)": v[1]}
                        for k, v in crosses.items()]
                with st.expander(tr("Show cross positions table", "顯示十字標記位置表")):
                    st.table(rows)

                # Helper for the Existing Pattern subsection: render the
                # Move: anchor + Target x/y inputs in `controls_col`, then the
                # Plotly figure (polygons + cross dots, shifted so the chosen
                # anchor lands at the target) in `plot_col`. Returns the
                # (shift_x, shift_y) applied so Overlayed can re-use it.
                def _render_pattern_alignment(polys, scale_to_mm, crosses,
                                            key_prefix,
                                            controls_col, plot_col):
                    cross_labels = list(crosses.keys())

                    with controls_col:
                        with st.container(border=True):
                            anchor = st.selectbox(
                                tr("Move:", "移動："), cross_labels,
                                key=f"{key_prefix}_anchor",
                            )
                            anchor_x, anchor_y = crosses[anchor]
                            st.session_state.setdefault(
                                f"{key_prefix}_target_x", float(anchor_x))
                            st.session_state.setdefault(
                                f"{key_prefix}_target_y", float(anchor_y))
                            
                            st.caption(tr("Mark position as seen on the SEM:",
                                          "如 SEM 所見的標記位置："))
                            target_x_input, target_y_input = st.columns(2)
                            with target_x_input:
                                target_x = st.number_input(
                                    tr("Target x", "目標 x"), format="%.4f", step=0.001,
                                    key=f"{key_prefix}_target_x",
                                )
                            with target_y_input:
                                target_y = st.number_input(
                                    tr("Target y", "目標 y"), format="%.4f", step=0.001,
                                    key=f"{key_prefix}_target_y",
                                )

                    shift_x = target_x - anchor_x
                    shift_y = target_y - anchor_y

                    fig = go.Figure()
                    if polys and len(polys) > _POLY_LIMIT:
                        for _t in _mask_overlay_traces(
                                polys, scale_to_mm, shift_x, shift_y,
                                "#1f77b4", tr("Pattern", "圖案")):
                            fig.add_trace(_t)
                    elif polys:
                        xs_all, ys_all = [], []
                        for xs, ys in polys:
                            mxs = [shift_x + p * scale_to_mm for p in xs]
                            mys = [shift_y + p * scale_to_mm for p in ys]
                            xs_all += mxs + [mxs[0], None]
                            ys_all += mys + [mys[0], None]
                        fig.add_trace(go.Scatter(
                            x=xs_all, y=ys_all,
                            mode="lines",
                            fill="toself",
                            line=dict(color="#1f77b4", width=0.5),
                            fillcolor="rgba(31,119,180,0.3)",
                            name="Pattern",
                            hoverinfo="skip",
                        ))
                    cross_x = [v[0] + shift_x for v in crosses.values()]
                    cross_y = [v[1] + shift_y for v in crosses.values()]
                    cross_text = list(crosses.keys())
                    fig.add_trace(go.Scatter(
                        x=cross_x, y=cross_y,
                        mode="markers+text",
                        marker=dict(size=10, color="#d62728", symbol="x"),
                        text=cross_text,
                        textposition="top right",
                        hovertemplate=("%{text}<br>"
                                    "(%{x:.4f}, %{y:.4f}) mm<extra></extra>"),
                        name="Crosses",
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
                    with plot_col:
                        st.plotly_chart(fig, width="stretch",
                                        key=f"{key_prefix}_chart")
                    return shift_x, shift_y

                # ── Existing Pattern ─────────────────────────────────────────
                st.subheader(tr("Existing Pattern on Chip", "晶片上的既有圖案"))

                ep_polys: list = []
                ep_shift_x = 0.0
                ep_shift_y = 0.0
                ep_ready = False

                ep_ctrl_col, ep_plot_col = st.columns([3, 7])

                with ep_ctrl_col:
                    if not _gds_cells_data:
                        st.caption(tr(
                            "No GDS file uploaded — showing crosses only.",
                            "未上傳 GDS 檔案 — 僅顯示十字標記。"))
                    else:
                        cell_name = _gds_selected_cell_name
                        ep_by_layer = (_gds_cells_data.get(cell_name, {})
                                    if cell_name else {})
                        ep_layer_keys = sorted(ep_by_layer.keys())
                        ep_layer_options = {
                            f"L{l}/D{d}  ({len(ep_by_layer[(l, d)]):,} polys)": (l, d)
                            for (l, d) in ep_layer_keys
                        }
                        ep_labels = list(ep_layer_options.keys())
                        ep_layer_label = st.selectbox(
                            tr("Existing pattern layer", "既有圖案圖層"),
                            ep_labels if ep_labels else ["—"],
                            disabled=not ep_labels,
                            key="ebc_sa_ep_layer",
                        )
                        if ep_labels:
                            ep_polys = ep_by_layer[
                                ep_layer_options[ep_layer_label]]

                ep_shift_x, ep_shift_y = _render_pattern_alignment(
                    ep_polys, _gds_unit_to_mm, crosses, "ebc_sa_ep",
                    ep_ctrl_col, ep_plot_col,
                )
                ep_ready = True

                # ── Second Alignment Pattern ─────────────────────────────────
                st.subheader(tr("Second Alignment Pattern", "二次對準圖案"))

                sap_polys = _gds_selected_polys
                # Dense layers can't be drawn/clipped per-polygon — fall back
                # to bounding box + vectorized area (see workflow notes above).
                _sa_huge = bool(sap_polys) and len(sap_polys) > _POLY_LIMIT
                _ep_huge = bool(ep_polys) and len(ep_polys) > _POLY_LIMIT

                _SAP_DEFAULTS = {
                    "ebc_sa_sap_cel_x": 9.7, "ebc_sa_sap_cel_y": 9.7,
                    "ebc_sa_sap_nx": 20, "ebc_sa_sap_ny": 20,
                }
                for _k, _v in _SAP_DEFAULTS.items():
                    st.session_state.setdefault(_k, _v)

                # ebc_chip_size is stored in μm (selectbox); convert to mm.
                chip_size_sap = float(st.session_state["ebc_chip_size"]) / 1000.0
                origin_x_sap = float(st.session_state["ebc_origin_x"])
                origin_y_sap = float(st.session_state["ebc_origin_y"])
                half_sap = chip_size_sap * 0.5

                # Anchor sap cel to (origin − half) on chip-size / origin
                # change so the mask aligns with the grid for any chip size
                # (see Dose Time Testing for the rationale).
                _sap_cel_token = (chip_size_sap, origin_x_sap, origin_y_sap)
                if st.session_state.get("_ebc_sap_cel_token") != _sap_cel_token:
                    st.session_state["ebc_sa_sap_cel_x"] = float(
                        origin_x_sap - half_sap)
                    st.session_state["ebc_sa_sap_cel_y"] = float(
                        origin_y_sap - half_sap)
                    st.session_state["_ebc_sap_cel_token"] = _sap_cel_token

                sap_ctrl_col, sap_plot_col = st.columns([3, 7])

                with sap_ctrl_col:
                    if sap_polys:
                        # Auto-fit Nx/Ny so the grid covers the mask's far
                        # edge given the cel offset:
                        #   Nx ≥ ceil((cel + max_x − (origin − half))
                        #             / chip_size)
                        # rounded up to the next even integer. Re-runs on
                        # layer, chip-size, or cel changes; manual edits to
                        # Nx/Ny stick until one of those changes again.
                        bbox_sap = _layer_bbox_mm(sap_polys, _gds_unit_to_mm)
                        if bbox_sap is not None:
                            bb_min_x, bb_min_y, bb_max_x, bb_max_y = bbox_sap
                            cel_x_v = float(
                                st.session_state["ebc_sa_sap_cel_x"])
                            cel_y_v = float(
                                st.session_state["ebc_sa_sap_cel_y"])
                            right_x = ((cel_x_v + bb_max_x)
                                    - (origin_x_sap - half_sap))
                            right_y = ((cel_y_v + bb_max_y)
                                    - (origin_y_sap - half_sap))
                            nx_req = _round_up_even(right_x / chip_size_sap)
                            ny_req = _round_up_even(right_y / chip_size_sap)
                            bb_w = bb_max_x - bb_min_x
                            bb_h = bb_max_y - bb_min_y
                            st.caption(
                                f"{tr('Mask layer size', '遮罩圖層尺寸')}: "
                                f"{bb_w:.3f} × {bb_h:.3f} mm"
                                f"  →  {tr('required grids', '所需網格數')}: "
                                f"{nx_req} × {ny_req}"
                            )
                            sap_token = (_gds_selected_token, chip_size_sap,
                                        cel_x_v, cel_y_v)
                            if (st.session_state.get("ebc_sa_sap_nxny_token")
                                    != sap_token):
                                st.session_state["ebc_sa_sap_nx"] = int(nx_req)
                                st.session_state["ebc_sa_sap_ny"] = int(ny_req)
                                st.session_state["ebc_sa_sap_nxny_token"] = sap_token
                    else:
                        st.caption(tr(
                            "No GDS layer selected — showing crosses only.",
                            "未選擇 GDS 圖層 — 僅顯示十字標記。"))

                    sp1, sp2 = st.columns(2)
                    with sp1:
                        with st.container(border=True):
                            st.markdown(f"**{tr('Cel Origin (mm) in job1', 'job1 中的 Cel 原點 (mm)')}**")
                            sap_cel_x = st.number_input(
                                "x", format="%.3f", step=0.1, min_value=0.0,
                                key="ebc_sa_sap_cel_x")
                            sap_cel_y = st.number_input(
                                "y", format="%.3f", step=0.1, min_value=0.0,
                                key="ebc_sa_sap_cel_y")
                    with sp2:
                        with st.container(border=True):
                            st.markdown(f"**{tr('Grid Count in job1', 'job1 中的網格數量')}**")
                            sap_nx = st.number_input(
                                "Nx", min_value=1, step=1,
                                key="ebc_sa_sap_nx")
                            sap_ny = st.number_input(
                                "Ny", min_value=1, step=1,
                                key="ebc_sa_sap_ny")
                    with st.container(border=True):
                        st.markdown(f"**{tr('Registration mark position in job1', 'job1 中的對位標記位置')}**")
                        celx = float(st.session_state["ebc_sa_sap_cel_x"])
                        cely = float(st.session_state["ebc_sa_sap_cel_y"])

                        # Load the 2 marks being used (chosen in the Overlayed
                        # section's Mark 1 / Mark 2 selectboxes — read from
                        # session_state since that block renders after this one).
                        # Fall back to the first two crosses on the first run
                        # before Overlayed has populated the selection.
                        _cross_keys = list(crosses.keys())
                        m1_name = st.session_state.get("ebc_sa_o_mark1")
                        if not m1_name or m1_name not in crosses:
                            m1_name = _cross_keys[0] if _cross_keys else None
                        m2_name = st.session_state.get("ebc_sa_o_mark2")
                        if (not m2_name or m2_name == m1_name
                                or m2_name not in crosses):
                            _others = [k for k in _cross_keys if k != m1_name]
                            m2_name = _others[0] if _others else m1_name

                        mk1x_mask = float(crosses[m1_name][0])
                        mk1y_mask = float(crosses[m1_name][1])
                        mk2x_mask = float(crosses[m2_name][0])
                        mk2y_mask = float(crosses[m2_name][1])

                        # add mark offset to cel origin to get mark positions in job1 coords:
                        mk1x, mk1y = mk1x_mask+celx, mk1y_mask+cely
                        mk2x, mk2y = mk2x_mask+celx, mk2y_mask+cely

                        m1_col, m2_col = st.columns(2)
                        with m1_col:
                            with st.container(border=True):
                                st.markdown(
                                    f"**{tr('Mark 1', '標記 1')} ({m1_name})**",
                                    help=f"x = {mk1x_mask} mm + {celx:.3f} mm, "
                                        f"y = {mk1y_mask} mm + {cely:.3f} mm",
                                )
                                st.markdown(f"x: {mk1x:.4f}")
                                st.markdown(f"y: {mk1y:.4f}")
                        with m2_col:
                            with st.container(border=True):
                                st.markdown(
                                    f"**{tr('Mark 2', '標記 2')} ({m2_name})**",
                                    help=f"x = {mk2x_mask} mm + {celx:.3f} mm, "
                                        f"y = {mk2y_mask} mm + {cely:.3f} mm",
                                )
                                st.markdown(f"x: {mk2x:.4f}")
                                st.markdown(f"y: {mk2y:.4f}")

                sap_nx_i = int(sap_nx)
                sap_ny_i = int(sap_ny)
                scale_sap = _gds_unit_to_mm

                # Plot: only render grid + mask if a GDS layer is selected;
                # crosses always render (and follow the cel origin).
                fig_sap = go.Figure()
                _sa_cells: list = []
                _sa_polys_mm: list = []
                if sap_polys:
                    sap_grid_xs: list = []
                    sap_grid_ys: list = []
                    for i in range(sap_nx_i):
                        for j in range(sap_ny_i):
                            gx = origin_x_sap - half_sap + i * chip_size_sap
                            gy = origin_y_sap - half_sap + j * chip_size_sap
                            sap_grid_xs += [gx, gx + chip_size_sap,
                                            gx + chip_size_sap,
                                            gx, gx, None]
                            sap_grid_ys += [gy, gy, gy + chip_size_sap,
                                            gy + chip_size_sap, gy, None]
                            _sa_cells.append(
                                (gx, gy,
                                gx + chip_size_sap, gy + chip_size_sap))
                    fig_sap.add_trace(go.Scatter(
                        x=sap_grid_xs, y=sap_grid_ys,
                        mode="lines",
                        fill="toself",
                        line=dict(color="#ff7f0e", width=1),
                        fillcolor="rgba(255,127,14,0.15)",
                        name=f"Grid ({sap_nx_i}×{sap_ny_i})",
                        hoverinfo="skip",
                    ))

                    if _sa_huge:
                        for _t in _mask_overlay_traces(
                                sap_polys, scale_sap, sap_cel_x, sap_cel_y,
                                "#2ca02c", tr("Mask", "遮罩")):
                            fig_sap.add_trace(_t)
                    else:
                        sap_mask_xs: list = []
                        sap_mask_ys: list = []
                        for xs, ys in sap_polys:
                            pxs = [sap_cel_x + p * scale_sap for p in xs]
                            pys = [sap_cel_y + p * scale_sap for p in ys]
                            sap_mask_xs += pxs + [pxs[0], None]
                            sap_mask_ys += pys + [pys[0], None]
                            _sa_polys_mm.append((pxs, pys))
                        fig_sap.add_trace(go.Scatter(
                            x=sap_mask_xs, y=sap_mask_ys,
                            mode="lines",
                            fill="toself",
                            line=dict(color="#2ca02c", width=0.5),
                            fillcolor="rgba(44,160,44,0.5)",
                            name=tr("Mask", "遮罩"),
                            hoverinfo="skip",
                        ))

                # Crosses follow the cel origin: as cel moves, the cross
                # marks move with the second alignment pattern.
                fig_sap.add_trace(go.Scatter(
                    x=[v[0] + sap_cel_x for v in crosses.values()],
                    y=[v[1] + sap_cel_y for v in crosses.values()],
                    mode="markers+text",
                    marker=dict(size=10, color="#d62728", symbol="x"),
                    text=list(crosses.keys()),
                    textposition="top right",
                    hovertemplate=("%{text}<br>"
                                "(%{x:.4f}, %{y:.4f}) mm<extra></extra>"),
                    name="Crosses",
                ))
                fig_sap.update_layout(
                    xaxis=dict(title="x (mm)"),
                    yaxis=dict(title="y (mm)",
                            scaleanchor="x", scaleratio=1),
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=500,
                    showlegend=True,
                    plot_bgcolor="white",
                )
                # Grid bounds in the SAP frame (no overlay shift).
                _gx_min_sap = origin_x_sap - half_sap
                _gy_min_sap = origin_y_sap - half_sap
                _gx_max_sap = _gx_min_sap + sap_nx_i * chip_size_sap
                _gy_max_sap = _gy_min_sap + sap_ny_i * chip_size_sap

                # Detect if either selected alignment mark would land
                # outside the SAP grid bounds (means user needs to expand
                # Nx/Ny or move cel).
                _cross_keys_sap = list(crosses.keys())
                _m1_sap = (st.session_state.get("ebc_sa_o_mark1")
                           or (_cross_keys_sap[0]
                               if _cross_keys_sap else None))
                _m2_sap = st.session_state.get("ebc_sa_o_mark2")
                if (_m2_sap is None or _m2_sap == _m1_sap
                        or _m2_sap not in crosses):
                    _others = [k for k in _cross_keys_sap if k != _m1_sap]
                    _m2_sap = _others[0] if _others else _m1_sap
                _marks_outside_sap = False
                _eps_m = 1e-9
                for _m_n in (_m1_sap, _m2_sap):
                    if _m_n in crosses:
                        _mx, _my = crosses[_m_n]
                        _ax = _mx + sap_cel_x
                        _ay = _my + sap_cel_y
                        if not (_gx_min_sap - _eps_m <= _ax <= _gx_max_sap + _eps_m
                                and _gy_min_sap - _eps_m <= _ay <= _gy_max_sap + _eps_m):
                            _marks_outside_sap = True
                            break

                with sap_plot_col:
                    if sap_polys:
                        _bbox = _layer_bbox_mm(sap_polys, scale_sap)
                        if _bbox is not None:
                            _mx_min, _my_min, _mx_max, _my_max = _bbox
                            _mx_min += sap_cel_x; _mx_max += sap_cel_x
                            _my_min += sap_cel_y; _my_max += sap_cel_y
                            _eps = 1e-9
                            if (_mx_min < _gx_min_sap - _eps
                                    or _mx_max > _gx_max_sap + _eps
                                    or _my_min < _gy_min_sap - _eps
                                    or _my_max > _gy_max_sap + _eps):
                                _show_outside_pattern_notice()
                    if _marks_outside_sap:
                        _show_outside_pattern_notice(
                            "selected alignment mark outside of exposure grids "
                            "— increase Nx/Ny or move cel"
                        )
                    st.plotly_chart(fig_sap, width="stretch",
                                    key="ebc_sa_sap_chart")
                sap_ready = True

                # ── Overlayed ────────────────────────────────────────────────
                st.subheader(tr("Overlayed", "疊圖"))

                ov_plot_col, ov_ctrl_col = st.columns([7, 3])

                with ov_ctrl_col:
                    cross_labels = list(crosses.keys())
                    with st.container(border=True):
                        mark1_input, mark2_input = st.columns(2)
                        with mark1_input:
                            mark1 = st.selectbox(
                                tr("Mark 1", "標記 1"), cross_labels, key="ebc_sa_o_mark1")
                            m1_pos = crosses[mark1]
                            st.caption(
                                f"{tr('Original', '原始')}: ({m1_pos[0]:.4f}, "
                                f"{m1_pos[1]:.4f}) mm"
                            )
                            mark2_options = [
                                m for m in cross_labels if m != mark1]
                            current_m2 = st.session_state.get("ebc_sa_o_mark2")
                            if (current_m2 == mark1
                                    or current_m2 not in mark2_options):
                                st.session_state["ebc_sa_o_mark2"] = mark2_options[0]
                        with mark2_input:
                            mark2 = st.selectbox(
                                tr("Mark 2", "標記 2"), mark2_options, key="ebc_sa_o_mark2")
                            m2_pos = crosses[mark2]
                            st.caption(
                                f"{tr('Original', '原始')}: ({m2_pos[0]:.4f}, "
                                f"{m2_pos[1]:.4f}) mm"
                            )

                    # Auto-populate the Shift inputs.
                    # SAP's crosses sit at (preset + sap_cel) because they
                    # follow the cel; EP's at (preset + ep_shift). To put
                    # SAP's marks on EP's marks we need
                    #   preset + sap_cel + shift = preset + ep_shift
                    #   ⇒ shift = ep_shift − sap_cel.
                    shift_token = (ep_shift_x, ep_shift_y,
                                sap_cel_x, sap_cel_y)
                    if (st.session_state.get("ebc_sa_o_shift_token")
                            != shift_token):
                        st.session_state["ebc_sa_o_shift_x"] = float(
                            ep_shift_x - sap_cel_x)
                        st.session_state["ebc_sa_o_shift_y"] = float(
                            ep_shift_y - sap_cel_y)
                        st.session_state["ebc_sa_o_shift_token"] = shift_token
                        
                    with st.container(border=True):
                        st.caption(tr("Use these numbers in job3:",
                                      "在 job3 中使用以下數值："))
                        overlay_shift_x_input, overlay_shift_y_input = st.columns(2)
                        with overlay_shift_x_input:
                            overlay_shift_x = st.number_input(
                                tr("Shift x (mm)", "位移 x (mm)"), format="%.4f", step=0.0005,
                                key="ebc_sa_o_shift_x",
                            )
                        with overlay_shift_y_input:
                            overlay_shift_y = st.number_input(
                                tr("Shift y (mm)", "位移 y (mm)"), format="%.4f", step=0.0005,
                                key="ebc_sa_o_shift_y",
                            )

                        # Clearing the token forces the auto-populate block above
                        # to re-fire on the next rerun, resetting Shift x/y to the
                        # computed center (ep_shift − sap_cel).
                        def _recenter_overlay_shift():
                            st.session_state.pop("ebc_sa_o_shift_token", None)

                        st.button(
                            tr("Re-center", "重新置中"), on_click=_recenter_overlay_shift,
                            key="ebc_sa_o_recenter",
                            help=tr("Reset Shift x/y to the auto-computed center.",
                                    "將位移 x/y 重設為自動計算的中心值。"),
                        )

                # Build cells + polys in the Overlay frame (apply
                # overlay_shift to everything from the SAP frame).
                _ov_cells: list = []
                for _cell in _sa_cells:
                    _x0, _y0, _x1, _y1 = _cell
                    _ov_cells.append((
                        _x0 + overlay_shift_x, _y0 + overlay_shift_y,
                        _x1 + overlay_shift_x, _y1 + overlay_shift_y,
                    ))
                _ov_polys: list = []
                for _xs, _ys in _sa_polys_mm:
                    _ov_polys.append((
                        [_x + overlay_shift_x for _x in _xs],
                        [_y + overlay_shift_y for _y in _ys],
                    ))

                # Per-cell area + clipped polys (for filling only inside
                # grids and for the time calculator).
                if _sa_huge and _sa_cells:
                    # Vectorized per-cell area in the overlay frame; the mask
                    # lands at (sap_cel + overlay_shift), grid starts at
                    # (origin − half + overlay_shift).
                    _ov_areas = _cell_areas_binned(
                        sap_polys, scale_sap,
                        sap_cel_x + overlay_shift_x,
                        sap_cel_y + overlay_shift_y,
                        origin_x_sap - half_sap + overlay_shift_x,
                        origin_y_sap - half_sap + overlay_shift_y,
                        chip_size_sap, sap_nx_i, sap_ny_i)
                    _ov_clipped = []
                elif _ov_cells and _ov_polys:
                    _ov_areas, _ov_clipped = _polygon_clip_per_cell_mm(
                        _ov_polys, _ov_cells)
                else:
                    _ov_areas = [0.0] * len(_ov_cells)
                    _ov_clipped = []

                _pattern_idx = {
                    _i for _i, _a in enumerate(_ov_areas) if _a > 0
                }

                # Map each selected mark to its enclosing cell index (or
                # −1 if outside the grid bounds).
                _ov_gx_min = origin_x_sap - half_sap + overlay_shift_x
                _ov_gy_min = origin_y_sap - half_sap + overlay_shift_y
                _ov_gx_max = _ov_gx_min + sap_nx_i * chip_size_sap
                _ov_gy_max = _ov_gy_min + sap_ny_i * chip_size_sap
                _mark_pos_ov: list = []
                _mark_cell_idx: list = []
                for _m_n in (mark1, mark2):
                    if _m_n in crosses:
                        _mx, _my = crosses[_m_n]
                        _ax = _mx + sap_cel_x + overlay_shift_x
                        _ay = _my + sap_cel_y + overlay_shift_y
                        _mark_pos_ov.append((_ax, _ay))
                        _eps = 1e-9
                        if (_ov_gx_min - _eps <= _ax <= _ov_gx_max + _eps
                                and _ov_gy_min - _eps <= _ay <= _ov_gy_max + _eps):
                            _i = int((_ax - _ov_gx_min) // chip_size_sap)
                            _j = int((_ay - _ov_gy_min) // chip_size_sap)
                            _i = max(0, min(sap_nx_i - 1, _i))
                            _j = max(0, min(sap_ny_i - 1, _j))
                            # Iteration order in _sa_cells: i outer, j inner.
                            _mark_cell_idx.append(_i * sap_ny_i + _j)
                        else:
                            _mark_cell_idx.append(-1)

                _marks_outside_ov = any(
                    _idx < 0 for _idx in _mark_cell_idx)
                _mark_in_grid_set = {
                    _idx for _idx in _mark_cell_idx if _idx >= 0}
                _active_idx = _pattern_idx | _mark_in_grid_set
                _active_cells_ov = [
                    _ov_cells[_i] for _i in sorted(_active_idx)]
                _pattern_cells_ov = [
                    _ov_cells[_i] for _i in sorted(_pattern_idx)]
                _mark_only_cells_ov = [
                    _ov_cells[_i]
                    for _i in sorted(_mark_in_grid_set - _pattern_idx)
                ]

                fig_ov = go.Figure()

                # Existing pattern (with EP's user shift); kept un-clipped
                # for visual context (it's prior fabrication).
                if ep_polys and _ep_huge:
                    for _t in _mask_overlay_traces(
                            ep_polys, _gds_unit_to_mm, ep_shift_x, ep_shift_y,
                            "#1f77b4", tr("Existing Pattern", "既有圖案")):
                        fig_ov.add_trace(_t)
                elif ep_polys:
                    ep_xs_all, ep_ys_all = [], []
                    for xs, ys in ep_polys:
                        mxs = [ep_shift_x + p * _gds_unit_to_mm
                            for p in xs]
                        mys = [ep_shift_y + p * _gds_unit_to_mm
                            for p in ys]
                        ep_xs_all += mxs + [mxs[0], None]
                        ep_ys_all += mys + [mys[0], None]
                    fig_ov.add_trace(go.Scatter(
                        x=ep_xs_all, y=ep_ys_all,
                        mode="lines",
                        fill="toself",
                        line=dict(color="#1f77b4", width=0.5),
                        fillcolor="rgba(31,119,180,0.35)",
                        name="Existing Pattern",
                        hoverinfo="skip",
                    ))

                # Active grids: cells with patterns or with a selected
                # mark inside (empty grids are hidden).
                if _active_cells_ov:
                    _agx, _agy = [], []
                    for _x0, _y0, _x1, _y1 in _active_cells_ov:
                        _agx += [_x0, _x1, _x1, _x0, _x0, None]
                        _agy += [_y0, _y0, _y1, _y1, _y0, None]
                    fig_ov.add_trace(go.Scatter(
                        x=_agx, y=_agy,
                        mode="lines", fill="toself",
                        line=dict(color="#ff7f0e", width=1),
                        fillcolor="rgba(255,127,14,0.20)",
                        name=f"Active grids ({len(_active_cells_ov)})",
                        hoverinfo="skip",
                    ))

                # SAP pattern clipped to the grid (only what gets exposed).
                if _ov_clipped:
                    _pxs, _pys = [], []
                    for _xs, _ys in _ov_clipped:
                        if not _xs:
                            continue
                        _pxs += list(_xs) + [_xs[0], None]
                        _pys += list(_ys) + [_ys[0], None]
                    if _pxs:
                        fig_ov.add_trace(go.Scatter(
                            x=_pxs, y=_pys, mode="lines",
                            fill="toself",
                            line=dict(color="#2ca02c", width=0.5),
                            fillcolor="rgba(44,160,44,0.4)",
                            name="Second Alignment Pattern",
                            hoverinfo="skip",
                        ))

                # Selected marks (hide unused marks).
                if _mark_pos_ov:
                    _sel_x = [_p[0] for _p in _mark_pos_ov]
                    _sel_y = [_p[1] for _p in _mark_pos_ov]
                    fig_ov.add_trace(go.Scatter(
                        x=_sel_x, y=_sel_y,
                        mode="markers+text",
                        marker=dict(size=14, color="#9467bd",
                                    symbol="circle-open",
                                    line=dict(width=2)),
                        text=[mark1, mark2],
                        textposition="top right",
                        hovertemplate=("%{text}<br>"
                                    "(%{x:.4f}, %{y:.4f}) mm"
                                    "<extra></extra>"),
                        name="Selected marks",
                    ))

                # 4 outermost grid corners (hoverable, like in other
                # workflow modes). Use the active-cell bbox.
                if _active_cells_ov:
                    _g_xs = [v for c in _active_cells_ov
                             for v in (c[0], c[2])]
                    _g_ys = [v for c in _active_cells_ov
                             for v in (c[1], c[3])]
                    _gx0, _gx1 = min(_g_xs), max(_g_xs)
                    _gy0, _gy1 = min(_g_ys), max(_g_ys)
                    fig_ov.add_trace(go.Scatter(
                        x=[_gx0, _gx1, _gx1, _gx0],
                        y=[_gy0, _gy0, _gy1, _gy1],
                        mode="markers",
                        marker=dict(size=8, color="#ff7f0e",
                                    symbol="square"),
                        hovertemplate=("Grid corner<br>"
                                       "(%{x:.3f}, %{y:.3f}) mm"
                                       "<extra></extra>"),
                        name="Grid corners",
                        showlegend=False,
                    ))

                fig_ov.update_layout(
                    xaxis=dict(title="x (mm)"),
                    yaxis=dict(title="y (mm)",
                            scaleanchor="x", scaleratio=1),
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=600,
                    showlegend=True,
                    plot_bgcolor="white",
                )
                with ov_plot_col:
                    st.plotly_chart(fig_ov, width="stretch",
                                    key="ebc_sa_overlay_chart")

                # Time Calculator inputs come from the Overlay state:
                # pattern cells = cells where SAP polys land (after the
                # overlay shift) and mark-only cells = cells holding a
                # selected mark but no patterns. Shared grids (both a
                # mark and patterns) are kept in pattern_cells only, so
                # they're counted once for stage movement.
                # For dense layers, pass per-active-cell areas (aligned to
                # _pattern_cells_ov = the sorted active cells) so the time
                # calculator skips the infeasible per-polygon gdstk clip.
                _sa_precomp = (
                    [_ov_areas[_i] for _i in sorted(_pattern_idx)]
                    if _sa_huge else None
                )
                if _sa_huge:
                    st.caption(_dense_layer_note(sap_polys))
                _render_time_calculator(
                    "ebc_sa", _ov_polys, _pattern_cells_ov,
                    chip_size_mm=chip_size_sap,
                    dotmap=int(st.session_state["ebc_dotmap"]),
                    mark_cells=_mark_only_cells_ov,
                    mark_labels=[mark1, mark2],
                    mark_positions=_mark_pos_ov,
                    extra_help_under_total=tr(
                        "Note: an additional 1–2 hours is typically needed "
                        "to find the alignment marks (not included in "
                        "this estimate).",
                        "備註：尋找對準標記通常需額外 1–2 小時（未計入此估計）。",
                    ),
                    disabled=_marks_outside_ov,
                    disabled_reason=tr(
                        "One or more selected marks fall outside the "
                        "exposure grid. Increase Nx/Ny or adjust cel "
                        "before calculating.",
                        "一個或多個選定標記超出曝光網格範圍。請增加 Nx/Ny 或"
                        "調整 cel 後再計算。",
                    ),
                    precomputed_cell_areas=_sa_precomp,
                    coverage_layer=sap_polys if _sa_huge else None,
                    coverage_scale=scale_sap,
                    coverage_ox=sap_cel_x + overlay_shift_x,
                    coverage_oy=sap_cel_y + overlay_shift_y,
                )


render_page()
