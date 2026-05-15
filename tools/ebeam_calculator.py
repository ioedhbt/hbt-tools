"""
ebeam_calculator.py — E-beam lithography position / dose calculator.

Helps the user map chip-corner positions in the e-beam holder, set the
left-computer origin, and run per-mode workflows (dose-time test,
first exposure, second alignment).

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.
"""
from __future__ import annotations

__version__ = "1.0"

import math
import os
import tempfile
import streamlit as st
import plotly.graph_objects as go

try:
    import gdstk
except ImportError:  # pragma: no cover
    gdstk = None


# ─── Page header ─────────────────────────────────────────────────────────────

st.title(f"🧮 E-Beam Lithography Calculator (v{__version__})")
st.caption("Use this tool to calculate JEOL ELS-7000 EBL chip positions.")

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
    "ebc_chip_size": 0.6,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


# ─── Section 1: Chip Position in the E-beam Holder ───────────────────────────

st.header("Chip Position in the E-beam Holder")
st.caption("Check the positions of the chip corners — especially the "
           "bottom-left and top-right.")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Corner Positions")

    shape_mode = st.radio(
        "Shape", ["Rectangular", "Custom"],
        horizontal=True, key="ebc_shape_mode",
    )
    is_rect = (shape_mode == "Rectangular")

    # In rectangular mode the user picks which diagonal pair drives the
    # rectangle; the other pair is computed from it and shown disabled.
    bltr_active = True  # default: BL/TR is the editable pair
    if is_rect:
        diag = st.radio(
            "Editable diagonal", ["BL / TR", "BR / TL"],
            horizontal=True, key="ebc_diag_mode",
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

    def _corner_inputs(label: str, kx: str, ky: str, disabled: bool):
        st.markdown(f"**{label}**")
        cx, cy = st.columns(2)
        cx.number_input(
            "x", key=kx, format="%.3f", step=0.0005, disabled=disabled,
        )
        cy.number_input(
            "y", key=ky, format="%.3f", step=0.0005, disabled=disabled,
        )

    # 2×2 grid: top row = top-left, top-right; bottom row = bottom-left, bottom-right
    row_top = st.columns(2)
    with row_top[0]:
        _corner_inputs("Top Left (TL)",  "ebc_tl_x", "ebc_tl_y", disabled=disabled_tl)
    with row_top[1]:
        _corner_inputs("Top Right (TR)", "ebc_tr_x", "ebc_tr_y", disabled=disabled_tr)

    row_bot = st.columns(2)
    with row_bot[0]:
        _corner_inputs("Bottom Left (BL)",  "ebc_bl_x", "ebc_bl_y", disabled=disabled_bl)
    with row_bot[1]:
        _corner_inputs("Bottom Right (BR)", "ebc_br_x", "ebc_br_y", disabled=disabled_br)

with col_right:
    st.subheader("Chip Position")

    bl = (st.session_state["ebc_bl_x"], st.session_state["ebc_bl_y"])
    br = (st.session_state["ebc_br_x"], st.session_state["ebc_br_y"])
    tr = (st.session_state["ebc_tr_x"], st.session_state["ebc_tr_y"])
    tl = (st.session_state["ebc_tl_x"], st.session_state["ebc_tl_y"])

    # Loop back to bl to close the polygon for fill='toself'.
    poly_x = [bl[0], br[0], tr[0], tl[0], bl[0]]
    poly_y = [bl[1], br[1], tr[1], tl[1], bl[1]]

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
        x=[bl[0], br[0], tr[0], tl[0]],
        y=[bl[1], br[1], tr[1], tl[1]],
        mode="markers+text",
        marker=dict(size=10, color="#1f77b4"),
        text=["BL", "BR", "TR", "TL"],
        textposition=["bottom center", "bottom center",
                      "top center", "top center"],
        hovertemplate="%{text}: (%{x:.3f}, %{y:.3f})<extra></extra>",
        showlegend=False,
    ))

    xs = [bl[0], br[0], tr[0], tl[0]]
    ys = [bl[1], br[1], tr[1], tl[1]]
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
    st.plotly_chart(fig, use_container_width=True)


# ─── Section 2: Left Computer Setup ──────────────────────────────────────────

st.header("Left Computer Setup, in job1")
with st.expander("Setup Instructions", expanded=False):
    st.caption("Make sure you already have the `.cel` file. In `job1`: ")
    st.caption("1. Type `pc`.")
    st.caption("2. Select folder where the `.cel` file is, usually in `Desktop/IOED/hbt/your_folder`.")
    st.caption("3. Type the chip name, the same name as the `.cel` file.")
    st.caption("4. Set chip origin, usualy `10.0,10.0`. Using `0, 0` is difficult to see.")
    st.caption("5. Click `Ax: chip dot` (white), it will be changed to `Ax: stage (mm)` (green).")
    st.caption("6. Type `0.0001g` to set grid spacing to 100 nm.")
    with st.expander("Dose Time Testing", expanded=False):
        st.caption("7. Click File -> Load CEL. Enter cel name.")
        st.caption("8. Origin: `9.7,9.7`.")
    with st.expander("First Exposure", expanded=False):
        st.caption("7. Type `mc` to create grid points.")
        st.caption("8. Click the square grid (click `i` to zoom in, and `o` to zoom out, then click the screen with the mouse pointer if needed).")
        st.caption("9. Are you sure? -> `Y`, All `cel_name`? -> `N`.")
        st.caption("10. dx, dy: `0.6,0.6`. This is the grid distance from each other.")
        st.caption("11. Nx, Ny -> `18,18`, or `17,17` depending on the size of the pattern.")
        st.caption("12. X direction? `Y` -> Auto reverse? `N`")
        st.caption("13. Click File -> Load CEL. Enter cel name.")
        st.caption("14. Origin: `9.7,9.7`.")
    with st.expander("Second Alignment", expanded=False):
        st.caption("7. Type `mc` to create grid points.")
        st.caption("8. Click the square grid (click `i` to zoom in, and `o` to zoom out, then click the screen with the mouse pointer if needed).")
        st.caption("9. Are you sure? -> `Y`, All `cel_name`? -> `N`.")
        st.caption("10. dx, dy: `0.6,0.6`. This is the grid distance from each other.")
        st.caption("11. Nx, Ny -> `18,18`, or `17,17` depending on the size of the pattern.")
        st.caption("12. X direction? `Y` -> Auto reverse? `N`")
        st.caption("13. Click File -> Load CEL. Enter cel name.")
        st.caption("14. Origin: `9.7,9.7`.")
        st.caption("15. Click Menu -> Chip -> Reg-2 Mark (R2). Input the positions for the 2 marks.")
    st.caption("Click File -> save -> press enter. Type the file `.con` name, the same as the `.cel` file.")
    st.caption("If successful, the grids will be green, your folder should have `.ccc, .cbc, .con` files.")

c_ox, c_oy, c_cs = st.columns(3)
c_ox.number_input("Chip Origin x (mm)", key="ebc_origin_x",
                  format="%.3f", step=0.0005)
c_oy.number_input("Chip Origin y (mm)", key="ebc_origin_y",
                  format="%.3f", step=0.0005)
c_cs.number_input("Chip Size (mm)", key="ebc_chip_size",
                  format="%.3f", step=0.0005, min_value=0.001)


# ─── Section 3: GDS Mask Viewer ──────────────────────────────────────────────

st.header("GDS Mask")

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_POLY_LIMIT = 50_000


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


@st.cache_data(show_spinner="Parsing GDS…")
def _load_gds(file_bytes: bytes):
    """Parse GDS → (unit_meters, {cell_name: {(l,d): [(xs, ys), ...]}})."""
    # gdstk.read_gds requires a filesystem path, so spill to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        lib = gdstk.read_gds(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    out: dict = {}
    for cell in lib.top_level():
        by_layer: dict = {}
        for p in cell.get_polygons(depth=None):
            key = (int(p.layer), int(p.datatype))
            pts = p.points
            by_layer.setdefault(key, []).append(
                (pts[:, 0].tolist(), pts[:, 1].tolist())
            )
        out[cell.name] = by_layer
    return float(lib.unit), out


def _layer_trace(name: str, polys: list, color: str) -> go.Scatter:
    xs_all, ys_all = [], []
    for xs, ys in polys:
        xs_all.extend(xs); xs_all.append(xs[0]); xs_all.append(None)
        ys_all.extend(ys); ys_all.append(ys[0]); ys_all.append(None)
    return go.Scatter(
        x=xs_all, y=ys_all,
        mode="lines",
        fill="toself",
        line=dict(color=color, width=0.5),
        fillcolor=_hex_to_rgba(color, 0.3),
        name=name,
        hoverinfo="skip",
    )


def _layer_bbox_mm(polys: list, scale_to_mm: float):
    """Return (min_x, min_y, max_x, max_y) of polygons in mm, or None."""
    if not polys:
        return None
    min_x = min(min(xs) for xs, _ in polys) * scale_to_mm
    max_x = max(max(xs) for xs, _ in polys) * scale_to_mm
    min_y = min(min(ys) for _, ys in polys) * scale_to_mm
    max_y = max(max(ys) for _, ys in polys) * scale_to_mm
    return min_x, min_y, max_x, max_y


def _round_up_even(value: float) -> int:
    """Smallest even integer >= value, with a floor of 2."""
    n = max(1, math.ceil(value))
    return n if n % 2 == 0 else n + 1


# Module-level outputs consumed by workflow modes:
_gds_selected_polys: list = []  # list of (xs, ys) in GDS user units (e.g. µm)
_gds_unit_to_mm: float = 1e-3   # multiplier from GDS user units to mm
_gds_selected_token = None      # opaque ID of current (cell, layer) selection
_gds_cells_data: dict = {}      # full {cell_name: {(l,d): polys}} from upload
_gds_selected_cell_name = None  # currently selected top-cell name

if gdstk is None:
    st.warning("`gdstk` is not installed. Run `pip install gdstk` to enable "
               "GDS viewing.")
else:
    cells_data: dict = {}
    parse_error: str | None = None

    col_upload, col_select = st.columns([1, 1])

    with col_upload:
        gds_upload = st.file_uploader(
            "Upload .gds file", type=["gds"], key="ebc_gds_upload",
        )
        if gds_upload is not None:
            try:
                unit_m, cells_data = _load_gds(gds_upload.getvalue())
                _gds_unit_to_mm = unit_m * 1000.0
                _gds_cells_data = cells_data
            except Exception as e:
                parse_error = str(e)

    with col_select:
        has_data = bool(cells_data)
        cell_names = list(cells_data.keys()) if has_data else []

        selected_cell = st.selectbox(
            "Top Cell",
            cell_names if has_data else ["—"],
            key="ebc_gds_cell",
            disabled=not has_data,
        )
        _gds_selected_cell_name = selected_cell if has_data else None

        by_layer = (
            cells_data.get(selected_cell, {}) if has_data else {}
        )
        layer_keys = sorted(by_layer.keys())
        layer_options = {
            f"L{l}/D{d}  ({len(by_layer[(l, d)]):,} polys)": (l, d)
            for (l, d) in layer_keys
        }
        labels = list(layer_options.keys())

        selected_label = st.selectbox(
            "Layer to expose",
            labels if labels else ["—"],
            key="ebc_gds_layer",
            disabled=not labels,
        )

    if parse_error:
        st.error(f"Failed to parse GDS: {parse_error}")
    elif gds_upload is not None and not cells_data:
        st.info("No top-level cells found in this GDS.")
    elif has_data and not by_layer:
        st.info("Selected cell has no polygons.")
    elif labels:
        layer_key = layer_options[selected_label]
        polys = by_layer[layer_key]
        _gds_selected_polys = polys
        _gds_selected_token = (selected_cell, layer_key)
        n_polys = len(polys)

        render = True
        if n_polys > _POLY_LIMIT:
            st.warning(
                f"Selected layer contains {n_polys:,} polygons "
                f"(limit {_POLY_LIMIT:,}). Rendering may be slow."
            )
            render = st.checkbox("Render anyway", key="ebc_gds_force")

        if render:
            fig = go.Figure()
            fig.add_trace(_layer_trace(
                selected_label, polys, _PALETTE[0],
            ))
            fig.update_layout(
                xaxis=dict(title="x (µm)"),
                yaxis=dict(title="y (µm)",
                           scaleanchor="x", scaleratio=1),
                margin=dict(l=40, r=20, t=20, b=40),
                height=550,
                showlegend=True,
                legend=dict(itemsizing="constant"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ─── Section 4: Mode selector ────────────────────────────────────────────────

st.header("Workflow")

mode = st.radio(
    "Mode",
    ["Choose mode:","Dose Time Testing", "First Exposure", "Second Alignment"],
    horizontal=True,
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

if mode == "Dose Time Testing":

    _DT_DEFAULTS = {
        "ebc_dt_cel_x": 9.7, "ebc_dt_cel_y": 9.7,
        "ebc_dt_dx": 0.6, "ebc_dt_dy": 0.6,
        "ebc_dt_nx": 5, "ebc_dt_ny": 5,
        "ebc_dt_shift_x": 94.0, "ebc_dt_shift_y": 104.0,
    }
    for _k, _v in _DT_DEFAULTS.items():
        st.session_state.setdefault(_k, _v)

    chip_size = float(st.session_state["ebc_chip_size"])
    origin_x = float(st.session_state["ebc_origin_x"])
    origin_y = float(st.session_state["ebc_origin_y"])
    half = chip_size * 0.5
    chip_cx = (st.session_state["ebc_bl_x"]
               + st.session_state["ebc_tr_x"]) * 0.5
    chip_cy = (st.session_state["ebc_bl_y"]
               + st.session_state["ebc_tr_y"]) * 0.5

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
        st.markdown("**Cel Origin (mm) in job1**")
        cel_x = st.number_input("x", format="%.3f",
                                step=0.0005, key="ebc_dt_cel_x")
        cel_y = st.number_input("y", format="%.3f",
                                step=0.0005, key="ebc_dt_cel_y")
    with p2:
        st.markdown("**Increment (mm) in job3**")
        dx = st.number_input("dx", format="%.3f",
                             step=0.0005, key="ebc_dt_dx")
        dy = st.number_input("dy", format="%.3f",
                             step=0.0005, key="ebc_dt_dy")
    with p3:
        st.markdown("**Grid Count in job3**")
        Nx = st.number_input("Nx", min_value=1,
                             step=1, key="ebc_dt_nx")
        Ny = st.number_input("Ny", min_value=1,
                             step=1, key="ebc_dt_ny")
    with p4:
        st.markdown("**Initial Shift (mm) in job3**")
        shift_x = st.number_input("x", format="%.3f",
                                  step=0.0005, key="ebc_dt_shift_x")
        shift_y = st.number_input("y", format="%.3f",
                                  step=0.0005, key="ebc_dt_shift_y")

    Nx_i, Ny_i = int(Nx), int(Ny)
    grid_xs: list = []
    grid_ys: list = []
    for i in range(Nx_i):
        for j in range(Ny_i):
            gx = origin_x - half + i * dx + shift_x
            gy = origin_y - half + j * dy + shift_y
            grid_xs += [gx, gx + chip_size, gx + chip_size, gx, gx, None]
            grid_ys += [gy, gy, gy + chip_size, gy + chip_size, gy, None]

    # Mask cell overlay: place the selected GDS layer's (0,0) at
    # (cel_origin + i·dx + shift, cel_origin + j·dy + shift); polygon
    # coords convert from GDS user units to mm via _gds_unit_to_mm.
    mask_xs: list = []
    mask_ys: list = []
    if _gds_selected_polys:
        scale = _gds_unit_to_mm
        for i in range(Nx_i):
            for j in range(Ny_i):
                cx = cel_x + i * dx + shift_x
                cy = cel_y + j * dy + shift_y
                for xs, ys in _gds_selected_polys:
                    pxs = [cx + p * scale for p in xs]
                    pys = [cy + p * scale for p in ys]
                    mask_xs += pxs + [pxs[0], None]
                    mask_ys += pys + [pys[0], None]

    bl = (st.session_state["ebc_bl_x"], st.session_state["ebc_bl_y"])
    br = (st.session_state["ebc_br_x"], st.session_state["ebc_br_y"])
    tr = (st.session_state["ebc_tr_x"], st.session_state["ebc_tr_y"])
    tl = (st.session_state["ebc_tl_x"], st.session_state["ebc_tl_y"])

    plot_left, plot_right = st.columns(2)

    # ── Left plot: single grid at the setup origin (no shift) ─────────
    with plot_left:
        st.markdown("**Single Grid (no shift)**")
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
        if _gds_selected_polys:
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
                name="Mask",
                hoverinfo="skip",
            ))

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
        st.plotly_chart(fig_single, use_container_width=True)

    # ── Right plot: full Nx×Ny pattern overlaid on chip (with shift) ──
    with plot_right:
        st.markdown("**Chip Position with Grids (after shift)**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[bl[0], br[0], tr[0], tl[0], bl[0]],
            y=[bl[1], br[1], tr[1], tl[1], bl[1]],
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
                name="Mask",
                hoverinfo="skip",
            ))

        # Chip corner dots (hover to read coords)
        fig.add_trace(go.Scatter(
            x=[bl[0], br[0], tr[0], tl[0]],
            y=[bl[1], br[1], tr[1], tl[1]],
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

        all_x = [bl[0], br[0], tr[0], tl[0]] + [v for v in grid_xs if v is not None]
        all_y = [bl[1], br[1], tr[1], tl[1]] + [v for v in grid_ys if v is not None]
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
        st.plotly_chart(fig, use_container_width=True)

elif mode == "First Exposure":
    _FE_DEFAULTS = {
        "ebc_fe_cel_x": 9.7, "ebc_fe_cel_y": 9.7,
        "ebc_fe_nx": 20, "ebc_fe_ny": 20,
        "ebc_fe_shift_x": 94.0, "ebc_fe_shift_y": 104.0,
    }
    for _k, _v in _FE_DEFAULTS.items():
        st.session_state.setdefault(_k, _v)

    chip_size_v = float(st.session_state["ebc_chip_size"])
    origin_x_v = float(st.session_state["ebc_origin_x"])
    origin_y_v = float(st.session_state["ebc_origin_y"])
    half_v = chip_size_v * 0.5

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
                f"Mask layer size: {layer_w:.3f} × {layer_h:.3f} mm"
                f"  →  required grids: {Nx_req} × {Ny_req} "
                f"(chip-size tile = {chip_size_v:.3f} mm, "
                f"cel = {cel_x_v:.3f}, {cel_y_v:.3f})"
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
        st.markdown("**Cel Origin (mm) in job1**")
        cel_x = st.number_input("x", format="%.3f", step=0.0005,
                                key="ebc_fe_cel_x")
        cel_y = st.number_input("y", format="%.3f", step=0.0005,
                                key="ebc_fe_cel_y")
    with p2:
        st.markdown("**Grid Count in job1 (make sure everything is inside the chip)**")
        Nx = st.number_input("Nx", min_value=1, step=1,
                             key="ebc_fe_nx")
        Ny = st.number_input("Ny", min_value=1, step=1,
                             key="ebc_fe_ny")
    with p3:
        st.markdown("**Shift (mm) in job3**")
        shift_x = st.number_input("x", format="%.3f", step=0.0005,
                                  key="ebc_fe_shift_x")
        shift_y = st.number_input("y", format="%.3f", step=0.0005,
                                  key="ebc_fe_shift_y")

    Nx_i, Ny_i = int(Nx), int(Ny)

    # Grid pattern — Nx × Ny tiles of chip_size, packed edge-to-edge,
    # representing the e-beam writable area (no mask duplication).
    grid_xs: list = []
    grid_ys: list = []
    for i in range(Nx_i):
        for j in range(Ny_i):
            gx = origin_x_v - half_v + i * chip_size_v + shift_x
            gy = origin_y_v - half_v + j * chip_size_v + shift_y
            grid_xs += [gx, gx + chip_size_v, gx + chip_size_v,
                        gx, gx, None]
            grid_ys += [gy, gy, gy + chip_size_v,
                        gy + chip_size_v, gy, None]

    # Single mask placement: GDS (0,0) lands at (cel + shift).
    mask_xs: list = []
    mask_ys: list = []
    if _gds_selected_polys:
        scale = _gds_unit_to_mm
        for xs, ys in _gds_selected_polys:
            pxs = [cel_x + shift_x + p * scale for p in xs]
            pys = [cel_y + shift_y + p * scale for p in ys]
            mask_xs += pxs + [pxs[0], None]
            mask_ys += pys + [pys[0], None]

    bl = (st.session_state["ebc_bl_x"], st.session_state["ebc_bl_y"])
    br = (st.session_state["ebc_br_x"], st.session_state["ebc_br_y"])
    tr = (st.session_state["ebc_tr_x"], st.session_state["ebc_tr_y"])
    tl = (st.session_state["ebc_tl_x"], st.session_state["ebc_tl_y"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[bl[0], br[0], tr[0], tl[0], bl[0]],
        y=[bl[1], br[1], tr[1], tl[1], bl[1]],
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
            name="Mask",
            hoverinfo="skip",
        ))

    # Chip corner dots (hover to read coords)
    fig.add_trace(go.Scatter(
        x=[bl[0], br[0], tr[0], tl[0]],
        y=[bl[1], br[1], tr[1], tl[1]],
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

    all_x = ([bl[0], br[0], tr[0], tl[0]]
             + [v for v in grid_xs if v is not None])
    all_y = ([bl[1], br[1], tr[1], tl[1]]
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
    )
    st.plotly_chart(fig, use_container_width=True)

elif mode == "Second Alignment":
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

    preset_name = st.radio(
        "Cross-position preset",
        list(_CROSS_PRESETS.keys()),
        horizontal=True,
        key="ebc_sa_preset",
    )

    if preset_name == "Custom":
        # Custom preset: collect 2 alignment-mark positions from the
        # user and use them as `crosses` for the rest of the flow.
        _CUSTOM_DEFAULTS = {
            "ebc_sa_custom_m1_x": 0.0, "ebc_sa_custom_m1_y": 0.0,
            "ebc_sa_custom_m2_x": 6.0, "ebc_sa_custom_m2_y": 6.0,
        }
        for _k, _v in _CUSTOM_DEFAULTS.items():
            st.session_state.setdefault(_k, _v)

        cm1, cm2 = st.columns(2)
        with cm1:
            st.markdown("**Mark M1 (mm)**")
            m1cx, m1cy = st.columns(2)
            m1_x = m1cx.number_input(
                "x", format="%.4f", step=0.0005,
                key="ebc_sa_custom_m1_x")
            m1_y = m1cy.number_input(
                "y", format="%.4f", step=0.0005,
                key="ebc_sa_custom_m1_y")
        with cm2:
            st.markdown("**Mark M2 (mm)**")
            m2cx, m2cy = st.columns(2)
            m2_x = m2cx.number_input(
                "x", format="%.4f", step=0.0005,
                key="ebc_sa_custom_m2_x")
            m2_y = m2cy.number_input(
                "y", format="%.4f", step=0.0005,
                key="ebc_sa_custom_m2_y")

        crosses = {"M1": (m1_x, m1_y), "M2": (m2_x, m2_y)}
    else:
        crosses = _CROSS_PRESETS[preset_name]

    if not crosses:
        st.info("No cross positions defined for this preset.")
    else:
        st.caption(f"{len(crosses)} cross positions loaded "
                   "(chip-relative, mm).")
        rows = [{"Cross": k, "x (mm)": v[0], "y (mm)": v[1]}
                for k, v in crosses.items()]
        with st.expander("Show cross positions table"):
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
                anchor = st.selectbox(
                    "Move:", cross_labels,
                    key=f"{key_prefix}_anchor",
                )
                anchor_x, anchor_y = crosses[anchor]
                st.session_state.setdefault(
                    f"{key_prefix}_target_x", float(anchor_x))
                st.session_state.setdefault(
                    f"{key_prefix}_target_y", float(anchor_y))
                st.caption("Mark position as seen on the SEM:")
                target_x_input, target_y_input = st.columns(2)
                with target_x_input:
                    target_x = st.number_input(
                        "Target x in SEM (mm)", format="%.4f", step=0.0005,
                        key=f"{key_prefix}_target_x",
                    )
                with target_y_input:
                    target_y = st.number_input(
                        "Target y in SEM (mm)", format="%.4f", step=0.0005,
                        key=f"{key_prefix}_target_y",
                    )

            shift_x = target_x - anchor_x
            shift_y = target_y - anchor_y

            fig = go.Figure()
            if polys:
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
            )
            with plot_col:
                st.plotly_chart(fig, use_container_width=True,
                                key=f"{key_prefix}_chart")
            return shift_x, shift_y

        # ── Existing Pattern ─────────────────────────────────────────
        st.subheader("Existing Pattern on Chip")

        ep_polys: list = []
        ep_shift_x = 0.0
        ep_shift_y = 0.0
        ep_ready = False

        ep_ctrl_col, ep_plot_col = st.columns([3, 7])

        with ep_ctrl_col:
            if not _gds_cells_data:
                st.caption("No GDS file uploaded — showing "
                           "crosses only.")
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
                    "Existing pattern layer",
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
        st.subheader("Second Alignment Pattern")

        sap_polys = _gds_selected_polys

        _SAP_DEFAULTS = {
            "ebc_sa_sap_cel_x": 9.7, "ebc_sa_sap_cel_y": 9.7,
            "ebc_sa_sap_nx": 20, "ebc_sa_sap_ny": 20,
        }
        for _k, _v in _SAP_DEFAULTS.items():
            st.session_state.setdefault(_k, _v)

        chip_size_sap = float(st.session_state["ebc_chip_size"])
        origin_x_sap = float(st.session_state["ebc_origin_x"])
        origin_y_sap = float(st.session_state["ebc_origin_y"])
        half_sap = chip_size_sap * 0.5

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
                        f"Mask layer size: {bb_w:.3f} × {bb_h:.3f} mm"
                        f"  →  required grids: {nx_req} × {ny_req}"
                    )
                    sap_token = (_gds_selected_token, chip_size_sap,
                                 cel_x_v, cel_y_v)
                    if (st.session_state.get("ebc_sa_sap_nxny_token")
                            != sap_token):
                        st.session_state["ebc_sa_sap_nx"] = int(nx_req)
                        st.session_state["ebc_sa_sap_ny"] = int(ny_req)
                        st.session_state["ebc_sa_sap_nxny_token"] = sap_token
            else:
                st.caption("No GDS layer selected — showing "
                           "crosses only.")

            sp1, sp2 = st.columns(2)
            with sp1:
                st.markdown("**Cel Origin (mm) in job1**")
                sap_cel_x = st.number_input(
                    "x", format="%.3f", step=0.0005,
                    key="ebc_sa_sap_cel_x")
                sap_cel_y = st.number_input(
                    "y", format="%.3f", step=0.0005,
                    key="ebc_sa_sap_cel_y")
            with sp2:
                st.markdown("**Grid Count in job1 (make sure everything is inside the chip)**")
                sap_nx = st.number_input(
                    "Nx", min_value=1, step=1,
                    key="ebc_sa_sap_nx")
                sap_ny = st.number_input(
                    "Ny", min_value=1, step=1,
                    key="ebc_sa_sap_ny")

        sap_nx_i = int(sap_nx)
        sap_ny_i = int(sap_ny)
        scale_sap = _gds_unit_to_mm

        # Plot: only render grid + mask if a GDS layer is selected;
        # crosses always render (and follow the cel origin).
        fig_sap = go.Figure()
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
            fig_sap.add_trace(go.Scatter(
                x=sap_grid_xs, y=sap_grid_ys,
                mode="lines",
                fill="toself",
                line=dict(color="#ff7f0e", width=1),
                fillcolor="rgba(255,127,14,0.15)",
                name=f"Grid ({sap_nx_i}×{sap_ny_i})",
                hoverinfo="skip",
            ))

            sap_mask_xs: list = []
            sap_mask_ys: list = []
            for xs, ys in sap_polys:
                pxs = [sap_cel_x + p * scale_sap for p in xs]
                pys = [sap_cel_y + p * scale_sap for p in ys]
                sap_mask_xs += pxs + [pxs[0], None]
                sap_mask_ys += pys + [pys[0], None]
            fig_sap.add_trace(go.Scatter(
                x=sap_mask_xs, y=sap_mask_ys,
                mode="lines",
                fill="toself",
                line=dict(color="#2ca02c", width=0.5),
                fillcolor="rgba(44,160,44,0.5)",
                name="Mask",
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
        )
        with sap_plot_col:
            st.plotly_chart(fig_sap, use_container_width=True,
                            key="ebc_sa_sap_chart")
        sap_ready = True

        # ── Overlayed ────────────────────────────────────────────────
        st.subheader("Overlayed")

        ov_plot_col, ov_ctrl_col = st.columns([7, 3])

        with ov_ctrl_col:
            cross_labels = list(crosses.keys())
            mark1_input, mark2_input = st.columns(2)
            with mark1_input:
                mark1 = st.selectbox(
                    "Mark 1", cross_labels, key="ebc_sa_o_mark1")
                m1_pos = crosses[mark1]
                st.caption(
                    f"Original: ({m1_pos[0]:.4f}, "
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
                    "Mark 2", mark2_options, key="ebc_sa_o_mark2")
                m2_pos = crosses[mark2]
                st.caption(
                    f"Original: ({m2_pos[0]:.4f}, "
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

            st.caption("Use these numbers in job3:")
            overlay_shift_x_input, overlay_shift_y_input = st.columns(2)
            with overlay_shift_x_input:
                overlay_shift_x = st.number_input(
                    "Shift x (mm)", format="%.4f", step=0.0005,
                    key="ebc_sa_o_shift_x",
                )
            with overlay_shift_y_input:
                overlay_shift_y = st.number_input(
                    "Shift y (mm)", format="%.4f", step=0.0005,
                    key="ebc_sa_o_shift_y",
                )

            # Clearing the token forces the auto-populate block above
            # to re-fire on the next rerun, resetting Shift x/y to the
            # computed center (ep_shift − sap_cel).
            def _recenter_overlay_shift():
                st.session_state.pop("ebc_sa_o_shift_token", None)

            st.button(
                "Re-center", on_click=_recenter_overlay_shift,
                key="ebc_sa_o_recenter",
                help="Reset Shift x/y to the auto-computed center.",
            )

        fig_ov = go.Figure()

        # Existing pattern (with EP's user shift); skipped if no file.
        if ep_polys:
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

        # SAP mask (at cel + overlay shift); skipped if no file.
        if sap_polys:
            sap_ov_xs, sap_ov_ys = [], []
            for xs, ys in sap_polys:
                mxs = [sap_cel_x + overlay_shift_x
                       + p * _gds_unit_to_mm for p in xs]
                mys = [sap_cel_y + overlay_shift_y
                       + p * _gds_unit_to_mm for p in ys]
                sap_ov_xs += mxs + [mxs[0], None]
                sap_ov_ys += mys + [mys[0], None]
            fig_ov.add_trace(go.Scatter(
                x=sap_ov_xs, y=sap_ov_ys,
                mode="lines",
                fill="toself",
                line=dict(color="#2ca02c", width=0.5),
                fillcolor="rgba(44,160,44,0.4)",
                name="Second Alignment Pattern",
                hoverinfo="skip",
            ))

        # Show only the two selected marks (in the SAP frame:
        # preset + sap_cel + overlay_shift). Hide the rest.
        sel_marks = [mark1, mark2]
        sel_x = [crosses[m][0] + sap_cel_x + overlay_shift_x
                 for m in sel_marks]
        sel_y = [crosses[m][1] + sap_cel_y + overlay_shift_y
                 for m in sel_marks]
        fig_ov.add_trace(go.Scatter(
            x=sel_x, y=sel_y,
            mode="markers+text",
            marker=dict(size=14, color="#9467bd",
                        symbol="circle-open",
                        line=dict(width=2)),
            text=sel_marks,
            textposition="top right",
            hovertemplate=("%{text}<br>"
                           "(%{x:.4f}, %{y:.4f}) mm"
                           "<extra></extra>"),
            name="Selected marks",
        ))

        fig_ov.update_layout(
            xaxis=dict(title="x (mm)"),
            yaxis=dict(title="y (mm)",
                       scaleanchor="x", scaleratio=1),
            margin=dict(l=40, r=20, t=20, b=40),
            height=600,
            showlegend=True,
        )
        with ov_plot_col:
            st.plotly_chart(fig_ov, use_container_width=True,
                            key="ebc_sa_overlay_chart")