"""
ebeam_calculator.py — E-beam lithography position / dose calculator.

Helps the user map chip-corner positions in the e-beam holder, set the
left-computer origin, and run per-mode workflows (dose-time test,
first exposure, second alignment).

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.
"""
from __future__ import annotations

__version__ = "1.4"

import gc
import hashlib
import math
import struct
import numpy as np
import streamlit as st
import plotly.graph_objects as go

try:
    import gdstk
except ImportError:  # pragma: no cover
    gdstk = None


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

st.title(_EBL_TITLE)
st.caption(_EBL_DESC)

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
def _chip_corner_guide_png() -> bytes:
    """Static reference sketch explaining the chip-corner labelling + the two
    diagonals that the "Editable diagonal" radio chooses between.  Drawn with
    matplotlib (Agg, no pyplot global state) and cached — it never changes."""
    import io
    from matplotlib.figure import Figure

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
            label="BL–TR diagonal")
    ax.plot([x1, x0], [y0, y1], ls="--", color=ORANGE, lw=1.6, zorder=2,
            label="BR–TL diagonal")

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


# ─── Section 1: Chip Position in the E-beam Holder ───────────────────────────
with st.container(border=True):
    st.header("Chip Position in the E-beam Holder",
              help="Check the positions of the chip corners - especially the "
                   "bottom-left and top-right.",
              anchor=False)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Corner Positions")

        with st.expander(_EBL_CORNER_GUIDE, expanded=False):
            st.image(_chip_corner_guide_png(), width="stretch")
            st.caption(_EBL_CORNER_NOTE)

        shape_mode = segmented_radio(
            "Shape", ["Rectangular", "Custom"],
            key="ebc_shape_mode",
        )
        is_rect = (shape_mode == "Rectangular")

        # In rectangular mode the user picks which diagonal pair drives the
        # rectangle; the other pair is computed from it and shown disabled.
        bltr_active = True  # default: BL/TR is the editable pair
        if is_rect:
            diag = segmented_radio(
                "Editable diagonal", ["BL / TR", "BR / TL"],
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
                _corner_inputs("Top Left (TL)",  "ebc_tl_x", "ebc_tl_y", disabled=disabled_tl)
        with row_top[1]:
            with st.container(border=True):
                _corner_inputs("Top Right (TR)", "ebc_tr_x", "ebc_tr_y", disabled=disabled_tr)

        row_bot = st.columns(2)
        with row_bot[0]:
            with st.container(border=True):
                _corner_inputs("Bottom Left (BL)",  "ebc_bl_x", "ebc_bl_y", disabled=disabled_bl)
        with row_bot[1]:
            with st.container(border=True):
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
        st.plotly_chart(fig, width="stretch")


# ─── Section 2: Left Computer Setup ──────────────────────────────────────────
with st.container(border=True):
    st.header("Left Computer Setup")
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

    c_oxy, c_csdm = st.columns(2)
    with c_oxy:
        with st.container(border=True):
            st.markdown("**In Job 1**")
            c_ox, c_oy = st.columns(2)
            c_ox.number_input("Chip Origin x (mm)", key="ebc_origin_x",
                        format="%.3f", step=0.0005, min_value=0.0)
            c_oy.number_input("Chip Origin y (mm)", key="ebc_origin_y",
                        format="%.3f", step=0.0005, min_value=0.0)
    with c_csdm:
        with st.container(border=True):
            st.markdown("**In Job 2**")
            c_cs, c_dm, c_res = st.columns(3)
            c_cs.selectbox(
                "Chip Size (μm)",
                [75, 150, 300, 600, 1200],
                key="ebc_chip_size",
            )
            c_dm.selectbox(
                "Dotmap",
                [20000, 60000, 240000],
                key="ebc_dotmap",
            )
            # Resolution = chip size (μm) / dotmap. Stored in μm under
            # ebc_resolution; also shown in nm for readability.
            _chip_um = int(st.session_state["ebc_chip_size"])
            _dotmap = int(st.session_state["ebc_dotmap"])
            st.session_state["ebc_resolution"] = _chip_um / _dotmap
            _res_um = st.session_state["ebc_resolution"]
            c_res.markdown("**Resolution**")
            c_res.markdown(f"{_chip_um} μm / {_dotmap} dots = {_res_um * 1000:g} nm")



# ─── Section 3: GDS Mask Viewer ──────────────────────────────────────────────
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# Safety budgets so a pathological GDS still fails with a clear message
# instead of OOM-killing the app on memory-limited hosts (Streamlit
# Community Cloud ≈ 1 GB). Geometry is stored as flat float64 arrays
# (~16 B per vertex / per placement row, no per-polygon Python objects),
# so 20 M of either ≈ 0.32 GB resident — the practical ceiling once the
# upload buffer (held live by the file_uploader widget) and the ~150 MB
# import baseline are accounted for.
_MAX_SRC_VERTICES = 20_000_000   # polygon vertices stored while parsing
_MAX_OFFSET_ROWS = 20_000_000    # reference placements after flattening
_MAX_VERTICES = 20_000_000       # vertices when expanding a layer flat

# Above this polygon count a layer is too dense to draw individually in
# the browser (Plotly chokes well before this) or to clip per-grid with
# gdstk on a 1 GB host. The viewer and workflow modes fall back to a
# bounding-box outline + a vectorized area estimate instead.
_POLY_LIMIT = 50_000


class _PolyLayer:
    """All polygons of one (layer, datatype), stored as flat float64
    coordinate arrays plus per-polygon start offsets.

    Behaves like the old ``list[(xs, ys)]`` (iteration yields per-polygon
    ``(xs, ys)`` numpy views, plus ``len`` and truthiness) so existing
    consumers work unchanged, but uses ~16 B/vertex with no per-polygon
    Python objects — the difference between ~250 MB and ~1 GB on a
    multi-million-polygon mask. Also offers vectorized bbox / area
    helpers used by the large-layer fast paths.
    """
    __slots__ = ("cx", "cy", "starts")

    def __init__(self, cx, cy, starts):
        self.cx = cx          # float64, all x concatenated
        self.cy = cy          # float64, all y concatenated
        self.starts = starts  # int64, length n_polys + 1 (CSR-style)

    def __len__(self):
        return int(self.starts.size - 1)

    def __bool__(self):
        return self.starts.size > 1

    def __iter__(self):
        cx, cy, st = self.cx, self.cy, self.starts
        for i in range(st.size - 1):
            a, b = int(st[i]), int(st[i + 1])
            yield cx[a:b], cy[a:b]

    def __getitem__(self, i):
        st = self.starts
        a, b = int(st[i]), int(st[i + 1])
        return self.cx[a:b], self.cy[a:b]

    def bbox(self):
        """(min_x, min_y, max_x, max_y) in source units, or None."""
        if self.cx.size == 0:
            return None
        return (float(self.cx.min()), float(self.cy.min()),
                float(self.cx.max()), float(self.cy.max()))

    def poly_areas(self):
        """|shoelace| area per polygon (source units²), vectorized."""
        cx, cy, st = self.cx, self.cy, self.starts
        if st.size <= 1:
            return np.zeros(0)
        nxt = np.arange(cx.size) + 1
        nxt[st[1:] - 1] = st[:-1]          # wrap each polygon's last → first
        cross = cx * cy[nxt] - cx[nxt] * cy
        return np.abs(0.5 * np.add.reduceat(cross, st[:-1]))

    def first_vertices(self):
        """First vertex of every polygon as (x, y) arrays."""
        st = self.starts[:-1]
        return self.cx[st], self.cy[st]


def _apply_ref_transform(cx, cy, rotation, magnification, x_reflection):
    """Apply a reference's magnification → x_reflection → rotation
    (translation handled separately via origin + repetition offsets).
    Identity transforms return the inputs unchanged (no copy) — callers
    treat the results as read-only."""
    if not rotation and magnification == 1.0 and not x_reflection:
        return cx, cy
    x = cx * magnification
    y = cy * magnification
    if x_reflection:
        y = -y
    if rotation:
        c = math.cos(rotation)
        s = math.sin(rotation)
        x, y = c * x - s * y, s * x + c * y
    return x, y


class _InstancedLayer:
    """A layer kept in its *instanced* form: one or more groups, each a
    small base pattern (``_PolyLayer``) plus an (K, 2) array of placement
    offsets. This preserves the array/AREF structure that the GDS already
    encodes instead of expanding it — so a unit cell tiled 150 k× stays a
    ~30-polygon base + a 150 k-row offset table, not 4.6 M polygons.

    Exposes ``len`` (total expanded polygon count), truthiness, and
    ``bbox`` so it drops into the same gating/auto-fit code as
    ``_PolyLayer``; the renderer and area helpers special-case it to draw
    the base once + the array extent and to compute area as
    base_area × instance_count.
    """
    __slots__ = ("groups", "_n", "_area", "_bbox", "_inst", "_base")

    def __init__(self, groups):
        # groups: list of (base_cx, base_cy, base_starts, offsets[K,2])
        self.groups = groups
        n = 0
        inst = 0
        base = 0
        area = 0.0
        xs0 = []
        ys0 = []
        xs1 = []
        ys1 = []
        for bcx, bcy, bst, off in groups:
            b = _PolyLayer(bcx, bcy, bst)
            k = int(off.shape[0])
            np_base = len(b)
            n += np_base * k
            inst += k
            base += np_base
            area += float(b.poly_areas().sum()) * k
            bb = b.bbox()
            if bb is not None and k:
                xs0.append(bb[0] + float(off[:, 0].min()))
                ys0.append(bb[1] + float(off[:, 1].min()))
                xs1.append(bb[2] + float(off[:, 0].max()))
                ys1.append(bb[3] + float(off[:, 1].max()))
        self._n = n
        self._inst = inst
        self._base = base
        self._area = area
        self._bbox = (None if not xs0 else
                      (min(xs0), min(ys0), max(xs1), max(ys1)))

    def __len__(self):
        return self._n

    def __bool__(self):
        return self._n > 0

    def bbox(self):
        return self._bbox

    def total_area_units2(self):
        return self._area

    def instance_count(self):
        return self._inst

    def base_poly_count(self):
        return self._base


# ─── Minimal streaming GDSII parser ──────────────────────────────────────────
# ``gdstk.read_gds`` materializes one C++ object per element and per
# reference (~190 B each): a 180 MB CAD dump holding ~6 M SREF
# placements needs >1.1 GB before the app sees a single polygon — an
# instant OOM on a 1 GB host. The reader below walks the raw GDSII
# records instead and stores only flat numpy arrays (~16 B/placement).
# Runs of byte-identical elements (how CAD tools emit arrayed SREFs and
# boundary dumps: the same few records repeated, only the XY payload
# changing) are detected and decoded in bulk with numpy at
# memory-bandwidth speed, so multi-million-reference files parse in
# seconds. gdstk is still used for geometry booleans (per-grid clipping)
# and for PATH → polygon conversion.

_REC = struct.Struct(">HH")

_T_UNITS = 0x0305;    _T_ENDLIB = 0x0400
_T_BGNSTR = 0x0502;   _T_STRNAME = 0x0606;  _T_ENDSTR = 0x0700
_T_BOUNDARY = 0x0800; _T_PATH = 0x0900
_T_SREF = 0x0A00;     _T_AREF = 0x0B00;     _T_TEXT = 0x0C00
_T_LAYER = 0x0D02;    _T_DATATYPE = 0x0E02; _T_WIDTH = 0x0F03
_T_XY = 0x1003;       _T_ENDEL = 0x1100;    _T_SNAME = 0x1206
_T_COLROW = 0x1302;   _T_NODE = 0x1500;     _T_STRANS = 0x1A01
_T_MAG = 0x1B05;      _T_ANGLE = 0x1C05;    _T_PATHTYPE = 0x2102
_T_BOX = 0x2D00;      _T_BGNEXTN = 0x3003;  _T_ENDEXTN = 0x3103

_EL_START = frozenset((_T_BOUNDARY, _T_PATH, _T_SREF, _T_AREF, _T_TEXT,
                       _T_NODE, _T_BOX))


def _gds_real8(b) -> float:
    """Decode one GDSII 8-byte excess-64 real."""
    exp = b[0]
    mant = int.from_bytes(bytes(b[1:8]), "big")
    v = mant * 16.0 ** ((exp & 0x7F) - 64) / float(1 << 56)
    return -v if exp & 0x80 else v


def _element_run(a, mv, p0, blk, xy_off, xy_len):
    """Bulk-decode the run of element blocks following the one at ``p0``.

    A "run" is consecutive ``blk``-byte blocks whose bytes equal the
    template block at ``p0`` everywhere except the XY payload
    (``xy_off``/``xy_len`` relative to the block start). Returns
    ``(next_pos, chunks)`` where each chunk is a flat int64 array of raw
    XY words from many blocks. A zero-copy memoryview probe of the very
    next block keeps the cost negligible when there is no run (mixed
    hand-drawn files); real runs are then consumed in ~4 MB numpy passes.
    """
    n = a.size
    q = p0 + blk
    pre = mv[p0:p0 + xy_off]
    post = mv[p0 + xy_off + xy_len:p0 + blk]
    if (q + blk > n
            or mv[q:q + xy_off] != pre
            or mv[q + xy_off + xy_len:q + blk] != post):
        return q, []

    tmpl = a[p0:p0 + blk]
    keep = np.ones(blk, dtype=bool)
    keep[xy_off:xy_off + xy_len] = False
    tfix = tmpl[keep]
    chunks = []
    step = max(1, (1 << 22) // blk)      # compare ~4 MB per numpy pass
    while q + blk <= n:
        k = min(step, (n - q) // blk)
        arr = a[q:q + k * blk].reshape(k, blk)
        ok = (arr[:, keep] == tfix).all(axis=1)
        r = k if bool(ok.all()) else int(np.argmin(ok))
        if r:
            xy = np.ascontiguousarray(arr[:r, xy_off:xy_off + xy_len])
            # int32 (exact: GDS coords are i4) — half the transient RAM
            # of int64 while the chunks wait for consolidation.
            chunks.append(xy.reshape(-1).view(">i4").astype(np.int32))
            q += r * blk
        if r < k:
            break
    return q, chunks


def _consolidate_cell(polys_acc, paths, refs_acc, db_user):
    """Merge one structure's parse accumulators into a raw-cell dict:

        {"polys": {(layer, datatype): (cx, cy, starts)},
         "refs": [(child_name, rotation_rad, magnification,
                   x_reflection, offsets[K, 2]), ...]}

    with all coordinates scaled from database to GDS user units
    (float64). PATH elements are converted to polygons here via gdstk
    geometry objects (no gdstk library/file involved)."""
    for (layer, dt, width, ptype, bext, eext, pts) in paths:
        if gdstk is None:
            continue
        ends = {0: "flush", 1: "round", 2: "extended"}.get(
            ptype, (float(bext), float(eext)))
        try:
            fp = gdstk.FlexPath(pts, abs(float(width)), ends=ends)
            path_polys = fp.to_polygons()
        except Exception:
            continue
        slot = polys_acc.setdefault((layer, dt), ([], [], []))
        for poly in path_polys:
            pp = poly.points
            slot[0].append(pp[:, 0].copy())
            slot[1].append(pp[:, 1].copy())
            slot[2].append(pp.shape[0])

    polys = {}
    for key, (cxs, cys, szs) in polys_acc.items():
        cx = np.concatenate(cxs).astype(np.float64)
        cy = np.concatenate(cys).astype(np.float64)
        cxs.clear(); cys.clear()          # release raw int chunks now
        cx *= db_user
        cy *= db_user
        sizes = np.concatenate(
            [np.atleast_1d(np.asarray(s, dtype=np.int64)) for s in szs])
        starts = np.zeros(sizes.size + 1, dtype=np.int64)
        np.cumsum(sizes, out=starts[1:])
        polys[key] = (cx, cy, starts)

    refs = []
    for (child, rot, mag, refl), lst in refs_acc.items():
        scalars = [e for e in lst if type(e) is tuple]
        parts = []
        if scalars:
            parts.append(np.asarray(scalars, dtype=np.float64))
        parts.extend(np.asarray(e, dtype=np.float64).reshape(-1, 2)
                     for e in lst if type(e) is not tuple)
        if not parts:
            continue
        lst.clear()                       # release raw int chunks now
        off = parts[0] if len(parts) == 1 else np.concatenate(parts)
        del parts
        off *= db_user
        refs.append((child, rot, mag, refl, off))
    return {"polys": polys, "refs": refs}


def _parse_gds(buf):
    """Single-pass streaming parse of a GDSII byte buffer.

    Returns ``(unit_meters, {structure_name: raw cell dict})`` (see
    ``_consolidate_cell`` for the dict layout). Geometry comes from
    BOUNDARY, PATH, SREF and AREF elements; TEXT/NODE/BOX are skipped.
    Raises ``ValueError`` with a user-facing message when the file
    exceeds the app's memory budgets.
    """
    mv = memoryview(buf)
    if mv.format != "B" or mv.ndim != 1:
        mv = mv.cast("B")
    a = np.frombuffer(mv, dtype=np.uint8)
    n = a.size
    unpack_rec = _REC.unpack_from
    unpack_h = struct.Struct(">h").unpack_from
    unpack_u16 = struct.Struct(">H").unpack_from
    unpack_i = struct.Struct(">i").unpack_from
    unpack_2i = struct.Struct(">2i").unpack_from
    unpack_2h = struct.Struct(">2h").unpack_from
    unpack_6i = struct.Struct(">6i").unpack_from

    db_user = 1e-3            # spec defaults; overwritten by UNITS
    db_meters = 1e-9
    cells: dict = {}

    cur_name = None
    cur_polys = None          # {(l, d): ([cx chunks], [cy chunks], [sizes])}
    cur_paths = None          # [(l, d, width, ptype, bext, eext, pts)]
    cur_refs = None           # {(sname, rot, mag, refl): [(x, y) | array]}

    el = 0                    # current element's start tag (0 = none)
    el_start = 0
    el_layer = el_dt = 0
    el_sname = None
    el_rot = 0.0
    el_mag = 1.0
    el_refl = False
    el_colrow = None
    el_xy = None
    el_width = 0
    el_ptype = 0
    el_bext = 0
    el_eext = 0

    total_verts = 0           # polygon vertices stored so far
    total_rows = 0            # reference placements stored so far

    p = 0
    while p + 4 <= n:
        rlen, tag = unpack_rec(mv, p)
        if rlen < 4 or p + rlen > n:
            break                        # trailing padding / truncation
        d0 = p + 4
        nxt = p + rlen

        if tag == _T_XY:
            el_xy = (d0, rlen - 4)
        elif tag == _T_ENDEL:
            p_next = nxt
            if el_xy is not None and cur_polys is not None:
                blk = nxt - el_start
                xd0, xdl = el_xy
                if el == _T_SREF and el_sname is not None:
                    key = (el_sname, el_rot, el_mag, el_refl)
                    lst = cur_refs.get(key)
                    if lst is None:
                        lst = []
                        cur_refs[key] = lst
                    lst.append(unpack_2i(mv, xd0))
                    total_rows += 1
                    # Bulk-decode the run of identical SREFs that CAD
                    # tools emit for arrayed placements.
                    p_next, chunks = _element_run(
                        a, mv, el_start, blk, xd0 - el_start, xdl)
                    for ch in chunks:
                        total_rows += ch.size >> 1
                        lst.append(ch.reshape(-1, 2))
                elif el == _T_BOUNDARY:
                    npts = xdl >> 3
                    pts = np.frombuffer(mv, dtype=">i4", count=2 * npts,
                                        offset=xd0).astype(np.int64)
                    bcx = pts[0::2]
                    bcy = pts[1::2]
                    drop = bool(npts > 1 and bcx[0] == bcx[-1]
                                and bcy[0] == bcy[-1])
                    if drop:
                        bcx = bcx[:-1]
                        bcy = bcy[:-1]
                    keep_pts = npts - 1 if drop else npts
                    slot = cur_polys.get((el_layer, el_dt))
                    if slot is None:
                        slot = ([], [], [])
                        cur_polys[(el_layer, el_dt)] = slot
                    slot[0].append(bcx)
                    slot[1].append(bcy)
                    slot[2].append(keep_pts)
                    total_verts += keep_pts
                    # Bulk-decode runs of same-shape boundaries (flat
                    # polygon dumps).
                    p_next, chunks = _element_run(
                        a, mv, el_start, blk, xd0 - el_start, xdl)
                    for ch in chunks:
                        rows = ch.size // (2 * npts)
                        rr = ch.reshape(rows, npts, 2)
                        if drop:
                            rr = rr[:, :-1, :]
                        slot[0].append(rr[:, :, 0].reshape(-1))
                        slot[1].append(rr[:, :, 1].reshape(-1))
                        slot[2].append(
                            np.full(rows, keep_pts, dtype=np.int64))
                        total_verts += rows * keep_pts
                elif (el == _T_AREF and el_sname is not None
                        and el_colrow is not None):
                    cols, arows = el_colrow
                    if cols > 0 and arows > 0 and xdl >= 24:
                        x0, y0, x1, y1, x2, y2 = unpack_6i(mv, xd0)
                        ii = (np.arange(cols, dtype=np.float64)[:, None]
                              / cols)
                        jj = (np.arange(arows, dtype=np.float64)[None, :]
                              / arows)
                        offx = x0 + ii * (x1 - x0) + jj * (x2 - x0)
                        offy = y0 + ii * (y1 - y0) + jj * (y2 - y0)
                        key = (el_sname, el_rot, el_mag, el_refl)
                        lst = cur_refs.get(key)
                        if lst is None:
                            lst = []
                            cur_refs[key] = lst
                        lst.append(np.stack(
                            [offx.ravel(), offy.ravel()], axis=1))
                        total_rows += cols * arows
                elif el == _T_PATH:
                    npts = xdl >> 3
                    pts = np.frombuffer(
                        mv, dtype=">i4", count=2 * npts,
                        offset=xd0).astype(np.float64).reshape(-1, 2)
                    cur_paths.append((el_layer, el_dt, el_width, el_ptype,
                                      el_bext, el_eext, pts))
                    total_verts += npts * 4   # rough polygon estimate
                if total_verts > _MAX_SRC_VERTICES:
                    raise ValueError(
                        f"GDS holds more than {_MAX_SRC_VERTICES:,} "
                        "polygon vertices — too much distinct geometry "
                        "for this app's memory budget. Expose a smaller "
                        "layer, then re-upload."
                    )
                if total_rows > _MAX_OFFSET_ROWS:
                    raise ValueError(
                        f"GDS places cell references more than "
                        f"{_MAX_OFFSET_ROWS:,} times — too much for "
                        "this app's memory budget. Expose a smaller "
                        "layer, then re-upload."
                    )
            el = 0
            el_sname = None
            el_xy = None
            el_colrow = None
            el_rot = 0.0
            el_mag = 1.0
            el_refl = False
            p = p_next
            continue
        elif tag in _EL_START:
            el = tag
            el_start = p
            el_layer = el_dt = 0
            el_xy = None
            el_width = 0
            el_ptype = 0
            el_bext = 0
            el_eext = 0
        elif tag == _T_LAYER:
            el_layer = unpack_h(mv, d0)[0]
        elif tag == _T_DATATYPE:
            el_dt = unpack_h(mv, d0)[0]
        elif tag == _T_SNAME:
            el_sname = bytes(mv[d0:nxt]).rstrip(b"\0").decode(
                "ascii", "replace")
        elif tag == _T_STRANS:
            el_refl = bool(unpack_u16(mv, d0)[0] & 0x8000)
        elif tag == _T_MAG:
            el_mag = _gds_real8(mv[d0:d0 + 8])
        elif tag == _T_ANGLE:
            el_rot = math.radians(_gds_real8(mv[d0:d0 + 8]))
        elif tag == _T_COLROW:
            el_colrow = unpack_2h(mv, d0)
        elif tag == _T_WIDTH:
            el_width = unpack_i(mv, d0)[0]
        elif tag == _T_PATHTYPE:
            el_ptype = unpack_h(mv, d0)[0]
        elif tag == _T_BGNEXTN:
            el_bext = unpack_i(mv, d0)[0]
        elif tag == _T_ENDEXTN:
            el_eext = unpack_i(mv, d0)[0]
        elif tag == _T_STRNAME:
            cur_name = bytes(mv[d0:nxt]).rstrip(b"\0").decode(
                "ascii", "replace")
            cur_polys = {}
            cur_paths = []
            cur_refs = {}
        elif tag == _T_ENDSTR:
            if cur_name is not None:
                cells[cur_name] = _consolidate_cell(
                    cur_polys, cur_paths, cur_refs, db_user)
            cur_name = None
            cur_polys = None
            cur_paths = None
            cur_refs = None
        elif tag == _T_UNITS:
            db_user = _gds_real8(mv[d0:d0 + 8])
            db_meters = _gds_real8(mv[d0 + 8:d0 + 16])
        elif tag == _T_ENDLIB:
            break
        p = nxt

    if db_user <= 0:
        db_user = 1e-3
    if db_meters <= 0:
        db_meters = db_user * 1e-6
    return (db_meters / db_user), cells


def _flatten_instanced(cell_name, cells, cache, budget):
    """Return ``{(layer, datatype): [group, ...]}`` for the named raw
    cell, where each group is ``(base_cx, base_cy, base_starts,
    offsets)``: the base geometry in this cell's frame plus the (K, 2)
    lattice of translations it's placed at.

    References are NOT expanded — the child's instanced geometry is
    reused and its offset lattice is combined (outer sum) with this
    reference's placement lattice. Nesting multiplies the offset tables,
    never the polygon arrays. Memoized per cell. The parser already
    grouped same-transform references, so a unit cell tiled 150 k× via
    individual SREFs arrives here as one (K, 2) placement array.
    """
    hit = cache.get(cell_name)
    if hit is not None:
        return hit
    cell = cells.get(cell_name)
    acc: dict = {}            # key -> list of groups
    if cell is None:          # dangling reference
        cache[cell_name] = acc
        return acc

    # Cell's own polygons: a single K=1 group per layer.
    for key, (cx, cy, starts) in cell["polys"].items():
        acc.setdefault(key, []).append(
            (cx, cy, starts, np.zeros((1, 2), dtype=np.float64)))

    for child, rot, mag, refl, placements in cell["refs"]:
        kp = int(placements.shape[0])
        child_groups = _flatten_instanced(child, cells, cache, budget)
        for ckey, groups in child_groups.items():
            for bcx, bcy, bst, coff in groups:
                # Transform the base coords (mag/reflection/rotation) and
                # the child's own offset vectors (linear part only).
                tbx, tby = _apply_ref_transform(bcx, bcy, rot, mag, refl)
                tcx, tcy = _apply_ref_transform(coff[:, 0], coff[:, 1],
                                                rot, mag, refl)
                kc = int(tcx.shape[0])
                # Guard before building the (kc*kp, 2) combined lattice.
                budget["rows"] += kc * kp
                if budget["rows"] > _MAX_OFFSET_ROWS:
                    raise ValueError(
                        f"GDS expands to more than {_MAX_OFFSET_ROWS:,} "
                        "cell placements — too much for this app's "
                        "memory budget. Expose a smaller layer, then "
                        "re-upload."
                    )
                if kp == 1:
                    # Single placement (typical top cell): one allocation
                    # instead of stack + broadcast (3× the array size —
                    # that was the RAM peak for multi-million-instance
                    # masks).
                    comb = np.empty((kc, 2), dtype=np.float64)
                    np.add(tcx, placements[0, 0], out=comb[:, 0])
                    np.add(tcy, placements[0, 1], out=comb[:, 1])
                else:
                    comb = (np.stack([tcx, tcy], axis=1)[:, None, :]
                            + placements[None, :, :]).reshape(-1, 2)
                acc.setdefault(ckey, []).append((tbx, tby, bst, comb))

    cache[cell_name] = acc
    return acc


def _expand_groups_to_flat(groups):
    """Tile every group's base by its offsets into one flat (cx, cy,
    starts). Used when a layer is small enough that the plain ``_PolyLayer``
    path is simpler than carrying the instanced structure."""
    cxs, cys, szs = [], [], []
    for bcx, bcy, bst, off in groups:
        sizes = np.diff(bst)
        k = off.shape[0]
        if k == 1:
            cxs.append(bcx + off[0, 0]); cys.append(bcy + off[0, 1])
            szs.append(sizes)
        else:
            cxs.append((bcx[None, :] + off[:, 0][:, None]).ravel())
            cys.append((bcy[None, :] + off[:, 1][:, None]).ravel())
            szs.append(np.tile(sizes, k))
    cx = np.concatenate(cxs) if cxs else np.zeros(0)
    cy = np.concatenate(cys) if cys else np.zeros(0)
    sizes = np.concatenate(szs) if szs else np.zeros(0, dtype=np.int64)
    starts = np.zeros(sizes.size + 1, dtype=np.int64)
    np.cumsum(sizes, out=starts[1:])
    return cx, cy, starts


@st.cache_resource(show_spinner="Parsing GDS…", max_entries=1)
def _load_gds(digest: str, _upload):
    """Parse GDS → ``(unit_meters, {cell_name: {(l,d): spec}})`` where each
    ``spec`` is a tuple:

      * ``("flat", cx, cy, starts)`` — fully expanded geometry, for layers
        small enough (≤ ``_POLY_LIMIT``) or with no usable repetition;
      * ``("inst", [(bcx, bcy, bstarts, offsets), ...])`` — the instanced
        (array/AREF) structure kept intact, for dense layers built from a
        small base pattern repeated many times.

    ``_upload`` is the ``st.file_uploader`` value (or raw bytes in tests);
    its buffer is parsed in place via ``getbuffer()`` — no ``getvalue()``
    copy, no temp file. The cache key is ``digest`` (content hash computed
    by the caller); the underscore keeps the buffer itself out of the key.
    ``st.cache_resource`` (not ``cache_data``) returns the parsed arrays
    by reference: with ``cache_data`` a multi-hundred-MB result would
    exist twice (pickled blob + a fresh unpickled copy per rerun). All
    consumers treat the arrays as immutable. The flattener
    (``_flatten_instanced``) combines offset lattices instead of expanding
    polygons, so a 28 M-polygon arrayed mask stays ~100 MB. Raises
    ``ValueError`` past the memory budgets so the caller can show a
    friendly error instead of getting OOM-killed.
    """
    buf = (_upload.getbuffer() if hasattr(_upload, "getbuffer")
           else memoryview(_upload))
    try:
        unit, cells = _parse_gds(buf)
    finally:
        buf.release()
        gc.collect()

    referenced = set()
    for cell in cells.values():
        for ref in cell["refs"]:
            referenced.add(ref[0])

    out: dict = {}
    budget = {"rows": 0}
    for name in cells:
        if name in referenced:
            continue
        geo = _flatten_instanced(name, cells, {}, budget)
        by_layer: dict = {}
        for key, groups in geo.items():
            expanded = sum(int(bst.size - 1) * int(off.shape[0])
                           for _, _, bst, off in groups)
            base_polys = sum(int(bst.size - 1)
                             for _, _, bst, _ in groups)
            repetitive = any(off.shape[0] > 1
                             for _, _, _, off in groups)
            # Keep the instanced form only when it actually pays off:
            # the layer is dense AND its unique base is small enough to
            # draw. Otherwise expand to a flat _PolyLayer (small layers
            # stay simple; huge non-repetitive dumps use the flat
            # bbox/raster fallback).
            if (expanded > _POLY_LIMIT and repetitive
                    and base_polys <= _POLY_LIMIT):
                by_layer[key] = ("inst", list(groups))
            else:
                exp_verts = sum(int(bcx.size) * int(off.shape[0])
                                for bcx, _, _, off in groups)
                if exp_verts > _MAX_VERTICES:
                    raise ValueError(
                        f"Layer L{key[0]}/D{key[1]} expands to "
                        f"{exp_verts:,} vertices with no small repeated "
                        "unit pattern — too much for this app's memory "
                        "budget. Expose a smaller layer, then re-upload."
                    )
                cx, cy, starts = _expand_groups_to_flat(groups)
                by_layer[key] = ("flat", cx, cy, starts)
        out[name] = by_layer
    return unit, out


def _spec_to_layer(spec):
    """Build a _PolyLayer or _InstancedLayer from a cached ``spec`` tuple."""
    if spec[0] == "inst":
        return _InstancedLayer(spec[1])
    _, cx, cy, starts = spec
    return _PolyLayer(cx, cy, starts)


def _load_gds_layers(upload):
    """Cached parse + wrap into layer objects.

    ``upload`` is the ``st.file_uploader`` value (or raw bytes in tests).
    Its content is only ever touched through ``getbuffer()`` memoryviews —
    with a 180 MB mask, a single ``getvalue()`` copy would burn a fifth
    of a 1 GB host's RAM. The layer objects (``_PolyLayer`` /
    ``_InstancedLayer``) are rebuilt per rerun from the cached primitives;
    wrapping is cheap (no coordinate copies).
    """
    if hasattr(upload, "getbuffer"):
        mv = upload.getbuffer()
    else:
        mv = memoryview(upload)
    try:
        digest = f"{hashlib.md5(mv).hexdigest()}:{mv.nbytes}"
    finally:
        mv.release()
    unit, raw = _load_gds(digest, upload)
    out = {
        cell: {key: _spec_to_layer(spec) for key, spec in by_layer.items()}
        for cell, by_layer in raw.items()
    }
    return unit, out


def _layer_trace(name: str, polys, color: str, scale: float = 1.0) -> go.Scatter:
    """Filled outline trace of every polygon, coordinates × ``scale``
    (e.g. user units → µm for the viewer plots)."""
    xs_all, ys_all = [], []
    for xs, ys in polys:
        if scale != 1.0:
            xs = xs * scale
            ys = ys * scale
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


def _layer_bbox_mm(polys, scale_to_mm: float):
    """Return (min_x, min_y, max_x, max_y) of polygons in mm, or None."""
    if not polys:
        return None
    if isinstance(polys, (_PolyLayer, _InstancedLayer)):
        bb = polys.bbox()
        if bb is None:
            return None
        return (bb[0] * scale_to_mm, bb[1] * scale_to_mm,
                bb[2] * scale_to_mm, bb[3] * scale_to_mm)
    min_x = min(min(xs) for xs, _ in polys) * scale_to_mm
    max_x = max(max(xs) for xs, _ in polys) * scale_to_mm
    min_y = min(min(ys) for _, ys in polys) * scale_to_mm
    max_y = max(max(ys) for _, ys in polys) * scale_to_mm
    return min_x, min_y, max_x, max_y


def _placed_bbox_mm(polys, scale_to_mm: float, ox: float, oy: float):
    """Bounding box (min_x, min_y, max_x, max_y) of ``polys`` after the
    affine placement (x*scale + ox, y*scale + oy), or None."""
    bb = _layer_bbox_mm(polys, scale_to_mm)
    if bb is None:
        return None
    return (bb[0] + ox, bb[1] + oy, bb[2] + ox, bb[3] + oy)


def _bbox_rect_trace(bbox_mm, color: str, name: str) -> go.Scatter:
    """A dashed rectangle outline for a placed mask bounding box, used in
    place of drawing every polygon when a layer is too dense to render."""
    x0, y0, x1, y1 = bbox_mm
    return go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode="lines",
        line=dict(color=color, width=1.5, dash="dash"),
        fill="toself",
        fillcolor=_hex_to_rgba(color, 0.12),
        name=name,
        hoverinfo="skip",
    )


def _bin_points(out, px, py, weights, gx0, gy0, chip_size_mm, nx, ny):
    """Accumulate ``weights`` into the flat (nx*ny) cell grid by the cell
    each (px, py) falls in. Cell order i outer, j inner."""
    i = np.floor((px - gx0) / chip_size_mm).astype(np.int64)
    j = np.floor((py - gy0) / chip_size_mm).astype(np.int64)
    valid = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    np.add.at(out, i[valid] * ny + j[valid], weights[valid])


def _fast_cell_areas_binned(layer: "_PolyLayer", scale_to_mm: float,
                            ox: float, oy: float, gx0: float, gy0: float,
                            chip_size_mm: float, nx: int, ny: int) -> list:
    """Vectorized per-cell mm² area for a mask placed once over an
    edge-to-edge Nx×Ny grid (cell order i outer, j inner — matching the
    workflow grids). Each polygon is assigned whole to the cell holding
    its first vertex: exact for the grand total and for the common case
    of features fully inside one grid; only boundary-straddling polygons
    are mis-binned. Polygons whose first vertex falls outside the grid
    are dropped (they aren't exposed)."""
    areas_mm2 = layer.poly_areas() * (scale_to_mm * scale_to_mm)
    fx, fy = layer.first_vertices()
    out = np.zeros(nx * ny, dtype=np.float64)
    _bin_points(out, fx * scale_to_mm + ox, fy * scale_to_mm + oy,
                areas_mm2, gx0, gy0, chip_size_mm, nx, ny)
    return out.tolist()


def _instanced_cell_areas_binned(layer: "_InstancedLayer", scale_to_mm: float,
                                 ox: float, oy: float, gx0: float, gy0: float,
                                 chip_size_mm: float, nx: int, ny: int) -> list:
    """Per-cell mm² area for an instanced layer, binned per *instance*
    (O(#instances), never expands polygons). Each instance contributes its
    whole base area to the cell containing the instance's center — exact
    total, exact per-cell when each tile fits within a grid."""
    out = np.zeros(nx * ny, dtype=np.float64)
    for bcx, bcy, bst, off in layer.groups:
        base = _PolyLayer(bcx, bcy, bst)
        base_area = float(base.poly_areas().sum()) * (scale_to_mm ** 2)
        bb = base.bbox()
        if bb is None:
            continue
        cxc = 0.5 * (bb[0] + bb[2])
        cyc = 0.5 * (bb[1] + bb[3])
        px = (cxc + off[:, 0]) * scale_to_mm + ox
        py = (cyc + off[:, 1]) * scale_to_mm + oy
        w = np.full(off.shape[0], base_area, dtype=np.float64)
        _bin_points(out, px, py, w, gx0, gy0, chip_size_mm, nx, ny)
    return out.tolist()


def _cell_areas_binned(layer, scale_to_mm, ox, oy, gx0, gy0,
                       chip_size_mm, nx, ny) -> list:
    """Dispatch per-cell area binning to the flat or instanced helper."""
    if isinstance(layer, _InstancedLayer):
        return _instanced_cell_areas_binned(
            layer, scale_to_mm, ox, oy, gx0, gy0, chip_size_mm, nx, ny)
    return _fast_cell_areas_binned(
        layer, scale_to_mm, ox, oy, gx0, gy0, chip_size_mm, nx, ny)


def _hex_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rasterize_coverage(layer: "_InstancedLayer", scale_to_mm: float,
                        ox: float, oy: float, color_hex: str, px: int = 360):
    """Bin every instance into a low-res pixel grid (coverage raster) so the
    full array footprint can be shown as ONE small image instead of tens of
    thousands of vector points. Empty pixels are WHITE so the view reads the
    same regardless of light/dark theme. Returns ``(rgba, x0, dx, y0, dy)``
    for a ``go.Image`` trace (row 0 = bottom; pair with a non-reversed
    y-axis), or None."""
    grid = _coverage_grid(layer, scale_to_mm, ox, oy, px)
    if grid is None:
        return None
    cnt, x0c, psz, y0c, _ = grid
    r, g, b = _hex_rgb(color_hex)
    rgba = np.empty((cnt.shape[0], cnt.shape[1], 4), dtype=np.uint8)
    rgba[:] = (255, 255, 255, 255)          # white background
    rgba[cnt > 0] = (r, g, b, 255)          # pattern coverage
    return rgba, x0c, psz, y0c, psz


def _coverage_grid(layer: "_InstancedLayer", scale_to_mm: float,
                   ox: float, oy: float, px: int):
    """Bin every instance center into a square-pixel grid over the placed
    bbox. Returns ``(cnt, x0_center, psz, y0_center, psz)`` (row 0 =
    bottom), or None."""
    bb = _placed_bbox_mm(layer, scale_to_mm, ox, oy)
    if bb is None:
        return None
    x0d, y0d, x1d, y1d = bb
    w = x1d - x0d
    h = y1d - y0d
    if w <= 0 or h <= 0:
        return None
    ncols = int(px)
    psz = w / ncols
    nrows = max(1, int(round(h / psz)))
    cnt = np.zeros((nrows, ncols), dtype=np.int64)
    for bcx, bcy, bst, off in layer.groups:
        base = _PolyLayer(bcx, bcy, bst)
        bbb = base.bbox()
        if bbb is None:
            continue
        cxc = 0.5 * (bbb[0] + bbb[2])
        cyc = 0.5 * (bbb[1] + bbb[3])
        pxs = (cxc + off[:, 0]) * scale_to_mm + ox
        pys = (cyc + off[:, 1]) * scale_to_mm + oy
        col = np.floor((pxs - x0d) / psz).astype(np.int64)
        row = np.floor((pys - y0d) / psz).astype(np.int64)
        m = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
        np.add.at(cnt, (row[m], col[m]), 1)
    return cnt, x0d + psz / 2, psz, y0d + psz / 2, psz


def _coverage_heatmap_trace(layer, scale_to_mm: float, ox: float, oy: float,
                            color: str, px: int = 260):
    """A low-res coverage trace as a ``go.Heatmap`` (pattern = ``color``,
    empty = transparent) for overlaying on the workflow / time-calculator
    plots without flipping their y-axis the way ``go.Image`` would. Only for
    instanced layers; returns None otherwise."""
    if not isinstance(layer, _InstancedLayer):
        return None
    grid = _coverage_grid(layer, scale_to_mm, ox, oy, px)
    if grid is None:
        return None
    cnt, x0c, dx, y0c, dy = grid
    z = np.where(cnt > 0, 1.0, np.nan)
    xs = x0c + dx * np.arange(cnt.shape[1])
    ys = y0c + dy * np.arange(cnt.shape[0])
    return go.Heatmap(
        z=z, x=xs, y=ys, colorscale=[[0.0, color], [1.0, color]],
        zmin=0.0, zmax=1.0, showscale=False, hoverinfo="skip",
        name="Mask coverage",
    )


def _unit_pattern_traces(layer: "_InstancedLayer", color: str,
                         scale: float = 1.0) -> list:
    """One filled trace per group showing the base (unit) pattern with
    coordinates × ``scale`` (user units → µm for the viewer) — a
    dedicated zoomed-in view of the repeated shape, which is otherwise
    sub-pixel at the full-array scale. When a layer has several distinct
    unit cells they're laid out left-to-right so they don't overlap."""
    palette = _PALETTE
    traces = []
    multi = len(layer.groups) > 1
    x_cursor = 0.0
    for gi, (bcx, bcy, bst, _off) in enumerate(layer.groups):
        if bst.size <= 1:
            continue
        base = _PolyLayer(bcx, bcy, bst)
        bb = base.bbox()
        if bb is None:
            continue
        w = (bb[2] - bb[0]) * scale
        # Shift so this unit's left edge sits at the running cursor
        # (cursor runs in scaled coordinates).
        dx = x_cursor - bb[0] * scale
        xs, ys = _nan_xy_from_flat(bcx, bcy, bst, scale, dx, 0.0)
        traces.append(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            line=dict(color=palette[gi % len(palette)] if multi else color,
                      width=1.2),
            fillcolor=_hex_to_rgba(
                palette[gi % len(palette)] if multi else color, 0.5),
            name=f"unit {gi + 1} ({int(bst.size - 1)} polys)"
                 if multi else "unit pattern",
            hoverinfo="skip", showlegend=multi,
        ))
        x_cursor += w * 1.4 if w > 0 else 1.0
    return traces


def _decimated_centers(layer: "_InstancedLayer", scale_to_mm: float,
                       ox: float, oy: float, cap: int = 5000):
    """Placed (x, y) of a decimated, evenly-spread sample of instance
    centers — a light, selectable scatter layer for box-select."""
    total = max(1, layer.instance_count())
    xs, ys = [], []
    for bcx, bcy, bst, off in layer.groups:
        base = _PolyLayer(bcx, bcy, bst)
        bb = base.bbox()
        if bb is None:
            continue
        cxc = 0.5 * (bb[0] + bb[2])
        cyc = 0.5 * (bb[1] + bb[3])
        k = off.shape[0]
        n = max(1, min(k, int(round(cap * k / total))))
        idx = np.unique(np.linspace(0, k - 1, n).round().astype(np.int64))
        xs.append((cxc + off[idx, 0]) * scale_to_mm + ox)
        ys.append((cyc + off[idx, 1]) * scale_to_mm + oy)
    if not xs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys)


def _nan_xy_from_flat(cx, cy, starts, scale_to_mm, ox, oy):
    """Flat (cx, cy, starts) → NaN-separated, polygon-closed x/y lists in
    placed mm coordinates, ready for a single filled Plotly line trace."""
    px = cx * scale_to_mm + ox
    py = cy * scale_to_mm + oy
    xs, ys = [], []
    s = starts
    for a, b in zip(s[:-1].tolist(), s[1:].tolist()):
        xs.extend(px[a:b].tolist()); xs.append(px[a]); xs.append(None)
        ys.extend(py[a:b].tolist()); ys.append(py[a]); ys.append(None)
    return xs, ys


# Max polygons to draw at full detail inside a selected region (KLayout-
# style inspect). Kept modest so the SVG detail plot stays responsive
# (~15 k polys ≈ 90 k points); bigger selections are refused with a hint
# to pick a smaller region.
_MAX_REGION_POLYS = 15_000


def _instances_in_window(layer: "_InstancedLayer", scale_to_mm: float,
                         ox: float, oy: float,
                         x0: float, x1: float, y0: float, y1: float):
    """Expand only the instances whose placed unit center falls in the
    window [x0,x1]×[y0,y1] to full per-polygon detail. Returns placed-mm
    ``(cx, cy, starts)``, ``None`` if the window is empty, or
    ``("over", n_polys)`` if it would exceed ``_MAX_REGION_POLYS``."""
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    sel_groups = []
    total = 0
    for bcx, bcy, bst, off in layer.groups:
        base = _PolyLayer(bcx, bcy, bst)
        bb = base.bbox()
        if bb is None:
            continue
        cxc = 0.5 * (bb[0] + bb[2])
        cyc = 0.5 * (bb[1] + bb[3])
        px = (cxc + off[:, 0]) * scale_to_mm + ox
        py = (cyc + off[:, 1]) * scale_to_mm + oy
        m = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
        sel = off[m]
        if sel.shape[0]:
            total += sel.shape[0] * int(bst.size - 1)
            if total > _MAX_REGION_POLYS:
                return ("over", total)
            sel_groups.append((bcx, bcy, bst, sel))
    if not sel_groups:
        return None
    cx, cy, starts = _expand_groups_to_flat(sel_groups)
    return cx * scale_to_mm + ox, cy * scale_to_mm + oy, starts


def _mask_overlay_traces(layer, scale_to_mm: float, ox: float, oy: float,
                         color: str, name: str) -> list:
    """Traces for a dense mask placed at (ox, oy): instanced layers draw
    base + tiles; flat dense layers draw a single bounding box."""
    if isinstance(layer, _InstancedLayer):
        # Low-res coverage raster (same view as the GDS viewer footprint).
        ht = _coverage_heatmap_trace(layer, scale_to_mm, ox, oy, color)
        return [ht] if ht is not None else []
    bb = _placed_bbox_mm(layer, scale_to_mm, ox, oy)
    if bb is None:
        return []
    return [_bbox_rect_trace(bb, color, f"{name} (bbox, {len(layer):,} polys)")]


def _dense_layer_note(layer) -> str:
    """One-line caption describing how a dense layer is being shown."""
    if isinstance(layer, _InstancedLayer):
        return (
            f":blue[Repetition detected: a {layer.base_poly_count():,}-polygon "
            f"unit pattern tiled {layer.instance_count():,}× "
            f"= {len(layer):,} polygons. Drawing the unit pattern + array "
            "footprint; the time estimate uses unit area × tile count.]"
        )
    return (
        f":orange[Layer has {len(layer):,} polygons (> {_POLY_LIMIT:,}); "
        "showing the mask bounding box. The time estimate uses each "
        "polygon's full area, binned to the grid cell holding its first "
        "vertex.]"
    )


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
    st.subheader("Time Calculator")

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
                "Initial dose (μs / dot)", min_value=0.0, step=0.01,
                format="%.3f", key=f"{prefix}_dose_init_us",
            )
        with c_step:
            st.number_input(
                "Incremental dose (μs / grid)", min_value=0.0, step=0.01,
                format="%.3f", key=f"{prefix}_dose_step_us",
            )
    else:
        c_dose, c_stage, c_btn = st.columns([2, 2, 1])
        with c_dose:
            st.number_input(
                "Dose time (μs / dot)", min_value=0.0, step=0.01,
                format="%.3f", key=f"{prefix}_dose_us",
            )
    with c_stage:
        st.number_input(
            "Stage movement time (s / grid)", min_value=0.0, step=0.1,
            format="%.2f", key=f"{prefix}_stage_s",
        )
    with c_btn:
        st.markdown("&nbsp;")  # vertical spacing to line up with inputs
        clicked = st.button(
            "Calculate time", key=f"{prefix}_calc",
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
            with st.spinner("Computing polygon area inside each grid…"):
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

    result = st.session_state.get(f"{prefix}_time_result")
    if not result:
        return

    col_vals, col_plot = st.columns(2)

    with col_vals:
        st.markdown("**Breakdown**")
        st.write(
            f"Resolution: **{result['res_mm'] * 1e6:.3f} nm** "
            f"({result['res_mm'] * 1000:g} μm) — "
            f"chip size / dotmap = "
            f"{result['chip_size_mm'] * 1000:g} μm / {result['dotmap']}"
        )
        st.write(
            f"Filled resolution boxes: "
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
            f"Active grids: **{result['active'] + _n_marks} / "
            f"{result['n_cells'] + _n_marks}** "
            "(empty grids are skipped)"
        )
        if result.get("dose_ramp"):
            st.write(
                f"Dose ramp: **{result['dose_init_us']:.3f} μs** "
                f"+ {result['dose_step_us']:.3f} μs/grid "
                f"→ active range "
                f"**{result['dose_min']:.3f} – "
                f"{result['dose_max']:.3f} μs**"
            )
            st.write(
                "Exposure: Σ(filled × per-grid dose) "
                f"= **{result['exposure_us'] / 1e6:,.3f} s**"
            )
        else:
            st.write(
                f"Exposure: {result['filled_total']:,.0f} × "
                f"{result['dose_us']:.3f} μs "
                f"= **{result['exposure_us'] / 1e6:,.3f} s**"
            )
        st.write(
            f"Stage movement: {result['active'] + _n_marks} × "
            f"{result['stage_s']:.2f} s "
            f"= **{result['stage_us'] / 1e6:,.3f} s**"
        )
        st.markdown(
            f"### Estimated Time: `{_format_hms(result['total_us'] / 1e6)}`",
            help=(f"{result['total_us'] / 1e6:,.3f} seconds "
                  "(hh:mm:ss.sss = hours:minutes:seconds)")
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

        # Mask: for instanced layers overlay the low-res coverage raster
        # (same view as the viewer / workflow). Otherwise draw the mask
        # clipped to the grids (the portion that lands inside a cell, so
        # what's plotted matches what was counted in the area total).
        if coverage_layer is not None and isinstance(
                coverage_layer, _InstancedLayer):
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
                    name="Mask",
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
with st.container(border=True):
    st.header("GDS Mask")



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
                    unit_m, cells_data = _load_gds_layers(gds_upload)
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
            # Viewer plots are labelled in µm, so convert from GDS user
            # units explicitly — files where 1 user unit ≠ 1 µm (the PSO
            # masks use 1 m) otherwise display meter values under a µm
            # axis label.
            _gds_unit_to_um = _gds_unit_to_mm * 1000.0

            if isinstance(polys, _InstancedLayer):
                # Repetitive layer: show the unit pattern zoomed (so the
                # repeated shape is actually visible) beside the full array
                # footprint with a decimated sample of real patterns.
                st.caption(_dense_layer_note(polys))
                col_unit, col_full = st.columns([4, 6])
                with col_unit:
                    st.markdown("**Unit pattern (zoomed)**")
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
                    st.markdown("**Array footprint** (low-res overview) — "
                                "drag a box to inspect that region below at "
                                "full detail")
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
                    st.caption("Double-click the plot to clear the selection.")

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
                        st.info("No patterns in the selected region.")
                    elif isinstance(_res[0], str):   # ("over", n_polys)
                        st.warning(
                            f"Selected region holds {_res[1]:,} polygons "
                            f"(> {_MAX_REGION_POLYS:,}). Select a smaller "
                            "region to inspect at full detail."
                        )
                    else:
                        _cx, _cy, _starts = _res
                        _n_sel = int(_starts.size - 1)
                        st.markdown(
                            f"**Selected region (full detail — "
                            f"{_n_sel:,} polygons)**")
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
                render = True
                if n_polys > _POLY_LIMIT:
                    # Dense, non-repetitive layer: drawing every polygon
                    # would stall the browser, so default off behind a box.
                    st.warning(
                        f"Selected layer contains {n_polys:,} polygons "
                        f"(limit {_POLY_LIMIT:,}) with no detected "
                        "repetition. Rendering every polygon may stall the "
                        "browser."
                    )
                    render = st.checkbox("Render anyway", key="ebc_gds_force")
                    if render:
                        fig.add_trace(_layer_trace(
                            selected_label, polys, _PALETTE[0],
                            scale=_gds_unit_to_um))
                else:
                    fig.add_trace(_layer_trace(
                        selected_label, polys, _PALETTE[0],
                        scale=_gds_unit_to_um))
                if render:
                    fig.update_layout(
                        xaxis=dict(title="x (µm)"),
                        yaxis=dict(title="y (µm)",
                                scaleanchor="x", scaleratio=1),
                        margin=dict(l=40, r=20, t=20, b=40),
                        height=550,
                        showlegend=True,
                        legend=dict(itemsizing="constant"),
                    )
                    st.plotly_chart(fig, width="stretch")


# ─── Section 4: Mode selector ────────────────────────────────────────────────
with st.container(border=True):
    st.header("Workflow")

    mode = segmented_radio(
        "Mode",
        ["Choose mode:","Dose Time Testing", "First Exposure", "Second Alignment"],
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
                st.markdown("**Cel Origin (mm) in job1**")
                p11, p12 = st.columns(2)
                cel_x = p11.number_input("x", format="%.3f",
                                        step=0.1, min_value=0.0, key="ebc_dt_cel_x")
                cel_y = p12.number_input("y", format="%.3f",
                                        step=0.1, min_value=0.0, key="ebc_dt_cel_y")
        with p2:
            with st.container(border=True):
                st.markdown("**Increment (mm) in job3**")
                p11, p12 = st.columns(2)
                dx = p11.number_input("dx", format="%.3f",
                                    min_value=chip_size,
                                    step=0.0005, key="ebc_dt_dx")
                dy = p12.number_input("dy", format="%.3f",
                                    min_value=chip_size,
                                    step=0.0005, key="ebc_dt_dy")
        with p3:
            with st.container(border=True):
                st.markdown("**Grid Count in job3**")
                p11, p12 = st.columns(2)
                Nx = p11.number_input("Nx", min_value=1,
                                        step=1, key="ebc_dt_nx")
                Ny = p12.number_input("Ny", min_value=1,
                                        step=1, key="ebc_dt_ny")
        with p4:
            with st.container(border=True):
                st.markdown("**Initial Shift (mm) in job3**")
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
        tr = (st.session_state["ebc_tr_x"], st.session_state["ebc_tr_y"])
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
                    name="Mask",
                    hoverinfo="skip",
                ))
            elif _dt_huge:
                _bb = _placed_bbox_mm(_gds_selected_polys, _gds_unit_to_mm,
                                      cel_x, cel_y)
                if _bb is not None:
                    fig_single.add_trace(_bbox_rect_trace(
                        _bb, "#2ca02c",
                        f"Mask bbox ({len(_gds_selected_polys):,} polys)"))

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
            st.plotly_chart(fig, width="stretch")

        if _dt_huge:
            if isinstance(_gds_selected_polys, _InstancedLayer):
                st.caption(
                    f":blue[Repetition detected: a "
                    f"{_gds_selected_polys.base_poly_count():,}-polygon unit "
                    f"pattern tiled {_gds_selected_polys.instance_count():,}× "
                    f"= {len(_gds_selected_polys):,} polygons. Showing the "
                    "mask bounding box per grid; the time estimate uses the "
                    "single-grid mask area replicated per grid.]"
                )
            else:
                st.caption(
                    f":orange[Layer has {len(_gds_selected_polys):,} polygons "
                    f"(> {_POLY_LIMIT:,}); showing the mask bounding box. The "
                    "time estimate uses the single-grid mask area (sum of "
                    "polygon areas inside the grid), replicated per grid.]"
                )

        _render_time_calculator(
            "ebc_dt", _dt_polys_mm, _dt_cells,
            chip_size_mm=chip_size,
            dotmap=int(st.session_state["ebc_dotmap"]),
            dose_ramp=True,
            precomputed_cell_areas=_dt_precomp,
        )

    elif mode == "First Exposure":
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
            with st.container(border=True):
                st.markdown("**Cel Origin (mm) in job1**")
                p11, p12 = st.columns(2)
                with p11:
                    cel_x = st.number_input("x", format="%.3f", step=0.1, min_value=0.0,
                                            key="ebc_fe_cel_x")
                with p12:
                    cel_y = st.number_input("y", format="%.3f", step=0.1, min_value=0.0,
                                            key="ebc_fe_cel_y")
        with p2:
            with st.container(border=True):
                st.markdown("**Grid Count in job1**")
                p21, p22 = st.columns(2)
                with p21:
                    Nx = st.number_input("Nx", min_value=1, step=1,
                                        key="ebc_fe_nx")
                with p22:
                    Ny = st.number_input("Ny", min_value=1, step=1,
                                    key="ebc_fe_ny")
        with p3:
            with st.container(border=True):
                st.markdown("**Shift (mm) in job3**")
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
        tr = (st.session_state["ebc_tr_x"], st.session_state["ebc_tr_y"])
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
        elif _fe_huge:
            for _t in _mask_overlay_traces(
                    _gds_selected_polys, _gds_unit_to_mm,
                    cel_x + shift_x, cel_y + shift_y, "#2ca02c", "Mask"):
                fig.add_trace(_t)

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

        preset_name = segmented_radio(
            "Cross-position preset",
            list(_CROSS_PRESETS.keys()),
            key="ebc_sa_preset",
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
                "Input unit", ["mm", "μm"],
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
                    st.markdown(f"**Mark M1 ({unit})**")
                    m1cx, m1cy = st.columns(2)
                    with m1cx:
                        m1_x = _mark_input("x", "ebc_sa_custom_m1_x")
                    with m1cy:
                        m1_y = _mark_input("y", "ebc_sa_custom_m1_y")
            with cm2:
                with st.container(border=True):
                    st.markdown(f"**Mark M2 ({unit})**")
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
                    with st.container(border=True):
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
                                "Target x", format="%.4f", step=0.001,
                                key=f"{key_prefix}_target_x",
                            )
                        with target_y_input:
                            target_y = st.number_input(
                                "Target y", format="%.4f", step=0.001,
                                key=f"{key_prefix}_target_y",
                            )

                shift_x = target_x - anchor_x
                shift_y = target_y - anchor_y

                fig = go.Figure()
                if polys and len(polys) > _POLY_LIMIT:
                    for _t in _mask_overlay_traces(
                            polys, scale_to_mm, shift_x, shift_y,
                            "#1f77b4", "Pattern"):
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
                    with st.container(border=True):
                        st.markdown("**Cel Origin (mm) in job1**")
                        sap_cel_x = st.number_input(
                            "x", format="%.3f", step=0.1, min_value=0.0,
                            key="ebc_sa_sap_cel_x")
                        sap_cel_y = st.number_input(
                            "y", format="%.3f", step=0.1, min_value=0.0,
                            key="ebc_sa_sap_cel_y")
                with sp2:
                    with st.container(border=True):
                        st.markdown("**Grid Count in job1**")
                        sap_nx = st.number_input(
                            "Nx", min_value=1, step=1,
                            key="ebc_sa_sap_nx")
                        sap_ny = st.number_input(
                            "Ny", min_value=1, step=1,
                            key="ebc_sa_sap_ny")
                with st.container(border=True):
                    st.markdown("**Registration mark position in job1**")
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
                                f"**Mark 1 ({m1_name})**",
                                help=f"x = {mk1x_mask} mm + {celx:.3f} mm, "
                                    f"y = {mk1y_mask} mm + {cely:.3f} mm",
                            )
                            st.markdown(f"x: {mk1x:.4f}")
                            st.markdown(f"y: {mk1y:.4f}")
                    with m2_col:
                        with st.container(border=True):
                            st.markdown(
                                f"**Mark 2 ({m2_name})**",
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
                            "#2ca02c", "Mask"):
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
            st.subheader("Overlayed")

            ov_plot_col, ov_ctrl_col = st.columns([7, 3])

            with ov_ctrl_col:
                cross_labels = list(crosses.keys())
                with st.container(border=True):
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
                    
                with st.container(border=True):
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
                        "#1f77b4", "Existing Pattern"):
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
                extra_help_under_total=(
                    "Note: an additional 1–2 hours is typically needed "
                    "to find the alignment marks (not included in "
                    "this estimate)."
                ),
                disabled=_marks_outside_ov,
                disabled_reason=(
                    "One or more selected marks fall outside the "
                    "exposure grid. Increase Nx/Ny or adjust cel "
                    "before calculating."
                ),
                precomputed_cell_areas=_sa_precomp,
                coverage_layer=sap_polys if _sa_huge else None,
                coverage_scale=scale_sap,
                coverage_ox=sap_cel_x + overlay_shift_x,
                coverage_oy=sap_cel_y + overlay_shift_y,
            )