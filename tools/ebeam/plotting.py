"""
plotting.py — Plotly trace builders, coverage-raster generation and
per-cell area binning for the EBL calculator's GDS mask viewer.

Split out of ``tools/ebeam/calculator.py`` (mechanical refactor: moved
as-is, no behaviour change, no numbers changed). Self-contained like the
rest of the EBL calculator (see ``tools/ebeam/AGENTS.md``): imports
nothing outside ``tools/ebeam`` itself.

``_PALETTE`` and ``_hex_to_rgba`` moved here from ``calculator.py``
alongside the trace builders that are their main users.
``tools/ebeam/calculator.py`` keeps its own separate copy of
``_hex_to_rgba`` for its own module-level UI script (see that file's
docstring for why it cannot just import this one back — the short version:
``calculator.py`` is always the Streamlit entry script, never importable
under its own dotted name without re-running the whole page) and imports
``_PALETTE`` from here since that one is plain data, not a function, so
there is no re-execution hazard either direction.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from tools.ebeam.gdsii.limits import _POLY_LIMIT, tr
from tools.ebeam.gdsii.parser import (
    _PolyLayer, _InstancedLayer, _expand_groups_to_flat, _FLAT_RASTER_CHUNK,
)
from tools.ebeam.gdsii.stream import _RASTER_N, _INSTANCE_CHUNK, _bin_points


_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


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
    if isinstance(polys, (_PolyLayer, _InstancedLayer, _StreamLayer)):
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
        k = int(off.shape[0])
        for s in range(0, k, _INSTANCE_CHUNK):
            e = min(s + _INSTANCE_CHUNK, k)
            px = (cxc + off[s:e, 0]) * scale_to_mm + ox
            py = (cyc + off[s:e, 1]) * scale_to_mm + oy
            w = np.full(e - s, base_area, dtype=np.float64)
            _bin_points(out, px, py, w, gx0, gy0, chip_size_mm, nx, ny)
    return out.tolist()


class _StreamLayer:
    """One ``(layer, datatype)`` of a ``_StreamSummary``, wearing enough of
    the ``_PolyLayer`` / ``_InstancedLayer`` interface that the viewer and
    the Time Calculator can consume it unchanged.

    This is what the page falls back to when a mask is too big for the full
    parse: there is no polygon array behind it at all, only the streamed
    reductions (counts, bbox, area, the coverage/area rasters). Anything
    that needs real geometry — drawing individual polygons, the unit
    pattern — has to go through ``_stream_window`` for a small region
    instead, because the whole layer never exists in memory.
    """
    __slots__ = ("summary", "key", "store")

    def __init__(self, summary, key, store):
        self.summary = summary
        self.key = key
        self.store = store

    @property
    def _agg(self):
        return self.summary.layers[self.key]

    def __len__(self):
        return int(self._agg.polygons)

    def __bool__(self):
        return self._agg.polygons > 0

    def bbox(self):
        return self._agg.bbox

    def instance_count(self):
        return int(self._agg.placements)

    def base_poly_count(self):
        return 0

    def total_area_units2(self):
        return float(self._agg.area_units2)


def _cell_areas_binned(layer, scale_to_mm, ox, oy, gx0, gy0,
                       chip_size_mm, nx, ny) -> list:
    """Dispatch per-cell area binning to the flat or instanced helper."""
    if isinstance(layer, _StreamLayer):
        # No geometry to bin — re-bin the streamed area raster instead.
        return layer.summary.cell_areas(
            scale_to_mm, ox, oy, gx0, gy0, chip_size_mm, nx, ny,
            layer=layer.key)
    if isinstance(layer, _InstancedLayer):
        return _instanced_cell_areas_binned(
            layer, scale_to_mm, ox, oy, gx0, gy0, chip_size_mm, nx, ny)
    return _fast_cell_areas_binned(
        layer, scale_to_mm, ox, oy, gx0, gy0, chip_size_mm, nx, ny)


def _hex_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _stream_coverage_grid(layer: "_StreamLayer", scale_to_mm: float,
                          ox: float, oy: float):
    """``_coverage_grid`` equivalent for a streamed layer: the scan already
    binned every placement into ``count_raster`` over the file's extent, so
    this only converts that fixed grid into placed-mm pixel geometry. Row 0
    is the bottom, matching what ``go.Image`` / the heatmap expect."""
    agg = layer._agg
    extent = layer.summary.extent
    if extent is None or not agg.placements:
        return None
    ex0, ey0, ex1, ey1 = extent
    dx = (ex1 - ex0) / _RASTER_N * scale_to_mm
    dy = (ey1 - ey0) / _RASTER_N * scale_to_mm
    if dx <= 0 or dy <= 0:
        return None
    x0c = ex0 * scale_to_mm + ox + 0.5 * dx
    y0c = ey0 * scale_to_mm + oy + 0.5 * dy
    return agg.count_raster, x0c, dx, y0c, dy


def _layer_coverage_grid(layer, scale_to_mm: float, ox: float, oy: float,
                         px: int):
    """``(cnt, x0_center, dx, y0_center, dy)`` coverage grid for *any* layer
    kind — streamed, instanced or flat — or None.

    Every layer can produce one, which is the point: a low-res raster is
    the one view that never depends on how much of the mask fits in
    memory, so it is what the viewer falls back to instead of showing
    nothing."""
    if isinstance(layer, _StreamLayer):
        return _stream_coverage_grid(layer, scale_to_mm, ox, oy)
    if isinstance(layer, _InstancedLayer):
        return _coverage_grid(layer, scale_to_mm, ox, oy, px)
    if isinstance(layer, _PolyLayer):
        return _flat_coverage_grid(layer, scale_to_mm, ox, oy, px)
    return None


def _rasterize_coverage(layer, scale_to_mm: float,
                        ox: float, oy: float, color_hex: str, px: int = 360):
    """Bin a layer into a low-res pixel grid (coverage raster) so a whole
    mask can be shown as ONE small image instead of millions of vector
    points. Empty pixels are WHITE so the view reads the same regardless of
    light/dark theme. Returns ``(rgba, x0, dx, y0, dy)`` for a ``go.Image``
    trace (row 0 = bottom; pair with a non-reversed y-axis), or None."""
    grid = _layer_coverage_grid(layer, scale_to_mm, ox, oy, px)
    if grid is None:
        return None
    cnt, x0c, psz, y0c, _ = grid
    r, g, b = _hex_rgb(color_hex)
    rgba = np.empty((cnt.shape[0], cnt.shape[1], 4), dtype=np.uint8)
    rgba[:] = (255, 255, 255, 255)          # white background
    rgba[cnt > 0] = (r, g, b, 255)          # pattern coverage
    return rgba, x0c, psz, y0c, psz


def _raster_shape(w: float, h: float, px: int):
    """``(ncols, nrows, pixel_size)`` for a ``w`` × ``h`` box drawn at most
    ``px`` pixels along its *longer* axis, with square pixels.

    Sizing off the width alone (the obvious reading of "px columns") makes
    the row count unbounded: a mask 30× taller than it is wide would want a
    ~10 000-row image, tens of MB of counts and RGBA for something the
    browser then downscales anyway."""
    psz = max(w, h) / max(1, int(px))
    return (max(1, int(round(w / psz))),
            max(1, int(round(h / psz))), psz)


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
    ncols, nrows, psz = _raster_shape(w, h, px)
    cnt = np.zeros((nrows, ncols), dtype=np.int64)
    for bcx, bcy, bst, off in layer.groups:
        base = _PolyLayer(bcx, bcy, bst)
        bbb = base.bbox()
        if bbb is None:
            continue
        cxc = 0.5 * (bbb[0] + bbb[2])
        cyc = 0.5 * (bbb[1] + bbb[3])
        # One point per instance is right while a unit cell is sub-pixel
        # (the usual EBL mask: sub-µm shapes, tens of µm per pixel). A cell
        # *bigger* than a pixel — a handful of large stepped blocks — would
        # otherwise light a handful of dots and read as an empty mask, so
        # spread each instance over the pixels its own extent covers.
        rx = int((bbb[2] - bbb[0]) * scale_to_mm / psz) // 2
        ry = int((bbb[3] - bbb[1]) * scale_to_mm / psz) // 2
        # Dilation needs this group's own counts isolated, so it gets a
        # scratch grid; without it the instances go straight into `cnt`.
        # Either way the grid is ~px², not placement-sized.
        acc = (np.zeros((nrows, ncols), dtype=np.int64)
               if (rx or ry) else cnt)
        k = int(off.shape[0])
        for s in range(0, k, _INSTANCE_CHUNK):   # see _INSTANCE_CHUNK
            e = min(s + _INSTANCE_CHUNK, k)
            pxs = (cxc + off[s:e, 0]) * scale_to_mm + ox
            pys = (cyc + off[s:e, 1]) * scale_to_mm + oy
            col = np.floor((pxs - x0d) / psz).astype(np.int64)
            row = np.floor((pys - y0d) / psz).astype(np.int64)
            m = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
            np.add.at(acc, (row[m], col[m]), 1)
        if rx or ry:
            cnt += _dilate_box(acc, ry, rx)
    return cnt, x0d + psz / 2, psz, y0d + psz / 2, psz


def _dilate_box(cnt, ry: int, rx: int):
    """Box-dilate a count grid by ±``ry`` rows / ±``rx`` columns.

    Via a summed-area table, so the cost is the grid (~130 k cells), not
    the instance count — the whole point is that this stays free however
    many placements were binned into it. Only ``cnt > 0`` is ever read
    downstream, so turning counts into "an instance covers this pixel" is
    the intended meaning, not a loss."""
    if ry <= 0 and rx <= 0:
        return cnt
    nr, nc = cnt.shape
    sat = np.zeros((nr + 1, nc + 1), dtype=np.int64)
    np.cumsum(np.cumsum(cnt, axis=0), axis=1, out=sat[1:, 1:])
    r0 = np.clip(np.arange(nr) - ry, 0, nr)
    r1 = np.clip(np.arange(nr) + ry + 1, 0, nr)
    c0 = np.clip(np.arange(nc) - rx, 0, nc)
    c1 = np.clip(np.arange(nc) + rx + 1, 0, nc)
    return (sat[np.ix_(r1, c1)] - sat[np.ix_(r0, c1)]
            - sat[np.ix_(r1, c0)] + sat[np.ix_(r0, c0)])


def _flat_coverage_grid(layer: "_PolyLayer", scale_to_mm: float,
                        ox: float, oy: float, px: int):
    """``_coverage_grid`` for a flat layer: bin every *vertex* into the
    pixel grid, in bounded-memory passes.

    Vertices rather than one point per polygon, because a flat layer is
    exactly the case with no repetition to summarise: a single centre would
    erase the shape of anything bigger than a pixel. On a real mask (sub-µm
    shapes across a ~cm field, so tens of µm per pixel) every polygon is
    sub-pixel and the two agree; where they don't, the outline is the
    honest picture, since nothing here fills polygon interiors."""
    bb = _placed_bbox_mm(layer, scale_to_mm, ox, oy)
    if bb is None:
        return None
    x0d, y0d, x1d, y1d = bb
    w = x1d - x0d
    h = y1d - y0d
    if w <= 0 or h <= 0:
        return None
    ncols, nrows, psz = _raster_shape(w, h, px)
    flat = np.zeros(nrows * ncols, dtype=np.int64)
    cx, cy = layer.cx, layer.cy
    for s in range(0, cx.size, _FLAT_RASTER_CHUNK):
        e = min(s + _FLAT_RASTER_CHUNK, cx.size)
        col = (cx[s:e] * scale_to_mm + (ox - x0d)) / psz
        row = (cy[s:e] * scale_to_mm + (oy - y0d)) / psz
        # int32 indices: the grid is at most px², far inside its range, and
        # it halves the per-pass working set against the int64 default.
        ci = np.floor(col, out=col).astype(np.int32)
        ri = np.floor(row, out=row).astype(np.int32)
        del col, row
        # The grid comes from this layer's own bbox, so the only way out of
        # range is a vertex sitting exactly on the far edge — clip puts it
        # in the last pixel, where it belongs. Nothing is invented.
        np.clip(ci, 0, ncols - 1, out=ci)
        np.clip(ri, 0, nrows - 1, out=ri)
        # bincount, not np.add.at: the grid is ~130 k bins against a million
        # points per pass, so the per-call O(bins) cost is noise and the
        # per-point cost is an order of magnitude lower.
        ri *= ncols
        ri += ci
        flat += np.bincount(ri, minlength=nrows * ncols)
    return (flat.reshape(nrows, ncols),
            x0d + psz / 2, psz, y0d + psz / 2, psz)


def _coverage_heatmap_trace(layer, scale_to_mm: float, ox: float, oy: float,
                            color: str, px: int = 260):
    """A low-res coverage trace as a ``go.Heatmap`` (pattern = ``color``,
    empty = transparent) for overlaying on the workflow / time-calculator
    plots without flipping their y-axis the way ``go.Image`` would. Works
    for every layer kind (see ``_layer_coverage_grid``)."""
    grid = _layer_coverage_grid(layer, scale_to_mm, ox, oy, px)
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
            name=(f"{tr('unit', '單元')} {gi + 1} "
                  f"({int(bst.size - 1)} {tr('polys', '個多邊形')})")
                 if multi else tr("unit pattern", "單元圖案"),
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
        npoly = int(bst.size - 1)
        k = int(off.shape[0])
        picks = []
        for s in range(0, k, _INSTANCE_CHUNK):   # see _INSTANCE_CHUNK
            e = min(s + _INSTANCE_CHUNK, k)
            px = (cxc + off[s:e, 0]) * scale_to_mm + ox
            py = (cyc + off[s:e, 1]) * scale_to_mm + oy
            m = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
            if not m.any():
                continue
            sel = off[s:e][m]
            total += sel.shape[0] * npoly
            if total > _MAX_REGION_POLYS:
                # Bail on the chunk that crosses the cap rather than after
                # the whole group: `total` is only ever shown as "more than
                # _MAX_REGION_POLYS", and finishing the group would keep
                # scanning a layer the caller has already refused to draw.
                return ("over", total)
            picks.append(sel)
        if picks:
            sel_groups.append((bcx, bcy, bst,
                               picks[0] if len(picks) == 1
                               else np.concatenate(picks)))
    if not sel_groups:
        return None
    cx, cy, starts = _expand_groups_to_flat(sel_groups)
    return cx * scale_to_mm + ox, cy * scale_to_mm + oy, starts


def _mask_overlay_traces(layer, scale_to_mm: float, ox: float, oy: float,
                         color: str, name: str) -> list:
    """Traces for a dense mask placed at (ox, oy): the low-res coverage
    raster, whatever the layer kind. Only if that cannot be built at all
    does this fall back to the bounding box — a rectangle says nothing
    about where the pattern actually sits, so it is the last resort rather
    than the answer for flat layers."""
    ht = _coverage_heatmap_trace(layer, scale_to_mm, ox, oy, color)
    if ht is not None:
        return [ht]
    bb = _placed_bbox_mm(layer, scale_to_mm, ox, oy)
    if bb is None:
        return []
    return [_bbox_rect_trace(
        bb, color,
        f"{name} (bbox, {len(layer):,} {tr('polys', '個多邊形')})")]


def _use_coverage_raster(layer) -> bool:
    """True when a layer should be shown as the low-res coverage raster
    rather than drawn polygon by polygon: streamed (no geometry in memory
    at all), instanced (repetition worth summarising), or simply past
    ``_POLY_LIMIT``. Below that a layer is drawn exactly, which is always
    the better picture when it is affordable."""
    if isinstance(layer, (_InstancedLayer, _StreamLayer)):
        return True
    return bool(layer) and len(layer) > _POLY_LIMIT


def _dense_layer_note(layer) -> str:
    """One-line caption describing how a dense layer is being shown."""
    if isinstance(layer, _StreamLayer):
        return tr(
            f":orange[Large-mask mode: this file is too big to load fully "
            f"on this machine, so it was streamed instead — "
            f"{layer.instance_count():,} placements / {len(layer):,} "
            "polygons measured without ever holding the geometry. The "
            "overview and the time estimate are exact; drawing individual "
            "polygons is unavailable.]",
            f":orange[大型遮罩模式：本機記憶體不足以完整載入此檔案，改以串流方式"
            f"掃描 — 已量測 {layer.instance_count():,} 個放置／{len(layer):,} "
            "個多邊形，過程中不需保留幾何資料。總覽與時間估算為精確值；"
            "但無法逐一繪製多邊形。]"
        )
    if isinstance(layer, _InstancedLayer):
        return tr(
            f":blue[Repetition detected: a {layer.base_poly_count():,}-polygon "
            f"unit pattern tiled {layer.instance_count():,}× "
            f"= {len(layer):,} polygons. Drawing the unit pattern + array "
            "footprint; the time estimate uses unit area × tile count.]",
            f":blue[偵測到重複結構：{layer.base_poly_count():,} 個多邊形的單元圖案"
            f"重複排列 {layer.instance_count():,} 次 = {len(layer):,} 個多邊形。"
            "顯示單元圖案 + 陣列覆蓋範圍；時間估算採用單元面積 × 重複次數。]"
        )
    return tr(
        f":orange[Layer has {len(layer):,} polygons (> {_POLY_LIMIT:,}) with "
        "no repeated unit pattern, so it is shown as a low-res coverage "
        "raster — one pixel lit wherever the mask has geometry — unless "
        "you ask for every polygon below. The time estimate is unaffected "
        "either way: it uses each polygon's full area, binned to the grid "
        "cell holding its first vertex.]",
        f":orange[圖層含有 {len(layer):,} 個多邊形（> {_POLY_LIMIT:,}）且無重複"
        "單元圖案，因此以低解析度覆蓋圖顯示 — 遮罩有幾何之處即點亮一個像素 — "
        "除非於下方選擇繪製每一個多邊形。時間估算兩者皆不受影響：仍採用各多邊形"
        "的完整面積，並依其第一個頂點所在的網格分組計算。]"
    )


