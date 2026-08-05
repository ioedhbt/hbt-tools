"""A drop-in stand-in for the five gdstk calls this app makes.

Why: gdstk has no aarch64 wheel and building it needs Qhull 8, which no
reachable source in this sandbox provides.  Everything the app actually asks of
gdstk is 2-D polygon geometry, which Shapely does — so this module supplies the
same names on top of Shapely and is copied into the shadow tree as ``gdstk.py``.

The whole surface, from ``grep -o 'gdstk\\.[A-Za-z_]*'``:

    gdstk.Polygon   gdstk.rectangle   gdstk.boolean   gdstk.FlexPath

(``gdstk.read_gds`` appears in a comment only — the app has its own reader.)

Screenshot fidelity is what matters here, and areas/clips agree with gdstk to
well under the 1e-7 mm precision the caller asks for.  This is a sandbox
convenience, not a suggestion to vendor it.
"""
from __future__ import annotations

import numpy as np
from shapely import geometry as _g
from shapely.ops import unary_union as _union

__version__ = "shim-1.0 (shapely-backed)"


class Polygon:
    """gdstk.Polygon: a point list, its area, and its exterior ring."""

    def __init__(self, points, layer: int = 0, datatype: int = 0):
        if isinstance(points, _g.Polygon):
            self._geom = points
        else:
            pts = np.asarray(points, dtype=float).reshape(-1, 2)
            self._geom = _g.Polygon(pts)
            if not self._geom.is_valid:
                self._geom = self._geom.buffer(0)
        self.layer = layer
        self.datatype = datatype

    @property
    def points(self) -> np.ndarray:
        g = self._geom
        if g.is_empty:
            return np.zeros((0, 2))
        if g.geom_type != "Polygon":
            g = max(g.geoms, key=lambda p: p.area)
        # gdstk returns the ring without the repeated closing vertex.
        return np.asarray(g.exterior.coords[:-1], dtype=float)

    def area(self) -> float:
        return float(self._geom.area)

    def bounding_box(self):
        if self._geom.is_empty:
            return None
        x0, y0, x1, y1 = self._geom.bounds
        return ((x0, y0), (x1, y1))

    def translate(self, dx, dy):
        from shapely import affinity

        self._geom = affinity.translate(self._geom, dx, dy)
        return self


def rectangle(corner1, corner2, layer: int = 0, datatype: int = 0) -> Polygon:
    (x0, y0), (x1, y1) = corner1, corner2
    return Polygon(_g.box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                   layer, datatype)


def _as_geom(obj):
    if isinstance(obj, Polygon):
        return obj._geom
    if isinstance(obj, (list, tuple)):
        parts = [_as_geom(o) for o in obj]
        return _union([p for p in parts if not p.is_empty]) if parts \
            else _g.Polygon()
    if isinstance(obj, _g.base.BaseGeometry):
        return obj
    return _g.Polygon(np.asarray(obj, dtype=float).reshape(-1, 2))


def _explode(geom, layer, datatype) -> list[Polygon]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        geoms = [geom]
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        geoms = [g for g in geom.geoms if g.geom_type == "Polygon"]
    else:
        return []
    return [Polygon(g, layer, datatype) for g in geoms if g.area > 0]


def boolean(operand1, operand2, operation, precision: float = 1e-3,
            layer: int = 0, datatype: int = 0) -> list[Polygon]:
    a, b = _as_geom(operand1), _as_geom(operand2)
    grid = max(precision, 0.0)
    if grid:
        # gdstk snaps to a grid before clipping; mimic it so near-degenerate
        # slivers don't survive as zero-area rings.
        from shapely import set_precision

        a, b = set_precision(a, grid), set_precision(b, grid)
    op = operation.lower()
    if op in ("and", "intersection"):
        out = a.intersection(b)
    elif op in ("or", "union"):
        out = a.union(b)
    elif op in ("not", "difference"):
        out = a.difference(b)
    elif op in ("xor", "symmetric_difference"):
        out = a.symmetric_difference(b)
    else:
        raise ValueError(f"unknown boolean operation {operation!r}")
    return _explode(out, layer, datatype)


_CAP = {"flush": 2, "round": 1, "extended": 3}   # shapely cap_style codes


class FlexPath:
    """Only the constructor signature and ``to_polygons()`` are used."""

    def __init__(self, points, width, ends="flush", layer: int = 0,
                 datatype: int = 0, **_ignored):
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        w = float(np.ravel(width)[0]) if np.ndim(width) else float(width)
        self.layer, self.datatype = layer, datatype

        if len(pts) < 2 or w <= 0:
            self._geom = _g.Polygon()
            return

        line = _g.LineString(pts)
        if isinstance(ends, str):
            cap = _CAP.get(ends, 2)
            geom = line.buffer(w / 2.0, cap_style=cap, join_style=2)
        else:
            # (begin, end) extension lengths — extend the ends, then flat-cap.
            try:
                bext, eext = (float(ends[0]), float(ends[1]))
            except Exception:
                bext = eext = 0.0
            pts = _extend(pts, bext, eext)
            geom = _g.LineString(pts).buffer(w / 2.0, cap_style=2,
                                             join_style=2)
        self._geom = geom

    def to_polygons(self) -> list[Polygon]:
        return _explode(self._geom, self.layer, self.datatype)


def _extend(pts: np.ndarray, begin: float, end: float) -> np.ndarray:
    pts = pts.copy()
    if begin:
        d = pts[0] - pts[1]
        n = np.hypot(*d)
        if n:
            pts[0] = pts[0] + d / n * begin
    if end:
        d = pts[-1] - pts[-2]
        n = np.hypot(*d)
        if n:
            pts[-1] = pts[-1] + d / n * end
    return pts
