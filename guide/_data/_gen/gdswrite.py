"""
guide/_data/_gen/gdswrite.py — minimal hand-rolled GDSII stream writer.

No gdstk in this sandbox (see guide/_harness/README.md), so this writes raw
GDSII records with `struct`, matching exactly what
tools/process/ebeam/gdsii/parser.py decodes (see its _T_* tag constants — HEADER /
BGNLIB / LIBNAME / UNITS / BGNSTR / STRNAME / BOUNDARY / ENDSTR / ENDLIB).

Units: 1 user unit = 1 um, 1 database unit = 1 nm (UNITS record = 1e-3,
1e-9) — every coordinate passed to `boundary()` is in um (float) and gets
rounded to the nearest nm integer here.
"""
from __future__ import annotations

import struct


def _rec(tag: int, data: bytes = b"") -> bytes:
    if len(data) % 2 == 1:
        data += b"\x00"
    length = 4 + len(data)
    return struct.pack(">HH", length, tag) + data


def _ascii(tag: int, s: str) -> bytes:
    b = s.encode("ascii")
    if len(b) % 2 == 1:
        b += b"\x00"
    return _rec(tag, b)


def _int2(tag: int, vals) -> bytes:
    return _rec(tag, struct.pack(f">{len(vals)}h", *vals))


def _int4(tag: int, vals) -> bytes:
    return _rec(tag, struct.pack(f">{len(vals)}i", *vals))


def _real8(v: float) -> bytes:
    """Encode one IEEE double as a GDSII 8-byte excess-64 base-16 real.

    Inverse of parser._gds_real8: v = mant/2**56 * 16**(exp-64), sign bit
    in the top bit of byte 0.
    """
    if v == 0:
        return b"\x00" * 8
    sign = 0x80 if v < 0 else 0
    v = abs(v)
    exp = 0
    while v >= 1.0:
        v /= 16.0
        exp += 1
    while v < 1.0 / 16.0:
        v *= 16.0
        exp -= 1
    mantissa = round(v * (1 << 56))
    if mantissa >= (1 << 56):
        mantissa >>= 4
        exp += 1
    exp_byte = (exp + 64) & 0x7F | sign
    return bytes([exp_byte]) + mantissa.to_bytes(7, "big")


def _units(tag: int, a: float, b: float) -> bytes:
    return _rec(tag, _real8(a) + _real8(b))


# ─── Record tags (must match tools/process/ebeam/gdsii/parser.py) ───────────────────
T_HEADER = 0x0002
T_BGNLIB = 0x0102
T_LIBNAME = 0x0206
T_UNITS = 0x0305
T_ENDLIB = 0x0400
T_BGNSTR = 0x0502
T_STRNAME = 0x0606
T_ENDSTR = 0x0700
T_BOUNDARY = 0x0800
T_LAYER = 0x0D02
T_DATATYPE = 0x0E02
T_XY = 0x1003
T_ENDEL = 0x1100


class GDSWriter:
    """Accumulates structures, each a list of (layer, datatype, [(x, y), ...])
    boundary polygons in user units (um); writes one GDSII stream on demand."""

    def __init__(self, libname: str = "LIB", db_user: float = 1e-3, db_meters: float = 1e-9):
        self.libname = libname
        self.db_user = db_user      # database units per user unit (1e-3: 1 um user, 1 nm db)
        self.db_meters = db_meters  # meters per database unit
        self._structs: dict[str, list] = {}
        self.polygon_count = 0
        self.total_area_um2 = 0.0

    def new_structure(self, name: str):
        self._structs.setdefault(name, [])
        return name

    def boundary(self, struct_name: str, layer: int, datatype: int, pts_um):
        """Add a closed BOUNDARY polygon. `pts_um` is a list of (x, y) in
        um; the closing vertex (first == last) is added automatically if
        not already present."""
        pts = list(pts_um)
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        self._structs[struct_name].append((layer, datatype, pts))
        self.polygon_count += 1
        self.total_area_um2 += _shoelace_area(pts)

    def to_bytes(self) -> bytes:
        out = bytearray()
        out += _int2(T_HEADER, [600])
        out += _int2(T_BGNLIB, [0] * 12)
        out += _ascii(T_LIBNAME, self.libname)
        out += _units(T_UNITS, self.db_user, self.db_meters)

        um_to_db = 1.0 / self.db_user  # db units per um (1000 for 1nm db unit)
        for name, elements in self._structs.items():
            out += _int2(T_BGNSTR, [0] * 12)
            out += _ascii(T_STRNAME, name)
            for layer, datatype, pts in elements:
                out += _rec(T_BOUNDARY)
                out += _int2(T_LAYER, [layer])
                out += _int2(T_DATATYPE, [datatype])
                xy = []
                for x, y in pts:
                    xy.append(round(x * um_to_db))
                    xy.append(round(y * um_to_db))
                out += _int4(T_XY, xy)
                out += _rec(T_ENDEL)
            out += _rec(T_ENDSTR)
        out += _rec(T_ENDLIB)
        return bytes(out)

    def write(self, path: str):
        with open(path, "wb") as fh:
            fh.write(self.to_bytes())


def _shoelace_area(pts):
    """|area| of a closed polygon (last point == first), um^2."""
    n = len(pts) - 1
    s = 0.0
    for k in range(n):
        x0, y0 = pts[k]
        x1, y1 = pts[k + 1]
        s += x0 * y1 - x1 * y0
    return abs(s) * 0.5


# ─── Shared shapes ───────────────────────────────────────────────────────────

def rect(cx, cy, w, h):
    """Rectangle corners (CCW), centered at (cx, cy), width w, height h."""
    hw, hh = w / 2.0, h / 2.0
    return [(cx - hw, cy - hh), (cx + hw, cy - hh),
            (cx + hw, cy + hh), (cx - hw, cy + hh)]


def cross(cx, cy, arm_len=40.0, arm_w=8.0):
    """12-vertex plus-sign (alignment cross) centered at (cx, cy)."""
    a, w = arm_len / 2.0, arm_w / 2.0
    return [
        (cx - w, cy + a), (cx + w, cy + a), (cx + w, cy + w),
        (cx + a, cy + w), (cx + a, cy - w), (cx + w, cy - w),
        (cx + w, cy - a), (cx - w, cy - a), (cx - w, cy - w),
        (cx - a, cy - w), (cx - a, cy + w), (cx - w, cy + w),
    ]
