"""
Synthetic GDSII generator for stress-testing the EBL page's mask viewer.

Builds masks out of the primitive shapes an e-beam pattern is actually
made of — rectangles (axis-aligned and rotated), polygonized circles and
triangles — each **under 1 µm**, packed into one 2 × 2 µm unit *cell*
that is then repeated over a grid of at most **1 × 1 cm**.  That is the
shape of every real EBL mask this app sees: a tiny unit pattern, copied
tens of thousands to millions of times.

Why a hand-rolled writer instead of ``gdstk``: emitting ~7 M cell
placements through gdstk would materialize ~7 M C++ objects (>1 GB) just
to write the file.  Here the records are built as flat numpy byte blocks
and streamed out in chunks, so a 250 MB file writes in a few seconds
inside a few hundred MB of RAM.  The bytes are also *byte-identical
except for the XY payload* between consecutive placements — exactly the
run structure `_element_run()` in ``tools/ebeam_calculator.py`` bulk
decodes, so the generated files exercise the parser's fast path the same
way CAD-tool output does.

Modes (``--mode``):

| mode        | what it emits                          | what it stresses |
|-------------|----------------------------------------|------------------|
| ``sref``    | N individual SREFs of the unit cell    | run-decode fast path, offset-table RAM (the realistic big-file case) |
| ``sref_norun`` | SREFs with cycling rotations        | worst case: runs broken, per-element Python loop |
| ``aref``    | one AREF (cols × rows)                 | tiny file, huge expansion — the placement budget |
| ``flat``    | every shape written out at every site  | no repetition at all — the distinct-vertex budget |
| ``mixed``   | flat half + referenced half, on two layers | both budgets loaded at once — the true worst case per MB |
| ``bigcell`` | one cell too big to stay instanced, stepped | forced flat expansion — the expanded-vertex budget |
| ``multi``   | several *different* cells, interleaved site-by-site | many cell types with the runs broken by the ordering |
| ``multi_grouped`` | the same cells, one contiguous span each | many cell types with the runs intact (what CAD tools emit) |

Usage (from repo root):

    .hbttools/bin/python gds/_gen_test_gds.py out.gds --mode sref --mb 220
    .hbttools/bin/python gds/_gen_test_gds.py out.gds --mode aref --places 50e6
    .hbttools/bin/python gds/_gen_test_gds.py out.gds --mode bigcell --places 9

See ``gds/_profile_gds_limits.py`` for the sweep that generates these
and measures parse time / peak RSS against the Streamlit Cloud budget.
"""
from __future__ import annotations

import argparse
import math
import struct
import numpy as np

# ─── GDSII record tags (mirror of the reader's table in ebeam_calculator) ────
_T_HEADER = 0x0002;   _T_BGNLIB = 0x0102;   _T_LIBNAME = 0x0206
_T_UNITS = 0x0305;    _T_ENDLIB = 0x0400
_T_BGNSTR = 0x0502;   _T_STRNAME = 0x0606;  _T_ENDSTR = 0x0700
_T_BOUNDARY = 0x0800; _T_SREF = 0x0A00;     _T_AREF = 0x0B00
_T_LAYER = 0x0D02;    _T_DATATYPE = 0x0E02
_T_XY = 0x1003;       _T_ENDEL = 0x1100;    _T_SNAME = 0x1206
_T_COLROW = 0x1302;   _T_STRANS = 0x1A01;   _T_ANGLE = 0x1C05

# Database unit = 1 nm, user unit = 1 µm (the usual EBL convention), so all
# integer coordinates below are nanometres.
_DBU_PER_UM = 1000
_UNIT_CELL_UM = 2.0                      # unit-cell pitch floor (footprint)
_MAX_SPAN_UM = 10_000.0                  # 1 cm — the hard area constraint
_COLROW_MAX = 32_767                     # AREF COLROW fields are int16


def _rec(tag: int, payload: bytes = b"") -> bytes:
    """One GDSII record: 2-byte length (incl. header), 2-byte tag, payload."""
    return struct.pack(">HH", 4 + len(payload), tag) + payload


def _name(text: str) -> bytes:
    """ASCII payload padded to an even length (GDSII requires even records)."""
    b = text.encode("ascii")
    return b + b"\0" if len(b) & 1 else b


def _pack_real8(v: float) -> bytes:
    """Encode a float as a GDSII 8-byte excess-64 real (inverse of
    ``_gds_real8()`` in the reader)."""
    if v == 0.0:
        return b"\x00" * 8
    sign = 0x80 if v < 0 else 0x00
    v = abs(v)
    exp = 0
    while v >= 1.0:
        v /= 16.0
        exp += 1
    while v < 1.0 / 16.0:
        v *= 16.0
        exp -= 1
    mant = int(round(v * (1 << 56)))
    if mant >= (1 << 56):                # rounding pushed us over one nibble
        mant >>= 4
        exp += 1
    return bytes([sign | ((exp + 64) & 0x7F)]) + mant.to_bytes(7, "big")


# ─── Unit-cell geometry (every shape < 1 µm, packed into 2 × 2 µm) ──────────

def _rect(w: float, h: float, angle_deg: float,
          cx: float, cy: float) -> np.ndarray:
    """Rectangle ``w`` × ``h`` (nm) rotated ``angle_deg`` about its center."""
    hw, hh = w / 2.0, h / 2.0
    pts = np.array([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)], float)
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + (cx, cy)


def _circle(r: float, n: int, cx: float, cy: float) -> np.ndarray:
    """Circle of radius ``r`` (nm) polygonized with ``n`` vertices."""
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def _triangle(side: float, angle_deg: float,
              cx: float, cy: float) -> np.ndarray:
    """Equilateral triangle of edge ``side`` (nm) about its centroid."""
    r = side / math.sqrt(3.0)            # centroid → vertex
    t = np.radians(np.array([90.0, 210.0, 330.0]) + angle_deg)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def _iso_triangle(base: float, height: float, angle_deg: float,
                  cx: float, cy: float) -> np.ndarray:
    """Isosceles triangle, ``base`` wide and ``height`` tall (nm), rotated
    ``angle_deg`` about its centroid."""
    pts = np.array([(-base / 2.0, -height / 3.0), (base / 2.0, -height / 3.0),
                    (0.0, 2.0 * height / 3.0)], float)
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return pts @ np.array([[c, -s], [s, c]]).T + (cx, cy)


def _star(r_out: float, r_in: float, points: int, angle_deg: float,
          cx: float, cy: float) -> np.ndarray:
    """``points``-pointed star: outer and inner radii alternating (nm)."""
    t = np.radians(angle_deg) + np.arange(2 * points) * (math.pi / points)
    r = np.where(np.arange(2 * points) & 1, r_in, r_out)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def unit_cell_shapes() -> list[np.ndarray]:
    """The repeated unit pattern: 8 polygons inside a 2 × 2 µm footprint,
    every one under 1 µm across (largest is the 900 × 120 nm slot, 907 nm
    on the diagonal).  Mixed vertex counts (3 / 4 / 24 / 32) on purpose —
    the parser bulk-decodes runs of *same-shape* elements, so a real mix
    keeps the flat mode honest."""
    return [
        _rect(800, 260, 0.0, 500, 400),        # axis-aligned bar
        _rect(620, 200, 30.0, 1500, 400),      # rotated bar
        _rect(450, 450, -15.0, 500, 1150),     # tilted square
        _rect(900, 120, 75.0, 1000, 900),      # steep thin slot
        _circle(300, 32, 1500, 1200),          # via-like disc
        _circle(180, 24, 1000, 1800),          # small dot
        _triangle(700, 0.0, 250, 1750),        # marker
        _triangle(520, 45.0, 1750, 1800),      # rotated marker
    ]


def cell_shape_sets() -> list[list[np.ndarray]]:
    """Distinct unit patterns for the ``multi`` modes — one *set* of shapes
    per cell definition, every shape under 1 µm inside the same 2 × 2 µm
    footprint so any set can sit at any grid site.

    Deliberately unequal in polygon and vertex count: a mask built from
    several different cells is not the same load as one cell repeated, and
    the difference has to show up in the numbers.
    """
    return [
        # square + circle + triangle                      (3 polys, 39 verts)
        [_rect(700, 700, 0.0, 550, 550),
         _circle(320, 32, 1450, 550),
         _triangle(760, 0.0, 550, 1450)],
        # tilted square + isosceles triangle               (2 polys,  7 verts)
        [_rect(620, 620, 25.0, 700, 700),
         _iso_triangle(820, 900, 10.0, 1300, 1350)],
        # rectangle + star                                 (2 polys, 14 verts)
        [_rect(940, 300, 0.0, 1000, 550),
         _star(420, 190, 5, 18.0, 1000, 1350)],
        # thin slot + dot + rotated bar + small circle     (4 polys, 46 verts)
        [_rect(900, 110, 75.0, 600, 600),
         _circle(150, 24, 1500, 500),
         _rect(700, 240, -40.0, 1200, 1300),
         _circle(240, 16, 400, 1550)],
        # two triangles + wide bar                         (3 polys, 10 verts)
        [_triangle(640, 0.0, 550, 600),
         _triangle(640, 180.0, 1400, 600),
         _rect(880, 380, 0.0, 1000, 1400)],
        # big circle + four dots                           (5 polys, 84 verts)
        [_circle(460, 48, 1000, 1000),
         _circle(120, 12, 350, 350), _circle(120, 12, 1650, 350),
         _circle(120, 12, 350, 1650), _circle(120, 12, 1650, 1650)],
    ]


def _spin(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a shape about the 2 × 2 µm cell centre (keeps it in footprint)."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return (pts - 1000.0) @ np.array([[c, -s], [s, c]]).T + 1000.0


def multi_cell_sets(k: int) -> list[list[np.ndarray]]:
    """``k`` distinct unit patterns.  The first few are the hand-written
    sets above; beyond that they are those sets spun by an angle unique to
    each, which keeps every cell definition genuinely distinct (different
    coordinates, so the reader cannot merge them) without inventing new
    geometry.  Used to push past the viewer's base-polygon ceiling."""
    pool = cell_shape_sets()
    if k <= len(pool):
        return pool[:k]
    out = list(pool)
    for i in range(len(pool), k):
        ang = 90.0 * ((i // len(pool)) % 4) + (i % len(pool)) * 1.7 + 0.3
        out.append([_spin(s, ang) for s in pool[i % len(pool)]])
    return out


def _shape_int(pts: np.ndarray) -> np.ndarray:
    """Round to whole database units and close the ring (GDSII BOUNDARY
    repeats the first point last)."""
    p = np.rint(pts).astype(np.int32)
    return np.vstack([p, p[:1]])


# ─── Grid planning ──────────────────────────────────────────────────────────

# On-disk bytes each mode spends per placement, so a target file size can
# be turned into a placement count:
#   sref        SREF 4 + SNAME 10 + XY 12 + ENDEL 4
#   sref_norun  + STRANS 6 + ANGLE 12
#   flat        per shape: BOUNDARY 4 + LAYER 6 + DATATYPE 6 + XY (4 + 8·pts)
#               + ENDEL 4, summed over the unit cell
_SREF_BYTES = 30
_SREF_NORUN_BYTES = 48
_HEAD_BYTES = 4096        # library header + unit-cell definition, rounded up


def plan_grid(n_places: int) -> tuple[int, int, int]:
    """Return ``(nx, ny, pitch_nm)`` for ``n_places`` placements laid out
    as a near-square grid that stays inside 1 × 1 cm with at least the
    2 µm unit-cell pitch (so shapes never overlap).  Raises when the
    request cannot fit."""
    nx = int(math.ceil(math.sqrt(n_places)))
    ny = int(math.ceil(n_places / nx))
    span_max = _MAX_SPAN_UM * _DBU_PER_UM
    pitch = int(span_max // max(nx, ny))
    floor = int(_UNIT_CELL_UM * _DBU_PER_UM)
    if pitch < floor:
        fit = int((_MAX_SPAN_UM / _UNIT_CELL_UM) ** 2)
        raise ValueError(
            f"{n_places:,} placements at the {_UNIT_CELL_UM:g} µm unit-cell "
            f"pitch need more than 1 × 1 cm; at most {fit:,} fit.")
    return nx, ny, pitch


def places_for_mb(target_mb: float, mode: str = "sref") -> int:
    """Placement count whose element stream lands at ``target_mb``."""
    if mode == "sref_norun":
        per = _SREF_NORUN_BYTES
    elif mode == "flat":
        per = sum(24 + 8 * (s.shape[0] + 1) for s in unit_cell_shapes())
    else:
        per = _SREF_BYTES
    return max(1, int((target_mb * 1024 * 1024 - _HEAD_BYTES) / per))


# ─── Writer ─────────────────────────────────────────────────────────────────

def _write_header(f, libname: str = "STRESS.DB") -> None:
    f.write(_rec(_T_HEADER, struct.pack(">h", 600)))
    stamp = struct.pack(">12h", *([2026, 1, 1, 0, 0, 0] * 2))
    f.write(_rec(_T_BGNLIB, stamp))
    f.write(_rec(_T_LIBNAME, _name(libname)))
    # user units per database unit (1 nm = 1e-3 µm), database unit in metres.
    f.write(_rec(_T_UNITS, _pack_real8(1.0 / _DBU_PER_UM)
                 + _pack_real8(1e-9)))


def _boundary_bytes(pts: np.ndarray, layer: int, dtype: int) -> bytes:
    """One BOUNDARY element (already-closed integer ring)."""
    return (_rec(_T_BOUNDARY)
            + _rec(_T_LAYER, struct.pack(">h", layer))
            + _rec(_T_DATATYPE, struct.pack(">h", dtype))
            + _rec(_T_XY, pts.astype(">i4").tobytes())
            + _rec(_T_ENDEL))


def _write_unit_cell(f, shapes, cname: str, layer: int, dtype: int) -> None:
    stamp = struct.pack(">12h", *([2026, 1, 1, 0, 0, 0] * 2))
    f.write(_rec(_T_BGNSTR, stamp))
    f.write(_rec(_T_STRNAME, _name(cname)))
    for pts in shapes:
        f.write(_boundary_bytes(pts, layer, dtype))
    f.write(_rec(_T_ENDSTR))


def _grid_xy(idx: np.ndarray, ny: int, pitch: int) -> np.ndarray:
    """(K, 2) int32 placement offsets for flat indices ``idx``."""
    i, j = np.divmod(idx, ny)
    out = np.empty((idx.size, 2), dtype=np.int32)
    out[:, 0] = i * pitch
    out[:, 1] = j * pitch
    return out


def _stream_blocks(f, tmpl: bytes, xy_off: int, xy_len: int,
                   fill, n: int, chunk: int) -> None:
    """Write ``n`` copies of the byte block ``tmpl``, with bytes
    ``[xy_off : xy_off + xy_len]`` of each copy replaced by ``fill(lo, hi)``.

    Building the copies as one (chunk, blk) uint8 array and patching the
    XY columns in place keeps this at memory-bandwidth speed, and every
    block stays byte-identical outside the XY window — the run structure
    the reader's ``_element_run()`` bulk-decodes.
    """
    base = np.frombuffer(tmpl, dtype=np.uint8)
    blk = base.size
    for lo in range(0, n, chunk):
        k = min(chunk, n - lo)
        arr = np.tile(base, (k, 1))
        arr[:, xy_off:xy_off + xy_len] = fill(lo, lo + k).reshape(k, xy_len)
        f.write(arr.tobytes())


def _sref_block(cname: str, angle: float | None) -> tuple[bytes, int]:
    """One SREF element block and the offset of its XY payload."""
    parts = [_rec(_T_SREF), _rec(_T_SNAME, _name(cname))]
    if angle is not None:
        parts.append(_rec(_T_STRANS, struct.pack(">H", 0)))
        parts.append(_rec(_T_ANGLE, _pack_real8(angle)))
    pre = b"".join(parts)
    return pre + _rec(_T_XY, b"\0" * 8) + _rec(_T_ENDEL), len(pre) + 4


def _write_srefs(f, cname: str, nx: int, ny: int, pitch: int, n: int,
                 chunk: int) -> None:
    """``n`` identical-except-XY SREF placements of ``cname`` on the grid."""
    tmpl, xy_off = _sref_block(cname, None)

    def fill(lo, hi):
        return (_grid_xy(np.arange(lo, hi, dtype=np.int64), ny, pitch)
                .astype(">i4").view(np.uint8))

    _stream_blocks(f, tmpl, xy_off, 8, fill, n, chunk)


def _write_srefs_norun(f, cname: str, nx: int, ny: int, pitch: int, n: int,
                       chunk: int, angles: list[float]) -> None:
    """SREFs whose rotation cycles element-by-element.

    Each block carries a different ANGLE payload than its neighbour, so
    ``_element_run()``'s "next block is identical except XY" probe fails
    every time and the reader is forced down its per-element Python path
    — the deliberate worst case for parse time.  Striding one angle at a
    time (all 0°, then all 15°, …) would *not* do this: it would just
    produce four long runs, each as fast as one.  So the tile written
    here is the whole ``len(angles)``-block group, patched at as many XY
    windows.
    """
    g = len(angles)
    parts, offs, pos = [], [], 0
    for ang in angles:
        blk, xy_off = _sref_block(cname, ang)
        parts.append(blk)
        offs.append(pos + xy_off)
        pos += len(blk)
    tile = np.frombuffer(b"".join(parts), dtype=np.uint8)

    groups = n // g
    for lo in range(0, groups, chunk):
        k = min(chunk, groups - lo)
        arr = np.tile(tile, (k, 1))
        base = (np.arange(lo, lo + k, dtype=np.int64) * g)
        for w, off in enumerate(offs):
            xy = _grid_xy(base + w, ny, pitch).astype(">i4").view(np.uint8)
            arr[:, off:off + 8] = xy.reshape(k, 8)
        f.write(arr.tobytes())

    for i in range(groups * g, n):        # ragged tail
        blk, xy_off = _sref_block(cname, angles[i % g])
        xy = _grid_xy(np.array([i]), ny, pitch).astype(">i4").tobytes()
        f.write(blk[:xy_off] + xy + blk[xy_off + 8:])


def _write_srefs_multi(f, cnames: list[str], nx: int, ny: int, pitch: int,
                       n: int, chunk: int, interleave: bool) -> None:
    """``n`` placements shared out over several *different* cells.

    Two orderings, because they are not the same file for the reader even
    though they hold the same geometry:

    ``interleave=True``  — CELL0, CELL1, CELL2, CELL0, …  Neighbouring
        blocks differ in their SNAME payload, so ``_element_run()``'s
        "next block identical except XY" probe fails at every element and
        the reader falls back to its per-element Python path.  This is
        what a layout written site-by-site (a chip with mixed devices
        stepped together) looks like on disk.

    ``interleave=False`` — all of CELL0, then all of CELL1, …  Each cell's
        span is one uninterrupted run, so the file costs the reader the
        same as a single-cell mask however many cells it holds.  This is
        what most CAD tools emit, since they write one reference list per
        child cell.
    """
    k = len(cnames)
    if interleave:
        parts, offs, pos = [], [], 0
        for cn in cnames:
            blk, xy_off = _sref_block(cn, None)
            parts.append(blk)
            offs.append(pos + xy_off)
            pos += len(blk)
        tile = np.frombuffer(b"".join(parts), dtype=np.uint8)
        groups = n // k
        # The tile is the whole k-block group, so with thousands of cells
        # it is already large — size the repeat count by bytes, not by
        # group count, or the staging array explodes.
        chunk = max(1, min(chunk, (16 << 20) // max(1, tile.size)))
        for lo in range(0, groups, chunk):
            m = min(chunk, groups - lo)
            arr = np.tile(tile, (m, 1))
            base = np.arange(lo, lo + m, dtype=np.int64) * k
            for w, off in enumerate(offs):
                xy = _grid_xy(base + w, ny, pitch).astype(">i4").view(np.uint8)
                arr[:, off:off + 8] = xy.reshape(m, 8)
            f.write(arr.tobytes())
        first = groups * k
    else:
        per = n // k
        for c, cn in enumerate(cnames):
            tmpl, xy_off = _sref_block(cn, None)
            start = c * per

            def fill(lo, hi, _s=start):
                return (_grid_xy(np.arange(_s + lo, _s + hi, dtype=np.int64),
                                 ny, pitch).astype(">i4").view(np.uint8))

            _stream_blocks(f, tmpl, xy_off, 8, fill, per, chunk)
        first = per * k

    for i in range(first, n):                 # ragged tail
        blk, xy_off = _sref_block(cnames[i % k], None)
        xy = _grid_xy(np.array([i]), ny, pitch).astype(">i4").tobytes()
        f.write(blk[:xy_off] + xy + blk[xy_off + 8:])


def _write_flat(f, shapes, nx: int, ny: int, pitch: int, n: int,
                layer: int, dtype: int, chunk: int) -> None:
    """Every shape written out at every grid site — no references at all."""
    for pts in shapes:
        m = pts.shape[0]
        pre = (_rec(_T_BOUNDARY)
               + _rec(_T_LAYER, struct.pack(">h", layer))
               + _rec(_T_DATATYPE, struct.pack(">h", dtype)))
        tmpl = (pre + _rec(_T_XY, b"\0" * (8 * m)) + _rec(_T_ENDEL))
        xy_off = len(pre) + 4

        def fill(lo, hi, _p=pts, _m=m):
            off = _grid_xy(np.arange(lo, hi, dtype=np.int64), ny, pitch)
            return ((_p[None, :, :].astype(np.int64) + off[:, None, :])
                    .astype(">i4").view(np.uint8))

        _stream_blocks(f, tmpl, xy_off, 8 * m, fill, n, chunk)


def _mixed_split(target_mb: float, verts_per_cell: int,
                 vert_budget: int) -> tuple[int, int]:
    """(flat placements, SREF placements) for ``mixed`` mode.

    The flat half is sized to land just under ``vert_budget`` distinct
    vertices — as much real geometry as the reader will accept — and the
    SREF half spends whatever bytes are left.  Both halves therefore sit
    just below their respective guards *at the same time*, which is the
    genuine worst case for a file of a given size: the guards are on
    vertices and on placements separately, so neither alone bounds the
    total.
    """
    per_flat = sum(24 + 8 * (s.shape[0] + 1) for s in unit_cell_shapes())
    budget = target_mb * 1024 * 1024 - _HEAD_BYTES
    # Vertex-limited, but never more than 70 % of the byte budget — flat
    # geometry costs ~880 B per unit pattern against 30 B for a
    # reference, so an unclamped flat half would overshoot the requested
    # file size on its own and leave no room for the referenced half.
    n_flat = max(1, min(int(vert_budget / verts_per_cell),
                        int(budget * 0.7 / per_flat)))
    n_ref = max(0, int((budget - n_flat * per_flat) / _SREF_BYTES))
    return n_flat, n_ref


def _write_big_cell(f, shapes, cname: str, layer: int, dtype: int,
                    sub_n: int, sub_pitch: int, chunk: int) -> None:
    """A cell definition whose body is ``sub_n`` copies of the unit
    shapes on an internal grid — i.e. one *large* cell (well past
    ``_POLY_LIMIT`` polygons) rather than a small repeated one.

    The reader keeps a layer instanced only when its base pattern is
    small enough to draw; a base this size forces the flat-expansion
    path instead, which is what ``_MAX_VERTICES`` guards.
    """
    stamp = struct.pack(">12h", *([2026, 1, 1, 0, 0, 0] * 2))
    f.write(_rec(_T_BGNSTR, stamp))
    f.write(_rec(_T_STRNAME, _name(cname)))
    sub_ny = int(math.ceil(math.sqrt(sub_n)))
    _write_flat(f, shapes, sub_ny, sub_ny, sub_pitch, sub_n,
                layer, dtype, chunk)
    f.write(_rec(_T_ENDSTR))


def _write_aref(f, cname: str, nx: int, ny: int, pitch: int) -> None:
    """One AREF covering the whole grid (a few dozen bytes on disk)."""
    f.write(_rec(_T_AREF))
    f.write(_rec(_T_SNAME, _name(cname)))
    f.write(_rec(_T_COLROW, struct.pack(">2h", nx, ny)))
    # XY = origin, origin + cols·pitch in x, origin + rows·pitch in y.
    f.write(_rec(_T_XY, struct.pack(">6i", 0, 0, nx * pitch, 0,
                                    0, ny * pitch)))
    f.write(_rec(_T_ENDEL))


def _write_marks(f, span_nm: int, layer: int, dtype: int) -> None:
    """Four 900 nm alignment squares written flat in the top cell — a
    second, tiny layer so the generated file also exercises the viewer's
    layer picker and the flat-layer code path."""
    for mx, my in ((0, 0), (span_nm, 0), (0, span_nm), (span_nm, span_nm)):
        f.write(_boundary_bytes(_shape_int(_rect(900, 900, 0.0, mx, my)),
                                layer, dtype))


def generate_gds(path, *, mode: str = "sref", target_mb: float | None = 220.0,
                 places: int | None = None, layer: int = 1, datatype: int = 0,
                 mark_layer: int = 2, ref_layer: int = 3, top: str = "TOP",
                 cell: str = "UCELL", ref_cell: str = "UCELL3",
                 vert_budget: int = 18_000_000, chunk: int = 200_000,
                 n_cells: int = 3) -> dict:
    """Write a synthetic mask to ``path`` and return its stats.

    Give either ``target_mb`` (file size to aim for) or ``places`` (exact
    placement count).  Returns a dict with the on-disk size, placement
    count, expanded polygon/vertex counts and the grid extent in mm —
    everything the profiler tabulates.
    """
    shapes = [_shape_int(s) for s in unit_cell_shapes()]
    # Closed rings: the reader drops the duplicated last point, so the
    # polygon vertex count it reports is one less than what we write.
    verts_per_cell = sum(s.shape[0] - 1 for s in shapes)

    multi = mode in ("multi", "multi_grouped")
    sets: list[list[np.ndarray]] = []
    cnames: list[str] = []
    if multi:
        if not 1 <= n_cells <= 99_999:
            raise ValueError("n_cells must be 1..99999")
        sets = [[_shape_int(s) for s in grp] for grp in multi_cell_sets(n_cells)]
        # 6 characters → SNAME needs no padding → the same 30 B per
        # placement as single-cell `sref`, so file size per placement is
        # directly comparable between the modes.
        cnames = [f"C{c:05d}" for c in range(n_cells)]

    n_flat = n_ref = n_sub = 0
    if mode == "bigcell":
        # ``places`` is the number of cell placements; the cell is then
        # filled with enough unit patterns to reach ``vert_budget``
        # vertices once expanded.
        places = int(places or 25)
        n_sub = max(1, int(vert_budget / (verts_per_cell * places)))
    elif mode == "mixed":
        # Two independent halves on two layers, each parked just under
        # its own guard (a single layer holding both would be expanded
        # flat and refused outright).
        n_flat, n_ref = _mixed_split(float(target_mb), verts_per_cell,
                                     vert_budget)
        places = n_flat + n_ref
    elif places is None:
        places = places_for_mb(float(target_mb), mode)
    places = int(places)
    if multi and n_cells > places:
        # Every cell must be placed at least once, or the unplaced ones
        # come back as extra top-level cells and the file no longer has
        # the single top the rest of the suite assumes.
        raise ValueError(f"{n_cells:,} cells need at least that many "
                         f"placements; this file has {places:,}.")

    if mode == "aref":
        # COLROW is int16, so an AREF grid tops out at 32767 × 32767.
        nx = min(_COLROW_MAX, int(math.ceil(math.sqrt(places))))
        ny = min(_COLROW_MAX, int(math.ceil(places / nx)))
        pitch = int(_MAX_SPAN_UM * _DBU_PER_UM // max(nx, ny))
        if pitch < 1:
            raise ValueError("AREF grid too dense for a 1 cm span")
        places = nx * ny
    else:
        nx, ny, pitch = plan_grid(places)

    # Last placement sits at (n-1)·pitch; the marks go on those corners so
    # nothing — marks included — reaches past the 1 cm budget.
    span_nm = int((max(nx, ny) - 1) * pitch)

    with open(path, "wb") as f:
        _write_header(f)
        # Define only the cell that is actually referenced, so the file
        # has exactly one top-level cell.  `mixed` puts its referenced
        # half on its own layer, so the flat half cannot merge with (and
        # force the expansion of) the instanced half.
        if multi:
            for cn, grp in zip(cnames, sets):
                _write_unit_cell(f, grp, cn, layer, datatype)
        elif mode == "mixed":
            _write_unit_cell(f, shapes, ref_cell, ref_layer, datatype)
        elif mode == "bigcell":
            # Cell footprint = one tile of the top-level grid, so the
            # stepped copies tile the 1 cm field without overlapping.
            _write_big_cell(f, shapes, cell, layer, datatype, n_sub,
                            max(int(_UNIT_CELL_UM * _DBU_PER_UM),
                                pitch // int(math.ceil(math.sqrt(n_sub)))),
                            chunk)
        elif mode != "flat":
            _write_unit_cell(f, shapes, cell, layer, datatype)

        stamp = struct.pack(">12h", *([2026, 1, 1, 0, 0, 0] * 2))
        f.write(_rec(_T_BGNSTR, stamp))
        f.write(_rec(_T_STRNAME, _name(top)))
        if multi:
            _write_srefs_multi(f, cnames, nx, ny, pitch, places, chunk,
                               interleave=(mode == "multi"))
        elif mode in ("sref", "bigcell"):
            _write_srefs(f, cell, nx, ny, pitch, places, chunk)
        elif mode == "sref_norun":
            _write_srefs_norun(f, cell, nx, ny, pitch, places, chunk,
                               angles=[0.0, 15.0, 30.0, 45.0])
        elif mode == "aref":
            _write_aref(f, cell, nx, ny, pitch)
        elif mode == "flat":
            _write_flat(f, shapes, nx, ny, pitch, places,
                        layer, datatype, chunk)
        elif mode == "mixed":
            _write_flat(f, shapes, nx, ny, pitch, n_flat,
                        layer, datatype, chunk)
            _write_srefs(f, ref_cell, nx, ny, pitch, n_ref, chunk)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        _write_marks(f, span_nm, mark_layer, datatype)
        f.write(_rec(_T_ENDSTR))
        f.write(_rec(_T_ENDLIB))
        size = f.tell()

    # Reported extent includes the unit-cell footprint on the far corner.
    span_mm = (span_nm + _UNIT_CELL_UM * _DBU_PER_UM) / (_DBU_PER_UM * 1000.0)
    if multi:
        # Both orderings hand cell c the same share: n//k placements plus
        # one more for the first n % k cells (the ragged tail).
        k = len(sets)
        share = [places // k + (1 if c < places % k else 0) for c in range(k)]
        base_polys = sum(len(g) for g in sets)
        polys = sum(m * len(g) for m, g in zip(share, sets)) + 4
        verts = sum(m * sum(s.shape[0] - 1 for s in g)
                    for m, g in zip(share, sets)) + 16
        return {
            "path": str(path), "mode": mode, "bytes": int(size),
            "mb": size / (1024 * 1024), "places": int(places),
            "grid": (int(nx), int(ny)), "pitch_um": pitch / _DBU_PER_UM,
            "cells": k, "cell_share": share,
            "base_polys": base_polys, "polys": int(polys), "verts": int(verts),
            "flat_places": 0, "ref_places": int(places),
            "span_mm": span_mm, "area_mm2": span_mm * span_mm,
        }
    return {
        "path": str(path),
        "mode": mode,
        "bytes": int(size),
        "mb": size / (1024 * 1024),
        "places": int(places),
        "grid": (int(nx), int(ny)),
        "pitch_um": pitch / _DBU_PER_UM,
        # In bigcell mode each placement carries n_sub unit patterns.
        "base_polys": len(shapes) * max(1, n_sub),
        "polys": int(places) * len(shapes) * max(1, n_sub) + 4,  # + marks
        "verts": int(places) * verts_per_cell * max(1, n_sub) + 16,
        "flat_places": int(n_flat),
        "ref_places": int(n_ref),
        "span_mm": span_mm,
        "area_mm2": span_mm * span_mm,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("out", help="output .gds path")
    ap.add_argument("--mode", default="sref",
                    choices=["sref", "sref_norun", "aref", "flat", "mixed",
                             "bigcell", "multi", "multi_grouped"])
    ap.add_argument("--cells", type=int, default=3,
                    help="distinct cell definitions for the multi modes")
    ap.add_argument("--mb", type=float, default=220.0,
                    help="target file size in MB (ignored with --places)")
    ap.add_argument("--places", type=float, default=None,
                    help="exact placement count (accepts 5e6)")
    ap.add_argument("--vert-budget", type=float, default=18e6,
                    help="vertex target for the mixed / bigcell modes")
    args = ap.parse_args()

    st = generate_gds(args.out, mode=args.mode, target_mb=args.mb,
                      places=None if args.places is None else int(args.places),
                      vert_budget=int(args.vert_budget), n_cells=args.cells)
    print(f"{st['path']}  [{st['mode']}]")
    print(f"  size        {st['mb']:.1f} MB")
    print(f"  placements  {st['places']:,}  grid {st['grid'][0]}×{st['grid'][1]}"
          f"  pitch {st['pitch_um']:.2f} µm")
    if "cells" in st:
        print(f"  cells       {st['cells']} distinct  "
              f"({st['base_polys']} base polygons)")
    print(f"  polygons    {st['polys']:,}   vertices {st['verts']:,}")
    print(f"  extent      {st['span_mm']:.2f} × {st['span_mm']:.2f} mm "
          f"({st['area_mm2']:.2f} mm²)")


if __name__ == "__main__":
    main()
