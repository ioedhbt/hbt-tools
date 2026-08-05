"""
gdsii/parser.py — single-pass streaming GDSII decoder and in-memory layer
objects (``_PolyLayer``, ``_InstancedLayer``).

Split out of ``tools/process/ebeam/calculator.py`` (mechanical refactor: moved
as-is, no behaviour change, no numbers changed). Self-contained like the
rest of the EBL calculator (see ``tools/process/ebeam/AGENTS.md``): imports nothing
outside ``tools/process/ebeam`` itself.
"""
from __future__ import annotations

import math
import struct

import numpy as np

try:
    import gdstk
except ImportError:  # pragma: no cover
    gdstk = None

from tools.process.ebeam.gdsii.limits import _Limits, _DEFAULT_LIMITS, _rows_budget_msg, tr

# Vertices per pass wherever a flat layer is walked whole — _PolyLayer.
# poly_areas and _flat_coverage_grid. Scaling, binning or cross-multiplying
# a 20 M-vertex layer in one shot would allocate several hundred MB of
# temporaries, on the very layers that are dense precisely because memory is
# tight. At 1 M the working set is ~24 MB and the pass still runs in ~0.1 s.
_FLAT_RASTER_CHUNK = 1 << 20


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
        """|shoelace| area per polygon (source units²), vectorized.

        Walked in vertex-bounded passes: the one-shot form allocates four
        vertex-sized temporaries (the wrap index, both gathers and the
        cross products), so a 13 M-vertex flat layer asked for ~400 MB of
        scratch — on every rerun, since the exposure grid recomputes this.
        Same bounded-pass reasoning as ``_flat_coverage_grid``.
        """
        cx, cy, st = self.cx, self.cy, self.starts
        n = int(st.size - 1)
        if n <= 0:
            return np.zeros(0)
        out = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            # As many whole polygons as fit in one vertex budget, but never
            # fewer than one (a single polygon bigger than the budget still
            # has to be done in one piece).
            j = int(np.searchsorted(st, st[i] + _FLAT_RASTER_CHUNK,
                                    side="right")) - 1
            j = min(max(j, i + 1), n)
            a, b = int(st[i]), int(st[j])
            sub = st[i:j + 1] - a          # this pass's starts, rebased
            x = cx[a:b]
            y = cy[a:b]
            nxt = np.arange(b - a) + 1
            nxt[sub[1:] - 1] = sub[:-1]    # wrap each polygon's last → first
            cross = x * y[nxt] - x[nxt] * y
            np.abs(0.5 * np.add.reduceat(cross, sub[:-1]), out=out[i:j])
            i = j
        return out

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


# ─── Repeating *groups* of elements ─────────────────────────────────────────
#
# ``_element_run`` only ever compares a block against the one directly
# after it, so a file that repeats a *group* of k different elements —
# CELL0, CELL1, CELL2, CELL0, CELL1, CELL2, … — has every run cut to
# length 1 even though its byte stream is perfectly regular. A layout that
# steps several device types together is written exactly like that, and it
# used to cost ~20× a single-cell mask of the same size (300 MB: 24 s to
# parse against 1.1 s, 65 s to stream against 0.7 s).
#
# ``_tile_probe`` finds the period of such a group and verifies, in one
# numpy pass, that the whole T-byte tile repeats with *only* its XY
# payloads changing. The element loop then walks the k template elements
# exactly as before — so every identity (SNAME, rotation, layer, vertex
# count) still comes from the ordinary parse — and each collects its own
# copies with a strided gather, picking slot i out of every repeat in one
# pass. Once the template elements are done the loop skips the whole
# verified region at a stroke.
#
# Verifying the entire tile up front, rather than each slot as it is
# reached, is what makes the per-slot passes safe to do without any byte
# comparison of their own: every byte outside the XY windows is by then
# known to be constant across all the repeats, so a slot cannot silently
# change identity underneath the template the loop parsed.

_TILE_MAX_BYTES = 1 << 16      # widest group period worth searching for
_TILE_MAX_ELEMS = 256          # most elements one group may hold
_TILE_MIN_REPS = 4             # fewer repeats than this and the probe
                               # costs more than the run it would save
_TILE_PROBE_STEP = 4096        # bytes skipped after a failed probe…
_TILE_PROBE_MAX = 1 << 20      # …doubling up to this, so a genuinely
                               # irregular file pays ~1 probe per MB
# Elements a group may contain: the two the loop bulk-decodes from an XY
# run, plus the three it ignores outright. PATH and AREF are excluded —
# both do per-element work that a strided gather would not reproduce.
_TILE_OK_TAGS = frozenset((_T_SREF, _T_BOUNDARY, _T_TEXT, _T_NODE, _T_BOX))


def _tile_windows(mv, start: int, end: int):
    """XY payload windows — ``(offset from start, length)`` — of the
    elements filling ``[start, end)``, or ``None`` if that range is not a
    whole number of group-safe elements each carrying exactly one XY."""
    unpack_rec = _REC.unpack_from
    wins = []
    p = start
    n_el = 0
    in_el = False
    have_xy = False
    while p < end:
        if p + 4 > end:
            return None
        rlen, tag = unpack_rec(mv, p)
        if rlen < 4 or p + rlen > end:
            return None
        if tag in _EL_START:
            if in_el or tag not in _TILE_OK_TAGS:
                return None
            in_el = True
        elif tag == _T_XY:
            if not in_el or have_xy:
                return None
            wins.append((p + 4 - start, rlen - 4))
            have_xy = True
        elif tag == _T_ENDEL:
            if not have_xy:
                return None
            in_el = have_xy = False
            n_el += 1
            if n_el > _TILE_MAX_ELEMS:
                return None
        p += rlen
    return wins if (p == end and not in_el and n_el) else None


def _tile_probe(a, mv, p0: int, blk: int, xy_off: int, n: int):
    """Search for a repeating group of elements beginning at ``p0``.

    Returns ``(period_bytes, repeats)`` — the group is ``[p0, p0+period)``
    and the ``repeats`` copies after it are verified byte-identical to it
    outside their XY payloads — or ``None`` when there is no such group.
    """
    head = bytes(mv[p0:p0 + xy_off])       # element header up to its XY
    if len(head) < 8:                      # too weak a signature to trust
        return None
    lo = p0 + blk
    hi = min(p0 + _TILE_MAX_BYTES, n)
    if lo >= hi:
        return None
    blob = bytes(mv[lo:hi])
    at = 0
    for _ in range(4):
        # The header can also occur *inside* a longer group (a second
        # reference to the same cell), which yields a period too short to
        # verify — so try the next few candidates before giving up.
        pos = blob.find(head, at)
        if pos < 0:
            return None
        at = pos + 1
        period = lo + pos - p0
        if p0 + period * (1 + _TILE_MIN_REPS) > n:
            continue
        wins = _tile_windows(mv, p0, p0 + period)
        if wins is None:
            continue
        keep = np.ones(period, dtype=bool)
        for off, ln in wins:
            keep[off:off + ln] = False
        tfix = a[p0:p0 + period][keep]
        reps = 0
        q = p0 + period
        step = max(1, (1 << 22) // period)
        while q + period <= n:
            k = min(step, (n - q) // period)
            arr = a[q:q + k * period].reshape(k, period)
            ok = (arr[:, keep] == tfix).all(axis=1)
            r = k if bool(ok.all()) else int(np.argmin(ok))
            reps += r
            q += r * period
            if r < k:
                break
        if reps >= _TILE_MIN_REPS:
            return period, reps
    return None


def _element_slot_run(a, p0: int, blk: int, xy_off: int, xy_len: int,
                      period: int, reps: int):
    """Gather one group slot's XY payloads across ``reps`` repeats.

    No byte comparison here, unlike ``_element_run``: ``_tile_probe`` has
    already proved every byte outside the group's XY windows identical in
    all of them.
    """
    # Gathered as a strided view of the XY windows alone, not by reshaping
    # whole ``period``-byte rows: a slot that starts part-way into the
    # group has its *block* run past the end of the verified region on the
    # last repeat even though its XY payload does not, and reshaping would
    # demand those bytes.
    base = p0 + period + xy_off
    reps = min(reps, (a.size - base - xy_len) // period + 1)
    chunks = []
    step = max(1, (1 << 22) // period)
    done = 0
    while done < reps:
        k = min(step, reps - done)
        view = np.lib.stride_tricks.as_strided(
            a[base + done * period:], shape=(k, xy_len),
            strides=(period, 1), writeable=False)
        chunks.append(np.ascontiguousarray(view)
                      .reshape(-1).view(">i4").astype(np.int32))
        done += k
    return chunks


def _decode_copies(a, mv, n: int, p0: int, blk: int, xy_off: int,
                   xy_len: int, tile: list):
    """Every further copy of the element at ``p0``, as XY chunks.

    Tries the plain run first; where there is none, looks for a repeating
    group (subject to the backoff in ``tile``) and gathers this element's
    slot from it. ``tile`` is the caller's mutable
    ``[period, start, repeats, probe_from, backoff]`` — zero period means
    "not inside a group". Returns ``(next_pos, chunks)``.
    """
    if tile[0]:
        return p0 + blk, _element_slot_run(a, p0, blk, xy_off, xy_len,
                                           tile[0], tile[2])
    p_next, chunks = _element_run(a, mv, p0, blk, xy_off, xy_len)
    if chunks or p0 < tile[3]:
        return p_next, chunks
    hit = _tile_probe(a, mv, p0, blk, xy_off, n)
    if hit is None:
        tile[4] = min(_TILE_PROBE_MAX, tile[4] * 2 + _TILE_PROBE_STEP)
        tile[3] = p0 + tile[4]
        return p_next, chunks
    tile[0], tile[1], tile[2] = hit[0], p0, hit[1]
    tile[3] = tile[4] = 0      # a hit clears the backoff: after a group of
                               # this shape, another is likely right behind
    return p_next, _element_slot_run(a, p0, blk, xy_off, xy_len, *hit)


def _tile_advance(p_next: int, tile: list) -> int:
    """Once the group's template elements have all been walked, skip the
    whole verified region — every repeat has already been decoded, slot by
    slot, by ``_element_slot_run``."""
    if tile[0] and p_next >= tile[1] + tile[0]:
        if p_next == tile[1] + tile[0]:
            p_next = tile[1] + (tile[2] + 1) * tile[0]
        # Landing *past* the group's end would mean its elements did not
        # tile it exactly after all — which ``_tile_windows`` ruled out, so
        # leave group mode rather than skip on a premise that no longer
        # holds.
        tile[0] = 0
    return p_next


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


def _parse_gds(buf, limits: "_Limits" = None):
    """Single-pass streaming parse of a GDSII byte buffer.

    Returns ``(unit_meters, {structure_name: raw cell dict})`` (see
    ``_consolidate_cell`` for the dict layout). Geometry comes from
    BOUNDARY, PATH, SREF and AREF elements; TEXT/NODE/BOX are skipped.
    Raises ``ValueError`` with a user-facing message when the file
    exceeds ``limits`` — the budgets derived from this host's free RAM
    (see ``_limits_for``); defaults to the conservative static set.
    """
    if limits is None:
        limits = _DEFAULT_LIMITS
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
    tile = [0, 0, 0, 0, 0]    # repeating-group state, see _decode_copies

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
                    # tools emit for arrayed placements — or, when the
                    # file steps a *group* of different cells, this
                    # element's slot out of every repeat of that group.
                    p_next, chunks = _decode_copies(
                        a, mv, n, el_start, blk, xd0 - el_start, xdl, tile)
                    for ch in chunks:
                        total_rows += ch.size >> 1
                        lst.append(ch.reshape(-1, 2))
                elif el == _T_BOUNDARY and xdl >= 8:
                    # xdl >= 8 (i.e. npts >= 1): a BOUNDARY whose XY record
                    # is empty or truncated carries no polygon, and letting
                    # npts reach 0 divides by zero in the run decode below.
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
                    # polygon dumps), or this shape's slot out of a
                    # repeating group of different shapes.
                    p_next, chunks = _decode_copies(
                        a, mv, n, el_start, blk, xd0 - el_start, xdl, tile)
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
                        n_inst = int(cols) * int(arows)
                        # Budget-check BEFORE building the lattice. An AREF
                        # is ~40 bytes on disk but expands to cols × rows
                        # placements (up to 32767² ≈ 1.07 G, since COLROW
                        # is int16) — allocating first would ask for tens
                        # of GB from a file small enough to pass every size
                        # limit, i.e. an instant OOM kill.
                        if total_rows + n_inst > limits.rows:
                            raise ValueError(
                                _rows_budget_msg(limits.rows))
                        x0, y0, x1, y1, x2, y2 = unpack_6i(mv, xd0)
                        ii = (np.arange(cols, dtype=np.float64)[:, None]
                              / cols)
                        jj = (np.arange(arows, dtype=np.float64)[None, :]
                              / arows)
                        # Fill one (2, K) buffer and hand over its
                        # transpose: separate offx/offy grids plus an
                        # np.stack would peak at 3× the final array.
                        lattice = np.empty((2, n_inst), dtype=np.float64)
                        gx = lattice[0].reshape(cols, arows)
                        gy = lattice[1].reshape(cols, arows)
                        np.add(ii * (x1 - x0), jj * (x2 - x0), out=gx)
                        gx += x0
                        np.add(ii * (y1 - y0), jj * (y2 - y0), out=gy)
                        gy += y0
                        key = (el_sname, el_rot, el_mag, el_refl)
                        lst = cur_refs.get(key)
                        if lst is None:
                            lst = []
                            cur_refs[key] = lst
                        lst.append(lattice.T)
                        total_rows += n_inst
                elif el == _T_PATH:
                    npts = xdl >> 3
                    pts = np.frombuffer(
                        mv, dtype=">i4", count=2 * npts,
                        offset=xd0).astype(np.float64).reshape(-1, 2)
                    cur_paths.append((el_layer, el_dt, el_width, el_ptype,
                                      el_bext, el_eext, pts))
                    total_verts += npts * 4   # rough polygon estimate
                if total_verts > limits.src_verts:
                    raise ValueError(tr(
                        f"GDS holds more than {limits.src_verts:,} "
                        "polygon vertices — more distinct geometry than "
                        "this machine's free memory allows. Expose a "
                        "smaller layer, then re-upload.",
                        f"GDS 檔案含有超過 {limits.src_verts:,} 個多邊形頂點 — "
                        "幾何資料量超出本機可用記憶體的負荷。請匯出較小的圖層後"
                        "重新上傳。"
                    ))
                if total_rows > limits.rows:
                    raise ValueError(_rows_budget_msg(limits.rows))
            el = 0
            el_sname = None
            el_xy = None
            el_colrow = None
            el_rot = 0.0
            el_mag = 1.0
            el_refl = False
            p = _tile_advance(p_next, tile)
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
            tile[0] = 0
        elif tag == _T_ENDSTR:
            if cur_name is not None:
                cells[cur_name] = _consolidate_cell(
                    cur_polys, cur_paths, cur_refs, db_user)
            cur_name = None
            cur_polys = None
            cur_paths = None
            cur_refs = None
            tile[0] = 0        # a group never spans a structure boundary
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

    ``budget`` is the shared mutable counter/limit pair
    ``{"rows": running_total, "limit_rows": ceiling}`` — the ceiling
    comes from ``_limits_for()`` and so tracks this host's free RAM.
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
                if budget["rows"] > budget["limit_rows"]:
                    raise ValueError(tr(
                        f"GDS expands to more than "
                        f"{budget['limit_rows']:,} cell placements — "
                        "more than this machine's free memory allows. "
                        "Expose a smaller layer, then re-upload.",
                        f"GDS 檔案展開後的元件放置數超過 "
                        f"{budget['limit_rows']:,} 個 — 超出本機可用記憶體的"
                        "負荷。請匯出較小的圖層後重新上傳。"
                    ))
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

