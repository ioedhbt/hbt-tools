"""
models/tuning/progressive_geometry.py — Progressive auto-fit (🪜) constants
and pure geometry helpers used by drivers_progressive.py.

Kept in their own module (rather than in the driver) because they are pure
functions with no Streamlit dependency — the grid geometry / bounds-shrink
logic can be unit-tested without a Streamlit session.

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).
"""
from __future__ import annotations
import numpy as np

from .ranges import _clamp_to_hard

# Combo budgets — how many candidate points a single pass may evaluate.
_PROG_GLOBAL_BUDGET_CPU = 100_000
_PROG_GLOBAL_BUDGET_GPU = 2_000_000
_PROG_GROUP_BUDGET_CPU  = 2_000
_PROG_GROUP_BUDGET_GPU  = 50_000
# Per-simulate chunk sizes (rows fed to simulate_batch at once).
_PROG_CHUNK_CPU = 16_384
_PROG_CHUNK_GPU = 262_144
# Bounds geometry per refinement cycle.
_PROG_SHRINK    = 0.35   # new half-span = 0.35 × old span (interior best)
_PROG_EXPAND    = 1.6    # widen a side by 1.6 × span when best sits on an edge
_PROG_EDGE_FRAC = 0.05   # "on an edge" = within 5 % of the span from a bound
_PROG_MAX_CYCLES = 1000   # cycles are memo-cheap; 200 cut off real-file runs
                          # (KY HEMT benchmark still improving ~0.1%/cycle)
_PROG_MIN_STEP   = 0.01  # default step floor when a spec has no explicit step
# Escape phase — fires when refinement stalls above the per-port goal: pinned
# params get one wide log re-scan (alone + with a group partner) and their
# boxes re-widened, so a bad local minimum can't trap the whole fit.
_PROG_PORT_GOAL    = 5.0   # per-port residual goal (%) — escape runs while any port is above
_PROG_STALL_CYCLES = 4     # consecutive low-improvement cycles before an escape fires
_PROG_STALL_REL    = 2e-3  # "low improvement" = relative best-Total drop below this
_PROG_ESCAPE_PTS   = 48    # points on a wide escape axis

# Canonical refinement groups — pair-wise (Zbe / Zbc / …) refinement mirrors the
# physical coupling between each pole's R and C.  Fit keys map onto a group when
# their canonical key is in that group's tuple.
_PROG_GROUPS = [
    ("Zbe",       ("Rbe", "Cbe")),
    ("Zbc",       ("Rbc", "Cbc")),
    ("extrinsic", ("Cbex", "Cbcx", "Rbcx")),
    ("transport", ("alpha0", "tauB", "tauC")),
    ("gm",        ("Gm0", "tau")),
    ("access",    ("Rpb", "Rpc", "Rpe")),
    ("base",      ("Rbi",)),
    # Kun-Yang HEMT — per-branch RC pairs, delay network, substrate pads,
    # plus lead-L / pad-C groups for full-scope fits.
    ("Ygs",       ("Cgs", "Ri")),
    ("Ygd",       ("Cgd", "Rgd")),
    ("Yds",       ("Cds", "Rds")),
    ("ky-delay",  ("R_delay", "C_delay")),
    ("ky-pad-gs", ("Cgsp", "Rsub1")),
    ("ky-pad-ds", ("Cdsp", "Rsub2")),
    ("ky-pad-gd", ("Cgdp",)),
    ("leads",     ("Lb", "Lc", "Le")),
    ("pads",      ("Cpbe", "Cpce", "Cpbc")),
]


def _prog_axis_values(lo, hi, n_pts, floor, include=None) -> np.ndarray:
    """Value list for one sweep axis over [lo, hi] in display units.

    • Log spacing when ``lo > 0 and hi / lo >= 50`` (many-decade span), else
      linear.
    • Spacing floor: never place points closer than ``floor`` — drop excess
      points so consecutive samples are ≥ floor apart.
    • Optionally insert ``include`` (e.g. the current best) into the list.
    • Always ≥ 1 point; clipped into [lo, hi].
    """
    lo = float(lo); hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    n_pts = max(1, int(n_pts))
    span = hi - lo
    if span <= 0:
        vals = np.array([lo], dtype=np.float64)
    else:
        # Cap the point count so spacing never drops below the floor.
        if floor is not None and floor > 0:
            max_by_floor = int(span / float(floor)) + 1
            n_pts = max(1, min(n_pts, max_by_floor))
        if n_pts <= 1:
            vals = np.array([0.5 * (lo + hi)], dtype=np.float64)
        elif lo > 0 and hi / lo >= 50.0:
            vals = np.geomspace(lo, hi, n_pts)
        else:
            vals = np.linspace(lo, hi, n_pts)
    if include is not None and np.isfinite(include):
        vals = np.append(vals, float(np.clip(include, lo, hi)))
    vals = np.clip(vals, lo, hi)
    # Dedup values closer than floor/2 apart (keeps the inserted value).
    vals = np.unique(np.round(vals, 12))
    if floor is not None and floor > 0 and vals.size > 1:
        keep = [vals[0]]
        for v in vals[1:]:
            if v - keep[-1] >= float(floor) * 0.5:
                keep.append(v)
        vals = np.asarray(keep, dtype=np.float64)
    if vals.size == 0:
        vals = np.array([lo], dtype=np.float64)
    return vals


def _prog_signature(row_vals, floors) -> tuple:
    """Quantised signature of one candidate row for the memo set.  Two rows
    whose every coordinate lands in the same ``floor/2`` bucket collide (never
    re-evaluated); values a full floor apart land in distinct buckets."""
    sig = []
    for v, fl in zip(row_vals, floors):
        f = float(fl) if fl and fl > 0 else _PROG_MIN_STEP
        sig.append(int(round(float(v) / (f * 0.5))))
    return tuple(sig)


def _prog_next_bounds(lo, hi, best, hard_lo, hard_hi, floor) -> tuple:
    """New (lo, hi) for one axis after a refinement cycle.

    • Best sits within ``_PROG_EDGE_FRAC × span`` of a bound → expand THAT side
      by ``_PROG_EXPAND × span`` (search ran into the wall — widen it).
      When the hard-limit clamp makes that expansion a no-op (best pinned
      against a hard limit — nowhere left to widen), fall through to the
      shrink rule instead: otherwise the axis would never shrink and stay
      coarsely sampled forever.
    • Otherwise shrink to ``best ± _PROG_SHRINK × span``.
    • Result clipped into hard limits; span kept ≥ 2 × floor so the box never
      collapses below the step floor.
    """
    lo = float(lo); hi = float(hi); best = float(best)
    span = hi - lo
    fl = float(floor) if floor and floor > 0 else _PROG_MIN_STEP
    if span <= 0:
        span = fl
    edge = _PROG_EDGE_FRAC * span
    expanded = False
    if best - lo <= edge:
        new_lo = lo - _PROG_EXPAND * span
        new_hi = hi
        expanded = True
    elif hi - best <= edge:
        new_lo = lo
        new_hi = hi + _PROG_EXPAND * span
        expanded = True
    else:
        half = _PROG_SHRINK * span
        new_lo = best - half
        new_hi = best + half
    new_lo, new_hi = _clamp_to_hard(new_lo, new_hi, hard_lo, hard_hi)
    if (expanded and abs(new_lo - lo) < 1e-15 and abs(new_hi - hi) < 1e-15):
        # Expansion fully clipped away — best is pinned against a hard limit.
        # Shrink instead so the axis still refines.
        half = _PROG_SHRINK * span
        new_lo, new_hi = _clamp_to_hard(best - half, best + half,
                                        hard_lo, hard_hi)
    # Keep the box at least 2×floor wide, centred on best where possible.
    if new_hi - new_lo < 2.0 * fl:
        c = float(np.clip(best, new_lo, new_hi))
        new_lo = c - fl
        new_hi = c + fl
        new_lo, new_hi = _clamp_to_hard(new_lo, new_hi, hard_lo, hard_hi)
    return new_lo, new_hi
