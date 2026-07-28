"""
models/tuning/ranges.py — Pad parameter specs, physics-informed sweep ranges,
and the hard physical limits every tuning driver clamps into.

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).
"""
from __future__ import annotations
import re
import numpy as np


# ── Pad parameter specs (shared across all models) ────────────────────────────
# (key, display_label, SI_scale, unit_string, format_string, step)

PAD_SPECS = [
    ("Cpbe","Cpbe", 1e15,"fF","%.4f",0.1),
    ("Cpce","Cpce", 1e15,"fF","%.4f",0.1),
    ("Cpbc","Cpbc", 1e15,"fF","%.4f",0.01),
    ("Lb",  "Lb",   1e12,"pH","%.3f",0.1),
    ("Lc",  "Lc",   1e12,"pH","%.3f",0.1),
    ("Le",  "Le",   1e12,"pH","%.3f",0.01),
    ("Rpb", "Rb",   1.0, "Ω", "%.4f",0.01),
    ("Rpc", "Rc",   1.0, "Ω", "%.4f",0.01),
    ("Rpe", "Re",   1.0, "Ω", "%.4f",0.01),
]
_PAD_KEYS = [k for k, *_ in PAD_SPECS]


# ── Physics-informed sweep ranges + hard physical limits ──────────────────────
#
# Everything here is in DISPLAY units (i.e. SI × spec-scale — fF / pH / Ω / …),
# so the tables can be consulted directly against the number_input widgets and
# the sweep rows without re-scaling.  The reference anchors are three published
# InP-HBT extraction columns: this work / Xu 2014 / Cheng 2022.

# Hard physical limits per parameter key: key → (lo, hi).  Every parameter not
# listed gets (0.0, None) — negative component values are never physical.
# alpha0 is additionally boxed to the physically plausible HBT range.
TUNE_HARD_LIMITS: dict = {"alpha0": (0.95, 0.99)}

# Physics-informed default sweep ranges (DISPLAY units), (lo, hi) per key.
TUNE_DEFAULT_RANGES: dict = {
    # access resistances (Ω)
    "Rpb": (2.0, 60.0), "Rpe": (5.0, 50.0), "Rpc": (0.5, 30.0),
    # extrinsic network
    "Cbex": (5.0, 200.0), "Cbcx": (1.0, 50.0),      # fF
    "Rbcx": (50.0, 500.0),                          # kΩ
    # intrinsic
    "Rbi": (2.0, 50.0), "Rbe": (1.0, 100.0),        # Ω
    "Cbe": (20.0, 3000.0), "Cbc": (1.0, 50.0),      # fF
    "Rbc": (20.0, 300.0),                           # kΩ
    "alpha0": (0.95, 0.99),
    "tauB": (0.05, 1.0), "tauC": (0.05, 1.0),       # ps
    # π-topology (τ is shared with the Kun-Yang HEMT — keep it wide)
    "Gm0": (10.0, 2000.0),                          # mS
    "tau": (0.05, 5.0),                             # ps
    # Kun-Yang HEMT — intrinsic RC branches, delay network, substrate pads.
    # (Pads reuse the canonical Rpb/Rpc/Rpe + Lb/Lc/Le keys, relabelled
    # Rg/Rd/Rs + Lg/Ld/Ls in the KY specs.)
    "Cgs": (100.0, 2000.0), "Cgd": (2.0, 150.0), "Cds": (20.0, 500.0),   # fF
    "Ri": (0.2, 20.0), "Rgd": (20.0, 5000.0), "Rds": (30.0, 2000.0),     # Ω
    "R_delay": (0.5, 100.0), "C_delay": (10.0, 50000.0),                 # Ω / fF
    "Cgsp": (5.0, 100.0), "Cdsp": (5.0, 100.0), "Cgdp": (0.5, 50.0),     # fF
    "Rsub1": (0.5, 100.0), "Rsub2": (0.5, 100.0),                        # Ω
    # parasitics (not in the default fit scope, but seeds still want them)
    "Cpbe": (0.0, 20.0), "Cpce": (0.0, 20.0), "Cpbc": (0.0, 20.0),  # fF
    "Lb": (0.0, 150.0), "Lc": (0.0, 150.0), "Le": (0.0, 150.0),     # pH
}

# Low-performance variants (fmax < fT, or fT below ~40 GHz): such devices sit
# at much higher Cbc/Cbcx/Rbi and slower τ — cf. the "this work" column of the
# reference table.
TUNE_LOW_PERF_RANGES: dict = {
    "Cbcx": (20.0, 300.0), "Cbc": (50.0, 400.0),
    "Rbi": (50.0, 2500.0),
    "tauB": (0.1, 8.0), "tauC": (0.05, 3.0), "tau": (0.5, 10.0),
}

_PARASITIC_KEYS = frozenset({"Cpbe", "Cpce", "Cpbc", "Lb", "Lc", "Le"})

# Union of every key that appears in any of the range tables — used to match a
# canonical key against.  Ordered largest-first so multi-letter keys (alpha0)
# win before their prefix tokens ever could.
_TUNE_TABLE_KEYS = tuple(sorted(
    set(TUNE_DEFAULT_RANGES) | set(TUNE_LOW_PERF_RANGES)
    | set(TUNE_HARD_LIMITS) | _PARASITIC_KEYS,
    key=len, reverse=True))

# Greek / typeset label aliases → canonical key (labels carry these glyphs).
_TUNE_LABEL_ALIASES = {"α₀": "alpha0", "τB": "tauB", "τC": "tauC", "τ": "tau"}

# Secondary token aliases, only consulted when no table key matched a token —
# custom-model access resistors are named "Rb"/"Rc"/"Re" (core.py access
# specs) while the tables key them as Rpb/Rpc/Rpe.  Kept separate from the
# primary pass so an intrinsic label like "Rbe … aka rE" still resolves to
# Rbe (its own token matches first) and never to the access Rpe.
_TUNE_TOKEN_ALIASES = {"rb": "Rpb", "rc": "Rpc", "re": "Rpe"}


def _canonical_tune_key(key, label="") -> str | None:
    """Map a spec key OR a custom-model component name to the canonical table
    key.  Returns None when nothing plausibly matches.

    Matching, in order:
      1. Exact key match against the union of table keys.
      2. Greek/typeset label aliases (α₀ → alpha0, τB → tauB, …).
      3. Case-insensitive standalone-token match: split both the key and the
         label into alpha[+digits] tokens and compare (lower-cased) against
         the (lower-cased) table keys.  So a custom component named "Rbe" or a
         Cheng label "Rbe (from Cheng's T) aka rE (from Xu)" both resolve to
         "Rbe".
    """
    key_s = str(key or "")
    label_s = str(label or "")
    # 1 — exact key.
    if key_s in _TUNE_TABLE_KEYS:
        return key_s
    # 2 — Greek / typeset label alias (also honour the aliases on the key).
    for src in (label_s, key_s):
        if src in _TUNE_LABEL_ALIASES:
            return _TUNE_LABEL_ALIASES[src]
    # 3 — standalone-token match.  Longest table keys first so alpha0 wins
    #     over a bare "alpha"/"a" prefix.
    tokens = set()
    for src in (key_s, label_s):
        tokens.update(t.lower() for t in re.findall(r"[A-Za-z]+[0-9]*", src))
    for tk in _TUNE_TABLE_KEYS:
        if tk.lower() in tokens:
            return tk
    # 4 — secondary token aliases (custom access names Rb/Rc/Re → Rpb/…).
    for tok, tk in _TUNE_TOKEN_ALIASES.items():
        if tok in tokens:
            return tk
    return None


def tune_hard_limits(key, label="") -> tuple:
    """Return the hard (lo, hi) physical limits for a parameter, in display
    units.  hi may be None (no upper bound).  Defaults to (0.0, None) — no
    negative component values — for anything not explicitly boxed."""
    if key in TUNE_HARD_LIMITS:
        return TUNE_HARD_LIMITS[key]
    canon = _canonical_tune_key(key, label)
    if canon is not None and canon in TUNE_HARD_LIMITS:
        return TUNE_HARD_LIMITS[canon]
    return (0.0, None)


def _clamp_to_hard(lo, hi, hard_lo, hard_hi):
    """Clamp a (lo, hi) display-unit interval into hard limits (hi may be
    None → unbounded above).  Hard limits win over everything else."""
    if hard_lo is not None:
        lo = max(lo, hard_lo)
        hi = max(hi, hard_lo)
    if hard_hi is not None:
        lo = min(lo, hard_hi)
        hi = min(hi, hard_hi)
    if hi < lo:
        hi = lo
    return lo, hi


def _range_step(span, spec_step=None):
    """Default sweep step for a span: ``span/20`` rounded to 2 significant
    digits, floored at ``spec_step`` (the fine per-param step) when given.
    Zero span → zero step."""
    if span <= 0:
        return 0.0
    step = float(f"{span / 20.0:.2g}")
    if step <= 0:
        step = span / 20.0
    if spec_step is not None and spec_step > 0:
        step = max(step, float(spec_step))
    return step


def informed_default_range(key, label, current_disp, *,
                           low_perf=False, spec_step=None) -> tuple:
    """Physics-informed default (min, step, max) for one parameter, in display
    units.

    Behaviour:
      • Canonicalise the key/label.  On a match, take (lo, hi) from
        ``TUNE_LOW_PERF_RANGES`` when ``low_perf`` and the canonical key is
        listed there, else ``TUNE_DEFAULT_RANGES``.
          – Zero-current parasitic rule: a parasitic key (pad C / lead L) whose
            current value is ~0 was deliberately zeroed (e.g. a pre-de-embedded
            file), so it stays at (0, 0, 0).
          – Otherwise widen (lo, hi) to include the current finite nonzero
            value so the seed sits inside the box.
      • No canonical match (custom / HEMT params): decade box around the
        current value (current > 0 → current/10 … current×10); current == 0 →
        (0, 0, 0) so unknown zero params stay put.
      • Clamp the result into the hard limits (hard limits win over
        widen-to-include-current).
    """
    cur = float(current_disp)
    cur_finite = np.isfinite(cur)
    hard_lo, hard_hi = tune_hard_limits(key, label)
    canon = _canonical_tune_key(key, label)

    rng = None
    if canon is not None:
        # Deliberately-zeroed parasitics stay zero.
        if canon in _PARASITIC_KEYS and abs(cur) < 1e-30:
            return (0.0, 0.0, 0.0)
        if low_perf and canon in TUNE_LOW_PERF_RANGES:
            rng = TUNE_LOW_PERF_RANGES[canon]
        else:
            rng = TUNE_DEFAULT_RANGES.get(canon)

    if rng is not None:
        lo, hi = float(rng[0]), float(rng[1])
        # Widen to include the current value (finite + nonzero only).
        if cur_finite and abs(cur) > 1e-30:
            lo = min(lo, cur)
            hi = max(hi, cur)
    else:
        # No table entry (unknown custom / HEMT param, or a key boxed only in
        # TUNE_HARD_LIMITS): decade box around the current value.
        if cur_finite and cur > 0:
            lo, hi = cur / 10.0, cur * 10.0
        else:
            return (0.0, 0.0, 0.0)

    lo, hi = _clamp_to_hard(lo, hi, hard_lo, hard_hi)
    step = _range_step(hi - lo, spec_step)
    return (lo, step, hi)


def _detect_low_perf_device(S_raw, freq) -> bool:
    """Heuristic: is this a low-performance device (slow / lossy)?

    Computes fT/fmax from |h21|² and Mason's U; falls back to the
    20 dB/dec extrapolated 0-dB crossing when there's no in-band crossing.
    Returns True iff (both fT and fmax are known and fmax < fT) or (fT is known
    and below ~40 GHz).  Any failure → False (assume high-perf default ranges).
    """
    try:
        from ...helpers.metrics import (compute_h21_U, find_ft_fmax,
                                         extrap_20dbdec)
        h21_db, U_db = compute_h21_U(S_raw)
        f_ghz = np.asarray(freq, dtype=float) / 1e9
        fT, fmax = find_ft_fmax(f_ghz, h21_db, U_db)

        def _extrap_zero(gain_db):
            try:
                _fe, _ge, f_zero = extrap_20dbdec(f_ghz, gain_db)
                if f_zero is None or not np.isfinite(f_zero):
                    return None
                return float(f_zero)
            except Exception:
                return None

        if fT is None:
            fT = _extrap_zero(h21_db)
        if fmax is None:
            fmax = _extrap_zero(U_db)

        if (fT is not None and fmax is not None
                and np.isfinite(fT) and np.isfinite(fmax) and fmax < fT):
            return True
        if fT is not None and np.isfinite(fT) and fT < 40.0:
            return True
        return False
    except Exception:
        return False
