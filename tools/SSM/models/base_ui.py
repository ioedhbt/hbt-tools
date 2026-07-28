"""
models/base_ui.py — Shared Streamlit UI helpers used by all model classes.

Kept separate from the abstract base so model files import one thing,
not a chain of ssm_* modules.
"""
from __future__ import annotations
import gc
import re
import time
import numpy as np
import threading
import streamlit as st
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from itertools import product as iterproduct
from io import BytesIO

from ..helpers import (extended_smith_grid, params_hash, s_to_y,
                        build_Y_pad_batch, build_Z_ser_batch,
                        plotly_with_dl,
                        quickset_buttons, apply_pending, segmented_radio)
from ..helpers.mem_budget import ram_available_bytes
from ...i18n import is_zh, tr

# Streamlit's "rerun current script" exception — raised when any st.* call
# happens after the user has clicked a widget that triggers a re-run (e.g.
# the Stop button below).  We catch it so the long-running tuning loop can
# break out cleanly without losing the persisted top-K state.
#
# We also catch `StopException`, which Streamlit raises through a *different*
# code path (e.g. `st.stop()` or some Stop-button paths in newer versions).
# Both indicate "the user wants the current run to end now"; we treat them
# identically — persist state, free GPU buffers, and re-raise so Streamlit
# can finish the rerun cleanly.
try:
    from streamlit.runtime.scriptrunner.script_runner import RerunException as _RerunException
except Exception:
    try:
        from streamlit.runtime.scriptrunner import RerunException as _RerunException
    except Exception:
        _RerunException = None  # type: ignore[assignment]
try:
    from streamlit.runtime.scriptrunner.script_runner import StopException as _StopException
except Exception:
    try:
        from streamlit.runtime.scriptrunner_utils.exceptions import StopException as _StopException
    except Exception:
        _StopException = None  # type: ignore[assignment]

# ── CUDA detection (runtime, zero-cost when CuPy is absent) ─────────────────

try:
    import cupy as _cp
    _HAS_CUDA = True
    _v = _cp.cuda.runtime.runtimeGetVersion()      # e.g. 13000
    _CUDA_VER = f"{_v // 1000}.{(_v % 1000) // 10}"
except Exception:
    _cp = None          # type: ignore[assignment]
    _HAS_CUDA = False
    _CUDA_VER = ""


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
        from ..helpers.metrics import (compute_h21_U, find_ft_fmax,
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


# ── Residual ──────────────────────────────────────────────────────────────────

def ssm_residual(S_mea: np.ndarray, S_mod: np.ndarray) -> float:
    """RMS relative S-parameter residual across all four ports (%)."""
    total = 0.0
    for r in range(2):
        for c in range(2):
            sm = S_mea[:,r,c]; sk = S_mod[:,r,c]
            denom = np.sum(np.abs(sm)**2)
            if denom > 0:
                total += np.sqrt(np.sum(np.abs(sm-sk)**2) / denom)
    return total / 4.0 * 100.0


# ── Per-port residuals ────────────────────────────────────────────────────────

def _port_residuals(S_mea, S_mod):
    """Return dict with Total, S11, S12, S21, S22 residuals in %."""
    res = {}
    total = 0.0
    for name, (r, c) in [("S11",(0,0)),("S12",(0,1)),("S21",(1,0)),("S22",(1,1))]:
        sm = S_mea[:, r, c]; sk = S_mod[:, r, c]
        denom = np.sum(np.abs(sm)**2)
        if denom > 0:
            val = np.sqrt(np.sum(np.abs(sm - sk)**2) / denom) * 100.0
        else:
            val = 0.0
        res[name] = val
        total += val
    res["Total"] = total / 4.0
    return res


def _port_residuals_batch(S_mea, S_mod_batch, xp):
    """Batched residuals — *fused* across all 4 ports for minimum kernel launches.

    S_mea       : (N, 2, 2)              — measured (already on device)
    S_mod_batch : (B, N, 2, 2)           — model
    Returns dict whose values are (B,) arrays on the *xp* device.

    Old impl ran ~5 ops per port × 4 ports + summation = ~22 kernel launches.
    This impl does ~7 launches total — significant Python/launch overhead saved
    in the GPU hot loop.
    """
    diff   = S_mea[None, :, :, :] - S_mod_batch        # (B, N, 2, 2)
    num    = xp.sum(xp.abs(diff)  ** 2, axis=1)        # (B, 2, 2)
    den    = xp.sum(xp.abs(S_mea) ** 2, axis=0)        # (2, 2)
    den_s  = xp.where(den > 0, den, 1.0)               # avoid div-by-0
    val    = xp.sqrt(num / den_s[None, :, :]) * 100.0  # (B, 2, 2)
    val    = xp.where(den[None, :, :] > 0, val, 0.0)

    s11 = val[:, 0, 0]
    s12 = val[:, 0, 1]
    s21 = val[:, 1, 0]
    s22 = val[:, 1, 1]
    total = (s11 + s12 + s21 + s22) * 0.25
    return {"S11": s11, "S12": s12, "S21": s21, "S22": s22, "Total": total}


# ── Smith chart ───────────────────────────────────────────────────────────────

_SMITH_COLORS = {"S11":"#1f77b4","S22":"#ff7f0e","S21":"#2ca02c","S12":"#d62728"}

def render_smith_chart(S_mea, S_sim, model_name, error_pct, scales=None, key="smith",
                       show_title=True, meas_label="Meas.", sim_label="Model",
                       *, compact: bool = False, height: int | None = None,
                       extra_download: tuple | None = None,
                       inline_labels: bool = False):
    """Render a Plotly Smith chart with measured (markers) + modeled (dashed).

    Parameters
    ----------
    compact : bool, default False
        When True, the layout is tuned for narrow containers (e.g. the
        side-by-side Smith+Bode in the Visual Tuning preview):
          • legend repositioned to the BOTTOM (instead of right side).
          • height shrunk to 380 unless ``height`` overrides it.
          • margins tightened.
        Default False preserves the original full-width layout.
    height : int, optional
        Explicit pixel height; overrides the compact / default values.
    """
    if scales is None:
        scales = {"S11":1.0,"S12":1.0,"S21":1.0,"S22":1.0}
    fig = go.Figure()
    for _grid_tr in extended_smith_grid(1.0):
        fig.add_trace(_grid_tr)
    inline_annos = []     # populated when inline_labels=True (Sxx near trace)
    for name, (r, c) in [("S11",(0,0)),("S22",(1,1)),("S21",(1,0)),("S12",(0,1))]:
        col = _SMITH_COLORS[name]; sc = scales.get(name, 1.0)
        sm = S_mea[:,r,c]*sc; sk = S_sim[:,r,c]*sc
        sc_lbl = "" if abs(sc-1.0)<1e-9 else (f" ×{sc:.2g}" if sc>=1 else f" ÷{1/sc:.2g}")
        fig.add_trace(go.Scattergl(x=sm.real, y=sm.imag, mode="markers",
                                  name=f"{name}{sc_lbl} {meas_label}",
                                  marker=dict(color=col, size=5, symbol="circle"),
                                  showlegend=not inline_labels,
                                  hovertemplate=f"{name} {meas_label}<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))
        fig.add_trace(go.Scattergl(x=sk.real, y=sk.imag, mode="lines",
                                  name=f"{name}{sc_lbl} {sim_label}",
                                  line=dict(color=col, width=2.0, dash="dash"),
                                  showlegend=not inline_labels,
                                  hovertemplate=f"{name} {sim_label}<br>Re=%{{x:.4f}}<br>Im=%{{y:.4f}}<extra></extra>"))
        if inline_labels:
            # Label each S-param near its (model) trace centroid, nudged
            # radially outward — mirrors the matplotlib Smith chart's on-trace
            # Sxx labels and frees the space a legend box would eat.
            with np.errstate(invalid="ignore"):
                cx = float(np.nanmean(sk.real)); cy = float(np.nanmean(sk.imag))
            if np.isfinite(cx) and np.isfinite(cy):
                rr = (cx*cx + cy*cy) ** 0.5
                if rr > 1e-6:
                    f = (rr + 0.13) / rr
                    cx *= f; cy *= f
                inline_annos.append(dict(x=cx, y=cy, xref="x", yref="y",
                                         text=f"{name}{sc_lbl}", showarrow=False,
                                         font=dict(size=13, color=col)))
    title_cfg = (dict(text=f"Smith Chart - {model_name}", font=dict(size=12))
                 if show_title else None)
    if compact:
        # The bottom-legend layout needs ~140 px of bottom margin to
        # keep the X-axis title "Re(Γ)" clear of the legend chips,
        # plus another ~30 px below for the "● Meas. — — Model"
        # annotation.  At height=540 the plot area is still ≈ 390 px
        # which keeps the unit-circle squareness visually reasonable.
        legend_cfg = dict(orientation="h", x=0.5, y=-0.22,
                          xanchor="center", yanchor="top",
                          font=dict(size=9))
        margin_cfg = dict(l=40, r=15, t=40 if show_title else 10, b=140)
        eff_height = height if height is not None else 540
        annotation_y = -0.42  # below the legend chips
    else:
        legend_cfg = dict(x=1.02, y=1.0, xanchor="left")
        margin_cfg = dict(l=50, r=30, t=70, b=50)
        eff_height = height if height is not None else 560
        annotation_y = -0.08
    if inline_labels:
        # Inline-label mode: no legend box; Sxx sit on the traces, plus one
        # small meas/model hint pinned bottom-left inside the plot.
        annotations = list(inline_annos)
        annotations.append(dict(
            x=-1.06, y=-1.06, xref="x", yref="y", xanchor="left", yanchor="bottom",
            showarrow=False,
            text=f"● {meas_label}   - - {sim_label}",
            font=dict(size=9, color="gray"), align="left"))
        if compact:
            margin_cfg = dict(l=40, r=15, t=40 if show_title else 10, b=40)
    else:
        annotations = [dict(x=0.5, y=annotation_y, xref="paper", yref="paper",
                            showarrow=False,
                            text=f"● {meas_label} (markers)  |  - - {sim_label} (dashed)",
                            font=dict(size=10, color="gray"), align="center")]
    fig.update_layout(
        title=title_cfg,
        xaxis=dict(title="Re(Γ)", range=[-1.1,1.1], scaleanchor="y", scaleratio=1,
                   showgrid=False, zeroline=False),
        yaxis=dict(title="Im(Γ)", range=[-1.1,1.1], showgrid=False, zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white", height=eff_height,
        margin=margin_cfg,
        legend=legend_cfg,
        showlegend=not inline_labels,
        hovermode="closest",
        annotations=annotations)
    plotly_with_dl(fig, key=key, filename=key, extra_download=extra_download)


def render_smith_with_ftfmax(S_raw, S_sim, freq, model_name: str,
                             model_short: str, fname: str, scales=None,
                             *, s2p_bytes: bytes | None = None,
                             s2p_filename: str | None = None):
    """
    Two-column layout: Smith chart (left) + fT/fmax mini-card (right).

    A sticky st.metric row (Total + per-port S11/S12/S21/S22 residuals, keyed
    container `hbt_resrow_<model_short>` pinned via ui_theme.py CSS) renders
    above the Smith chart in place of the old title-line residual sentence;
    the mini-card on the right shows |h21|² and Mason U for both measured and
    modeled with 20 dB/dec extrapolation when needed.  Use this in place of
    the bare `render_smith_chart()` call inside each model's
    `render_override_and_smith`.

    When ``s2p_bytes`` is supplied, the Smith chart's download row gains a
    second button (📥 S2P) right next to the standard xlsx — this replaced
    the standalone "Download Modeled DUT S2P" section that used to live at
    the bottom of the SSM extraction tab.
    """
    # Local import — ssm_plots imports back from base_ui at module load time,
    # so a top-level import here would create a circular dependency.
    from ..ssm_plots import render_ft_fmax_card

    port_res = _port_residuals(S_raw, S_sim)
    err = float(port_res["Total"])
    # Sticky residual strip — keyed container so ui_theme.py can pin it
    # (position:sticky) while the user scrolls the tuning expanders.  The key
    # deliberately excludes fname (dots/commas would break the st-key-* CSS
    # class); model_short is unique per rendered page.
    _prev_key = f"hbt_res_prev_{model_short}_{fname}"
    _prev = st.session_state.get(_prev_key)
    with st.container(key=f"hbt_resrow_{model_short}"):
        mc = st.columns(5)
        for i, pname in enumerate(("Total", "S11", "S12", "S21", "S22")):
            cur = float(port_res[pname])
            delta = None
            if isinstance(_prev, dict) and pname in _prev:
                d = cur - float(_prev[pname])
                if abs(d) >= 0.005:
                    delta = f"{d:+.2f}%"
            mc[i].metric(
                tr("Total residual", "總殘差") if pname == "Total" else pname,
                f"{cur:.2f}%", delta=delta, delta_color="inverse",
                help=tr("RMS relative S-parameter error across all four ports —"
                        " the number to minimise.  Deltas compare against the"
                        " previous simulation of this model.",
                        "四個埠上的均方根相對 S 參數誤差 — 需最小化的數值。"
                        "差值與此模型上一次模擬結果比較。")
                     if pname == "Total" else None)
    st.session_state[_prev_key] = {k: float(v) for k, v in port_res.items()}
    extra_dl = None
    if s2p_bytes is not None and s2p_filename is not None:
        extra_dl = (tr("📥 modeled S2P", "📥 模型 S2P"),
                    s2p_bytes, s2p_filename, "text/plain")
    col_l, col_r = st.columns([1.05, 1])
    with col_l:
        render_smith_chart(S_raw, S_sim, model_name, err, scales,
                           key=f"smith_{model_short}_{fname}",
                           extra_download=extra_dl)
    with col_r:
        render_ft_fmax_card(S_raw, S_sim, freq,
                            model_name=model_name,
                            key=f"ftfmax_card_{model_short}_{fname}",
                            height=560)


def smith_scale_controls(fname, topo_key) -> dict:
    st.markdown(
        f"<b>{tr('S display scale', 'S 顯示縮放')}</b>"
        f" <span class='hbt-help' title='{tr(
            'Multiply each trace before plotting.'
            ' Display only - does not affect the residual.',
            '繪圖前乘上此係數，僅影響顯示，不影響殘差。')}'>?</span>",
        unsafe_allow_html=True)
    sc = {}
    for col_w, name, default in zip(st.columns(4),
                                    ["S11","S12","S21","S22"],
                                    [1.0,   1.0,   1.0,   1.0]):
        sk = f"smith_scale_{topo_key}_{name}_{fname}"
        if sk not in st.session_state:
            st.session_state[sk] = default
        # NOTE: do NOT pass `value=` alongside `key=` when the key is
        # already initialized in session_state — Streamlit's
        # check_session_state_rules logs a warning ("created via
        # st.session_state.X and value parameter").  The widget reads
        # the current value from session_state via the key alone.
        sc[name] = col_w.number_input(f"{name} ×",
                                       min_value=0.01, max_value=1000.0,
                                       step=0.5, format="%.2f", key=sk)
    return sc


# ── Pad sync (call once per model's render_override_and_smith) ────────────────

def sync_pad_from_preov(fname: str, topo_key: str, para_eff: dict):
    """
    Copy current pre-extraction pad values into the per-topology session state
    only when the pre-extraction values have changed (detected via MD5 hash).
    Prevents stale pad values in the Smith chart override form.
    """
    preov_hash  = params_hash({k: para_eff.get(k, 0.0) for k in _PAD_KEYS})
    sync_key    = f"smith_pad_synced_{topo_key}_{fname}"
    # Also reseed when a widget key was GC'd by a page switch — Streamlit
    # drops session_state for widgets that skip a run, while this sync hash
    # (a plain value, not a widget) survives and would otherwise block the
    # reseed, leaving the pad inputs to recreate at 0.
    _keys_missing = any(f"sim_{topo_key}_{key}_{fname}" not in st.session_state
                        for key, *_ in PAD_SPECS)
    if _keys_missing or st.session_state.get(sync_key) != preov_hash:
        for key, _, scale, *_ in PAD_SPECS:
            st.session_state[f"sim_{topo_key}_{key}_{fname}"] = float(para_eff.get(key, 0.0)) * scale
        st.session_state[sync_key] = preov_hash

def _render_cbex_sweep_tool(*, cbex_arr, freq, f_ghz, f_min_v, f_max_v,
                            cbex_scale, cbex_unit, cbex_param_key,
                            model_short, fname, g_idx, rng_tag,
                            cbex_sweep_fn, param_groups):
    """Render the Min / Step / Max inputs + Calculate button for the Cbex
    sweep.  On button click: generate candidate Cbex values, call
    `cbex_sweep_fn(cbex_SI_array, mask)` to get per-candidate std(Cbcx_arr),
    pick the argmin, and write the best value into the Cbex number_input's
    session-state key so the widget below renders the new value on the next
    natural rerun.

    The std window (mask) comes from the Cbcx group's slider session state
    if the user has already moved it, otherwise the full frequency range.
    """
    # Defaults: min = min|Cbex_arr| (fF), max = max|Cbex_arr| (fF), step = 10 fF
    fin = cbex_arr[np.isfinite(cbex_arr)]
    if len(fin) == 0:
        st.caption(tr("Cbex sweep unavailable — no finite Cbex samples.",
                      "Cbex 掃描無法使用 — 沒有有限的 Cbex 樣本。"))
        return
    abs_disp = np.abs(fin) * cbex_scale
    default_min = float(np.min(abs_disp))
    default_max = float(np.max(abs_disp))
    if default_max <= default_min:
        default_max = default_min + 1.0
    default_step = 0.1

    # Per-group state keys — tied to rng_tag so defaults refresh on slider move
    sweep_key_base = f"cbex_sweep_{model_short}_{fname}_{rng_tag}"
    k_min  = f"{sweep_key_base}_min"
    k_step = f"{sweep_key_base}_step"
    k_max  = f"{sweep_key_base}_max"
    k_res  = f"cbex_sweep_result_{model_short}_{fname}"  # persists across reruns

    st.markdown("---")
    st.markdown(f"**🔍 {tr('Cbex sweep — minimise std(Cbcx)', 'Cbex 掃描 — 最小化 std(Cbcx)')}**")

    c_min, c_step, c_max = st.columns(3)
    sweep_min = c_min.number_input(
        f"{tr('Min', '最小值')} ({cbex_unit})",
        value=float(st.session_state.get(k_min, default_min)),
        min_value=0.0, format="%.4f", key=k_min)
    sweep_step = c_step.number_input(
        f"{tr('Step', '步進')} ({cbex_unit})",
        value=float(st.session_state.get(k_step, default_step)),
        min_value=1e-6, format="%.4f", key=k_step)
    sweep_max = c_max.number_input(
        f"{tr('Max', '最大值')} ({cbex_unit})",
        value=float(st.session_state.get(k_max, default_max)),
        min_value=0.0, format="%.4f", key=k_max)
    run_sweep = st.button(
        tr("Calculate", "計算"),
        key=f"{sweep_key_base}_btn",
        width="stretch")

    if run_sweep:
        if sweep_max < sweep_min or sweep_step <= 0:
            st.error(tr("Sweep range is invalid: need max ≥ min and step > 0.",
                        "掃描範圍無效：需 max ≥ min 且 step > 0。"))
        else:
            # Build candidate array in display units, then convert to SI
            n_pts = int(np.floor((sweep_max - sweep_min) / sweep_step)) + 1
            n_pts = max(2, min(n_pts, 10000))  # sanity cap
            cand_disp = sweep_min + np.arange(n_pts) * sweep_step
            cand_disp = cand_disp[cand_disp <= sweep_max + 1e-12]
            cand_SI = cand_disp / float(cbex_scale)

            # Mask: use the Cbcx group's slider range if the user has set it,
            # otherwise the full freq range.  Find the Cbcx group by searching
            # param_groups for one whose params reference "Cbcx_arr" (was
            # `g_idx + 1` but Ccex now sits between Cbex and Cbcx for ChengT).
            cbcx_g_idx = next(
                (i for i, g in enumerate(param_groups)
                 if any(spec[0] == "Cbcx_arr" for spec in g.get("params", []))),
                g_idx + 1,
            )
            cbcx_sl_key = f"pfp_sl_{model_short}_{cbcx_g_idx}_{fname}"
            cbcx_range = st.session_state.get(cbcx_sl_key, (f_min_v, f_max_v))
            try:
                f_lo_cbcx, f_hi_cbcx = float(cbcx_range[0]), float(cbcx_range[1])
            except Exception:
                f_lo_cbcx, f_hi_cbcx = f_min_v, f_max_v
            mask = (f_ghz >= f_lo_cbcx) & (f_ghz <= f_hi_cbcx)
            if not mask.any():
                mask = np.ones_like(f_ghz, dtype=bool)

            try:
                stds = cbex_sweep_fn(cand_SI, mask)
            except Exception as exc:
                st.error(tr(f"Sweep failed: {exc!r}", f"掃描失敗：{exc!r}"))
                return
            stds = np.asarray(stds, dtype=float)
            if not np.any(np.isfinite(stds)):
                st.error(tr("All candidates produced non-finite std(Cbcx) — "
                            "try a different range.",
                            "所有候選值皆產生非有限的 std(Cbcx) — 請嘗試其他範圍。"))
                return
            best_k = int(np.nanargmin(stds))
            best_cbex_SI = float(cand_SI[best_k])
            best_cbex_disp = best_cbex_SI * float(cbex_scale)
            best_std = float(stds[best_k])

            # Stage the result in a "pending" key.  We CAN'T write directly
            # to the number_input's session-state key here because that
            # widget has already been instantiated earlier in this same run
            # (it lives in cols[0] of the same row).  On the next rerun the
            # plot loop will see this pending value and copy it into the
            # widget key BEFORE the widget is created — see the
            # `pfp_pending_*` lookup just above the number_input render.
            pending_key = (f"pfp_pending_{model_short}_{cbex_param_key}_"
                           f"{fname}")
            st.session_state[pending_key] = float(best_cbex_disp)

            # Persist the message so it survives the rerun
            st.session_state[k_res] = {
                "best_disp": best_cbex_disp,
                "best_std":  best_std,
                "n_pts":     len(cand_SI),
                "f_lo":      f_lo_cbcx,
                "f_hi":      f_hi_cbcx,
                "rng_tag":   rng_tag,
            }
            st.rerun()

    # Show the last sweep result (if any) — scoped to this rng_tag so it
    # clears when the Cbex slider is moved.
    last = st.session_state.get(k_res)
    if last and last.get("rng_tag") == rng_tag:
        st.success(tr(
            f"Best Cbex = **{last['best_disp']:.4f} {cbex_unit}**  "
            f"(std(Cbcx) = {last['best_std']:.3e}, "
            f"{last['n_pts']} candidates, "
            f"Cbcx window {last['f_lo']:.2f}–{last['f_hi']:.2f} GHz)",
            f"最佳 Cbex = **{last['best_disp']:.4f} {cbex_unit}**  "
            f"（std(Cbcx) = {last['best_std']:.3e}，"
            f"{last['n_pts']} 個候選值，"
            f"Cbcx 視窗 {last['f_lo']:.2f}–{last['f_hi']:.2f} GHz）"))


def _render_tau_total_fit_section(*, all_data, fname, model_short,
                                  params, para_eff):
    """
    Multi-file 1/(2πfT) vs 1/IC linear fit (T-model reference only).

    For each bias file in ``all_data``, computes τ_total = 1/(2π f_T) from
    the de-embedded |h21|² 0-dB crossing.  Plots τ_total (ps) versus 1/IC
    (1/mA), linear-fits, and reports:
      - Cje (from slope):  slope = (η kT/q) · CJE  →  CJE = slope / (η · Vt).
        (Cbc contribution to slope is neglected per Cheng et al., paper Eq. 1.)
      - τB + τC (from intercept): intercept − (RC + REE) · CBC.
      - Per-file τCC = (rE + REE + RC) · CBC and τE = rE · CJE,
        with rE = η kT / (q IC).

    References:
      - Equation (1) of K. Y. D. Cheng et al., "Hot electron injection
        effect on the microwave performance of type-I/II AlInP/GaAsSb/InP
        DHBTs" — supplies the total-delay formula.  Paper notation:
        REE/RC/rE → code: Rpe/Rpc/(ηkT/qIC).
      - H. G. Liu, N. Tao, S. P. Watkins, C. R. Bolognesi, "Extraction
        of the Average Collector Velocity in High-Speed Type-II
        InP–GaAsSb–InP DHBTs," IEEE EDL 25(12), 2004 — supplies the
        v_c = W_C/(2 τ_C) split with default v_c = 4×10⁷ cm/s (peak
        across a 2000 Å InP collector).

    Hidden when ``all_data`` has < 2 files (the fit needs ≥ 2 bias points).
    The extracted Cje is for reference only — it does NOT feed back into
    the model's own extracted parameters.
    """
    import pandas as pd
    from pathlib import Path
    from ..helpers import peel_parasitics, compute_metrics, extract_limit

    if not all_data or len(all_data) < 2:
        return False

    with st.expander(tr("📐 Cje / τB+τC / τCC / τE from 1/(2πfT) vs 1/IC fit  "
                        "(T-model reference)",
                        "📐 由 1/(2πfT) 對 1/IC 擬合萃取 Cje / τB+τC / τCC / τE"
                        "（T 模型參考）"),
                     expanded=False):
        st.caption(
            tr("Reference extraction (extracted values do NOT feed back into "
               "the model fit). Liu, Tao, Watkins, Bolognesi, IEEE EDL 25(12), 2004 'Extraction of the average collector velocity in high-speed Type-II InP-GaAsSb-InP_DHBTs.pdf'",
               "僅供參考的萃取（萃取值不會回饋至模型擬合）。Liu, Tao, Watkins, "
               "Bolognesi, IEEE EDL 25(12), 2004 'Extraction of the average "
               "collector velocity in high-speed Type-II InP-GaAsSb-InP_DHBTs.pdf'"))
        st.latex(
            r"\frac{1}{2\pi f_T}=\tau_B+\tau_C+\frac{\eta k T}{q I_C}\,C_{JE}"
            r"+\left(R_C+R_{EE}+\frac{\eta k T}{q I_C}\right)C_{BC}")
        st.caption(
            tr("Notation: REE → emitter access resistance (Rpe here); "
               "RC → collector access (Rpc here); rE = ηkT/(qIC) → intrinsic "
               "base-emitter resistance.",
               "符號說明：REE → 射極存取電阻（此處為 Rpe）；"
               "RC → 集極存取電阻（此處為 Rpc）；"
               "rE = ηkT/(qIC) → 本徵射基電阻。"))

        # ── fT per file (Open+Short de-embedded, access R RETAINED) ─────────
        # The Cheng formula's RC/REE refer to the access resistance that the
        # device sees at the fT-measurement plane.  If we used the same
        # `para_eff` that the model extraction uses, ``peel_parasitics`` would
        # also strip Rpb/Rpc/Rpe (when sourced from Z-param / open-collector
        # / Cold-HBT) — that puts the fT plane past the access R and the
        # (RC+REE)·CBC intercept correction over-subtracts.
        #
        # Smart behavior: zero out Rpb/Rpc/Rpe before peeling.  When the user
        # has NOT entered any pad caps / lead L in the previous section
        # (because the files are already pre-de-embedded), every C and L in
        # ``para_eff`` is zero — and peel_parasitics becomes an algebraic
        # no-op (Y_pad=0, Z_ser=0 → returns Y_dut unchanged).  Otherwise it
        # peels only the caps and leads, exactly as requested.
        _para_pad_lead_only = dict(para_eff)
        _para_pad_lead_only["Rpb"] = 0.0
        _para_pad_lead_only["Rpc"] = 0.0
        _para_pad_lead_only["Rpe"] = 0.0

        recs = []
        for fn, d in all_data.items():
            try:
                Y     = peel_parasitics(d["S_raw"], d["freq"], d["z0"],
                                        _para_pad_lead_only)
                dfm   = compute_metrics(Y, d["freq"])
                f_ghz = dfm["Freq (GHz)"].to_numpy()
                fT_v, _, _ = extract_limit(
                    f_ghz, dfm["|h21|² (dB)"].to_numpy(),
                    dfm["fT Plateau (GHz)"].to_numpy(),
                    n_pts=2,
                    f_min=float(f_ghz[0]), f_max=float(f_ghz[-1]))
                fT_GHz   = float(fT_v) if np.isfinite(fT_v) else np.nan
                tau_tot  = (1.0 / (2.0 * np.pi * fT_GHz * 1e9)
                            if (np.isfinite(fT_GHz) and fT_GHz > 0) else np.nan)
                recs.append({"fn": fn, "stem": Path(fn).stem,
                             "fT_GHz": fT_GHz, "tau_s": tau_tot})
            except Exception as ex:
                recs.append({"fn": fn, "stem": Path(fn).stem,
                             "fT_GHz": np.nan, "tau_s": np.nan,
                             "err": str(ex)})

        # Sort by fT descending (matches Z-param method's "sort by extracted-
        # quantity desc" convention — higher fT files appear first).
        recs.sort(key=lambda r: (r["fT_GHz"] if np.isfinite(r["fT_GHz"])
                                  else -np.inf),
                  reverse=True)

        # st.markdown(
        #     "**Files (fT measured after Open+Short pad/lead de-embedding "
        #     "— access R RETAINED so RC/REE in the formula remain meaningful; "
        #     "no-op when the file is already pre-de-embedded):**")
        hcols = st.columns([0.3, 2.0, 1.2, 1.4, 1.4])
        for h, t in zip(hcols, ["", tr("File", "檔案"), "fT (GHz)",
                                  "τ_total (ps)", "IC (mA)"]):
            h.markdown(f"<small><b>{t}</b></small>", unsafe_allow_html=True)

        # Per-file checkbox + IC input.  Seed IC from rz12_Ie_{fn} (Z-param's
        # IE input) since IE ≈ IC in normal HBT operation; the user can refine.
        points = []   # list of (1/IC[1/mA], τ_total[ps], stem, IC_mA)
        for r in recs:
            c0, c1, c2, c3, c4 = st.columns([0.3, 2.0, 1.2, 1.4, 1.4])
            use_key = f"taut_use_{r['fn']}__{fname}__{model_short}"
            if use_key not in st.session_state:
                st.session_state[use_key] = True
            use = c0.checkbox(f"{tr('Use', '使用')} {r['stem']}",
                              key=use_key + "_w",
                              value=st.session_state[use_key],
                              label_visibility="collapsed")
            st.session_state[use_key] = use

            c1.markdown(f"<small>{r['stem']}</small>", unsafe_allow_html=True)
            c2.markdown(
                (f"<small>{r['fT_GHz']:.3f}</small>"
                 if np.isfinite(r["fT_GHz"]) else "<small>—</small>"),
                unsafe_allow_html=True)
            c3.markdown(
                (f"<small>{r['tau_s']*1e12:.4f}</small>"
                 if np.isfinite(r["tau_s"]) else "<small>—</small>"),
                unsafe_allow_html=True)

            ic_key = f"taut_Ic_{r['fn']}__{fname}__{model_short}"
            if ic_key not in st.session_state:
                # Default to Z-param Ie (≈ Ic in normal mode), else 0.
                st.session_state[ic_key] = float(
                    st.session_state.get(f"rz12_Ie_{r['fn']}", 0.0))
            ic_mA = c4.number_input(
                f"IC {tr('for', '對象')} {r['stem']}",
                min_value=0.0, step=0.1, format="%.3f",
                value=float(st.session_state[ic_key]),
                key=ic_key + "_w",
                label_visibility="collapsed")
            st.session_state[ic_key] = ic_mA

            if use and ic_mA > 0 and np.isfinite(r["tau_s"]):
                points.append((1.0 / ic_mA,
                                r["tau_s"] * 1e12,
                                r["stem"], ic_mA))

        if len(points) < 2:
            st.info(tr("Enter IC for at least two enabled files to fit.",
                       "請至少為兩個已啟用的檔案輸入 IC 才能進行擬合。"))
            return

        x = np.array([p[0] for p in points])     # 1/IC (1/mA)
        y = np.array([p[1] for p in points])     # τ_total (ps)
        lbls = [p[2] for p in points]
        try:
            slope, intercept = np.polyfit(x, y, 1)    # slope: ps·mA, int: ps
        except Exception as ex:
            st.error(tr(f"Linear fit failed: {ex}", f"線性擬合失敗：{ex}"))
            return

        # ── Plot ──────────────────────────────────────────────────────────────
        x_fit = np.linspace(0.0, float(x.max() * 1.08), 200)
        y_fit = slope * x_fit + intercept
        fig = go.Figure()
        # Trace names become Excel sheet names in the xlsx download, so they
        # must avoid characters Excel forbids in sheet titles: / \ ? * [ ]
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="markers+text", text=lbls,
            textposition="top center", name="τ_total",
            marker=dict(size=11, color="#1f77b4",
                        line=dict(color="#0d4a7a", width=1.5))))
        fig.add_trace(go.Scattergl(
            x=x_fit, y=y_fit, mode="lines",
            name=f"{tr('Fit', '擬合')}  slope={slope:.4g} ps·mA   int={intercept:.4g} ps",
            line=dict(color="#d62728", width=2, dash="dash")))
        fig.add_trace(go.Scattergl(
            x=[0.0], y=[intercept], mode="markers",
            name=f"{tr('Intercept', '截距')} = {intercept:.4f} ps",
            marker=dict(size=14, symbol="star", color="#d62728")))
        fig.update_layout(
            title=f"1/(2π f_T) vs 1/I_C  —  {model_short} {tr('model (reference)', '模型（參考）')}",
            xaxis=dict(title="1/I_C (1/mA)", rangemode="tozero",
                       showgrid=True, gridcolor="#ebebeb"),
            yaxis=dict(title="τ_total = 1/(2π f_T) (ps)",
                       showgrid=True, gridcolor="#ebebeb"),
            plot_bgcolor="white", paper_bgcolor="white", height=380,
            legend=dict(x=0.45, y=0.05,
                        xanchor="left", yanchor="bottom",
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#ccc", borderwidth=1,
                        font=dict(size=10)),
            margin=dict(l=55, r=20, t=50, b=50))
        plotly_with_dl(fig,
                       key=f"taut_fit_{model_short}_{fname}",
                       filename=f"taut_fit_{model_short}_{fname}")

        # ── Inputs (defaults from current file's extraction) ─────────────────
        _inputs_hdr = tr("Inputs (defaults from this file's extraction):",
                         "輸入值（預設取自此檔案的萃取結果）：")
        st.markdown(f"**{_inputs_hdr}**")
        Re_def  = float(para_eff.get("Rpe", 0.0))
        Rc_def  = float(para_eff.get("Rpc", 0.0))
        Cbc_def = (float(params.get("Cbc",  0.0))
                   + float(params.get("Cbcx", 0.0)))   # total = intrinsic + extrinsic

        ci1, ci2, ci3, ci4, ci5 = st.columns(5)
        Re_val = ci1.number_input(
            "Re — REE (Ω)", min_value=0.0, value=Re_def, format="%.4f",
            key=f"taut_Re_{model_short}_{fname}",
            help=tr("Emitter access resistance.  Default = Rpe used in extraction.",
                    "射極存取電阻。預設值 = 萃取所用的 Rpe。"))
        Rc_val = ci2.number_input(
            "Rc (Ω)", min_value=0.0, value=Rc_def, format="%.4f",
            key=f"taut_Rc_{model_short}_{fname}",
            help=tr("Collector access resistance.  Default = Rpc used in extraction.",
                    "集極存取電阻。預設值 = 萃取所用的 Rpc。"))
        Cbc_val_fF = ci3.number_input(
            f"Cbc {tr('total', '總計')} (fF)", min_value=0.0,
            value=Cbc_def * 1e15, format="%.4f",
            key=f"taut_Cbc_{model_short}_{fname}",
            help=tr("Total base-collector cap.  Default = Cbc + Cbcx (intrinsic + extrinsic).",
                    "總基極-集極電容。預設值 = Cbc + Cbcx（本徵 + 外徵）。"))
        eta_val = ci4.number_input(
            f"η（{tr('ideality', '理想因子')}）", min_value=0.5, max_value=3.0,
            value=1.0, step=0.05, format="%.3f",
            key=f"taut_eta_{model_short}_{fname}",
            help=tr("Ideality factor for r_E = η kT/(q IC).  Set this from a "
                    "Gummel-plot fit of your device (typical InP HBT: 1.0–1.2).",
                    "r_E = η kT/(q IC) 的理想因子。請由元件的 Gummel 圖擬合設定"
                    "（典型 InP HBT：1.0–1.2）。"))
        T_K = ci5.number_input(
            "T (K)", min_value=1.0, value=300.0, step=5.0, format="%.1f",
            key=f"taut_T_{model_short}_{fname}",
            help=tr("Temperature for kT/q.", "kT/q 計算所用的溫度。"))

        Cbc_val = Cbc_val_fF * 1e-15
        Vt = 1.380649e-23 * T_K / 1.602176634e-19         # kT/q  (V)

        # ── Derived (slope → Cje; intercept → τB+τC) ─────────────────────────
        # Units conversion: slope is in ps·mA = (s·1e-12)·(A·1e-3) = s·A · 1e-15.
        # In SI, slope_SI = (η · Vt) · Cje  with  [V · F] = [s · A].
        # ⇒ Cje[F] = slope_SI / (η · Vt) = slope[ps·mA] · 1e-15 / (η · Vt).
        # ⇒ Cje[fF] = slope[ps·mA] / (η · Vt[V]).
        Cje_fF = (slope / (eta_val * Vt)) if (eta_val * Vt) > 0 else 0.0
        Cje_F  = Cje_fF * 1e-15

        tau_BC_ps = intercept - (Re_val + Rc_val) * Cbc_val * 1e12

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(tr("Slope", "斜率"), f"{slope:.4g} ps·mA",
                  help=tr("d(τ_total)/d(1/I_C) — drives Cje.",
                          "d(τ_total)/d(1/I_C) — 決定 Cje。"))
        m2.metric(tr("Intercept", "截距"), f"{intercept:.4f} ps",
                  help=tr("τ_total extrapolated to 1/I_C → 0.",
                          "τ_total 外插至 1/I_C → 0 的值。"))
        m3.metric(f"Cje  ({tr('ref.', '參考')})", f"{Cje_fF:.4f} fF",
                  help=tr("Cje = slope / (η · kT/q).  Reference only.",
                          "Cje = slope / (η · kT/q)。僅供參考。"))
        m4.metric(f"τB + τC  ({tr('ref.', '參考')})", f"{tau_BC_ps:.4f} ps",
                  help=tr("τB+τC = intercept − (Rc + Re) · Cbc.",
                          "τB+τC = intercept − (Rc + Re) · Cbc。"))

        # ── Split τB / τC using assumed collector velocity v_c ───────────────
        # Liu, Tao, Watkins, Bolognesi, IEEE EDL 25(12), 2004 — "Extraction
        # of the Average Collector Velocity in High-Speed Type-II
        # InP–GaAsSb–InP DHBTs" — found v_c peaks at 4×10⁷ cm/s across a
        # 2000 Å InP collector at V_CB ≈ 0.4 V.  Using the same definition
        # v_c = W_C / (2 τ_C):
        #     τ_C = W_C / (2 v_c)
        #     τ_B = (τ_B + τ_C)_intercept − τ_C
        # Default v_c is the Liu peak for InP collectors; user can edit for
        # other collector materials / thicknesses / biases.  Publishes to
        # session state so the τB / τC number_inputs farther down offer a
        # "v_c = …" quickset button.
        st.markdown(tr(
            "**Split τB / τC using assumed average collector velocity "
            "(Liu et al. 2004 — default for InP collector):**",
            "**依假設之平均集極速度拆分 τB / τC**"
            "**（Liu et al. 2004 — InP 集極預設值）：**"))
        cv1, cv2, cv3, cv4 = st.columns(4)
        Wc_nm = cv1.number_input(
            "W_C (nm)", min_value=1.0, value=120.0, step=10.0, format="%.2f",
            key=f"taut_Wc_{model_short}_{fname}",
            help=tr("Collector depletion width.", "集極空乏區寬度。"))
        v_c_cms = cv2.number_input(
            "v_c (cm/s)", min_value=1.0e5, value=4.0e7,
            step=1.0e6, format="%.3e",
            key=f"taut_vc_{model_short}_{fname}",
            help=tr("Average collector velocity.  Default 4×10⁷ cm/s — peak "
                    "value extracted for 2000 Å InP collectors in Liu, Tao, "
                    "Watkins, Bolognesi, IEEE EDL 25(12), 2004.  Adjust for "
                    "other collector materials / thicknesses / biases.",
                    "平均集極速度。預設 4×10⁷ cm/s — 取自 Liu, Tao, Watkins, "
                    "Bolognesi, IEEE EDL 25(12), 2004 對 2000 Å InP 集極萃取"
                    "之峰值。可依不同集極材料 / 厚度 / 偏壓調整。"))
        v_c_ms     = v_c_cms * 1e-2                      # cm/s → m/s
        Wc_m       = Wc_nm * 1e-9
        tauC_vc_s  = Wc_m / (2.0 * v_c_ms)               # seconds
        tauC_vc_ps = tauC_vc_s * 1e12
        tauB_vc_ps = tau_BC_ps - tauC_vc_ps              # ps
        tauB_vc_s  = tauB_vc_ps * 1e-12

        cv3.metric(f"τC  ({tr('from v_c', '由 v_c 計算')})", f"{tauC_vc_ps:.4f} ps",
                   help=tr("τ_C = W_C / (2 v_c).", "τ_C = W_C / (2 v_c)。"))
        cv4.metric(f"τB  ({tr('from v_c', '由 v_c 計算')})", f"{tauB_vc_ps:.4f} ps",
                   help=tr("τ_B = (τ_B+τ_C) − τ_C.", "τ_B = (τ_B+τ_C) − τ_C。"))

        # Publish v_c-derived values (SI seconds) so τB / τC number_inputs
        # can read them via the "v_c = …" quickset button.  Stored only when
        # finite; deleted otherwise so the button auto-hides when the fit
        # degrades.
        for pub_key, val in (
            (f"taut_pub_tauB_{model_short}_{fname}", tauB_vc_s),
            (f"taut_pub_tauC_{model_short}_{fname}", tauC_vc_s),
        ):
            if np.isfinite(val) and abs(val) > 0:
                st.session_state[pub_key] = float(val)
            else:
                st.session_state.pop(pub_key, None)

        # ── Per-file τCC and τE ─────────────────────────────────────────────
        rows = []
        for _, tau_total_ps, stem, ic_mA in points:
            ic_A   = ic_mA * 1e-3
            rE_i   = (eta_val * Vt) / ic_A if ic_A > 0 else np.nan
            tau_cc = (rE_i + Re_val + Rc_val) * Cbc_val * 1e12  # ps
            tau_E  = rE_i * Cje_F * 1e12                         # ps
            rows.append({
                "File":              stem,
                "IC (mA)":           f"{ic_mA:.4f}",
                "rE = ηkT/(qIC) (Ω)": f"{rE_i:.4f}",
                "τCC = (rE+Re+Rc)·Cbc (ps)": f"{tau_cc:.4f}",
                "τE = rE·Cje (ps)":  f"{tau_E:.4f}",
                "τ_total measured (ps)": f"{tau_total_ps:.4f}",
            })
        st.markdown(f"**{tr('Per-file derived delays (using inputs above):', '各檔案推導延遲（依上述輸入值）：')}**")
        st.dataframe(pd.DataFrame(rows),
                     width="stretch", hide_index=True)
    return True


# Component grouping for the diagram-mode fine-tune editor (the "✏️ Fine-tune"
# override expander).  Keys are matched against each model's spec list; any spec
# key not named here lands in a trailing "Other" group, so nothing is hidden.
_FINETUNE_DIAGRAM_GROUPS_EN = [
    ("Pad parasitics",    ["Cpbe", "Cpbc", "Cpce", "Cgsp", "Cdsp", "Cgdp",
                           "Rsub1", "Rsub2"]),
    ("Lead inductance",   ["Lb", "Lc", "Le"]),
    ("Access resistance", ["Rpb", "Rpc", "Rpe"]),
    ("Extrinsic C",       ["Cbex", "Cbcx", "Rbcx"]),
    ("Delay",             ["tauB", "tauC", "tau", "R_delay", "C_delay"]),
    ("Intrinsic",         ["Rbi", "Rbe", "Cbe", "Cbc", "Rbc", "alpha0", "Gm0"]),
]
# label → zh translation — looked up at render time (via _finetune_diagram_groups())
# so a language switch mid-session relabels the groups immediately, instead of
# baking in whatever language was active when the module was first imported.
_FINETUNE_DIAGRAM_GROUP_ZH = {
    "Pad parasitics":    "焊墊寄生",
    "Lead inductance":   "引線電感",
    "Access resistance": "存取電阻",
    "Extrinsic C":        "外徵電容",
    "Delay":              "延遲",
    "Intrinsic":           "本徵",
}


def _finetune_diagram_groups():
    """Localized ``(label, keys)`` pairs — call fresh each render (see module
    docstring above ``_FINETUNE_DIAGRAM_GROUP_ZH``); never cache at import
    time since :func:`tr` depends on the current session's language."""
    return [(tr(lbl, _FINETUNE_DIAGRAM_GROUP_ZH[lbl]), keys)
            for lbl, keys in _FINETUNE_DIAGRAM_GROUPS_EN]


def render_finetune_diagram(*, all_specs, state_key_for, active_state,
                            calc_vals, render_illustration):
    """Diagram-mode alternative for a model's "✏️ Fine-tune" override expander.

    Two columns:
      • Left  — the model topology schematic, with the component the user last
        edited ringed in red (via ``render_illustration(preview, highlight)``).
      • Right — every override value, grouped (parasitics / lead L / access R /
        extrinsic C / delay / intrinsic).  Each ``number_input`` writes the SAME
        session-state key the list view uses, so the two modes stay in lock-step
        and the caller's downstream parameter assembly is unchanged.  Editing a
        field also moves the highlight to that component.

    Parameters
    ----------
    all_specs        : list of ``(key, label, scale, unit, fmt, step)``.
    state_key_for    : ``f(param_key) -> session_state key`` — lets each caller
                       map a param to its own widget-key scheme (SSM extraction
                       uses ``sim_{topo}_{key}_{fname}``; the RF simulator uses
                       ``rfsim_{prefix}_{pad|ext|int}_{key}``).
    active_state     : session_state key holding the highlighted param.
    calc_vals        : SI defaults used to seed an unset widget (``{}`` ⇒ 0).
    render_illustration : ``f(preview_all_p_SI, highlight_key) -> None``.
    """
    _diagram_groups = _finetune_diagram_groups()
    spec_lookup = {s[0]: s for s in all_specs}
    ordered: list[str] = []
    for _lbl, keys in _diagram_groups:
        ordered += [k for k in keys if k in spec_lookup]
    groups = [(lbl, [k for k in keys if k in spec_lookup])
              for lbl, keys in _diagram_groups]
    other = [s[0] for s in all_specs if s[0] not in ordered]
    if other:
        groups.append((tr("Other", "其他"), other))

    col_diag, col_inp = st.columns([1.1, 1], gap="medium")

    with col_inp:
        st.caption(tr("Edit any value — the diagram highlights the component you "
                      "last changed.",
                      "編輯任一數值 — 示意圖會標示你最後修改的元件。"))
        for g_label, keys in groups:
            if not keys:
                continue
            st.markdown(f"**{g_label}**")
            for i in range(0, len(keys), 2):
                cols = st.columns(2)
                for col_w, key in zip(cols, keys[i:i + 2]):
                    _k, lbl, sc, unit, fmt, step = spec_lookup[key]
                    skey = state_key_for(key)
                    if skey not in st.session_state:
                        st.session_state[skey] = float(calc_vals.get(key, 0.0)) * sc

                    def _mk(_key=key):
                        def _cb():
                            st.session_state[active_state] = _key
                        return _cb

                    col_w.number_input(
                        f"{lbl} ({unit})" if unit else lbl,
                        key=skey, format=fmt, step=step, on_change=_mk())

    with col_diag:
        active = st.session_state.get(active_state)
        preview = {
            s[0]: float(st.session_state.get(
                state_key_for(s[0]),
                float(calc_vals.get(s[0], 0.0)) * s[2])) / s[2]
            for s in all_specs}
        render_illustration(preview, active)
        if active:
            st.caption(f"{tr('Editing', '編輯中')} **{active}**")


def render_interactive_param_groups(params, arrays, freq, fname, model_short, param_groups,
                                    cold_res=None, cold_param_map=None, reextract_fn=None,
                                    cbex_sweep_fn=None,
                                    all_data=None, para_eff=None):

    """
    For each parameter group, render:
      - A labelled section heading with dependency info
      - A "same range as previous" button (for dependent groups)
      - A frequency range slider
      - Per-frequency line plots with a dashed line at the current median
      - A number_input per parameter for manual override
      - (Groups flagged `cbex_sweep_group` only) a Cbex sweep tool that
        searches for the Cbex value that minimises std(Cbcx_arr) across
        the Cbcx-group's selected frequency window.
      - (Groups flagged `tau_total_fit_group` only) the multi-file
        1/(2πfT) vs 1/IC reference fit — auto-hidden when only one s2p
        file is loaded.  Needs ``all_data`` and ``para_eff`` to be passed
        through; otherwise the group is silently skipped.
    Returns a copy of params with all overrides applied (SI units).
    """
    
    f_ghz   = freq * 1e-9
    f_min_v = float(f_ghz[0])
    f_max_v = float(f_ghz[-1])
    step_v  = max(round((f_max_v - f_min_v) / 100, 3), 0.001)

    params_out = dict(params)
    live_arrays = dict(arrays)   # updated mid-loop after re-extraction
    live_params = dict(params)   # updated mid-loop; used for change detection
    prev_range = (f_min_v, f_max_v)

    with st.expander(tr("📊 Interactive Parameter Extraction", "📊 互動式參數萃取"), expanded=False):
        # Thick outline on the large per-group boxes so each parameter group
        # reads as a bold card, set apart from the thin per-parameter
        # sub-containers inside it.  Streamlit puts the user `key` class on the
        # inner `stVerticalBlock` element (NOT the border wrapper), so we draw
        # the border directly on that keyed element and disable Streamlit's own
        # `border=True` wrapper (below) to avoid a double frame.  Scoped to the
        # `st-key-pfp_groupbox_*` prefix, so no other container is affected.
        st.markdown(
            """<style>
            div[class*="st-key-pfp_groupbox_"] {
                border: 3px solid #9aa0a6 !important;
                border-radius: 0.5rem !important;
                padding: 0.75rem !important;
            }
            </style>""",
            unsafe_allow_html=True)
        for g_idx, group in enumerate(param_groups):
            g_label  = group["label"]
            g_params = group["params"]
            g_deps   = group.get("depends_on", [])

            # ── tau_total_fit_group: multi-file 1/(2πfT) vs 1/IC reference ──
            # Rendered as its own nested expander, so no leading heading.
            # Silently skips when called from a single-file context or
            # when the caller didn't pass all_data/para_eff.
            if group.get("tau_total_fit_group"):
                _tau_rendered = False
                if all_data is not None and para_eff is not None:
                    _tau_rendered = bool(_render_tau_total_fit_section(
                        all_data=all_data, fname=fname,
                        model_short=model_short,
                        params=live_params, para_eff=para_eff))
                # Only emit the trailing separator when the section actually
                # rendered.  With a single s2p file the fit is hidden, so
                # skipping the rule avoids a double "---" (the previous group
                # already drew one) showing as two empty lines.
                if _tau_rendered and g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # Heading: special fit groups (Z-plots / Fbi / F1) print it here;
            # normal parameter groups print it inside their bordered box below.
            _is_special_group = any(group.get(_k) for _k in
                                    ("z_plots_group", "fbi_fit_group",
                                     "f1_fit_group"))
            if _is_special_group:
                st.markdown(f"**{g_label}**")

            # ── z_plots_group: Z1, Z3, Z4 Re/Im plots ───────────────────────────
            if group.get("z_plots_group"):
                for formula_type, formula_content in group.get("formulas", []):
                    if formula_type == "latex":
                        st.latex(formula_content)
                    else:
                        st.markdown(formula_content)

                sl_key = f"pfp_sl_{model_short}_{g_idx}_{fname}"
                if sl_key not in st.session_state:
                    st.session_state[sl_key] = (f_min_v, f_max_v)
                f_lo, f_hi = st.slider(
                    tr("Frequency range (GHz)", "頻率範圍 (GHz)"),
                    min_value=f_min_v, max_value=f_max_v,
                    value=st.session_state[sl_key],
                    step=step_v, format="%.2f", key=sl_key)
                mask   = (f_ghz >= f_lo) & (f_ghz <= f_hi)
                f_plot = f_ghz[mask]

                for zlabel, zarr in [("Z1", live_arrays.get("Z1")),
                                      ("Z3", live_arrays.get("Z3")),
                                      ("Z4", live_arrays.get("Z4"))]:
                    if zarr is None:
                        continue
                    c1, c2 = st.columns(2)
                    for col_w, part_fn, part_lbl in [(c1, np.real, "Re"),
                                                      (c2, np.imag, "Im")]:
                        fig = go.Figure()
                        fig.add_trace(go.Scattergl(
                            x=f_plot, y=part_fn(zarr[mask]), mode="lines",
                            line=dict(color="#1f77b4", width=2)))
                        fig.update_layout(
                            title=dict(text=f"{part_lbl}({zlabel})", font=dict(size=12)),
                            xaxis_title=tr("Frequency (GHz)", "頻率 (GHz)"),
                            yaxis_title=f"{part_lbl}({zlabel}) (Ω)",
                            plot_bgcolor="white", paper_bgcolor="white", height=220,
                            margin=dict(l=50, r=20, t=35, b=40), showlegend=False)
                        fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                        fig.update_yaxes(showgrid=True, gridcolor="#ebebeb")
                        plotly_with_dl(fig,
                                       key=f"pfp_z_{zlabel}_{part_lbl}_{model_short}_{fname}",
                                       filename=f"pfp_z_{zlabel}_{part_lbl}_{model_short}_{fname}",
                                       container=col_w)

                prev_range = (f_lo, f_hi)
                if g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # ── fbi_fit_group: Fbi vs ω², single fit-window slider ───────────────
            if group.get("fbi_fit_group"):
                for formula_type, formula_content in group.get("formulas", []):
                    if formula_type == "latex":
                        st.latex(formula_content)
                    else:
                        st.markdown(formula_content)

                sl_key = f"pfp_fbi_sl_{model_short}_{fname}"
                if sl_key not in st.session_state:
                    st.session_state[sl_key] = f_max_v
                f_hi_fbi = st.slider(
                    tr("Fbi linear fit upper frequency (GHz)",
                       "Fbi 線性擬合上限頻率 (GHz)"),
                    min_value=f_min_v, max_value=f_max_v,
                    value=float(st.session_state[sl_key]),
                    step=step_v, format="%.2f", key=sl_key)

                n_fit_new  = int(np.sum(f_ghz <= f_hi_fbi))
                n_fit_key  = f"pfp_nfit_{model_short}_{fname}"
                n_fit_prev = st.session_state.get(n_fit_key, n_fit_new)
                st.session_state[n_fit_key] = n_fit_new

                if n_fit_new != n_fit_prev and reextract_fn is not None:
                    params_out["_n_fit"] = n_fit_new
                    try:
                        _new_p, _new_a = reextract_fn(params_out, 1, live_arrays)
                        live_params.update(_new_p)
                        params_out.update(_new_p)
                        live_arrays.update(_new_a)
                    except Exception:
                        pass
                    params_out.pop("_n_fit", None)

                omega2  = live_arrays.get("omega2")
                Fbi_arr = live_arrays.get("Fbi")
                A0      = live_params.get("A0", 0.0)
                B0      = live_params.get("B0", 0.0)

                if Fbi_arr is not None and omega2 is not None:
                    valid_mask = (np.isfinite(Fbi_arr) & (np.abs(Fbi_arr) < 1e15)
                                  & (Fbi_arr > 0))
                    win_mask   = valid_mask & (f_ghz <= f_hi_fbi)
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=omega2[valid_mask], y=Fbi_arr[valid_mask], mode="markers",
                        name=tr("All data", "全部資料"), marker=dict(size=4, color="#aec7e8")))
                    fig.add_trace(go.Scattergl(
                        x=omega2[win_mask], y=Fbi_arr[win_mask], mode="markers",
                        name=tr("Fit window", "擬合視窗"), marker=dict(size=6, color="#1f77b4")))
                    if A0 > 1e-30 and win_mask.any():
                        xf = np.linspace(0, float(omega2[win_mask].max()) * 1.1, 200)
                        fig.add_trace(go.Scattergl(
                            x=xf, y=A0 + B0 * xf, mode="lines",
                            name=f"{tr('Fit', '擬合')}  A₀={A0:.3e}  B₀={B0:.3e}",
                            line=dict(color="#d62728", dash="dash", width=2)))
                    fig.update_layout(
                        title=dict(text="Fbi vs ω²  [Eq. 8]", font=dict(size=12)),
                        xaxis_title="ω² (rad²/s²)", yaxis_title="Fbi (rad/s)",
                        plot_bgcolor="white", paper_bgcolor="white", height=300,
                        margin=dict(l=55, r=10, t=40, b=42), showlegend=True,
                        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                                    font=dict(size=9)))
                    fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb")
                    plotly_with_dl(fig, key=f"pfp_fbi_{model_short}_{fname}",
                                   filename=f"pfp_fbi_{model_short}_{fname}")

                    Tbi_fit = float(np.sqrt(max(B0 / A0, 0.0))) if A0 > 1e-30 else 0.0
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("A₀", f"{A0:.4e}", help=tr("Intercept of Fbi vs ω²", "Fbi 對 ω² 的截距"))
                    mc2.metric("B₀", f"{B0:.4e}", help=tr("Slope of Fbi vs ω²", "Fbi 對 ω² 的斜率"))
                    mc3.metric("Tbi = √(B₀/A₀)", f"{Tbi_fit*1e12:.4f} ps",
                               help=tr("Intrinsic base time constant from fit",
                                       "由擬合求得的本徵基極時間常數"))

                prev_range = (f_hi_fbi, f_hi_fbi)
                if g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # ── f1_fit_group: F1 vs ω² fit + Tbe number_input ───────────────────
            if group.get("f1_fit_group"):
                if g_deps:
                    st.caption(f"{tr('Depends on', '依存於')}: {', '.join(g_deps)}")
                for formula_type, formula_content in group.get("formulas", []):
                    if formula_type == "latex":
                        st.latex(formula_content)
                    else:
                        st.markdown(formula_content)

                sl_key = f"pfp_f1_sl_{model_short}_{fname}"
                if sl_key not in st.session_state:
                    st.session_state[sl_key] = f_max_v
                f_hi_f1 = st.slider(
                    tr("F1 linear fit upper frequency (GHz)",
                       "F1 線性擬合上限頻率 (GHz)"),
                    min_value=f_min_v, max_value=f_max_v,
                    value=float(st.session_state[sl_key]),
                    step=step_v, format="%.2f", key=sl_key)

                n_fit_f1_new = int(np.sum(f_ghz <= f_hi_f1))
                nf1_key      = f"pfp_nfit_f1_{model_short}_{fname}"
                nf1_prev     = st.session_state.get(nf1_key, n_fit_f1_new)
                st.session_state[nf1_key] = n_fit_f1_new

                if n_fit_f1_new != nf1_prev and reextract_fn is not None:
                    params_out["_n_fit_f1"] = n_fit_f1_new
                    try:
                        _new_p, _new_a = reextract_fn(params_out, 4, live_arrays)
                        live_params.update(_new_p)
                        params_out.update(_new_p)
                        live_arrays.update(_new_a)
                    except Exception:
                        pass
                    params_out.pop("_n_fit_f1", None)

                omega2 = live_arrays.get("omega2")
                F1_arr = live_arrays.get("F1")
                A1     = live_params.get("A1", 0.0)
                B1     = live_params.get("B1", 0.0)

                if F1_arr is not None and omega2 is not None:
                    valid_mask = (np.isfinite(F1_arr) & (np.abs(F1_arr) < 1e15)
                                  & (F1_arr > 0))
                    win_mask   = valid_mask & (f_ghz <= f_hi_f1)
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=omega2[valid_mask], y=F1_arr[valid_mask], mode="markers",
                        name=tr("All data", "全部資料"), marker=dict(size=4, color="#aec7e8")))
                    fig.add_trace(go.Scattergl(
                        x=omega2[win_mask], y=F1_arr[win_mask], mode="markers",
                        name=tr("Fit window", "擬合視窗"), marker=dict(size=6, color="#1f77b4")))
                    if A1 > 1e-30 and win_mask.any():
                        xf = np.linspace(0, float(omega2[win_mask].max()) * 1.1, 200)
                        fig.add_trace(go.Scattergl(
                            x=xf, y=A1 + B1 * xf, mode="lines",
                            name=f"{tr('Fit', '擬合')}  A={A1:.3e}  B={B1:.3e}",
                            line=dict(color="#d62728", dash="dash", width=2)))
                    fig.update_layout(
                        title=dict(text="F1 vs ω²  [Eq. 19]", font=dict(size=12)),
                        xaxis_title="ω² (rad²/s²)", yaxis_title="F1 (rad/s)",
                        plot_bgcolor="white", paper_bgcolor="white", height=300,
                        margin=dict(l=55, r=10, t=40, b=42), showlegend=True,
                        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                                    font=dict(size=9)))
                    fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb")
                    plotly_with_dl(fig, key=f"pfp_f1_{model_short}_{fname}",
                                   filename=f"pfp_f1_{model_short}_{fname}")

                    alpha   = 1.0 / A1 if A1 > 1e-30 else 0.0
                    Tbe_fit = float(np.sqrt(max(B1 / A1, 0.0))) if A1 > 1e-30 else 0.0
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("A", f"{A1:.4e}", help=tr("Intercept of F1 vs ω²", "F1 對 ω² 的截距"))
                    mc2.metric("B", f"{B1:.4e}", help=tr("Slope of F1 vs ω²", "F1 對 ω² 的斜率"))
                    mc3.metric("α = 1/A", f"{alpha:.4e}", help="α = R(T − Tbe)")
                    mc4.metric("Tbe = √(B/A)", f"{Tbe_fit*1e12:.4f} ps",
                               help=tr("Emitter time constant from fit",
                                       "由擬合求得的射極時間常數"))

                # Tbe number_input (user-overridable)
                _upstream_vals = tuple(
                    round(params_out.get(spec[1], 0.0) * 1e15)
                    for gi in range(g_idx)
                    for spec in param_groups[gi]["params"]
                )
                upstream_tag = str(hash(_upstream_vals) % (10 ** 9))
                rng_tag      = f"{f_hi_f1:.3f}"

                for arr_key, param_key, label, scale, unit in group.get("params", []):
                    if arr_key not in live_arrays:
                        continue
                    raw     = live_arrays[arr_key]
                    arr_num = np.abs(raw) if np.iscomplexobj(raw) else np.real(raw)
                    fin     = arr_num[np.isfinite(arr_num)]
                    auto_SI   = float(np.median(fin)) if len(fin) > 0 \
                                else float(params_out.get(param_key, 0.0))
                    auto_disp = auto_SI * scale
                    inp_key   = (f"pfp_inp_{model_short}_{param_key}_{fname}"
                                 f"_{rng_tag}_{upstream_tag}")
                    actual_val = st.number_input(
                        f"{label} ({unit})" if unit else label,
                        value=float(auto_disp), format="%.5g", key=inp_key)
                    params_out[param_key] = actual_val / scale

                # Re-extract downstream if Tbe changed
                if reextract_fn is not None:
                    _grp_pk  = {spec[1] for spec in group.get("params", [])}
                    _grp_ak  = {spec[0] for spec in group.get("params", [])}
                    _changed = any(
                        abs(params_out.get(pk, 0.0) - live_params.get(pk, 0.0))
                        > 1e-9 * (abs(live_params.get(pk, 0.0)) + 1e-30)
                        for pk in _grp_pk
                    )
                    if _changed:
                        try:
                            _new_p, _new_a = reextract_fn(params_out, g_idx, live_arrays)
                            _processed_pk = {
                                spec[1]
                                for gi in range(g_idx + 1)
                                for spec in param_groups[gi]["params"]
                            }
                            _processed_ak = {
                                spec[0]
                                for gi in range(g_idx + 1)
                                for spec in param_groups[gi]["params"]
                            }
                            for _k, _v in _new_p.items():
                                if _k not in _processed_pk:
                                    live_params[_k] = _v
                                    params_out[_k]  = _v
                            for _k, _v in _new_a.items():
                                if _k not in _processed_ak:
                                    live_arrays[_k] = _v
                        except Exception:
                            pass

                prev_range = (f_hi_f1, f_hi_f1)
                if g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # Wrap the whole parameter group (title + slider + per-parameter
            # plots) in one large card, so each group (Cbex, Cbcx, intrinsic,
            # τB, τC, …) reads as distinct.  `border=False` here — the thick
            # outline is drawn by the scoped `st-key-pfp_groupbox_*` CSS above
            # (Streamlit's own border wrapper would otherwise add a second,
            # thin frame).  All of this group's top-level widgets render into
            # `box`; nested per-parameter plots inherit it via `box.columns()`.
            box = st.container(
                border=False,
                key=f"pfp_groupbox_{model_short}_{g_idx}_{fname}")
            box.markdown(f"**{g_label}**")
            if g_deps:
                box.caption(f"{tr('Depends on', '依存於')}: {', '.join(g_deps)}")

            slider_key = f"pfp_sl_{model_short}_{g_idx}_{fname}"

            # "Use previous range" button for dependent groups
            if g_deps:
                if box.button(tr("↩ Same range as previous group",
                                 "↩ 與上一組相同範圍"),
                             key=f"pfp_useprev_{model_short}_{g_idx}_{fname}"):
                    st.session_state[slider_key] = prev_range
                    st.rerun()

            # Render per-group formulas before the slider
            for formula_type, formula_content in group.get("formulas", []):
                if formula_type == "markdown":
                    box.markdown(formula_content)
                elif formula_type == "latex":
                    box.latex(formula_content)


            # Initialize slider — pre-seed session_state and DO NOT
            # pass `value=` alongside `key=`, otherwise Streamlit logs
            # `check_session_state_rules` warnings about a widget being
            # created with both a default value and a Session-State entry.
            # Also clamp any pre-existing value to the current [min, max]
            # bounds in case a previous file had a different freq range.
            if slider_key not in st.session_state:
                st.session_state[slider_key] = (f_min_v, f_max_v)
            else:
                _cur = st.session_state[slider_key]
                try:
                    _lo = max(f_min_v, min(f_max_v, float(_cur[0])))
                    _hi = max(f_min_v, min(f_max_v, float(_cur[1])))
                    if _lo > _hi:
                        _lo, _hi = f_min_v, f_max_v
                    st.session_state[slider_key] = (_lo, _hi)
                except (TypeError, ValueError, IndexError):
                    st.session_state[slider_key] = (f_min_v, f_max_v)

            f_lo, f_hi = box.slider(
                tr("Frequency range (GHz)", "頻率範圍 (GHz)"),
                min_value=f_min_v, max_value=f_max_v,
                step=step_v, format="%.2f",
                key=slider_key)

            mask   = (f_ghz >= f_lo) & (f_ghz <= f_hi)
            f_plot = f_ghz[mask]
            # Tag used in widget keys — changing it recreates inputs fresh on slider move
            rng_tag = f"{f_lo:.3f}_{f_hi:.3f}"

            valid_specs = [
                s for s in g_params
                if s[0] in live_arrays and isinstance(live_arrays[s[0]], np.ndarray)
            ]


            _is_cbex_sweep_group = (group.get("cbex_sweep_group")
                                    and cbex_sweep_fn is not None
                                    and "Cbex_arr" in live_arrays)
            _cbex_sweep_rendered = False

            for row_start in range(0, len(valid_specs), 2):
                row  = valid_specs[row_start:row_start + 2]
                cols = box.columns(2)
                for _col_outer, (arr_key, param_key, label, scale, unit) in zip(cols, row):
                    # Wrap each parameter's plot + input + quickset buttons in
                    # a bordered container so individual extracted parameters
                    # are visually distinct from each other within a group.
                    col_w = _col_outer.container(border=True)
                    raw_masked = live_arrays[arr_key][mask]
                    arr_plot   = (np.abs(raw_masked) if np.iscomplexobj(raw_masked)
                                  else np.real(raw_masked)) * scale

                    # Recompute median from current slider range
                    arr_num = np.abs(raw_masked) if np.iscomplexobj(raw_masked) else np.real(raw_masked)
                    fin     = arr_num[np.isfinite(arr_num)]
                    _use_first = param_key in group.get("use_first_params", set())
                    if _use_first:
                        auto_SI = float(fin[0]) if len(fin) > 0 else float(params.get(param_key, 0.0))
                    else:
                        auto_SI = float(np.median(fin)) if len(fin) > 0 else float(params.get(param_key, 0.0))

                    auto_disp = auto_SI * scale

                    # Pre-read session state so plot can use it before number_input renders
                    # Hash of all upstream groups' current values — changes when any
                    # upstream param is overridden, forcing downstream inputs to reset.
                    _upstream_vals = tuple(
                        round(params_out.get(spec[1], 0.0) * 1e15)
                        for gi in range(g_idx)
                        for spec in param_groups[gi]["params"]
                    )
                    upstream_tag = str(hash(_upstream_vals) % (10 ** 9))
                    inp_key   = f"pfp_inp_{model_short}_{param_key}_{fname}_{rng_tag}_{upstream_tag}"

                    # Transfer any pending override (e.g. from the Cbex
                    # sweep tool) into the widget key.  Must happen BEFORE
                    # the number_input is instantiated, or Streamlit raises
                    # "session_state ... cannot be modified after widget".
                    pending_key = f"pfp_pending_{model_short}_{param_key}_{fname}"
                    if pending_key in st.session_state:
                        st.session_state[inp_key] = st.session_state.pop(pending_key)
                    # Same for quickset-button writes
                    apply_pending(inp_key)

                    user_disp = float(st.session_state.get(inp_key, auto_disp))
                    user_SI   = user_disp / scale

                    ylabel = f"{label} ({unit})" if unit else label
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=f_plot, y=arr_plot, mode="lines", name=label,
                        line=dict(color="#1f77b4", width=2)))
                    if np.isfinite(user_disp):
                        fig.add_hline(
                            y=user_disp,
                            line=dict(color="#d62728", width=1.8, dash="dash"),
                            annotation_text=f"{user_disp:.4g} {unit}",
                            annotation_position="right",
                            annotation_font=dict(size=9, color="#d62728"))
                    # Cold-section value (computed once — used for both the
                    # plot annotation and the "cold = …" quickset button).
                    _cold_disp = None
                    if cold_res is not None and cold_param_map is not None:
                        _ck = cold_param_map.get(param_key)
                        if _ck and _ck in cold_res:
                            _v = float(cold_res[_ck]) * scale
                            if np.isfinite(_v):
                                _cold_disp = _v
                    if _cold_disp is not None:
                        fig.add_hline(
                            y=_cold_disp,
                            line=dict(color="#2ca02c", width=1.5, dash="dot"),
                            annotation_text=f"{tr('Cold', '冷測')}: {_cold_disp:.4g} {unit}",
                            annotation_position="left",
                            annotation_font=dict(size=9, color="#2ca02c"))
                    fig.update_layout(
                        title=dict(text=label, font=dict(size=12)),
                        xaxis_title=tr("Frequency (GHz)", "頻率 (GHz)"), yaxis_title=ylabel,
                        plot_bgcolor="white", paper_bgcolor="white", height=240,
                        margin=dict(l=50, r=60, t=35, b=40),
                        showlegend=False, hovermode="x unified")

                    fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                    _y_range = None
                    if np.isfinite(user_disp) and abs(user_disp) > 1e-30:
                        _v5  = 5.0 * abs(user_disp)
                        _fin = arr_plot[np.isfinite(arr_plot)]
                        if len(_fin) > 0 and (_fin.max() > _v5 or _fin.min() < -_v5):
                            _y_range = [-_v5, _v5]
                    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb",
                                     **({"range": _y_range} if _y_range is not None else {}))

                    plotly_with_dl(fig,
                                   key=f"pfp_{model_short}_{arr_key}_{fname}",
                                   filename=f"pfp_{model_short}_{arr_key}_{fname}",
                                   container=col_w)

                    # Number input — key includes rng_tag so it resets to new median on slider move
                    actual_val = col_w.number_input(
                        f"{label} ({unit})" if unit else label,
                        value=float(auto_disp),
                        format="%.5g",
                        key=inp_key)

                    # Quickset buttons row beneath the input (Step 3 layout).
                    # cold_disp surfaces beside "default" when the parameter
                    # has a value extracted in the Cold-HBT section.
                    quickset_buttons(container=col_w,
                                      key_prefix=inp_key,
                                      target_key=inp_key,
                                      arr_disp=arr_plot,
                                      default_disp=auto_disp,
                                      cold_disp=_cold_disp,
                                      unit=unit,
                                      fmt="%.4g", layout="below")

                    # Extra quickset buttons sourced from elsewhere in the UI:
                    #   - τB / τC : v_c-method values published by the
                    #     tau_total_fit_group section (T-models, ≥2 files).
                    #   - Rbe     : Z-parameter method result (when run and
                    #     this DUT is part of the fit).
                    _extra_qs = []
                    if param_key in ("tauB", "tauC"):
                        _pub = st.session_state.get(
                            f"taut_pub_{param_key}_{model_short}_{fname}")
                        if _pub is not None and np.isfinite(_pub):
                            _extra_qs.append(("v_c", float(_pub) * scale))
                    elif param_key == "Rbe":
                        _rz_rbe = st.session_state.get(f"rz12_Rbe_{fname}")
                        if _rz_rbe is not None and np.isfinite(_rz_rbe) \
                                and abs(_rz_rbe) > 0:
                            _extra_qs.append(("Z-param", float(_rz_rbe) * scale))

                    if _extra_qs:
                        _bcols = col_w.columns(len(_extra_qs))
                        for _bc, (_lbl, _val) in zip(_bcols, _extra_qs):
                            _btn_text = (f"{_lbl} = {_val:.4g} {unit}"
                                         if unit else f"{_lbl} = {_val:.4g}")
                            if _bc.button(_btn_text,
                                          key=f"{inp_key}_qs_extra_{_lbl}",
                                          width="stretch",
                                          help=tr(f"Set {label} to the {_lbl} reference value",
                                                  f"將 {label} 設為 {_lbl} 參考值")):
                                st.session_state[inp_key + "_pending"] = float(_val)
                                st.rerun()

                    params_out[param_key] = actual_val / scale

                # Render the Cbex sweep tool in the unused 2nd column,
                # immediately beside the Cbex plot.
                if (_is_cbex_sweep_group
                        and not _cbex_sweep_rendered
                        and len(row) < 2):
                    with cols[1]:
                        _render_cbex_sweep_tool(
                            cbex_arr=live_arrays["Cbex_arr"],
                            freq=freq,
                            f_ghz=f_ghz,
                            f_min_v=f_min_v,
                            f_max_v=f_max_v,
                            cbex_scale=g_params[0][3],   # 1e15 (fF)
                            cbex_unit=g_params[0][4],    # "fF"
                            cbex_param_key=g_params[0][1],  # "Cbex"
                            model_short=model_short,
                            fname=fname,
                            g_idx=g_idx,
                            rng_tag=rng_tag,
                            cbex_sweep_fn=cbex_sweep_fn,
                            param_groups=param_groups,
                        )
                    _cbex_sweep_rendered = True

            # ── Re-extract downstream groups if any param in this group changed ──
            if reextract_fn is not None and g_idx < len(param_groups) - 1:
                _grp_param_keys = {spec[1] for spec in g_params}
                _grp_arr_keys   = {spec[0] for spec in g_params}
                _any_changed = any(
                    abs(params_out.get(pk, 0.0) - live_params.get(pk, 0.0))
                    > 1e-9 * (abs(live_params.get(pk, 0.0)) + 1e-30)
                    for pk in _grp_param_keys
                    if pk in params_out
                )
                if _any_changed:
                    try:
                        _new_p, _new_a = reextract_fn(params_out, g_idx, live_arrays)
                        # Only overwrite params/arrays from groups not yet processed.
                        # Protecting all groups 0..g_idx prevents a downstream
                        # re-extraction (e.g. tauB change) from clobbering user
                        # overrides set in earlier groups (e.g. Rbi, Rbe, …).
                        _processed_param_keys = {
                            spec[1]
                            for gi in range(g_idx + 1)
                            for spec in param_groups[gi]["params"]
                        }
                        _processed_arr_keys = {
                            spec[0]
                            for gi in range(g_idx + 1)
                            for spec in param_groups[gi]["params"]
                        }
                        for _k, _v in _new_p.items():
                            if _k not in _processed_param_keys:
                                live_params[_k] = _v
                                params_out[_k]  = _v
                        for _k, _v in _new_a.items():
                            if _k not in _processed_arr_keys:
                                live_arrays[_k] = _v
                    except Exception:
                        pass  # silently ignore re-extraction failures

            prev_range = (f_lo, f_hi)
            if g_idx < len(param_groups) - 1:
                st.markdown("---")

    return params_out


# ── Streamlit fragment decorator (1.36 → st.experimental_fragment;
#    1.37+ → st.fragment).  Wrapping the slider-preview render functions
#    in a fragment confines slider-drag reruns to JUST the fragment —
#    the rest of the SSM script (Sections 1-5, all other models) does
#    NOT re-execute.  That's the order-of-magnitude speedup for Live
#    mode (1-2 s per drag → ~150 ms).
_FRAGMENT = (getattr(st, "fragment", None)
             or getattr(st, "experimental_fragment", None)
             or (lambda f: f))


# ── Tuning (parameter sweep + residual table) ───────────────────────────────

def _make_sweep_values(min_val, max_val, step):
    """Generate sweep values, always including max_val as the last point."""
    if step <= 0 or abs(max_val - min_val) < 1e-15:
        return np.array([min_val])
    if max_val < min_val:
        return np.array([min_val])
    values = np.arange(min_val, max_val + step * 0.5, step)
    if len(values) == 0:
        return np.array([min_val])
    # Ensure max is included
    if abs(values[-1] - max_val) > 1e-12:
        values = np.append(values, max_val)
    return values


def _fmt_eta(seconds) -> str:
    """Human-readable ETA: seconds under a minute, ``Xm Ys`` under an hour,
    ``Xh Ym Zs`` beyond an hour."""
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{s:.1f}s"
    total = int(round(s))
    if total < 3600:
        m, sec = divmod(total, 60)
        return f"{m}m {sec}s"
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m {sec}s"


def _fmt_eval_time(seconds) -> str:
    """Total run time as ``xx s (xx h: xx m: xx s)`` — raw seconds plus an
    hours:minutes:seconds breakdown."""
    s = max(0.0, float(seconds))
    total = int(s)
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{s:.1f} s ({h} h: {m:02d} m: {sec:02d} s)"


def _render_slider_preview(model_cls, all_p, S_raw, freq, z0,
                           tuning_specs, fname, topo_key):
    """Sandbox-style slider preview at the top of the Tuning expander.

    Two flavors selectable via the mode selector:

      🎯 All sweep           — drag any number of sliders; every drag-tick
                                triggers a Streamlit rerun + a full sim.
                                Slow with many params or many freq points,
                                but supports multi-param sliding.

      ⚡ Smooth sweep        — click ``🧮 Build animation`` once, then the
                                embedded Plotly figure scrubs through
                                pre-computed frames entirely client-side
                                (no Streamlit rerun per drag-tick).
                                Multi-param: frames are the cartesian
                                product of every selected sweep axis.

    Sliders in either mode write into ``slpreview_*`` session keys.
    ✅ "Use these values" (live mode) copies them into the main ``sim_*``
    keys; the auto-save gate in ``render_override_and_smith`` then picks
    that change up and persists it to the fit cache.
    """
    mode_key = f"slpreview_mode_{topo_key}_{fname}"
    mode = segmented_radio(
        tr("Preview mode", "預覽模式"),
        [tr("🎯 All sweep", "🎯 全參數掃描"),
         tr("⚡ Smooth sweep", "⚡ 平滑掃描")],
        index=0,
        key=mode_key,
        help=tr(
            "🎯 All sweep — best for a few small changes: the plots "
            "re-compute on every drag.  ⚡ Smooth sweep — best for exploring a "
            "large range: pre-computes the whole range once so dragging is "
            "instant afterwards.",
            "🎯 全參數掃描 — 適合少量微調：每次拖曳都會重新計算圖表。"
            "⚡ 平滑掃描 — 適合探索大範圍：一次預先計算整個範圍，之後拖曳即時反應。"))
    if mode.startswith("⚡"):
        _render_plotly_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                       tuning_specs, fname, topo_key)
    else:
        _render_live_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                     tuning_specs, fname, topo_key)


def _slider_default_range(current_disp, key="", label=""):
    """Sane default (min, max, step) for one slider given the current value.

    The (min, max) are clamped into ``tune_hard_limits(key, label)`` so R / L / C
    sliders floor at 0 and alpha0 stays inside [0.95, 0.99].  A zero current
    value defaults to (0.0, 1.0, 0.01) rather than a symmetric ±1 span.
    """
    hard_lo, hard_hi = tune_hard_limits(key, label)
    if abs(current_disp) < 1e-30:
        d_min, d_max = 0.0, 1.0
    else:
        lo = current_disp * 0.1 if current_disp > 0 else current_disp * 10
        hi = current_disp * 10  if current_disp > 0 else current_disp * 0.1
        d_min, d_max = min(lo, hi), max(lo, hi)
    d_min, d_max = _clamp_to_hard(d_min, d_max, hard_lo, hard_hi)
    d_step = max((d_max - d_min) / 100, 1e-9)
    return d_min, d_max, d_step


def _multi_metric_top_n(arr, per_metric: int = 10):
    """Return the union of top-``per_metric`` rows by each of the first five
    columns (Total, S11, S12, S21, S22), deduped, sorted by Total Residual.

    ``arr`` is an (N, n_cols) float ndarray.  Rows whose Total Residual is
    non-finite or >= 1e30 (the BIG sentinel used to mark filtered combos)
    are dropped first.  Result row count: 10 (all five metrics' bests are
    the same row) ≤ R ≤ 50 (all distinct).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or arr.shape[0] == 0:
        return arr
    finite_mask = np.isfinite(arr[:, 0]) & (arr[:, 0] < 1e30)
    arr = arr[finite_mask]
    if arr.shape[0] == 0:
        return arr
    keep: set[int] = set()
    for c in range(min(5, arr.shape[1])):
        col_vals = arr[:, c]
        col_safe = np.where(np.isfinite(col_vals), col_vals, np.inf)
        n_take   = min(per_metric, arr.shape[0])
        idx      = np.argpartition(col_safe, n_take - 1)[:n_take]
        keep.update(idx.tolist())
    sub = arr[sorted(keep)]
    unique = np.unique(sub, axis=0)
    return unique[np.argsort(unique[:, 0])]


# ── Progressive auto-fit (🪜) — constants + pure geometry helpers ─────────────
#
# The progressive driver (nested inside render_tuning_expander) leans on these
# pure module-level helpers so the grid geometry / bounds-shrink logic can be
# unit-tested without a Streamlit session.

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


@_FRAGMENT
def _render_live_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                tuning_specs, fname, topo_key):
    """Streamlit-rerun-per-drag preview.

    Layout (v4 — fixed-height scroll containers, same primitive
    Streamlit's own sidebar uses for independent scrolling)
    --------------------------------------------------------
    Two top-level columns, each wrapped in ``st.container(height=…)``
    so they get their own scrollbar.  The page itself does not grow
    when many variable cards are added — the LEFT container scrolls
    internally, the RIGHT container stays put.

      ┌─ Left scroll-box ──────────┐ ┌─ Right scroll-box ─────────────┐
      │ [Parameters to slide]      │ │   Smith         |     Bode     │
      │  (narrow multiselect)      │ │   (compact)     |   (compact)   │
      │ ─────────                  │ │   legend below  |  legend below │
      │ variable card 1            │ │                                │
      │ variable card 2            │ │                                │
      │ variable card 3            │ │                                │
      │ ... (1 per row, scrolls)   │ │                                │
      └────────────────────────────┘ └────────────────────────────────┘
      ┌─ Below the columns (always visible) ─────────────────────────┐
      │  ✅ Use these values     ↩️ Reset preview                     │
      └───────────────────────────────────────────────────────────────┘

    Buttons live OUTSIDE the scroll boxes so the user doesn't have to
    scroll the left panel to find them.
    """
    from ..ssm_plots import render_ft_fmax_card

    # Fixed height for the two scroll boxes — picks a value comfortable
    # on a typical 1080p laptop, deliberately taller than the plots so
    # the right box never shows its own scrollbar (the plots fit) while
    # the left box gets scrollbars when the user adds many variables.
    _BOX_HEIGHT = 680

    label_for = {s[0]: s[1] for s in tuning_specs}

    # ── Top-level split — keep the slider column compact so the plots get
    #    most of the width.
    controls_col, plots_col = st.columns([0.6, 1.4])

    # Pre-declare BOTH scroll containers so we can append into them in
    # any order (simulation happens after slider values are read).
    with controls_col:
        sliders_box = st.container(height=_BOX_HEIGHT, border=False)
    with plots_col:
        plots_box = st.container(height=_BOX_HEIGHT, border=False)

    # ── Controls box: multiselect (narrowed) + variable cards ─────────
    with sliders_box:
        # Constrain multiselect width with a sub-column so the chips
        # don't wrap awkwardly across the full panel.
        sel_key  = f"slpreview_sel_{topo_key}_{fname}"
        ms_col, _ms_pad = st.columns([3, 1])
        with ms_col:
            selected = st.multiselect(
                tr("Parameters to slide", "可拖曳參數"),
                options=[s[0] for s in tuning_specs],
                default=st.session_state.get(sel_key, []),
                format_func=lambda k: label_for.get(k, k),
                key=sel_key,
                placeholder=tr("Choose options", "請選擇項目"),
                help=tr(
                    "Pick parameter(s) to drag.  Plots on the right "
                    "show the slider-substituted model in real time.  "
                    "The main Smith / fT-fmax plots above stay frozen "
                    "until you click ✅ Use these values.",
                    "選擇要拖曳的參數。右側圖表即時顯示套用滑桿值後的模型。"
                    "上方主要的 Smith / fT-fmax 圖會維持不變，"
                    "直到你點擊「✅ 使用這些數值」為止。"))

        selected_specs = [s for s in tuning_specs if s[0] in selected]
        preview_overrides: dict[str, float] = {}

        # Initialize ranges + sliders
        for spec in selected_specs:
            key, _label, scale = spec[0], spec[1], spec[2]
            current_disp  = float(all_p.get(key, 0.0)) * scale
            kp = f"slpreview_{topo_key}_{key}_{fname}"
            if f"{kp}_min" not in st.session_state:
                d_min, d_max, d_step = _slider_default_range(
                    current_disp, key, _label)
                st.session_state[f"{kp}_min"]  = float(d_min)
                st.session_state[f"{kp}_max"]  = float(d_max)
                st.session_state[f"{kp}_step"] = float(d_step)
            if kp not in st.session_state:
                st.session_state[kp] = float(current_disp)

        # Variable cards — ONE per row.  The fixed-height scroll box
        # handles overflow internally.
        if not selected_specs:
            st.caption(tr("Select one or more parameters above to begin.",
                          "請先在上方選擇一個或多個參數以開始。"))
        else:
            for spec in selected_specs:
                key, label, scale = spec[0], spec[1], spec[2]
                unit = spec[3] if len(spec) > 3 else ""
                fmt  = spec[4] if len(spec) > 4 else "%.4g"
                current_disp = float(all_p.get(key, 0.0)) * scale
                kp = f"slpreview_{topo_key}_{key}_{fname}"
                mn = float(st.session_state[f"{kp}_min"])
                mx = float(st.session_state[f"{kp}_max"])
                sp = float(st.session_state[f"{kp}_step"])
                if mx <= mn:
                    mx = mn + max(sp, abs(mn) * 1e-6 + 1e-9)
                sp_safe = sp if sp > 0 else max((mx - mn) / 100, 1e-12)
                # Clamp the persisted slider value into the CURRENT
                # min/max bounds and write it back to session_state
                # BEFORE the widget renders.  Passing `value=` to a
                # widget that also has `key=` (where the key is in
                # session_state) triggers Streamlit's
                # check_session_state_rules warning every render —
                # the supported pattern is "set the key in
                # session_state, then omit value=".
                cur_v = min(max(float(st.session_state.get(kp, current_disp)),
                                mn), mx)
                st.session_state[kp] = cur_v
                label_unit = f"{label} ({unit})" if unit else label
                with st.container(border=True):
                    head = st.columns([1.4, 1, 1, 1])
                    head[0].markdown(
                        f"<div style='padding-top:1.6em;font-weight:600'>"
                        f"{label_unit}</div>",
                        unsafe_allow_html=True)
                    head[1].number_input(tr("Min", "最小值"), format=fmt,
                                         key=f"{kp}_min")
                    head[2].number_input(tr("Step", "步進"), format=fmt,
                                         key=f"{kp}_step",
                                         min_value=0.0)
                    head[3].number_input(tr("Max", "最大值"), format=fmt,
                                         key=f"{kp}_max")
                    v = st.slider(label_unit, min_value=mn, max_value=mx,
                                  step=sp_safe, format=fmt,
                                  key=kp, label_visibility="collapsed")
                    st.caption(f"{tr('main', '目前')}: **{current_disp:.4g}**  →  "
                               f"{tr('preview', '預覽')}: **{v:.4g}** {unit}".rstrip())
                preview_overrides[key] = float(v) / scale

    # ── Preview simulation (same logic as before, narrowed plot heights
    #    because each plot now occupies a half-width column).
    if preview_overrides:
        all_p_prev = dict(all_p)
        all_p_prev.update(preview_overrides)
        swept_keys = list(preview_overrides.keys())
        xp_live = _cp if _HAS_CUDA else np

        static_cache_key  = f"slpreview_static_cache_{topo_key}_{fname}"
        static_hash_input = {k: float(v) for k, v in all_p.items()
                              if k not in swept_keys
                              and isinstance(v, (int, float))}
        static_hash_input["__swept"] = tuple(sorted(swept_keys))
        static_hash_input["__nf"]    = int(len(freq))
        static_hash_input["__xp"]    = "cuda" if xp_live is not np else "cpu"
        static_hash = params_hash({k: str(v) for k, v in static_hash_input.items()})
        cached_static = st.session_state.get(static_cache_key)
        if cached_static is None or cached_static.get("hash") != static_hash:
            try:
                static_cache = model_cls.build_static_cache(
                    all_p, freq, xp=xp_live, swept_keys=swept_keys)
            except Exception:
                static_cache = None
            st.session_state[static_cache_key] = {"hash": static_hash,
                                                  "cache": static_cache}
            cached_static = st.session_state[static_cache_key]
        static_cache = cached_static.get("cache")

        p_batch = dict(all_p)
        for k, v in preview_overrides.items():
            p_batch[k] = xp_live.asarray([v], dtype=float)
        S_prev = None
        try:
            S_b = model_cls.simulate_batch(p_batch, freq, z0,
                                            xp=xp_live, cache=static_cache)
            if xp_live is not np:
                S_b = _cp.asnumpy(S_b)
            S_prev = np.asarray(S_b)[0]
        except Exception:
            try:
                S_prev = model_cls.simulate_vec(all_p_prev, freq, z0)
            except Exception as e:
                st.error(tr(f"Preview simulation failed: {e}",
                            f"預覽模擬失敗：{e}"))
                S_prev = None
        if S_prev is not None and not np.all(np.isfinite(S_prev)):
            st.warning(tr("Preview S-parameters contain non-finite values — "
                          "adjust slider ranges to avoid singular combinations.",
                          "預覽 S 參數包含非有限值 — 請調整滑桿範圍以避免奇異組合。"))
            S_prev = None
        if S_prev is not None:
            # Render INTO the right scroll box.  Smith + Bode go in two
            # sub-columns so they sit side-by-side, with legends below
            # each plot (compact mode on the smith chart).
            with plots_box:
                st.markdown(
                    f"<div style='font-size:0.85em;color:#555;"
                    f"margin-bottom:4px'>{tr('Preview', '預覽')} — {model_cls.NAME} "
                    f"({tr('residual', '殘差')} {ssm_residual(S_raw, S_prev):.2f}%)"
                    f"</div>",
                    unsafe_allow_html=True)
                smith_col, bode_col = st.columns(2)
                with smith_col:
                    # Default to the per-trace display multipliers set above the
                    # main Smith chart (smith_scale_controls) so the preview
                    # matches it; inline Sxx labels keep the legend off the plot.
                    _scales = {
                        nm: float(st.session_state.get(
                            f"smith_scale_{topo_key}_{nm}_{fname}", 1.0))
                        for nm in ("S11", "S12", "S21", "S22")}
                    render_smith_chart(
                        S_raw, S_prev, model_cls.NAME,
                        ssm_residual(S_raw, S_prev),
                        scales=_scales,
                        key=f"slpreview_smith_{topo_key}_{fname}",
                        show_title=False,
                        compact=True, height=540, inline_labels=True)
                with bode_col:
                    render_ft_fmax_card(
                        S_raw, S_prev, freq,
                        model_name=model_cls.NAME,
                        key=f"slpreview_bode_{topo_key}_{fname}",
                        height=540, compact=True)

    # ── Commit / Reset buttons — appended to the left column BELOW the
    #    sliders_box scroll container, so they're always visible
    #    without scrolling the slider list.
    with controls_col:
        bc1, bc2 = st.columns(2)
        commit_clicked = bc1.container(key=f"hbt_amber_slcommit_{topo_key}").button(
            tr("✅ Use these values", "✅ 使用這些數值"),
            key=f"slpreview_commit_{topo_key}_{fname}",
            disabled=(len(preview_overrides) == 0),
            help=tr(
                "Copy slider values into the fine-tune Smith-chart override "
                "fields above.  Does NOT auto-save to the persistent fit "
                "cache — only direct edits in the fine-tune number_inputs do.",
                "將滑桿數值複製到上方的微調 Smith 圖覆寫欄位。"
                "不會自動儲存到永久擬合快取 — 只有直接編輯微調數字輸入框才會。"),
            width="stretch")
        reset_clicked = bc2.button(
            tr("↩️ Reset preview", "↩️ 重設預覽"),
            key=f"slpreview_reset_{topo_key}_{fname}",
            help=tr("Discard slider drags and clear remembered min/step/max.",
                    "捨棄滑桿拖曳並清除記住的最小值/步進/最大值。"),
            width="stretch")

    if commit_clicked:
        for k, v_si in preview_overrides.items():
            sc = next(s[2] for s in tuning_specs if s[0] == k)
            st.session_state[f"sim_{topo_key}_{k}_{fname}"] = float(v_si) * sc
        st.rerun()

    if reset_clicked:
        for s in tuning_specs:
            kp = f"slpreview_{topo_key}_{s[0]}_{fname}"
            for suf in ("", "_min", "_step", "_max"):
                st.session_state.pop(kp + suf, None)
        st.rerun()


def _chunked_simulate_batch_to_host(model_cls, p_batch, freq, z0, *,
                                     xp, chunk_size: int = 5000,
                                     dtype=np.complex64):
    """Run ``simulate_batch`` in chunks of ``chunk_size`` so OOM doesn't bite
    on million-frame sweeps.  Returns one host-side ``np.ndarray`` of the
    requested complex ``dtype``.

    The default storage dtype is **complex64** (2× memory savings vs.
    complex128 with imperceptible visual difference on Smith + Bode).
    Pass ``dtype=np.complex128`` for full fp64 storage if you need it
    for downstream residual calculations.
    """
    if xp is np:
        # Shrink the effective chunk size to whatever CPU RAM is actually
        # available right now — same per-row working-set model as the
        # Full Auto Tune driver (_run_progressive) and _run_one_sweep's
        # per_combo_bytes. Prevents a caller-supplied (or default) chunk_size
        # from allocating more than the Streamlit Cloud cgroup limit in one
        # simulate_batch call, which would get SIGKILLed before any
        # except MemoryError recovery path could run.
        per_row_bytes = 16 * 4 * len(freq) * 12
        cap = max(256, int(ram_available_bytes() * 0.25 // max(per_row_bytes, 1)))
        chunk_size = min(chunk_size, cap)

    sweep_keys = []
    B = None
    for k, v in p_batch.items():
        if isinstance(v, np.ndarray) or (_HAS_CUDA and isinstance(v, _cp.ndarray)):
            sweep_keys.append(k)
            if B is None:
                B = int(v.shape[0])
    if B is None or B <= chunk_size:
        out = model_cls.simulate_batch(p_batch, freq, z0, xp=xp)
        if xp is not np:
            out = _cp.asnumpy(out)
        return np.asarray(out).astype(dtype, copy=False)

    chunks = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        sub_p = {k: (v[start:end] if k in sweep_keys else v)
                 for k, v in p_batch.items()}
        S_c = model_cls.simulate_batch(sub_p, freq, z0, xp=xp)
        if xp is not np:
            S_c = _cp.asnumpy(S_c)
            try:
                _cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        chunks.append(np.asarray(S_c).astype(dtype, copy=False))
    return np.concatenate(chunks, axis=0)


@_FRAGMENT
def _render_plotly_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                  tuning_specs, fname, topo_key):
    """Pre-computed Plotly slider — joint (cartesian) multi-param scan.

    Each selected param becomes one Plotly slider.  Frames are the *full
    cartesian product* of all sliders' sweep values, so dragging slider
    B reflects the model at the *current position of every other slider*
    (true M × N × K joint behaviour).  Coordination between sliders is
    done by injected JS listening to ``plotly_sliderchange``.

    Performance notes
    -----------------
    • All traces use ``Scattergl`` (WebGL).
    • Frequency axis defaults to full fidelity (≤ 1001 points per trace);
      the **Freq points** input decimates it when the payload grows large.
    • Single ``simulate_batch`` call runs the full cartesian product —
      one GPU pass when CUDA is available.

    Cache
    -----
    Built batch + slider specs stash in session_state until the user
    changes the selection / ranges and clicks ``🧮 Build`` again.
    """
    from ..helpers.plotly_plots import build_smith_bode_slider_payload
    from ..components import smith_bode_slider

    label_for = {s[0]: s[1] for s in tuning_specs}
    options   = [s[0] for s in tuning_specs]

    sel_key = f"slprev_pl_sel_{topo_key}_{fname}"
    selected = st.multiselect(
        tr("Sweep parameters", "掃描參數"),
        options=options,
        default=st.session_state.get(sel_key, [options[0]] if options else []),
        format_func=lambda k: label_for.get(k, k),
        key=sel_key,
        placeholder=tr("Choose options", "請選擇項目"),
        help=tr(
            "Each selected param gets its own Plotly slider in the figure. "
            "Frames are the FULL cartesian product, so dragging slider B "
            "reflects the model at the current position of every other "
            "slider (true joint scan).  Watch the total frame count below "
            "— it grows multiplicatively.",
            "每個選取的參數在圖表中都有各自的 Plotly 滑桿。"
            "各幀是所有滑桿值的完整笛卡兒積，因此拖曳滑桿 B 時，"
            "會反映其他每個滑桿目前位置下的模型（真正的聯合掃描）。"
            "請留意下方的總幀數 — 它會以乘法方式增長。"))

    selected_specs = [s for s in tuning_specs if s[0] in selected]
    if not selected_specs:
        st.caption(tr("Select one or more parameters above and click "
                      "**🧮 Build animation**.",
                      "請先在上方選擇一個或多個參數，再點擊"
                      "**🧮 建立動畫**。"))
        return

    # ── Initialize per-param ranges ────────────────────────────────────
    # Default frames-per-axis shrinks as more params are selected so the
    # cartesian product (and thus the embedded payload) stays manageable
    # at full frequency fidelity: 11 for 1-2 params, 7 for 3, 5 for 4+.
    n_sel = len(selected_specs)
    default_frames = 11 if n_sel <= 2 else (7 if n_sel == 3 else 5)
    for spec in selected_specs:
        key, _label, scale = spec[0], spec[1], spec[2]
        current_disp  = float(all_p.get(key, 0.0)) * scale
        kp = f"slprev_pl_{topo_key}_{key}_{fname}"
        if f"{kp}_min" not in st.session_state:
            d_min, d_max, _ = _slider_default_range(current_disp, key, _label)
            st.session_state[f"{kp}_min"]    = float(d_min)
            st.session_state[f"{kp}_max"]    = float(d_max)
            st.session_state[f"{kp}_frames"] = default_frames

    # ── 2-column variable-card grid (min / max / frames per card) ──────
    def _render_one_range_card(spec):
        key, label, scale = spec[0], spec[1], spec[2]
        unit = spec[3] if len(spec) > 3 else ""
        fmt  = spec[4] if len(spec) > 4 else "%.4g"
        kp   = f"slprev_pl_{topo_key}_{key}_{fname}"
        label_unit = f"{label} ({unit})" if unit else label
        with st.container(border=True):
            head = st.columns([1.4, 1, 1, 1])
            # Top-pad the variable name so it sits at the same vertical
            # level as the input boxes (whose own "Min"/"Max"/"Frames"
            # labels add ~1.6 em of header height above them).
            head[0].markdown(
                f"<div style='padding-top:1.6em;font-weight:600'>"
                f"{label_unit}</div>",
                unsafe_allow_html=True)
            head[1].number_input(tr("Min", "最小值"), format=fmt, key=f"{kp}_min")
            head[2].number_input(tr("Max", "最大值"), format=fmt, key=f"{kp}_max")
            head[3].number_input(tr("Frames", "幀數"), min_value=2, max_value=100, step=1,
                                 key=f"{kp}_frames",
                                 help=tr("Frames per axis (2–100). "
                                         "Total = product across params.",
                                         "每軸幀數（2–100）。"
                                         "總數 = 各參數幀數的乘積。"))

    for row_start in range(0, len(selected_specs), 2):
        row_specs = selected_specs[row_start:row_start + 2]
        l_col, r_col = st.columns(2)
        with l_col:
            _render_one_range_card(row_specs[0])
        with r_col:
            if len(row_specs) > 1:
                _render_one_range_card(row_specs[1])
            else:
                st.empty()

    # ── Decimation (fidelity) control + payload estimate ───────────────
    n_freq_full = int(len(freq))
    decim_default = min(1001, n_freq_full)
    decim_key   = f"slprev_pl_decim_{topo_key}_{fname}"
    if decim_key not in st.session_state:
        st.session_state[decim_key] = decim_default

    dims_preview = []
    for spec in selected_specs:
        kp = f"slprev_pl_{topo_key}_{spec[0]}_{fname}"
        dims_preview.append(int(st.session_state.get(f"{kp}_frames",
                                                     default_frames)))
    total_frames = int(np.prod(dims_preview)) if dims_preview else 0

    # Estimated payload: (5-sig-fig ≈ 7 chars / number) × 10 numbers / sample.
    decim_n = int(st.session_state.get(decim_key, decim_default))
    decim_n = min(decim_n, n_freq_full)
    est_mb  = total_frames * decim_n * 10 * 7 / 1024 / 1024

    fd_col1, fd_col2 = st.columns([1, 2])
    with fd_col1:
        st.number_input(
            f"{tr('Freq points', '頻率點數')} (max: {n_freq_full})",
            min_value=20, max_value=n_freq_full, step=10,
            key=decim_key,
            help=tr(
                f"Frequency points kept per trace (max = {n_freq_full} = "
                "full fidelity).  Lower this (~120 still looks smooth on "
                "Smith / Bode) when the payload estimate grows large.",
                f"每條曲線保留的頻率點數（最大 = {n_freq_full} = 完整精度）。"
                "當估計負載變大時可調低此值（約 120 在 Smith / Bode 圖上"
                "仍相當平滑）。"))
    with fd_col2:
        st.caption(
            f"{tr('Cartesian sweep', '笛卡兒掃描')}: "
            + " × ".join(str(d) for d in dims_preview)
            + f" = **{total_frames}** {tr('frames', '幀')} · {decim_n} "
            f"{tr('freq pts', '頻率點')} · "
            f"{tr('estimated payload', '估計負載')} ≈ **{est_mb:.0f} MB**")

    if est_mb > 180:
        st.error(tr(
            f"❌ Estimated payload ≈ {est_mb:.0f} MB will exceed "
            "Streamlit's 200 MB browser-message limit.  Lower the "
            "**Freq points** value, reduce per-axis frame counts, or "
            "raise the limit via `.streamlit/config.toml` → "
            "`[server] maxMessageSize = 500`.",
            f"❌ 估計負載 ≈ {est_mb:.0f} MB 將超過 Streamlit 的 200 MB "
            "瀏覽器訊息上限。請降低 **頻率點數**、減少每軸幀數，"
            "或透過 `.streamlit/config.toml` → `[server] maxMessageSize = 500` "
            "提高上限。"))
    elif est_mb > 120:
        st.warning(tr(f"⚠️ Estimated payload ≈ {est_mb:.0f} MB is close "
                      "to Streamlit's 200 MB limit.",
                      f"⚠️ 估計負載 ≈ {est_mb:.0f} MB 已接近 Streamlit 的 "
                      "200 MB 上限。"))
    elif total_frames > 2000:
        st.warning(tr(f"⚠️ {total_frames} frames may stutter on "
                      "slider drag.",
                      f"⚠️ {total_frames} 幀可能會導致拖曳滑桿時卡頓。"))

    # ── CUDA checkbox + Build button (button next to checkbox when CUDA available)
    cuda_toggle_key = f"slprev_pl_cuda_{topo_key}_{fname}"
    if _HAS_CUDA:
        cuda_col, btn_col = st.columns([1.6, 1])
        with cuda_col:
            use_cuda = st.checkbox(tr(f"⚡ Use CUDA (cupy {_CUDA_VER}) for "
                                      "batched simulation",
                                      f"⚡ 使用 CUDA（cupy {_CUDA_VER}）"
                                      "進行批次模擬"),
                                    value=st.session_state.get(cuda_toggle_key, True),
                                    key=cuda_toggle_key,
                                    help=tr("Off-load the joint cartesian "
                                            "batched simulation to the GPU.  "
                                            "Result is brought back to host as "
                                            "fp64 for Plotly embedding.",
                                            "將聯合笛卡兒批次模擬卸載至 GPU 運算。"
                                            "結果會以 fp64 帶回主機供 Plotly 嵌入。"))
        with btn_col:
            build_clicked = st.button(tr("🧮 Build animation", "🧮 建立動畫"),
                                      key=f"slprev_pl_build_{topo_key}_{fname}",
                                      width="stretch",
                                      help=tr("Pre-compute the cartesian joint "
                                              "sweep and embed with JS-"
                                              "coordinated multi-sliders.",
                                              "預先計算笛卡兒聯合掃描，"
                                              "並以 JS 協調的多重滑桿嵌入。"))
    else:
        use_cuda = False
        build_clicked = st.button(tr("🧮 Build animation", "🧮 建立動畫"),
                                  key=f"slprev_pl_build_{topo_key}_{fname}",
                                  width="stretch",
                                  help=tr("Pre-compute the cartesian joint "
                                          "sweep and embed with JS-coordinated "
                                          "multi-sliders.",
                                          "預先計算笛卡兒聯合掃描，"
                                          "並以 JS 協調的多重滑桿嵌入。"))

    state_key = f"slprev_pl_state_{topo_key}_{fname}"

    if build_clicked:
        import time as _time
        xp = _cp if (use_cuda and _HAS_CUDA) else np
        if xp is not np:
            device_label = f"GPU (cupy {_CUDA_VER})"
        else:
            # CPU path routes through simulate_batch → the Rust end-to-end
            # kernel whenever it's available (mirrors _detect_rust_active),
            # so the badge must reflect what actually ran, not a hardcoded
            # "numpy".  Only true NumPy composition gets the numpy label.
            try:
                from ..helpers.rust_kernels import (
                    HAS_RUST as _HR,
                    _phase2_dispatch_enabled as _p2on,
                    SIM_FOR_TOPOLOGY as _SIMTOPO,
                )
                from ..helpers import rust_kernels as _RKMOD
                _rust_used = bool(
                    _HR and _p2on() and (
                        _SIMTOPO.get(model_cls.SHORT) is not None
                        or (getattr(model_cls, "USES_RUST_BATCH", False)
                            and getattr(getattr(_RKMOD, "_rk", None),
                                        "sim_custom_batch", None) is not None)))
            except Exception:
                _rust_used = False
            device_label = (tr("CPU (🦀 Rust)", "CPU（🦀 Rust）") if _rust_used
                            else tr("CPU (numpy)", "CPU（numpy）"))
        # Build per-axis sweeps then meshgrid → cartesian product
        sweep_disps  : list[np.ndarray] = []
        sweep_sis    : list[np.ndarray] = []
        slider_specs_out: list[dict] = []
        for spec in selected_specs:
            key, label, scale = spec[0], spec[1], spec[2]
            unit = spec[3] if len(spec) > 3 else ""
            fmt  = spec[4] if len(spec) > 4 else "%.4g"
            kp   = f"slprev_pl_{topo_key}_{key}_{fname}"
            mn = float(st.session_state[f"{kp}_min"])
            mx = float(st.session_state[f"{kp}_max"])
            nf = int(st.session_state[f"{kp}_frames"])
            if mx <= mn:
                mx = mn + abs(mn) * 1e-6 + 1e-9
            sd = np.linspace(mn, mx, nf)
            sweep_disps.append(sd)
            sweep_sis.append(sd / scale)
            slider_specs_out.append(dict(
                key=key, label=label, unit=unit, fmt=fmt,
                values_disp=sd.tolist()))
        meshes = np.meshgrid(*sweep_sis, indexing="ij")
        flats  = [m.ravel() for m in meshes]
        n_total = int(flats[0].size) if flats else 0

        # Spinner so the user sees that compute is happening (the Plotly
        # figure below stays "stale" — the prior build — until the new
        # batch finishes and we replace it).  For very large sweeps the
        # chunked path avoids GPU OOM by simulating in slabs of 5 000.
        t0 = _time.perf_counter()
        with st.spinner(tr(f"Computing {n_total} frames on {device_label}…",
                           f"正在 {device_label} 上計算 {n_total} 幀…")):
            p_batch = dict(all_p)
            for spec, flat in zip(selected_specs, flats):
                p_batch[spec[0]] = xp.asarray(flat, dtype=float)
            try:
                S_b = _chunked_simulate_batch_to_host(
                    model_cls, p_batch, freq, z0, xp=xp)
            except Exception as e:
                st.error(tr(f"Batched preview simulation failed: {e}",
                            f"批次預覽模擬失敗：{e}"))
                return
        elapsed = _time.perf_counter() - t0

        if not np.all(np.isfinite(S_b)):
            st.warning(tr("Some frames contain non-finite S-parameters — "
                          "narrow the ranges to avoid singular combinations.",
                          "部分幀包含非有限的 S 參數 — 請縮小範圍以避免奇異組合。"))

        st.session_state[state_key] = {
            "slider_specs":  slider_specs_out,
            "S_batch":       S_b,
            "selected_keys": [s[0] for s in selected_specs],
            "elapsed_s":     elapsed,
            "device":        device_label,
            "n_total":       n_total,
        }

    state = st.session_state.get(state_key)
    if state is None:
        st.info(tr("Click **🧮 Build animation** to compute frames for the "
                   "Plotly slider(s).",
                   "點擊 **🧮 建立動畫** 以計算 Plotly 滑桿的幀。"))
        return

    cached_keys  = state.get("selected_keys", [])
    current_keys = [s[0] for s in selected_specs]
    if cached_keys != current_keys:
        st.warning(tr("Selection changed since last build "
                      f"(cached: {cached_keys}, current: {current_keys}).  "
                      "Click **🧮 Build animation** to refresh.",
                      f"自上次建立以來選擇已變更"
                      f"（快取：{cached_keys}，目前：{current_keys}）。"
                      "請點擊 **🧮 建立動畫** 以更新。"))
        return

    if state.get("S_batch") is None:
        # e.g. a stale build from the removed server-cached/int16 path.
        st.warning(tr("Cached build is unusable — click **🧮 Build animation** "
                      "to refresh.",
                      "快取的建構結果無法使用 — 請點擊 **🧮 建立動畫** 以更新。"))
        return

    # Device + elapsed banner so the user can confirm GPU vs CPU.
    elapsed = float(state.get("elapsed_s", 0.0))
    n_total = int(state.get("n_total", 0)) or len(state["S_batch"])
    device  = str(state.get("device", "?"))
    ms_each = (elapsed / max(1, n_total)) * 1000.0
    st.caption(tr(
        f"✅ Built **{n_total}** frames on **{device}** in "
        f"**{elapsed:.2f} s** ({ms_each:.1f} ms/frame).  "
        "Drag any slider below to scrub.",
        f"✅ 已在 **{device}** 上建立 **{n_total}** 幀，耗時 "
        f"**{elapsed:.2f} 秒**（{ms_each:.1f} 毫秒/幀）。"
        "拖曳下方任一滑桿即可瀏覽。"))

    payload = build_smith_bode_slider_payload(
        S_batch_joint=state["S_batch"],
        freq=freq,
        slider_specs=state["slider_specs"],
        model_name=model_cls.NAME,
        S_meas=S_raw,
        decimate_points=int(st.session_state.get(decim_key, decim_default)),
        # Mirror the Plotly Smith chart's per-trace display scale (set via
        # smith_scale_controls) so the slider's Smith view matches it.
        smith_mults={
            nm: float(st.session_state.get(
                f"smith_scale_{topo_key}_{nm}_{fname}", 1.0))
            for nm in ("S11", "S12", "S21", "S22")
        },
    )
    # Bidirectional component: renders the Smith+Bode figure with client-side
    # scrub sliders (the per-frame data is inflated in the browser from a
    # gzip+base64 blob via native DecompressionStream, so no multi-MB HTML
    # string is re-embedded on every rerun) plus a "✅ Use these values" button
    # that posts the chosen slider indices back.  Height is self-measured by
    # the component so the button is never clipped.
    n_sl   = len(state["slider_specs"])
    height = 500 + 30 + 36 * n_sl + 52
    ret = smith_bode_slider(
        payload=payload, height=height,
        use_label=tr("✅ Use these values", "✅ 使用這些數值"),
        key=f"sbslider_{topo_key}_{fname}")

    # Commit chosen slider values into the fine-tune sim_* override fields —
    # same target keys + display-unit convention as the Live-mode commit
    # above (`sim_{topo_key}_{k}_{fname}` = value × scale).  values_disp is
    # already in display units, so it's written straight through.  A nonce
    # guard stops the persisted component value from re-committing on every
    # rerun.
    if isinstance(ret, dict) and ret.get("nonce") is not None:
        _nonce_key = f"_sbslider_nonce_{topo_key}_{fname}"
        if ret["nonce"] != st.session_state.get(_nonce_key):
            st.session_state[_nonce_key] = ret["nonce"]
            idxs = ret.get("indices") or []
            for i, sp in enumerate(state["slider_specs"]):
                if i < len(idxs):
                    vals = sp.get("values_disp", [])
                    ii = int(idxs[i])
                    if 0 <= ii < len(vals):
                        st.session_state[
                            f"sim_{topo_key}_{sp['key']}_{fname}"] = float(vals[ii])
            st.rerun()


def render_visual_tuning_expander(model_cls, all_p, S_raw, freq, z0,
                                   tuning_specs, fname, topo_key):
    """🎚️ Visual Tuning expander — drag sliders to see the model react.

    Strictly UI-only: writes into the same fine-tune ``sim_*`` session
    keys via the ✅ commit button.  Auto-save is suppressed for slider
    commits (see ``render_override_and_smith``).
    """
    with st.container(key="hbt_exp_tune_vis_" + topo_key), \
         st.expander(tr("🎚️ Visual Tuning", "🎚️ 視覺化調諧"), expanded=False):
        # ── Backend status badges — show ALL active accelerators.
        #    When both CUDA and Rust are available, the Visual Tuning
        #    Live mode picks CUDA via `xp=cupy`, but the Plotly slider
        #    "Build animation" path (cache-less batched sim) can route
        #    through Rust.  Showing both lets the user know what's
        #    available, not just which one wins per click.
        from ..helpers.rust_kernels import (
            HAS_RUST as _HAS_RUST_BACKEND,
            rust_diagnostic as _rust_diag,
        )
        _chips = []
        if _HAS_CUDA:
            _chips.append(
                "<span style='background:#e3f2fd;color:#0d47a1;"
                f"padding:2px 8px;border-radius:4px;font-size:0.8em;"
                f"font-weight:600'>⚡ CUDA (cupy {_CUDA_VER})</span>")
        if _HAS_RUST_BACKEND:
            _chips.append(
                "<span style='background:#fff3e0;color:#e65100;"
                "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                "font-weight:600'>🦀 Rust kernels</span>")
        if not _chips:
            _chips.append(
                "<span style='background:#eceff1;color:#37474f;"
                "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                "font-weight:600'"
                f" title='{tr('Build the Rust crate for ~10x speedup.'
                             ' See tools/SSM/rust_kernels/README.md.',
                             '建置 Rust crate 可加速約 10 倍。'
                             '詳見 tools/SSM/rust_kernels/README.md。')}'"
                f">🐢 {tr('NumPy fallback', 'NumPy 備援')}</span>")
        st.markdown(
            tr("Backends", "運算後端") + ": " + "  ".join(_chips),
            unsafe_allow_html=True)

        # ── Diagnostic: when the binary exists on disk but Rust didn't
        #    load, surface the actual import error inside an expander so
        #    the user doesn't have to dig through the terminal.
        if not _HAS_RUST_BACKEND:
            _d = _rust_diag()
            if _d["binary_files"] and not _d["force_numpy"]:
                with st.expander(tr("🛠️ Why is Rust not active?",
                                    "🛠️ 為什麼 Rust 未啟用？"), expanded=False):
                    st.code(
                        f"arch_tag       : {_d['arch_tag']}\n"
                        f"bin_dir        : {_d['bin_dir']}\n"
                        f"bin_dir_exists : {_d['bin_dir_exists']}\n"
                        f"binary_files   : {_d['binary_files']}\n"
                        f"import_error   : {_d['import_error']}\n"
                        f"force_numpy    : {_d['force_numpy']}\n",
                        language="text")
                    st.caption(tr(
                        "A binary exists on disk but couldn't be imported. "
                        "Most common cause: the Streamlit process was started "
                        "*before* the binary was placed in this folder — "
                        "restart the launcher (`python LAUNCH_Tool.py`) to "
                        "pick it up.  If the import error persists after a "
                        "fresh restart, the .pyd may be from a different ABI; "
                        "delete it and run `python rust_things/build_rust_kernels.py`.",
                        "磁碟上存在二進位檔，但無法匯入。"
                        "最常見的原因：Streamlit 程序是在二進位檔放入此資料夾"
                        "*之前*啟動的 — 請重新啟動啟動器"
                        "（`python LAUNCH_Tool.py`）以載入。"
                        "若重新啟動後匯入錯誤仍持續發生，"
                        "該 .pyd 可能來自不同的 ABI；"
                        "請刪除後執行 `python rust_things/build_rust_kernels.py`。"))
        _render_slider_preview(model_cls, all_p, S_raw, freq, z0,
                               tuning_specs, fname, topo_key)


def render_tuning_expander(model_cls, all_p, S_raw, freq, z0,
                           tuning_specs, fname, topo_key,
                           *, default_fit_keys=None):
    """🔧 Auto Tuning for Minimum Residuals — grid sweep + residual ranking.

    Renders the full sweep-grid / brute-force / optimized / prioritize /
    minimize-deviation toolset, plus a 🪜 Progressive auto-fit card (coarse →
    fine, physics-informed ranges).  Kept as ``render_tuning_expander`` for
    call-site stability (still imported under that name); the visible
    label is "Auto Tuning for Minimum Residuals".

    Parameters
    ----------
    model_cls : AbstractSSMModel subclass with .simulate(params, freq, z0)
    all_p     : dict — current parameter values in SI units
    S_raw     : np.ndarray — measured S-parameters (N, 2, 2)
    freq      : np.ndarray — frequency array
    z0        : float
    tuning_specs : list of (key, label, scale, unit, ...) tuples
                   scale converts SI → display unit (e.g. 1e15 for fF)
    fname     : str — filename for unique widget keys
    topo_key  : str — topology short name for unique widget keys
    default_fit_keys : iterable of spec keys, optional
        Spec keys pre-selected in the 🪜 Progressive auto-fit "Parameters to
        fit" multiselect.  When None, defaults to every spec whose canonical
        key is NOT a parasitic (pads / leads) — so intrinsic + extrinsic +
        access params (and unknown custom params) are on by default.
    """
    import pandas as pd

    # Physics-informed default sweep ranges (and the low-perf flag they key on)
    # are used at every seeding site below; cache the low-perf detection once
    # per (topo, file) so we don't recompute fT/fmax on every rerun.
    _lp_key = f"tune_lowperf_{topo_key}_{fname}"
    if _lp_key not in st.session_state:
        st.session_state[_lp_key] = _detect_low_perf_device(S_raw, freq)
    _low_perf = bool(st.session_state[_lp_key])

    with st.container(key="hbt_exp_tune_auto_" + topo_key), \
         st.expander(tr("🔧 Auto Tuning for Minimum Residuals",
                        "🔧 自動調諧至最小殘差"), expanded=False):

        # ── Compute backend badge — cached in session_state so the
        #    label stays stable across reruns.  Without the cache the
        #    badge sometimes flipped Rust→NumPy after a Stop/cancel
        #    rerun (re-import races on `hbt_rust_kernels` after the
        #    sweep loop unwinds); caching guarantees the badge reflects
        #    a one-time, definitive detection.
        _BADGE_KEY = f"_rust_active_cached_{model_cls.SHORT}_{fname}"

        def _detect_rust_active() -> bool:
            try:
                from ..helpers.rust_kernels import (
                    HAS_RUST as _HR,
                    _phase2_dispatch_enabled as _phase2_on,
                    SIM_FOR_TOPOLOGY as _SIM_TOPO,
                )
                from ..helpers import rust_kernels as _RKMOD
            except Exception:                              # pragma: no cover
                return False
            if not (_HR and _phase2_on()):
                return False
            # Built-in fixed-topology end-to-end kernels.
            if _SIM_TOPO.get(model_cls.SHORT) is not None:
                return True
            # Custom model — the data-driven `sim_custom_batch` kernel isn't in
            # SIM_FOR_TOPOLOGY; it's active when the adapter opts in *and* the
            # compiled symbol is present in the loaded binary.
            if getattr(model_cls, "USES_RUST_BATCH", False):
                return getattr(getattr(_RKMOD, "_rk", None),
                               "sim_custom_batch", None) is not None
            return False

        # Sticky-True caching: re-evaluate every render, but never flip
        # an active=True back to False within the session.  This keeps
        # the original "no Rust→NumPy flicker after Stop" guarantee
        # *and* allows the badge to flip from stale-False → True the
        # moment the Rust binary becomes importable (e.g. user just
        # finished building it without restarting Streamlit).
        _now_active = _detect_rust_active()
        _prev = bool(st.session_state.get(_BADGE_KEY, False))
        st.session_state[_BADGE_KEY] = bool(_now_active or _prev)
        _rust_active_here = st.session_state[_BADGE_KEY]
        if _HAS_CUDA and _rust_active_here:
            _bk = ("<span style='background:#e3f2fd;color:#0d47a1;"
                   "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                   "font-weight:600'"
                   f" title='{tr('CUDA buttons use cupy; CPU buttons use the Rust kernel.',
                                 'CUDA 按鈕使用 cupy；CPU 按鈕使用 Rust 核心。')}'"
                   f">⚡ {tr('CUDA available', 'CUDA 可用')} + 🦀 {tr('Rust active', 'Rust 已啟用')}</span>")
        elif _HAS_CUDA:
            _bk = ("<span style='background:#e3f2fd;color:#0d47a1;"
                   "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                   "font-weight:600'"
                   f" title='{tr('CUDA buttons use cupy; CPU buttons use NumPy.'
                                 ' Set HBT_USE_RUST_SIM_BATCH=1 to enable Rust on the'
                                 ' CPU path for ~25x faster sweeps.',
                                 'CUDA 按鈕使用 cupy；CPU 按鈕使用 NumPy。'
                                 '設定 HBT_USE_RUST_SIM_BATCH=1 可在 CPU 路徑'
                                 '啟用 Rust，掃描速度提升約 25 倍。')}'"
                   f">⚡ {tr('CUDA available', 'CUDA 可用')}</span>")
        elif _rust_active_here:
            _bk = ("<span style='background:#fff3e0;color:#e65100;"
                   "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                   "font-weight:600'"
                   f" title='{tr('CPU buttons use the Rust kernel.',
                                 'CPU 按鈕使用 Rust 核心。')}'"
                   f">🦀 {tr('Rust active', 'Rust 已啟用')}</span>")
        else:
            _bk = ("<span style='background:#eceff1;color:#37474f;"
                   "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                   "font-weight:600'"
                   f" title='{tr('CPU buttons use NumPy. Build the Rust crate'
                                 ' (python rust_things/build_rust_kernels.py) and set'
                                 ' HBT_USE_RUST_SIM_BATCH=1 for ~25x speedup.',
                                 'CPU 按鈕使用 NumPy。建置 Rust crate'
                                 '（python rust_things/build_rust_kernels.py）'
                                 '並設定 HBT_USE_RUST_SIM_BATCH=1 可提升約 25 倍速度。')}'"
                   ">🐢 NumPy</span>")
        st.markdown(f"{tr('Compute backend', '運算後端')}: {_bk}", unsafe_allow_html=True)

        # ── Mode-card shared bits — used by the Full Auto Tune card here
        #    AND the Semi-Auto cards further below, so they live above both.
        _cpu_tag = "🦀 Rust" if _rust_active_here else "🐢 NumPy"
        # The backend (Rust vs NumPy) is no longer shown on the CPU button
        # face — it's surfaced in the hover tooltip instead.
        _cpu_help_suffix = f"\n\n{tr('CPU backend', 'CPU 後端')}: {_cpu_tag}."

        # Modes share a 2-column inner grid (CPU left, CUDA right).
        # When no CUDA is available the right column is dropped.
        def _action_cols():
            return st.columns(2) if _HAS_CUDA else (st.columns(1)[0],
                                                     st.empty())

        # ── 🪜 Full Auto Tune (recommended) — first card under the badge ──
        # Global coarse scan from physics-informed ranges, then pair-wise
        # (Zbe / Zbc / …) refinement with shrinking steps.  Never
        # re-evaluates a combo; keeps the best result on Stop.
        _prog_label_map = {}
        for _s in tuning_specs:
            _u = _s[3] if len(_s) > 3 else ""
            _prog_label_map[_s[0]] = f"{_s[1]} ({_u})" if _u else _s[1]
        _prog_opt_keys = [s[0] for s in tuning_specs]

        # Default selection: caller-supplied keys, else all non-parasitic
        # (canonical) keys.  Intersect with the actual option keys so
        # st.multiselect never sees an unknown default.
        if default_fit_keys is None:
            _prog_default = [
                s[0] for s in tuning_specs
                if _canonical_tune_key(s[0], s[1]) not in _PARASITIC_KEYS]
        else:
            _prog_default = list(default_fit_keys)
        _prog_default = [k for k in _prog_default if k in _prog_opt_keys]

        with st.container(border=True):
            st.markdown(
                f"**🪜 {tr('Full Auto Tune', '全自動調諧')}** "
                f"<span style='color:#666;font-size:0.85em'>({tr('recommended', '建議')})</span>"
                f" <span class='hbt-help' title='{tr(
                    'Coarse-to-fine: global scan'
                    ' from physics-informed ranges, then pair-wise refinement with'
                    ' shrinking steps. Best result is kept when you press Stop.'
                    ' Deselect a parameter to pin it at its current value'
                    ' (including 0).',
                    '由粗到細：先依物理範圍進行全域掃描，'
                    '再以逐漸縮小的步進進行成對細化。'
                    '按下停止時會保留最佳結果。'
                    '取消勾選某參數會將其固定在目前值（含 0）。')}'>?</span>",
                unsafe_allow_html=True)
            _prog_scope_key = f"tune_prog_scope_{topo_key}_{fname}"
            # Once the widget owns its state, use that as the default (matches
            # the slider-preview multiselect pattern) so Streamlit doesn't warn
            # about a keyed widget also carrying a fixed default.
            _prog_seed = st.session_state.get(_prog_scope_key, _prog_default)
            _prog_seed = [k for k in _prog_seed if k in _prog_opt_keys]
            prog_fit_keys = st.multiselect(
                tr("Parameters to fit", "要擬合的參數"),
                options=_prog_opt_keys,
                default=_prog_seed,
                format_func=lambda k: _prog_label_map.get(k, k),
                key=_prog_scope_key,
                placeholder=tr("Choose options", "請選擇項目"))
            _prog_help = tr(
                "Coarse global scan from the physics-informed ranges, then "
                "pair-wise refinement with shrinking steps. Memoises every "
                "combo so no work is repeated; keeps the best result on "
                "Stop. Fitted parameters are floored above 0 — deselect one "
                "to pin it at its current value instead.",
                "先依物理範圍進行粗略全域掃描，再以逐漸縮小的步進進行成對細化。"
                "每個組合都會被記錄，不會重複計算；按下停止時保留最佳結果。"
                "擬合參數下限為 0 — 取消勾選可改為固定在目前值。")
            _c_cpu, _c_cuda = _action_cols()
            prog_cpu_clicked = _c_cpu.button(
                tr("Evaluate with CPU", "以 CPU 評估"),
                key=f"tune_calc_prog_{topo_key}_{fname}",
                help=_prog_help + _cpu_help_suffix,
                width="stretch")
            prog_cuda_clicked = (
                _HAS_CUDA
                and _c_cuda.button(
                    tr("⚡ Evaluate with CUDA", "⚡ 以 CUDA 評估"),
                    key=f"tune_calc_prog_cuda_{topo_key}_{fname}",
                    help=_prog_help,
                    width="stretch"))

        # ── Pre-initialize session state defaults (only on first render of
        #    each key).  MUST stay outside the Semi-Auto toggle: every driver
        #    (Full Auto included) and the closed-path row build below read
        #    these keys.
        for spec in tuning_specs:
            key, label, scale = spec[0], spec[1], spec[2]
            spec_step = spec[5] if len(spec) > 5 else None
            current_si = float(all_p.get(key, 0.0))
            current_disp = current_si * scale
            is_zero = abs(current_si) < 1e-30
            kp = f"tune_{topo_key}_{key}_{fname}"
            if f"{kp}_chk" not in st.session_state:
                d_min, d_step, d_max = informed_default_range(
                    key, label, current_disp,
                    low_perf=_low_perf, spec_step=spec_step)
                st.session_state[f"{kp}_chk"]  = not is_zero
                st.session_state[f"{kp}_min"]  = d_min
                st.session_state[f"{kp}_step"] = d_step
                st.session_state[f"{kp}_max"]  = d_max

        # ── Per-parameter sweep-row data — ONE source of truth ──────────
        # param_rows feeds every driver (_run_one_sweep / _run_nelder_mead /
        # _run_progressive col-names + constant fill) and the sensitivity
        # section, so it must exist on every run whether or not the
        # Semi-Auto section below is open.  The row widgets are keyed, so
        # the closed path reads the very same session keys the widgets
        # would return — identical values by construction.
        def _clamp_row_session(kp, h_lo, h_hi):
            """Pre-clamp stale session Min/Max into the hard limits (the
            keyed number_inputs would otherwise raise
            StreamlitAPIException) and floor Step at 0.  Runs in BOTH the
            open (widget) and closed (session-read) paths."""
            for _sfx in ("_min", "_max"):
                _sk = f"{kp}{_sfx}"
                if _sk in st.session_state:
                    _v = float(st.session_state[_sk])
                    _v = max(_v, h_lo)
                    if h_hi is not None:
                        _v = min(_v, float(h_hi))
                    st.session_state[_sk] = _v
            _step_sk = f"{kp}_step"
            if _step_sk in st.session_state:
                st.session_state[_step_sk] = max(
                    0.0, float(st.session_state[_step_sk]))

        def _row_entry(spec, enabled, min_val, step_val, max_val):
            """Build one param_rows dict — the single place sweep/n_calc
            come from, so the open and closed paths can never diverge."""
            key, label, scale = spec[0], spec[1], spec[2]
            unit = spec[3] if len(spec) > 3 else ""
            if enabled:
                sweep = _make_sweep_values(min_val, max_val, step_val)
            else:
                sweep = np.array([float(all_p.get(key, 0.0)) * scale])
            return {"key": key, "label": label, "scale": scale, "unit": unit,
                    "enabled": enabled, "sweep": sweep, "n_calc": len(sweep)}

        # Mode-click flags must exist even when the Semi-Auto section is
        # closed (the dispatch below reads all of them) — default everything
        # False / neutral and let the open branch's widgets overwrite.
        cpu_clicked = cuda_clicked = False
        opt_cpu_clicked = opt_cuda_clicked = False
        prio_cpu_clicked = prio_cuda_clicked = False
        bal_cpu_clicked = bal_cuda_clicked = False
        bal_dev_threshold = 3.0
        _bal_use_res = False
        bal_res_threshold = 5.0
        prio_metric = st.session_state.get(
            f"tune_prio_metric_{topo_key}_{fname}", "S11")

        # ── 🔬 Semi-Auto Tune (toggle-collapsed section) ────────────────
        # Streamlit forbids nesting st.expander, so the collapsible section
        # is a toggle-gated bordered container instead (toggle off =
        # collapsed).
        param_rows = []
        semi_open = st.toggle(
            tr("🔬 Semi-Auto Tune", "🔬 半自動調諧"),
            key=f"tune_semi_open_{topo_key}_{fname}",
            help=tr("Manual sweep grid: pick parameters, set Min/Step/Max, "
                    "then run Brute force / Optimized / Prioritized passes.",
                    "手動掃描表格：選擇參數、設定最小值/步進/最大值，"
                    "再執行暴力法／最佳化／優先化計算。"))
        if semi_open:
            with st.container(border=True):
                # ── Toolbar: Use default values + Select all + De-select ──
                # All three buttons live above the table so users hit the
                # bulk actions before scanning per-row.  Each writes to
                # session_state and reruns so the table picks up the new
                # values on the next render pass.
                _tb1, _tb2, _tb3, _ = st.columns([1.0, 1.0, 1.0, 3.0])
                if _tb1.button(tr("↩️ Use default values", "↩️ 使用預設值"),
                                key=f"tune_defaults_{topo_key}_{fname}"):
                    for spec in tuning_specs:
                        key, label, scale = spec[0], spec[1], spec[2]
                        spec_step = spec[5] if len(spec) > 5 else None
                        current_disp = float(all_p.get(key, 0.0)) * scale
                        kp = f"tune_{topo_key}_{key}_{fname}"
                        d_min, d_step, d_max = informed_default_range(
                            key, label, current_disp,
                            low_perf=_low_perf, spec_step=spec_step)
                        st.session_state[f"{kp}_min"] = d_min
                        st.session_state[f"{kp}_step"] = d_step
                        st.session_state[f"{kp}_max"] = d_max
                    st.rerun()
                if _tb2.button(tr("✅ Select all", "✅ 全選"),
                                key=f"tune_select_all_{topo_key}_{fname}"):
                    for spec in tuning_specs:
                        key = spec[0]
                        st.session_state[
                            f"tune_{topo_key}_{key}_{fname}_chk"] = True
                    st.rerun()
                if _tb3.button(tr("❌ De-select all", "❌ 全部取消"),
                                key=f"tune_deselect_all_{topo_key}_{fname}"):
                    for spec in tuning_specs:
                        key = spec[0]
                        st.session_state[
                            f"tune_{topo_key}_{key}_{fname}_chk"] = False
                    st.rerun()

                # Column headers
                hdr = st.columns([0.5, 1.5, 1.2, 1.2, 1.2, 1.0])
                hdr[0].markdown(f"**{tr('Sweep', '掃描')}**")
                hdr[1].markdown(f"**{tr('Parameter', '參數')}**")
                hdr[2].markdown(f"**{tr('Min', '最小值')}**")
                hdr[3].markdown(f"**{tr('Step', '步進')}**")
                hdr[4].markdown(f"**{tr('Max', '最大值')}**")
                hdr[5].markdown(f"**{tr('# Calc', '計算數')}**")

                # Per-parameter rows — widget path
                for spec in tuning_specs:
                    key, label, scale = spec[0], spec[1], spec[2]
                    unit = spec[3] if len(spec) > 3 else ""
                    current_disp = float(all_p.get(key, 0.0)) * scale
                    kp = f"tune_{topo_key}_{key}_{fname}"

                    cols = st.columns([0.5, 1.5, 1.2, 1.2, 1.2, 1.0])
                    enabled = cols[0].checkbox(
                        f"sweep_{key}", value=False, key=f"{kp}_chk",
                        label_visibility="collapsed")
                    cur_str = f"{current_disp:.2f} {unit}".strip()
                    cols[1].markdown(
                        f"**{label}** ({unit}) ({cur_str})" if unit
                        else f"**{label}** ({cur_str})")

                    if enabled:
                        # Hard physical limits: floor Min/Max at (h_lo,
                        # h_hi) — pads can't go negative, alpha0 is boxed
                        # to [0.95, 0.99], etc.
                        h_lo, h_hi = tune_hard_limits(key, label)
                        h_lo = float(h_lo)
                        _clamp_row_session(kp, h_lo, h_hi)
                        _min_kwargs = {"min_value": h_lo}
                        _max_kwargs = {"min_value": h_lo}
                        if h_hi is not None:
                            _min_kwargs["max_value"] = float(h_hi)
                            _max_kwargs["max_value"] = float(h_hi)
                        min_val = cols[2].number_input(
                            tr("Min", "最小值"), format="%.5g", key=f"{kp}_min",
                            label_visibility="collapsed", **_min_kwargs)
                        step_val = cols[3].number_input(
                            tr("Step", "步進"), format="%.5g", key=f"{kp}_step",
                            min_value=0.0, label_visibility="collapsed")
                        max_val = cols[4].number_input(
                            tr("Max", "最大值"), format="%.5g", key=f"{kp}_max",
                            label_visibility="collapsed", **_max_kwargs)
                    else:
                        min_val = current_disp
                        step_val = 0.0
                        max_val = current_disp
                        cols[2].markdown(f"{current_disp:.2f}")
                        cols[3].markdown("0")
                        cols[4].markdown(f"{current_disp:.2f}")

                    row = _row_entry(spec, enabled,
                                     min_val, step_val, max_val)
                    cols[5].markdown(f"**{row['n_calc']}**")
                    param_rows.append(row)

                # Total calculation count
                total_calcs = 1
                for row in param_rows:
                    if row["enabled"]:
                        total_calcs *= row["n_calc"]
                    # unchecked params contribute 1 (single current value)

                st.markdown(f"**{tr('Total calculations', '總計算數')}: {total_calcs:,}**")

                if total_calcs > 500_000:
                    st.warning(tr("More than 500,000 combinations — this may "
                                  "take a long time.",
                                  "超過 500,000 種組合 — 可能需要很長時間。"))

                # ── CUDA availability note ──────────────────────────────
                if _HAS_CUDA:
                    st.caption(tr(f"CUDA {_CUDA_VER} detected — GPU "
                                  "acceleration available",
                                  f"偵測到 CUDA {_CUDA_VER} — GPU 加速可用"))

                # ── Grid-mode cards ── Brute force / Optimized / Prioritized
                #    (min-dev stays hidden behind _SHOW_MIN_DEV).  Flags were
                #    pre-neutralised above, so no else-branches are needed here.
                # ── 🧮 Brute-force all combos ───────────────────────────────────
                _calc_all_help = tr(
                    "Evaluates EVERY combination in the sweep grid above and ranks "
                    "the top-100 results by lowest TOTAL residual.",
                    "評估上方掃描表格中的每一種組合，並依最低總殘差排出前 100 名。")
                with st.container(border=True):
                    st.markdown(
                        f"**🧮 {tr('Brute force — all combos', '暴力法 — 所有組合')}**"
                        f" <span class='hbt-help' title='{tr(
                            'Evaluate every'
                            ' combination in the grid, rank by total residual.',
                            '評估表格中每一種組合，並依總殘差排序。')}'"
                        ">?</span>",
                        unsafe_allow_html=True)
                    _c_cpu, _c_cuda = _action_cols()
                    cpu_clicked = _c_cpu.button(
                        tr("Brute force with CPU", "以 CPU 執行暴力法"),
                        key=f"tune_calc_{topo_key}_{fname}",
                        help=_calc_all_help + _cpu_help_suffix,
                        width="stretch")
                    cuda_clicked = (
                        _HAS_CUDA
                        and _c_cuda.button(
                            tr("⚡ Brute force with CUDA", "⚡ 以 CUDA 執行暴力法"),
                            key=f"tune_calc_cuda_{topo_key}_{fname}",
                            help=_calc_all_help,
                            width="stretch"))

                # ── 🎯 Optimized recursive bisection ────────────────────────────
                _opt_help = tr(
                    "Repeatedly subsamples 5 evenly-spaced values per swept parameter "
                    "(e.g. min=1, step=1, max=100 → 1, 25, 50, 75, 100), picks the 2 "
                    "combos with the lowest total residual, then narrows the search "
                    "box to those two values and recurses.  Iteration stops once a "
                    "brute-force sweep at the user's chosen step would fit ≤ 5,000,000 "
                    "combos, and that final refinement is run as the closing pass.",
                    "對每個掃描參數重複取 5 個等間距值進行子取樣"
                    "（例如 min=1, step=1, max=100 → 1, 25, 50, 75, 100），"
                    "挑出總殘差最低的 2 個組合，將搜尋範圍縮小至該兩值後遞迴。"
                    "當以使用者所選步進進行暴力掃描的組合數 ≤ 5,000,000 時停止疊代，"
                    "並以該次細化作為最終回合。")
                with st.container(border=True):
                    st.markdown(
                        f"**🎯 {tr('Optimized — recursive bisection', '最佳化 — 遞迴二分法')}**"
                        f" <span class='hbt-help' title='{tr(
                            'Iteratively narrows the'
                            ' search box: 5-point subsample, keep top-2, recurse'
                            ' until the final pass fits at most 5 million combos.',
                            '逐步縮小搜尋範圍：5 點子取樣，保留前 2 名，'
                            '遞迴直到最終回合組合數不超過 500 萬。')}'"
                        ">?</span>",
                        unsafe_allow_html=True)
                    _c_cpu, _c_cuda = _action_cols()
                    opt_cpu_clicked = _c_cpu.button(
                        tr("Optimized with CPU", "以 CPU 執行最佳化"),
                        key=f"tune_calc_opt_{topo_key}_{fname}",
                        help=_opt_help + _cpu_help_suffix,
                        width="stretch")
                    opt_cuda_clicked = (
                        _HAS_CUDA
                        and _c_cuda.button(
                            tr("⚡ Optimized with CUDA", "⚡ 以 CUDA 執行最佳化"),
                            key=f"tune_calc_opt_cuda_{topo_key}_{fname}",
                            help=_opt_help,
                            width="stretch"))

                # ── 🥇 Prioritized by single S-parameter ────────────────────────
                _prio_help = tr(
                    "Runs the same full-grid sweep but ranks results by the residual "
                    "of a single chosen S-parameter instead of the total.",
                    "執行相同的全網格掃描，但改依單一選定 S 參數的殘差排序，"
                    "而非總殘差。")
                with st.container(border=True):
                    st.markdown(
                        f"**🥇 {tr('Prioritized — rank by one S-parameter', '優先排序 — 依單一 S 參數排序')}**"
                        f" <span class='hbt-help' title='{tr(
                            'Same full-grid sweep,'
                            ' but the top-100 selection is sorted by a single'
                            ' S-parameter residual rather than the total.',
                            '相同的全網格掃描，但前 100 名的排序改依'
                            '單一 S 參數殘差，而非總殘差。')}'"
                        ">?</span>",
                        unsafe_allow_html=True)
                    _prio_row = st.columns([0.6, 2])
                    _prio_row[0].markdown(
                        f"<div style='padding-top:0.4em'>"
                        f"<small><b>{tr('Sort by', '排序依據')}</b></small></div>",
                        unsafe_allow_html=True)
                    prio_metric = _prio_row[1].radio(
                        tr("Prioritize sort metric", "優先排序指標"),
                        ["S11", "S12", "S21", "S22"],
                        horizontal=True,
                        label_visibility="collapsed",
                        key=f"tune_prio_metric_{topo_key}_{fname}")
                    _c_cpu, _c_cuda = _action_cols()
                    prio_cpu_clicked = _c_cpu.button(
                        tr("Prioritized with CPU", "以 CPU 執行優先排序"),
                        key=f"tune_calc_prio_{topo_key}_{fname}",
                        help=_prio_help + _cpu_help_suffix,
                        width="stretch")
                    prio_cuda_clicked = (
                        _HAS_CUDA
                        and _c_cuda.button(
                            tr("⚡ Prioritized with CUDA", "⚡ 以 CUDA 執行優先排序"),
                            key=f"tune_calc_prio_cuda_{topo_key}_{fname}",
                            help=_prio_help,
                            width="stretch"))

                # ── 🎚️  Minimize deviation (HIDDEN for now — to re-enable, flip
                #    `_SHOW_MIN_DEV` to True.  All the underlying logic at the
                #    `bal_cpu_clicked or bal_cuda_clicked` dispatch site below
                #    stays wired up; the only thing the toggle gates is the
                #    UI section.).
                _SHOW_MIN_DEV = False
                if _SHOW_MIN_DEV:
                    _bal_help = tr(
                        "Runs the full-grid sweep but discards combos whose per-port "
                        "residuals are unbalanced (peak-to-peak spread > threshold).  "
                        "Surviving balanced combos are then ranked by lowest total "
                        "residual.  Optionally also drop combos where any single port "
                        "residual exceeds a quality floor.",
                        "執行全網格掃描，但捨棄各埠殘差不平衡的組合"
                        "（峰對峰差 > 閾值）。倖存的平衡組合再依最低總殘差排序。"
                        "亦可選擇捨棄任一埠殘差超過品質下限的組合。")
                    with st.container(border=True):
                        st.markdown(
                            f"**🎚️ {tr('Minimize deviation — peak-to-peak filter', '最小化偏差 — 峰對峰篩選')}**  "
                            f"<span style='color:#666;font-size:0.85em'>"
                            f"{tr('Discard unbalanced combos, then rank survivors by '
                                 'total residual.',
                                 '捨棄不平衡的組合，再依總殘差排序倖存者。')}</span>",
                            unsafe_allow_html=True)
                        _bal_inputs = st.columns([1, 1, 2])
                        bal_dev_threshold = _bal_inputs[0].number_input(
                            tr("Max per-port deviation (%)", "每埠最大偏差 (%)"),
                            min_value=0.0, max_value=100.0,
                            value=float(st.session_state.get(
                                f"tune_bal_dev_{topo_key}_{fname}", 3.0)),
                            step=0.5, format="%.2f",
                            key=f"tune_bal_dev_{topo_key}_{fname}",
                            help=tr("Max allowed |max(S11,S12,S21,S22) − min(...)| residual "
                                    "spread (in %).  Smaller = more balanced.",
                                    "允許的最大 |max(S11,S12,S21,S22) − min(...)| 殘差"
                                    "分佈（%）。數值越小越平衡。"))
                        _bal_use_res = _bal_inputs[1].checkbox(
                            tr("Use residual cap", "使用殘差上限"),
                            value=False,
                            key=f"tune_bal_use_res_{topo_key}_{fname}")
                        bal_res_threshold = _bal_inputs[2].number_input(
                            tr("Max per-port residual (%)", "每埠最大殘差 (%)"),
                            min_value=0.0, max_value=100.0,
                            value=float(st.session_state.get(
                                f"tune_bal_res_{topo_key}_{fname}", 5.0)),
                            step=0.5, format="%.2f",
                            key=f"tune_bal_res_{topo_key}_{fname}",
                            disabled=(not _bal_use_res),
                            help=tr("Quality floor — drop combos where any port's residual "
                                    "exceeds this.",
                                    "品質下限 — 捨棄任一埠殘差超過此值的組合。"))
                        _c_cpu, _c_cuda = _action_cols()
                        bal_cpu_clicked = _c_cpu.button(
                            f"CPU — {_cpu_tag}",
                            key=f"tune_calc_bal_{topo_key}_{fname}",
                            help=_bal_help,
                            width="stretch")
                        bal_cuda_clicked = (
                            _HAS_CUDA
                            and _c_cuda.button(
                                "⚡ CUDA",
                                key=f"tune_calc_bal_cuda_{topo_key}_{fname}",
                                help=_bal_help,
                                width="stretch"))
        else:
            # Semi-Auto collapsed — the row widgets aren't rendered, but every
            # driver (Full Auto included) still needs param_rows.  Read the
            # same keyed session state the widgets would return; _row_entry is
            # the shared source of truth for sweep/n_calc, so the two paths
            # cannot diverge.  (The pre-init seeding loop above guarantees the
            # _chk/_min/_step/_max keys exist.)
            for spec in tuning_specs:
                key, label, scale = spec[0], spec[1], spec[2]
                kp = f"tune_{topo_key}_{key}_{fname}"
                h_lo, h_hi = tune_hard_limits(key, label)
                _clamp_row_session(kp, float(h_lo), h_hi)
                enabled = bool(st.session_state.get(f"{kp}_chk", False))
                current_disp = float(all_p.get(key, 0.0)) * scale
                if enabled:
                    min_val  = float(st.session_state.get(
                        f"{kp}_min", current_disp))
                    step_val = float(st.session_state.get(f"{kp}_step", 0.0))
                    max_val  = float(st.session_state.get(
                        f"{kp}_max", current_disp))
                else:
                    min_val, step_val, max_val = (current_disp, 0.0,
                                                  current_disp)
                param_rows.append(
                    _row_entry(spec, enabled, min_val, step_val, max_val))

        # # ── Auto (Nelder-Mead) ──────────────────────────────────────────
        # st.caption(
        #     "🤖 **Auto (Nelder-Mead)** seeds the simplex from the current "
        #     "parameter values, expands/contracts adaptively (large moves when "
        #     "residual is high, small moves when low), and converges to a "
        #     "local minimum within the **Min/Max** boxes above.  Cheap, "
        #     "gradient-free, and respects the Stop button.  Most useful as a "
        #     "polish step after a coarse grid sweep.")
        # _auto_inputs = st.columns([1, 1, 2])
        # auto_max_iter = int(_auto_inputs[0].number_input(
        #     "Max iterations",
        #     min_value=10, max_value=10_000,
        #     value=int(st.session_state.get(
        #         f"tune_auto_iter_{topo_key}_{fname}", 200)),
        #     step=10,
        #     key=f"tune_auto_iter_{topo_key}_{fname}",
        #     help="Nelder-Mead iteration cap (function evals ≈ iters × (n+1))."))
        # auto_restart = _auto_inputs[1].checkbox(
        #     "Random restart on collapse",
        #     value=True,
        #     key=f"tune_auto_restart_{topo_key}_{fname}",
        #     help="If the simplex shrinks below tol before convergence, "
        #          "perturb x0 and run a second pass.")
        # _auto_btn = _auto_inputs[2].columns([1, 1] if _HAS_CUDA else [1])
        # auto_cpu_clicked = _auto_btn[0].button(
        #     "🤖 Auto (Nelder-Mead)",
        #     key=f"tune_calc_auto_{topo_key}_{fname}")
        # auto_cuda_clicked = (
        #     _HAS_CUDA
        #     and _auto_btn[1].button(
        #         "⚡ Auto with CUDA",
        #         key=f"tune_calc_auto_cuda_{topo_key}_{fname}"))

        # ── Helper: format a "best result" markdown line including all params ──
        #
        # • Bold the parameter name AND value for any param that is currently
        #   ticked for sweep (its `_chk` session key is True).
        # • Hide all-zero pad caps (Cpbe/Cpce/Cpbc) and lead inductances
        #   (Lb/Lc/Le) since they're often left at 0 and just add noise.
        _HIDE_IF_ZERO = {"Cpbe", "Cpce", "Cpbc", "Lb", "Lc", "Le"}
        def _best_summary_md(best_row, label=None):
            if label is None:
                label = tr("Best so far", "目前最佳")
            head = (f"**{label} — {tr('Total', '總計')}: {best_row['Total Residual (%)']:.2f}%  |  "
                    f"S11: {best_row['S11 (%)']:.2f}%  S12: {best_row['S12 (%)']:.2f}%  "
                    f"S21: {best_row['S21 (%)']:.2f}%  S22: {best_row['S22 (%)']:.2f}%**")
            parts = []
            for spec in tuning_specs:
                key = spec[0]; label_p = spec[1]
                unit = spec[3] if len(spec) > 3 else ""
                fmt = spec[4] if len(spec) > 4 else "%.4g"
                col = f"{label_p} ({unit})" if unit else label_p
                if col not in best_row.index:
                    continue
                v = float(best_row[col])
                if key in _HIDE_IF_ZERO and np.isfinite(v) and abs(v) < 1e-30:
                    continue
                is_swept = bool(st.session_state.get(
                    f"tune_{topo_key}_{key}_{fname}_chk", False))
                sval = (fmt % v) if np.isfinite(v) else "NaN"
                name_md = f"<b>{label_p}</b>" if is_swept else label_p
                val_md  = (f"<b>{sval} {unit}</b>".rstrip()
                           if is_swept else f"{sval} {unit}".rstrip())
                parts.append(f"{name_md}: {val_md}")
            if parts:
                head += "  \n<small>" + ", ".join(parts) + "</small>"
            return head

        def _run_one_sweep(*, use_cuda: bool,
                            sweep_lists_override: dict | None = None,
                            sort_metric: str = "Total",
                            phase_label: str = "",
                            phase_suffix: str = "",
                            dev_threshold: float | None = None,
                            res_threshold: float | None = None):
            """Run one full parameter-sweep pass.

            sweep_lists_override : dict[str, np.ndarray] | None
                When provided, overrides the per-row sweep arrays for the listed
                keys. Other rows fall back to their checkbox/min/step/max state.
            sort_metric : "Total" | "S11" | "S12" | "S21" | "S22"
                Selects the residual metric used to rank the top-K results.
                Total residual / per-port residuals are still recorded for
                every kept combo regardless of choice.
            phase_label / phase_suffix
                phase_label is shown in the progress text; phase_suffix is
                appended to the Stop-button key so multi-pass runs (e.g.
                Optimized) don't clash on a duplicate Streamlit widget key.
            dev_threshold : float | None
                Max allowed peak-to-peak (max-min) per-port residual deviation
                in %.  Combos exceeding this are pushed to the bottom of the
                ranking — surviving combos are ranked normally by sort_metric.
                Used by the "Minimize deviation" button to pick the lowest
                total residual *among balanced* combinations.
            res_threshold : float | None
                Max allowed per-port residual in %.  Combos with any port
                exceeding this are pushed to the bottom of the ranking.
            """
            xp = _cp if use_cuda else np
            has_batch = hasattr(model_cls, "simulate_batch")
            _mode_label = "CUDA" if use_cuda else "CPU"

            # Only the top-K combos (lowest residuals) are kept in memory.
            # Saves >99% RAM on huge sweeps and side-steps MemoryError.
            TOP_K = 100

            # Build sweep lists (display units) — short host arrays of unique
            # values per parameter (length 1 for unswept params).
            sweep_keys, sweep_lists = [], []
            sweep_scales, sweep_labels, sweep_units = [], [], []
            for row in param_rows:
                sweep_keys.append(row["key"])
                sweep_scales.append(row["scale"])
                sweep_labels.append(row["label"])
                sweep_units.append(row["unit"])
                if sweep_lists_override and row["key"] in sweep_lists_override:
                    sweep_lists.append(np.asarray(
                        sweep_lists_override[row["key"]], dtype=np.float64))
                elif row["enabled"]:
                    sweep_lists.append(np.asarray(row["sweep"], dtype=np.float64))
                else:
                    sweep_lists.append(np.array(
                        [float(all_p.get(row["key"], 0.0)) * row["scale"]],
                        dtype=np.float64))

            n_params = len(sweep_keys)
            L_list = [int(len(sl)) for sl in sweep_lists]
            n_total = 1
            for L in L_list:
                n_total *= L

            # Strides for "ij"-order linear→multi-index decomposition:
            #   combo k → (i_0, i_1, …, i_{n-1})
            #   where i_p = (k // strides[p]) % L_list[p]
            strides = [1] * n_params
            for i in range(n_params - 2, -1, -1):
                strides[i] = strides[i + 1] * L_list[i + 1]

            # Indices of params that actually vary — only these need to be
            # carried per-row in the top-K state. Constants get filled in at
            # display time from sweep_lists.
            swept_idx_full = [i for i in range(n_params) if L_list[i] > 1]
            n_swept = len(swept_idx_full)

            # ── N-D parameter-sweep layout ──────────────────────────────────
            # Each *swept* parameter gets its own broadcast axis.  Sub-networks
            # whose inputs are not all swept are computed at their *native*
            # dimensionality and broadcast across the rest of the grid — so
            # `h/i` in `g·h/i` is built once with shape `(1,Lh,Li,1)` and
            # reused for every value of `g`, with no recomputation.
            #
            # Layout convention:
            #   axis 0 .. n_swept_dims-1  → one length-L_i axis per swept param
            #   axis n_swept_dims         → frequency (length N_freq)
            # Constants stay as Python scalars (broadcast to anything).
            #
            # Output S has shape  (L_0, L_1, ..., L_{n-1}, N_freq, 2, 2)
            # — flattened to (B_inner, N_freq, 2, 2) for residual scoring.

            swept_pos_to_param_idx = [i for i in range(n_params) if L_list[i] > 1]
            n_swept_dims = len(swept_pos_to_param_idx)
            inner_lens = [int(L_list[i]) for i in swept_pos_to_param_idx]

            # Lookup table of *display* values for index→display mapping
            L_max = max(max(L_list), 1)
            sweep_table_h = np.zeros((n_params, L_max), dtype=np.float64)
            for i, sl in enumerate(sweep_lists):
                sweep_table_h[i, :L_list[i]] = sl
            sweep_table_dev = xp.asarray(sweep_table_h)                            # (n_params, L_max)
            if n_swept > 0:
                swept_indices_dev = xp.asarray(np.asarray(swept_idx_full, dtype=np.int64))
            else:
                swept_indices_dev = None

            # Constant scalars passed straight into simulate_batch
            const_si = {sweep_keys[i]: float(sweep_lists[i][0]) / float(sweep_scales[i])
                        for i in range(n_params) if L_list[i] == 1}
            varies_mask = [L_list[i] > 1 for i in range(n_params)]
            swept_set = {sweep_keys[i] for i in range(n_params) if varies_mask[i]}

            # ── FP32 sweep mode (GPU only, model-opt-in) ─────────────────────
            # Cheng T/Pi's batched intrinsic-Y kernels honour cache["_cdtype"]
            # so the main loop can run entirely in complex64.  fp64 rerank
            # below replaces the residuals on the surviving top-K combos so
            # the final ranking and the persisted residuals are double-precision.
            #
            # CPU mode is left at fp64: scalar AVX is the same width for
            # both, and reduced precision saves no wall time on this code.
            use_fp32_sweep = bool(use_cuda and getattr(
                model_cls, "SUPPORTS_FP32_SWEEP", False))
            cdtype_main = np.complex64 if use_fp32_sweep else np.complex128
            rdtype_main = np.float32   if use_fp32_sweep else np.float64

            # Move S_mea onto the compute device once.  Two copies for fp32:
            # the cast (complex64) version drives the main loop, the original
            # complex128 stays alive for the fp64 rerank.
            S_mea_dev_fp64 = xp.asarray(S_raw)
            if use_fp32_sweep:
                S_mea_dev = S_mea_dev_fp64.astype(np.complex64)
            else:
                S_mea_dev = S_mea_dev_fp64

            # ── N-D omega: (1,)*n_swept_dims + (N_freq,) ────────────────────
            # Number of leading 1s = number of swept axes, so omega broadcasts
            # cleanly against any swept-param tensor regardless of which axes
            # it occupies.
            N_freq = len(freq)
            omega_shape = (1,) * n_swept_dims + (N_freq,)
            omega_dev = xp.asarray(
                2.0 * np.pi * np.asarray(freq, dtype=np.float64)
            ).reshape(omega_shape)

            # ── Pre-bake fully-constant sub-networks (lifted out of the loop)
            # If a sub-network has *zero* swept inputs, its tensor is built
            # once here at shape (1,...,1,N_freq) and reused every iteration.
            # If a sub-network has *some* swept inputs, the model code will
            # build it inside simulate_batch — but at its own native dims
            # (smaller than the full inner block), thanks to N-D broadcasting.
            _PAD_CAP_KEYS = ("Cpbe", "Cpce", "Cpbc")
            _SER_LEAD_KEYS = ("Rpb", "Rpc", "Rpe", "Lb", "Lc", "Le")
            _CHENG_EXTR_KEYS = ("Cbex", "Cbcx")
            # Cheng-T intrinsic sub-expression groups: each is pre-bakeable
            # whenever *none* of its inputs are swept.
            _T_ZBE_KEYS   = ("Rbe", "Cbe")
            _T_ZBC_KEYS   = ("Rbc", "Cbc")
            _T_ALPHA_KEYS = ("alpha0", "tauB", "tauC")
            _T_INT_ALL    = ("Rbi",) + _T_ZBE_KEYS + _T_ZBC_KEYS + _T_ALPHA_KEYS
            # Cheng-π intrinsic sub-expression groups
            _PI_YBE_KEYS  = ("Rbe", "Cbe")
            _PI_YBC_KEYS  = ("Rbc", "Cbc")
            _PI_GM_KEYS   = ("Gm0", "tau")
            _PI_INT_ALL   = ("Rbi",) + _PI_YBE_KEYS + _PI_YBC_KEYS + _PI_GM_KEYS

            static_p = dict(all_p)
            for _k, _v in const_si.items():
                static_p[_k] = _v

            # The fp64 cache is the canonical one — built first, then
            # cast to a parallel fp32 cache for the main loop if eligible.
            # The fp64 cache is also kept alive for the fp64 rerank below.
            static_cache_fp64 = {"omega": omega_dev}
            _cached_msgs = []
            if not (set(_PAD_CAP_KEYS) & swept_set):
                static_cache_fp64["Y_pad"] = build_Y_pad_batch(
                    static_p, omega_dev, 1, N_freq, xp)
                _cached_msgs.append("Y_pad")
            if not (set(_SER_LEAD_KEYS) & swept_set):
                static_cache_fp64["Z_ser"] = build_Z_ser_batch(
                    static_p, omega_dev, 1, N_freq, xp)
                _cached_msgs.append("Z_ser")
            if not (set(_CHENG_EXTR_KEYS) & swept_set):
                _Cbex_c = float(static_p.get("Cbex", 0.0))
                _Cbcx_c = float(static_p.get("Cbcx", 0.0))
                static_cache_fp64["Y_extr"] = (1j * omega_dev * _Cbex_c,
                                               1j * omega_dev * _Cbcx_c)
                _cached_msgs.append("Y_extr")

            # ── Cheng-specific intrinsic sub-expression caches ──────────────
            # These mirror what _Y_int_T_batch / _Y_int_Pi_batch will look up:
            # if a (Rbe,Cbe) / (Rbc,Cbc) / (alpha0,tauB,tauC) pair has all
            # constant inputs we can pre-build the result once and reuse it
            # every chunk.  When *every* intrinsic param is constant we go
            # one step further and pre-build the four intrinsic-Y planes
            # outright, skipping the entire per-chunk inv() of Z_in.
            _model_short = getattr(model_cls, "SHORT", "")
            if _model_short == "T":
                if not (set(_T_ZBE_KEYS) & swept_set):
                    _Rbe_c = float(static_p.get("Rbe", 1.0))
                    _Cbe_c = float(static_p.get("Cbe", 0.0))
                    static_cache_fp64["Zbe"] = (
                        _Rbe_c / (1.0 + 1j * omega_dev * _Rbe_c * _Cbe_c))
                    _cached_msgs.append("Zbe")
                if not (set(_T_ZBC_KEYS) & swept_set):
                    _Rbc_c = float(static_p.get("Rbc", 1.0))
                    _Cbc_c = float(static_p.get("Cbc", 0.0))
                    static_cache_fp64["Zbc"] = (
                        _Rbc_c / (1.0 + 1j * omega_dev * _Rbc_c * _Cbc_c))
                    _cached_msgs.append("Zbc")
                if not (set(_T_ALPHA_KEYS) & swept_set):
                    _a0 = float(static_p.get("alpha0", 0.0))
                    _tC = float(static_p.get("tauC",   0.0))
                    _tB = float(static_p.get("tauB",   0.0))
                    static_cache_fp64["alpha"] = (
                        _a0 * xp.exp(-1j * omega_dev * _tC)
                        / (1.0 + 1j * omega_dev * _tB))
                    _cached_msgs.append("alpha")
                if not (set(_T_INT_ALL) & swept_set):
                    # Inline what _Y_int_T_batch would compute, once.
                    _Rbi_c = float(static_p.get("Rbi", 0.0))
                    _Zbe_c = static_cache_fp64["Zbe"]
                    _Zbc_c = static_cache_fp64["Zbc"]
                    _alpha_c = static_cache_fp64["alpha"]
                    _z00 = _Rbi_c + _Zbe_c
                    _z01 = _Zbe_c
                    _z10 = _Zbe_c - _alpha_c * _Zbc_c
                    _z11 = (1.0 - _alpha_c) * _Zbc_c + _Zbe_c
                    _det = _z00 * _z11 - _z01 * _z10
                    _inv_det = 1.0 / _det
                    static_cache_fp64["T_int_planes"] = (
                         _z11 * _inv_det,
                        -_z01 * _inv_det,
                        -_z10 * _inv_det,
                         _z00 * _inv_det,
                    )
                    _cached_msgs.append("T_int_planes")
            elif _model_short == "pi":
                if not (set(_PI_YBE_KEYS) & swept_set):
                    _Rbe_c = float(static_p.get("Rbe", 1.0))
                    _Cbe_c = float(static_p.get("Cbe", 0.0))
                    static_cache_fp64["Ybe"] = (
                        1.0 / _Rbe_c + 1j * omega_dev * _Cbe_c)
                    _cached_msgs.append("Ybe")
                if not (set(_PI_YBC_KEYS) & swept_set):
                    _Rbc_c = float(static_p.get("Rbc", 1e9))
                    _Cbc_c = float(static_p.get("Cbc", 0.0))
                    static_cache_fp64["Ybc"] = (
                        1.0 / _Rbc_c + 1j * omega_dev * _Cbc_c)
                    _cached_msgs.append("Ybc")
                if not (set(_PI_GM_KEYS) & swept_set):
                    _Gm0_c = float(static_p.get("Gm0", 0.0))
                    _tau_c = float(static_p.get("tau", 0.0))
                    static_cache_fp64["gm"] = (
                        _Gm0_c * xp.exp(-1j * omega_dev * _tau_c))
                    _cached_msgs.append("gm")
                if not (set(_PI_INT_ALL) & swept_set):
                    # Inline what _Y_int_Pi_batch would compute, once.
                    _Rbi_c   = float(static_p.get("Rbi", 0.0))
                    _Ybe_c   = static_cache_fp64["Ybe"]
                    _Ybc_c   = static_cache_fp64["Ybc"]
                    _gm_c    = static_cache_fp64["gm"]
                    _yc00 = _Ybe_c + _Ybc_c
                    _yc01 = -_Ybc_c
                    _yc10 = _gm_c  - _Ybc_c
                    _yc11 = _Ybc_c
                    _idc  = 1.0 / (_yc00 * _yc11 - _yc01 * _yc10)
                    _zc00 =  _yc11 * _idc + _Rbi_c
                    _zc01 = -_yc01 * _idc
                    _zc10 = -_yc10 * _idc
                    _zc11 =  _yc00 * _idc
                    _idi  = 1.0 / (_zc00 * _zc11 - _zc01 * _zc10)
                    static_cache_fp64["Pi_int_planes"] = (
                         _zc11 * _idi,
                        -_zc01 * _idi,
                        -_zc10 * _idi,
                         _zc00 * _idi,
                    )
                    _cached_msgs.append("Pi_int_planes")

            # ── Build the fp32 cache (cast of fp64 cache) for main loop ─────
            # Everything except `omega` and the dtype tag is a complex tensor;
            # cast each in-place to complex64.  `omega` is real — keep its
            # own copy at float32 so multiply-by-J stays in c64.
            if use_fp32_sweep:
                omega_dev_fp32 = omega_dev.astype(np.float32)
                static_cache = {"omega": omega_dev_fp32, "_cdtype": np.complex64}
                for _k, _v in static_cache_fp64.items():
                    if _k in ("omega", "_cdtype"):
                        continue
                    if isinstance(_v, tuple):
                        static_cache[_k] = tuple(
                            _p.astype(np.complex64) if hasattr(_p, "astype") else _p
                            for _p in _v
                        )
                    elif hasattr(_v, "astype"):
                        static_cache[_k] = _v.astype(np.complex64)
                    else:
                        static_cache[_k] = _v
            else:
                static_cache = static_cache_fp64
                static_cache["_cdtype"] = np.complex128

            if _cached_msgs:
                _prec_lbl = "fp32" if use_fp32_sweep else "fp64"
                st.caption(f"Pre-baked constant networks ({_prec_lbl}): "
                           + ", ".join(_cached_msgs))
                print(f"[tune] pre-baked ({_prec_lbl}): "
                      f"{', '.join(_cached_msgs)}", flush=True)

            col_names = ["Total Residual (%)", "S11 (%)", "S12 (%)", "S21 (%)", "S22 (%)"]
            for lbl, u in zip(sweep_labels, sweep_units):
                col_names.append(f"{lbl} ({u})" if u else lbl)

            sess_key = f"tune_df_{topo_key}_{fname}"

            def _topk_to_df(top_arr_host):
                """Display DataFrame is the top-10 per metric across
                {Total, S11, S12, S21, S22}, deduped — 10 ≤ R ≤ 50 rows
                sorted by Total Residual.  See ``_multi_metric_top_n``."""
                return pd.DataFrame(
                    _multi_metric_top_n(top_arr_host, per_metric=10),
                    columns=col_names)

            # ── Auto slab sizing from device free memory ────────────────────
            # Per-combo working-set estimate.
            #
            # The model uses *scalar plane* representation (4 separate (B,N)
            # complex128 planes per 2×2 matrix, not full (B,N,2,2) tensors)
            # — see _sim_wrap_batch in models/cheng.py.  Realistic peak live
            # planes per combo:
            #   • Y_int (4 planes) + Y_extr (2) + Y_ex (4)       = 10 planes
            #   • Z_ex (4) + Z_ser (4) + Z_tot (4)               = 12 planes
            #   • Y_tot (4) + Y_pad (4) + Y_total (4)            = 12 planes
            #   • Y_norm/M/M_inv/N/S working set                 = 12 planes
            #   • Final stacked (B,N,2,2)                        =  4 planes
            #   • Residual diff/num/val                          =  6 planes
            # Each plane = 16 bytes/element × N_freq elements per combo
            # in complex128, or 8 bytes/element in complex64 (fp32 sweep).
            # Initial guess: ~60 planes × 16 = 960 B/combo per N_freq, with a
            # 1.25× safety margin → 12× N_freq complex128 tensor-equivalents.
            # In fp32 mode the same 60 planes are 8 B each, so the bytes
            # estimate halves.  Replaced after iter 1 by an empirical
            # measurement (see `_calibrated` below) — the initial guess only
            # governs the *first* block size before we have real data.
            _bytes_per_complex = 8 if use_fp32_sweep else 16
            per_combo_bytes = (_bytes_per_complex * 4) * N_freq * 12
            _calibrated = False
            free_label = ""
            if use_cuda:
                try:
                    # Free up any cached blocks first so the query reflects
                    # what we can *actually* allocate now (not what's been
                    # pinned by previous calculations).
                    try:
                        _cp.get_default_memory_pool().free_all_blocks()
                        _cp.get_default_pinned_memory_pool().free_all_blocks()
                    except Exception:
                        pass
                    free_b, total_b = _cp.cuda.runtime.memGetInfo()
                    dev = _cp.cuda.Device(0)
                    sm_count = dev.attributes.get("MultiProcessorCount", 0)
                    free_label = (f"GPU{dev.id}: {free_b/1024**3:.2f}/"
                                  f"{total_b/1024**3:.2f} GiB free  ·  {sm_count} SMs")
                    # 0.55 keeps ~45% of free VRAM as headroom for pool
                    # 0.95 uses more VRAM
                    # fragmentation, top-K scratch, persistent buffers, and
                    # the measurement S_mea_dev tensor.
                    budget = int(free_b * 0.95)
                except Exception:
                    budget = 1 * 1024**3
            else:
                # ram_available_bytes() is cgroup-aware (Streamlit Cloud runs
                # inside a memory-limited container) and never raises, so no
                # try/except is needed here — see helpers/mem_budget.py.
                avail = ram_available_bytes()
                free_label = f"CPU RAM: {avail/1024**3:.2f} GiB free"
                budget = int(avail * 0.25)

            max_inner = max(1, budget // max(per_combo_bytes, 1))
            if not use_cuda:
                # CPU mode: cap inner block size more aggressively
                max_inner = min(max_inner, 65_536)
            else:
                # GPU mode: clamp to a sane upper bound to bound output size
                max_inner = min(max_inner, 16_777_216)  # 16M combos per slab

            # ── Multi-axis block sizing ─────────────────────────────────────
            # Pick a per-axis block_shape (one length per swept axis) such
            # that prod(block_shape) <= max_inner.  We start with the full
            # inner_lens and greedily halve the *largest* axis until the
            # product fits.  This generalises the old "slab one axis" logic
            # to handle the case where multiple axes need slabbing — i.e.
            # when the product of all "other" axes alone exceeds VRAM.
            def _shrink_to_budget(shape, budget):
                """Greedy: halve the largest axis until prod(shape) <= budget."""
                bs = list(shape)
                if not bs:
                    return bs
                while True:
                    p = 1
                    for v in bs:
                        p *= v
                    if p <= budget or budget < 1:
                        return bs
                    k_max = 0
                    for k in range(1, len(bs)):
                        if bs[k] > bs[k_max]:
                            k_max = k
                    if bs[k_max] <= 1:
                        return bs  # cannot shrink further
                    bs[k_max] = max(1, bs[k_max] // 2)

            def _shrink_one_step(shape):
                """Halve the largest axis once. Returns (new_shape, did_shrink)."""
                bs = list(shape)
                if not bs:
                    return bs, False
                k_max = 0
                for k in range(1, len(bs)):
                    if bs[k] > bs[k_max]:
                        k_max = k
                if bs[k_max] <= 1:
                    return bs, False
                bs[k_max] = max(1, bs[k_max] // 2)
                return bs, True

            def _grow_to_budget(shape, full_lens, budget):
                """Greedy: double the smallest still-growable axis until either
                doubling again would exceed `budget`, or every axis is at its
                maximum (full_lens[k]).  Returns the new shape (a list)."""
                bs = list(shape)
                if not bs:
                    return bs
                while True:
                    p = 1
                    for v in bs:
                        p *= v
                    if p * 2 > budget:
                        return bs
                    # Pick smallest axis that can still grow
                    cand_k = -1
                    cand_v = None
                    for k in range(len(bs)):
                        if bs[k] < full_lens[k]:
                            if cand_k < 0 or bs[k] < cand_v:
                                cand_k = k
                                cand_v = bs[k]
                    if cand_k < 0:
                        return bs  # nothing left to grow
                    bs[cand_k] = min(full_lens[cand_k], max(2, bs[cand_k] * 2))

            if n_swept_dims == 0:
                block_shape = []
                inner_block = 1
            else:
                block_shape = _shrink_to_budget(inner_lens, max_inner)
                inner_block = 1
                for v in block_shape:
                    inner_block *= v

            CHUNK_MAX = inner_block  # upper bound on per-iter combo count

            import sys as _sys, time as _time
            _t_start = _time.time()
            if free_label:
                st.caption(free_label)
            if n_swept_dims == 0:
                _slab_label = "single combo"
            elif inner_block >= n_total:
                _slab_label = (f"single N-D block, "
                               f"{inner_block:,} combos / iter")
            else:
                # Shape summary: e.g. "block=(8,32,7), 1,792 combos / iter"
                _shape_str = ",".join(str(v) for v in block_shape)
                _full_str  = ",".join(str(v) for v in inner_lens)
                # Conservative iter count from current block (may rise if
                # shrunk later, fall if grown — we don't grow).
                _n_iters_pre = 1
                for k, v in enumerate(block_shape):
                    _n_iters_pre *= (inner_lens[k] + v - 1) // v
                _slab_label = (f"block=({_shape_str})/({_full_str}), "
                               f"{inner_block:,} combos / iter, "
                               f"≥{_n_iters_pre} iters")
            print(f"\n[tune] start  mode={_mode_label}  total={n_total:,}  "
                  f"{_slab_label}  batched={has_batch}  {free_label}", flush=True)

            # ── UI placeholders ────────────────────────────────────────────
            ui_cols = st.columns([5, 1])
            with ui_cols[0]:
                _tuning_word = tr("Tuning", "調諧")
                _prog_lbl = f"{_tuning_word} {phase_label} ({_mode_label})…" \
                            if phase_label \
                            else f"{_tuning_word} ({_mode_label})…"
                progress = st.progress(0, text=_prog_lbl)
            with ui_cols[1]:
                stop_box = st.empty()
            best_box = st.empty()      # live "best so far" line

            # The Stop button works by triggering a Streamlit re-run on click;
            # the next st.* call inside the loop raises RerunException, which
            # we catch and turn into a clean cancellation.  No on_click needed.
            stop_key = f"tune_stop_{topo_key}_{fname}{phase_suffix}"
            stop_box.button(
                tr("⏹ Stop", "⏹ 停止"),
                key=stop_key,
                help=tr("Stop the calculation. The best results found so far "
                        "are kept.",
                        "停止計算。目前找到的最佳結果將被保留。"),
                type="secondary",
            )

            cancelled = False

            # ── Persistent device buffers for the top-K accumulator ─────────
            # Inf placeholders ensure new finite values always displace them.
            # top_4 must also start at inf — Prioritize sorts by view_4[:, _smap[sort_metric]],
            # so zero placeholders would otherwise out-rank every real residual and the
            # "Best so far" panel would stay empty for the entire sweep.
            top_res    = xp.full(TOP_K, xp.inf, dtype=xp.float64)
            top_4      = xp.full((TOP_K, 4), xp.inf, dtype=xp.float64)
            top_swept  = xp.zeros((TOP_K, max(n_swept, 1)), dtype=xp.float64)

            # ── Pre-allocated scratch buffers for the merge step ────────────
            # Sized for TOP_K + the *maximum* chunk we'd ever submit.  Reused
            # every iteration → zero per-chunk allocation churn for top-K.
            SCRATCH = TOP_K + CHUNK_MAX
            scratch_res   = xp.empty(SCRATCH, dtype=xp.float64)
            scratch_4     = xp.empty((SCRATCH, 4), dtype=xp.float64)
            scratch_swept = xp.empty((SCRATCH, max(n_swept, 1)), dtype=xp.float64)

            # Threshold used to distinguish "real residual" from the BIG=1e308
            # clamp the inner loop assigns to combos that fail the
            # deviation / residual filters or produced NaN/inf simulations.
            # Real residuals are percentages (typically 0.01–1000); 1e100 is a
            # comfortable separator below BIG and above any plausible value.
            _VALID_RES_MAX = 1.0e100

            def _sync_topk_host():
                """Pull the top-K state to host as a (n, 5+n_params) numpy array.
                Drops inf placeholders and BIG-clamped (filter-rejected) rows.
                Constants are filled from sweep_lists.
                """
                if use_cuda:
                    tr = _cp.asnumpy(top_res)
                    t4 = _cp.asnumpy(top_4)
                    ts = _cp.asnumpy(top_swept) if n_swept > 0 else \
                         np.zeros((TOP_K, 0), dtype=np.float64)
                else:
                    tr = np.asarray(top_res)
                    t4 = np.asarray(top_4)
                    ts = np.asarray(top_swept) if n_swept > 0 else \
                         np.zeros((TOP_K, 0), dtype=np.float64)
                valid = np.isfinite(tr) & (tr < _VALID_RES_MAX)
                n = int(valid.sum())
                if n == 0:
                    return None
                out = np.empty((n, 5 + n_params), dtype=np.float64)
                out[:, 0]   = tr[valid]
                out[:, 1:5] = t4[valid]
                j = 0
                for i in range(n_params):
                    if L_list[i] == 1:
                        out[:, 5 + i] = float(sweep_lists[i][0])
                    else:
                        out[:, 5 + i] = ts[valid, j]
                        j += 1
                return out

            def _persist_topk():
                """Best-effort persist to session_state. Safe to call from
                anywhere (including the finally clause)."""
                try:
                    h = _sync_topk_host()
                    if h is not None:
                        st.session_state[sess_key] = _topk_to_df(h)
                except Exception:
                    pass

            def _release_gpu_memory():
                """Best-effort: synchronize the device, run gc twice (to break
                cycles), then return all idle pool blocks to the driver.

                Order matters:
                  1. Synchronize first — async kernels may still be holding
                     tensor inputs alive on the stream.  Without sync,
                     `free_all_blocks()` would skip those blocks.
                  2. gc.collect() twice — CuPy ndarrays often participate in
                     reference cycles via residual/topk dicts; one pass may
                     not break them all.
                  3. Free both device and pinned-host memory pools.
                """
                if not use_cuda:
                    gc.collect()
                    gc.collect()
                    return
                try:
                    _cp.cuda.runtime.deviceSynchronize()
                except Exception:
                    pass
                gc.collect()
                gc.collect()
                try:
                    _cp.get_default_memory_pool().free_all_blocks()
                    _cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

            try:
                if not has_batch:
                    raise RuntimeError(
                        f"Model {model_cls.__name__} has no simulate_batch — "
                        f"cannot run batched tuning. Implement simulate_batch.")

                processed = 0
                last_ui = 0.0
                chunks_done = 0
                last_chunk_ms = 0.0
                # axis_offsets[k] = current row index along swept axis k.
                # Advances by block_shape[k] after each successful iteration,
                # carrying over to the next-outer axis at L_k.
                axis_offsets = [0] * n_swept_dims
                # Resize gate — never re-poll the driver more than once per
                # this many seconds.  Polling memGetInfo() is cheap (~µs) but
                # the *real* cost we're avoiding is fragmenting the allocator
                # by trying to resize too aggressively.
                _last_resize_t = 0.0
                _RESIZE_INTERVAL = 2.0
                # Consecutive-OOM counter — drives the escalating recovery
                # strategy: 1st OOM just shrinks (cheap), 2nd consecutive OOM
                # also flushes the pool (expensive but reclaims everything).
                # Reset to 0 after any successful iteration.
                _oom_streak = 0

                while processed < n_total:
                    # ── Adaptive multi-axis block sizing (gated). ──────────
                    # We re-query free VRAM at most once every _RESIZE_INTERVAL
                    # seconds.  Crucially we do NOT call free_all_blocks() —
                    # flushing the pool every iteration kills allocator
                    # amortization and forces every alloc to fall through to
                    # cudaMalloc, which on Windows WDDM saturates the Copy
                    # engine with page-table updates and starves Compute.
                    #
                    # Instead we use the pool's own bookkeeping (free_bytes
                    # = bytes already cached and re-allocatable for free) to
                    # build an "effective free" estimate without disturbing
                    # the pool.  Once `_calibrated`, the budget is also used
                    # to decide whether the block can grow.
                    _now_t = _time.time()
                    if (use_cuda and n_swept_dims > 0
                            and (_now_t - _last_resize_t) >= _RESIZE_INTERVAL):
                        _last_resize_t = _now_t
                        try:
                            _mp = _cp.get_default_memory_pool()
                            _pool_free = _mp.free_bytes()    # cached, re-allocatable
                            _drv_free, _ = _cp.cuda.runtime.memGetInfo()
                            # Effective free = what driver reports + what's
                            # already cached in the pool (the pool's cached
                            # blocks count against driver-reported free, but
                            # are available to us without a malloc).
                            _eff_free = _drv_free + _pool_free
                            _budget_now = int(_eff_free * 0.55)
                            _max_inner_now = max(
                                1, _budget_now // max(per_combo_bytes, 1))
                            _cur_prod = 1
                            for v in block_shape:
                                _cur_prod *= v
                            # Shrink if we're now over budget
                            _shrunk = _shrink_to_budget(block_shape, _max_inner_now)
                            _shrunk_prod = 1
                            for v in _shrunk:
                                _shrunk_prod *= v
                            if _shrunk_prod < _cur_prod:
                                block_shape = _shrunk
                                print(f"\n[tune] free VRAM dropped → block shrunk to "
                                      f"({','.join(str(v) for v in block_shape)}) "
                                      f"= {_shrunk_prod:,} combos / iter",
                                      flush=True)
                            elif _calibrated:
                                # Try to grow.  Only after calibration — using
                                # the pessimistic initial estimate to compute
                                # a grow target would let us grow into an OOM.
                                _grown = _grow_to_budget(
                                    block_shape, inner_lens, _max_inner_now)
                                _grown_prod = 1
                                for v in _grown:
                                    _grown_prod *= v
                                # Only act if growth is meaningful (≥ +50%)
                                if _grown_prod >= int(_cur_prod * 1.5):
                                    block_shape = _grown
                                    print(f"\n[tune] free VRAM ample → block grown to "
                                          f"({','.join(str(v) for v in block_shape)}) "
                                          f"= {_grown_prod:,} combos / iter",
                                          flush=True)
                        except Exception:
                            pass

                    # ── Compute this iteration's per-axis extent ────────────
                    # cur_shape[k] = how many rows of axis k this iteration
                    # covers, capped at the remainder of the axis.
                    if n_swept_dims == 0:
                        cur_shape = ()
                        inner_shape = ()
                    else:
                        cur_shape = tuple(
                            min(block_shape[k], inner_lens[k] - axis_offsets[k])
                            for k in range(n_swept_dims)
                        )
                        inner_shape = cur_shape
                    B_inner = 1
                    for L in inner_shape:
                        B_inner *= L
                    if B_inner == 0:
                        break
                    _t_chunk = _time.time()

                    # ── Build N-D parameter tensors for this slab ───────────
                    # Each swept param gets shape (1,..,L,..,1,1) — its own
                    # length-L axis at its position, 1s elsewhere, and a
                    # trailing 1 for the freq slot.  Constants stay scalar.
                    # On host the per-param values are tiny: only the *device*
                    # tensors matter for VRAM, and they're (1,..,L,..,1,1).
                    p_nd = None
                    S_batch_nd = None
                    S_flat = None
                    res = None
                    cur_total = None
                    cur_4 = None
                    cur_swept = None
                    lin = None
                    idx_2d = None
                    view_res = None
                    idx = None
                    try:
                        p_nd = dict(all_p)
                        for j, key in enumerate(sweep_keys):
                            if not varies_mask[j]:
                                p_nd[key] = const_si[key]
                                continue
                            swept_pos = swept_pos_to_param_idx.index(j)
                            o = axis_offsets[swept_pos]
                            ln = cur_shape[swept_pos]
                            vals_disp = sweep_lists[j][o:o + ln]
                            vals_si = vals_disp / float(sweep_scales[j])
                            nd_shape = [1] * (n_swept_dims + 1)
                            nd_shape[swept_pos] = len(vals_si)
                            # rdtype_main is float32 in fp32 sweep mode, so
                            # the swept tensor doesn't get promoted back to
                            # float64 inside the model kernels.
                            p_nd[key] = xp.asarray(
                                vals_si, dtype=rdtype_main).reshape(nd_shape)
                    except Exception as exc:
                        is_oom = (isinstance(exc, MemoryError) or
                                  "out of memory" in str(exc).lower() or
                                  "OutOfMemoryError" in type(exc).__name__)
                        # Drop any partially-built tensors so the retry has room
                        p_nd = None
                        if is_oom:
                            new_block, did = _shrink_one_step(block_shape)
                            if did:
                                block_shape = new_block
                                _oom_streak += 1
                                print(f"\n[tune] OOM in param-gen → block "
                                      f"({','.join(str(v) for v in block_shape)}) "
                                      f"[streak={_oom_streak}]",
                                      flush=True)
                                gc.collect()
                                # Only flush the pool on the *second* OOM in a
                                # row.  A single OOM is usually solved by the
                                # shrink alone — flushing every time would
                                # destroy allocator amortization (the same
                                # bug we're fixing in this commit).
                                if use_cuda and _oom_streak >= 2:
                                    try:
                                        _cp.get_default_memory_pool().free_all_blocks()
                                        _cp.get_default_pinned_memory_pool().free_all_blocks()
                                        print("[tune]   pool flushed (last resort)",
                                              flush=True)
                                    except Exception:
                                        pass
                                continue
                        raise

                    # ── Run simulate + residuals + top-K merge on device ────
                    try:
                        # Snapshot pool size BEFORE simulate so we can
                        # measure the actual per-combo allocation footprint
                        # of this iteration and replace the static estimate
                        # with an empirical one (only on the first call,
                        # gated by `_calibrated`).
                        if use_cuda and not _calibrated:
                            try:
                                _bytes_before = _cp.get_default_memory_pool().total_bytes()
                            except Exception:
                                _bytes_before = 0
                        S_batch_nd = model_cls.simulate_batch(
                            p_nd, freq, z0, xp=xp, cache=static_cache)
                        # Shape: inner_shape + (N_freq, 2, 2)
                        S_flat = S_batch_nd.reshape(B_inner, N_freq, 2, 2)
                        res = _port_residuals_batch(S_mea_dev, S_flat, xp)
                        cur_total = res["Total"]                                  # (B_inner,)
                        cur_4 = xp.stack(
                            [res["S11"], res["S12"], res["S21"], res["S22"]],
                            axis=1)                                                # (B_inner, 4)

                        # NaN/inf protection — sentinel value sorts to bottom
                        # without forcing a host-side .all() check (no sync).
                        BIG = 1.0e308
                        cur_total = xp.where(xp.isfinite(cur_total), cur_total, BIG)
                        # Per-port residuals are also the sort key in
                        # "Prioritize" mode, so clamp NaN/inf there too.
                        cur_4 = xp.where(xp.isfinite(cur_4), cur_4, BIG)

                        # ── Balance / quality filters ───────────────────────
                        # Push unbalanced or poor-residual combos to the
                        # bottom of the ranking by clamping their sort key
                        # to BIG.  Balanced survivors are then sorted normally
                        # → "lowest total among balanced".
                        if dev_threshold is not None:
                            _dev = cur_4.max(axis=1) - cur_4.min(axis=1)
                            cur_total = xp.where(_dev > dev_threshold, BIG, cur_total)
                            cur_4 = xp.where(
                                _dev[:, None] > dev_threshold, BIG, cur_4)
                        if res_threshold is not None:
                            _peak = cur_4.max(axis=1)
                            cur_total = xp.where(_peak > res_threshold, BIG, cur_total)
                            cur_4 = xp.where(
                                _peak[:, None] > res_threshold, BIG, cur_4)

                        # ── Build display values for swept params ───────────
                        # Decompose flat index → multi-axis index using
                        # row-major strides over `inner_shape`, then look up
                        # via sweep_table_dev[swept_indices, axis_index].
                        if n_swept_dims > 0:
                            local_strides_h = np.ones(n_swept_dims, dtype=np.int64)
                            for k in range(n_swept_dims - 2, -1, -1):
                                local_strides_h[k] = local_strides_h[k + 1] * inner_shape[k + 1]
                            local_strides_dev = xp.asarray(local_strides_h)
                            local_lens_dev    = xp.asarray(np.asarray(inner_shape, dtype=np.int64))

                            lin = xp.arange(B_inner, dtype=xp.int64)
                            idx_2d = (lin[:, None] // local_strides_dev[None, :]) \
                                      % local_lens_dev[None, :]                    # (B_inner, n_swept)

                            if any(o > 0 for o in axis_offsets):
                                offset_h = np.asarray(axis_offsets, dtype=np.int64)
                                idx_2d = idx_2d + xp.asarray(offset_h)[None, :]

                            cur_swept = sweep_table_dev[swept_indices_dev[None, :], idx_2d]

                        # ── Top-K merge into pre-allocated scratch ──────────
                        total_in = TOP_K + B_inner
                        scratch_res[:TOP_K]              = top_res
                        scratch_res[TOP_K:total_in]      = cur_total
                        scratch_4[:TOP_K]                = top_4
                        scratch_4[TOP_K:total_in]        = cur_4
                        if n_swept_dims > 0:
                            scratch_swept[:TOP_K]         = top_swept
                            scratch_swept[TOP_K:total_in] = cur_swept

                        view_res = scratch_res[:total_in]
                        view_4   = scratch_4[:total_in]
                        # When prioritizing one S-parameter, sort by that
                        # column instead of the total residual. We still keep
                        # `top_res` = total so the displayed "Total Residual"
                        # column stays meaningful.
                        if sort_metric == "Total":
                            sort_view = view_res
                        else:
                            _smap = {"S11": 0, "S12": 1, "S21": 2, "S22": 3}
                            sort_view = view_4[:, _smap[sort_metric]]
                        idx = xp.argpartition(sort_view, TOP_K)[:TOP_K]
                        idx = idx[xp.argsort(sort_view[idx])]

                        top_res[:] = view_res[idx]
                        top_4[:]   = view_4[idx]
                        if n_swept_dims > 0:
                            top_swept[:] = scratch_swept[:total_in][idx]

                        # ── Empirical calibration (first iteration only) ─
                        # Measure how many bytes the pool actually grew by
                        # during this iteration, divide by B_inner, and use
                        # that as the ground-truth per-combo footprint.
                        # Apply a 1.3× safety to absorb spike differences
                        # between iterations.  This replaces the (rough)
                        # initial estimate so subsequent shrink/grow
                        # decisions are made on real data.
                        if use_cuda and not _calibrated and B_inner > 0:
                            try:
                                _bytes_after = _cp.get_default_memory_pool().total_bytes()
                                _delta = max(0, _bytes_after - _bytes_before)
                                if _delta > 0:
                                    _measured = _delta // B_inner
                                    _new_pcb = max(1, int(_measured * 1.3))
                                    print(f"\n[tune] calibrated per_combo_bytes "
                                          f"= {_new_pcb:,}  ({_measured:,} "
                                          f"measured × 1.3 safety; was "
                                          f"{per_combo_bytes:,})",
                                          flush=True)
                                    per_combo_bytes = _new_pcb
                                _calibrated = True
                                # Force a resize check on the *next* iter so
                                # the new estimate can immediately grow the
                                # block if there's headroom.
                                _last_resize_t = 0.0
                            except Exception:
                                _calibrated = True   # don't keep retrying
                    except Exception as exc:
                        is_oom = (isinstance(exc, MemoryError) or
                                  "out of memory" in str(exc).lower() or
                                  "OutOfMemoryError" in type(exc).__name__)
                        if is_oom:
                            new_block, did = _shrink_one_step(block_shape)
                            if did:
                                old_str = ",".join(str(v) for v in block_shape)
                                new_str = ",".join(str(v) for v in new_block)
                                _oom_streak += 1
                                print(f"\n[tune] OOM at block=({old_str}) → "
                                      f"retry with ({new_str}) "
                                      f"[streak={_oom_streak}]",
                                      flush=True)
                                block_shape = new_block
                                # Drop intermediates before retrying — finally
                                # block will null these out, but we run gc.collect
                                # immediately to release memory before continue.
                                p_nd = None
                                S_batch_nd = None
                                S_flat = None
                                res = None
                                cur_total = None
                                cur_4 = None
                                cur_swept = None
                                lin = None
                                idx_2d = None
                                view_res = None
                                idx = None
                                gc.collect()
                                # Only flush the pool on the *second*
                                # consecutive OOM.  A single OOM is usually
                                # solved by the shrink alone.  Flushing every
                                # OOM destroys allocator amortization.
                                if use_cuda and _oom_streak >= 2:
                                    try:
                                        _cp.get_default_memory_pool().free_all_blocks()
                                        _cp.get_default_pinned_memory_pool().free_all_blocks()
                                        print("[tune]   pool flushed (last resort)",
                                              flush=True)
                                    except Exception:
                                        pass
                                continue   # do NOT advance counters
                            # Already at all-1s floor — bail out cleanly with
                            # whatever top-K we have, instead of spinning
                            # forever or crashing the Streamlit script.
                            print(f"\n[tune] OOM at block=(1,1,...) — single combo "
                                  f"won't fit in VRAM, giving up",
                                  flush=True)
                            st.error(tr(
                                f"GPU ran out of memory even at block=(1,1,...): "
                                f"a single combination's working set "
                                f"(~{per_combo_bytes/1024**2:.1f} MiB for {N_freq} freq pts) "
                                f"won't fit in available VRAM. Reduce the number "
                                f"of frequency points or free GPU memory.",
                                f"即使在 block=(1,1,...) 下 GPU 仍記憶體不足："
                                f"單一組合的工作集"
                                f"（約 {per_combo_bytes/1024**2:.1f} MiB，"
                                f"{N_freq} 個頻率點）無法容納於可用 VRAM 中。"
                                f"請減少頻率點數或釋放 GPU 記憶體。"))
                            cancelled = True
                            break
                        off_str = ",".join(str(o) for o in axis_offsets)
                        print(f"\n[tune] block @offsets ({off_str}) failed: {exc!r}",
                              flush=True)
                        # Skip this block on non-OOM errors — advance below
                        # via the post-loop counters by treating it as done.
                        # Fall through to normal advance.
                    finally:
                        # Drop per-iteration intermediates so OOM retry has room
                        # *before* re-allocating next iteration.  We don't
                        # touch `top_*` / `scratch_*` (persistent).
                        p_nd = None
                        S_batch_nd = None
                        S_flat = None
                        res = None
                        cur_total = None
                        cur_4 = None
                        cur_swept = None
                        lin = None
                        idx_2d = None
                        view_res = None
                        idx = None

                    # Successful (or skipped) iteration — advance counters.
                    # Reset OOM streak: we got through a full iteration, so
                    # any past OOMs are no longer "consecutive".
                    _oom_streak = 0
                    # Multi-axis carry: advance the *innermost* axis by its
                    # current block step; if it overflows L_k, reset to 0
                    # and carry into the next-outer axis.
                    if n_swept_dims == 0:
                        processed = 1
                    else:
                        processed += B_inner
                        k = n_swept_dims - 1
                        while k >= 0:
                            axis_offsets[k] += block_shape[k]
                            if axis_offsets[k] < inner_lens[k]:
                                break
                            axis_offsets[k] = 0
                            k -= 1
                        # k < 0  →  every axis wrapped, we're done.  The
                        # while-loop guard `processed < n_total` will exit.
                    chunks_done += 1
                    last_chunk_ms = (_time.time() - _t_chunk) * 1000.0

                    # ── Throttled UI tick: ~2 Hz, the only host sync point ──
                    now = _time.time()
                    if now - last_ui > 0.5 or processed >= n_total:
                        elapsed = now - _t_start
                        rate = processed / elapsed if elapsed > 0 else 0.0
                        eta = (n_total - processed) / rate if rate > 0 else 0.0
                        _phase_str = f" {phase_label}" if phase_label else ""
                        _tuning_word = tr("Tuning", "調諧")
                        _combos_word = tr("combos", "組合")
                        progress.progress(
                            min(1.0, processed / max(n_total, 1)),
                            text=(f"{_tuning_word}{_phase_str} ({_mode_label})… "
                                  f"{processed:,}/{n_total:,} {_combos_word}  "
                                  f"({rate:,.0f}/s, ETA {_fmt_eta(eta)})  "
                                  f"slab={B_inner:,}  ({last_chunk_ms:.1f} ms/iter)"))
                        top_arr_host = _sync_topk_host()
                        if top_arr_host is not None:
                            best_series = pd.Series(top_arr_host[0], index=col_names)
                            best_box.markdown(
                                _best_summary_md(best_series, label=tr("Best so far", "目前最佳")),
                                unsafe_allow_html=True,
                            )
                            # Persist every tick so a later crash leaves a result
                            st.session_state[sess_key] = _topk_to_df(top_arr_host)
                        elif dev_threshold is not None or res_threshold is not None:
                            # All combos so far failed the deviation/residual
                            # filters — surface this so the user can widen
                            # thresholds instead of staring at a blank panel.
                            best_box.markdown(tr(
                                "*No combos have passed the deviation/residual "
                                "filters yet — consider widening the thresholds "
                                "if this persists.*",
                                "*尚無組合通過偏差/殘差篩選 — "
                                "若持續發生，請考慮放寬閾值。*"))
                        last_ui = now

                    _sys.stdout.write(
                        f"\r[tune] {processed:>11,}/{n_total:,}  "
                        f"({100.0*processed/max(n_total,1):5.1f}%)  "
                        f"{(processed / max(_time.time()-_t_start, 1e-9)):>10,.0f} calc/s  "
                        f"slab={B_inner:,}  {last_chunk_ms:6.1f} ms/iter")
                    _sys.stdout.flush()
            except (KeyboardInterrupt, SystemExit):
                cancelled = True
                st.warning(tr(
                    "Computation cancelled — keeping the best results found so far.",
                    "計算已取消 — 已保留目前找到的最佳結果。"))
                print("\n[tune] cancelled by user (KeyboardInterrupt)", flush=True)
            except MemoryError as me:
                st.error(tr(
                    f"Out of memory: {me}. Keeping the best results found so far.",
                    f"記憶體不足：{me}。已保留目前找到的最佳結果。"))
                print(f"\n[tune] MemoryError: {me}", flush=True)
            except BaseException as exc:
                # Streamlit raises RerunException OR StopException when the
                # user clicks any widget (incl. our Stop button) — which one
                # depends on Streamlit version and the exact widget path.
                # Treat both identically: persist state, free GPU buffers,
                # then re-raise so Streamlit can finish the rerun cleanly.
                _is_rerun = (_RerunException is not None
                             and isinstance(exc, _RerunException))
                _is_stop  = (_StopException  is not None
                             and isinstance(exc, _StopException))
                # Belt-and-suspenders: also match by class name in case the
                # exception class moved between Streamlit versions and our
                # imports above silently fell through to None.
                _name = type(exc).__name__
                _is_streamlit_stop = _is_rerun or _is_stop or (
                    _name in ("RerunException", "StopException"))
                if _is_streamlit_stop:
                    cancelled = True
                    print(f"\n[tune] cancelled (Streamlit {_name}, "
                          f"e.g. Stop button)", flush=True)
                    _persist_topk()
                    # Free GPU buffers before re-raising so the rerun starts clean
                    try:
                        S_mea_dev = None
                        S_mea_dev_fp64 = None
                        top_res = None; top_4 = None; top_swept = None
                        scratch_res = None; scratch_4 = None; scratch_swept = None
                        sweep_table_dev = None
                        swept_indices_dev = None
                        omega_dev = None
                        # Per-iteration intermediates that may still be alive
                        # if the exception was raised mid-iteration.  The
                        # inner finally usually nulls these, but we
                        # belt-and-suspenders here.
                        try:
                            del p_nd, S_batch_nd, S_flat, res, cur_total, cur_4
                            del cur_swept, lin, idx_2d, view_res, idx
                        except (NameError, UnboundLocalError):
                            pass
                        static_cache.clear()
                        if static_cache_fp64 is not static_cache:
                            static_cache_fp64.clear()
                    except Exception:
                        pass
                    _release_gpu_memory()
                    raise
                # Anything else: log, persist, re-raise
                print(f"\n[tune] unexpected exception: {exc!r}", flush=True)
                _persist_topk()
                raise
            finally:
                # Always persist whatever we have so a crash never wipes results.
                _persist_topk()

            progress.empty()
            stop_box.empty()

            n_kept = (0 if top_res is None
                      else int(xp.sum(xp.isfinite(top_res)
                                       & (top_res < _VALID_RES_MAX)).item()))

            # Keep a warning visible if a deviation/residual sweep rejected
            # everything; otherwise clear the live "best so far" line because
            # the persistent results table will render the final ranking.
            if (n_kept == 0
                    and (dev_threshold is not None or res_threshold is not None)):
                best_box.warning(tr(
                    "No combos passed the deviation/residual filters. "
                    "Widen the thresholds and re-run.",
                    "沒有組合通過偏差/殘差篩選。請放寬閾值後重新執行。"))
                # Clear any stale prior-sweep results so the table below
                # doesn't misleadingly show data from a different setting.
                st.session_state.pop(sess_key, None)
                st.session_state.pop(f"tune_elapsed_{topo_key}_{fname}", None)
            else:
                best_box.empty()
            print(f"\n[tune] done   processed={'?' if cancelled else f'{n_total:,}'}  "
                  f"top={n_kept}  in {_time.time()-_t_start:.2f}s", flush=True)

            # ── FP64 rerank of the surviving top-K ──────────────────────────
            # The main loop ran in complex64 (cache_fp32) so the residuals
            # are fp32-accurate.  Re-evaluate every finite top-K combo at
            # complex128 using `static_cache_fp64`, replace the residuals,
            # and re-sort.  TOP_K is small (~100) so this is one batched
            # simulate_batch call — negligible cost vs the main sweep.
            if (use_fp32_sweep and not cancelled
                    and top_res is not None and n_kept > 0):
                try:
                    _top_h = _sync_topk_host()
                    if _top_h is not None and len(_top_h) > 0:
                        K_rk = int(len(_top_h))
                        # Build (K,) SI arrays for swept params, scalars for
                        # constants — same param dict shape `simulate_batch`
                        # already understands.
                        _p_rk = dict(all_p)
                        for j, key in enumerate(sweep_keys):
                            col_disp = _top_h[:, 5 + j]
                            if L_list[j] == 1:
                                _p_rk[key] = (float(col_disp[0])
                                              / float(sweep_scales[j]))
                            else:
                                _p_rk[key] = xp.asarray(
                                    col_disp / float(sweep_scales[j]),
                                    dtype=np.float64)
                        # Force fp64 cdtype on the rerank cache.
                        static_cache_fp64["_cdtype"] = np.complex128
                        S_rk = model_cls.simulate_batch(
                            _p_rk, freq, z0, xp=xp, cache=static_cache_fp64)
                        S_rk_flat = S_rk.reshape(K_rk, N_freq, 2, 2)
                        res_rk = _port_residuals_batch(
                            S_mea_dev_fp64, S_rk_flat, xp)
                        tot_rk = xp.where(
                            xp.isfinite(res_rk["Total"]),
                            res_rk["Total"], 1.0e308)
                        s4_rk  = xp.stack(
                            [res_rk["S11"], res_rk["S12"],
                             res_rk["S21"], res_rk["S22"]], axis=1)

                        # Re-sort by fp64 residuals — match sort_metric chosen
                        # for the main loop so Prioritize keeps its ranking.
                        if sort_metric == "Total":
                            sort_rk = tot_rk
                        else:
                            _smap = {"S11": 0, "S12": 1, "S21": 2, "S22": 3}
                            _col  = _smap[sort_metric]
                            sort_rk = xp.where(xp.isfinite(s4_rk[:, _col]),
                                                s4_rk[:, _col], 1.0e308)
                        order_dev = xp.argsort(sort_rk)
                        tot_rk_s  = tot_rk[order_dev]
                        s4_rk_s   = s4_rk[order_dev]

                        order_h = (_cp.asnumpy(order_dev) if use_cuda
                                   else np.asarray(order_dev))
                        _top_h_s = _top_h[order_h]
                        _top_h_s[:, 0]   = (_cp.asnumpy(tot_rk_s) if use_cuda
                                            else np.asarray(tot_rk_s))
                        _top_h_s[:, 1:5] = (_cp.asnumpy(s4_rk_s) if use_cuda
                                            else np.asarray(s4_rk_s))

                        # Push back into the device top-K accumulators so
                        # display and persist see fp64 values.
                        top_res[:K_rk] = xp.asarray(_top_h_s[:, 0])
                        if K_rk < TOP_K:
                            top_res[K_rk:] = xp.inf
                        top_4[:K_rk] = xp.asarray(_top_h_s[:, 1:5])
                        if n_swept > 0:
                            _ts = np.empty((K_rk, n_swept), dtype=np.float64)
                            _jswept = 0
                            for i in range(n_params):
                                if L_list[i] > 1:
                                    _ts[:, _jswept] = _top_h_s[:, 5 + i]
                                    _jswept += 1
                            top_swept[:K_rk] = xp.asarray(_ts)
                        _persist_topk()
                        print(f"[tune] fp64 rerank: top-{K_rk} "
                              f"re-evaluated and re-sorted", flush=True)
                except Exception as _rk_exc:
                    print(f"[tune] fp64 rerank failed: {_rk_exc!r} — "
                          f"keeping fp32 ranking", flush=True)

            # ── Aggressive cleanup: free everything except the persisted
            # top-100 dataframe (already in st.session_state).  Drop refs
            # first so the GC can collect, then return memory pools to the
            # device / OS.  See _release_gpu_memory() for the sync+gc+flush
            # sequence — without it, async kernels in flight would prevent
            # the pool from actually returning blocks to the driver.
            try:
                S_mea_dev = None
                S_mea_dev_fp64 = None
                top_res = None; top_4 = None; top_swept = None
                scratch_res = None; scratch_4 = None; scratch_swept = None
                sweep_table_dev = None
                swept_indices_dev = None
                omega_dev = None
                static_cache.clear()
                if static_cache_fp64 is not static_cache:
                    static_cache_fp64.clear()
            except Exception:
                pass
            _release_gpu_memory()

            # Persist total wall-clock run time so the results panel can show
            # "Evaluated in …" above the best-residual line (survives reruns).
            st.session_state[f"tune_elapsed_{topo_key}_{fname}"] = (
                _time.time() - _t_start)

        # ── Nelder-Mead "Auto" tuning ──────────────────────────────────
        def _run_nelder_mead(*, max_iter: int, restart: bool,
                             use_cuda: bool = False):
            """Local-search optimisation using scipy's adaptive Nelder-Mead.

            x is in *display units* so the simplex moves at comparable
            magnitudes across mixed parameters (fF / pH / Ω).  Bounds come
            from the per-row Min/Max boxes; the seed is the current
            parameter value clipped into bounds.

            When ``use_cuda`` is True and CuPy is available, each objective
            evaluation uses ``simulate_vec(xp=cupy)`` and computes residuals
            on the GPU, with a single host sync per eval to read the scalar
            total back.  Helps when N_freq is large enough that the GPU
            outruns CPU even at batch=1.

            Top-K accumulator and session-state layout match _run_one_sweep
            so the existing display / "Use best values" button work
            unchanged.
            """
            try:
                from scipy.optimize import minimize as _nm_minimize
            except ImportError:
                st.error(tr(
                    "scipy is required for Nelder-Mead Auto tuning. "
                    "Install with `pip install scipy`.",
                    "Nelder-Mead 自動調諧需要 scipy。"
                    "請以 `pip install scipy` 安裝。"))
                return

            import pandas as pd

            # Only enabled rows participate in optimisation; constants stay put.
            swept_rows = [r for r in param_rows if r["enabled"]]
            if not swept_rows:
                st.warning(tr(
                    "Auto needs at least one parameter with the **Sweep** "
                    "checkbox enabled.",
                    "自動調諧至少需要一個已啟用**掃描**核取方塊的參數。"))
                return

            sweep_keys_nm   = [r["key"]   for r in swept_rows]
            sweep_scales_nm = [r["scale"] for r in swept_rows]

            # Seed + bounds in display units.
            x0_disp = []
            bounds  = []
            for r in swept_rows:
                cur  = float(all_p.get(r["key"], 0.0)) * r["scale"]
                arr  = np.asarray(r["sweep"], dtype=np.float64)
                lo   = float(np.min(arr))
                hi   = float(np.max(arr))
                if hi <= lo:
                    hi = lo + max(abs(lo), 1.0) * 0.5  # avoid degenerate box
                # Clip the box into the hard physical limits so the simplex
                # can never wander into negative components / out-of-range α₀.
                _h_lo, _h_hi = tune_hard_limits(r["key"], r["label"])
                lo, hi = _clamp_to_hard(lo, hi, _h_lo, _h_hi)
                if hi <= lo:
                    hi = lo + max(abs(lo), 1.0) * 0.5
                cur  = max(lo, min(hi, cur))
                x0_disp.append(cur)
                bounds.append((lo, hi))
            x0      = np.asarray(x0_disp, dtype=np.float64)
            lo_arr  = np.asarray([b[0] for b in bounds], dtype=np.float64)
            hi_arr  = np.asarray([b[1] for b in bounds], dtype=np.float64)

            TOP_K = 100
            col_names = [
                "Total Residual (%)", "S11 (%)", "S12 (%)", "S21 (%)", "S22 (%)"]
            for r in param_rows:
                u   = r["unit"]
                lbl = r["label"]
                col_names.append(f"{lbl} ({u})" if u else lbl)
            sess_key = f"tune_df_{topo_key}_{fname}"

            # Top-K kept on host as a small max-heap-like sorted list.
            top_rows: list[list[float]] = []
            n_eval   = [0]

            def _push_topk(row: list[float]):
                # Keep top-K by total residual.  Linear insert is fine — TOP_K
                # is 100 and Nelder-Mead does <~10k evals total.
                if len(top_rows) < TOP_K:
                    top_rows.append(row)
                    top_rows.sort(key=lambda r: r[0])
                    return
                if row[0] < top_rows[-1][0]:
                    top_rows[-1] = row
                    top_rows.sort(key=lambda r: r[0])

            use_vec = hasattr(model_cls, "simulate_vec")

            # ── CUDA hot path setup ───────────────────────────────────
            # When use_cuda is requested we move S_raw to device once and
            # run residuals there, doing a single .item() per eval to pull
            # the scalar total back to host for scipy.
            xp_dev = _cp if (use_cuda and _HAS_CUDA and use_vec) else np
            if use_cuda and not _HAS_CUDA:
                st.warning(tr("CUDA not available — falling back to CPU.",
                              "CUDA 不可用 — 已回退至 CPU。"))
            elif use_cuda and not use_vec:
                st.warning(tr(
                    f"{model_cls.__name__} has no simulate_vec — "
                    "CUDA path needs it; falling back to CPU.",
                    f"{model_cls.__name__} 沒有 simulate_vec — "
                    "CUDA 路徑需要它；已回退至 CPU。"))
            S_mea_dev = xp_dev.asarray(S_raw)
            _den_dev  = xp_dev.sum(xp_dev.abs(S_mea_dev) ** 2, axis=0)  # (2,2)

            def _residuals_dev(S_sim_dev):
                """Return (total, s11, s12, s21, s22) as Python floats.
                One host sync per call when xp_dev is cupy."""
                diff = S_mea_dev - S_sim_dev                            # (N,2,2)
                num  = xp_dev.sum(xp_dev.abs(diff) ** 2, axis=0)        # (2,2)
                den_s = xp_dev.where(_den_dev > 0, _den_dev, 1.0)
                val   = xp_dev.sqrt(num / den_s) * 100.0
                val   = xp_dev.where(_den_dev > 0, val, 0.0)
                s11 = val[0, 0]; s12 = val[0, 1]
                s21 = val[1, 0]; s22 = val[1, 1]
                tot = (s11 + s12 + s21 + s22) * 0.25
                if xp_dev is np:
                    return (float(tot), float(s11), float(s12),
                            float(s21), float(s22))
                # Single fused host sync — pulls all five scalars in one shot.
                arr = xp_dev.stack([tot, s11, s12, s21, s22])
                arr_h = _cp.asnumpy(arr)
                return (float(arr_h[0]), float(arr_h[1]), float(arr_h[2]),
                        float(arr_h[3]), float(arr_h[4]))

            def _objective(x_disp):
                # Hard-clip into bounds — scipy NM with bounds usually stays
                # inside but the simplex initialisation can poke out by a hair.
                x_disp = np.minimum(np.maximum(x_disp, lo_arr), hi_arr)
                p = dict(all_p)
                for k, sc, v in zip(sweep_keys_nm, sweep_scales_nm, x_disp):
                    p[k] = float(v) / float(sc)
                try:
                    if use_vec:
                        S_sim = model_cls.simulate_vec(p, freq, z0, xp=xp_dev)
                    else:
                        S_sim = model_cls.simulate(p, freq, z0)
                except Exception:
                    return 1.0e8
                if S_sim is None:
                    return 1.0e8
                if xp_dev is np:
                    r = _port_residuals(S_raw, S_sim)
                    r_tot, r_s11, r_s12, r_s21, r_s22 = (
                        r["Total"], r["S11"], r["S12"], r["S21"], r["S22"])
                else:
                    r_tot, r_s11, r_s12, r_s21, r_s22 = _residuals_dev(S_sim)
                if not np.isfinite(r_tot):
                    return 1.0e8
                # Repack to dict shape used downstream (top-K row build)
                r = {"Total": r_tot, "S11": r_s11, "S12": r_s12,
                     "S21": r_s21, "S22": r_s22}
                # Build the (5 + n_params) row in the same column order as the
                # full-sweep dataframe.
                row = [r["Total"], r["S11"], r["S12"], r["S21"], r["S22"]]
                x_dict = dict(zip(sweep_keys_nm, x_disp))
                for pr in param_rows:
                    k = pr["key"]
                    if k in x_dict:
                        row.append(float(x_dict[k]))
                    else:
                        row.append(float(all_p.get(k, 0.0)) * pr["scale"])
                _push_topk(row)
                n_eval[0] += 1
                return float(r["Total"])

            # ── UI placeholders (mirror the sweep UI) ─────────────────
            ui_cols = st.columns([5, 1])
            with ui_cols[0]:
                progress = st.progress(
                    0, text=tr("Auto (Nelder-Mead)…", "自動（Nelder-Mead）…"))
            with ui_cols[1]:
                stop_box = st.empty()
            best_box = st.empty()
            stop_key = f"tune_stop_{topo_key}_{fname}_nm"
            stop_box.button(
                tr("⏹ Stop", "⏹ 停止"), key=stop_key, type="secondary",
                help=tr("Stop the calculation. Best results so far are kept.",
                        "停止計算。目前的最佳結果將被保留。"))

            # NM does roughly maxiter * (n+1) evals worst case.
            ev_budget = max(1, max_iter * (len(x0) + 1))

            def _persist_now():
                if not top_rows:
                    return
                df = pd.DataFrame(
                    _multi_metric_top_n(np.asarray(top_rows, dtype=float),
                                         per_metric=10),
                    columns=col_names)
                st.session_state[sess_key] = df

            def _callback(xk, *args, **kwargs):
                # Periodic progress + best-so-far card.  Streamlit's Stop
                # button works by raising RerunException on any st.* call,
                # so calling progress.progress() here also gives us a clean
                # cancellation point.
                if top_rows:
                    best_series = pd.Series(top_rows[0], index=col_names)
                    best_box.markdown(
                        _best_summary_md(best_series, label=tr("Best so far", "目前最佳")),
                        unsafe_allow_html=True,
                    )
                _auto_nm_word = tr("Auto (Nelder-Mead)", "自動（Nelder-Mead）")
                _evals_word = tr("evals", "次評估")
                if top_rows:
                    _txt = (f"{_auto_nm_word}… {n_eval[0]} {_evals_word}  "
                            f"({tr('best Total', '最佳總計')}: "
                            f"{top_rows[0][0]:.3f}%)")
                else:
                    _txt = f"{_auto_nm_word}… {n_eval[0]} {_evals_word}"
                progress.progress(min(1.0, n_eval[0] / ev_budget), text=_txt)
                _persist_now()

            cancelled = False
            options = {
                "maxiter":  int(max_iter),
                "maxfev":   int(max_iter) * (len(x0) + 1),
                "xatol":    1e-6,
                "fatol":    1e-4,
                "adaptive": True,
            }

            try:
                _nm_minimize(
                    _objective, x0,
                    method="Nelder-Mead",
                    bounds=bounds,
                    callback=_callback,
                    options=options,
                )
                if restart and len(top_rows) > 0:
                    # Random perturbation around current best (10% of box width
                    # per axis) for a single second pass — catches premature
                    # simplex collapse without doubling cost.
                    best_disp = np.asarray(top_rows[0][5:5 + len(param_rows)],
                                           dtype=np.float64)
                    # Pull just the swept components out of best_disp
                    swept_idx = [i for i, pr in enumerate(param_rows)
                                 if pr["enabled"]]
                    x_seed = np.asarray(
                        [best_disp[i] for i in swept_idx], dtype=np.float64)
                    rng = np.random.default_rng(0)
                    span = hi_arr - lo_arr
                    x_seed = np.clip(
                        x_seed + rng.uniform(-0.1, 0.1, size=x_seed.shape) * span,
                        lo_arr, hi_arr)
                    _nm_minimize(
                        _objective, x_seed,
                        method="Nelder-Mead",
                        bounds=bounds,
                        callback=_callback,
                        options=options,
                    )
            except BaseException as exc:
                _is_rerun = (_RerunException is not None
                             and isinstance(exc, _RerunException))
                _is_stop  = (_StopException is not None
                             and isinstance(exc, _StopException))
                _name = type(exc).__name__
                if _is_rerun or _is_stop or _name in (
                        "RerunException", "StopException"):
                    cancelled = True
                    _persist_now()
                    # Drop device buffers before re-raising so the next
                    # rerun starts from a clean pool.
                    if xp_dev is not np:
                        try:
                            S_mea_dev = None
                            _cp.cuda.runtime.deviceSynchronize()
                            gc.collect()
                            _cp.get_default_memory_pool().free_all_blocks()
                        except Exception:
                            pass
                    raise
                st.error(tr(f"Nelder-Mead failed: {exc!r}",
                            f"Nelder-Mead 執行失敗：{exc!r}"))
            finally:
                _persist_now()
                if xp_dev is not np:
                    try:
                        S_mea_dev = None
                        gc.collect()
                        _cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass

            progress.empty()
            best_box.empty()
            stop_box.empty()
            if not cancelled and top_rows:
                _mode = "CUDA" if xp_dev is not np else "CPU"
                st.success(tr(
                    f"Auto tuning done ({_mode}) — {n_eval[0]} evals, "
                    f"best Total = {top_rows[0][0]:.3f}%",
                    f"自動調諧完成（{_mode}）— {n_eval[0]} 次評估，"
                    f"最佳總計 = {top_rows[0][0]:.3f}%"))

        # ── 🪜 Full Auto Tune driver (progressive coarse → fine) ────────
        def _run_progressive(*, use_cuda: bool, fit_keys: list,
                             target_pct: float):
            """Coarse → fine Full Auto Tune (progressive refinement).

            A global coarse scan over physics-informed ranges seeds a set of
            pair-wise (Zbe / Zbc / …) refinement cycles with shrinking boxes.
            Every candidate is memoised (quantised to floor/2) so no combo is
            ever re-evaluated; the best result is kept regardless of how the
            run ends (step floor / target residual / ⏹ Stop).  Selected
            parameters carry a strictly-positive lower limit (max(hard_lo,
            step floor)) so the fit can never propose 0 — deselecting a
            parameter pins it at its current value (incl. 0) instead.

            ``target_pct`` is kept in the signature for programmatic use; the
            UI card no longer exposes it and dispatches with 0.0 (= refine
            until the step floor or ⏹ Stop).

            Closes over param_rows / all_p / tuning_specs / S_raw / freq / z0 /
            model_cls / topo_key / fname / _low_perf.  Structured around the
            module-level pure helpers (_prog_axis_values / _prog_signature /
            _prog_next_bounds) so the geometry is unit-testable.
            """
            import pandas as pd

            xp = _cp if (use_cuda and _HAS_CUDA) else np
            if use_cuda and not _HAS_CUDA:
                st.warning(tr("CUDA not available — falling back to CPU.",
                              "CUDA 不可用 — 已回退至 CPU。"))
            if not hasattr(model_cls, "simulate_batch"):
                st.error(tr(
                    f"{model_cls.__name__} has no simulate_batch — "
                    "Full Auto Tune needs it.",
                    f"{model_cls.__name__} 沒有 simulate_batch — "
                    "全自動調諧需要它。"))
                return
            fit_keys = [k for k in fit_keys if k in
                        {s[0] for s in tuning_specs}]
            if not fit_keys:
                st.warning(tr(
                    "Full Auto Tune needs at least one parameter "
                    "selected under **Parameters to fit**.",
                    "全自動調諧至少需要在**要擬合的參數**下選取一個參數。"))
                return

            _global_budget = (_PROG_GLOBAL_BUDGET_GPU if xp is not np
                              else _PROG_GLOBAL_BUDGET_CPU)
            _group_budget  = (_PROG_GROUP_BUDGET_GPU if xp is not np
                              else _PROG_GROUP_BUDGET_CPU)
            if xp is not np:
                _chunk_max = _PROG_CHUNK_GPU
            else:
                # CPU: derive the chunk size from RAM actually available right
                # now instead of the fixed _PROG_CHUNK_CPU constant — same
                # per-row working-set model as _run_one_sweep's
                # per_combo_bytes. At high freq-point counts the fixed constant
                # could allocate several GB in one simulate_batch call and
                # get SIGKILLed by the Streamlit Cloud cgroup OOM-killer
                # before the try/except MemoryError path below ever runs.
                per_row_bytes = 16 * 4 * len(freq) * 12
                _chunk_max = max(256, min(_PROG_CHUNK_CPU,
                                  int(ram_available_bytes() * 0.25 // per_row_bytes)))

            # Per-fit-key metadata: label / scale / hard limits / initial box /
            # step floor.  Order = fit_keys order.
            spec_by_key = {s[0]: s for s in tuning_specs}
            n_fit  = len(fit_keys)
            labels, scales = [], []
            bounds  = {}    # key -> [lo, hi] (display units), mutated per cycle
            floors  = []    # per fit key step floor
            pos_floors = []  # per fit key strictly-positive lower limit
            hard_lims = {}
            for k in fit_keys:
                s = spec_by_key[k]
                lbl   = s[1]
                scale = s[2]
                sstep = s[5] if len(s) > 5 else None
                labels.append(lbl)
                scales.append(scale)
                cur_disp = float(all_p.get(k, 0.0)) * scale
                lo, _st, hi = informed_default_range(
                    k, lbl, cur_disp, low_perf=_low_perf, spec_step=sstep)
                # A parasitic pinned to (0,0,0) can't be fit — give it a tiny
                # informed box anyway so the axis has room to move.
                if hi <= lo:
                    _dlo, _dhi = TUNE_DEFAULT_RANGES.get(
                        _canonical_tune_key(k, lbl) or "", (0.0, 1.0))
                    lo, hi = float(_dlo), float(_dhi)
                floor_val = (float(sstep) if sstep and sstep > 0
                             else _PROG_MIN_STEP)
                floors.append(floor_val)
                h_lo, h_hi = tune_hard_limits(k, lbl)
                hard_lims[k] = (h_lo, h_hi)
                # No-zero floor: a *selected* parameter is never proposed at 0
                # (a 0-valued Cbe/tau just compensates degenerately elsewhere).
                # Keys whose hard lower limit is 0 get one step floor (>0) as
                # the effective lower limit; alpha0's 0.95 is already above.
                # Deselected params stay pinned at their current value —
                # including 0 — that's the user's escape hatch.
                pos_floor = max(float(h_lo) if h_lo is not None else 0.0,
                                floor_val)
                pos_floors.append(pos_floor)
                lo = max(float(lo), pos_floor)
                if hi <= lo:
                    # Floor swallowed the box (tiny decade range) — keep a
                    # minimal viable span above the floor.
                    hi = lo + 2.0 * floor_val
                    if h_hi is not None:
                        hi = min(hi, float(h_hi))
                bounds[k] = [float(lo), float(hi)]
            floors_arr = np.asarray(floors, dtype=np.float64)

            # ── Result table plumbing (mirrors _run_one_sweep / _run_nm) ──
            TOP_K = 100
            col_names = [
                "Total Residual (%)", "S11 (%)", "S12 (%)", "S21 (%)", "S22 (%)"]
            for r in param_rows:
                u = r["unit"]; lbl = r["label"]
                col_names.append(f"{lbl} ({u})" if u else lbl)
            sess_key = f"tune_df_{topo_key}_{fname}"
            top_rows: list = []

            def _push_topk(row: list):
                if len(top_rows) < TOP_K:
                    top_rows.append(row)
                    top_rows.sort(key=lambda r: r[0])
                    return
                if row[0] < top_rows[-1][0]:
                    top_rows[-1] = row
                    top_rows.sort(key=lambda r: r[0])

            def _persist_now():
                if not top_rows:
                    return
                try:
                    st.session_state[sess_key] = pd.DataFrame(
                        _multi_metric_top_n(np.asarray(top_rows, dtype=float),
                                             per_metric=10),
                        columns=col_names)
                except Exception:
                    pass

            # ── Refinement groups: map canonical groups onto fit keys, then
            #    chunk leftovers into pairs (spec order). ──────────────────
            _fit_set = list(fit_keys)
            _assigned = set()
            groups: list = []
            for _gname, _gkeys in _PROG_GROUPS:
                members = [k for k in _fit_set
                           if _canonical_tune_key(k, spec_by_key[k][1]) in _gkeys
                           and k not in _assigned]
                if members:
                    groups.append((_gname, members))
                    _assigned.update(members)
            _leftover = [k for k in _fit_set if k not in _assigned]
            for _i in range(0, len(_leftover), 2):
                groups.append(("misc", _leftover[_i:_i + 2]))

            # ── State / memo / UI ─────────────────────────────────────────
            memo: set = set()
            best_disp = {k: float(all_p.get(k, 0.0)) * scales[i]
                         for i, k in enumerate(fit_keys)}
            best_tot = [float("inf")]
            n_eval = [0]
            n_skip = [0]

            ui_cols = st.columns([5, 1])
            with ui_cols[0]:
                progress = st.progress(
                    0, text=tr("Full Auto Tune…", "全自動調諧…"))
            with ui_cols[1]:
                stop_box = st.empty()
            best_box = st.empty()
            stop_key = f"tune_stop_{topo_key}_{fname}_prog"
            stop_box.button(
                tr("⏹ Stop", "⏹ 停止"), key=stop_key, type="secondary",
                help=tr("Stop the calculation. Best result so far is kept.",
                        "停止計算。目前的最佳結果將被保留。"))

            last_ui = [0.0]
            _t_start = time.time()
            S_mea_dev = xp.asarray(S_raw)

            def _release():
                if xp is np:
                    gc.collect(); gc.collect(); return
                try:
                    _cp.cuda.runtime.deviceSynchronize()
                except Exception:
                    pass
                gc.collect(); gc.collect()
                try:
                    _cp.get_default_memory_pool().free_all_blocks()
                    _cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

            _PORT_NAMES = ("S11", "S12", "S21", "S22")

            def _worst_port():
                """(name, value) of the worst per-port residual on the current
                best row — inf before any result exists."""
                if not top_rows:
                    return _PORT_NAMES[0], float("inf")
                vals = top_rows[0][1:5]
                i = int(np.argmax(vals))
                return _PORT_NAMES[i], float(vals[i])

            def _ui_tick(phase, cyc, frac):
                if time.time() - last_ui[0] < 0.5:
                    return
                last_ui[0] = time.time()
                _txt = (f"Full Auto… {phase} | cycle {cyc} | "
                        f"{n_eval[0]:,} evals ({n_skip[0]:,} memo-skipped) | "
                        f"best Total {best_tot[0]:.2f}%")
                if top_rows:
                    _wn, _wv = _worst_port()
                    _txt += f" | worst {_wn} {_wv:.2f}%"
                # Any st.* call is the Stop-button cancellation point.
                progress.progress(min(1.0, max(0.0, float(frac))), text=_txt)
                if top_rows:
                    best_box.markdown(
                        _best_summary_md(pd.Series(top_rows[0], index=col_names),
                                         label=tr("Best so far", "目前最佳")),
                        unsafe_allow_html=True)
                _persist_now()

            def _row_for(cand_disp):
                """Assemble a (5 + n_params) result-table row from a fitted
                candidate (display units) + the constant params."""
                cand = {k: float(v) for k, v in zip(fit_keys, cand_disp)}
                row = [0.0, 0.0, 0.0, 0.0, 0.0]
                for pr in param_rows:
                    k = pr["key"]
                    if k in cand:
                        row.append(cand[k])
                    else:
                        row.append(float(all_p.get(k, 0.0)) * pr["scale"])
                return row

            def _eval_block(cands, phase, cyc):
                """Evaluate a (B, n_fit) host array of display-unit candidates.

                Filters memoised rows, chunks the survivors through
                simulate_batch, scores residuals, and pushes into the top-K.
                Returns the number of *new* (non-skipped) rows evaluated.
                """
                cands = np.asarray(cands, dtype=np.float64)
                if cands.ndim != 2 or cands.shape[0] == 0:
                    return 0
                B = cands.shape[0]
                # Vectorised signature buckets, then a python loop to filter.
                buckets = np.round(cands / (floors_arr * 0.5)).astype(np.int64)
                survivors = []
                for bi in range(B):
                    sig = tuple(int(x) for x in buckets[bi])
                    if sig in memo:
                        n_skip[0] += 1
                        continue
                    memo.add(sig)
                    survivors.append(cands[bi])
                if not survivors:
                    return 0
                surv = np.asarray(survivors, dtype=np.float64)
                n_new = surv.shape[0]

                start = 0
                chunk = _chunk_max
                while start < surv.shape[0]:
                    sub = surv[start:start + chunk]
                    Bc = sub.shape[0]
                    try:
                        p = dict(all_p)
                        for j, k in enumerate(fit_keys):
                            p[k] = xp.asarray(sub[:, j] / scales[j],
                                              dtype=np.float64)
                        S = model_cls.simulate_batch(p, freq, z0, xp=xp)
                        S_flat = xp.asarray(S).reshape(Bc, len(freq), 2, 2)
                        res = _port_residuals_batch(S_mea_dev, S_flat, xp)
                        if xp is np:
                            tot = np.asarray(res["Total"], dtype=float)
                            s11 = np.asarray(res["S11"], dtype=float)
                            s12 = np.asarray(res["S12"], dtype=float)
                            s21 = np.asarray(res["S21"], dtype=float)
                            s22 = np.asarray(res["S22"], dtype=float)
                        else:
                            tot = _cp.asnumpy(res["Total"])
                            s11 = _cp.asnumpy(res["S11"])
                            s12 = _cp.asnumpy(res["S12"])
                            s21 = _cp.asnumpy(res["S21"])
                            s22 = _cp.asnumpy(res["S22"])
                    except Exception as exc:
                        _is_oom = (isinstance(exc, MemoryError)
                                   or "out of memory" in str(exc).lower()
                                   or "OutOfMemoryError" in type(exc).__name__)
                        if _is_oom and chunk > 1:
                            # Halve the chunk and retry from the same offset.
                            chunk = max(1, chunk // 2)
                            _release()
                            continue
                        if _is_oom:
                            st.error(tr(
                                "Out of memory — a single candidate row "
                                "won't fit. Aborting Full Auto Tune.",
                                "記憶體不足 — 單一候選列都無法容納。"
                                "正在中止全自動調諧。"))
                            return n_new
                        raise
                    for _a in (tot, s11, s12, s21, s22):
                        _a[~np.isfinite(_a)] = np.inf
                    # Only the chunk's TOP_K lowest totals can possibly enter
                    # the (Total-sorted) global top-K, so prefilter with one
                    # argsort instead of building a Python row per candidate —
                    # keeps the host loop negligible even for 262k-row GPU
                    # chunks.
                    n_eval[0] += Bc
                    for ri in np.argsort(tot)[:TOP_K]:
                        if not np.isfinite(tot[ri]):
                            break        # sorted → the rest are inf too
                        row = _row_for(sub[ri])
                        row[0] = float(tot[ri]); row[1] = float(s11[ri])
                        row[2] = float(s12[ri]); row[3] = float(s21[ri])
                        row[4] = float(s22[ri])
                        _push_topk(row)
                        if row[0] < best_tot[0]:
                            best_tot[0] = row[0]
                            for j, k in enumerate(fit_keys):
                                best_disp[k] = float(sub[ri, j])
                    start += Bc
                    _ui_tick(phase, cyc, min(1.0, start / max(1, surv.shape[0])))
                return n_new

            # ── Cartesian grid builder within current bounds ───────────────
            def _grid(keys, n_pts_per):
                """Cartesian product over ``keys`` axis value-lists (current
                bounds, best value inserted, floor-limited).  Non-listed fit
                keys are pinned at their current best.  Returns (B, n_fit)."""
                axis_vals = []
                for k in fit_keys:
                    if k in keys:
                        i = fit_keys.index(k)
                        lo, hi = bounds[k]
                        axis_vals.append(_prog_axis_values(
                            lo, hi, n_pts_per, floors[i],
                            include=best_disp[k]))
                    else:
                        axis_vals.append(np.array([best_disp[k]],
                                                  dtype=np.float64))
                mesh = np.meshgrid(*axis_vals, indexing="ij")
                return np.stack([m.ravel() for m in mesh], axis=1)

            # ── Escape pass — wide re-scan of box-pinned params ────────────
            def _escape_pass(cyc):
                """One wide log re-scan for every fitted param pinned at a box
                edge (all fit keys when none are pinned): 1-D over a much
                wider range, then 2-D with the first group partner so
                compensating pairs can move together.  Each scanned param's
                box is re-widened so later cycles keep exploring the
                discovery.  All work goes through _eval_block (memo / top-K /
                UI tick / ⏹ Stop).  Returns True when best_tot improved."""
                _before = best_tot[0]
                pinned = []
                for i, k in enumerate(fit_keys):
                    lo, hi = bounds[k]
                    span = hi - lo
                    if (best_disp[k] <= lo + 3.0 * floors[i]
                            or best_disp[k] >= hi - _PROG_EDGE_FRAC * span):
                        pinned.append(k)
                if not pinned:
                    pinned = list(fit_keys)
                for k in pinned:
                    i = fit_keys.index(k)
                    s = spec_by_key[k]
                    sstep = s[5] if len(s) > 5 else None
                    _ilo, _istep, inf_hi = informed_default_range(
                        k, s[1], best_disp[k],
                        low_perf=_low_perf, spec_step=sstep)
                    wide_hi = max(float(inf_hi), bounds[k][1],
                                  abs(best_disp[k]) * 100.0,
                                  100.0 * floors[i])
                    _h_hi = hard_lims[k][1]
                    if _h_hi is not None:
                        wide_hi = min(wide_hi, float(_h_hi))
                    # 1-D wide scan — log spacing kicks in automatically for
                    # the many-decade span; others pinned at the current best.
                    axis_w = _prog_axis_values(
                        pos_floors[i], wide_hi, _PROG_ESCAPE_PTS,
                        floors[i], include=best_disp[k])
                    base = np.asarray([best_disp[kk] for kk in fit_keys],
                                      dtype=np.float64)
                    cands = np.tile(base, (len(axis_w), 1))
                    cands[:, i] = axis_w
                    _eval_block(cands, f"escape {k}", cyc)
                    # 2-D with the FIRST other member of k's group present in
                    # the fit scope — compensating pairs move together.
                    partner = None
                    for _gn, members in groups:
                        if k in members:
                            partner = next(
                                (m for m in members if m != k), None)
                            break
                    if partner is not None:
                        j = fit_keys.index(partner)
                        ax_k = _prog_axis_values(
                            pos_floors[i], wide_hi, 24, floors[i],
                            include=best_disp[k])
                        ax_p = _prog_axis_values(
                            bounds[partner][0], bounds[partner][1], 12,
                            floors[j], include=best_disp[partner])
                        mk, mp = np.meshgrid(ax_k, ax_p, indexing="ij")
                        cands2 = np.tile(base, (mk.size, 1))
                        cands2[:, i] = mk.ravel()
                        cands2[:, j] = mp.ravel()
                        _eval_block(cands2, f"escape {k}", cyc)
                    # Re-widen so later cycles keep exploring the discovery.
                    bounds[k] = [pos_floors[i], wide_hi]
                return best_tot[0] < _before - 1e-15

            cancelled = False
            try:
                # 0 — evaluate the current point first (baseline in the table).
                _eval_block(np.asarray([[best_disp[k] for k in fit_keys]],
                                       dtype=np.float64), "baseline", 0)

                # 1 — GLOBAL coarse pass.
                _n_pts_global = 3
                for _cand in (5, 4, 3):
                    if _cand ** n_fit <= _global_budget:
                        _n_pts_global = _cand
                        break
                if 3 ** n_fit > _global_budget:
                    # Too many dims for even a 3-point full cartesian — draw
                    # `budget` random combos from per-axis 3-point lists.
                    rng = np.random.default_rng(0)
                    axis3 = []
                    for i, k in enumerate(fit_keys):
                        lo, hi = bounds[k]
                        axis3.append(_prog_axis_values(
                            lo, hi, 3, floors[i], include=best_disp[k]))
                    cols_rand = [rng.choice(a, size=_global_budget) for a in axis3]
                    _eval_block(np.stack(cols_rand, axis=1), "global", 0)
                else:
                    _eval_block(_grid(set(fit_keys), _n_pts_global), "global", 0)

                # 2 — cycle loop, with an automatic escape phase.
                #
                # `stall` counts consecutive low-improvement cycles.  While
                # any port residual is above _PROG_PORT_GOAL, a stall (or a
                # fully-memoised cycle) triggers ONE escape: box-pinned
                # params get a wide log re-scan (alone + with a group
                # partner) and re-widened bounds; a fruitless escape falls
                # back to a single global random re-scan.  Only after that
                # may the normal termination paths end the run above the
                # goal.
                _converged = False
                stall = 0
                escape_spent = False
                for cyc in range(1, _PROG_MAX_CYCLES + 1):
                    _prev_best = best_tot[0]
                    _cycle_new = 0
                    for _gname, members in groups:
                        _k = len(members)
                        if _k == 0:
                            continue
                        _pts = max(3, int(round(_group_budget ** (1.0 / _k))))
                        _cycle_new += _eval_block(
                            _grid(set(members), _pts), _gname, cyc)

                    # Shrink / expand each axis around its best.  The no-zero
                    # floor doubles as the effective hard lower limit so an
                    # edge-expansion can never re-open the box down to 0.
                    for i, k in enumerate(fit_keys):
                        lo, hi = bounds[k]
                        h_lo, h_hi = hard_lims[k]
                        h_lo_eff = max(float(h_lo) if h_lo is not None else 0.0,
                                       pos_floors[i])
                        bounds[k] = list(_prog_next_bounds(
                            lo, hi, best_disp[k], h_lo_eff, h_hi, floors[i]))

                    # Stall tracking + worst-port status.
                    _rel_impr = ((_prev_best - best_tot[0])
                                 / max(_prev_best, 1e-9))
                    stall = 0 if _rel_impr > _PROG_STALL_REL else stall + 1
                    _wname, _wval = _worst_port()

                    # Termination: explicit target reached.
                    if target_pct > 0 and best_tot[0] <= target_pct:
                        _converged = True
                        break

                    # Escape phase — refinement is stuck above the per-port
                    # goal (stalled, or the grids collapsed onto memoised
                    # points) and the escape hasn't been spent yet.
                    if (not escape_spent and _wval > _PROG_PORT_GOAL
                            and (stall >= _PROG_STALL_CYCLES
                                 or _cycle_new == 0)):
                        if _escape_pass(cyc):
                            stall = 0
                            continue      # keep cycling from the discovery
                        # Escape found nothing — one global random re-scan
                        # over the (re-widened) bounds, then let the normal
                        # termination paths end the run.
                        escape_spent = True
                        rng_esc = np.random.default_rng(1)
                        axis5 = []
                        for i, k in enumerate(fit_keys):
                            lo, hi = bounds[k]
                            axis5.append(_prog_axis_values(
                                lo, hi, 5, floors[i]))
                        _n_rand = max(1, _global_budget // 4)
                        cols_esc = [rng_esc.choice(a, size=_n_rand)
                                    for a in axis5]
                        _before_r = best_tot[0]
                        _eval_block(np.stack(cols_esc, axis=1),
                                    "escape rescan", cyc)
                        if best_tot[0] < _before_r - 1e-15:
                            stall = 0
                        continue

                    if _cycle_new == 0:
                        # Planned cycle produced no new evaluations and no
                        # escape is available (spent, or the per-port goal
                        # is met).  Stop.
                        _converged = True
                        break
                    _all_at_floor = all(
                        (bounds[k][1] - bounds[k][0]) <= 2.0 * floors[i] + 1e-30
                        for i, k in enumerate(fit_keys))
                    if (_all_at_floor and _rel_impr < 1e-3
                            and (_wval <= _PROG_PORT_GOAL or escape_spent)):
                        _converged = True
                        break

                _persist_now()
                progress.empty(); best_box.empty(); stop_box.empty()
                st.session_state[f"tune_elapsed_{topo_key}_{fname}"] = (
                    time.time() - _t_start)
                _wname, _wval = _worst_port()
                if _wval <= _PROG_PORT_GOAL:
                    st.success(tr(
                        f"Full Auto Tune done — {n_eval[0]:,} evals, "
                        f"best Total = {best_tot[0]:.2f}% "
                        f"(all ports ≤ {_PROG_PORT_GOAL:.0f}%)",
                        f"全自動調諧完成 — {n_eval[0]:,} 次評估，"
                        f"最佳總計 = {best_tot[0]:.2f}%"
                        f"（所有埠 ≤ {_PROG_PORT_GOAL:.0f}%）"))
                else:
                    st.info(tr(
                        f"Full Auto Tune stopped above the per-port goal — "
                        f"best kept (worst {_wname} {_wval:.2f}%, Total "
                        f"{best_tot[0]:.2f}%, {n_eval[0]:,} evals). "
                        f"Re-running Evaluate continues from the best values "
                        f"after 🏆 Use best values.",
                        f"全自動調諧已停止於高於各埠目標之處 — "
                        f"已保留最佳結果（最差 {_wname} {_wval:.2f}%，總計 "
                        f"{best_tot[0]:.2f}%，{n_eval[0]:,} 次評估）。"
                        f"按下🏆使用最佳值後，重新執行評估將由最佳值繼續。"))
            except BaseException as exc:
                _is_rerun = (_RerunException is not None
                             and isinstance(exc, _RerunException))
                _is_stop  = (_StopException is not None
                             and isinstance(exc, _StopException))
                _name = type(exc).__name__
                if _is_rerun or _is_stop or _name in (
                        "RerunException", "StopException"):
                    cancelled = True
                    _persist_now()
                    try:
                        S_mea_dev = None
                    except Exception:
                        pass
                    _release()
                    raise
                st.error(tr(f"Full Auto Tune failed: {exc!r}",
                            f"全自動調諧失敗：{exc!r}"))
            finally:
                _persist_now()
                try:
                    S_mea_dev = None
                except Exception:
                    pass
                _release()

        # ── Helper: column-name lookup for a tuning_specs row ──────────
        def _col_name(row):
            return f"{row['label']} ({row['unit']})" if row["unit"] else row["label"]

        # ── Dispatch ────────────────────────────────────────────────────
        if cpu_clicked or cuda_clicked:
            _run_one_sweep(use_cuda=bool(cuda_clicked), sort_metric="Total")
        elif opt_cpu_clicked or opt_cuda_clicked:
            # Recursive bisection: each iteration subsamples N_SUB evenly-
            # spaced points per swept parameter inside the current range, runs
            # the sweep, picks the top-2, and narrows the range to between
            # those two values.  Recursion stops once a final refinement at
            # the user's chosen step fits inside MAX_FINAL combos — that
            # final refinement is then run as the closing pass.
            #
            # N_SUB adapts to the dimensionality so per-iter cost stays
            # tractable: 5^13 ≈ 1.2 B combos is unreasonable, but 3^13 ≈
            # 1.6 M is fine and the box still narrows (just by ½ per iter
            # instead of ¼).  We pick the largest N_SUB ∈ {3, 4, 5} whose
            # full grid fits MAX_PER_ITER.
            _use_cuda      = bool(opt_cuda_clicked)
            MAX_FINAL      = 5_000_000
            # Per-iter cap: GPU can absorb ~100M combos in seconds, CPU
            # struggles past ~10M.  Both still drop to N_SUB=3 for very
            # high-dim sweeps — that's expected and still cheap.
            MAX_PER_ITER   = 100_000_000 if _use_cuda else 10_000_000
            # 3-point bisection halves the box per iter; 5-point quarters
            # it.  Going from a width of ~1 to ~1e-4 takes 14 iters at 3
            # points, 7 iters at 5 points — bumped to 30 to leave headroom.
            MAX_ITERS      = 30

            current_ranges = {}   # key -> (lo, hi) — search box this iter
            user_steps     = {}   # key -> float    — step from the row UI
            rows_by_key    = {}
            for row in param_rows:
                if not row["enabled"]:
                    continue
                full = np.asarray(row["sweep"], dtype=np.float64)
                if len(full) == 0:
                    continue
                current_ranges[row["key"]] = (float(full[0]), float(full[-1]))
                kp = f"tune_{topo_key}_{row['key']}_{fname}"
                user_steps[row["key"]] = (
                    float(st.session_state.get(f"{kp}_step", 0.0)) or 0.0)
                rows_by_key[row["key"]] = row

            if not current_ranges:
                st.warning(tr(
                    "Optimized mode needs at least one parameter with "
                    "the **Sweep** checkbox enabled.",
                    "最佳化模式至少需要一個已啟用**掃描**核取方塊的參數。"))
            else:
                # Adaptive N_SUB — largest in {3,4,5} that fits MAX_PER_ITER.
                # If even 3^N exceeds the budget we still use 3 and warn.
                _n_swept = len(current_ranges)
                N_SUB = 3
                for _cand in (5, 4, 3):
                    if (_cand ** _n_swept) <= MAX_PER_ITER:
                        N_SUB = _cand
                        break
                _per_iter = N_SUB ** _n_swept
                if N_SUB == 3 and _per_iter > MAX_PER_ITER:
                    st.warning(tr(
                        f"3-point subsample of {_n_swept} swept params is "
                        f"{_per_iter:,} combos — over the {MAX_PER_ITER:,} "
                        "per-iter target. The run will still proceed but "
                        "each iteration will be slow. Consider sweeping "
                        "fewer parameters at once.",
                        f"對 {_n_swept} 個掃描參數進行 3 點子取樣即為 "
                        f"{_per_iter:,} 種組合 — 超過每次疊代 "
                        f"{MAX_PER_ITER:,} 的目標。仍會繼續執行，但每次疊代"
                        "會較慢。建議一次掃描較少的參數。"))
                else:
                    st.caption(tr(
                        f"🎯 N_SUB = **{N_SUB}** points/param "
                        f"({_per_iter:,} combos per iter for {_n_swept} "
                        "swept params)",
                        f"🎯 N_SUB = **{N_SUB}** 點/參數"
                        f"（{_n_swept} 個掃描參數，每次疊代 "
                        f"{_per_iter:,} 種組合）"))
                def _estimate_final_combos(ranges, steps):
                    """How many combos would a brute-force sweep of these
                    ranges at the user's steps generate?"""
                    n = 1
                    for k, (lo, hi) in ranges.items():
                        step = steps.get(k, 0.0)
                        if step <= 0 or abs(hi - lo) < 1e-15:
                            pts = 1
                        else:
                            pts = max(1, int(round(abs(hi - lo) / step)) + 1)
                        n *= pts
                    return n

                # Short-circuit: if the user's full grid already fits the
                # final-refine budget, skip the subsample iterations and run
                # the brute force directly. No point doing 5^N subsampling
                # when we could just sweep everything.
                _initial_full = _estimate_final_combos(current_ranges, user_steps)
                _stop_recursion = False
                _final_iter_idx = None
                _skip_recursion = (_initial_full <= MAX_FINAL)
                if _skip_recursion:
                    st.caption(tr(
                        f"🎯 Full sweep is already **{_initial_full:,}** "
                        f"combos ≤ {MAX_FINAL:,} — skipping subsample, "
                        "running brute force at the user step directly.",
                        f"🎯 全掃描已為 **{_initial_full:,}** 組合 "
                        f"≤ {MAX_FINAL:,} — 略過子取樣，"
                        "直接以使用者步進執行暴力法。"))
                    _final_iter_idx = 0
                    iter_idx = 0
                # The recursion loop only runs when we actually need it.
                # Wrapped in `if not _skip_recursion` rather than `else:` so
                # the existing for-else block keeps working at its original
                # indentation.
                _do_loop = not _skip_recursion
                _loop_converged = False  # set True on any clean break
                for iter_idx in range(1, MAX_ITERS + 1) if _do_loop else range(0):
                    # Build 5-sample lists within the current ranges
                    sub = {}
                    for k, (lo, hi) in current_ranges.items():
                        if abs(hi - lo) < 1e-15:
                            sub[k] = np.array([lo], dtype=np.float64)
                        else:
                            sub[k] = np.linspace(lo, hi, N_SUB)

                    est_final = _estimate_final_combos(current_ranges, user_steps)
                    st.caption(tr(
                        f"🎯 Optimized iter {iter_idx} — current final-refine "
                        f"estimate: **{est_final:,}** combos "
                        f"(target ≤ {MAX_FINAL:,})",
                        f"🎯 最佳化疊代 {iter_idx} — 目前最終細化"
                        f"預估：**{est_final:,}** 組合"
                        f"（目標 ≤ {MAX_FINAL:,}）"))

                    _run_one_sweep(
                        use_cuda=_use_cuda, sweep_lists_override=sub,
                        sort_metric="Total",
                        phase_label=tr(
                            f"Iter {iter_idx} (subsample {N_SUB})",
                            f"疊代 {iter_idx}（子取樣 {N_SUB}）"),
                        phase_suffix=f"_optP{iter_idx}")

                    _df = st.session_state.get(f"tune_df_{topo_key}_{fname}")
                    if _df is None or len(_df) == 0:
                        # No finite residuals — abandon bisection but
                        # still run the closing pass on current_ranges so
                        # the user gets a brute-force result.
                        st.warning(tr(
                            f"Iteration {iter_idx} produced no finite "
                            "residuals — running the final refinement on "
                            "the current range without further narrowing.",
                            f"第 {iter_idx} 次疊代未產生有限殘差 — "
                            "將在目前範圍上執行最終細化，不再進一步縮小。"))
                        _final_iter_idx = iter_idx
                        _loop_converged = True
                        break

                    # Narrow to the box between the top-2 combos.  With
                    # only one survivor, centre a quarter-width box on it
                    # so the optimised pass keeps making progress.
                    single = len(_df) < 2
                    new_ranges = {}
                    for k, (lo_old, hi_old) in current_ranges.items():
                        col = _col_name(rows_by_key[k])
                        if col not in _df.columns:
                            new_ranges[k] = (lo_old, hi_old)
                            continue
                        v0 = float(_df.iloc[0][col])
                        if single:
                            half_w = abs(hi_old - lo_old) * 0.25
                            new_ranges[k] = (max(lo_old, v0 - half_w),
                                             min(hi_old, v0 + half_w))
                        else:
                            v1 = float(_df.iloc[1][col])
                            new_ranges[k] = (min(v0, v1), max(v0, v1))

                    # Bail if the box can't shrink anymore (guards against
                    # the corner case where the top-2 are identical).
                    if all(abs(new_ranges[k][1] - new_ranges[k][0]) < 1e-15
                           for k in new_ranges):
                        current_ranges = new_ranges
                        _final_iter_idx = iter_idx
                        _loop_converged = True
                        break

                    current_ranges = new_ranges

                    # Done once the final brute-force fits the budget
                    if _estimate_final_combos(current_ranges, user_steps) \
                            <= MAX_FINAL:
                        _final_iter_idx = iter_idx
                        _loop_converged = True
                        break
                # for-else replaced by explicit flag so the warning only
                # fires when the loop actually ran *and* didn't converge.
                if _do_loop and not _loop_converged:
                    st.warning(tr(
                        f"Reached MAX_ITERS={MAX_ITERS} without converging "
                        f"under {MAX_FINAL:,} combos — running the final "
                        "refinement on the last narrowed range anyway.",
                        f"已達 MAX_ITERS={MAX_ITERS} 仍未收斂至 "
                        f"{MAX_FINAL:,} 組合以下 — 仍會在最後縮小的範圍上"
                        "執行最終細化。"))

                # ── Closing pass: brute-force at the user's step ─────────
                if not _stop_recursion:
                    final_lists = {}
                    for k, (lo, hi) in current_ranges.items():
                        step = user_steps.get(k, 0.0)
                        if step <= 0 or abs(hi - lo) < 1e-15:
                            final_lists[k] = np.array([lo], dtype=np.float64)
                        else:
                            final_lists[k] = _make_sweep_values(lo, hi, step)
                    final_total = 1
                    for arr in final_lists.values():
                        final_total *= len(arr)
                    st.caption(tr(
                        f"🎯 Final refine — running **{final_total:,}** "
                        "combos at user-defined step.",
                        f"🎯 最終細化 — 正在以使用者定義步進執行 "
                        f"**{final_total:,}** 種組合。"))
                    _run_one_sweep(
                        use_cuda=_use_cuda, sweep_lists_override=final_lists,
                        sort_metric="Total",
                        phase_label=tr(
                            f"Final refine ({final_total:,} combos)",
                            f"最終細化（{final_total:,} 組合）"),
                        phase_suffix="_optFinal")
        elif prio_cpu_clicked or prio_cuda_clicked:
            _run_one_sweep(use_cuda=bool(prio_cuda_clicked),
                            sort_metric=str(prio_metric),
                            phase_label=tr(f"Prioritize {prio_metric}",
                                           f"優先排序 {prio_metric}"),
                            phase_suffix=f"_prio{prio_metric}")
        elif bal_cpu_clicked or bal_cuda_clicked:
            _res_thr = float(bal_res_threshold) if _bal_use_res else None
            _run_one_sweep(
                use_cuda=bool(bal_cuda_clicked),
                sort_metric="Total",
                phase_label=tr(f"Balance ≤{bal_dev_threshold:.2f}%",
                               f"平衡 ≤{bal_dev_threshold:.2f}%"),
                phase_suffix="_bal",
                dev_threshold=float(bal_dev_threshold),
                res_threshold=_res_thr,
            )
        elif prog_cpu_clicked or prog_cuda_clicked:
            _run_progressive(
                use_cuda=bool(prog_cuda_clicked),
                fit_keys=list(prog_fit_keys),
                target_pct=0.0)
        # elif auto_cpu_clicked or auto_cuda_clicked:
        #     _run_nelder_mead(
        #         max_iter=int(auto_max_iter),
        #         restart=bool(auto_restart),
        #         use_cuda=bool(auto_cuda_clicked),
        #     )

        # ── 📈 Parameter sensitivity (toggle-collapsed section) ────────
        # Same toggle-not-expander workaround as Semi-Auto Tune (Streamlit
        # forbids nested expanders).  Reads the persisted results df, so a
        # fresh run's charts appear on the rerun after it finishes.
        sens_open = st.toggle(
            tr("📈 Parameter sensitivity", "📈 參數敏感度"),
            key=f"tune_sens_open_{topo_key}_{fname}",
            help=tr(
                "Per-parameter residual curves swept around the best "
                "result, other parameters held at their best values.  "
                "Curves are plotted for parameters with a ticked Sweep "
                "checkbox in Semi-Auto Tune.",
                "在最佳結果附近，逐一掃描每個參數繪出殘差曲線，"
                "其餘參數固定在其最佳值。僅繪製半自動調諧中"
                "已勾選「掃描」的參數。"))
        if sens_open:
            with st.container(border=True):
                df_sens = st.session_state.get(f"tune_df_{topo_key}_{fname}")
                ticked_specs = [
                    spec for spec in tuning_specs
                    if st.session_state.get(
                        f"tune_{topo_key}_{spec[0]}_{fname}_chk", False)
                ]
                if df_sens is None:
                    st.caption(tr(
                        "Run a tune first — the sensitivity plots "
                        "sweep each parameter around the best result.",
                        "請先執行一次調諧 — 敏感度圖會在最佳結果附近"
                        "掃描每個參數。"))
                elif not ticked_specs:
                    st.caption(tr(
                        "Tick at least one **Sweep** checkbox in "
                        "Semi-Auto Tune to choose which parameters "
                        "to plot.",
                        "請在半自動調諧中至少勾選一個**掃描**核取方塊，"
                        "以選擇要繪製的參數。"))
                else:
                    best = df_sens.iloc[0]
                    st.markdown(tr(
                        "**Parameter sensitivity** *(other params held at best values)*",
                        "**參數敏感度** *（其餘參數固定於最佳值）*"))
                    # Build baseline param dict from best row (in SI).
                    # Missing column (stale parameter-less table) → keep
                    # the all_p value.
                    best_p = dict(all_p)
                    for spec in tuning_specs:
                        key, label, scale = spec[0], spec[1], spec[2]
                        unit = spec[3] if len(spec) > 3 else ""
                        col_name = f"{label} ({unit})" if unit else label
                        if col_name in best.index:
                            best_p[key] = float(best[col_name]) / scale

                    _sp_colors = {"S11": "#1f77b4", "S12": "#d62728",
                                  "S21": "#2ca02c", "S22": "#ff7f0e"}

                    # Lay sensitivity charts out in a 2-column grid.  Even
                    # index → left column, odd index → right column.  An
                    # odd total leaves the final chart alone in the left
                    # column (right column stays empty).
                    _sens_col_pair = None  # current (left_col, right_col)
                    _sens_rendered = 0

                    for spec in ticked_specs:
                        key, label, scale = spec[0], spec[1], spec[2]
                        unit = spec[3] if len(spec) > 3 else ""
                        kp = f"tune_{topo_key}_{key}_{fname}"
                        sweep_min = float(st.session_state.get(f"{kp}_min", 0))
                        sweep_step = float(st.session_state.get(f"{kp}_step", 0))
                        sweep_max = float(st.session_state.get(f"{kp}_max", 0))
                        sweep_vals = _make_sweep_values(sweep_min, sweep_max, sweep_step)

                        # Cap the sensitivity chart at 50 points — when the
                        # user-defined sweep is denser, drop to 50 evenly-spaced
                        # samples (always keeping the endpoints).
                        SENS_MAX_PTS = 50
                        if len(sweep_vals) > SENS_MAX_PTS:
                            idxs = np.linspace(0, len(sweep_vals) - 1,
                                                SENS_MAX_PTS).round().astype(int)
                            idxs = np.unique(idxs)
                            sweep_vals = sweep_vals[idxs]

                        if len(sweep_vals) < 2:
                            continue  # nothing to plot for a single point

                        # Residuals for the whole sweep in ONE batched call (the
                        # swept param as a length-M array) instead of M sequential
                        # simulate_vec calls — keeps the sensitivity panel cheap even
                        # when many parameters are ticked (matters most for the custom
                        # model, whose per-call path crosses the Rust/PyO3 boundary).
                        res_s11 = res_s12 = res_s21 = res_s22 = None
                        if hasattr(model_cls, "simulate_batch"):
                            try:
                                p_b = dict(best_p)
                                p_b[key] = np.asarray(sweep_vals, dtype=float) / scale
                                S_b = np.asarray(
                                    model_cls.simulate_batch(p_b, freq, z0, xp=np)
                                ).reshape(len(sweep_vals), len(freq), 2, 2)
                                rb = _port_residuals_batch(S_raw, S_b, np)
                                res_s11 = np.asarray(rb["S11"], dtype=float)
                                res_s12 = np.asarray(rb["S12"], dtype=float)
                                res_s21 = np.asarray(rb["S21"], dtype=float)
                                res_s22 = np.asarray(rb["S22"], dtype=float)
                                for _a in (res_s11, res_s12, res_s21, res_s22):
                                    _a[~np.isfinite(_a)] = float("inf")
                            except Exception:
                                res_s11 = None       # fall back to the per-point loop

                        if res_s11 is None:
                            res_s11 = np.zeros(len(sweep_vals))
                            res_s12 = np.zeros(len(sweep_vals))
                            res_s21 = np.zeros(len(sweep_vals))
                            res_s22 = np.zeros(len(sweep_vals))
                            _sens_has_vec = hasattr(model_cls, "simulate_vec")
                            for vi, sv in enumerate(sweep_vals):
                                p = dict(best_p)
                                p[key] = sv / scale  # convert display -> SI
                                try:
                                    if _sens_has_vec:
                                        S_sim = model_cls.simulate_vec(p, freq, z0, xp=np)
                                    else:
                                        S_sim = model_cls.simulate(p, freq, z0)
                                except Exception:
                                    S_sim = None
                                if S_sim is None:
                                    res_s11[vi] = res_s12[vi] = float("inf")
                                    res_s21[vi] = res_s22[vi] = float("inf")
                                else:
                                    r = _port_residuals(S_raw, S_sim)
                                    res_s11[vi] = r["S11"]
                                    res_s12[vi] = r["S12"]
                                    res_s21[vi] = r["S21"]
                                    res_s22[vi] = r["S22"]

                        x_label = f"{label} ({unit})" if unit else label
                        fig = go.Figure()
                        for sp_name, y_data in [("S11", res_s11), ("S12", res_s12),
                                                ("S21", res_s21), ("S22", res_s22)]:
                            fig.add_trace(go.Scattergl(
                                x=sweep_vals, y=y_data, mode="lines+markers",
                                name=sp_name,
                                line=dict(color=_sp_colors[sp_name], width=2),
                                marker=dict(size=4),
                            ))
                        fig.update_layout(
                            title=tr(f"Sensitivity -- {x_label}",
                                     f"敏感度 -- {x_label}"),
                            xaxis_title=x_label,
                            yaxis_title=tr("Residual (%)", "殘差 (%)"),
                            height=350,
                            margin=dict(l=50, r=30, t=40, b=50),
                            legend=dict(orientation="h", y=1.12),
                            hovermode="x unified",
                        )
                        # Open a fresh 2-column pair every even-indexed chart.
                        if _sens_rendered % 2 == 0:
                            _sens_col_pair = st.columns(2)
                        _target = _sens_col_pair[_sens_rendered % 2]
                        with _target:
                            plotly_with_dl(
                                fig,
                                key=f"tune_sens_{topo_key}_{key}_{fname}",
                                filename=f"tune_sens_{topo_key}_{key}_{fname}")
                        _sens_rendered += 1

        # ── Results — evaluated-in line + best residual + top-K table ──
        df = st.session_state.get(f"tune_df_{topo_key}_{fname}")
        if df is not None:
            best = df.iloc[0]
            _elapsed = st.session_state.get(f"tune_elapsed_{topo_key}_{fname}")
            if _elapsed is not None:
                st.markdown(
                    f"**{tr('Evaluated in', '評估耗時')} "
                    f"{_fmt_eval_time(_elapsed)}**")
            st.markdown(
                _best_summary_md(best, label=tr("Best residual", "最佳殘差")),
                unsafe_allow_html=True,
            )

            # "Use best values" — push best-row params into the fine-tune widgets.
            # Uses on_click callback so writes happen *before* widgets re-render.
            def _apply_best(best_row, specs, topo, fn, low_perf):
                for spec in specs:
                    key, label, scale = spec[0], spec[1], spec[2]
                    unit = spec[3] if len(spec) > 3 else ""
                    spec_step = spec[5] if len(spec) > 5 else None
                    col_name = f"{label} ({unit})" if unit else label
                    # A stale results table (e.g. persisted before the
                    # closed-path param_rows fix) may lack parameter
                    # columns — skip those instead of KeyError'ing.
                    if col_name not in best_row.index:
                        continue
                    disp_val = float(best_row[col_name])
                    st.session_state[f"sim_{topo}_{key}_{fn}"] = disp_val
                    kp = f"tune_{topo}_{key}_{fn}"
                    # Re-seed the sweep box around the newly-applied best value
                    # from the physics-informed ranges (widened to include it).
                    d_min, d_step, d_max = informed_default_range(
                        key, label, disp_val,
                        low_perf=low_perf, spec_step=spec_step)
                    st.session_state[f"{kp}_min"] = d_min
                    st.session_state[f"{kp}_step"] = d_step
                    st.session_state[f"{kp}_max"] = d_max

            st.container(key=f"hbt_amber_best_{topo_key}").button(
                tr("🏆 Use best values", "🏆 使用最佳值"),
                key=f"tune_best_{topo_key}_{fname}",
                on_click=_apply_best,
                args=(best, tuning_specs, topo_key, fname, _low_perf))

            st.dataframe(df, width="stretch", hide_index=True)

            # Excel download
            buf = BytesIO()
            df.to_excel(buf, index=False, engine="openpyxl")
            buf.seek(0)
            st.download_button(
                tr("📥 Download as Excel", "📥 下載為 Excel"),
                data=buf,
                file_name=f"tuning_{topo_key}_{fname}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"tune_dl_{topo_key}_{fname}",
            )



# ════════════════════════════════════════════════════════════════════════════════
# SSMModelTemplate — shared scaffold for concrete model classes
# ════════════════════════════════════════════════════════════════════════════════
#
# This is a *mixin-style* parent: it provides default implementations of
# simulate_vec, simulate_batch, render_override_and_smith, and
# render_results_table that work for any model whose forward simulation
# follows the standard pad→lead→intrinsic chain.
#
# Concrete model classes inherit from BOTH this template AND AbstractSSMModel
# (the ABC contract in models/__init__.py).  Python's MRO combines them:
#
#     class MyModel(SSMModelTemplate, AbstractSSMModel):
#         NAME, SHORT, TOPOLOGY_CHAR = "...", "...", "..."
#         _INT_SPECS = [...]   # internal param specs (key, label, scale, unit, ...)
#         _EXT_SPECS = [...]   # external param specs
#         _Y_INT_VEC_FN     = staticmethod(_Y_int_xxx_vec)
#         _Y_INT_BATCH_FN   = staticmethod(_Y_int_xxx_batch)
#         _SIM_WRAP_VEC_FN  = staticmethod(_sim_wrap_vec)   # model's wrap
#         _SIM_WRAP_BATCH_FN= staticmethod(_sim_wrap_batch) # model's wrap
#
#         @classmethod
#         def _do_override_ui(cls, fname, calc_vals, cache_ctx=None): ...  # call model's _override_ui
#         @classmethod
#         def _render_topology(cls, all_p, fname): ...      # render topology illus
#         @classmethod
#         def _results_rows(cls, params): ...               # list[(sym, val, unit)]
#         @classmethod
#         def _render_results_trace(cls): pass              # optional formula trace
#
#         # Still required by AbstractSSMModel (model-specific math):
#         extract, simulate, reextract
#
# The template intentionally does NOT inherit AbstractSSMModel so it can live
# in base_ui.py without a circular import — concrete classes inherit both.

class SSMModelTemplate:
    """
    Shared scaffold for HBT small-signal model classes.

    Provides default bodies for the UI-side and forward-sim wrapper methods;
    concrete subclasses supply the math kernels and per-model UI bits via
    class attributes and a handful of classmethod hooks.  See the module
    docstring above for the contract.
    """

    # Subclasses must set these:
    _INT_SPECS: list      = []
    _EXT_SPECS: list      = []
    _Y_INT_VEC_FN         = None    # staticmethod or plain function
    _Y_INT_BATCH_FN       = None
    _SIM_WRAP_VEC_FN      = None
    _SIM_WRAP_BATCH_FN    = None
    # Pad-spec list used by the tuning expander.  Default is the shared
    # PAD_SPECS; override to customise displayed pad-parameter labels
    # (e.g. XuModel relabels Cpce → "Cpce / Cpad").
    _TUNING_PAD_SPECS     = None    # falls back to PAD_SPECS in render_override_and_smith

    # Default: no pre-bakeable sub-networks.  Concrete subclasses override
    # ``STATIC_SUBNETWORKS`` with a dict of {subnet_name: frozenset(deps)}.
    STATIC_SUBNETWORKS: dict = {}

    # ── Pre-bake truth table ──────────────────────────────────────────────────

    @classmethod
    def prebake_static_keys(cls, swept_keys):
        """Given the iterable of *swept* (i.e. changing) param keys, return
        the list of pre-bakeable sub-network names for this model.  A sub-
        network is pre-bakeable iff none of the keys it depends on are
        swept.  Caller can use this to decide which entries to populate
        in the ``cache`` argument to ``simulate_batch``."""
        if not cls.STATIC_SUBNETWORKS:
            return []
        swept_set = set(swept_keys)
        return [name for name, deps in cls.STATIC_SUBNETWORKS.items()
                if not (deps & swept_set)]

    @classmethod
    def build_static_cache(cls, all_p, freq, *, xp=None, swept_keys=()):
        """Build the ``cache`` dict (matching what ``_sim_wrap_batch`` reads)
        for the given fixed-param baseline and the set of params being
        swept.  Concrete subclasses override to fill in model-specific
        sub-networks (Zbe / Zbc / alpha / Y_extr / etc.).

        Default implementation only handles the shared sub-networks
        (``Y_pad``, ``Z_ser``) — enough to give the live mode a meaningful
        speed-up even on models without intrinsic pre-bake support.
        """
        from ..helpers.deembed_math import build_Y_pad_batch, build_Z_ser_batch
        if xp is None:
            xp = np
        omega = xp.asarray(2 * np.pi * np.asarray(freq), dtype=float)
        N = int(omega.shape[0])
        cache: dict = {"omega": omega, "_cdtype": np.complex128}
        static_p = dict(all_p)
        pbk = set(cls.prebake_static_keys(swept_keys))
        if "Y_pad" in pbk and not (cls.STATIC_SUBNETWORKS.get("Y_pad", set()) & set(swept_keys)):
            try:
                cache["Y_pad"] = build_Y_pad_batch(static_p, omega, 1, N, xp)
            except Exception:
                pass
        if "Z_ser" in pbk and not (cls.STATIC_SUBNETWORKS.get("Z_ser", set()) & set(swept_keys)):
            try:
                cache["Z_ser"] = build_Z_ser_batch(static_p, omega, 1, N, xp)
            except Exception:
                pass
        # Concrete subclasses extend via _build_intrinsic_static_cache().
        try:
            cls._build_intrinsic_static_cache(static_p, omega, cache, xp, pbk)
        except (AttributeError, NotImplementedError):
            pass
        return cache

    @classmethod
    def _build_intrinsic_static_cache(cls, p, omega, cache, xp, prebakeable):
        """Override in concrete classes to populate intrinsic sub-networks
        (Y_extr / Zbe / Zbc / alpha / Ybe / Ybc / gm / T_int_planes / etc.).
        Default is no-op."""
        return

    # ── Forward simulation ────────────────────────────────────────────────────

    @classmethod
    def simulate_vec(cls, params, freq, z0=50.0, xp=None):
        """Vectorised simulate — no per-freq loop.  Pass xp=cupy for GPU."""
        if xp is None:
            xp = np
        return cls._SIM_WRAP_VEC_FN(cls._Y_INT_VEC_FN, params, freq, z0, xp)

    @classmethod
    def simulate_batch(cls, params, freq, z0=50.0, xp=None, cache=None):
        """Batched simulate over (param_combo × freq).  Pass xp=cupy for GPU.

        params dict values may be scalars or (B,) arrays.
        Returns (B, N_freq, 2, 2) on the *xp* device (no host transfer).
        Optional ``cache`` carries pre-computed constant sub-networks.

        Phase 2 opt-in
        --------------
        When the env var ``HBT_USE_RUST_SIM_BATCH=1`` is set AND the
        Rust crate is loaded AND a Rust kernel exists for this model's
        ``SHORT`` identifier, the CPU path (xp is numpy) routes through
        the Rust end-to-end batched simulator instead of the NumPy
        composition chain.  The CUDA path (xp is cupy) stays on cupy.

        Cache + Rust
        ------------
        When a non-empty ``cache`` is provided, Rust still runs (the
        whole composition is so much faster than NumPy that the cache
        savings can't beat it).  The cache is preserved for the NumPy
        FALLBACK path so a Rust failure (e.g. exotic pad mode raises
        ``NotImplementedError``) still gets the pre-bake speedup.

        The default behaviour (env var unset) is identical to before:
        CPU goes through ``cls._SIM_WRAP_BATCH_FN`` exactly as it does
        today, with no measurable overhead from the opt-in check.
        """
        if xp is None:
            xp = np

        # Phase 2 dispatch — opt-in, CPU-only.  The lazy import keeps
        # this branch zero-cost when Phase 2 isn't activated.
        if xp is np:
            try:
                from ..helpers.rust_kernels import (
                    HAS_RUST as _HAS_RUST,
                    SIM_FOR_TOPOLOGY as _SIM_FOR_TOPOLOGY,
                    _phase2_dispatch_enabled as _phase2_on,
                )
            except Exception:                              # pragma: no cover
                _HAS_RUST = False
                _SIM_FOR_TOPOLOGY = {}

                def _phase2_on() -> bool:
                    return False

            if _HAS_RUST and _phase2_on():
                rust_wrapper = _SIM_FOR_TOPOLOGY.get(cls.SHORT)
                if rust_wrapper is not None:
                    # NumPy fallback closure — preserves the ORIGINAL
                    # cache so the pre-bake speedup isn't lost if Rust
                    # raises (NotImplementedError for exotic pad modes
                    # is the common case).
                    _saved_cache = cache
                    def _np_fallback(_p, _f, _z0):
                        return cls._SIM_WRAP_BATCH_FN(
                            cls._Y_INT_BATCH_FN, _p, _f, _z0, np,
                            _saved_cache)
                    return rust_wrapper(params, freq, z0,
                                         np_fallback=_np_fallback)

        return cls._SIM_WRAP_BATCH_FN(cls._Y_INT_BATCH_FN, params, freq, z0, xp, cache)

    # ── Cached simulate-vec (used by render_override_and_smith) ───────────────

    @classmethod
    def _cached_simulate_vec(cls, all_p, freq, z0, fname):
        """Hash-cached wrapper around ``simulate_vec``.

        Skips re-simulation when the parameter dict + freq length match the
        last invocation.  Errors are caught and reported in-place; the cache
        is populated with a NaN array so the rest of the UI keeps rendering.
        """
        cache_key = f"sim_result_{cls.SHORT}_{fname}"
        hash_key  = f"sim_phash_{cls.SHORT}_{fname}"
        cur_hash  = params_hash({k: str(v) for k, v in
                                  {**all_p, "__nf": len(freq)}.items()})

        def _run():
            with st.spinner(tr(f"Simulating {cls.NAME}…",
                               f"正在模擬 {cls.NAME}…")):
                try:
                    return cls.simulate_vec(all_p, freq, z0)
                except Exception as e:
                    st.error(tr(f"Simulation error ({cls.NAME}): {e}",
                                f"模擬錯誤（{cls.NAME}）：{e}"))
                    return np.full((len(freq), 2, 2), np.nan + 0j)

        if st.session_state.get(hash_key) != cur_hash:
            S_sim = _run()
            st.session_state[cache_key] = S_sim
            st.session_state[hash_key]  = cur_hash
        else:
            S_sim = st.session_state.get(cache_key)
            if S_sim is None or S_sim.shape[0] != len(freq):
                S_sim = _run()
                st.session_state[cache_key] = S_sim
                st.session_state[hash_key]  = cur_hash
        return S_sim

    # ── UI scaffolding ────────────────────────────────────────────────────────

    @classmethod
    def render_results_table(cls, params):
        """Render JUST the scalar-parameter dataframe.

        The formula-trace expander is rendered separately via
        :meth:`render_formula_trace` so the call site can place both
        side-by-side in their own columns / expanders.
        """
        import pandas as pd
        rows = cls._results_rows(params)
        st.dataframe(pd.DataFrame(rows, columns=["Symbol", "Value", "Unit"]),
                     width="stretch", hide_index=True)

    @classmethod
    def render_formula_trace(cls):
        """Render the 📐 Full formula trace expander, if the subclass
        provides one.  Default dispatches to the private template hook
        ``_render_results_trace`` — subclasses (Cheng T / π, Xu T)
        override that hook to supply their own LaTeX dependency chain.
        Models without a trace (e.g. Degachi) silently render nothing.
        """
        cls._render_results_trace()

    @classmethod
    def _render_results_trace(cls):
        """Optional: render an expander with the full extraction/sim formula chain.
        Default is a no-op; override to add a 📐 trace expander."""
        return

    @classmethod
    def has_formula_trace(cls) -> bool:
        """True iff the subclass overrides ``_render_results_trace``.

        Used by callers that want to drop the side-by-side layout when
        no formula trace is available (so the parameter table can use
        full width instead of leaving an empty right column).
        """
        return (cls._render_results_trace.__func__
                is not SSMModelTemplate._render_results_trace.__func__)

    @classmethod
    def render_override_and_smith(cls, fname, S_raw, freq, z0,
                                  para_eff, extract_result, *,
                                  show_tuning: bool = True,
                                  prefer_calc_vals: bool = False,
                                  show_cache_banner: bool = True, **kwargs):
        """
        Standard override-UI → cached sim → Smith chart → topology illustration
        → matplotlib Smith expander → tuning expander.

        ``show_tuning=False`` (used by the Extraction page) suppresses the
        Visual Tuning + Auto Tuning expanders — tuning now lives on the
        Simulation & Fitting page.

        ``prefer_calc_vals=True`` (set by the Simulation & Fitting page when a
        fresh extraction handoff arrives) makes calc_vals win over both stale
        widget state and any cached fit: the sim widget keys and sync hashes
        for this (model, fname) are cleared so ``_override_ui`` reseeds
        everything from calc_vals; the cache auto-restore is skipped (the
        banner + "📌 Use cache" button inside Fine-tune remain available).

        ``show_cache_banner=False`` suppresses the "📌 Cached fit for ..."
        banner entirely — used by RF_simulator.py's fit-mode call, which
        renders its own compact cache pill in the header row instead.  The
        "📌 Use cache" button inside the Fine-tune expander (see cache_ctx
        below) still works regardless of this flag.

        Concrete subclasses supply the per-model UI pieces via:
          cls._do_override_ui(fname, calc_vals, cache_ctx=None)  → all_p
              cache_ctx is {"has_cache": bool, "ts": str|None, "req_key": str}
              — forwarded to the module-level _override_ui so it can render a
              "📌 Use cache" button that sets session_state[req_key] = True
              and reruns; this function applies the cache on the next render
              before any sim widgets instantiate (see _cache_req_key below).
          cls._render_topology(all_p, fname)     → render illustration
          cls._INT_SPECS / _EXT_SPECS            → tuning specs

        Also performs persistent fit-cache restore (on first render after
        upload) and auto-save (when the user has fine-tuned vs. extraction
        defaults).  See helpers/fit_cache.py.
        """
        from ..helpers.fit_cache import (get_fit, get_fit_timestamp,
                                          save_fit, delete_fit, differs_from)

        params, _arrays = extract_result
        calc_vals = {**para_eff, **params}

        pad_specs   = cls._TUNING_PAD_SPECS if cls._TUNING_PAD_SPECS is not None else PAD_SPECS
        all_specs   = pad_specs + cls._EXT_SPECS + cls._INT_SPECS
        int_ext_keys = [k for k, *_ in cls._EXT_SPECS + cls._INT_SPECS]
        scale_for    = {k: sc for k, _, sc, *_ in all_specs}

        # ── Cache restore — runs on the first call after Run SSM is clicked
        #    (applied_key resides in session_state, which IOED's "Clear SSM
        #    results" button wipes for the file; so a fresh Run SSM cycle
        #    re-applies the cache).
        cached        = get_fit(fname, cls.SHORT)
        cached_ts     = get_fit_timestamp(fname, cls.SHORT)
        applied_key   = f"cache_applied_{cls.SHORT}_{fname}"
        dismissed_key = f"cache_dismissed_{cls.SHORT}_{fname}"

        # ── Fresh-handoff priority — the Sim & Fitting page sets this when an
        #    extraction handoff just arrived: the forwarded values must win
        #    over both stale widget state and the cached fit (previously the
        #    cache silently clobbered a fresh extraction in a new session).
        if prefer_calc_vals:
            for _k in list(st.session_state.keys()):
                if (_k.startswith(f"sim_{cls.SHORT}_")
                        and _k.endswith(f"_{fname}")):
                    del st.session_state[_k]
            st.session_state.pop(f"sim_synchash_{cls.SHORT}_{fname}", None)
            st.session_state.pop(f"smith_pad_synced_{cls.SHORT}_{fname}", None)
            st.session_state[applied_key] = True

        # Helper closure — write cached values (including pad) into the
        # fine-tune session_state and align the sync hashes so subsequent
        # renders don't overwrite us.
        #
        # Pad strategy: prefer cached pad if the cache has it (preserves
        # the user's fine-tuned pad, e.g. Rpb/Rpc/Rpe from Cold-HBT that
        # don't get re-extracted from Step 1b alone).  Fall back to
        # `para_eff` only when the cache has no value for that key
        # (older cache files written with the no-pad filter, or never
        # touched).  The matching `main_ssm_extraction` change ensures
        # `calc_vals[pad] = para_eff[pad]` (live Step 1), so the pad
        # sync hash below stays aligned with para_eff and
        # `sync_pad_from_preov` doesn't clobber the cached pad on the
        # next render.
        def _apply_cached_to_simstate(cached_dict):
            if not isinstance(cached_dict, dict):
                return
            # Pad: cached value first, else live para_eff.
            for key, _, sc, *_ in pad_specs:
                cv = cached_dict.get(key)
                if not isinstance(cv, (int, float)):
                    cv = para_eff.get(key, 0.0)
                st.session_state[f"sim_{cls.SHORT}_{key}_{fname}"] = float(cv) * sc
            # Int/ext: from cache.
            for k, v in cached_dict.items():
                if (k in scale_for and k not in _PAD_KEYS
                        and isinstance(v, (int, float))):
                    st.session_state[f"sim_{cls.SHORT}_{k}_{fname}"] = (
                        float(v) * scale_for[k])
            # Intrinsic sync hash: hash calc_vals[int_ext].  Since
            # main_ssm filters pad out of `params`, calc_vals[int_ext]
            # equals cached[int_ext] — so _override_ui's intrinsic sync
            # is a no-op.
            st.session_state[f"sim_synchash_{cls.SHORT}_{fname}"] = params_hash(
                {k: str(round(float(calc_vals.get(k, 0.0)), 15))
                 for k in int_ext_keys})
            # Pad sync hash matches live para_eff (NOT cached pad), so
            # sync_pad_from_preov stays a no-op until Step 1 actually
            # changes.  This is what protects the just-loaded cached pad
            # values from being overwritten by para_eff on the next
            # render.
            st.session_state[f"smith_pad_synced_{cls.SHORT}_{fname}"] = params_hash(
                {k: para_eff.get(k, 0.0) for k in _PAD_KEYS})

        # ── "Use cache" request — set by the "📌 Use cache" button inside the
        #    Fine-tune expander (module-level _override_ui in cheng/xu/kunyang).
        #    Runs before any sim widgets instantiate this run, so writing their
        #    session_state values here is legal (same pattern as the cache
        #    auto-restore below).
        _cache_req_key = f"cache_apply_request_{cls.SHORT}_{fname}"
        if st.session_state.pop(_cache_req_key, False) and cached:
            _apply_cached_to_simstate(cached)
            st.session_state[applied_key] = True

        # ── Cache restore — runs on the first call after Run SSM is clicked
        #    (applied_key resides in session_state, which IOED's "Clear SSM
        #    results" button wipes for the file; so a fresh Run SSM cycle
        #    re-applies the cache).
        #
        #    `applied_key` is set on this first render *whether or not* a cache
        #    existed.  Critically, when NO cache exists yet, the user's first
        #    fine-tune edit triggers the auto-save below, which creates a cache
        #    file.  If `applied_key` were still unset on the next render, this
        #    branch would see the freshly-written cache and re-apply it —
        #    clobbering the user's in-progress edit with the previous value
        #    (the "have to type every value twice" bug).  Setting the flag now
        #    guarantees the override widgets own session_state from here on.
        # Widget keys can be GC'd by a page switch (see the keep-alive in
        # IOED_Tool_Web.py); if that happened, re-apply the cache — thanks to
        # auto-save the cache IS the user's last-seen state — instead of
        # letting every input recreate at 0.
        _widgets_missing = any(
            f"sim_{cls.SHORT}_{k}_{fname}" not in st.session_state
            for k in int_ext_keys)
        if (not prefer_calc_vals
                and (not st.session_state.get(applied_key) or _widgets_missing)
                and not st.session_state.get(dismissed_key)):
            if cached:
                _apply_cached_to_simstate(cached)
            st.session_state[applied_key] = True

        # The container renders unconditionally so the element tree above the
        # fine-tune expander stays stable whether or not the banner shows —
        # conditional siblings above an expander reset its client-side open
        # state (the "expander always closes" bug).
        _banner_slot = st.container()
        if show_cache_banner and cached_ts and not st.session_state.get(dismissed_key):
            with _banner_slot:
                st.markdown(
                    f"<small>📌 Cached fit for <b>{cls.NAME}</b> — saved {cached_ts}"
                    " <span class='hbt-help' title='A fit saved earlier for this"
                    " file and model. It is applied automatically on entry -"
                    " except right after a handoff from Extraction, where the"
                    " freshly extracted values take priority. Load it into the"
                    " fields below with “📌 Use cache” inside the Fine-tune"
                    " expander.'>?</span></small>",
                    unsafe_allow_html=True)

        all_p = cls._do_override_ui(fname, calc_vals, cache_ctx={
            "has_cache": bool(cached), "ts": cached_ts, "req_key": _cache_req_key})

        S_sim = cls._cached_simulate_vec(all_p, freq, z0, fname)

        # Build the modeled S2P bytes once here so render_smith_with_ftfmax
        # can wire a "📥 modeled S2P" button next to the xlsx download under
        # the Smith chart.  This replaces the standalone "Download Modeled
        # DUT S2P" section that used to live at the bottom of the SSM tab.
        from ..helpers import write_s2p
        from pathlib import Path as _Path
        s2p_params = {}
        for _k in ("Cpbe", "Cpce", "Cpbc"):
            s2p_params[_k] = f"{para_eff.get(_k, 0.0) * 1e15:.4f} fF"
        for _k_raw, _label in (("Rpb", "Rb"), ("Rpc", "Rc"), ("Rpe", "Re")):
            s2p_params[_label] = f"{para_eff.get(_k_raw, 0.0):.4f} Ω"
        for _k in ("Lb", "Lc", "Le"):
            s2p_params[_k] = f"{para_eff.get(_k, 0.0) * 1e12:.4f} pH"
        s2p_bytes = write_s2p(
            freq, S_sim,
            title=f"DUT {cls.NAME} — {_Path(fname).stem}",
            params=s2p_params,
        )
        s2p_filename = f"model_dut_{_Path(fname).stem}_{cls.SHORT}.s2p"

        sc = smith_scale_controls(fname, cls.SHORT)
        render_smith_with_ftfmax(S_raw, S_sim, freq,
                                 model_name=cls.NAME, model_short=cls.SHORT,
                                 fname=fname, scales=sc,
                                 s2p_bytes=s2p_bytes,
                                 s2p_filename=s2p_filename)

        # τ_total + calculated fmax expander (HBT T/π models only — Kun-Yang
        # HEMT has no Cbcx/Cbc/Rbi/Rb to evaluate the fmax formula).
        if cls.SHORT in ("T", "pi", "XuT"):
            from ..ssm_plots import render_tau_fmax_expander
            _CBC = float(all_p.get("Cbcx", 0.0)) + float(all_p.get("Cbc", 0.0))
            _Rbb = float(all_p.get("Rbi", 0.0)) + float(all_p.get("Rpb", 0.0))
            if cls.SHORT == "pi":
                _tau_sum = float(all_p.get("tau", 0.0))
                _tau_lbl, _tau_tex = "τ", r"\tau"
            else:
                _tau_sum = float(all_p.get("tauB", 0.0)) + float(all_p.get("tauC", 0.0))
                _tau_lbl, _tau_tex = "τB + τC", r"\tau_B+\tau_C"
            render_tau_fmax_expander(key=f"taufmax_{cls.SHORT}_{fname}",
                                     freq=freq, S_meas=S_raw, S_model=S_sim,
                                     CBC=_CBC, Rbb=_Rbb,
                                     tau_sum=_tau_sum, tau_sum_label=_tau_lbl,
                                     tau_sum_tex=_tau_tex,
                                     extrap_key=f"ftfmax_card_{cls.SHORT}_{fname}")

        # Persist the *current* (post-override) param dict so the Complete
        # Parameter Summary can read live values instead of extraction-time ones.
        st.session_state[f"current_p_{cls.SHORT}_{fname}"] = dict(all_p)

        # ── Auto-save — fires whenever the fine-tune section's `all_p`
        # differs from the current "baseline" snapshot.  The baseline is
        # `calc_vals` (extraction defaults) overlaid with any cached
        # values — so after a cache restore, `all_p == baseline` and no
        # redundant save fires; only genuine user edits beyond the
        # cached state trigger a write.  Pad is included so users can
        # fine-tune Rpb/Rpc/Rpe (etc.) and have those values persist.
        baseline = dict(calc_vals)
        if isinstance(cached, dict):
            for _k, _v in cached.items():
                if _k in scale_for and isinstance(_v, (int, float)):
                    baseline[_k] = float(_v)
        check_keys = _PAD_KEYS + int_ext_keys
        # Refuse to persist a degenerate/uninitialised state: a genuine fit
        # has (nearly) all intrinsic/extrinsic values nonzero, while a
        # GC-wiped widget set is all zeros except whatever the user just
        # touched — exactly the state that poisoned the cache before.
        _n_nonzero = sum(
            1 for k in int_ext_keys
            if isinstance(all_p.get(k), (int, float)) and float(all_p[k]) != 0.0)
        if (_n_nonzero >= max(2, len(int_ext_keys) // 2)
                and not st.session_state.get(dismissed_key)
                and differs_from(all_p, baseline, keys=check_keys)):
            save_fit(fname, cls.SHORT, dict(all_p))

        # Topology illustration in its own expander (collapsed by default).
        # For no-parasitics models we also pass the user's *customized* Smith
        # chart (rebuilt from session_state via phase="chart", return_png=True —
        # creates no widgets) so the topology view can overlay it bottom-right.
        from ..ssm_plots import render_matplotlib_smith
        _smith_png = None
        if S_sim is not None:
            try:
                _smith_png = render_matplotlib_smith(
                    S_raw, S_sim, fname, cls.SHORT,
                    default_multiplier=sc,
                    phase="chart", freq_hz=freq, return_png=True)
            except Exception:                            # noqa: BLE001
                _smith_png = None
        with st.container(key=f"hbt_exp_view_topo_{cls.SHORT}"), \
             st.expander(tr("🖼️ Topology illustration", "🖼️ 拓樸示意圖"),
                         expanded=False):
            # Models registered with a built-in custom-model preset (Cheng
            # T/π, Xu T, Kun-Yang HEMT — see svg_topology._PRESET_LABEL) can
            # switch to a live SVG schematic that omits every zero-valued
            # component.  Degachi (no preset) has no ``_SVG_TOPOLOGY`` attr,
            # so it keeps the plain PNG-template path unchanged.
            svg_mode = False
            if getattr(cls, "_SVG_TOPOLOGY", False):
                svg_mode = st.toggle(
                    tr("Simplified schematic (non-zero components only)",
                       "簡化示意圖（僅顯示非零元件）"),
                    key=f"topo_svg_{cls.SHORT}_{fname}",
                    help=tr("ON: a live SVG schematic built from the model "
                            "topology, showing only components with a "
                            "non-zero current value. OFF: the standard "
                            "labeled template illustration.",
                            "開啟：由模型拓樸即時產生的 SVG 示意圖，"
                            "僅顯示目前為非零值的元件。"
                            "關閉：標準的標籤範本示意圖。"))
            if svg_mode:
                from .svg_topology import render_svg_topology
                render_svg_topology(cls.SHORT, all_p, fname)
            else:
                cls._render_topology(all_p, fname, smith_png=_smith_png)

        # NOTE: the Open/Short pad-dummy schematics (svg_topology.
        # render_pad_topology) deliberately do NOT appear here.  A device
        # model's own topology illustration above already draws its pad
        # parasitics in place, so a second pad-only drawing was redundant on
        # every model page (and on the custom model).  The pad schematics now
        # live only where they are the subject: the RF simulator's "Open and
        # Short Pad" model (tools/RF_simulator.py).

        # Smith chart (matplotlib) + its controls live in a single
        # expander, rendered side-by-side — matches the RF simulator
        # layout (RF_simulator.py "🍩 Smith Chart (Matplotlib)").  The
        # split-call pattern (controls in the right column, chart in the
        # left) preserves the order-of-operations requirement that
        # widgets render BEFORE the chart so session_state is fresh when
        # the chart half reads it.
        with st.container(key=f"hbt_exp_view_mplsmith_{cls.SHORT}"), \
             st.expander(tr("🍩 Smith Chart (Matplotlib)",
                            "🍩 Smith 圖 (Matplotlib)"), expanded=False):
            col_mpl_left, col_mpl_right = st.columns([1.2, 1])
            with col_mpl_right:
                render_matplotlib_smith(S_raw, S_sim, fname, cls.SHORT,
                                         default_multiplier=sc,
                                         phase="controls", freq_hz=freq)
            with col_mpl_left:
                render_matplotlib_smith(S_raw, S_sim, fname, cls.SHORT,
                                         default_multiplier=sc,
                                         phase="chart", freq_hz=freq)

        if show_tuning:
            render_visual_tuning_expander(cls, all_p, S_raw, freq, z0,
                                           all_specs,
                                           fname, cls.SHORT)
            render_tuning_expander(
                cls, all_p, S_raw, freq, z0, all_specs, fname, cls.SHORT,
                default_fit_keys=(
                    [k for k, *_ in cls._EXT_SPECS + cls._INT_SPECS]
                    + ["Rpb", "Rpc", "Rpe"]))
        return S_sim
