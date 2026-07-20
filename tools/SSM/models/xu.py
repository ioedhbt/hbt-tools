"""
models/xu.py — Xu's T (2014) T-topology HBT small-signal model.

Like Cheng's T-topology but: (i) no extrinsic Cbex, and (ii) a parallel
Rbcx alongside Cbcx so the base-collector extrinsic admittance is
Ybcx = 1/Rbcx + jωCbcx.  Rbcx is not extracted from measurement — it
defaults to 285 kΩ and is user-tunable via the Fine-tune UI.
"""
from __future__ import annotations
import os as _os
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path as _Path
from ..helpers         import (y_to_z, z_to_y, y_to_s_single, y_to_s_vec,
                                inv2x2, mm2x2,
                                safe_median, params_hash,
                                extended_smith_grid,
                                build_Y_pad, build_Z_ser,
                                build_Y_pad_vec, build_Z_ser_vec,
                                build_Y_pad_batch, build_Z_ser_batch,
                                segmented_radio)
from .base_ui         import (sync_pad_from_preov, PAD_SPECS, SSMModelTemplate,
                              render_finetune_diagram)
from ._shared          import (_b1, _detect_B, _stack22,
                                _try_download_inter, has_inter, _load_font,
                                _FONT_CACHE_DIR)
from . import AbstractSSMModel
from ...i18n import tr

# Default Rbcx (Ω): user-tunable; not extracted from data.
_RBCX_DEFAULT = 285e3

# _b1, _detect_B, _stack22 — see ._shared (lifted to share with cheng.py).


def _step2_T(Y_ex1, freq):
    """
    Xu — Extract Cbcx directly from Y_ex1 (no Cbex peel).  Rbcx is not
    extracted from data; it defaults to 285 kΩ (user-tunable).

    Cbcx = −[Im(Yms)·Re(YL) − Re(Yms)·Im(YL)] / [ω · denominator]
    where  Yms = Y12+Y22,  YL = det(Y_ex1),  Ytot = sum(Yij)
    """
    omega = 2.0*np.pi*freq

    Yms   = Y_ex1[:,0,1] + Y_ex1[:,1,1]
    YL    = Y_ex1[:,0,0]*Y_ex1[:,1,1] - Y_ex1[:,0,1]*Y_ex1[:,1,0]
    Ytot  = Y_ex1[:,0,0] + Y_ex1[:,0,1] + Y_ex1[:,1,0] + Y_ex1[:,1,1]
    num   = np.imag(Yms)*np.real(YL) - np.real(Yms)*np.imag(YL)
    den   = np.real(Yms)*np.real(Ytot) + np.imag(Ytot)*np.imag(Yms)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cbcx_arr = -np.where(np.abs(den) > 1e-40, num/(omega*den), np.nan)
    n0, n1 = len(freq)//4, 3*len(freq)//4
    Cbcx = abs(safe_median(Cbcx_arr[n0:n1]))
    Rbcx = _RBCX_DEFAULT
    return ({"Cbcx": Cbcx, "Rbcx": Rbcx},
            {"Cbcx_arr": Cbcx_arr})

def _step3_T(Y_ex1, freq, Cbcx, Rbcx, n_low):
    """
    Xu — Extract intrinsic parameters for T-topology.

    NOTE: the intrinsic extraction below (Z_in → Rbi, Rbe, Cbe, Rbc, Cbc,
    α, τB, τC) is taken directly from Cheng's T-topology Step 3.  Xu's
    contribution is the *outer* base-collector network — replacing
    Cheng's series Cbex + parallel Cbcx with a single parallel
    Rbcx ∥ Cbcx — so once that network is peeled the inside math is
    identical to Cheng.  See ``models/cheng.py::_step3_T`` for the
    original derivation (Cheng 2022, Eqs. 16, 29–31).

    Z_in = [Y_ex1 − Ybcx·shunt]⁻¹   with Ybcx = 1/Rbcx + jωCbcx
    Zbe = Z12,   Zbc = Z22−Z21,   Zbi = Z11−Z12
    α = (Z12−Z21) / (Z22−Z21),   α₀ = |α|ω→0
    τB = √(U−1)/ω   where U = (α₀/|α|)²
    τC = −arctan(V/√(1−V²)) / (2ω)   where V=2ωτB/U
    """
    omega = 2.0*np.pi*freq

    # Peel Ybcx (= 1/Rbcx + jωCbcx, parallel) to get intrinsic Y
    Y_in = Y_ex1.copy()
    for i, w in enumerate(omega):
        Ybcx = 1.0/Rbcx + 1j*w*Cbcx
        Y_in[i,0,0] -= Ybcx; Y_in[i,0,1] += Ybcx
        Y_in[i,1,0] += Ybcx; Y_in[i,1,1] -= Ybcx

    Z_in = y_to_z(Y_in)  # [Eq. 16] ↓
    Zbe_arr = Z_in[:,0,1]
    Zbc_arr = Z_in[:,1,1] - Z_in[:,1,0]
    Zbi_arr = Z_in[:,0,0] - Z_in[:,0,1]

    with np.errstate(divide="ignore", invalid="ignore"):
        Ybe_arr = 1.0 / (Zbe_arr + 1e-40)
        Ybc_arr = 1.0 / (Zbc_arr + 1e-40)
        alpha_arr = (Z_in[:,0,1] - Z_in[:,1,0]) / (Zbc_arr + 1e-40)  # [Eq. 29]

    Rbe_a = 1.0 / np.real(Ybe_arr).clip(1e-6)
    Cbe_a = np.imag(Ybe_arr) / omega
    Rbc_a = 1.0 / np.real(Ybc_arr).clip(1e-12)
    Cbc_a = np.imag(Ybc_arr) / omega
    Rbi_a = np.real(Zbi_arr)

    Rbe  = safe_median(Rbe_a, n_low)
    Cbe  = safe_median(Cbe_a, n_low)
    Rbc  = safe_median(Rbc_a, n_low)
    Cbc  = safe_median(Cbc_a, n_low)
    Rbi  = safe_median(Rbi_a, n_low)
    alpha0 = safe_median(np.abs(alpha_arr), n_low)   # [Eq. 29]

    # [Eq. 30] τB
    U_arr    = (alpha0 / (np.abs(alpha_arr) + 1e-30))**2
    tauB_arr = np.sqrt(np.maximum(U_arr - 1.0, 0.0)) / omega
    tauB     = safe_median(tauB_arr[n_low:])

    # [Eq. 31 corrected] τC from phase of α
    # arg(α) = −ω·τC − arctan(ω·τB)  →  τC = [−arg(α) − arctan(ω·τB)] / ω
    with np.errstate(divide="ignore", invalid="ignore"):
        tauC_arr = (-np.angle(alpha_arr) - np.arctan(omega * tauB_arr)) / (omega + 1e-40)
    tauC = safe_median(tauC_arr[n_low:])

    params = dict(Rbi=Rbi, Rbe=Rbe, Cbe=Cbe, Rbc=Rbc, Cbc=Cbc,
                  alpha0=alpha0, tauB=tauB, tauC=tauC)
    arrays = dict(Rbi=Rbi_a, Rbe=Rbe_a, Cbe=Cbe_a, Rbc=Rbc_a, Cbc=Cbc_a,
                  alpha=alpha_arr, tauB=tauB_arr, tauC=tauC_arr)
    return params, arrays

# ── Vectorised forward simulation (no per-freq loop) ─────────────────────────
def _sim_wrap(Y_int_fn, p, freq, z0):
    """Add extrinsic Ybcx + pad/lead parasitics around the intrinsic Y matrix."""
    omega = 2.0*np.pi*freq
    S = np.zeros((len(freq), 2, 2), dtype=complex)
    Rbcx = p.get("Rbcx", _RBCX_DEFAULT)
    for i, w in enumerate(omega):
        Y_in  = Y_int_fn(p, w)
        Ybcx  = 1.0/Rbcx + 1j*w*p["Cbcx"]
        Y_ex  = Y_in + Ybcx*np.array([[1,-1],[-1,1]])
        Z_ser = build_Z_ser(p, w)
        try:    Y_tot = np.linalg.inv(np.linalg.inv(Y_ex) + Z_ser)
        except: Y_tot = np.zeros((2,2), dtype=complex)
        Y_pad = build_Y_pad(p, w)
        S[i]  = y_to_s_single(Y_tot + Y_pad, z0)
    return S

def _sim_wrap_vec(Y_int_vec_fn, p, freq, z0, xp):
    """Vectorised version of _sim_wrap — processes all freq points at once.

    Works identically with numpy (CPU) and cupy (GPU).
    *Y_int_vec_fn(p, omega, xp)* must return an (N, 2, 2) intrinsic-Y array.
    """
    omega = xp.asarray(2.0 * np.pi * freq, dtype=np.float64)

    # Intrinsic admittance (N, 2, 2)
    Y_in = Y_int_vec_fn(p, omega, xp)

    # Extrinsic Ybcx (parallel Rbcx + Cbcx)
    Rbcx = p.get("Rbcx", _RBCX_DEFAULT)
    Ybcx = 1.0/Rbcx + 1j * omega * p["Cbcx"]   # (N,)

    Y_ex = Y_in.copy()
    Y_ex[:, 0, 0] += Ybcx
    Y_ex[:, 0, 1] -= Ybcx
    Y_ex[:, 1, 0] -= Ybcx
    Y_ex[:, 1, 1] += Ybcx

    # Series-lead impedance (N, 2, 2)
    Z_ser = build_Z_ser_vec(p, omega, xp)

    # Y_tot = inv(inv(Y_ex) + Z_ser)   — batched 2×2 inversions
    Y_tot = xp.linalg.inv(xp.linalg.inv(Y_ex) + Z_ser)

    # Pad admittance (N, 2, 2)
    Y_pad = build_Y_pad_vec(p, omega, xp)

    # Y → S (batched)
    S = y_to_s_vec(Y_tot + Y_pad, z0, xp)

    # If CuPy, bring result back to host
    if xp is not np:
        S = xp.asnumpy(S)
    return S


def _sim_wrap_batch(Y_int_batch_fn, p, freq, z0, xp, cache=None):
    """Batched forward simulation over (param_combo × frequency).

    *p* is a dict whose values are scalars or (B,) arrays — both may be mixed.
    Returns S of shape (B, N_freq, 2, 2), still on the *xp* device.

    Hand-inlined 2×2 algebra throughout: every matrix inverse is the
    analytic adjugate formula, every matmul is 8 scalar mults.  This
    skips cuSOLVER/cuBLAS entirely, which have launch overhead orders of
    magnitude larger than the actual 2×2 arithmetic.

    Optional ``cache`` (built once before the chunk loop in
    ``render_tuning_expander``) lets us skip recomputing constant
    sub-networks every chunk.  Recognised keys:

      - ``"omega"``  : pre-built (1, N) angular-frequency array
      - ``"Y_pad"``  : 4 planes (yp00, yp01, yp10, yp11)
      - ``"Z_ser"``  : 4 planes (zs00, zs01, zs10, zs11)
      - ``"_cdtype"``: complex dtype for the entire compute path.  Use
        ``np.complex64`` for the fp32 sweep main loop, ``np.complex128``
        for the final fp64 rerank.  Defaults to ``np.complex128`` so
        callers without a cache get the original behaviour.
    """
    N = len(freq)
    B = _detect_B(p, xp)
    cdtype = (cache or {}).get("_cdtype", np.complex128)
    rdtype = np.float32 if cdtype == np.complex64 else np.float64
    J = xp.asarray(1j, dtype=cdtype)
    if cache is not None and "omega" in cache:
        omega = cache["omega"]
    else:
        omega = xp.asarray(2.0 * np.pi * freq, dtype=rdtype).reshape(1, N)  # (1, N)

    # Intrinsic Y matrix as 4 (B, N) planes — no (B, N, 2, 2) tensor yet
    yi00, yi01, yi10, yi11 = Y_int_batch_fn(p, omega, B, N, xp, cache)

    # Extrinsic Ybcx = 1/Rbcx + jωCbcx  (broadcastable to (B, N)).
    # No Y_extr cache for Xu: Rbcx is not in the Cheng-style cache schema,
    # so we always recompute fresh.
    Cbcx = _b1(p, "Cbcx", 0.0,           xp, rdtype)
    Rbcx = _b1(p, "Rbcx", _RBCX_DEFAULT, xp, rdtype)
    Ybcx = (1.0 / Rbcx) + (J * omega) * Cbcx

    # Y_ex = Y_in + Ybcx · [[1,-1],[-1,1]]
    ye00 = yi00 + Ybcx
    ye01 = yi01 - Ybcx
    ye10 = yi10 - Ybcx
    ye11 = yi11 + Ybcx

    # Z_ex = inv(Y_ex)  — analytic 2×2
    inv_det_e = 1.0 / (ye00 * ye11 - ye01 * ye10)
    ze00 =  ye11 * inv_det_e
    ze01 = -ye01 * inv_det_e
    ze10 = -ye10 * inv_det_e
    ze11 =  ye00 * inv_det_e

    # Z_ser as 4 planes (cache-aware) — no (B,N,2,2) build
    if cache is not None and "Z_ser" in cache:
        zs00, zs01, zs10, zs11 = cache["Z_ser"]
    else:
        zs00, zs01, zs10, zs11 = build_Z_ser_batch(p, omega, B, N, xp)

    zt00 = ze00 + zs00
    zt01 = ze01 + zs01
    zt10 = ze10 + zs10
    zt11 = ze11 + zs11

    # Y_tot = inv(Z_tot) — analytic 2×2
    inv_det_t = 1.0 / (zt00 * zt11 - zt01 * zt10)
    yt00 =  zt11 * inv_det_t
    yt01 = -zt01 * inv_det_t
    yt10 = -zt10 * inv_det_t
    yt11 =  zt00 * inv_det_t

    # Y_pad as 4 planes (cache-aware)
    if cache is not None and "Y_pad" in cache:
        yp00, yp01, yp10, yp11 = cache["Y_pad"]
    else:
        yp00, yp01, yp10, yp11 = build_Y_pad_batch(p, omega, B, N, xp)

    # Y_total network = Y_tot + Y_pad
    ya00 = yt00 + yp00
    ya01 = yt01 + yp01
    ya10 = yt10 + yp10
    ya11 = yt11 + yp11

    # ── Y → S, fully inlined.  M = I + Yn ; S = (I − Yn) · M⁻¹ ──
    yn00 = ya00 * z0
    yn01 = ya01 * z0
    yn10 = ya10 * z0
    yn11 = ya11 * z0

    m00 = 1.0 + yn00
    m11 = 1.0 + yn11
    inv_det_m = 1.0 / (m00 * m11 - yn01 * yn10)
    mi00 =  m11 * inv_det_m
    mi01 = -yn01 * inv_det_m
    mi10 = -yn10 * inv_det_m
    mi11 =  m00 * inv_det_m

    n00 = 1.0 - yn00
    n11 = 1.0 - yn11
    s00 = n00 * mi00 + (-yn01) * mi10
    s01 = n00 * mi01 + (-yn01) * mi11
    s10 = (-yn10) * mi00 + n11 * mi10
    s11 = (-yn10) * mi01 + n11 * mi11

    return _stack22(s00, s01, s10, s11, xp)   # (B, N, 2, 2)


def _Y_int_T_vec(p, omega, xp):
    """Vectorised T-topology intrinsic Y matrix → (N, 2, 2)."""
    Zbe = p["Rbe"] / (1.0 + 1j * omega * p["Rbe"] * p["Cbe"])
    Zbc = p["Rbc"] / (1.0 + 1j * omega * p["Rbc"] * p["Cbc"])
    alpha = (p["alpha0"] * xp.exp(-1j * omega * p["tauC"])
             / (1.0 + 1j * omega * p["tauB"]))
    N = len(omega)
    Z_in = xp.zeros((N, 2, 2), dtype=complex)
    Z_in[:, 0, 0] = p["Rbi"] + Zbe
    Z_in[:, 0, 1] = Zbe
    Z_in[:, 1, 0] = Zbe - alpha * Zbc
    Z_in[:, 1, 1] = (1.0 - alpha) * Zbc + Zbe
    return xp.linalg.inv(Z_in)


def _Y_int_T_batch(p, omega, B, N, xp, cache=None):
    """Batched T-topology intrinsic Y → 4 (B, N) planes (y00, y01, y10, y11).

    Returning planes (not a (B, N, 2, 2) tensor) lets ``_sim_wrap_batch``
    keep the algebra fully inlined.

    Cache-aware: skip whichever sub-expressions are constant for the
    sweep (only their inputs aren't being swept):

      - ``"T_int_planes"`` : full (yi00..yi11) tuple — used when *every*
        intrinsic param (Rbi, Rbe, Cbe, Rbc, Cbc, alpha0, tauB, tauC) is
        constant, so we can return the four planes directly.
      - ``"Zbe"``          : pre-built when both Rbe and Cbe are constant.
      - ``"Zbc"``          : pre-built when both Rbc and Cbc are constant.
      - ``"alpha"``        : pre-built when alpha0, tauB, tauC are all constant.

    The dtype is propagated through ``cache["_cdtype"]`` (complex64 for
    fp32 sweep, complex128 otherwise).
    """
    if cache is not None and "T_int_planes" in cache:
        return cache["T_int_planes"]

    cdtype = (cache or {}).get("_cdtype", np.complex128)
    rdtype = np.float32 if cdtype == np.complex64 else np.float64
    J = xp.asarray(1j, dtype=cdtype)

    Rbi = _b1(p, "Rbi", 0.0, xp, rdtype)

    if cache is not None and "Zbe" in cache:
        Zbe = cache["Zbe"]
    else:
        Rbe = _b1(p, "Rbe", 1.0, xp, rdtype)
        Cbe = _b1(p, "Cbe", 0.0, xp, rdtype)
        Zbe = Rbe / (1.0 + J * omega * Rbe * Cbe)

    if cache is not None and "Zbc" in cache:
        Zbc = cache["Zbc"]
    else:
        Rbc = _b1(p, "Rbc", 1.0, xp, rdtype)
        Cbc = _b1(p, "Cbc", 0.0, xp, rdtype)
        Zbc = Rbc / (1.0 + J * omega * Rbc * Cbc)

    if cache is not None and "alpha" in cache:
        alpha = cache["alpha"]
    else:
        alpha0 = _b1(p, "alpha0", 0.0, xp, rdtype)
        tauC   = _b1(p, "tauC",   0.0, xp, rdtype)
        tauB   = _b1(p, "tauB",   0.0, xp, rdtype)
        alpha = alpha0 * xp.exp(-J * omega * tauC) / (1.0 + J * omega * tauB)

    # Z_in 2×2
    z00 = Rbi + Zbe
    z01 = Zbe
    z10 = Zbe - alpha * Zbc
    z11 = (1.0 - alpha) * Zbc + Zbe

    # Y_in = inv(Z_in)
    inv_det = 1.0 / (z00 * z11 - z01 * z10)
    y00 =  z11 * inv_det
    y01 = -z01 * inv_det
    y10 = -z10 * inv_det
    y11 =  z00 * inv_det
    return y00, y01, y10, y11

# ── Override UI specs (used by render_override_and_smith) ─────────────────────

_EXT_SPECS = [
    ("Cbcx","Cbcx",1e15,"fF", "%.4f", 0.1),
    ("Rbcx","Rbcx",1e-3,"kΩ", "%.4f", 1.0),
]
_EXT_T_SPECS  = _EXT_SPECS

# Dual-name label helper: show both the Cheng-T label and the Xu alias so
# users coming from either paper recognise the parameter.
def _aka(cheng_lbl: str, xu_lbl: str) -> str:
    return f"{cheng_lbl} (from Cheng's T) aka {xu_lbl} (from Xu)"

_INT_T_SPECS = [
    ("Rbi",   "Rbi",                    1.0, "Ω",  "%.4f", 0.1),
    ("Rbe",   _aka("Rbe", "rE"),        1.0, "Ω",  "%.3f", 1.0),
    ("Cbe",   "Cbe",                    1e15,"fF", "%.4f", 0.1),
    ("Rbc",   _aka("Rbc", "Rbci"),      1e-3,"kΩ", "%.4f", 0.01),
    ("Cbc",   _aka("Cbc", "Cbci"),      1e15,"fF", "%.4f", 0.01),
    ("alpha0","α₀",                     1.0, "",   "%.5f", 0.001),
    ("tauB",  "τB",                     1e12,"ps", "%.4f", 0.01),
    ("tauC",  "τC",                     1e12,"ps", "%.4f", 0.01),
]

# Xu-flavoured PAD spec overrides: key stays identical (so session-state /
# de-embedding plumbing still finds the value), only the display label
# differs from base_ui.PAD_SPECS.  Format: "Cheng-label aka Xu-label".
_XU_PAD_LABEL_OVERRIDES = {
    "Rpb":  _aka("Rb",   "Rbx"),
    "Rpe":  _aka("Re",   "Rex"),
    "Cpce": _aka("Cpce", "Cpad"),
}
_XU_PAD_SPECS = [
    (key, _XU_PAD_LABEL_OVERRIDES.get(key, lbl), sc, unit, fmt, step)
    for (key, lbl, sc, unit, fmt, step) in PAD_SPECS
]


# ════════════════════════════════════════════════════════════════════════════════
# Topology illustration helpers
# ════════════════════════════════════════════════════════════════════════════════

_ILLUS_DIR = _Path(__file__).parent / "illus_template"
# _FONT_CACHE_DIR / _try_download_inter / has_inter / _load_font — see ._shared.

ohm_sign = "Ω" if has_inter() else "Ohm"

# Display units for each parameter key: (SI→display scale factor, base unit string)
_PARAM_DISPLAY: dict[str, tuple] = {
    "Cpbe":   (1e15, "fF"),  "Cpce":  (1e15, "fF"),  "Cpbc":  (1e15, "fF"),
    "Lb":     (1e12, "pH"),  "Lc":    (1e12, "pH"),   "Le":    (1e12, "pH"),
    "Rpb":    (1,    ohm_sign),   "Rpc":   (1,    ohm_sign),    "Rpe":   (1,    ohm_sign),
    "Cbcx":   (1e15, "fF"),
    "Rbcx":   (1e-3, f"k{ohm_sign}"),
    "Rbi":    (1,    ohm_sign),   "Rbe":   (1,    ohm_sign),
    "Cbe":    (1e15, "fF"),  "Cbc":   (1e15, "fF"),
    "Rbc":    (1e-3, f"k{ohm_sign}"),
    "alpha0": (1,    ""),
    "tauB":   (1e12, "ps"),  "tauC":  (1e12, "ps"),
}

# Unit ladder: when display value >= 1000, scale to the next prefix
_UNIT_LADDER: dict[str, str] = {
    "fF": "pF",  "pF": "nF",
    "pH": "nH",  "nH": "μH",
    "ps": "ns",  "ns": "μs",
    "mS": "S",
    f"{ohm_sign}":  f"k{ohm_sign}",  f"k{ohm_sign}": f"M{ohm_sign}", 
    f"M{ohm_sign}": f"G{ohm_sign}", f"G{ohm_sign}": f"T{ohm_sign}",
}

# Keys that get 3 decimal places instead of 2
_3DP_PARAMS = {"alpha0"}

# Pixel (x, y[, anchor]) positions for overlaid value text — image is 1014 × 831 px.
# anchor is a PIL anchor string (default "mm"). First char: l/m/r = horizontal align
# (left/center/right). Second char: t/m/b = vertical align (top/middle/bottom).
# Positions are shared for pad/ext params; intrinsic differs per topology.
_COMMON_OVERLAY: dict[str, tuple] = {
    # Pad parasitics
    "Cpbc": (485,  58),
    "Cpbe": (100, 610),
    "Cpce": (915, 610),

    # Lead inductances
    "Lb":  (118, 340),
    "Le":  (510, 745),
    "Lc":  (890, 340),

    # Access or series resistance
    "Rpb": (247, 340),
    "Rpe": (475, 665, "lm"),
    "Rpc": (762, 340),

    # External base-collector network (parallel Rbcx ∥ Cbcx)
    "Cbcx": (512, 160),
    "Rbcx": (360, 200, "lm"),

    # Intrinsic base resistance
    "Rbi": (390, 340),
}

_T_EXTRA_OVERLAY: dict[str, tuple] = {
    "Rbe":    (408, 515, "lm"),
    "Cbe":    (545, 557, "lm"),
    "Rbc":    (592, 340),
    "Cbc":    (590, 250),
    "alpha0": (582, 433, "lm"),
    "tauB":   (582, 458, "lm"),
    "tauC":   (582, 485, "lm"),
}

_T_OVERLAY  = {**_COMMON_OVERLAY, **_T_EXTRA_OVERLAY}


def _fmt_param(key: str, val_si: float) -> str:
    """Format a parameter SI value for display on the topology illustration."""
    if not np.isfinite(val_si):
        return "—"
    if key not in _PARAM_DISPLAY:
        return f"{val_si:.3g}"
    scale, unit = _PARAM_DISPLAY[key]
    decimals = 3 if key in _3DP_PARAMS else 2
    display = val_si * scale
    while abs(display) >= 1000 and unit in _UNIT_LADDER:
        display /= 1000
        unit = _UNIT_LADDER[unit]
    if abs(display - round(display)) < 0.005:
        text = f"{int(round(display))}"
    else:
        text = f"{display:.{decimals}f}"
    return f"{text} {unit}" if unit else text


def _render_topology_illustration(all_p: dict, fname: str,
                                  smith_png=None, highlight_key=None) -> None:
    """
    Overlay live parameter values on the Xu T-topology schematic PNG and display
    it via st.image().  Called inside a Streamlit expander by render_override_and_smith.

    Parameters
    ----------
    all_p     : dict — current (post Fine-tune) parameters in SI units.
    fname     : file name tag used only as an image key for Streamlit.
    smith_png : optional PNG bytes of the user's customized matplotlib Smith
                chart.  When given AND the model has no parasitics, a second
                column shows the schematic with that Smith chart composited at
                the bottom-right (placement tunable via SMITH_OVERLAY_* in
                models/_shared.py).
    """
    import io
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        st.info(tr("Install *pillow* to see the topology illustration.",
                   "請安裝 *pillow* 套件以顯示拓樸示意圖。"))
        return

    _PARASITIC_KEYS = ("Cpce", "Cpbe", "Cpbc", "Lb", "Le", "Lc")
    def _is_zero(key):
        v = all_p.get(key)
        try:
            return v is None or float(v) == 0.0
        except Exception:
            return False
    no_parasitics = all(_is_zero(k) for k in _PARASITIC_KEYS)

    tpl_name = ("XuT_template_noparasitics.png" if no_parasitics
                else "XuT_template.png")
    tpl_path = _ILLUS_DIR / tpl_name
    if not tpl_path.exists():
        st.warning(tr(f"Template not found: {tpl_path}",
                      f"找不到範本檔案：{tpl_path}"))
        return

    overlay = _T_OVERLAY

    # Category colour sets (dark, readable on white)
    _C_PAD = (180,  2,   2)   # red
    _R_ACC = (175,  90,   5)   # orange
    _C_EXT = (  0, 130,  55)   # green
    _C_INT = ( 20,  95, 160)   # blue

    _PAD_KEYS = {"Cpbe", "Cpce", "Cpbc", "Lb", "Lc", "Le"}
    _ACCESSRES_KEYS = {"Rpb", "Rpc", "Rpe"}
    _EXT_KEYS = {"Cbcx", "Rbcx"}

    def _color(key: str) -> tuple:
        if key in _PAD_KEYS:
            return _C_PAD
        if key in _ACCESSRES_KEYS:
            return _R_ACC
        if key in _EXT_KEYS:
            return _C_EXT
        return _C_INT

    font = _load_font(18)

    img  = Image.open(tpl_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Diagram-mode highlight: red ring on the component being edited.
    if highlight_key and highlight_key in overlay and not (
            no_parasitics and highlight_key in _PARASITIC_KEYS):
        hx, hy, *_ = overlay[highlight_key]
        rx, ry = 52, 26
        draw.ellipse([hx - rx, hy - ry, hx + rx, hy + ry],
                     outline=(220, 38, 38), width=5)

    for key, pos in overlay.items():
        if no_parasitics and key in _PARASITIC_KEYS:
            continue
        val_si = all_p.get(key)
        if val_si is None:
            continue
        try:
            text = _fmt_param(key, float(val_si))
        except Exception:
            continue
        px, py, *rest = pos
        anchor = rest[0] if rest else "mm"
        color = _color(key)
        # White stroke for readability, then colored text on top
        try:
            draw.text((px, py), text, font=font, fill=color,
                      anchor=anchor, stroke_width=2, stroke_fill=(255, 255, 255))
        except TypeError:
            # Older PIL: manual halo (no anchor support)
            for dx, dy in [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]:
                draw.text((px+dx, py+dy), text, font=font, fill=(255, 255, 255))
            draw.text((px, py), text, font=font, fill=color)

    if no_parasitics and smith_png is not None:
        from ._shared import composite_smith_overlay
        c_topo, c_smith = st.columns(2)
        with c_topo:
            buf = io.BytesIO(); img.save(buf, format="PNG")
            st.image(buf.getvalue(), width="stretch",
                     caption=tr("Topology", "拓樸圖"))
        with c_smith:
            try:
                img2 = composite_smith_overlay(img, smith_png)
                buf2 = io.BytesIO(); img2.save(buf2, format="PNG")
                st.image(buf2.getvalue(), width="stretch",
                         caption=tr("Topology + Smith", "拓樸圖 + Smith 圖"))
            except Exception as e:                       # noqa: BLE001
                st.warning(tr(f"Smith overlay unavailable: {e}",
                              f"無法疊加 Smith 圖：{e}"))
    else:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.image(buf.getvalue(), width="stretch")


def _override_ui(fname, tK, calc_vals, int_specs, label, ext_specs=_EXT_SPECS,
                  cache_ctx=None):
    """Render the override expander for the Xu T-topology."""
    all_specs = _XU_PAD_SPECS + ext_specs + int_specs
    sync_pad_from_preov(fname, tK, calc_vals)

    _int_keys = [k for k, *_ in ext_specs + int_specs]
    _sync_hash_key = f"sim_synchash_{tK}_{fname}"
    _sync_hash = params_hash({k: str(round(float(calc_vals.get(k, 0.0)), 15))
                            for k in _int_keys})
    # Also reseed when a widget key was GC'd by a page switch — see cheng.py.
    _keys_missing = any(f"sim_{tK}_{k}_{fname}" not in st.session_state
                        for k in _int_keys)
    if _keys_missing or st.session_state.get(_sync_hash_key) != _sync_hash:
        for key, _, scale, *_ in ext_specs + int_specs:
            st.session_state[f"sim_{tK}_{key}_{fname}"] = float(calc_vals.get(key, 0.0)) * scale
        st.session_state[_sync_hash_key] = _sync_hash

    with st.container(key=f"hbt_exp_edit_{tK}"), \
         st.expander(tr(f"✏️ Fine-tune {label} intrinsic/extrinsic parameters",
                        f"✏️ 微調 {label} 內部/外部參數"), expanded=False):
        rc1, rc2, rc3 = st.columns(3)
        if rc1.button(tr(f"↩️ Reset {label} to interactive section values",
                         f"↩️ 將 {label} 重設為互動區段數值"),
                     key=f"rst_sim_{tK}_{fname}", width="stretch"):
            for key, _, scale, *_ in all_specs:
                st.session_state[f"sim_{tK}_{key}_{fname}"] = float(calc_vals.get(key, 0.0)) * scale
            st.rerun()

        if (cache_ctx or {}).get("has_cache"):
            _ts = (cache_ctx or {}).get("ts")
            if rc2.container(key=f"hbt_amber_usecache_{tK}").button(
                    tr("📌 Use cache", "📌 使用快取"),
                    key=f"use_cache_{tK}_{fname}", width="stretch",
                    help=tr(f"Load the fit saved {_ts} for this device+model into "
                            "these fields. Pad fields keep following the "
                            "pre-extraction override.",
                            f"將此元件＋模型於 {_ts} 儲存的擬合結果載入這些欄位。"
                            "Pad 欄位仍會依循萃取前的覆寫值。")):
                st.session_state[cache_ctx["req_key"]] = True
                st.rerun()

        if rc3.button(tr("0️⃣ Reset all to 0", "0️⃣ 全部重設為 0"),
                     key=f"zero_sim_{tK}_{fname}",
                     width="stretch",
                     help=tr("Set every field in this expander to 0. The saved "
                             "cache is untouched — recover with Use cache.",
                             "將此展開區內所有欄位重設為 0。已儲存的快取不受影響 — "
                             "可用「使用快取」復原。")):
            for key, *_ in all_specs:
                st.session_state[f"sim_{tK}_{key}_{fname}"] = 0.0
            st.rerun()

        _mode_opts = [tr("List", "清單"), tr("Diagram", "示意圖")]
        _mode = segmented_radio(
            tr("Editor mode", "編輯模式"), _mode_opts,
            key=f"sim_mode_{tK}_{fname}",
            help=tr("List: grouped number inputs.  Diagram: set values on the "
                    "model schematic — the component you edit is highlighted.",
                    "清單：以分組數字輸入框編輯。示意圖：直接在模型示意圖上設定"
                    "數值 — 目前編輯的元件會以紅框標示。"))

        if _mode == _mode_opts[1]:
            render_finetune_diagram(
                all_specs=all_specs, calc_vals=calc_vals,
                state_key_for=lambda k: f"sim_{tK}_{k}_{fname}",
                active_state=f"sim_dia_active_{tK}_{fname}",
                render_illustration=lambda p, hl: _render_topology_illustration(
                    p, fname, highlight_key=hl))
        else:
            st.markdown(tr("**Pad Parasitics** *(auto-synced from pre-extraction override)*",
                           "**Pad 寄生參數** *(自動同步萃取前覆寫值)*"))
            for row_start in range(0, len(_XU_PAD_SPECS), 3):
                row = _XU_PAD_SPECS[row_start:row_start+3]
                for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                    col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                                       key=f"sim_{tK}_{key}_{fname}", format=fmt, step=step)

            st.markdown(tr("**Extrinsic Caps**", "**外部電容**"))
            for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(ext_specs)), ext_specs):
                col_w.number_input(f"{lbl} ({unit})", key=f"sim_{tK}_{key}_{fname}", format=fmt, step=step)

            st.markdown(tr("**Intrinsic**", "**內部參數**"))
            for row_start in range(0, len(int_specs), 4):
                row = int_specs[row_start:row_start+4]
                for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                    col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                                       key=f"sim_{tK}_{key}_{fname}", format=fmt, step=step)

    all_p = {key: st.session_state.get(f"sim_{tK}_{key}_{fname}",
                                        float(calc_vals.get(key, 0.0))*scale) / scale
             for key, _, scale, *_ in all_specs}
    # Propagate extended open/short params
    for ek in ["Cpbe_mode","Cpbe_extra","Cpce_mode","Cpce_extra",
               "Cpbc_mode","Cpbc_extra","Cpar_Lb","Cpar_Lc","Cpar_Le"]:
        all_p[ek] = calc_vals.get(ek, "None" if "mode" in ek else 0.0)
    return all_p


class XuModel(SSMModelTemplate, AbstractSSMModel):
    """
    Xu's T (2014) extraction.

    Ybcx = 1/Rbcx + jωCbcx is a parallel R∥C network between base and
    collector (no separate Cbex node).  Rbcx defaults to 285 kΩ and is
    user-tunable; Cbcx is extracted from Y_ex1 directly.

    Step 2 → Cbcx (Rbcx defaulted)  |  Step 3 → Rbi, Rbe, Cbe, Rbc, Cbc, α₀, τB, τC
    """
    NAME          = "Xu's T (2014)"
    SHORT         = "XuT"
    TOPOLOGY_CHAR = "T"
    _SVG_TOPOLOGY = True    # built-in custom-model preset available (svg_topology.py)

    SUPPORTS_FP32_SWEEP = True
    # ── Template hooks (see SSMModelTemplate in base_ui.py) ──────────────────
    _INT_SPECS         = _INT_T_SPECS
    _EXT_SPECS         = _EXT_T_SPECS
    _Y_INT_VEC_FN      = _Y_int_T_vec
    _Y_INT_BATCH_FN    = _Y_int_T_batch
    _SIM_WRAP_VEC_FN   = _sim_wrap_vec
    _SIM_WRAP_BATCH_FN = _sim_wrap_batch
    _TUNING_PAD_SPECS  = _XU_PAD_SPECS   # relabels Cpce → "Cpce / Cpad" in tuning UI
    # Pre-bake truth table: Xu shares Cheng-T's intrinsic topology, plus
    # the extrinsic Ybcx = 1/Rbcx + jωCbcx network (no Cbex node in Xu).
    STATIC_SUBNETWORKS = {
        "Y_pad":  frozenset({"Cpbe", "Cpce", "Cpbc"}),
        "Z_ser":  frozenset({"Rpb", "Rpc", "Rpe", "Lb", "Lc", "Le"}),
        "Ybcx":   frozenset({"Rbcx", "Cbcx"}),
        "Zbe":    frozenset({"Rbe", "Cbe"}),
        "Zbc":    frozenset({"Rbc", "Cbc"}),
        "alpha":  frozenset({"alpha0", "tauB", "tauC"}),
        "T_int_planes": frozenset({"Rbi", "Rbe", "Cbe", "Rbc", "Cbc",
                                    "alpha0", "tauB", "tauC"}),
    }
    PARAM_GROUPS  = [
        {
            "label":      "Step 2 — Cbcx  (from Y_ex1, parallel-Rbcx∥Cbcx network)",
            "params":     [("Cbcx_arr", "Cbcx", "Cbcx", 1e15, "fF")],
            "depends_on": [],
            "formulas": [
                ("latex", r"C_{bcx}=-\frac{\mathrm{Im}(Y_{ms})\mathrm{Re}(Y_L)-\mathrm{Re}(Y_{ms})\mathrm{Im}(Y_L)}{\omega[\mathrm{Re}(Y_{ms})\mathrm{Re}(Y_{tot})+\mathrm{Im}(Y_{tot})\mathrm{Im}(Y_{ms})]}"),
                ("markdown", "Rbcx is **not** extracted from data — defaults to 285 kΩ (tune in Fine-tune UI)."),
            ],

        },
        {
            "label":      "Step 3 — Intrinsic  (Rbi, Rbe, Cbe, Rbc, Cbc, α₀ — all from Z_in)",
            "params": [
                ("Rbi",   "Rbi",    "Rbi",  1.0,  "Ω"),
                ("Rbe",   "Rbe",    "Rbe",  1.0,  "Ω"),
                ("Cbe",   "Cbe",    "Cbe",  1e15, "fF"),
                ("Rbc",   "Rbc",    "Rbc (low frequency range)",  1e-3, "kΩ"),
                ("Cbc",   "Cbc",    "Cbc (low frequency range)",  1e15, "fF"),
                ("alpha", "alpha0", "α (low frequency range)",    1.0,  ""),
            ],
            "depends_on": ["Cbcx"],
            "use_first_params": {"Rbc", "Cbc", "alpha0"},
            "formulas": [
                ("latex", r"Z_{be}=Z_{12},\;Z_{bc}=Z_{22}-Z_{21},\;Z_{bi}=Z_{11}-Z_{12}"),
                ("latex", r"\alpha=\frac{Z_{12}-Z_{21}}{Z_{22}-Z_{21}},\;\alpha_0=|\alpha|_{\omega\to0}"),
            ],

        },
        {
            # Multi-file 1/(2πfT) vs 1/IC reference fit — rendered only when
            # ≥2 s2p files are loaded.  Reference Cje / τB+τC / τCC / τE;
            # does NOT feed back into the model fit.  See
            # ``base_ui._render_tau_total_fit_section``.
            "label":              "Reference: Cje / τB+τC / τCC / τE from 1/(2πfT) vs 1/IC",
            "params":             [],
            "depends_on":         [],
            "tau_total_fit_group": True,
        },
        {
            "label":      "τB  (depends on α₀)",
            "params":     [("tauB", "tauB", "τB", 1e12, "ps")],
            "depends_on": ["alpha0"],
            "formulas": [
                ("latex", r"\tau_B=\frac{\sqrt{U-1}}{\omega},\quad U=\left(\frac{\alpha_0}{|\alpha|}\right)^2"),
            ],

        },
        {
            "label":      "τC  (depends on τB)",
            "params":     [("tauC", "tauC", "τC", 1e12, "ps")],
            "depends_on": ["tauB"],
            "formulas": [
                ("latex", r"\tau_C=-\frac{\arctan\bigl[V(1-V^2)^{-1/2}\bigr]}{2\omega},\quad V=\frac{2\omega\tau_B}{U}"),
            ],

        },
    ]


    @classmethod
    def extract(cls, Y_ex1, freq, n_low, **kwargs):

        """
        Full extraction: Step 2 (Cbcx; Rbcx defaulted) → Step 3 (intrinsic T params).
        See _step2_T and _step3_T for formula references.
        """
        res_ext, arr_ext = _step2_T(Y_ex1, freq)
        res_int, arr_int = _step3_T(Y_ex1, freq,
                                    res_ext["Cbcx"], res_ext["Rbcx"], n_low)
        params = {**res_ext, **res_int}
        arrays = {**arr_ext, **arr_int, "_res_ext": res_ext, "_res_int": res_int}
        return params, arrays

    @classmethod
    def simulate(cls, params, freq, z0=50.0):
        """
        Forward sim — inside-out:
          [Z_in]  →  add Ybcx (= 1/Rbcx + jωCbcx)  →  [Y_ex]  →  add Z_ser  →  add Y_pad  →  S
        """
        def _Y_int(p, w):
            # T intrinsic Y matrix (Z_in inverse)
            Zbe_v = p["Rbe"] / (1.0 + 1j*w*p["Rbe"]*p["Cbe"])
            Zbc_v = p["Rbc"] / (1.0 + 1j*w*p["Rbc"]*p["Cbc"])
            alpha = p["alpha0"] * np.exp(-1j*w*p["tauC"]) / (1.0 + 1j*w*p["tauB"])
            Z_in  = np.array([[p["Rbi"]+Zbe_v, Zbe_v],
                               [Zbe_v - alpha*Zbc_v, (1-alpha)*Zbc_v + Zbe_v]])
            try:
                return np.linalg.inv(Z_in)
            except np.linalg.LinAlgError:
                return np.zeros((2, 2), dtype=complex)
        return _sim_wrap(_Y_int, params, freq, z0)

    # simulate_vec, simulate_batch — inherited from SSMModelTemplate

    @classmethod
    def reextract(cls, Y_ex1, freq, n_low, overrides, changed_group_idx, live_arrays):
        """
        Re-derive all downstream parameters when an upstream group is overridden.
        Group indices match PARAM_GROUPS (note: index 2 is the multi-file
        tau_total_fit reference group inserted between Step 3 and τB; it
        contains no params and never triggers reextract):
          changed_group_idx=0 (Cbcx changed)  → use overrides["Cbcx"], re-run Step 3
          changed_group_idx=1 (Step-3 α₀)     → recompute τB and τC arrays
          changed_group_idx=3 (τB changed)    → recompute τC array only

        Rbcx is read from overrides (or _RBCX_DEFAULT) — it's not extracted
        from data but the user can override it via Fine-tune.
        """
        omega = 2.0 * np.pi * freq

        # ── Recompute Cbcx_arr from Y_ex1 directly (no Cbex peel) ───────────
        Yms  = Y_ex1[:, 0, 1] + Y_ex1[:, 1, 1]
        YL   = Y_ex1[:, 0, 0]*Y_ex1[:, 1, 1] - Y_ex1[:, 0, 1]*Y_ex1[:, 1, 0]
        Ytot = Y_ex1[:, 0, 0] + Y_ex1[:, 0, 1] + Y_ex1[:, 1, 0] + Y_ex1[:, 1, 1]
        num  = np.imag(Yms)*np.real(YL) - np.real(Yms)*np.imag(YL)
        den  = np.real(Yms)*np.real(Ytot) + np.imag(Ytot)*np.imag(Yms)
        with np.errstate(divide="ignore", invalid="ignore"):
            Cbcx_arr = -np.where(np.abs(den) > 1e-40, num / (omega * den), np.nan)
        n0c, n1c = len(freq) // 4, 3 * len(freq) // 4
        Cbcx_recomp = abs(safe_median(Cbcx_arr[n0c:n1c]))

        # User-overridden Cbcx only when Group 0 (Cbcx) was explicitly changed
        Cbcx = float(overrides["Cbcx"]) if (changed_group_idx >= 0
                                             and "Cbcx" in overrides) else Cbcx_recomp
        Rbcx = float(overrides.get("Rbcx", _RBCX_DEFAULT))

        # ── Re-run Step 3 with current Ybcx peel ─────────────────────────────
        res_int, arr_int = _step3_T(Y_ex1, freq, Cbcx, Rbcx, n_low)

        # ── Handle within-Step-3 overrides (α₀ → τB → τC) ───────────────────
        alpha_arr = arr_int["alpha"]   # complex, per-frequency

        if changed_group_idx >= 3 and "alpha0" in overrides:
            alpha0_ov = float(overrides["alpha0"])
            U_arr     = (alpha0_ov / (np.abs(alpha_arr) + 1e-30)) ** 2
            tauB_arr  = np.sqrt(np.maximum(U_arr - 1.0, 0.0)) / omega
            tauB_ov   = safe_median(tauB_arr[n_low:])
            with np.errstate(divide="ignore", invalid="ignore"):
                V_arr    = 2.0 * omega * tauB_ov / (U_arr + 1e-30)
                tauC_arr = -np.arctan(
                    V_arr / np.sqrt(np.maximum(1.0 - V_arr**2, 1e-30))
                ) / (2.0 * omega)
            tauC_ov = safe_median(tauC_arr[n_low:])
            arr_int["tauB"] = tauB_arr;  res_int["tauB"] = tauB_ov
            arr_int["tauC"] = tauC_arr;  res_int["tauC"] = tauC_ov

        elif changed_group_idx >= 4 and "tauB" in overrides:
            alpha0_cur = float(overrides.get("alpha0", res_int["alpha0"]))
            U_arr      = (alpha0_cur / (np.abs(alpha_arr) + 1e-30)) ** 2
            tauB_ov    = float(overrides["tauB"])
            with np.errstate(divide="ignore", invalid="ignore"):
                V_arr    = 2.0 * omega * tauB_ov / (U_arr + 1e-30)
                tauC_arr = -np.arctan(
                    V_arr / np.sqrt(np.maximum(1.0 - V_arr**2, 1e-30))
                ) / (2.0 * omega)
            tauC_ov = safe_median(tauC_arr[n_low:])
            arr_int["tauC"] = tauC_arr;  res_int["tauC"] = tauC_ov

        new_params = {"Cbcx": Cbcx, "Rbcx": Rbcx, **res_int}
        new_arrays = {
            "Cbcx_arr": Cbcx_arr,
            **arr_int,
        }
        return new_params, new_arrays

    @classmethod
    def _results_rows(cls, params):
        ri = params
        return [
            ("Cbcx", f"{ri['Cbcx']*1e15:.4f}", "fF"),   # Step 2 — extracted
            ("Rbcx", f"{ri.get('Rbcx', _RBCX_DEFAULT)*1e-3:.4f}", "kΩ"),  # defaulted (user-tunable)
            ("Rbi",  f"{ri['Rbi']:.4f}",        "Ω"),   # Step 3
            ("Rbe",  f"{ri['Rbe']:.4f}" if ri['Rbe']<1000 else f"{ri['Rbe']*1e-3:.4f}k", "Ω"),
            ("Cbe",  f"{ri['Cbe']*1e15:.4f}" if ri['Cbe']<1e-12 else f"{ri['Cbe']*1e12:.4f}",
                     "fF" if ri['Cbe']<1e-12 else "pF"),
            ("Rbc",  f"{ri['Rbc']*1e-3:.4f}", "kΩ"),
            ("Cbc",  f"{ri['Cbc']*1e15:.4f}", "fF"),
            ("α₀",   f"{ri['alpha0']:.5f}",   ""),
            ("τB",   f"{ri['tauB']*1e12:.4f}", "ps"),
            ("τC",   f"{ri['tauC']*1e12:.4f}", "ps"),
        ]

    @classmethod
    def _render_results_trace(cls):
        with st.expander(tr("📐 Full formula trace — Xu's T (2014)",
                            "📐 完整公式推導 — Xu's T (2014)"), expanded=False):
            st.markdown(tr("**Dependency chain:** Y_ex1 → peel Ybcx (= 1/Rbcx + jωCbcx) → Z_in → intrinsic",
                           "**相依鏈：** Y_ex1 → 剝離 Ybcx (= 1/Rbcx + jωCbcx) → Z_in → 內部參數"))
            st.markdown(tr("**Step 2** *(input: Y_ex1; Rbcx defaulted to 285 kΩ)*",
                           "**步驟 2** *(輸入：Y_ex1；Rbcx 預設為 285 kΩ)*"))
            st.latex(r"C_{bcx}=-\frac{\mathrm{Im}(Y_{ms})\mathrm{Re}(Y_L)"
                     r"-\mathrm{Re}(Y_{ms})\mathrm{Im}(Y_L)}{\omega"
                     r"[\mathrm{Re}(Y_{ms})\mathrm{Re}(Y_{tot})+\mathrm{Im}(Y_{tot})\mathrm{Im}(Y_{ms})]}")
            st.markdown(tr("**Step 3** *(input: Y_ex1, Cbcx, Rbcx)*",
                           "**步驟 3** *(輸入：Y_ex1, Cbcx, Rbcx)*"))
            st.latex(r"Z_{be}=Z_{12},\;Z_{bc}=Z_{22}-Z_{21},\;Z_{bi}=Z_{11}-Z_{12}")
            st.latex(r"\alpha=\frac{Z_{12}-Z_{21}}{Z_{22}-Z_{21}},\;"
                     r"\alpha_0=|\alpha||_{\omega\to0}")
            st.latex(r"\tau_B=\frac{\sqrt{U-1}}{\omega}")
            st.latex(r"\tau_C=-\frac{\arctan[V(1-V^2)^{-1/2}]}{2\omega}")
            st.markdown(tr("**Forward simulation** *(inside → outside)*",
                           "**正向模擬** *(由內而外)*"))
            st.latex(r"Z_{be}^{sim}=\frac{R_{be}}{1+j\omega R_{be}C_{be}},\;"
                     r"\alpha=\alpha_0 e^{-j\omega\tau_C}/(1+j\omega\tau_B)")
            st.latex(r"[Z_{in}^{sim}]=\begin{bmatrix}R_{bi}+Z_{be}&Z_{be}\\"
                     r"Z_{be}-\alpha Z_{bc}&(1-\alpha)Z_{bc}+Z_{be}\end{bmatrix}")
            st.latex(r"[Y_{ex}]=[Z_{in}]^{-1}+\left(\tfrac{1}{R_{bcx}}+j\omega C_{bcx}\right)"
                     r"\begin{pmatrix}1&-1\\-1&1\end{pmatrix}")
            st.latex(r"[Y_{tot}]=([Y_{ex}]^{-1}+[Z_{ser}])^{-1}\;,\quad "
                     r"S=(I-Z_0[Y_{tot}+Y_{pad}])(I+Z_0[Y_{tot}+Y_{pad}])^{-1}")

    @classmethod
    def _do_override_ui(cls, fname, calc_vals, cache_ctx=None):
        return _override_ui(fname, cls.SHORT, calc_vals, _INT_T_SPECS, cls.NAME,
                            ext_specs=_EXT_T_SPECS, cache_ctx=cache_ctx)

    @classmethod
    def _render_topology(cls, all_p, fname, smith_png=None, highlight_key=None):
        _render_topology_illustration(all_p, fname, smith_png=smith_png,
                                      highlight_key=highlight_key)

    @classmethod
    def _build_intrinsic_static_cache(cls, p, omega, cache, xp, prebakeable):
        """Xu T uses Cheng-T's intrinsic topology + a parallel
        Ybcx = 1/Rbcx + jωCbcx extrinsic network (no Cbex node)."""
        # Reuse Cheng-T's builder for the intrinsic block.
        from .cheng import ChengT as _ChengT
        _ChengT._build_intrinsic_static_cache(p, omega, cache, xp,
                                               prebakeable)
        if "Ybcx" in prebakeable:
            rbcx_c = float(p.get("Rbcx", 285e3))
            cbcx_c = float(p.get("Cbcx", 0.0))
            cache["Ybcx"] = (1.0 / rbcx_c) + 1j * omega * cbcx_c

