"""
models/cheng.py — Cheng (2022) T-topology and π-topology HBT small-signal models.

Reference: Cheng et al., Microelectronics Journal vol. 121, 2022
           Eqs. 13, 16, 18–19, 21–22, 26–33

Both models share the same two-step extrinsic extraction (Step 2) that produces
Y_ex2, then diverge in Step 3.  Each is its own class so the registry and UI
treat them independently, but shared logic lives in module-level helpers below.
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path
import numpy as np
import pandas as pd
import streamlit as st

from ..helpers         import (y_to_z, z_to_y, y_to_s_single, y_to_s_vec,
                                inv2x2, mm2x2,
                                safe_median, params_hash,
                                extended_smith_grid,
                                build_Y_pad, build_Z_ser,
                                build_Y_pad_vec, build_Z_ser_vec,
                                build_Y_pad_batch, build_Z_ser_batch)
from .base_ui         import (smith_scale_controls,
                               sync_pad_from_preov, PAD_SPECS,
                               render_tuning_expander, render_smith_with_ftfmax)
from . import AbstractSSMModel


# ════════════════════════════════════════════════════════════════════════════════
# Shared Step-2 helpers
# ════════════════════════════════════════════════════════════════════════════════

def _sweep_cbex_stds_cheng(Y_ex1, freq, cbex_SI_array, mask):
    """For each candidate Cbex, rebuild Y_ex2 and return std(Cbcx_arr[mask]).

    Both Cheng T and π use the same Cbcx formula [Eq. 22] once Y_ex2 has been
    built by peeling Cbex from Y_ex1[0,0], so the sweep helper is shared.

    Arguments
    ---------
    Y_ex1          : (N, 2, 2) complex — de-embedded Y before Cbex peel
    freq           : (N,) real — frequency axis (Hz)
    cbex_SI_array  : (K,) real — candidate Cbex values in SI units (F)
    mask           : (N,) bool — frequency window over which to compute std

    Returns
    -------
    stds : (K,) real — std of Cbcx_arr[mask] for each candidate Cbex.
           Candidates whose mask yields < 2 finite points get `inf`.
    """
    omega = 2.0 * np.pi * freq
    K = len(cbex_SI_array)
    stds = np.empty(K, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    for k, Cb in enumerate(cbex_SI_array):
        Y_ex2 = Y_ex1.copy()
        # Vectorised Y[0,0] peel: -= j ω Cb for all frequencies at once
        Y_ex2[:, 0, 0] = Y_ex2[:, 0, 0] - 1j * omega * float(Cb)
        Yms  = Y_ex2[:, 0, 1] + Y_ex2[:, 1, 1]
        YL   = Y_ex2[:, 0, 0]*Y_ex2[:, 1, 1] - Y_ex2[:, 0, 1]*Y_ex2[:, 1, 0]
        Ytot = Y_ex2[:, 0, 0] + Y_ex2[:, 0, 1] + Y_ex2[:, 1, 0] + Y_ex2[:, 1, 1]
        num  = np.imag(Yms)*np.real(YL) - np.real(Yms)*np.imag(YL)
        den  = np.real(Yms)*np.real(Ytot) + np.imag(Ytot)*np.imag(Yms)
        with np.errstate(divide="ignore", invalid="ignore"):
            Cbcx_arr = -np.where(np.abs(den) > 1e-40, num/(omega*den), np.nan)
        vals = Cbcx_arr[mask]
        vals = vals[np.isfinite(vals)]
        stds[k] = float(np.std(vals)) if len(vals) >= 2 else np.inf
    return stds


def _step2_T(Y_ex1, freq, n_low):
    """
    Cheng [Eqs. 13, 22] — Extract Cbex and Cbcx.

    Cbex_T = Im(Y11 + Y12) / ω  [Eq. 13]

    Cbcx = −[Im(Yms)·Re(YL) − Re(Yms)·Im(YL)] / [ω · denominator]  [Eq. 22]
    where  Yms = Y12+Y22,  YL = det(Y_ex2),  Ytot = sum(Yij)
    """
    omega = 2.0*np.pi*freq

    # [Eq. 13] Cbex from low-frequency Im(Y11+Y12)/ω
    Cbex_arr = np.imag(Y_ex1[:,0,0] + Y_ex1[:,0,1]) / omega
    Cbex = abs(safe_median(Cbex_arr, n_low))


    # Peel Cbex (Y11) (Y22) to get Y_ex2
    Y_ex2 = Y_ex1.copy()
    for i, w in enumerate(omega): # at all frequency
        Y_ex2[i,0,0] -= 1j*w*Cbex

    Yms   = Y_ex2[:,0,1] + Y_ex2[:,1,1] # eq 21
    YL    = Y_ex2[:,0,0]*Y_ex2[:,1,1] - Y_ex2[:,0,1]*Y_ex2[:,1,0] # eq 18
    Ytot  = Y_ex2[:,0,0] + Y_ex2[:,0,1] + Y_ex2[:,1,0] + Y_ex2[:,1,1] # eq 19
    # [Eq. 22] Cbcx
    num   = np.imag(Yms)*np.real(YL) - np.real(Yms)*np.imag(YL)
    den   = np.real(Yms)*np.real(Ytot) + np.imag(Ytot)*np.imag(Yms)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cbcx_arr = -np.where(np.abs(den) > 1e-40, num/(omega*den), np.nan)
    n0, n1 = len(freq)//4, 3*len(freq)//4
    Cbcx = abs(safe_median(Cbcx_arr[n0:n1]))
    return ({"Cbex": Cbex, "Cbcx": Cbcx},
            {"Cbex_arr": Cbex_arr, "Cbcx_arr": Cbcx_arr,
            "Y_ex2": Y_ex2})


def _step2_pi(Y_ex1, freq, n_low):
    """
    Cheng [Eqs. 26–28] — Extract Cbex (π variant) and Cbcx.

    B = Y12+Y22,  C = Y11+Y21
    Cbex_π = [Re(B)·Re(C) + Im(B)·Im(C)] / [ω·Im(B)]  [Eqs. 26–28]
    Cbcx same formula as T-topology [Eq. 22] applied on Y_ex2.
    """
    omega = 2.0*np.pi*freq
    B = Y_ex1[:,0,1] + Y_ex1[:,1,1]
    C = Y_ex1[:,0,0] + Y_ex1[:,1,0]

    # [Eqs. 26–28]
    with np.errstate(divide="ignore", invalid="ignore"):
        Cbex_arr = np.where(
            np.abs(np.imag(B)) > 1e-40,
            (np.real(B)*np.real(C) + np.imag(B)*np.imag(C)) / (omega*np.imag(B)),
            np.nan)
    Cbex = safe_median(Cbex_arr, n_low)

    Y_ex2 = Y_ex1.copy()
    for i, w in enumerate(omega):
        Y_ex2[i,0,0] -= 1j*w*Cbex

    # Cbcx [Eq. 22] — same as T
    Yms  = Y_ex2[:,0,1] + Y_ex2[:,1,1]
    YL   = Y_ex2[:,0,0]*Y_ex2[:,1,1] - Y_ex2[:,0,1]*Y_ex2[:,1,0]
    Ytot = Y_ex2[:,0,0] + Y_ex2[:,0,1] + Y_ex2[:,1,0] + Y_ex2[:,1,1]
    num  = np.imag(Yms)*np.real(YL) - np.real(Yms)*np.imag(YL)
    den  = np.real(Yms)*np.real(Ytot) + np.imag(Ytot)*np.imag(Yms)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cbcx_arr = -np.where(np.abs(den) > 1e-40, num/(omega*den), np.nan)
    n0, n1 = len(freq)//4, 3*len(freq)//4
    Cbcx = safe_median(Cbcx_arr[n0:n1])

    return ({"Cbex": Cbex, "Cbcx": Cbcx},
            {"Cbex_arr": Cbex_arr, "Cbcx_arr": Cbcx_arr, "Y_ex2": Y_ex2})


def _step3_T(Y_ex2, freq, Cbcx, n_low):
    """
    Cheng [Eqs. 16, 29–33] — Extract intrinsic parameters for T-topology.

    Z_in = [Y_ex2 − Cbcx·shunt]⁻¹
    Zbe = Z12,   Zbc = Z22−Z21,   Zbi = Z11−Z12          [Eq. 16]
    α = (Z12−Z21) / (Z22−Z21),   α₀ = |α|ω→0             [Eq. 29]
    τB = √(U−1)/ω   where U = (α₀/|α|)²                  [Eq. 30]
    τC = −arctan(V/√(1−V²)) / (2ω)   where V=2ωτB/U      [Eq. 31]
    """
    omega = 2.0*np.pi*freq

    # Peel Cbcx to get intrinsic Y
    Y_in = Y_ex2.copy()
    for i, w in enumerate(omega):
        Ybcx = 1j*w*Cbcx
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


def _step3_pi(Y_ex2, freq, Cbcx, n_low):
    """
    Zhang et al. [4] — Extract intrinsic parameters for π-topology.

    Z_bc = Z22−Z21
    gm = (Z12−Z21) / (Zbc·Z12)  →  Gm0 = |gm|,  τ = −∠gm / ω
    Ybe = (Z22−Z12) / (Z12·Zbc)  →  Rbe, Cbe
    Rbi = Re(Z11−Z12)
    """
    omega = 2.0*np.pi*freq

    Y_in = Y_ex2.copy()
    for i, w in enumerate(omega):
        Ybcx = 1j*w*Cbcx
        Y_in[i,0,0] -= Ybcx; Y_in[i,0,1] += Ybcx
        Y_in[i,1,0] += Ybcx; Y_in[i,1,1] -= Ybcx

    Z_in = y_to_z(Y_in)
    Z12  = Z_in[:,0,1]; Z21 = Z_in[:,1,0]
    Z11  = Z_in[:,0,0]; Z22 = Z_in[:,1,1]
    Zbc  = Z22 - Z21  # [from Eq. 16 shared]

    with np.errstate(divide="ignore", invalid="ignore"):
        Ybc_arr = 1.0 / (Zbc + 1e-40)
        gm_arr  = (Z12 - Z21) / ((Zbc + 1e-40)*(Z12 + 1e-40))
        Ybe_arr = (Z22 - Z12) / ((Z12 + 1e-40)*(Zbc + 1e-40))
        Rbi_a   = np.real(Z11 - Z12)

    Rbc_a = 1.0 / np.real(Ybc_arr).clip(1e-12)
    Cbc_a = np.imag(Ybc_arr) / omega
    Rbe_a = 1.0 / np.real(Ybe_arr).clip(1e-6)
    Cbe_a = np.imag(Ybe_arr) / omega
    Gm0_a = np.abs(gm_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_a = -np.angle(gm_arr) / omega

    Rbi  = safe_median(Rbi_a, n_low)
    Cbc  = safe_median(Cbc_a, n_low)
    Rbe  = safe_median(Rbe_a, n_low); Cbe = safe_median(Cbe_a, n_low)
    Gm0  = safe_median(Gm0_a, n_low); tau = safe_median(tau_a,  n_low)

    params = dict(Rbi=Rbi, Rbe=Rbe, Cbe=Cbe, Cbc=Cbc, Gm0=Gm0, tau=tau)
    arrays = dict(Rbi=Rbi_a, Rbe=Rbe_a, Cbe=Cbe_a, Cbc=Cbc_a, Gm0=Gm0_a, tau=tau_a)
    return params, arrays


# ════════════════════════════════════════════════════════════════════════════════
# Shared forward-simulator helpers
# ════════════════════════════════════════════════════════════════════════════════

def _sim_wrap(Y_int_fn, p, freq, z0):
    """Add extrinsic caps + pad/lead parasitics around the intrinsic Y matrix."""
    omega = 2.0*np.pi*freq
    S = np.zeros((len(freq), 2, 2), dtype=complex)
    for i, w in enumerate(omega):
        Y_in  = Y_int_fn(p, w)
        Ybcx  = 1j*w*p["Cbcx"]
        Ybex  = 1j*w*p["Cbex"]
        Y_ex  = (Y_in
                 + Ybcx*np.array([[1,-1],[-1,1]])
                 + Ybex*np.array([[1, 0],[ 0,0]]))
        Z_ser = build_Z_ser(p, w)
        try:    Y_tot = np.linalg.inv(np.linalg.inv(Y_ex) + Z_ser)
        except: Y_tot = np.zeros((2,2), dtype=complex)
        Y_pad = build_Y_pad(p, w)
        S[i]  = y_to_s_single(Y_tot + Y_pad, z0)
    return S


# ── Vectorised forward simulation (no per-freq loop) ─────────────────────────

def _sim_wrap_vec(Y_int_vec_fn, p, freq, z0, xp):
    """Vectorised version of _sim_wrap — processes all freq points at once.

    Works identically with numpy (CPU) and cupy (GPU).
    *Y_int_vec_fn(p, omega, xp)* must return an (N, 2, 2) intrinsic-Y array.
    """
    omega = xp.asarray(2.0 * np.pi * freq, dtype=np.float64)

    # Intrinsic admittance (N, 2, 2)
    Y_in = Y_int_vec_fn(p, omega, xp)

    # Extrinsic caps: Cbcx, Cbex
    Ybcx = 1j * omega * p["Cbcx"]              # (N,)
    Ybex = 1j * omega * p["Cbex"]              # (N,)

    Y_ex = Y_in.copy()
    Y_ex[:, 0, 0] += Ybcx + Ybex
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


def _Y_int_Pi_vec(p, omega, xp):
    """Vectorised Pi-topology intrinsic Y matrix → (N, 2, 2)."""
    Ybe = 1.0 / p["Rbe"] + 1j * omega * p["Cbe"]
    Ybc = 1.0 / p.get("Rbc", 1e9) + 1j * omega * p["Cbc"]
    gm  = p["Gm0"] * xp.exp(-1j * omega * p["tau"])
    N = len(omega)
    Y_core = xp.zeros((N, 2, 2), dtype=complex)
    Y_core[:, 0, 0] = Ybe + Ybc
    Y_core[:, 0, 1] = -Ybc
    Y_core[:, 1, 0] = gm - Ybc
    Y_core[:, 1, 1] = Ybc
    Z_core = xp.linalg.inv(Y_core)
    Z_core[:, 0, 0] += p["Rbi"]
    return xp.linalg.inv(Z_core)


# ── Batched (B, N, 2, 2) forward simulation for parameter-sweep tuning ─────

def _b1(p, key, default, xp, dtype=None):
    """Fetch p[key] (or default) and reshape (B,) → (B,1).  Scalars stay scalar.

    If ``dtype`` is given, the value is coerced to that dtype.  Used by the
    fp32 sweep path so a scalar Python ``float`` constant doesn't promote
    a (B,N) ``float32`` swept tensor back up to ``float64``.
    """
    v = p.get(key, default)
    if dtype is not None:
        a = xp.asarray(v, dtype=dtype)
    else:
        a = xp.asarray(v)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _detect_B(p, xp):
    """Determine batch size B from any (B,)-shaped value in p."""
    B = 1
    for v in p.values():
        if isinstance(v, str):
            continue
        try:
            a = xp.asarray(v)
        except Exception:
            continue
        if a.ndim == 1 and a.shape[0] > B:
            B = a.shape[0]
    return B


def _stack22(a00, a01, a10, a11, xp):
    """Stack four (..., ) planes into a (..., 2, 2) tensor.

    Avoids the ``xp.zeros + scatter assignments`` pattern (5 kernel
    launches) — does it in 3 launches via xp.stack and amortises better
    on the GPU.  Inputs may be any broadcastable shapes; the result has
    the broadcast shape with two extra trailing axes.
    """
    return xp.stack(
        [xp.stack([a00, a01], axis=-1),
         xp.stack([a10, a11], axis=-1)],
        axis=-2,
    )


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
      - ``"Y_extr"`` : (Ybex, Ybcx) — extrinsic-cap admittances
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

    # Extrinsic caps (broadcastable to (B, N)) — cache-aware.
    if cache is not None and "Y_extr" in cache:
        Ybex, Ybcx = cache["Y_extr"][:2]
    else:
        Cbcx = _b1(p, "Cbcx", 0.0, xp, rdtype)
        Cbex = _b1(p, "Cbex", 0.0, xp, rdtype)
        jw   = J * omega
        Ybcx = jw * Cbcx
        Ybex = jw * Cbex

    # Y_ex = Y_in + extrinsic-cap network (additions on the four planes)
    ye00 = yi00 + Ybcx + Ybex
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


def _Y_int_Pi_batch(p, omega, B, N, xp, cache=None):
    """Batched Pi-topology intrinsic Y → 4 (B, N) planes (y00, y01, y10, y11).

    Cache-aware (mirrors ``_Y_int_T_batch``):

      - ``"Pi_int_planes"`` : full (yi00..yi11) tuple — used when *every*
        intrinsic param (Rbi, Rbe, Cbe, Rbc, Cbc, Gm0, tau) is constant.
      - ``"Ybe"``           : pre-built when Rbe and Cbe are both constant.
      - ``"Ybc"``           : pre-built when Rbc and Cbc are both constant.
      - ``"gm"``            : pre-built when Gm0 and tau are both constant.

    The dtype is propagated through ``cache["_cdtype"]``.
    """
    if cache is not None and "Pi_int_planes" in cache:
        return cache["Pi_int_planes"]

    cdtype = (cache or {}).get("_cdtype", np.complex128)
    rdtype = np.float32 if cdtype == np.complex64 else np.float64
    J = xp.asarray(1j, dtype=cdtype)

    Rbi = _b1(p, "Rbi", 0.0, xp, rdtype)

    if cache is not None and "Ybe" in cache:
        Ybe = cache["Ybe"]
    else:
        Rbe = _b1(p, "Rbe", 1.0, xp, rdtype)
        Cbe = _b1(p, "Cbe", 0.0, xp, rdtype)
        Ybe = 1.0 / Rbe + J * omega * Cbe

    if cache is not None and "Ybc" in cache:
        Ybc = cache["Ybc"]
    else:
        Rbc = _b1(p, "Rbc", 1e9, xp, rdtype)
        Cbc = _b1(p, "Cbc", 0.0, xp, rdtype)
        Ybc = 1.0 / Rbc + J * omega * Cbc

    if cache is not None and "gm" in cache:
        gm = cache["gm"]
    else:
        Gm0 = _b1(p, "Gm0", 0.0, xp, rdtype)
        tau = _b1(p, "tau", 0.0, xp, rdtype)
        gm  = Gm0 * xp.exp(-J * omega * tau)

    # Y_core 2×2
    yc00 = Ybe + Ybc
    yc01 = -Ybc
    yc10 = gm - Ybc
    yc11 = Ybc

    # Z_core = inv(Y_core)
    inv_det_c = 1.0 / (yc00 * yc11 - yc01 * yc10)
    zc00 =  yc11 * inv_det_c
    zc01 = -yc01 * inv_det_c
    zc10 = -yc10 * inv_det_c
    zc11 =  yc00 * inv_det_c

    # Add Rbi to Z_core[0,0]
    zc00 = zc00 + Rbi

    # Y_in = inv(Z_core)
    inv_det_i = 1.0 / (zc00 * zc11 - zc01 * zc10)
    y00 =  zc11 * inv_det_i
    y01 = -zc01 * inv_det_i
    y10 = -zc10 * inv_det_i
    y11 =  zc00 * inv_det_i
    return y00, y01, y10, y11


# ── Override UI specs (used by render_override_and_smith) ─────────────────────

_EXT_SPECS = [
    ("Cbex","Cbex",1e15,"fF","%.4f",0.1),
    ("Cbcx","Cbcx",1e15,"fF","%.4f",0.1),
]
# Per-topology extrinsic specs
_EXT_T_SPECS  = _EXT_SPECS
_EXT_PI_SPECS = _EXT_SPECS
_INT_T_SPECS = [
    ("Rbi",   "Rbi", 1.0, "Ω",  "%.4f", 0.1),
    ("Rbe",   "Rbe", 1.0, "Ω",  "%.3f", 1.0),
    ("Cbe",   "Cbe", 1e15,"fF", "%.4f", 0.1),
    ("Rbc",   "Rbc", 1e-3,"kΩ", "%.4f", 0.01),
    ("Cbc",   "Cbc", 1e15,"fF", "%.4f", 0.01),
    ("alpha0","α₀",  1.0, "",   "%.5f", 0.001),
    ("tauB",  "τB",  1e12,"ps", "%.4f", 0.01),
    ("tauC",  "τC",  1e12,"ps", "%.4f", 0.01),
]
_INT_PI_SPECS = [
    ("Rbi","Rbi", 1.0, "Ω",  "%.4f", 0.1),
    ("Rbe","Rbe", 1.0, "Ω",  "%.3f", 1.0),
    ("Cbe","Cbe", 1e15,"fF", "%.4f", 0.1),
    ("Cbc","Cbc", 1e15,"fF", "%.4f", 0.01),
    ("Gm0","Gm0", 1e3, "mS", "%.4f", 0.01),
    ("tau","τ",   1e12,"ps", "%.4f", 0.01),
]


# ════════════════════════════════════════════════════════════════════════════════
# Topology illustration helpers
# ════════════════════════════════════════════════════════════════════════════════

_ILLUS_DIR = _Path(__file__).parent / "illus_template"
_FONT_CACHE_DIR = _Path(__file__).parent / "fonts"
_INTER_DOWNLOAD_URLS = (
    "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf",
    "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Regular.ttf",
)

def _try_download_inter():
    """Attempt to download Inter once and cache it. Returns the cached path or None."""
    target = _FONT_CACHE_DIR / "Inter-Regular.ttf"
    if target.exists():
        return target
    try:
        _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        import urllib.request
        for url in _INTER_DOWNLOAD_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = resp.read()
                if data and len(data) > 10_000:
                    target.write_bytes(data)
                    return target
            except Exception:
                continue
    except Exception:
        pass
    return None

def has_inter():
    for name in ("Inter-Regular.ttf", "Inter.ttf"):
        if _os.path.exists(name):
            return True
    cached = _FONT_CACHE_DIR / "Inter-Regular.ttf"
    if cached.exists():
        return True
    return _try_download_inter() is not None

ohm_sign = "Ω" if has_inter() else "Ohm"

# Display units for each parameter key: (SI→display scale factor, base unit string)
_PARAM_DISPLAY: dict[str, tuple] = {
    "Cpbe":   (1e15, "fF"),  "Cpce":  (1e15, "fF"),  "Cpbc":  (1e15, "fF"),
    "Lb":     (1e12, "pH"),  "Lc":    (1e12, "pH"),   "Le":    (1e12, "pH"),
    "Rpb":    (1,    ohm_sign),   "Rpc":   (1,    ohm_sign),    "Rpe":   (1,    ohm_sign),
    "Cbex":   (1e15, "fF"),  "Cbcx":  (1e15, "fF"),
    "Rbi":    (1,    ohm_sign),   "Rbe":   (1,    ohm_sign),
    "Cbe":    (1e15, "fF"),  "Cbc":   (1e15, "fF"),
    "Rbc":    (1e-3, f"k{ohm_sign}"),
    "alpha0": (1,    ""),
    "tauB":   (1e12, "ps"),  "tauC":  (1e12, "ps"),
    "Gm0":    (1e3,  "mS"),
    "tau":    (1e12, "ps"),
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
_3DP_PARAMS = {"alpha0", "Gm0"}

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

    # External
    "Cbex": (300, 502, "rm"),
    "Cbcx": (512, 160),

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

_PI_EXTRA_OVERLAY: dict[str, tuple] = {
    "Rbe":  (408, 475, "lm"),
    "Cbe":  (550, 515, "lm"),
    "Cbc":  (590, 340),
    "Gm0":  (770, 470, "lm"),
    "tau":  (770, 502, "lm"),
}

_T_OVERLAY  = {**_COMMON_OVERLAY, **_T_EXTRA_OVERLAY}
_PI_OVERLAY = {**_COMMON_OVERLAY, **_PI_EXTRA_OVERLAY}


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


def _load_font(size: int):
    """Load a TrueType font at the given size, with Inter → Arial → fallback chain."""
    from PIL import ImageFont
    candidates = [
        "Inter-Regular.ttf", "Inter.ttf",
        "arial.ttf", "Arial.ttf",
        "segoeui.ttf", "tahoma.ttf", "calibri.ttf",
    ]
    win_fonts = _os.path.join(_os.environ.get("WINDIR", "C:/Windows"), "Fonts")
    dirs = [
        str(_FONT_CACHE_DIR),
        win_fonts,
        "/usr/share/fonts/truetype",
        "/usr/share/fonts/truetype/liberation",
        "/System/Library/Fonts",
    ]
    for name in candidates:
        for d in dirs:
            path = _os.path.join(d, name)
            if _os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _render_topology_illustration(all_p: dict, topology: str, fname: str) -> None:
    """
    Overlay live parameter values on the circuit schematic template PNG and display
    it via st.image().  Called inside a Streamlit expander by render_override_and_smith.

    Parameters
    ----------
    all_p    : dict — current (post Fine-tune) parameters in SI units.
    topology : "T" for ChengT, "pi" for ChengPi.
    fname    : file name tag used only as an image key for Streamlit.
    """
    import io
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        st.info("Install *pillow* to see the topology illustration.")
        return

    _PARASITIC_KEYS = ("Cpce", "Cpbe", "Cpbc", "Lb", "Le", "Lc")
    def _is_zero(key):
        v = all_p.get(key)
        try:
            return v is None or float(v) == 0.0
        except Exception:
            return False
    no_parasitics = all(_is_zero(k) for k in _PARASITIC_KEYS)

    if no_parasitics:
        tpl_name = "ChengT_template_noparasitics.png" if topology == "T" else "ChengPi_template_noparasitics.png"
    else:
        tpl_name = "ChengT_template.png" if topology == "T" else "ChengPi_template.png"
    tpl_path = _ILLUS_DIR / tpl_name
    if not tpl_path.exists():
        st.warning(f"Template not found: {tpl_path}")
        return

    overlay = _T_OVERLAY if topology == "T" else _PI_OVERLAY

    # Category colour sets (dark, readable on white)
    _C_PAD = (180,  2,   2)   # red
    _R_ACC = (175,  90,   5)   # orange
    _C_EXT = (  0, 130,  55)   # green
    _C_INT = ( 20,  95, 160)   # blue

    _PAD_KEYS = {"Cpbe", "Cpce", "Cpbc", "Lb", "Lc", "Le"}
    _ACCESSRES_KEYS = {"Rpb", "Rpc", "Rpe"}
    _EXT_KEYS = {"Cbex", "Cbcx"}

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

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), use_container_width=True)


def _override_ui(fname, tK, calc_vals, int_specs, label, ext_specs=_EXT_SPECS):
    """Render the override expander for one Cheng topology."""
    all_specs = PAD_SPECS + ext_specs + int_specs
    sync_pad_from_preov(fname, tK, calc_vals)

    _int_keys = [k for k, *_ in ext_specs + int_specs]
    _sync_hash_key = f"sim_synchash_{tK}_{fname}"
    _sync_hash = params_hash({k: str(round(float(calc_vals.get(k, 0.0)), 15))
                            for k in _int_keys})
    if st.session_state.get(_sync_hash_key) != _sync_hash:
        for key, _, scale, *_ in ext_specs + int_specs:
            st.session_state[f"sim_{tK}_{key}_{fname}"] = float(calc_vals.get(key, 0.0)) * scale
        st.session_state[_sync_hash_key] = _sync_hash

    with st.expander(f"✏️ Fine-tune {label} intrinsic/extrinsic parameters", expanded=False):
        if st.button(f"↩️ Reset {label} to interactive section values",
                     key=f"rst_sim_{tK}_{fname}"):
            for key, _, scale, *_ in all_specs:
                st.session_state[f"sim_{tK}_{key}_{fname}"] = float(calc_vals.get(key, 0.0)) * scale
            st.rerun()

        st.markdown("**Pad Parasitics** *(auto-synced from pre-extraction override)*")
        for row_start in range(0, len(PAD_SPECS), 3):
            row = PAD_SPECS[row_start:row_start+3]
            for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                                   key=f"sim_{tK}_{key}_{fname}", format=fmt, step=step)

        st.markdown("**Extrinsic Caps**")
        for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(ext_specs)), ext_specs):
            col_w.number_input(f"{lbl} ({unit})", key=f"sim_{tK}_{key}_{fname}", format=fmt, step=step)

        st.markdown("**Intrinsic**")
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


# ════════════════════════════════════════════════════════════════════════════════
# ChengT
# ════════════════════════════════════════════════════════════════════════════════
def _render_step2_plots(arrays, params, freq, fname, tK):
    """Plot Cbex and Cbcx vs frequency with modeled (median) value overlaid."""
    import matplotlib.pyplot as plt
    f_ghz = freq * 1e-9
    Cbex_arr = arrays.get("Cbex_arr")
    Cbcx_arr = arrays.get("Cbcx_arr")
    if Cbex_arr is None or Cbcx_arr is None:
        return

class ChengT(AbstractSSMModel):
    """
    Cheng (2022) T-topology.
    Two-step extraction:  Step 2 → Cbex, Cbcx  |  Step 3 → Rbi, Rbe, Cbe, …, α, τB, τC
    """
    NAME          = "T-topology (Cheng 2022)"
    SHORT         = "T"
    TOPOLOGY_CHAR = "T"
    # Cheng's batched intrinsic-Y kernels honour ``cache["_cdtype"]``,
    # so the tuning loop is allowed to run in complex64 ↓ ~5–10× speedup
    # on consumer GPUs (fp64 is gimped 1/64 vs fp32 on RTX 3050).  The
    # final top-K is reranked at fp64 to keep the published residuals
    # fully precise.
    SUPPORTS_FP32_SWEEP = True
    PARAM_GROUPS  = [
        {
            "label":      "Step 2 — Cbex  (from Im(Y₁₁+Y₁₂)/ω, low-freq range)",
            "params":     [("Cbex_arr", "Cbex", "Cbex", 1e15, "fF")],
            "depends_on": [],
            "cbex_sweep_group": True,
            "formulas": [
                ("markdown", "**[Eq. 13]:**"),
                ("latex", r"C_{bex}^T=\frac{\mathrm{Im}(Y_{11}+Y_{12})}{\omega}\big|_{\omega\to0}"),
            ],
        },
        {
            "label":      "Step 2 — Cbcx  (from Y_ex2 after peeling Cbex)",
            "params":     [("Cbcx_arr", "Cbcx", "Cbcx", 1e15, "fF")],
            "depends_on": ["Cbex"],
            "formulas": [
                ("markdown", "**[Eq. 22]:**"),
                ("latex", r"C_{bcx}=-\frac{\mathrm{Im}(Y_{ms})\mathrm{Re}(Y_L)-\mathrm{Re}(Y_{ms})\mathrm{Im}(Y_L)}{\omega[\mathrm{Re}(Y_{ms})\mathrm{Re}(Y_{tot})+\mathrm{Im}(Y_{tot})\mathrm{Im}(Y_{ms})]}"),
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
            "depends_on": ["Cbex", "Cbcx"],
            "use_first_params": {"Rbc", "Cbc", "alpha0"},
            "formulas": [
                ("markdown", "**[Eq. 16]:**"),
                ("latex", r"Z_{be}=Z_{12},\;Z_{bc}=Z_{22}-Z_{21},\;Z_{bi}=Z_{11}-Z_{12}"),
                ("markdown", "**[Eq. 29]:**"),
                ("latex", r"\alpha=\frac{Z_{12}-Z_{21}}{Z_{22}-Z_{21}},\;\alpha_0=|\alpha|_{\omega\to0}"),
            ],

        },
        {
            "label":      "τB  (depends on α₀)",
            "params":     [("tauB", "tauB", "τB", 1e12, "ps")],
            "depends_on": ["alpha0"],
            "formulas": [
                ("markdown", "**[Eq. 30]:**"),
                ("latex", r"\tau_B=\frac{\sqrt{U-1}}{\omega},\quad U=\left(\frac{\alpha_0}{|\alpha|}\right)^2"),
            ],

        },
        {
            "label":      "τC  (depends on τB)",
            "params":     [("tauC", "tauC", "τC", 1e12, "ps")],
            "depends_on": ["tauB"],
            "formulas": [
                ("markdown", "**[Eq. 31]:**"),
                ("latex", r"\tau_C=-\frac{\arctan\bigl[V(1-V^2)^{-1/2}\bigr]}{2\omega},\quad V=\frac{2\omega\tau_B}{U}"),
            ],

        },
    ]


    @classmethod
    def extract(cls, Y_ex1, freq, n_low, **kwargs):

        """
        Full extraction: Step 2 (Cbex_T, Cbcx) → Step 3 (intrinsic T params).
        See _step2_T and _step3_T for formula references.
        """
        res_ext, arr_ext = _step2_T(Y_ex1, freq, n_low)
        res_int, arr_int = _step3_T(arr_ext["Y_ex2"], freq, res_ext["Cbcx"], n_low)
        params = {**res_ext, **res_int}
        arrays = {**arr_ext, **arr_int, "_res_ext": res_ext, "_res_int": res_int}
        return params, arrays

    @classmethod
    def sweep_cbex(cls, Y_ex1, freq, cbex_SI_array, mask):
        """Sweep helper for the interactive Cbex-vs-Cbcx-stability search.

        See _sweep_cbex_stds_cheng for the full docstring.  Returns an array
        of std(Cbcx_arr[mask]) values, one per candidate Cbex; the caller
        (base_ui.render_interactive_param_groups) picks the argmin.
        """
        return _sweep_cbex_stds_cheng(Y_ex1, freq, cbex_SI_array, mask)

    @classmethod
    def simulate(cls, params, freq, z0=50.0):
        """
        Forward sim — inside-out:
          [Z_in]  →  add Cbcx/Cbex  →  [Y_ex]  →  add Z_ser  →  add Y_pad  →  S
        """
        def _Y_int(p, w):
            # T intrinsic Y matrix [Eq. 16 inverse]
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

    @classmethod
    def simulate_vec(cls, params, freq, z0=50.0, xp=None):
        """Vectorised simulate — no per-freq loop.  Pass xp=cupy for GPU."""
        if xp is None:
            xp = np
        return _sim_wrap_vec(_Y_int_T_vec, params, freq, z0, xp)

    @classmethod
    def simulate_batch(cls, params, freq, z0=50.0, xp=None, cache=None):
        """Batched simulate over (param_combo × freq).  Pass xp=cupy for GPU.

        params dict values may be scalars or (B,) arrays.
        Returns (B, N_freq, 2, 2) on the *xp* device (no host transfer).

        Optional ``cache`` dict (built once per sweep) carries pre-computed
        constant sub-networks (Y_pad, Z_ser, Y_extr, omega).
        """
        if xp is None:
            xp = np
        return _sim_wrap_batch(_Y_int_T_batch, params, freq, z0, xp, cache)

    @classmethod
    def reextract(cls, Y_ex1, freq, n_low, overrides, changed_group_idx, live_arrays):
        """
        Re-derive all downstream parameters when an upstream group is overridden.
          changed_group_idx=0 (Cbex changed)  → recompute Y_ex2, Cbcx, all Step 3
          changed_group_idx=2 (Cbcx changed)  → keep Y_ex2 from overrides["Cbex"],
                                                 use overrides["Cbcx"], re-run Step 3
          changed_group_idx=3 (α₀ changed)   → recompute τB and τC arrays
          changed_group_idx=4 (τB changed)   → recompute τC array only
        """
        omega = 2.0 * np.pi * freq

        # ── Always recompute Y_ex2 from current Cbex ───────────────
        Cbex_arr = np.imag(Y_ex1[:, 0, 0] + Y_ex1[:, 0, 1]) / omega
        Cbex = float(overrides.get("Cbex") or abs(safe_median(Cbex_arr, n_low)))

        Y_ex2 = Y_ex1.copy()
        for i, w in enumerate(omega):
            Y_ex2[i, 0, 0] -= 1j * w * Cbex

        # ── Recompute Cbcx_arr from new Y_ex2 ────────────────────────────────
        Yms  = Y_ex2[:, 0, 1] + Y_ex2[:, 1, 1]
        YL   = Y_ex2[:, 0, 0]*Y_ex2[:, 1, 1] - Y_ex2[:, 0, 1]*Y_ex2[:, 1, 0]
        Ytot = Y_ex2[:, 0, 0] + Y_ex2[:, 0, 1] + Y_ex2[:, 1, 0] + Y_ex2[:, 1, 1]
        num  = np.imag(Yms)*np.real(YL) - np.real(Yms)*np.imag(YL)
        den  = np.real(Yms)*np.real(Ytot) + np.imag(Ytot)*np.imag(Yms)
        with np.errstate(divide="ignore", invalid="ignore"):
            Cbcx_arr = -np.where(np.abs(den) > 1e-40, num / (omega * den), np.nan)
        n0c, n1c = len(freq) // 4, 3 * len(freq) // 4
        Cbcx_recomp = abs(safe_median(Cbcx_arr[n0c:n1c]))

        # Use user-overridden Cbcx only when the user explicitly changed Group 2
        Cbcx = float(overrides["Cbcx"]) if (changed_group_idx >= 2
                                             and "Cbcx" in overrides) else Cbcx_recomp

        # ── Re-run Step 3 ─────────────────────────────────────────────────────
        res_int, arr_int = _step3_T(Y_ex2, freq, Cbcx, n_low)

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

        new_params = {"Cbex": Cbex, "Cbcx": Cbcx, **res_int}
        new_arrays = {
            "Cbex_arr": Cbex_arr,
            "Cbcx_arr": Cbcx_arr,
            "Y_ex2":    Y_ex2,
            **arr_int,
        }
        return new_params, new_arrays

    @classmethod
    def render_results_table(cls, params):
        ri = params
        rows = [
            ("Cbex", f"{ri['Cbex']*1e15:.4f}", "fF"),   # Step 2 — extracted first
            ("Cbcx", f"{ri['Cbcx']*1e15:.4f}", "fF"),   # Step 2
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

        st.dataframe(pd.DataFrame(rows, columns=["Symbol","Value","Unit"]),
                     width="stretch", hide_index=True)

        with st.expander("📐 Full formula trace — T-topology (Cheng 2022)", expanded=False):
            st.markdown("**Dependency chain:** Y_ex1 → peel Cbex → Y_ex2 → peel Cbcx → Z_in → intrinsic")
            st.markdown("**Step 2** *(input: Y_ex1)*")
            st.latex(r"[Eq.13]\;C_{bex}^T=\frac{\mathrm{Im}(Y_{11}+Y_{12})}{\omega}\big|_{\omega\to0}")
            st.latex(r"[Eq.22]\;C_{bcx}=-\frac{\mathrm{Im}(Y_{ms})\mathrm{Re}(Y_L)"
                     r"-\mathrm{Re}(Y_{ms})\mathrm{Im}(Y_L)}{\omega"
                     r"[\mathrm{Re}(Y_{ms})\mathrm{Re}(Y_{tot})+\mathrm{Im}(Y_{tot})\mathrm{Im}(Y_{ms})]}")
            st.markdown("**Step 3** *(input: Y_ex2, Cbcx)*")
            st.latex(r"[Eq.16]\;Z_{be}=Z_{12},\;Z_{bc}=Z_{22}-Z_{21},\;Z_{bi}=Z_{11}-Z_{12}")
            st.latex(r"[Eq.29]\;\alpha=\frac{Z_{12}-Z_{21}}{Z_{22}-Z_{21}},\;"
                     r"\alpha_0=|\alpha||_{\omega\to0}")
            st.latex(r"[Eq.30]\;\tau_B=\frac{\sqrt{U-1}}{\omega}")
            st.latex(r"[Eq.31]\;\tau_C=-\frac{\arctan[V(1-V^2)^{-1/2}]}{2\omega}")
            st.markdown("**Forward simulation** *(inside → outside)*")
            st.latex(r"Z_{be}^{sim}=\frac{R_{be}}{1+j\omega R_{be}C_{be}},\;"
                     r"\alpha=\alpha_0 e^{-j\omega\tau_C}/(1+j\omega\tau_B)")
            st.latex(r"[Z_{in}^{sim}]=\begin{bmatrix}R_{bi}+Z_{be}&Z_{be}\\"
                     r"Z_{be}-\alpha Z_{bc}&(1-\alpha)Z_{bc}+Z_{be}\end{bmatrix}")
            st.latex(r"[Y_{ex}]=[Z_{in}]^{-1}+j\omega C_{bcx}\begin{pmatrix}1&-1\\-1&1\end{pmatrix}"
                     r"+j\omega C_{bex}\begin{pmatrix}1&0\\0&0\end{pmatrix}")
            st.latex(r"[Y_{tot}]=([Y_{ex}]^{-1}+[Z_{ser}])^{-1}\;,\quad "
                     r"S=(I-Z_0[Y_{tot}+Y_{pad}])(I+Z_0[Y_{tot}+Y_{pad}])^{-1}")

    @classmethod
    def render_override_and_smith(cls, fname, S_raw, freq, z0,
                                  para_eff, extract_result, **kwargs):
        params, arrays = extract_result
        calc_vals = {**para_eff, **params}
        all_p = _override_ui(fname, cls.SHORT, calc_vals, _INT_T_SPECS, cls.NAME,
                             ext_specs=_EXT_T_SPECS)

        # Cached simulation
        cache_key  = f"sim_result_{cls.SHORT}_{fname}"
        hash_key   = f"sim_phash_{cls.SHORT}_{fname}"
        cur_hash   = params_hash({k: str(v) for k, v in {**all_p, "__nf": len(freq)}.items()})
        if st.session_state.get(hash_key) != cur_hash:
            with st.spinner(f"Simulating {cls.NAME}…"):
                try:
                    S_sim = cls.simulate(all_p, freq, z0)
                except Exception as e:
                    st.error(f"Simulation error ({cls.NAME}): {e}")
                    S_sim = np.full((len(freq), 2, 2), np.nan + 0j)
            st.session_state[cache_key] = S_sim
            st.session_state[hash_key]  = cur_hash
        else:
            S_sim = st.session_state.get(cache_key)
            if S_sim is None or S_sim.shape[0] != len(freq):
                with st.spinner(f"Simulating {cls.NAME}…"):
                    try:
                        S_sim = cls.simulate(all_p, freq, z0)
                    except Exception as e:
                        st.error(f"Simulation error ({cls.NAME}): {e}")
                        S_sim = np.full((len(freq), 2, 2), np.nan + 0j)
                st.session_state[cache_key] = S_sim
                st.session_state[hash_key]  = cur_hash

        sc = smith_scale_controls(fname, cls.SHORT)
        render_smith_with_ftfmax(S_raw, S_sim, freq,
                                 model_name=cls.NAME, model_short=cls.SHORT,
                                 fname=fname, scales=sc)

        # Persist the *current* (post-override) param dict so the Complete
        # Parameter Summary can read live values instead of extraction-time ones.
        st.session_state[f"current_p_{cls.SHORT}_{fname}"] = dict(all_p)

        with st.expander("🖼️ Topology Illustration", expanded=False):
            _render_topology_illustration(all_p, "T", fname)

        with st.expander("📐 Plot Smith chart with matplotlib", expanded=False):
            from ..ssm_plots import render_matplotlib_smith
            render_matplotlib_smith(S_raw, S_sim, fname, cls.SHORT)

        render_tuning_expander(cls, all_p, S_raw, freq, z0,
                               PAD_SPECS + _EXT_T_SPECS + _INT_T_SPECS, fname, cls.SHORT)
        _render_step2_plots(arrays, params, freq, fname, cls.NAME)
        return S_sim



# ════════════════════════════════════════════════════════════════════════════════
# ChengPi
# ════════════════════════════════════════════════════════════════════════════════

class ChengPi(AbstractSSMModel):
    """
    Cheng (2022) π-topology  (Step 3 after Zhang et al. 2015).
    Same Step 2 as T but with π variant of Cbex formula.
    """
    NAME          = "π-topology (Cheng 2022)"
    SHORT         = "pi"
    TOPOLOGY_CHAR = "pi"
    SUPPORTS_FP32_SWEEP = True   # see ChengT for rationale
    PARAM_GROUPS = [
        {
            "label":      "Step 2 — Cbex  (from Im(B·C)/Im(B), low-freq range)",
            "params":     [("Cbex_arr", "Cbex", "Cbex", 1e15, "fF")],
            "depends_on": [],
            "cbex_sweep_group": True,
            "formulas": [
                ("md",    "**Step 2 — Cbex [Eqs. 26–28]**"),
                ("latex", r"B=Y_{12}+Y_{22},\quad C=Y_{11}+Y_{21}"),
                ("latex", r"C_{bex}^\pi=\frac{\mathrm{Re}(B)\mathrm{Re}(C)"
                        r"+\mathrm{Im}(B)\mathrm{Im}(C)}{\omega\,\mathrm{Im}(B)}"),
            ],
        },
        {
            "label":      "Step 2 — Cbcx  (from Y_ex2 after peeling Cbex)",
            "params":     [("Cbcx_arr", "Cbcx", "Cbcx", 1e15, "fF")],
            "depends_on": ["Cbex"],
            "formulas": [
                ("md",    "**Step 2 — Cbcx [Eq. 22]**"),
                ("latex", r"Y_{ms}=Y_{12}+Y_{22},\quad Y_L=\det(Y_{ex2}),"
                        r"\quad Y_{tot}=\textstyle\sum Y_{ij}"),
                ("latex", r"C_{bcx}=-\frac{\mathrm{Im}(Y_{ms})\mathrm{Re}(Y_L)"
                        r"-\mathrm{Re}(Y_{ms})\mathrm{Im}(Y_L)}{\omega\cdot\mathrm{denom}}"),
            ],
        },
        {
            "label":      "Step 3 — Intrinsic  (all from Z_in, depends on Cbex, Cbcx)",
            "params": [
                ("Rbi",  "Rbi",  "Rbi",  1.0,  "Ω"),
                ("Rbe",  "Rbe",  "Rbe",  1.0,  "Ω"),
                ("Cbe",  "Cbe",  "Cbe",  1e15, "fF"),
                ("Cbc",  "Cbc",  "Cbc",  1e15, "fF"),
                ("Gm0",  "Gm0",  "Gm0",  1e3,  "mS"),
                ("tau",  "tau",  "τ",    1e12, "ps"),
            ],
            "depends_on": ["Cbex", "Cbcx"],
            "use_first_params": {"Cbc"},
            "formulas": [
                ("md",    "**Step 3 — Intrinsic (Zhang 2015)**"),
                ("latex", r"Z_{bc}=Z_{22}-Z_{21}"),
                ("latex", r"g_m=\frac{Z_{12}-Z_{21}}{Z_{bc}\cdot Z_{12}}"
                        r"\;\Rightarrow\;G_{m0}=|g_m|,\;\tau=-\frac{\angle g_m}{\omega}"),
                ("latex", r"Y_{be}=\frac{Z_{22}-Z_{12}}{Z_{12}\cdot Z_{bc}}"
                        r"\;\Rightarrow\;R_{be}=\frac{1}{\mathrm{Re}(Y_{be})},"
                        r"\;C_{be}=\frac{\mathrm{Im}(Y_{be})}{\omega}"),
                ("latex", r"R_{bi}=\mathrm{Re}(Z_{11}-Z_{12}),"
                        r"\quad C_{bc}=\frac{\mathrm{Im}(Y_{bc})}{\omega}"),
            ],
        },
    ]


    @classmethod
    def extract(cls, Y_ex1, freq, n_low, **kwargs):

        res_ext, arr_ext = _step2_pi(Y_ex1, freq, n_low)
        res_int, arr_int = _step3_pi(arr_ext["Y_ex2"], freq, res_ext["Cbcx"], n_low)
        params = {**res_ext, **res_int}
        arrays = {**arr_ext, **arr_int, "_res_ext": res_ext, "_res_int": res_int}
        return params, arrays

    @classmethod
    def sweep_cbex(cls, Y_ex1, freq, cbex_SI_array, mask):
        """Same sweep as ChengT — Cbcx formula [Eq. 22] is identical for
        both topologies once Y_ex2 has been built."""
        return _sweep_cbex_stds_cheng(Y_ex1, freq, cbex_SI_array, mask)

    @classmethod
    def simulate(cls, params, freq, z0=50.0):
        def _Y_int(p, w):
            Ybe_v = 1.0/p["Rbe"] + 1j*w*p["Cbe"]
            Ybc_v = 1.0/p.get("Rbc", 1e9) + 1j*w*p["Cbc"]
            gm_v  = p["Gm0"] * np.exp(-1j*w*p["tau"])
            Y_core = np.array([[Ybe_v+Ybc_v, -Ybc_v],
                                [gm_v-Ybc_v,   Ybc_v]])
            try:
                Z_core = np.linalg.inv(Y_core) + np.array([[p["Rbi"], 0],[0, 0]])
                return np.linalg.inv(Z_core)
            except np.linalg.LinAlgError:
                return np.zeros((2, 2), dtype=complex)
        return _sim_wrap(_Y_int, params, freq, z0)

    @classmethod
    def simulate_vec(cls, params, freq, z0=50.0, xp=None):
        """Vectorised simulate — no per-freq loop.  Pass xp=cupy for GPU."""
        if xp is None:
            xp = np
        return _sim_wrap_vec(_Y_int_Pi_vec, params, freq, z0, xp)

    @classmethod
    def simulate_batch(cls, params, freq, z0=50.0, xp=None, cache=None):
        """Batched simulate over (param_combo × freq)."""
        if xp is None:
            xp = np
        return _sim_wrap_batch(_Y_int_Pi_batch, params, freq, z0, xp, cache)

    # @classmethod
    # def render_step_formulas(cls):
    #     with st.expander("Formulas"):
    #         c1,c2=st.columns(2)
    #         with c1:
    #             st.markdown("**π-topology — Step 2 [Eqs. 26–28]:**")
    #             st.latex(r"B=Y_{12}+Y_{22},\;C=Y_{11}+Y_{21}")
    #             st.latex(r"C_{bex}^\pi=\frac{\mathrm{Re}(B)\mathrm{Re}(C)"
    #                     r"+\mathrm{Im}(B)\mathrm{Im}(C)}{\omega\,\mathrm{Im}(B)}")
    #         with c2:
    #             st.markdown("**π-topology — Step 3 (Zhang et al. 2015):**")
    #             st.latex(r"g_m=\frac{Z_{12}-Z_{21}}{Z_{bc}Z_{12}}\;\Rightarrow\;"
    #                     r"G_{m0}=|g_m|,\;\tau=-\angle g_m/\omega")
    #             st.latex(r"Y_{be}=\frac{Z_{22}-Z_{12}}{Z_{12}Z_{bc}}\;\Rightarrow\;"
    #                 r"R_{be}=1/\mathrm{Re}(Y_{be}),\;C_{be}=\mathrm{Im}(Y_{be})/\omega")
    @classmethod
    def reextract(cls, Y_ex1, freq, n_low, overrides, changed_group_idx, live_arrays):
        """
        Re-derive all downstream parameters when an upstream group is overridden.
          changed_group_idx=0 (Cbex changed)  → recompute Y_ex2, Cbcx, all Step 3
          changed_group_idx=1 (Cbcx changed)  → keep Y_ex2 from overrides["Cbex"],
                                                 use overrides["Cbcx"], re-run Step 3
        """
        omega = 2.0 * np.pi * freq

        # ── Cbex_arr (π formula) — for reference only, Cbex is taken from overrides ─
        B = Y_ex1[:, 0, 1] + Y_ex1[:, 1, 1]
        C = Y_ex1[:, 0, 0] + Y_ex1[:, 1, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            Cbex_arr = np.where(
                np.abs(np.imag(B)) > 1e-40,
                (np.real(B)*np.real(C) + np.imag(B)*np.imag(C)) / (omega * np.imag(B)),
                np.nan)
        Cbex = float(overrides.get("Cbex") or safe_median(Cbex_arr, n_low))

        # ── Recompute Y_ex2 from current Cbex ────────────────────────────────
        Y_ex2 = Y_ex1.copy()
        for i, w in enumerate(omega):
            Y_ex2[i, 0, 0] -= 1j * w * Cbex

        # ── Recompute Cbcx_arr from new Y_ex2 (same formula as T topology) ───
        Yms  = Y_ex2[:, 0, 1] + Y_ex2[:, 1, 1]
        YL   = Y_ex2[:, 0, 0]*Y_ex2[:, 1, 1] - Y_ex2[:, 0, 1]*Y_ex2[:, 1, 0]
        Ytot = Y_ex2[:, 0, 0] + Y_ex2[:, 0, 1] + Y_ex2[:, 1, 0] + Y_ex2[:, 1, 1]
        num  = np.imag(Yms)*np.real(YL) - np.real(Yms)*np.imag(YL)
        den  = np.real(Yms)*np.real(Ytot) + np.imag(Ytot)*np.imag(Yms)
        with np.errstate(divide="ignore", invalid="ignore"):
            Cbcx_arr = -np.where(np.abs(den) > 1e-40, num / (omega * den), np.nan)
        n0c, n1c = len(freq) // 4, 3 * len(freq) // 4
        Cbcx_recomp = safe_median(Cbcx_arr[n0c:n1c])

        Cbcx = float(overrides["Cbcx"]) if (changed_group_idx >= 1
                                             and "Cbcx" in overrides) else Cbcx_recomp

        # ── Re-run Step 3 ─────────────────────────────────────────────────────
        res_int, arr_int = _step3_pi(Y_ex2, freq, Cbcx, n_low)

        new_params = {"Cbex": Cbex, "Cbcx": Cbcx, **res_int}
        new_arrays = {
            "Cbex_arr": Cbex_arr,
            "Cbcx_arr": Cbcx_arr,
            "Y_ex2":    Y_ex2,
            **arr_int,
        }
        return new_params, new_arrays

    @classmethod
    def render_results_table(cls, params):
        ri = params
        rows = [
            ("Cbex", f"{ri['Cbex']*1e15:.4f}", "fF"),   # Step 2 — extracted first
            ("Cbcx", f"{ri['Cbcx']*1e15:.4f}", "fF"),   # Step 2
            ("Rbi",  f"{ri['Rbi']:.4f}", "Ω"),           # Step 3
            ("Rbe",  f"{ri['Rbe']:.4f}" if ri['Rbe']<1000 else f"{ri['Rbe']*1e-3:.4f}k", "Ω"),
            ("Cbe",  f"{ri['Cbe']*1e15:.4f}" if ri['Cbe']<1e-12 else f"{ri['Cbe']*1e12:.4f}",
                     "fF" if ri['Cbe']<1e-12 else "pF"),
            ("Cbc",  f"{ri['Cbc']*1e15:.4f}", "fF"),
            ("Gm0",  f"{ri['Gm0']*1e3:.4f}",  "mS"),
            ("τ",    f"{ri['tau']*1e12:.4f}",  "ps"),
        ]

        st.dataframe(pd.DataFrame(rows, columns=["Symbol","Value","Unit"]),
                     width="stretch", hide_index=True)

    @classmethod
    def render_override_and_smith(cls, fname, S_raw, freq, z0,
                                  para_eff, extract_result, **kwargs):
        params, arrays = extract_result
        calc_vals = {**para_eff, **params}
        all_p = _override_ui(fname, cls.SHORT, calc_vals, _INT_PI_SPECS, cls.NAME,
                             ext_specs=_EXT_PI_SPECS)

        cache_key = f"sim_result_{cls.SHORT}_{fname}"
        hash_key  = f"sim_phash_{cls.SHORT}_{fname}"
        cur_hash  = params_hash({k: str(v) for k, v in {**all_p, "__nf": len(freq)}.items()})
        if st.session_state.get(hash_key) != cur_hash:
            with st.spinner(f"Simulating {cls.NAME}…"):
                try:
                    S_sim = cls.simulate(all_p, freq, z0)
                except Exception as e:
                    st.error(f"Simulation error ({cls.NAME}): {e}")
                    S_sim = np.full((len(freq), 2, 2), np.nan + 0j)
            st.session_state[cache_key] = S_sim
            st.session_state[hash_key]  = cur_hash
        else:
            S_sim = st.session_state.get(cache_key)
            if S_sim is None or S_sim.shape[0] != len(freq):
                with st.spinner(f"Simulating {cls.NAME}…"):
                    try:
                        S_sim = cls.simulate(all_p, freq, z0)
                    except Exception as e:
                        st.error(f"Simulation error ({cls.NAME}): {e}")
                        S_sim = np.full((len(freq), 2, 2), np.nan + 0j)
                st.session_state[cache_key] = S_sim
                st.session_state[hash_key]  = cur_hash

        sc = smith_scale_controls(fname, cls.SHORT)
        render_smith_with_ftfmax(S_raw, S_sim, freq,
                                 model_name=cls.NAME, model_short=cls.SHORT,
                                 fname=fname, scales=sc)

        # Persist the *current* (post-override) param dict so the Complete
        # Parameter Summary can read live values instead of extraction-time ones.
        st.session_state[f"current_p_{cls.SHORT}_{fname}"] = dict(all_p)

        with st.expander("🖼️ Topology Illustration", expanded=False):
            _render_topology_illustration(all_p, "pi", fname)

        with st.expander("📐 Plot Smith chart with matplotlib", expanded=False):
            from ..ssm_plots import render_matplotlib_smith
            render_matplotlib_smith(S_raw, S_sim, fname, cls.SHORT)

        render_tuning_expander(cls, all_p, S_raw, freq, z0,
                               PAD_SPECS + _EXT_PI_SPECS + _INT_PI_SPECS, fname, cls.SHORT)
        _render_step2_plots(arrays, params, freq, fname, cls.NAME)
        return S_sim

