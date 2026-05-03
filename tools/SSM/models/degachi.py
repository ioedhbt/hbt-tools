"""
models/degachi.py — Degachi & Ghannouchi (2008) augmented π HBT model.

Reference: Degachi & Ghannouchi, IEEE TED vol. 55 no. 4, 2008, Eqs. 1–26.

Extraction follows the paper structure exactly:

  GROUP 1 — Preparation
    Step 1  [Eqs. 3–5]   Z1, Z3, Z4
    Step 2  [Eq. 8]      Fbi vs ω² linear fit → A0, B0
    Step 3  [Eq. 12]     Tbi = √(B0/A0)

  GROUP 2 — Extraction of Small-Signal Model Parameters
    Part A  [Eqs. 13–14] Rbi/Rbc, Rbi·Cbc
    Part B  [Eqs. 19–21] F1 vs ω² linear fit → A, B, α → Tbe = √(B/A)
    Part C  [Eqs. 23–24] R, R·T from F2
    Part D  [Eq. 25]     Rbe, Rbi (2×2 linear solve)
    Part E               Rbc, Cbc, Cbe, Cbi (derived)
    Part F  [ref.8/Eq.26] Gm0, τ, Rcx, Ccx

PARAM_GROUPS:
  G0  z_plots_group    — Step 1: Z1/Z3/Z4 plots
  G1  fbi_fit_group    — Step 2: Fbi fit window + A0/B0 display
  G2  standard         — Step 3: Tbi (user-overridable)
  G3  standard         — Part A: Rbi/Rbc, Rbi·Cbc
  G4  f1_fit_group     — Part B: F1 fit window + A/B/α display + Tbe
  G5  standard         — Part C: R, R·T
  G6  standard         — Part D: Rbe, Rbi
  G7  standard         — Part E: Rbc, Cbc, Cbe, Cbi
  G8  standard         — Part F: Gm0, τ, Rcx, Ccx

Special flags handled by base_ui.render_interactive_param_groups:
  "z_plots_group"  → show Re/Im of Z1/Z3/Z4 vs frequency, no params
  "fbi_fit_group"  → single-slider fit window, Fbi vs ω² plot, metrics, no params
  "f1_fit_group"   → single-slider fit window, F1 vs ω² plot, metrics, then Tbe input
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ..helpers          import (y_to_z, y_to_s_single, y_to_s_vec,
                                 safe_median, params_hash,
                                 extended_smith_grid,
                                 build_Y_pad, build_Z_ser,
                                 build_Y_pad_vec, build_Z_ser_vec,
                                 build_Y_pad_batch, build_Z_ser_batch)
from .base_ui           import (smith_scale_controls,
                                 sync_pad_from_preov, PAD_SPECS,
                                 render_tuning_expander, render_smith_with_ftfmax)
from . import AbstractSSMModel


# ════════════════════════════════════════════════════════════════════════════════
# Stateless helpers
# ════════════════════════════════════════════════════════════════════════════════

def _h_Tbi(Z1, Z3, omega, omega2, n_fit):
    """[Eqs. 8,12]  Fbi = ω/Im(Z1/Z3) = A0 + ω²B0  →  Tbi = √(B0/A0)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        Fbi = omega / np.imag(Z1 / Z3)
    mask = np.isfinite(Fbi[:n_fit]) & (np.abs(Fbi[:n_fit]) < 1e15) & (Fbi[:n_fit] > 0)
    if mask.sum() >= 3:
        try:
            c = np.polyfit(omega2[:n_fit][mask], Fbi[:n_fit][mask], 1)
            B0, A0 = float(c[0]), float(c[1])
        except Exception:
            A0, B0 = 1.0, 0.0
    else:
        A0, B0 = 1.0, 0.0
    Tbi = float(np.sqrt(max(B0 / A0, 0.0))) if A0 > 1e-30 else 0.0
    return Tbi, Fbi, A0, B0


def _h_Tbe(Z1, omega, omega2, Tbi, n_fit):
    """[Eqs. 15,19,20]  F1 = ω/Im(Z1(1+jωTbi)) = A + ω²B  →  Tbe = √(B/A)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        F1 = omega / np.imag(Z1 * (1.0 + 1j * omega * Tbi))
    mask = np.isfinite(F1[:n_fit]) & (np.abs(F1[:n_fit]) < 1e15) & (F1[:n_fit] > 0)
    if mask.sum() >= 3:
        try:
            c = np.polyfit(omega2[:n_fit][mask], F1[:n_fit][mask], 1)
            B1, A1 = float(c[0]), float(c[1])
        except Exception:
            A1, B1 = 1.0, 0.0
    else:
        A1, B1 = 1.0, 0.0
    Tbe = float(np.sqrt(max(B1 / A1, 0.0))) if A1 > 1e-30 else 0.0
    return Tbe, F1, A1, B1


def _h_ratios(Z1, Z3, omega, Tbi):
    """[Eqs. 13–14]  Re[(Z1/Z3)(1+jωTbi)] = Rbi/Rbc,  Im[·]/ω = Rbi·Cbc"""
    c13 = (Z1 / Z3) * (1.0 + 1j * omega * Tbi)
    ror_arr = np.real(c13)
    with np.errstate(divide="ignore", invalid="ignore"):
        rbc_arr = np.where(omega > 0, np.imag(c13) / omega, np.nan)
    return ror_arr, rbc_arr


def _h_R_RT(Z1, omega, Tbi, Tbe):
    """[Eqs. 23–24]  F2 = Z1(1+jωTbe)(1+jωTbi),  R=Re(F2), R·T=Im(F2)/ω"""
    F2 = Z1 * (1.0 + 1j * omega * Tbe) * (1.0 + 1j * omega * Tbi)
    R_arr = np.real(F2)
    with np.errstate(divide="ignore", invalid="ignore"):
        RT_arr = np.where(omega > 0, np.imag(F2) / omega, np.nan)
    return R_arr, RT_arr, F2


def _h_solve_pf(ror_arr, rbc_arr, R_arr, RT_arr, Tbi, Tbe):
    """[Eq. 25] per-frequency 2×2 solve for Rbe, Rbi."""
    N    = len(R_arr)
    mats = np.stack([
        np.column_stack([1.0 + ror_arr,       np.ones(N)]),
        np.column_stack([Tbi + rbc_arr, np.full(N, Tbe)]),
    ], axis=1)
    rhs = np.column_stack([R_arr, RT_arr])
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.linalg.solve(mats, rhs)
        Rbe_a = np.where((s[:, 0] > 0) & np.isfinite(s[:, 0]), s[:, 0], np.nan)
        Rbi_a = np.where((s[:, 1] > 0) & np.isfinite(s[:, 1]), s[:, 1], np.nan)
    except Exception:
        Rbe_a = Rbi_a = np.full(N, np.nan)
    return Rbe_a, Rbi_a


def _h_solve_sc(ror, rbc, R, RT, Tbi, Tbe, Z1, Z3, n_low):
    """[Eq. 25] scalar solve with heuristic fallback."""
    try:
        sol = np.linalg.solve(
            np.array([[1.0 + ror, 1.0], [Tbi + rbc, Tbe]]),
            np.array([R, RT]))
        Rbe, Rbi = float(sol[0]), float(sol[1])
        if Rbe <= 0 or Rbi <= 0:
            raise ValueError
    except Exception:
        Rbi = safe_median(np.real(Z1 - Z3), n_low)
        Rbe = max(R - Rbi, 1.0)
    return Rbe, Rbi


def _h_derived_pf(Rbe_a, Rbi_a, ror_a, rbc_a, Tbi, Tbe):
    """Rbc, Cbc, Cbe, Cbi per-frequency arrays from Rbe/Rbi arrays."""
    with np.errstate(divide="ignore", invalid="ignore"):
        Rbc_a = np.where(np.abs(ror_a) > 1e-30, Rbi_a / ror_a, np.nan)
        Cbc_a = np.where(np.abs(Rbi_a) > 1e-30, rbc_a / Rbi_a, np.nan)
        Cbe_a = np.where(np.abs(Rbe_a) > 1e-30, Tbe   / Rbe_a, np.nan)
        Cbi_a = np.where(np.abs(Rbi_a) > 1e-30, Tbi   / Rbi_a, np.nan)
    return Rbc_a, Cbc_a, Cbe_a, Cbi_a


def _h_derived_sc(Rbi, Rbe, ror, rbc, Tbi, Tbe):
    """Rbc, Cbc, Cbe, Cbi scalars from Rbe/Rbi scalars."""
    Rbc = float(Rbi / ror) if abs(ror) > 1e-30 else 1e6
    Cbc = float(rbc / Rbi) if abs(Rbi) > 1e-30 else 0.0
    Cbe = float(Tbe / Rbe) if abs(Rbe) > 1e-30 else 0.0
    Cbi = float(Tbi / Rbi) if abs(Rbi) > 1e-30 else 0.0
    return Rbc, Cbc, Cbe, Cbi


def _h_Rcx(Z4, Rbc, Cbc, omega):
    """[Eq. 26]  1/Rcx = Re(1/Z4 − 1/Z2),  Z2 = Rbc/(1+jωRbcCbc)"""
    Z2 = Rbc / (1.0 + 1j * omega * Rbc * Cbc)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.real(1.0 / Z4 - 1.0 / Z2)
        arr = np.where(inv > 1e-12, 1.0 / inv, np.nan)
    ok  = np.isfinite(arr) & (arr < 1e9)
    Rcx = max(float(safe_median(arr[ok])), 1.0) if ok.sum() >= 2 else 1e6
    return Rcx, arr


def _h_Gm0_tau(Y_ex1, freq):
    """[ref.8]  Gm0 = Re(Y22)|f→0,  τ = −(1/2π)·d∠Y21/df"""
    Gm0_a = np.real(Y_ex1[:, 1, 1])
    Gm0   = max(float(Gm0_a[0]), 0.0)
    phi   = np.unwrap(np.angle(Y_ex1[:, 1, 0]))
    tau   = max(-float(np.polyfit(freq, phi, 1)[0]) / (2.0 * np.pi), 0.0)
    tau_a = np.maximum(-np.gradient(phi, freq) / (2.0 * np.pi), 0.0)
    return Gm0, tau, Gm0_a, tau_a, phi


def _h_Ccx(Y_ex1, omega, n_hi):
    """[ref.8]  Ccx ≈ −Im(Y12)/ω at high frequency."""
    with np.errstate(divide="ignore", invalid="ignore"):
        arr = -np.imag(Y_ex1[:, 0, 1]) / omega
    return max(safe_median(arr[n_hi:]), 0.0), arr


# ════════════════════════════════════════════════════════════════════════════════
# Extraction
# ════════════════════════════════════════════════════════════════════════════════

def _extract(Y_ex1, freq, n_low, n_fit=None, n_fit_f1=None):
    omega  = 2.0 * np.pi * freq
    omega2 = omega ** 2
    N      = len(freq)
    if n_fit is None:
        n_fit = max(4, 2 * N // 3)
    if n_fit_f1 is None:
        n_fit_f1 = n_fit
    n_hi = max(N // 2, n_low + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        Z1 = 1.0 / (Y_ex1[:, 0, 0] + Y_ex1[:, 0, 1])
        Z3 = (Y_ex1[:, 1, 0] + Y_ex1[:, 0, 0]) / \
             ((Y_ex1[:, 0, 0] + Y_ex1[:, 0, 1]) * (Y_ex1[:, 1, 1] + Y_ex1[:, 0, 1]))
        Z4 = -1.0 / Y_ex1[:, 0, 1]

    Tbi, Fbi, A0, B0            = _h_Tbi(Z1, Z3, omega, omega2, n_fit)
    Tbe, F1,  A1, B1            = _h_Tbe(Z1, omega, omega2, Tbi, n_fit_f1)
    ror_a, rbc_a                = _h_ratios(Z1, Z3, omega, Tbi)
    ror, rbc                    = safe_median(ror_a, n_low), safe_median(rbc_a, n_low)
    R_a, RT_a, F2               = _h_R_RT(Z1, omega, Tbi, Tbe)
    R, RT                       = safe_median(R_a, n_low), safe_median(RT_a, n_low)
    Rbe_a, Rbi_a                = _h_solve_pf(ror_a, rbc_a, R_a, RT_a, Tbi, Tbe)
    Rbe, Rbi                    = _h_solve_sc(ror, rbc, R, RT, Tbi, Tbe, Z1, Z3, n_low)
    Rbc_a, Cbc_a, Cbe_a, Cbi_a = _h_derived_pf(Rbe_a, Rbi_a, ror_a, rbc_a, Tbi, Tbe)
    Rbc, Cbc, Cbe, Cbi          = _h_derived_sc(Rbi, Rbe, ror, rbc, Tbi, Tbe)
    Gm0, tau, Gm0_a, tau_a, phi = _h_Gm0_tau(Y_ex1, freq)
    Rcx, Rcx_a                  = _h_Rcx(Z4, Rbc, Cbc, omega)
    Ccx, Ccx_a                  = _h_Ccx(Y_ex1, omega, n_hi)

    params = dict(
        Rbi=Rbi, Cbi=Cbi, Rbe=Rbe, Cbe=Cbe,
        Rbc=Rbc, Cbc=Cbc, Rcx=Rcx, Ccx=Ccx, Gm0=Gm0, tau=tau,
        Tbi=Tbi, Tbe=Tbe, A0=A0, B0=B0, A1=A1, B1=B1,
        RbiOverRbc=ror, RbiCbc=rbc, R=R, RT=RT,
    )
    arrays = dict(
        Z1=Z1, Z3=Z3, Z4=Z4, F2=F2, Fbi=Fbi, F1=F1, omega2=omega2,
        phase_Y21=phi,
        Tbi_arr=np.full(N, Tbi),
        Tbe_arr=np.full(N, Tbe),
        RbiOverRbc_arr=ror_a, RbiCbc_arr=rbc_a,
        R_arr=R_a, RT_arr=RT_a,
        Rbe_arr=Rbe_a, Rbi_arr=Rbi_a,
        Rbc_arr=Rbc_a, Cbc_arr=Cbc_a, Cbe_arr=Cbe_a, Cbi_arr=Cbi_a,
        Gm0_a=Gm0_a, tau_a=tau_a,
        Rcx_arr=Rcx_a, Ccx_arr=Ccx_a,
    )
    return params, arrays


# ════════════════════════════════════════════════════════════════════════════════
# Forward simulator
# ════════════════════════════════════════════════════════════════════════════════

def _simulate(p, freq, z0=50.0):
    r"""
    5-layer inside-out build-up.
    $$Z_{xy}=R_{xy}/(1+j\omega R_{xy}C_{xy})$$  [Eq.1]
    $$[Y_{core}]=[[Y_{be}+Y_{bc},-Y_{bc}],[g_m-Y_{bc},Y_{bc}]]$$
    $$[Z_{core}]=[Y_{core}]^{-1}+[[Z_{bi},0],[0,0]]$$
    $$[Y_{int}]=[Z_{core}]^{-1}+(1/Z_{cx})[[1,-1],[-1,1]]$$
    $$[Y_{tot}]=([Y_{int}]^{-1}+[Z_{ser}])^{-1},\quad
      S=(I-Z_0 Y)(I+Z_0 Y)^{-1},\;Y=Y_{tot}+Y_{pad}$$
    """
    omega = 2.0 * np.pi * freq
    S = np.zeros((len(freq), 2, 2), dtype=complex)
    for i, w in enumerate(omega):
        Rbi, Cbi = p["Rbi"], p["Cbi"]
        Rbe, Cbe = p["Rbe"], p["Cbe"]
        Rbc, Cbc = p["Rbc"], p["Cbc"]
        Rcx, Ccx = p["Rcx"], p["Ccx"]
        Gm0, tau = p["Gm0"], p["tau"]

        Zbi_v = Rbi / (1.0 + 1j*w*Rbi*Cbi) if Cbi > 1e-40 else complex(Rbi)
        Zbe_v = Rbe / (1.0 + 1j*w*Rbe*Cbe)
        Zbc_v = Rbc / (1.0 + 1j*w*Rbc*Cbc)
        Zcx_v = (1.0/(1j*w*Ccx) if (Rcx > 1e4 and Ccx > 1e-40)
                 else Rcx/(1.0+1j*w*Rcx*Ccx) if Ccx > 1e-40
                 else complex(1e9))

        Ybe  = 1.0/Zbe_v; Ybc = 1.0/Zbc_v
        gm_v = Gm0 * np.exp(-1j*w*tau)

        Y_core = np.array([[Ybe+Ybc, -Ybc], [gm_v-Ybc, Ybc]])
        try:
            Z_core = np.linalg.inv(Y_core) + np.array([[Zbi_v, 0.0], [0.0, 0.0]])
            Y_int  = np.linalg.inv(Z_core) + (1.0/Zcx_v)*np.array([[1,-1],[-1,1]])
        except np.linalg.LinAlgError:
            Y_int = np.zeros((2, 2), dtype=complex)

        Z_ser = build_Z_ser(p, w)
        try:    Y_tot = np.linalg.inv(np.linalg.inv(Y_int) + Z_ser)
        except: Y_tot = np.zeros((2,2), dtype=complex)
        S[i] = y_to_s_single(Y_tot + build_Y_pad(p, w), z0)
    return S


def _simulate_vec(p, freq, z0=50.0, xp=None):
    """Vectorised Degachi simulate — no per-freq loop.  Pass xp=cupy for GPU."""
    if xp is None:
        xp = np
    omega = xp.asarray(2.0 * np.pi * freq, dtype=np.float64)
    N = len(freq)

    Rbi, Cbi = p["Rbi"], p["Cbi"]
    Rbe, Cbe = p["Rbe"], p["Cbe"]
    Rbc, Cbc = p["Rbc"], p["Cbc"]
    Rcx, Ccx = p["Rcx"], p["Ccx"]
    Gm0, tau = p["Gm0"], p["tau"]

    # Impedances (N,)
    Zbi = Rbi / (1.0 + 1j * omega * Rbi * Cbi) if Cbi > 1e-40 else xp.full(N, complex(Rbi))
    Zbe = Rbe / (1.0 + 1j * omega * Rbe * Cbe)
    Zbc = Rbc / (1.0 + 1j * omega * Rbc * Cbc)

    # Zcx — conditional on Rcx/Ccx values
    if Rcx > 1e4 and Ccx > 1e-40:
        Zcx = 1.0 / (1j * omega * Ccx)
    elif Ccx > 1e-40:
        Zcx = Rcx / (1.0 + 1j * omega * Rcx * Ccx)
    else:
        Zcx = xp.full(N, complex(1e9))

    Ybe = 1.0 / Zbe
    Ybc = 1.0 / Zbc
    gm  = Gm0 * xp.exp(-1j * omega * tau)

    # Y_core (N, 2, 2)
    Y_core = xp.zeros((N, 2, 2), dtype=complex)
    Y_core[:, 0, 0] = Ybe + Ybc
    Y_core[:, 0, 1] = -Ybc
    Y_core[:, 1, 0] = gm - Ybc
    Y_core[:, 1, 1] = Ybc

    # Z_core = inv(Y_core) + [[Zbi, 0], [0, 0]]
    Z_core = xp.linalg.inv(Y_core)
    Z_core[:, 0, 0] += Zbi

    # Y_int = inv(Z_core) + (1/Zcx) * [[1,-1],[-1,1]]
    Y_int = xp.linalg.inv(Z_core)
    inv_Zcx = 1.0 / Zcx   # (N,)
    Y_int[:, 0, 0] += inv_Zcx
    Y_int[:, 0, 1] -= inv_Zcx
    Y_int[:, 1, 0] -= inv_Zcx
    Y_int[:, 1, 1] += inv_Zcx

    # Series leads + pad
    Z_ser = build_Z_ser_vec(p, omega, xp)
    Y_tot = xp.linalg.inv(xp.linalg.inv(Y_int) + Z_ser)
    Y_pad = build_Y_pad_vec(p, omega, xp)

    S = y_to_s_vec(Y_tot + Y_pad, z0, xp)
    if xp is not np:
        S = xp.asnumpy(S)
    return S


# ── Batched (B, N, 2, 2) simulate for parameter-sweep tuning ─────────────────

def _b1(p, key, default, xp):
    """Fetch p[key] (or default) and reshape (B,) → (B,1).  Scalars stay scalar."""
    v = p.get(key, default)
    a = xp.asarray(v)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _detect_B(p, xp):
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


def _simulate_batch(p, freq, z0=50.0, xp=None, cache=None):
    """Batched Degachi simulate over (param_combo × freq).

    Returns S of shape (B, N, 2, 2) on the *xp* device.
    Param values may be scalars or (B,) arrays.

    Hand-inlined 2×2 algebra throughout — every matrix inverse uses the
    analytic adjugate formula instead of cuSOLVER's batched LU, which
    has launch overhead orders of magnitude larger than the actual
    arithmetic for 2×2 matrices.

    Optional ``cache`` dict (built once per sweep) carries pre-computed
    constant sub-networks: ``"omega"``, ``"Y_pad"``, ``"Z_ser"``.
    """
    if xp is None:
        xp = np
    N = len(freq)
    B = _detect_B(p, xp)
    if cache is not None and "omega" in cache:
        omega = cache["omega"]
    else:
        omega = xp.asarray(2.0 * np.pi * freq, dtype=np.float64).reshape(1, N)  # (1, N)

    Rbi = _b1(p, "Rbi", 0.0, xp)
    Cbi = _b1(p, "Cbi", 0.0, xp)
    Rbe = _b1(p, "Rbe", 1.0, xp)
    Cbe = _b1(p, "Cbe", 0.0, xp)
    Rbc = _b1(p, "Rbc", 1.0, xp)
    Cbc = _b1(p, "Cbc", 0.0, xp)
    Rcx = _b1(p, "Rcx", 0.0, xp)
    Ccx = _b1(p, "Ccx", 0.0, xp)
    Gm0 = _b1(p, "Gm0", 0.0, xp)
    tau = _b1(p, "tau", 0.0, xp)

    # Zbi: Rbi/(1+jωRbiCbi) — when Cbi tiny, falls back to Rbi (handled by xp.where)
    Cbi_safe = xp.where(xp.abs(Cbi) > 1e-40, Cbi, 1e-40)
    Zbi_full = Rbi / (1.0 + 1j * omega * Rbi * Cbi_safe)
    Zbi      = xp.where(xp.abs(Cbi) > 1e-40, Zbi_full, Rbi + 0j*omega)

    Zbe = Rbe / (1.0 + 1j * omega * Rbe * Cbe)
    Zbc = Rbc / (1.0 + 1j * omega * Rbc * Cbc)

    # Zcx — three branches:
    #   Rcx > 1e4 and Ccx > 1e-40 → 1/(jωCcx)
    #   Ccx > 1e-40              → Rcx/(1+jωRcxCcx)
    #   else                     → 1e9 (open)
    Ccx_safe = xp.where(xp.abs(Ccx) > 1e-40, Ccx, 1e-40)
    Zcx_a = 1.0 / (1j * omega * Ccx_safe)
    Zcx_b = Rcx / (1.0 + 1j * omega * Rcx * Ccx_safe)
    Zcx_open = xp.full_like(Zcx_a, complex(1e9))
    cond_a = (xp.abs(Rcx) > 1e4) & (xp.abs(Ccx) > 1e-40)
    cond_b = (xp.abs(Ccx) > 1e-40) & ~cond_a
    Zcx = xp.where(cond_a, Zcx_a, xp.where(cond_b, Zcx_b, Zcx_open))

    Ybe = 1.0 / Zbe
    Ybc = 1.0 / Zbc
    gm  = Gm0 * xp.exp(-1j * omega * tau)

    # Y_core 2×2 → 4 planes
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

    # Add Zbi to Z_core[0,0]
    zc00 = zc00 + Zbi

    # Y_int = inv(Z_core)
    inv_det_i = 1.0 / (zc00 * zc11 - zc01 * zc10)
    yi00 =  zc11 * inv_det_i
    yi01 = -zc01 * inv_det_i
    yi10 = -zc10 * inv_det_i
    yi11 =  zc00 * inv_det_i

    # Add Cbcx-style network: inv_Zcx in/out shunt across the (b,c) port
    inv_Zcx = 1.0 / Zcx
    yi00 = yi00 + inv_Zcx
    yi01 = yi01 - inv_Zcx
    yi10 = yi10 - inv_Zcx
    yi11 = yi11 + inv_Zcx

    # Z_int = inv(Y_int)
    inv_det_yi = 1.0 / (yi00 * yi11 - yi01 * yi10)
    zi00 =  yi11 * inv_det_yi
    zi01 = -yi01 * inv_det_yi
    zi10 = -yi10 * inv_det_yi
    zi11 =  yi00 * inv_det_yi

    # Add Z_ser (planes) — cache-aware
    if cache is not None and "Z_ser" in cache:
        zs00, zs01, zs10, zs11 = cache["Z_ser"]
    else:
        zs00, zs01, zs10, zs11 = build_Z_ser_batch(p, omega, B, N, xp)
    zt00 = zi00 + zs00
    zt01 = zi01 + zs01
    zt10 = zi10 + zs10
    zt11 = zi11 + zs11

    # Y_tot = inv(Z_tot)
    inv_det_t = 1.0 / (zt00 * zt11 - zt01 * zt10)
    yt00 =  zt11 * inv_det_t
    yt01 = -zt01 * inv_det_t
    yt10 = -zt10 * inv_det_t
    yt11 =  zt00 * inv_det_t

    # Y_total = Y_tot + Y_pad — cache-aware
    if cache is not None and "Y_pad" in cache:
        yp00, yp01, yp10, yp11 = cache["Y_pad"]
    else:
        yp00, yp01, yp10, yp11 = build_Y_pad_batch(p, omega, B, N, xp)
    ya00 = yt00 + yp00
    ya01 = yt01 + yp01
    ya10 = yt10 + yp10
    ya11 = yt11 + yp11

    # Y → S, fully inlined
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

    return xp.stack(
        [xp.stack([s00, s01], axis=-1),
         xp.stack([s10, s11], axis=-1)],
        axis=-2,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Smith chart fine-tune specs
# ════════════════════════════════════════════════════════════════════════════════

_INT_D_SPECS = [
    ("Rbi","Rbi", 1.0, "Ω",  "%.4f", 0.1),
    ("Cbi","Cbi", 1e15,"fF", "%.4f", 0.1),
    ("Rbe","Rbe", 1.0, "Ω",  "%.3f", 1.0),
    ("Cbe","Cbe", 1e15,"fF", "%.4f", 0.1),
    ("Rbc","Rbc", 1e-3,"kΩ", "%.4f", 0.01),
    ("Cbc","Cbc", 1e15,"fF", "%.4f", 0.01),
    ("Rcx","Rcx", 1e-3,"kΩ", "%.2f", 10.0),
    ("Ccx","Ccx", 1e15,"fF", "%.4f", 0.1),
    ("Gm0","Gm0", 1e3, "mS", "%.4f", 0.01),
    ("tau","τ",   1e12,"ps", "%.4f", 0.01),
]


# ════════════════════════════════════════════════════════════════════════════════
# Model class
# ════════════════════════════════════════════════════════════════════════════════

class Degachi(AbstractSSMModel):
    NAME          = "Degachi (2008) augmented π"
    SHORT         = "D"
    TOPOLOGY_CHAR = "D"

    # ── PARAM_GROUPS ─────────────────────────────────────────────────────────
    # Standard tuple format: (arr_key, param_key, label, scale, unit)
    # Special flags:
    #   "z_plots_group"  → base_ui renders Z1/Z3/Z4 Re/Im plots, no params
    #   "fbi_fit_group"  → base_ui renders Fbi fit, single fit-window slider, no params
    #   "f1_fit_group"   → base_ui renders F1 fit, single fit-window slider, then Tbe input
    #
    # Cascade: G0,G1(fit) → G2(Tbi) → G3(ror,rbc) → G4(Tbe) → G5(R,RT)
    #          → G6(Rbe,Rbi) → G7(Rbc,Cbc,Cbe,Cbi) → G8(Gm0,τ,Rcx,Ccx)
    PARAM_GROUPS = [

        # ── Group 0: Preparation — Step 1: Z1, Z3, Z4 ────────────────────────
        {
            "label":         "Preparation — Step 1: Z₁, Z₃, Z₄  [Eqs. 3–5]",
            "params":        [],
            "depends_on":    [],
            "z_plots_group": True,
            "formulas": [
                ("latex", r"Z_1=\frac{1}{Y_{11}+Y_{12}}"),
                ("latex", r"Z_3=\frac{Y_{21}+Y_{11}}{(Y_{11}+Y_{12})(Y_{22}+Y_{12})}"),
                ("latex", r"Z_4=-\frac{1}{Y_{12}}"),
            ],
        },

        # ── Group 1: Preparation — Step 2: Fbi fit ───────────────────────────
        {
            "label":          "Preparation — Step 2: Fbi vs ω²  [Eq. 8]",
            "params":         [],
            "depends_on":     [],
            "fbi_fit_group":  True,
            "formulas": [
                ("latex", r"F_{bi}\triangleq\frac{\omega}{\mathrm{Im}(Z_1/Z_3)}=A_0+\omega^2 B_0"),
            ],
        },

        # ── Group 2: Preparation — Step 3: Tbi ───────────────────────────────
        {
            "label":      "Preparation — Step 3: Tbi  [Eq. 12]",
            "params":     [("Tbi_arr", "Tbi", "Tbi", 1e12, "ps")],
            "depends_on": [],
            "formulas": [
                ("latex", r"T_{bi}=\sqrt{\frac{B_0}{A_0}}"),
            ],
        },

        # ── Group 3: Part A — Rbi/Rbc and Rbi·Cbc ────────────────────────────
        {
            "label":      "Part A — Rbi/Rbc and Rbi·Cbc  [Eqs. 13–14]",
            "params":     [("RbiOverRbc_arr", "RbiOverRbc", "Rbi/Rbc", 1.0,  ""),
                           ("RbiCbc_arr",     "RbiCbc",     "Rbi·Cbc", 1e12, "ps")],
            "depends_on": ["Tbi"],
            "formulas": [
                ("latex", r"\mathrm{Re}\!\left[\frac{Z_1}{Z_3}(1+j\omega T_{bi})\right]"
                          r"=\frac{R_{bi}}{R_{bc}}"),
                ("latex", r"\frac{\mathrm{Im}\!\left[\frac{Z_1}{Z_3}(1+j\omega T_{bi})\right]}{\omega}"
                          r"=R_{bi}C_{bc}"),
            ],
        },

        # ── Group 4: Part B — F1 fit → Tbe ───────────────────────────────────
        {
            "label":        "Part B — Determination of Tbe = Rbe·Cbe  [Eqs. 19–21]",
            "params":       [("Tbe_arr", "Tbe", "Tbe", 1e12, "ps")],
            "depends_on":   ["Tbi"],
            "f1_fit_group": True,
            "formulas": [
                ("latex", r"F_1\triangleq\frac{\omega}{\mathrm{Im}[Z_1(1+j\omega T_{bi})]}"
                          r"=A+\omega^2 B"),
                ("latex", r"\alpha=R(T-T_{be})=\frac{1}{A},\quad "
                          r"T_{be}=\sqrt{\frac{B}{A}}"),
            ],
        },

        # ── Group 5: Part C — R and R·T ──────────────────────────────────────
        {
            "label":      "Part C — R and R·T  [Eqs. 23–24]",
            "params":     [("R_arr",  "R",  "R",   1.0,  "Ω"),
                           ("RT_arr", "RT", "R·T", 1e12, "Ω·ps")],
            "depends_on": ["Tbi", "Tbe"],
            "formulas": [
                ("latex", r"F_2\triangleq Z_1(1+j\omega T_{be})(1+j\omega T_{bi})"
                          r"=R(1+j\omega T)"),
                ("latex", r"R=\mathrm{Re}(F_2),\quad "
                          r"R\cdot T=\frac{\mathrm{Im}(F_2)}{\omega}"),
            ],
        },

        # ── Group 6: Part D — Rbe and Rbi ────────────────────────────────────
        {
            "label":      "Part D — Rbe and Rbi  [Eq. 25]",
            "params":     [("Rbe_arr", "Rbe", "Rbe", 1.0, "Ω"),
                           ("Rbi_arr", "Rbi", "Rbi", 1.0, "Ω")],
            "depends_on": ["RbiOverRbc", "RbiCbc", "R", "RT"],
            "formulas": [
                ("latex", r"\begin{pmatrix}R_{be}\\R_{bi}\end{pmatrix}="
                          r"\begin{pmatrix}1+\dfrac{R_{bi}}{R_{bc}}&1\\"
                          r"T_{bi}+R_{bi}C_{bc}&T_{be}\end{pmatrix}^{\!-1}"
                          r"\begin{pmatrix}R\\R\cdot T\end{pmatrix}"),
            ],
        },

        # ── Group 7: Part E — Rbc, Cbc, Cbe, Cbi ────────────────────────────
        {
            "label":      "Part E — Rbc, Cbc, Cbe, Cbi",
            "params":     [("Rbc_arr", "Rbc", "Rbc", 1e-3, "kΩ"),
                           ("Cbc_arr", "Cbc", "Cbc", 1e15, "fF"),
                           ("Cbe_arr", "Cbe", "Cbe", 1e15, "fF"),
                           ("Cbi_arr", "Cbi", "Cbi", 1e15, "fF")],
            "depends_on": ["Rbe", "Rbi"],
            "formulas": [
                ("latex", r"R_{bc}=\frac{R_{bi}}{R_{bi}/R_{bc}},\quad "
                          r"C_{bc}=\frac{R_{bi}C_{bc}}{R_{bi}},\quad "
                          r"C_{be}=\frac{T_{be}}{R_{be}},\quad "
                          r"C_{bi}=\frac{T_{bi}}{R_{bi}}"),
            ],
        },

        # ── Group 8: Part F — Gm0, τ, Rcx, Ccx ──────────────────────────────
        {
            "label":            "Part F — Gm0, τ, Rcx, Ccx",
            "params":           [("Gm0_a",   "Gm0", "Gm0", 1e3,  "mS"),
                                 ("tau_a",   "tau", "τ",   1e12, "ps"),
                                 ("Rcx_arr", "Rcx", "Rcx", 1e-3, "kΩ"),
                                 ("Ccx_arr", "Ccx", "Ccx", 1e15, "fF")],
            "use_first_params": {"Gm0"},
            "depends_on":       ["Rbc", "Cbc"],
            "formulas": [
                ("latex", r"G_{m0}=\mathrm{Re}(Y_{22})\big|_{f\to 0}"),
                ("latex", r"\tau=-\frac{1}{2\pi}\frac{d\,\angle Y_{21}}{df}"),
                ("latex", r"\frac{1}{R_{cx}}=\mathrm{Re}\!\left(\frac{1}{Z_4}-\frac{1}{Z_2}\right),"
                          r"\quad Z_2=\frac{R_{bc}}{1+j\omega R_{bc}C_{bc}}"),
                ("latex", r"C_{cx}\approx-\frac{\mathrm{Im}(Y_{12})}{\omega}"
                          r"\bigg|_{\mathrm{high\,freq}}"),
            ],
        },
    ]

    @classmethod
    def extract(cls, Y_ex1, freq, n_low, **kwargs):
        return _extract(Y_ex1, freq, n_low,
                        n_fit=kwargs.get("n_fit"),
                        n_fit_f1=kwargs.get("n_fit_f1"))

    @classmethod
    def simulate(cls, params, freq, z0=50.0):
        return _simulate(params, freq, z0)

    @classmethod
    def simulate_vec(cls, params, freq, z0=50.0, xp=None):
        """Vectorised simulate — no per-freq loop.  Pass xp=cupy for GPU."""
        if xp is None:
            xp = np
        return _simulate_vec(params, freq, z0, xp)

    @classmethod
    def simulate_batch(cls, params, freq, z0=50.0, xp=None, cache=None):
        """Batched simulate over (param_combo × freq) — used by tuning sweep.

        Optional ``cache`` dict carries pre-computed constant sub-networks
        (Y_pad, Z_ser, omega) that don't depend on the swept parameters.
        """
        return _simulate_batch(params, freq, z0, xp, cache)

    @classmethod
    def reextract(cls, Y_ex1, freq, n_low, overrides, changed_group_idx, live_arrays):
        """
        Cascade re-extraction called by render_interactive_param_groups.

        Group → param mapping (new numbering):
          G0  z_plots_group   — no params
          G1  fbi_fit_group   — no params; _n_fit forces Tbi recompute
          G2                  — Tbi
          G3                  — RbiOverRbc, RbiCbc
          G4  f1_fit_group    — Tbe; _n_fit_f1 forces Tbe recompute
          G5                  — R, RT
          G6                  — Rbe, Rbi
          G7                  — Rbc, Cbc, Cbe, Cbi
          G8                  — Gm0, τ, Rcx, Ccx (always recomputed)

        Special override keys:
          "_n_fit"    — fit window changed for Fbi → force Tbi from fit
          "_n_fit_f1" — fit window changed for F1  → force Tbe from fit
        """
        omega  = 2.0 * np.pi * freq
        omega2 = omega ** 2
        N      = len(freq)
        n_fit_fbi = int(overrides.get("_n_fit",    max(4, 2 * N // 3)))
        n_fit_f1  = int(overrides.get("_n_fit_f1", max(4, 2 * N // 3)))
        n_hi      = max(N // 2, n_low + 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            Z1 = 1.0 / (Y_ex1[:, 0, 0] + Y_ex1[:, 0, 1])
            Z3 = (Y_ex1[:, 1, 0] + Y_ex1[:, 0, 0]) / \
                 ((Y_ex1[:, 0, 0] + Y_ex1[:, 0, 1]) * (Y_ex1[:, 1, 1] + Y_ex1[:, 0, 1]))
            Z4 = -1.0 / Y_ex1[:, 0, 1]

        # G2 — Tbi
        # Fit window change (_n_fit) or changed_group_idx < 2 → recompute from fit
        _Tbi_fit, Fbi, A0, B0 = _h_Tbi(Z1, Z3, omega, omega2, n_fit_fbi)
        if "_n_fit" in overrides or changed_group_idx < 2:
            Tbi = _Tbi_fit
        else:
            Tbi = float(overrides.get("Tbi", _Tbi_fit))

        # G3 — RbiOverRbc, RbiCbc
        ror_a, rbc_a = _h_ratios(Z1, Z3, omega, Tbi)
        if changed_group_idx >= 3:
            ror = float(overrides.get("RbiOverRbc", safe_median(ror_a, n_low)))
            rbc = float(overrides.get("RbiCbc",     safe_median(rbc_a, n_low)))
        else:
            ror = safe_median(ror_a, n_low)
            rbc = safe_median(rbc_a, n_low)

        # G4 — Tbe
        # Fit window change (_n_fit_f1) or changed_group_idx < 4 → recompute from fit
        _Tbe_fit, F1, A1, B1 = _h_Tbe(Z1, omega, omega2, Tbi, n_fit_f1)
        if "_n_fit_f1" in overrides or changed_group_idx < 4:
            Tbe = _Tbe_fit
        else:
            Tbe = float(overrides.get("Tbe", _Tbe_fit))

        # G5 — R, RT
        R_a, RT_a, F2 = _h_R_RT(Z1, omega, Tbi, Tbe)
        if changed_group_idx >= 5:
            R  = float(overrides.get("R",  safe_median(R_a,  n_low)))
            RT = float(overrides.get("RT", safe_median(RT_a, n_low)))
        else:
            R  = safe_median(R_a,  n_low)
            RT = safe_median(RT_a, n_low)

        # G6 — Rbe, Rbi
        Rbe_a, Rbi_a   = _h_solve_pf(ror_a, rbc_a, R_a, RT_a, Tbi, Tbe)
        Rbe_sc, Rbi_sc = _h_solve_sc(ror, rbc, R, RT, Tbi, Tbe, Z1, Z3, n_low)
        if changed_group_idx >= 6:
            Rbe = float(overrides.get("Rbe", Rbe_sc))
            Rbi = float(overrides.get("Rbi", Rbi_sc))
        else:
            Rbe, Rbi = Rbe_sc, Rbi_sc

        # G7 — Rbc, Cbc, Cbe, Cbi
        Rbc_a, Cbc_a, Cbe_a, Cbi_a = _h_derived_pf(Rbe_a, Rbi_a, ror_a, rbc_a, Tbi, Tbe)
        _Rbc, _Cbc, _Cbe, _Cbi     = _h_derived_sc(Rbi, Rbe, ror, rbc, Tbi, Tbe)
        if changed_group_idx >= 7:
            Rbc = float(overrides.get("Rbc", _Rbc))
            Cbc = float(overrides.get("Cbc", _Cbc))
            Cbe = float(overrides.get("Cbe", _Cbe))
            Cbi = float(overrides.get("Cbi", _Cbi))
        else:
            Rbc, Cbc, Cbe, Cbi = _Rbc, _Cbc, _Cbe, _Cbi

        # G8 — always recomputed from raw data
        Gm0, tau, Gm0_a, tau_a, phi = _h_Gm0_tau(Y_ex1, freq)
        Ccx, Ccx_a                  = _h_Ccx(Y_ex1, omega, n_hi)
        Rcx, Rcx_a                  = _h_Rcx(Z4, Rbc, Cbc, omega)

        new_params = dict(
            Rbi=Rbi, Cbi=Cbi, Rbe=Rbe, Cbe=Cbe,
            Rbc=Rbc, Cbc=Cbc, Rcx=Rcx, Ccx=Ccx, Gm0=Gm0, tau=tau,
            Tbi=Tbi, Tbe=Tbe, A0=A0, B0=B0, A1=A1, B1=B1,
            RbiOverRbc=ror, RbiCbc=rbc, R=R, RT=RT,
        )
        new_arrays = dict(
            Z1=Z1, Z3=Z3, Z4=Z4, F2=F2, Fbi=Fbi, F1=F1, omega2=omega2,
            phase_Y21=phi,
            Tbi_arr=np.full(N, Tbi),
            Tbe_arr=np.full(N, Tbe),
            RbiOverRbc_arr=ror_a, RbiCbc_arr=rbc_a,
            R_arr=R_a, RT_arr=RT_a,
            Rbe_arr=Rbe_a, Rbi_arr=Rbi_a,
            Rbc_arr=Rbc_a, Cbc_arr=Cbc_a, Cbe_arr=Cbe_a, Cbi_arr=Cbi_a,
            Gm0_a=Gm0_a, tau_a=tau_a,
            Rcx_arr=Rcx_a, Ccx_arr=Ccx_a,
        )
        return new_params, new_arrays

    @classmethod
    def render_results_table(cls, params):
        ri = params
        rows = [
            ("Tbi",  f"{ri['Tbi']*1e12:.4f}",  "ps"),
            ("Tbe",  f"{ri['Tbe']*1e12:.4f}",  "ps"),
            ("Rbi",  f"{ri['Rbi']:.4f}",        "Ω"),
            ("Cbi",  f"{ri['Cbi']*1e15:.4f}",   "fF"),
            ("Rbe",  f"{ri['Rbe']:.4f}" if ri['Rbe'] < 1000 else f"{ri['Rbe']*1e-3:.4f}k", "Ω"),
            ("Cbe",  f"{ri['Cbe']*1e15:.4f}" if ri['Cbe'] < 1e-12 else f"{ri['Cbe']*1e12:.4f}",
                     "fF" if ri['Cbe'] < 1e-12 else "pF"),
            ("Rbc",  f"{ri['Rbc']*1e-3:.4f}",  "kΩ"),
            ("Cbc",  f"{ri['Cbc']*1e15:.4f}",  "fF"),
            ("Gm0",  f"{ri['Gm0']*1e3:.4f}",   "mS"),
            ("τ",    f"{ri['tau']*1e12:.4f}",   "ps"),
            ("Rcx",  f"{ri['Rcx']*1e-3:.4f}",  "kΩ"),
            ("Ccx",  f"{ri['Ccx']*1e15:.4f}",  "fF"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Symbol", "Value", "Unit"]),
                     width="stretch", hide_index=True)

    @classmethod
    def render_diagnostic_plots(cls, params, arrays, freq, fname):
        """Fbi and F1 diagnostic fits (now shown inline in the expander). No-op."""
        pass

    @classmethod
    def render_override_and_smith(cls, fname, S_raw, freq, z0,
                                  para_eff, extract_result, **kwargs):
        params, arrays = extract_result
        calc_vals = {**para_eff, **params}
        all_specs = PAD_SPECS + _INT_D_SPECS
        sync_pad_from_preov(fname, cls.SHORT, calc_vals)

        for key, _, scale, *_ in _INT_D_SPECS:
            sk = f"sim_{cls.SHORT}_{key}_{fname}"
            if sk not in st.session_state:
                st.session_state[sk] = float(calc_vals.get(key, 0.0)) * scale

        with st.expander(f"✏️ Fine-tune {cls.NAME} intrinsic parameters", expanded=False):
            if st.button(f"↩️ Reset {cls.NAME} to calculated",
                         key=f"rst_sim_{cls.SHORT}_{fname}"):
                for key, _, scale, *_ in all_specs:
                    st.session_state[f"sim_{cls.SHORT}_{key}_{fname}"] = \
                        float(calc_vals.get(key, 0.0)) * scale
                st.rerun()

            st.markdown("**Pad Parasitics** *(auto-synced)*")
            for row_start in range(0, len(PAD_SPECS), 3):
                row = PAD_SPECS[row_start:row_start+3]
                for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                    col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                                       key=f"sim_{cls.SHORT}_{key}_{fname}",
                                       format=fmt, step=step)

            st.markdown("**Intrinsic (Degachi)**")
            for row_start in range(0, len(_INT_D_SPECS), 4):
                row = _INT_D_SPECS[row_start:row_start+4]
                for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                    col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                                       key=f"sim_{cls.SHORT}_{key}_{fname}",
                                       format=fmt, step=step)

        all_p = {
            key: st.session_state.get(f"sim_{cls.SHORT}_{key}_{fname}",
                                       float(calc_vals.get(key, 0.0))*scale) / scale
            for key, _, scale, *_ in all_specs
        }
        for ek in ["Cpbe_mode","Cpbe_extra","Cpce_mode","Cpce_extra",
                   "Cpbc_mode","Cpbc_extra","Cpar_Lb","Cpar_Lc","Cpar_Le"]:
            all_p[ek] = calc_vals.get(ek, "None" if "mode" in ek else 0.0)

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

        # Persist current (post-override) param dict for the summary table.
        st.session_state[f"current_p_{cls.SHORT}_{fname}"] = dict(all_p)

        render_tuning_expander(cls, all_p, S_raw, freq, z0,
                               PAD_SPECS + _INT_D_SPECS, fname, cls.SHORT)
        return S_sim
