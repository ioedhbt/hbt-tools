"""
helpers/deembed_math.py — Pure-math half of Open/Short de-embedding.

Mirrors the math functions currently in tools/SSM/ssm_deembedding.py.
The Streamlit `render_*` functions stay in ssm_deembedding.py; only the
stateless extraction / vectorised builder helpers live here.

References: Gao, HBT for Circuit Design, Wiley 2015, §4.2, §5.5.1~3.

Also includes two thin wrappers used by IOED's `process_dut`:
  - deembed_open_short  (Y_dut, Y_open, Y_short)
  - deembed_thru_half   (Y_dut, Y_thru_deemb)
which were previously inlined in IOED_HBT_RF_extract.py.
"""
from __future__ import annotations
import numpy as np

from .rf_math import (s_to_y, y_to_z, z_to_y,
                      open_elem_Y, short_lead_Z)
from .s2p_io  import build_Y_pad, build_Z_ser


# ── Aggregation helper ────────────────────────────────────────────────────────

def _agg_arr(arr, n0, n1, method="Median", trim_pct=20):
    a = np.asarray(arr[n0:n1], dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    if method == "Trimmed mean":
        k = max(0, int(len(a) * trim_pct / 100))
        s = np.sort(a)
        s = s[k: len(s) - k] if len(s) > 2 * k else s
        return float(np.mean(s)) if len(s) > 0 else np.nan
    return float(np.nanmedian(a))


# ── Vectorised pad / series builders (no per-freq loop) ──────────────────────

def _open_elem_Y_vec(C, mode, extra, omega, xp):
    """Vectorised admittance of one pad cap over an (N,) omega array."""
    if mode == "Parallel L" and extra > 0:
        return 1j*omega*C + 1.0/(1j*omega*extra + 1e-60)
    if mode == "Series L" and extra > 0:
        denom = 1.0 - omega**2 * extra * C
        denom = xp.where(xp.abs(denom) < 1e-10, 1e-10, denom)
        return 1j*omega*C / denom
    if mode == "Series R" and extra > 0:
        return 1j*omega*C / (1.0 + 1j*omega*extra*C)
    return 1j*omega*C


def _short_lead_Z_vec(R, L, Cpar, omega, xp):
    """Vectorised impedance of one series lead over an (N,) omega array."""
    Z = R + 1j*omega*L
    if Cpar > 0:
        return 1.0 / (1.0/Z + 1j*omega*Cpar)
    return Z


def build_Y_pad_vec(p, omega, xp):
    """(N,2,2) pad admittance matrix — fully vectorised."""
    N = len(omega)
    Ypbe = _open_elem_Y_vec(p["Cpbe"], p.get("Cpbe_mode","None"), p.get("Cpbe_extra",0.0), omega, xp)
    Ypce = _open_elem_Y_vec(p["Cpce"], p.get("Cpce_mode","None"), p.get("Cpce_extra",0.0), omega, xp)
    Ypbc = _open_elem_Y_vec(p["Cpbc"], p.get("Cpbc_mode","None"), p.get("Cpbc_extra",0.0), omega, xp)
    Y = xp.zeros((N, 2, 2), dtype=complex)
    Y[:, 0, 0] = Ypbe + Ypbc
    Y[:, 0, 1] = -Ypbc
    Y[:, 1, 0] = -Ypbc
    Y[:, 1, 1] = Ypce + Ypbc
    return Y


def build_Z_ser_vec(p, omega, xp):
    """(N,2,2) series-lead impedance matrix — fully vectorised."""
    N = len(omega)
    Zb = _short_lead_Z_vec(p["Rpb"], p["Lb"], p.get("Cpar_Lb", 0.0), omega, xp)
    Zc = _short_lead_Z_vec(p["Rpc"], p["Lc"], p.get("Cpar_Lc", 0.0), omega, xp)
    Ze = _short_lead_Z_vec(p["Rpe"], p["Le"], p.get("Cpar_Le", 0.0), omega, xp)
    Z = xp.zeros((N, 2, 2), dtype=complex)
    Z[:, 0, 0] = Zb + Ze
    Z[:, 0, 1] = Ze
    Z[:, 1, 0] = Ze
    Z[:, 1, 1] = Zc + Ze
    return Z


# ── Batched (B, N, 2, 2) builders for parameter-sweep tuning ────────────────
# Accept omega shaped (1, N) and parameters that may be either scalars
# or (B, 1) arrays. Result broadcasts to (B, N, 2, 2).

def _open_elem_Y_batch(C, mode, extra, omega, xp):
    """Same formulae as _open_elem_Y_vec but C may be a (B,1) array."""
    if mode == "Parallel L" and extra > 0:
        return 1j*omega*C + 1.0/(1j*omega*extra + 1e-60)
    if mode == "Series L" and extra > 0:
        denom = 1.0 - omega**2 * extra * C
        denom = xp.where(xp.abs(denom) < 1e-10, 1e-10, denom)
        return 1j*omega*C / denom
    if mode == "Series R" and extra > 0:
        return 1j*omega*C / (1.0 + 1j*omega*extra*C)
    return 1j*omega*C


def _short_lead_Z_batch(R, L, Cpar, omega, xp):
    """Same as _short_lead_Z_vec but R/L may be (B,1) arrays."""
    Z = R + 1j*omega*L
    if Cpar > 0:
        return 1.0 / (1.0/Z + 1j*omega*Cpar)
    return Z


def _b1(p, key, default, xp):
    """Fetch p[key] (or default) and reshape (B,) → (B,1).  Scalars stay scalar."""
    v = p.get(key, default)
    a = xp.asarray(v)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def build_Y_pad_batch(p, omega, B, N, xp):
    """Pad admittance as 4 planes (y00, y01, y10, y11), each broadcastable to (B, N).

    Returning planes (rather than a (B, N, 2, 2) tensor) lets the caller
    keep the 2×2 algebra inlined and avoids the cost of allocating /
    scatter-writing a full 4-D complex tensor every chunk.  ``B`` and
    ``N`` are kept in the signature for backward compatibility but the
    result is purely broadcast-shape.
    """
    Cpbe = _b1(p, "Cpbe", 0.0, xp)
    Cpce = _b1(p, "Cpce", 0.0, xp)
    Cpbc = _b1(p, "Cpbc", 0.0, xp)
    Ypbe = _open_elem_Y_batch(Cpbe, p.get("Cpbe_mode","None"), p.get("Cpbe_extra",0.0), omega, xp)
    Ypce = _open_elem_Y_batch(Cpce, p.get("Cpce_mode","None"), p.get("Cpce_extra",0.0), omega, xp)
    Ypbc = _open_elem_Y_batch(Cpbc, p.get("Cpbc_mode","None"), p.get("Cpbc_extra",0.0), omega, xp)
    return (Ypbe + Ypbc, -Ypbc, -Ypbc, Ypce + Ypbc)


def build_Z_ser_batch(p, omega, B, N, xp):
    """Series-lead impedance as 4 planes (z00, z01, z10, z11)."""
    Rpb = _b1(p, "Rpb", 0.0, xp); Lb = _b1(p, "Lb", 0.0, xp)
    Rpc = _b1(p, "Rpc", 0.0, xp); Lc = _b1(p, "Lc", 0.0, xp)
    Rpe = _b1(p, "Rpe", 0.0, xp); Le = _b1(p, "Le", 0.0, xp)
    Zb = _short_lead_Z_batch(Rpb, Lb, p.get("Cpar_Lb", 0.0), omega, xp)
    Zc = _short_lead_Z_batch(Rpc, Lc, p.get("Cpar_Lc", 0.0), omega, xp)
    Ze = _short_lead_Z_batch(Rpe, Le, p.get("Cpar_Le", 0.0), omega, xp)
    return (Zb + Ze, Ze, Ze, Zc + Ze)


# ── Step 1a — Open dummy → pad capacitances ───────────────────────────────────

def step_open(open_data, n0=None, n1=None, method="Median", trim_pct=20):
    """
    Extract pad shunt capacitances from Open dummy.
    Also returns raw conductance arrays for diagnostic plots.

    Returns
    -------
    params : dict  {Cpbe, Cpce, Cpbc}  (SI units, Farads)
    arrays : dict  {Cpbe, Cpce, Cpbc, Gpbe, Gpce, Gpbc, omega}  (per-frequency)

    Formulas [Gao §4.2]:
        Cpbe = Im(Y11_open + Y12_open) / ω
        Cpce = Im(Y22_open + Y12_open) / ω
        Cpbc = −Im(Y12_open) / ω
        Gpbe = Re(Y11_open + Y12_open)   ← nonzero only if series R or parallel G
    """
    f, S_o, z0 = open_data
    omega = 2.0*np.pi*f
    N = len(f)
    if n0 is None: n0 = N // 2
    if n1 is None: n1 = N
    Y_o = s_to_y(S_o, z0)

    Cpbe_arr = np.imag(Y_o[:,0,0] + Y_o[:,0,1]) / omega
    Cpce_arr = np.imag(Y_o[:,1,1] + Y_o[:,0,1]) / omega
    Cpbc_arr = -np.imag(Y_o[:,0,1]) / omega
    Gpbe_arr = np.real(Y_o[:,0,0] + Y_o[:,0,1])
    Gpce_arr = np.real(Y_o[:,1,1] + Y_o[:,0,1])
    Gpbc_arr = -np.real(Y_o[:,0,1])

    params = dict(
        Cpbe=abs(_agg_arr(Cpbe_arr, n0, n1, method, trim_pct)),
        Cpce=abs(_agg_arr(Cpce_arr, n0, n1, method, trim_pct)),
        Cpbc=abs(_agg_arr(Cpbc_arr, n0, n1, method, trim_pct)),
    )

    arrays = dict(Cpbe=Cpbe_arr, Cpce=Cpce_arr, Cpbc=Cpbc_arr,
                  Gpbe=Gpbe_arr, Gpce=Gpce_arr, Gpbc=Gpbc_arr, omega=omega)
    return params, arrays


# ── Step 1b — Short dummy → lead inductances & series resistances ──────────────

def step_short(short_data, freq, Cpbe, Cpce, Cpbc,
                 open_data=None, n0=None, n1=None, method="Median", trim_pct=20,
                 measured_open=True,
                 Cpbe_mode="None", Cpbe_extra=0.0,
                 Cpce_mode="None", Cpce_extra=0.0,
                 Cpbc_mode="None", Cpbc_extra=0.0):
    """
    Extract lead inductances and series resistances from Short dummy.
    The Open pad admittance is subtracted first (measured or modelled).

    Returns
    -------
    params : dict  {Le, Lb, Lc, Rpe, Rpb, Rpc}  (SI units)
    arrays : dict  {Le, Lb, Lc, Rpe, Rpb, Rpc, warnings}  (per-frequency)

    Formulas [Gao §4.2]:
        Z_corr = [Y_short − Y_open]⁻¹
        Re = Re(Z12_corr)
        Rb = Re(Z11_corr − Z12_corr)
        Rc = Re(Z22_corr − Z21_corr)    ← Note: Gao text has erratum (Z11 vs Z22)
        Le = Im(Z12_corr) / ω,   Lb = Im(Z11−Z12) / ω,   Lc = Im(Z22−Z21) / ω
    """
    _, S_s, z0 = short_data
    omega = 2.0*np.pi*freq
    N = len(freq)
    if n0 is None: n0 = 0
    if n1 is None: n1 = max(3, int(N * 0.20))

    Y_s = s_to_y(S_s, z0)

    # Build Open admittance (measured or modelled)
    if measured_open and open_data is not None:
        _, S_o, z0_o = open_data
        Y_open_eff = s_to_y(S_o, z0_o)
    else:
        Y_open_eff = np.zeros((N,2,2), dtype=complex)
        for i, w in enumerate(omega):
            Ypbe = open_elem_Y(Cpbe, Cpbe_mode, Cpbe_extra, w)
            Ypce = open_elem_Y(Cpce, Cpce_mode, Cpce_extra, w)
            Ypbc = open_elem_Y(Cpbc, Cpbc_mode, Cpbc_extra, w)
            Y_open_eff[i] = np.array([[Ypbe+Ypbc, -Ypbc],
                                       [-Ypbc, Ypce+Ypbc]])

    Z_corr = y_to_z(Y_s - Y_open_eff)
    Rpe_arr = np.real(Z_corr[:,0,1])
    Rpb_arr = np.real(Z_corr[:,0,0] - Z_corr[:,0,1])
    Rpc_arr = np.real(Z_corr[:,1,1] - Z_corr[:,1,0])
    Le_arr  = np.imag(Z_corr[:,0,1]) / omega
    Lb_arr  = np.imag(Z_corr[:,0,0] - Z_corr[:,0,1]) / omega
    Lc_arr  = np.imag(Z_corr[:,1,1] - Z_corr[:,1,0]) / omega

    Le_raw  = abs(_agg_arr(Le_arr,  n0, n1, method, trim_pct))
    Lb_raw  = abs(_agg_arr(Lb_arr,  n0, n1, method, trim_pct))
    Lc_raw  = abs(_agg_arr(Lc_arr,  n0, n1, method, trim_pct))
    Rpe_raw = abs(_agg_arr(Rpe_arr, n0, n1, method, trim_pct))
    Rpb_raw = abs(_agg_arr(Rpb_arr, n0, n1, method, trim_pct))
    Rpc_raw = abs(_agg_arr(Rpc_arr, n0, n1, method, trim_pct))

    NOISE = 3e-12
    warnings_list = []
    if Le_raw  < -NOISE: warnings_list.append("Le significantly negative — Open caps may over-correct.")
    if Lb_raw  < -NOISE: warnings_list.append(f"⚠️ Lb negative ({Lb_raw*1e12:.1f} pH).")
    if Lc_raw  < -NOISE: warnings_list.append(f"⚠️ Lc negative ({Lc_raw*1e12:.1f} pH).")

    params = dict(Le=Le_raw, Lb=Lb_raw, Lc=Lc_raw,
                  Rpe=Rpe_raw, Rpb=Rpb_raw, Rpc=Rpc_raw)
    arrays = dict(Le=Le_arr, Lb=Lb_arr, Lc=Lc_arr,
                  Rpe=Rpe_arr, Rpb=Rpb_arr, Rpc=Rpc_arr,
                  warnings=warnings_list)
    return params, arrays


# ── Pad peeling (used by all models) ──────────────────────────────────────────

def peel_parasitics(S_raw, freq, z0, p: dict) -> np.ndarray:
    """
    Remove Open+Short pad parasitics from DUT S-parameters.
    Returns Y_ex1 (admittance after full de-embedding), ready for model extraction.

    p must contain: Cpbe/ce/bc (+ optional _mode/_extra),
                    Lb/Lc/Le, Rpb/Rpc/Rpe (+ optional Cpar_Lb/Lc/Le).
    """
    omega = 2.0*np.pi*freq
    Y_dut = s_to_y(S_raw, z0)

    # 1. Build and subtract pad shunt admittance (Open de-embedding)
    Y_pad = np.zeros((len(freq),2,2), dtype=complex)
    for i, w in enumerate(omega):
        Y_pad[i] = build_Y_pad(p, w)
    Z1 = y_to_z(Y_dut - Y_pad)

    # 2. Build and subtract series lead impedance (Short de-embedding)
    Z_ser = np.zeros((len(freq),2,2), dtype=complex)
    for i, w in enumerate(omega):
        Z_ser[i] = build_Z_ser(p, w)

    return z_to_y(Z1 - Z_ser)   # → Y_ex1


# ── Thin wrappers (previously inline in IOED_HBT_RF_extract.py) ──────────────

def deembed_open_short(Y_dut, Y_open, Y_short):
    """Standard open-short de-embedding (Gao 2015 §4.2)."""
    return z_to_y(y_to_z(Y_dut - Y_open) - y_to_z(Y_short - Y_open))


def deembed_thru_half(Y_dut, Y_thru_deemb):
    """THRU/2 half-impedance subtraction."""
    return z_to_y(y_to_z(Y_dut) - 0.5*y_to_z(Y_thru_deemb))
