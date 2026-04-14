"""
ssm_deembedding.py — Open/Short de-embedding (Steps 1a and 1b).

References: Gao, HBT for Circuit Design, Wiley 2015, §4.2, §5.5.1~3, and Table 5.3.

Contains three methods for extracting series/access resistances:
  - render_rz12_section: Z-parameter method (Re(Z12) vs 1/IE)
  - render_open_collector_section: Open-collector method (Re(Zij) vs 1/IB)
  - _render_cold_hbt: Cold-HBT extraction (cut-off bias S2P)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .ssm_core import (s_to_y, y_to_z, z_to_y,
                       open_elem_Y, short_lead_Z, safe_median)
from .ssm_s2p          import parse_s2p_bytes, interpolate_s2f, build_Y_pad, build_Z_ser
from .ssm_chart_utils  import plotly_with_dl


# ── Pad matrix builders (also used by forward simulators) ─────────────────────

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

# ── Vectorised pad / series builders (no per-freq loop) ──���──────────────────

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
# These accept omega shaped (1, N) and parameters that may be either scalars
# or (B, 1) arrays.  Result is broadcast to (B, N, 2, 2).

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


    # Implementation of the formulas above ↓
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

    # Implementation of the formulas above ↓
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



# ════════════════════════════════════════════════════════════════════════════════
# Helper: Re(Z12) vs 1/IE
# ════════════════════════════════════════════════════════════════════════════════

def render_rz12_section(all_data, para_eff, fname):
    """
    Z-parameter method (Re(Z12) vs 1/IE).
    """
    st.markdown("#### 📈 Z-Parameter Method  *(Gao [3] Ch. 5.5.1)*")
    st.caption("NOTE: This method is only valid when devices are biased in the active or linear region.")
    st.caption("Valid for emitter access/series resistance (Re) extraction.")
    st.caption("NOT Valid for analysis of files with varying VCB.")
    st.caption("Drawback: Not suitable for Rb and Rc (Gao Table 5.3, pg. 145).")
    st.caption("Re(Z₁₂) = (ηkT/q)·(1/IE) + Re")
    st.latex(r"\mathrm{Re}(Z_{12})=\frac{\eta kT}{q}\cdot\frac{1}{I_E}+R_e")


    if not all_data:
        st.info("No DUT files loaded."); return

    for fn in all_data:
        for suf, dv in [("use", True), ("Ie", 0.0)]:
            gk = f"rz12_{suf}_{fn}"
            if gk not in st.session_state: st.session_state[gk] = dv

    ref_fn  = list(all_data.keys())[0]
    f_ref   = all_data[ref_fn]["freq"] * 1e-9
    f_min_v = float(f_ref[max(1, np.searchsorted(f_ref, 0.01))])
    f_max_v = float(f_ref[-1])
    col_fq, _ = st.columns([1, 2])
    f_extract = col_fq.number_input(
        "Z₁₂ freq (GHz)", min_value=f_min_v, max_value=f_max_v,
        value=min(1.0, f_max_v*0.05), step=0.5, format="%.2f",
        key=f"rz12_fext_{fname}")

    st.markdown("**Files:**")
    for h, t in zip(st.columns([0.3,2.3,1.2,1.4,1.4]),
                    ["","File","IE (mA)","Re(Z₁₂) (Ω)","Rbe*"]):
        h.markdown(f"<small><b>{t}</b></small>", unsafe_allow_html=True)

    points = []; all_rez12 = []; Re_ref = para_eff.get("Rpe", 0.0)
    for fn, d in all_data.items():
        c0,c1,c2,c3,c4 = st.columns([0.3,2.3,1.2,1.4,1.4])
        use = c0.checkbox("", key=f"rz12_use_{fn}__{fname}", value=st.session_state[f"rz12_use_{fn}"],
                           label_visibility="collapsed")
        st.session_state[f"rz12_use_{fn}"] = use
        c1.markdown(f"<small>{Path(fn).stem}</small>", unsafe_allow_html=True)
        if not use: continue
        Ie = c2.number_input("", min_value=0.0, step=0.1, format="%.3f",
                              key=f"rz12_Ie_{fn}__{fname}",
                              value=float(st.session_state[f"rz12_Ie_{fn}"]),
                              label_visibility="collapsed")
        st.session_state[f"rz12_Ie_{fn}"] = Ie
        try:
            idx    = int(np.argmin(np.abs(d["freq"]*1e-9 - f_extract)))
            Y_ex1f = peel_parasitics(d["S_raw"], d["freq"], d["z0"], para_eff)
            ReZ12  = float(y_to_z(Y_ex1f)[idx,0,1].real)
            c3.markdown(f"**{ReZ12:.4f}**")
            c4.markdown(f"<small>{ReZ12-Re_ref:.4f}</small>", unsafe_allow_html=True)
            all_rez12.append((ReZ12, Path(fn).stem))
            if Ie > 0: points.append((1.0/(Ie*1e-3), ReZ12, Path(fn).stem))
        except Exception as ex:
            c3.markdown(f"*err:{ex}*")

    if not all_rez12:
        st.caption("Enable files above to begin."); return

    fig = go.Figure()

    if points:
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        lbl = [p[2] for p in points]
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers+text", text=lbl,
            textposition="top center", name="Re(Z₁₂)",
            marker=dict(size=11, color="#1f77b4", line=dict(color="#0d4a7a", width=1.5))))
    else:
        # No Ie entered yet — show Re(Z₁₂) values on y-axis at x=0
        y0 = np.array([p[0] for p in all_rez12])
        lbl0 = [p[1] for p in all_rez12]
        fig.add_trace(go.Scatter(x=np.zeros(len(y0)), y=y0, mode="markers+text", text=lbl0,
            textposition="top right", name="Re(Z₁₂) (no IE yet)",
            marker=dict(size=11, symbol="circle-open", color="#1f77b4",
                        line=dict(color="#0d4a7a", width=1.5))))

    slope, Re_fit, eta = None, None, None
    if len(points) >= 2:
        try:
            x = np.array([p[0] for p in points])
            y = np.array([p[1] for p in points])
            slope, Re_fit = np.polyfit(x, y, 1)
            eta = slope / (1.381e-23 * 300 / 1.602e-19)
            x_fit = np.linspace(0, max(x)*1.08, 200)
            y_fit = slope*x_fit + Re_fit
            fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines",
                name=f"Fit Re={Re_fit:.4f} Ω  η={eta:.3f}",
                line=dict(color="#d62728", width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=[0], y=[Re_fit], mode="markers",
                name=f"Re={Re_fit:.4f} Ω",
                marker=dict(size=14, symbol="star", color="#d62728")))
        except Exception as ex:
            st.error(f"Fit failed: {ex}")
    elif points:
        st.caption("Need ≥ 2 files with IE for fit.")

    fig.update_layout(
        title=f"Re(Z₁₂) vs 1/IE @ {f_extract:.2f} GHz",
        xaxis=dict(title="1/IE (A⁻¹)", rangemode="tozero",
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Re(Z₁₂) (Ω)", showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=380,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=9)),
        margin=dict(l=55,r=20,t=50,b=50))
    plotly_with_dl(fig, key=f"rz12_{fname}", filename=f"rz12_{fname}")

    if Re_fit is not None:
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        lbl = [p[2] for p in points]
        current_stem  = Path(fname).stem
        current_idx   = next((i for i, l in enumerate(lbl) if l == current_stem), None)
        if current_idx is not None:
            st.session_state[f"rz12_Rbe_{fname}"] = y[current_idx] - Re_fit
        else:
            st.session_state.pop(f"rz12_Rbe_{fname}", None)
        st.session_state[f"rz12_Re_{fname}"] = Re_fit

        mc1, mc2 = st.columns(2)
        mc1.metric("Re (intercept)", f"{Re_fit:.4f} Ω",
                   delta=f"{Re_fit - para_eff.get('Rpe',0):+.4f} vs open-short")
        if current_idx is not None:
            mc2.metric(f"Rbe ({current_stem})", f"{y[current_idx]-Re_fit:.4f} Ω")
        else:
            mc2.info("Current file not in fit.")
    else:
        st.session_state.pop(f"rz12_Re_{fname}", None)
        st.session_state.pop(f"rz12_Rbe_{fname}", None)


def render_open_collector_section(all_data, para_eff, fname):
    """
    Open-collector method: Re(Zij) vs 1/IB linear extrapolation → Rb, Rc, Re.
    Results written into session state as ocm_Rb/Rc/Re_{fname}.
    """
    st.caption("NOTE: This method is only valid when devices are biased with high base current (Ib ≈ 10 ~ 100mA). With high Ib, Ic is assumed to be 0.")
    st.caption("Valid for base/emitter/collector access/series resistance (Rb, Re, Rc) extraction.")
    st.caption("Drawback: Assumption that Rbi tends to 0 (Gao, Table 5.3, pg. 145)")
    st.caption("Re(Z₁₁−Z₁₂) vs 1/IB → Rb,  Re(Z₂₂−Z₁₂) vs 1/IB → Rc,  Re(Z₁₂) vs 1/IB → Re")
    st.latex(r"\mathrm{Re}(Z_{11}-Z_{12})=R_b+f(I_B),\quad"
             r"\mathrm{Re}(Z_{22}-Z_{12})=R_c+f(I_B),\quad"
             r"\mathrm{Re}(Z_{12})=R_e+f(I_B)")

    if not all_data:
        st.info("No DUT files loaded."); return

    for fn in all_data:
        for suf, dv in [("ocm_use", True), ("ocm_Ib", 0.0)]:
            gk = f"{suf}_{fn}"
            if gk not in st.session_state: st.session_state[gk] = dv

    st.markdown("**Files:**")
    hcols = st.columns([0.3, 2.0, 1.2, 1.4, 1.4, 1.4])
    for h, t in zip(hcols, ["", "File", "IB (mA)",
                              "Re(Z11-Z12)", "Re(Z22-Z12)", "Re(Z12)"]):
        h.markdown(f"<small><b>{t}</b></small>", unsafe_allow_html=True)

    ocm_points = []  # (1/Ib, ReZ11Z12, ReZ22Z12, ReZ12, stem)
    for fn, d in all_data.items():
        c0, c1, c2, c3, c4, c5 = st.columns([0.3, 2.0, 1.2, 1.4, 1.4, 1.4])
        use = c0.checkbox("", key=f"ocm_use_{fn}__{fname}",
                          value=st.session_state[f"ocm_use_{fn}"],
                          label_visibility="collapsed")
        st.session_state[f"ocm_use_{fn}"] = use
        c1.markdown(f"<small>{Path(fn).stem}</small>", unsafe_allow_html=True)
        if not use: continue
        Ib = c2.number_input("", min_value=0.0, step=0.1, format="%.3f",
                              key=f"ocm_Ib_{fn}__{fname}",
                              value=float(st.session_state[f"ocm_Ib_{fn}"]),
                              label_visibility="collapsed")
        st.session_state[f"ocm_Ib_{fn}"] = Ib
        try:
            ref_fn2 = list(all_data.keys())[0]
            f_ref2  = all_data[ref_fn2]["freq"] * 1e-9
            f_ext2  = st.session_state.get(f"rz12_fext_{fname}",
                                            min(1.0, float(f_ref2[-1]) * 0.05))
            idx2   = int(np.argmin(np.abs(d["freq"] * 1e-9 - f_ext2)))
            Y_ex1f = peel_parasitics(d["S_raw"], d["freq"], d["z0"], para_eff)
            Z_f    = y_to_z(Y_ex1f)[idx2]
            v1 = float(np.real(Z_f[0, 0] - Z_f[0, 1]))
            v2 = float(np.real(Z_f[1, 1] - Z_f[0, 1]))
            v3 = float(np.real(Z_f[0, 1]))
            c3.markdown(f"**{v1:.4f}**")
            c4.markdown(f"**{v2:.4f}**")
            c5.markdown(f"**{v3:.4f}**")
            if Ib > 0:
                ocm_points.append((1.0 / (Ib * 1e-3), v1, v2, v3,
                                   Path(fn).stem))
        except Exception as ex:
            c3.markdown(f"*err:{ex}*")

    if len(ocm_points) < 2:
        st.caption("Need ≥ 2 files with IB for fit."); return

    xo   = np.array([p[0] for p in ocm_points])
    y1o  = np.array([p[1] for p in ocm_points])
    y2o  = np.array([p[2] for p in ocm_points])
    y3o  = np.array([p[3] for p in ocm_points])
    lblo = [p[4] for p in ocm_points]
    x_fit_o = np.linspace(0, max(xo) * 1.08, 200)

    fig_o = go.Figure()
    for y_arr, name, color in [
        (y1o, "Re(Z11-Z12) → Rb", "#1f77b4"),
        (y2o, "Re(Z22-Z12) → Rc", "#2ca02c"),
        (y3o, "Re(Z12) → Re",     "#ff7f0e"),
    ]:
        try:
            sl, ic = np.polyfit(xo, y_arr, 1)
            fig_o.add_trace(go.Scatter(
                x=xo, y=y_arr, mode="markers+text", text=lblo,
                textposition="top center", name=name,
                marker=dict(size=10, color=color)))
            fig_o.add_trace(go.Scatter(
                x=x_fit_o, y=sl * x_fit_o + ic, mode="lines",
                name=f"{name.split('→')[1].strip()} intercept={ic:.4f} Ω",
                line=dict(color=color, width=1.5, dash="dash")))
            fig_o.add_trace(go.Scatter(
                x=[0], y=[ic], mode="markers",
                marker=dict(size=12, symbol="star", color=color),
                name=f"intercept {ic:.4f} Ω", showlegend=False))
        except Exception:
            pass

    fig_o.update_layout(
        title="Open-collector: Re(Zij) vs 1/IB",
        xaxis=dict(title="1/IB (A⁻¹)", rangemode="tozero",
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Re(Zij) (Ω)", showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=400,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=9)),
        margin=dict(l=55, r=20, t=50, b=50))
    plotly_with_dl(fig_o, key=f"ocm_{fname}", filename=f"open_collector_{fname}")

    try:
        Rb_ocm = float(np.polyfit(xo, y1o, 1)[1])
        Rc_ocm = float(np.polyfit(xo, y2o, 1)[1])
        Re_ocm = float(np.polyfit(xo, y3o, 1)[1])
        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Rb (intercept)", f"{Rb_ocm:.4f} Ω")
        oc2.metric("Rc (intercept)", f"{Rc_ocm:.4f} Ω")
        oc3.metric("Re (intercept)", f"{Re_ocm:.4f} Ω")
        st.session_state[f"ocm_Rb_{fname}"] = Rb_ocm
        st.session_state[f"ocm_Rc_{fname}"] = Rc_ocm
        st.session_state[f"ocm_Re_{fname}"] = Re_ocm
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════════
# Cold-HBT extraction
# ════════════════════════════════════════════════════════════════════════════════

def _render_cold_hbt(fname, open_data, para_caps_ov, do_measured, freq):
    """Cold-HBT extraction UI. Returns cold_res dict or None."""

    st.caption("Used to extract series/access resistances Rb, Rc. Upload cut-off bias (Vce=0, Vbe≤0) S2P 'cold'.")
    st.caption("Drawback: High-frequency measurement (Gao, Table 5.3, pg. 145)")
    cold_file = st.file_uploader("Cold HBT S2P", type=["s2p"], key=f"cold_upload_{fname}")
    if cold_file is None:
        return None
    if open_data is None:
        st.warning("Cold-HBT extraction requires an Open dummy file.")
        return None

    try:
        f_c_raw, S_c_raw, z0_c = parse_s2p_bytes(cold_file.getvalue())
        f_o, S_o, z0_o = open_data
        # Interpolate if grids differ
        if len(f_c_raw) != len(f_o) or not np.allclose(f_c_raw, f_o, rtol=1e-4):
            f_c_use = f_o; S_c_use = interpolate_s2f(f_c_raw, S_c_raw, f_o)
            st.info("Cold S2P interpolated to DUT grid.")
        else:
            f_c_use = f_c_raw; S_c_use = S_c_raw
        omega_c = 2.0*np.pi*f_o; N_c = len(f_o)
        Y_cold  = s_to_y(S_c_use, z0_c)

        # Open admittance (measured or modelled)
        if do_measured:
            Y_open_eff = s_to_y(S_o, z0_o)
        else:
            Y_open_eff = np.zeros((N_c,2,2), dtype=complex)
            for i, w in enumerate(omega_c):
                Y_open_eff[i] = build_Y_pad(para_caps_ov, w)

        # ── Cold-HBT extraction formulas [Gao §5.5.2] ────────────────────
        Z_cor = y_to_z(Y_cold - Y_open_eff)

        z12_choice = st.radio("Use for Z₁₂ in intermediate quantities:",
                               ["Z12", "Z21"], horizontal=True,
                               key=f"cold_z12sel_{fname}")
        Z12_sel = Z_cor[:,0,1] if z12_choice == "Z12" else Z_cor[:,1,0]
        A = np.imag(Z_cor[:,0,0] - Z12_sel)
        B = np.imag(Z_cor[:,1,1] - Z12_sel)
        C = np.real(Z12_sel)

        with np.errstate(divide="ignore", invalid="ignore"):
            disc    = A**2*B**2 + 4.0*A*B*C**2
            D_arr   = np.where(np.abs(C)>1e-30,
                               (A*B + np.sqrt(np.maximum(disc,0.0))) / (2.0*C**2), np.nan)
            Cex_arr = np.where(np.isfinite(D_arr),
                               -((C/B)**2) / (omega_c*A*((1.0+1.0/D_arr)**2+(C/B)**2)), np.nan)
            CbcCex_arr = np.where(np.isfinite(D_arr),
                                   -1.0/(omega_c*B*(1.0+A**2/(C**2*D_arr**2))), np.nan)
            Cbc_arr = CbcCex_arr - Cex_arr
            Rbi_arr = np.where(np.abs(omega_c*Cex_arr)>1e-40,
                               D_arr/(omega_c*Cex_arr), np.nan)
            num_cbe = Rbi_arr * Cex_arr
            den_cbe = Cex_arr + Cbc_arr + 1j*omega_c*Rbi_arr*Cbc_arr*Cex_arr
            Cbe_arr = np.where(np.abs(den_cbe)>1e-40,
                               -1.0/(omega_c*np.imag(Z_cor[:,0,1]-num_cbe/den_cbe)), np.nan)
            Zex_arr = np.where(np.abs(Cex_arr)>1e-40, 1.0/(1j*omega_c*Cex_arr), np.nan+0j)
            Zbc_z   = np.where(np.abs(Cbc_arr)>1e-40, 1.0/(1j*omega_c*Cbc_arr), np.nan+0j)
            Zbe_arr = np.where(np.abs(Cbe_arr)>1e-40, 1.0/(1j*omega_c*Cbe_arr), np.nan+0j)
            denom_b = Zbc_z + Zex_arr + Rbi_arr
            Rb_arr  = np.real((Z_cor[:,0,0]-Z12_sel) -
                              np.where(np.abs(denom_b)>1e-40, Zex_arr*Rbi_arr/denom_b, np.nan+0j))
            Rc_arr  = np.real((Z_cor[:,1,1]-Z12_sel) -
                              np.where(np.abs(denom_b)>1e-40, Zbc_z*Zex_arr/denom_b, np.nan+0j))
            Re_arr  = np.real(Z12_sel - Zbe_arr -
                              np.where(np.abs(denom_b)>1e-40, Zbc_z*Rbi_arr/denom_b, np.nan+0j))

        with st.expander("📊 Intermediate quantities A, B, C, D vs frequency", expanded=False):
            st.markdown("**Definitions**")
            st.latex(r"[Z_{cor}]=[Y_{cold}-Y_{open}]^{-1}")
            st.latex(r"A=\mathrm{Im}(Z_{11}-Z_{12}),\quad B=\mathrm{Im}(Z_{22}-Z_{12}),\quad C=\mathrm{Re}(Z_{12})")
            st.latex(r"D=\frac{AB+\sqrt{A^2B^2+4ABC^2}}{2C^2}")
            f_GHz = f_o / 1e9
            fig_abcd, axes_abcd = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
            for ax, arr_abcd, lbl in zip(
                axes_abcd.flat,
                [A, B, C, D_arr],
                ["A = Im(Z₁₁−Z₁₂)", "B = Im(Z₂₂−Z₁₂)", "C = Re(Z₁₂)", "D"]
            ):
                ax.plot(f_GHz, arr_abcd, "b-", lw=1.5)
                ax.axhline(0, color="k", lw=0.8, ls="--")
                ax.set_ylabel(lbl)
                ax.grid(True, lw=0.4)
                ax.set_ylim(-500, 500)
            for ax in axes_abcd[1]:
                ax.set_xlabel("Frequency (GHz)")
            fig_abcd.suptitle("Cold-HBT intermediate quantities", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_abcd)
            plt.close(fig_abcd)

        with st.expander("📊 Extracted Parameters vs Frequency — Interactive", expanded=False):
            f_ghz_c = f_o * 1e-9
            f_min_v = float(f_ghz_c[0])
            f_max_v = float(f_ghz_c[-1])
            step_v  = max(round((f_max_v - f_min_v) / 100, 3), 0.001)

            cold_res = {}

            def _cold_plot(col_w, arr, res_key, label, scale, unit, formulas, upstream_tag=""):
                """Render formula + slider + Plotly chart + number_input.
                upstream_tag changes the widget key when an upstream scalar changes,
                which resets this input to the new median automatically."""
                for kind, content in formulas:
                    if kind == "latex":
                        col_w.latex(content)
                    else:
                        col_w.markdown(content)

                sl_key = f"cold_pfp_sl_{res_key}_{fname}"
                if sl_key not in st.session_state:
                    st.session_state[sl_key] = (f_min_v, f_max_v)
                f_lo, f_hi = col_w.slider(
                    "Frequency range (GHz)",
                    min_value=f_min_v, max_value=f_max_v,
                    value=st.session_state[sl_key],
                    step=step_v, format="%.2f",
                    key=sl_key)
                rng_tag = f"{f_lo:.3f}_{f_hi:.3f}"

                mask     = (f_ghz_c >= f_lo) & (f_ghz_c <= f_hi)
                f_plot   = f_ghz_c[mask]
                raw      = np.abs(arr[mask]) if np.iscomplexobj(arr) else np.real(arr[mask])
                arr_disp = raw * scale
                fin      = raw[np.isfinite(raw)]
                auto_SI  = abs(float(np.median(fin))) if len(fin) > 0 else 0.0
                auto_disp = auto_SI * scale

                inp_key   = f"cold_pfp_inp_{res_key}_{fname}_{rng_tag}_{upstream_tag}"
                user_disp = float(st.session_state.get(inp_key, auto_disp))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=f_plot, y=arr_disp, mode="lines", name=label,
                    line=dict(color="#1f77b4", width=2)))
                if np.isfinite(user_disp):
                    fig.add_hline(
                        y=user_disp,
                        line=dict(color="#d62728", width=1.8, dash="dash"),
                        annotation_text=f"{user_disp:.4g} {unit}",
                        annotation_position="right",
                        annotation_font=dict(size=9, color="#d62728"))
                fig.update_layout(
                    title=dict(text=label, font=dict(size=12)),
                    xaxis_title="Frequency (GHz)",
                    yaxis_title=f"{label} ({unit})" if unit else label,
                    plot_bgcolor="white", paper_bgcolor="white", height=240,
                    margin=dict(l=50, r=60, t=35, b=40),
                    showlegend=False, hovermode="x unified")

                fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                _y_range = None
                if np.isfinite(user_disp) and abs(user_disp) > 1e-30:
                    _v5  = 5.0 * abs(user_disp)
                    _fin = arr_disp[np.isfinite(arr_disp)]
                    if len(_fin) > 0 and (_fin.max() > _v5 or _fin.min() < -_v5):
                        _y_range = [-_v5, _v5]
                fig.update_yaxes(showgrid=True, gridcolor="#ebebeb",
                                 **({"range": _y_range} if _y_range is not None else {}))


                plotly_with_dl(fig, key=f"cold_pfp_{res_key}_{fname}",
                               filename=f"cold_{res_key}_{fname}", container=col_w)

                actual_val = col_w.number_input(
                    f"{label} ({unit})" if unit else label,
                    value=float(auto_disp),
                    format="%.5g",
                    key=inp_key)
                return abs(actual_val) / scale

            # ── Step 2: Cex ─────────────────────────────────────────────────
            # Depends only on A, B, C, D (all from Z_cor — no user-set upstream)
            st.markdown("**Step 2 — Cex**")
            st.caption("Depends on: A, B, C, D only")
            col_cex, _ = st.columns(2)
            Cex_scalar = _cold_plot(col_cex, Cex_arr, "Cex_cold", "Cex", 1e15, "fF", [
                ("md",    "**Cex** [Gao §5.5.2]"),
                ("latex", r"C_{ex}=-\frac{(C/B)^2}{\omega A\!\left[\left(1+\tfrac{1}{D}\right)^{\!2}+(C/B)^2\right]}"),
            ])
            cold_res["Cex_cold"] = Cex_scalar

            # ── Step 3: Cbc, Rbi ────────────────────────────────────────────
            # Both depend on Cex scalar. Arrays are recomputed live from Cex_scalar.
            # upstream_tag = Cex value → widget key changes when Cex changes
            #   → input resets to new median automatically.
            st.divider()
            st.markdown("**Step 3 — Cbc, Rbi**")
            st.caption("Depend on: Cex")
            up_cex = f"{Cex_scalar:.6e}"
            with np.errstate(divide="ignore", invalid="ignore"):
                Cbc_arr_live = CbcCex_arr - Cex_scalar
                Rbi_arr_live = np.where(
                    np.abs(omega_c * Cex_scalar) > 1e-40,
                    D_arr / (omega_c * Cex_scalar), np.nan)

            col_cbc, col_rbi = st.columns(2)
            Cbc_scalar = _cold_plot(col_cbc, Cbc_arr_live, "Cbc_cold", "Cbc", 1e15, "fF", [
                ("md",    "**Cbc** [Gao §5.5.2]"),
                ("latex", r"C_{bc}+C_{ex}=-\frac{1}{\omega B\!\left[1+\dfrac{A^2}{C^2D^2}\right]}"),
                ("latex", r"C_{bc}=\left(C_{bc}+C_{ex}\right)-C_{ex}"),
            ], upstream_tag=up_cex)
            cold_res["Cbc_cold"] = Cbc_scalar

            Rbi_scalar = _cold_plot(col_rbi, Rbi_arr_live, "Rbi_cold", "Rbi", 1.0, "Ω", [
                ("md",    "**Rbi** [Gao §5.5.2]"),
                ("latex", r"R_{bi}=\frac{D}{\omega\,C_{ex}}"),
            ], upstream_tag=up_cex)
            cold_res["Rbi_cold"] = Rbi_scalar

            # ── Step 4: Cbe ─────────────────────────────────────────────────
            # Depends on Cex, Cbc, Rbi scalars. Array recomputed live.
            st.divider()
            st.markdown("**Step 4 — Cbe**")
            st.caption("Depends on: Cex, Cbc, Rbi")
            up_s3 = f"{Cex_scalar:.6e}_{Cbc_scalar:.6e}_{Rbi_scalar:.6e}"
            with np.errstate(divide="ignore", invalid="ignore"):
                num_cbe      = Rbi_scalar * Cex_scalar
                den_cbe      = (Cex_scalar + Cbc_scalar
                                + 1j * omega_c * Rbi_scalar * Cbc_scalar * Cex_scalar)
                Cbe_arr_live = np.where(
                    np.abs(den_cbe) > 1e-40,
                    -1.0 / (omega_c * np.imag(Z_cor[:, 0, 1] - num_cbe / den_cbe)),
                    np.nan)

            col_cbe, _ = st.columns(2)
            Cbe_scalar = _cold_plot(col_cbe, Cbe_arr_live, "Cbe_cold", "Cbe", 1e15, "fF", [
                ("md",    "**Cbe** [Gao §5.5.2]"),
                ("latex", r"C_{be}=\frac{-1}{\omega\,\mathrm{Im}\!\left("
                          r"Z_{12}-\dfrac{R_{bi}C_{ex}}{C_{ex}+C_{bc}+j\omega R_{bi}C_{bc}C_{ex}}"
                          r"\right)}"),
            ], upstream_tag=up_s3)
            cold_res["Cbe_cold"] = Cbe_scalar

            # ── Step 5: Rb, Rc, Re ──────────────────────────────────────────
            # Depend on Cex, Cbc, Rbi, Cbe scalars. Arrays recomputed live.
            st.divider()
            st.markdown("**Step 5 — Rb, Rc, Re**")
            st.caption("Depend on: Cex, Cbc, Rbi, Cbe")
            up_s4 = f"{up_s3}_{Cbe_scalar:.6e}"
            with np.errstate(divide="ignore", invalid="ignore"):
                Zex_l = 1.0 / (1j * omega_c * Cex_scalar)
                Zbc_l = 1.0 / (1j * omega_c * Cbc_scalar)
                Zbe_l = 1.0 / (1j * omega_c * Cbe_scalar)
                den_l = Zbc_l + Zex_l + Rbi_scalar
                Rb_arr_live = np.real(
                    (Z_cor[:, 0, 0] - Z12_sel) -
                    np.where(np.abs(den_l) > 1e-40, Zex_l * Rbi_scalar / den_l, np.nan + 0j))
                Rc_arr_live = np.real(
                    (Z_cor[:, 1, 1] - Z12_sel) -
                    np.where(np.abs(den_l) > 1e-40, Zbc_l * Zex_l / den_l, np.nan + 0j))
                Re_arr_live = np.real(
                    Z12_sel - Zbe_l -
                    np.where(np.abs(den_l) > 1e-40, Zbc_l * Rbi_scalar / den_l, np.nan + 0j))

            col_rb, col_rc = st.columns(2)
            Rb_scalar = _cold_plot(col_rb, Rb_arr_live, "Rb_cold", "Rb", 1.0, "Ω", [
                ("md",    "**Rb** [Gao §5.5.2]"),
                ("latex", r"R_{bx}=\mathrm{Re}\!\left(Z_{11}-Z_{12}"
                          r"-\frac{Z_{ex}\,R_{bi}}{Z_{bc}+Z_{ex}+R_{bi}}\right)"),
            ], upstream_tag=up_s4)
            cold_res["Rb_cold"] = Rb_scalar

            Rc_scalar = _cold_plot(col_rc, Rc_arr_live, "Rc_cold", "Rc", 1.0, "Ω", [
                ("md",    "**Rc** [Gao §5.5.2]"),
                ("latex", r"R_c=\mathrm{Re}\!\left(Z_{22}-Z_{12}"
                          r"-\frac{Z_{bc}\,Z_{ex}}{Z_{bc}+Z_{ex}+R_{bi}}\right)"),
            ], upstream_tag=up_s4)
            cold_res["Rc_cold"] = Rc_scalar

            col_re, _ = st.columns(2)
            Re_scalar = _cold_plot(col_re, Re_arr_live, "Re_cold", "Re", 1.0, "Ω", [
                ("md",    "**Re** [Gao §5.5.2]"),
                ("latex", r"R_e=\mathrm{Re}\!\left(Z_{12}-Z_{be}"
                          r"-\frac{Z_{bc}\,R_{bi}}{Z_{bc}+Z_{ex}+R_{bi}}\right)"),
            ], upstream_tag=up_s4)
            cold_res["Re_cold"] = Re_scalar

        # ── Model fit verification plots ─────────────────────────────────────
        with st.expander("📊 Cold-HBT Model Fit Verification", expanded=False):
            st.caption(
                "Measured Z_cor vs model reconstructed from extracted parameters.  \n"
                "A good fit confirms the extracted values are self-consistent.")
            f_GHz    = f_o / 1e9
            Zex_m    = 1.0 / (1j * omega_c * cold_res["Cex_cold"])
            Zbc_m    = 1.0 / (1j * omega_c * cold_res["Cbc_cold"])
            Zbe_m    = 1.0 / (1j * omega_c * cold_res["Cbe_cold"])
            denom_m  = Zbc_m + Zex_m + cold_res["Rbi_cold"]
            Z11Z12_meas  = Z_cor[:, 0, 0] - Z_cor[:, 0, 1]
            Z12_meas     = Z_cor[:, 0, 1]
            Z22Z12_meas  = Z_cor[:, 1, 1] - Z_cor[:, 0, 1]
            Z11Z12_model = Zex_m * cold_res["Rbi_cold"] / denom_m + cold_res["Rb_cold"]
            Z12_model    = Zbc_m * cold_res["Rbi_cold"] / denom_m + Zbe_m + cold_res["Re_cold"]
            Z22Z12_model = Zbc_m * Zex_m / denom_m + cold_res["Rc_cold"]
            fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
            plot_data = [
                (Z11Z12_meas, Z11Z12_model, "Z11-Z12"),
                (Z12_meas,    Z12_model,    "Z12"),
                (Z22Z12_meas, Z22Z12_model, "Z22-Z12"),
            ]
            for row, (meas, model, lbl) in enumerate(plot_data):
                for col, (part_fn, part_lbl) in enumerate(
                        [(np.real, "Re"), (np.imag, "Im")]):
                    ax = axes[row, col]
                    ax.plot(f_GHz, part_fn(meas),  "b-",  lw=1.5, label="Measured")
                    ax.plot(f_GHz, part_fn(model), "r--", lw=1.5, label="Model")
                    ax.set_ylabel(f"{part_lbl}({lbl}) (Ω)")
                    ax.legend(fontsize=7)
                    ax.grid(True, lw=0.4)
                    ax.set_ylim(-200, 200)
                    if row == 0:
                        ax.set_title(f"{part_lbl} part")
            for ax in axes[-1]:
                ax.set_xlabel("Frequency (GHz)")
            fig.suptitle("Cold-HBT: Measured vs Model", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        return cold_res
    except Exception as e:
        st.error(f"Cold-HBT failed: {e}")
        return None
