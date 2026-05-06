"""
ssm_deembedding.py — Streamlit render functions for series/access resistance
extraction methods. The pure-math half (step_open, step_short, peel_parasitics,
build_Y_pad_*, build_Z_ser_*, etc.) lives in tools/SSM/helpers/deembed_math.py.

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

from .helpers import (s_to_y, y_to_z, z_to_y,
                      open_elem_Y,
                      parse_s2p_bytes, interpolate_s2f,
                      build_Y_pad,
                      peel_parasitics,
                      plotly_with_dl,
                      quickset_buttons, apply_pending)


# ════════════════════════════════════════════════════════════════════════════════

# Math helpers (_agg_arr, build_Y_pad/Z_ser_vec/_batch, step_open, step_short,
# peel_parasitics) live in tools/SSM/helpers/deembed_math.py and are re-exported
# via `tools.SSM.helpers`. The render_* functions below use them via that surface.


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

def _render_cold_hbt(fname, open_data, para_step1, do_measured, freq,
                       re_zparam=None, open_arr=None, short_arr=None):
    """Cold-HBT extraction UI. Returns cold_res dict or None.

    Per Gao §5.5.2, Z_cor must have pad caps, series inductances, and Re
    removed before A/B/C/D are computed. Missing pieces default to 0 and
    are flagged in the UI.

    open_arr / short_arr (optional) are the per-frequency arrays from
    step_open / step_short — when supplied, the parasitic override inputs
    grow quickset buttons (mean / median / low f / high f / default).
    """

    st.caption("Used to extract series/access resistances Rb, Rc. Upload cut-off bias (Vce=0, Vbe≤0) S2P 'cold'.")
    st.caption("Drawback: High-frequency measurement (Gao, Table 5.3, pg. 145)")
    cold_file = st.file_uploader("Cold HBT S2P", type=["s2p"], key=f"cold_upload_{fname}")
    if cold_file is None:
        return None

    # ── Pull parasitics that should be stripped before extraction ─────────
    Cpbe = float(para_step1.get("Cpbe", 0.0))
    Cpce = float(para_step1.get("Cpce", 0.0))
    Cpbc = float(para_step1.get("Cpbc", 0.0))
    Lb   = float(para_step1.get("Lb",   0.0))
    Lc   = float(para_step1.get("Lc",   0.0))
    Le   = float(para_step1.get("Le",   0.0))
    Re_v = float(re_zparam) if re_zparam is not None else 0.0

    has_open  = open_data is not None
    has_short = (Lb != 0.0) or (Lc != 0.0) or (Le != 0.0)
    has_re    = re_zparam is not None

    st.markdown("**Parasitics being subtracted from Z_cor** (editable — defaults to extracted values)")

    def _cold_input(container, label, key, init_disp, fmt, step,
                    arr_si=None, scale=1.0, unit=""):
        """number_input + quickset buttons (below) for one Cold-HBT parasitic.

        Returns the user's displayed value (in display units).
        """
        apply_pending(key)
        if key not in st.session_state:
            st.session_state[key] = float(init_disp)
        container.number_input(label, key=key, format=fmt, step=step)
        arr_disp = (np.asarray(arr_si) * scale) if arr_si is not None else None
        quickset_buttons(container=container,
                          key_prefix=key, target_key=key,
                          arr_disp=arr_disp,
                          default_disp=float(init_disp),
                          unit=unit, fmt="%.4g", layout="below")
        return float(st.session_state[key])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<small><b>Pad caps (Open)</b></small>", unsafe_allow_html=True)
        if not has_open:
            st.caption("⚠️ no Open file — defaulting to 0 fF")
        Cpce_in = _cold_input(c1, "Cpce (fF)", f"cold_Cpce_in_{fname}",
                              Cpce*1e15, "%.3f", 0.1,
                              arr_si=(open_arr or {}).get("Cpce"),
                              scale=1e15, unit="fF")
        Cpbe_in = _cold_input(c1, "Cpbe (fF)", f"cold_Cpbe_in_{fname}",
                              Cpbe*1e15, "%.3f", 0.1,
                              arr_si=(open_arr or {}).get("Cpbe"),
                              scale=1e15, unit="fF")
        Cpbc_in = _cold_input(c1, "Cpbc (fF)", f"cold_Cpbc_in_{fname}",
                              Cpbc*1e15, "%.3f", 0.1,
                              arr_si=(open_arr or {}).get("Cpbc"),
                              scale=1e15, unit="fF")
        Cpce = Cpce_in * 1e-15
        Cpbe = Cpbe_in * 1e-15
        Cpbc = Cpbc_in * 1e-15
    with c2:
        st.markdown("<small><b>Series L (Short)</b></small>", unsafe_allow_html=True)
        if not has_short:
            st.caption("⚠️ no Short file — defaulting to 0 pH")
        Lb_in = _cold_input(c2, "Lb (pH)", f"cold_Lb_in_{fname}",
                            Lb*1e12, "%.3f", 0.1,
                            arr_si=(short_arr or {}).get("Lb"),
                            scale=1e12, unit="pH")
        Le_in = _cold_input(c2, "Le (pH)", f"cold_Le_in_{fname}",
                            Le*1e12, "%.3f", 0.1,
                            arr_si=(short_arr or {}).get("Le"),
                            scale=1e12, unit="pH")
        Lc_in = _cold_input(c2, "Lc (pH)", f"cold_Lc_in_{fname}",
                            Lc*1e12, "%.3f", 0.1,
                            arr_si=(short_arr or {}).get("Lc"),
                            scale=1e12, unit="pH")
        Lb = Lb_in * 1e-12
        Le = Le_in * 1e-12
        Lc = Lc_in * 1e-12
    with c3:
        st.markdown("<small><b>Re (Z-param)</b></small>", unsafe_allow_html=True)
        if not has_re:
            st.caption("⚠️ Z-param fit unavailable — defaulting to 0 Ω")
        # No per-frequency array for Re here → quickset is skipped (scalar fit).
        Re_v = st.number_input("Re (Ω)", value=float(Re_v),
                               format="%.4f", step=0.01,
                               key=f"cold_Re_in_{fname}")

    para_eff = dict(para_step1)
    para_eff["Cpbe"] = Cpbe
    para_eff["Cpce"] = Cpce
    para_eff["Cpbc"] = Cpbc

    try:
        f_c_raw, S_c_raw, z0_c = parse_s2p_bytes(cold_file.getvalue())
        if has_open:
            f_o, S_o, z0_o = open_data
            f_grid = f_o
            if len(f_c_raw) != len(f_o) or not np.allclose(f_c_raw, f_o, rtol=1e-4):
                S_c_use = interpolate_s2f(f_c_raw, S_c_raw, f_o)
                st.info("Cold S2P interpolated to Open grid.")
            else:
                S_c_use = S_c_raw
        else:
            f_grid  = f_c_raw
            S_c_use = S_c_raw
        omega_c = 2.0*np.pi*f_grid; N_c = len(f_grid)
        Y_cold  = s_to_y(S_c_use, z0_c)

        # Open admittance (measured or modelled). When no open file is
        # provided, all pad caps are 0 → Y_open_eff = 0.
        if has_open and do_measured:
            Y_open_eff = s_to_y(S_o, z0_o)
        else:
            Y_open_eff = np.zeros((N_c,2,2), dtype=complex)
            for i, w in enumerate(omega_c):
                Y_open_eff[i] = build_Y_pad(para_eff, w)

        # ── Cold-HBT extraction formulas [Gao §5.5.2] ────────────────────
        # Z_cor = (cold − pad) − (series-L T-network) − Re·1
        Z_cor = y_to_z(Y_cold - Y_open_eff)
        Z_cor[:,0,0] -= 1j*omega_c*(Lb + Le) + Re_v
        Z_cor[:,1,1] -= 1j*omega_c*(Lc + Le) + Re_v
        Z_cor[:,0,1] -= 1j*omega_c*Le        + Re_v
        Z_cor[:,1,0] -= 1j*omega_c*Le        + Re_v

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
                               -D_arr/(omega_c*Cex_arr), np.nan)
            den_rb  = Cex_arr + Cbc_arr + 1j*omega_c*Rbi_arr*Cbc_arr*Cex_arr
            den_rc  = 1j*omega_c*(Cex_arr + Cbc_arr) - (omega_c**2)*Rbi_arr*Cbc_arr*Cex_arr
            Rb_arr  = np.real((Z_cor[:,0,0]-Z12_sel) -
                              np.where(np.abs(den_rb)>1e-40, Rbi_arr*Cbc_arr/den_rb, np.nan+0j))
            Rc_arr  = np.real((Z_cor[:,1,1]-Z12_sel) -
                              np.where(np.abs(den_rc)>1e-40, 1.0/den_rc, np.nan+0j))

        with st.expander("📊 Intermediate quantities A, B, C, D vs frequency", expanded=False):
            st.markdown("**Definitions**")
            st.latex(r"[Z_{cor}]=[Y_{cold}-Y_{open}]^{-1}")
            st.latex(r"A=\mathrm{Im}(Z_{11}-Z_{12}),\quad B=\mathrm{Im}(Z_{22}-Z_{12}),\quad C=\mathrm{Re}(Z_{12})")
            st.latex(r"D=\frac{AB+\sqrt{A^2B^2+4ABC^2}}{2C^2}")
            f_GHz = f_grid / 1e9
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
            f_ghz_c = f_grid * 1e-9
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
                auto_SI  = abs(float(fin[-1])) if len(fin) > 0 else 0.0
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
                    -D_arr / (omega_c * Cex_scalar), np.nan)

            col_cbc, col_rbi = st.columns(2)
            Cbc_scalar = _cold_plot(col_cbc, Cbc_arr_live, "Cbc_cold", "Cbc", 1e15, "fF", [
                ("md",    "**Cbc** [Gao §5.5.2]"),
                ("latex", r"C_{bc}+C_{ex}=-\frac{1}{\omega B\!\left[1+\dfrac{A^2}{C^2D^2}\right]}"),
                ("latex", r"C_{bc}=\left(C_{bc}+C_{ex}\right)-C_{ex}"),
            ], upstream_tag=up_cex)
            cold_res["Cbc_cold"] = Cbc_scalar

            Rbi_scalar = _cold_plot(col_rbi, Rbi_arr_live, "Rbi_cold", "Rbi", 1.0, "Ω", [
                ("md",    "**Rbi** [Gao §5.5.2]"),
                ("latex", r"R_{bi}=-\frac{D}{\omega\,C_{ex}}"),
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
                    1.0 / (omega_c * np.imag(Z_cor[:, 0, 1] - num_cbe / den_cbe)),
                    np.nan)

            col_cbe, _ = st.columns(2)
            Cbe_scalar = _cold_plot(col_cbe, Cbe_arr_live, "Cbe_cold", "Cbe", 1e15, "fF", [
                ("md",    "**Cbe** [Gao §5.5.2]"),
                ("latex", r"C_{be}=\frac{1}{\omega\,\mathrm{Im}\!\left("
                          r"Z_{12}-\dfrac{R_{bi}C_{ex}}{C_{ex}+C_{bc}+j\omega R_{bi}C_{bc}C_{ex}}"
                          r"\right)}"),
            ], upstream_tag=up_s3)
            cold_res["Cbe_cold"] = Cbe_scalar

            # ── Step 5: Rb, Rc ──────────────────────────────────────────────
            # Depend on Cex, Cbc, Rbi scalars. Arrays recomputed live.
            st.divider()
            st.markdown("**Step 5 — Rb, Rc**")
            st.caption("Depend on: Cex, Cbc, Rbi")
            with np.errstate(divide="ignore", invalid="ignore"):
                den_rb_l = (Cex_scalar + Cbc_scalar
                            + 1j * omega_c * Rbi_scalar * Cbc_scalar * Cex_scalar)
                den_rc_l = (1j * omega_c * (Cex_scalar + Cbc_scalar)
                            - (omega_c ** 2) * Rbi_scalar * Cbc_scalar * Cex_scalar)
                Rb_arr_live = np.real(
                    (Z_cor[:, 0, 0] - Z12_sel) -
                    np.where(np.abs(den_rb_l) > 1e-40,
                             Rbi_scalar * Cbc_scalar / den_rb_l, np.nan + 0j))
                Rc_arr_live = np.real(
                    (Z_cor[:, 1, 1] - Z12_sel) -
                    np.where(np.abs(den_rc_l) > 1e-40, 1.0 / den_rc_l, np.nan + 0j))

            col_rb, col_rc = st.columns(2)
            Rb_scalar = _cold_plot(col_rb, Rb_arr_live, "Rb_cold", "Rb", 1.0, "Ω", [
                ("md",    "**Rb** [Gao §5.5.2]"),
                ("latex", r"R_{bx}=\mathrm{Re}\!\left(Z_{11}-Z_{12}"
                          r"-\frac{R_{bi}\,C_{bc}}"
                          r"{C_{ex}+C_{bc}+j\omega R_{bi}C_{bc}C_{ex}}\right)"),
            ], upstream_tag=up_s3)
            cold_res["Rb_cold"] = Rb_scalar

            Rc_scalar = _cold_plot(col_rc, Rc_arr_live, "Rc_cold", "Rc", 1.0, "Ω", [
                ("md",    "**Rc** [Gao §5.5.2]"),
                ("latex", r"R_c=\mathrm{Re}\!\left(Z_{22}-Z_{12}"
                          r"-\frac{1}{j\omega(C_{ex}+C_{bc})"
                          r"-\omega^{2}R_{bi}C_{bc}C_{ex}}\right)"),
            ], upstream_tag=up_s3)
            cold_res["Rc_cold"] = Rc_scalar

        # ── Model fit verification plots ─────────────────────────────────────
        with st.expander("📊 Cold-HBT Model Fit Verification", expanded=False):
            st.caption(
                "Measured Z_cor vs model reconstructed from extracted parameters.  \n"
                "A good fit confirms the extracted values are self-consistent.")
            f_GHz    = f_grid / 1e9
            Zex_m    = 1.0 / (1j * omega_c * cold_res["Cex_cold"])
            Zbc_m    = 1.0 / (1j * omega_c * cold_res["Cbc_cold"])
            Zbe_m    = 1.0 / (1j * omega_c * cold_res["Cbe_cold"])
            denom_m  = Zbc_m + Zex_m + cold_res["Rbi_cold"]
            Z11Z12_meas  = Z_cor[:, 0, 0] - Z_cor[:, 0, 1]
            Z12_meas     = Z_cor[:, 0, 1]
            Z22Z12_meas  = Z_cor[:, 1, 1] - Z_cor[:, 0, 1]
            den_rb_m = (cold_res["Cex_cold"] + cold_res["Cbc_cold"]
                        + 1j * omega_c * cold_res["Rbi_cold"]
                        * cold_res["Cbc_cold"] * cold_res["Cex_cold"])
            den_rc_m = (1j * omega_c * (cold_res["Cex_cold"] + cold_res["Cbc_cold"])
                        - (omega_c ** 2) * cold_res["Rbi_cold"]
                        * cold_res["Cbc_cold"] * cold_res["Cex_cold"])
            Z11Z12_model = (cold_res["Rbi_cold"] * cold_res["Cbc_cold"] / den_rb_m
                            + cold_res["Rb_cold"])
            Z12_model    = Zbc_m * cold_res["Rbi_cold"] / denom_m + Zbe_m
            Z22Z12_model = 1.0 / den_rc_m + cold_res["Rc_cold"]
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
