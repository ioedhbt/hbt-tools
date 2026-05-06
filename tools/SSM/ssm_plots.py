"""
ssm_plots.py — Step 1 diagnostic plots and helper visualisations.

All functions render directly into Streamlit and return any UI-state values
needed by the caller (e.g. extended element modes, Cpar values).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from .helpers          import (open_elem_Y, s_to_y, y_to_z,
                                peel_parasitics, simulate_open,
                                plotly_with_dl,
                                compute_h21_U, find_ft_fmax, extrap_20dbdec)
from .models.base_ui   import render_smith_chart, ssm_residual


# ── Open element mode constants ───────────────────────────────────────────────

OPEN_MODES      = ["None", "Parallel L", "Series L", "Series R"]
_MODE_UNIT      = {"None": None, "Parallel L": "pH", "Series L": "pH", "Series R": "Ω"}
_MODE_SCALE     = {"None": 1,    "Parallel L": 1e12, "Series L": 1e12, "Series R": 1.0}


# ════════════════════════════════════════════════════════════════════════════════
# Open dummy plots
# ════════════════════════════════════════════════════════════════════════════════

def render_open_plots(open_data, para_caps, open_arr, fname=""):
    """
    Render four expanders for Open dummy diagnostics:
      1. Extra-element model controls (Parallel L / Series L / Series R per cap)
      2. Capacitance Im(Y)/ω vs frequency  [0–50 fF, fixed]
      3. Conductance Re(Y) vs frequency    [series R indicator]
      4. Im(Y)/ω vs 1/ω²                  [Parallel L linearisation]
      5. Smith chart: measured vs forward-simulated Open

    Returns
    -------
    dict  {cap: (mode_str, extra_SI)}  — propagated into para_eff and forward sims.
    """
    f_o, S_o, z0_o = open_data
    f_ghz = f_o * 1e-9
    omega  = 2.0*np.pi*f_o

    # ── 1. Per-cap extra element controls ─────────────────────────────────────
    with st.expander("🔧 Open Pad Element Model — extra parasitic options", expanded=False):
        st.markdown(
            "Each pad capacitor can include one secondary parasitic element.  \n"
            "**Parallel L** = inductor ∥ C → resonance at 1/√LC, affects Im(Y)/ω.  \n"
            "**Series L**   = inductor in series with C → increases effective C near resonance.  \n"
            "**Series R**   = resistor in series with C → adds Re(Y) that rises then saturates.  \n"
            "These choices propagate into de-embedding, Cold-HBT correction, and S2P downloads."
        )
        for cap, cap_lbl in [("Cpbe","Cpbe (B-E)"), ("Cpce","Cpce (C-E)"), ("Cpbc","Cpbc (B-C)")]:
            st.markdown(f"**{cap_lbl}**")
            c1, c2 = st.columns([2, 1])
            mode_sk  = f"open_mode_{cap}_{fname}"
            extra_sk = f"open_extra_{cap}_{fname}"
            if mode_sk  not in st.session_state: st.session_state[mode_sk]  = "None"
            if extra_sk not in st.session_state: st.session_state[extra_sk] = 0.0
            c1.radio("", OPEN_MODES, horizontal=True, key=mode_sk, label_visibility="collapsed")
            mode = st.session_state[mode_sk]
            if mode != "None":
                unit = _MODE_UNIT[mode]
                c2.number_input(f"Extra {unit}", min_value=0.0,
                                step=0.1 if unit == "pH" else 0.01,
                                format="%.3f" if unit == "pH" else "%.4f",
                                key=extra_sk)

    def _get_mode_extra(cap):
        """Return (mode_str, extra_in_SI) for one cap."""
        mode  = st.session_state.get(f"open_mode_{cap}_{fname}", "None")
        extra_disp = float(st.session_state.get(f"open_extra_{cap}_{fname}", 0.0))
        sc    = _MODE_SCALE.get(mode, 1)
        return mode, extra_disp / sc

    # ── 2. Capacitance plot ───────────────────────────────────────────────────
    with st.expander("📊 Open — Pad Capacitances vs Frequency", expanded=True):
        fig_cap = go.Figure()
        for key, lbl, col in [("Cpbe","Cpbe","#1f77b4"),
                               ("Cpce","Cpce","#ff7f0e"),
                               ("Cpbc","Cpbc","#2ca02c")]:
            arr_fF = open_arr[key] * 1e15
            val_fF = para_caps[key] * 1e15
            mode, extra = _get_mode_extra(key)
            # Measured trace
            fig_cap.add_trace(go.Scatter(x=f_ghz, y=arr_fF,
                name=f"{lbl} (meas.)", line=dict(color=col, width=2), mode="lines"))
            # Median dashed line
            fig_cap.add_trace(go.Scatter(x=[f_ghz[0], f_ghz[-1]], y=[val_fF, val_fF],
                name=f"{lbl}={val_fF:.3f} fF", line=dict(color=col, width=1.8, dash="dash"), mode="lines"))
            # Modelled effective C overlay (when extra element chosen)
            if mode != "None":
                # See ssm_core.open_elem_Y for the formula
                Y_mod_arr = np.array([open_elem_Y(para_caps[key], mode, extra, w) for w in omega])
                Ceff_fF   = np.imag(Y_mod_arr) / omega * 1e15
                fig_cap.add_trace(go.Scatter(x=f_ghz, y=Ceff_fF,
                    name=f"{lbl} model ({mode})", line=dict(color=col, width=2, dash="dot"), mode="lines"))
        fig_cap.update_layout(title="Pad Capacitances — Im(Y)/ω",
            xaxis_title="Frequency (GHz)", yaxis_title="Cap (fF)",
            plot_bgcolor="white", paper_bgcolor="white", height=360,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=9)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_cap.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_cap.update_yaxes(showgrid=True, gridcolor="#ebebeb", range=[0, 50])
        plotly_with_dl(fig_cap, key=f"step1_cap_{fname}", filename=f"open_cap_{fname}")
        st.caption("Flat line = pure C. Slope/resonance = inductive effect. Range fixed 0–50 fF.")

    # ── 3. Conductance plot (Series R indicator) ──────────────────────────────
    with st.expander("📊 Open — Pad Conductance vs Frequency (Re(Y) — series R indicator)",
                     expanded=False):
        fig_g = go.Figure()
        for key, lbl, col in [("Cpbe","Gpbe","#1f77b4"),
                               ("Cpce","Gpce","#ff7f0e"),
                               ("Cpbc","Gpbc","#2ca02c")]:
            arr_mS = open_arr[f"G{key[1:]}"] * 1e3   # Gpbe/Gpce/Gpbc keys
            mode, extra = _get_mode_extra(key)
            fig_g.add_trace(go.Scatter(x=f_ghz, y=arr_mS,
                name=f"{lbl} (meas.)", line=dict(color=col, width=2), mode="lines"))
            if mode == "Series R" and extra > 0:
                # Re[Y_series_R] = ω²RC² / (1+ω²R²C²)  → see ssm_core.open_elem_Y
                G_mod = np.array([np.real(open_elem_Y(para_caps[key], mode, extra, w))*1e3
                                  for w in omega])
                fig_g.add_trace(go.Scatter(x=f_ghz, y=G_mod,
                    name=f"{lbl} model (R={extra:.3f} Ω)",
                    line=dict(color=col, width=2, dash="dot"), mode="lines"))
        fig_g.add_hline(y=0, line_color="#aaa", line_width=1)
        fig_g.update_layout(
            title="Pad Conductance Re(Y) — nonzero = series R or parallel G loss",
            xaxis_title="Frequency (GHz)", yaxis_title="Conductance (mS)",
            plot_bgcolor="white", paper_bgcolor="white", height=320,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=9)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_g.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_g.update_yaxes(showgrid=True, gridcolor="#ebebeb")
        plotly_with_dl(fig_g, key=f"step1_cond_{fname}", filename=f"open_conductance_{fname}")
        st.caption(
            "Pure C → Re(Y)=0.  "
            "Series R → Re(Y) = ω²RC² / (1+ω²R²C²) — rises then saturates.  \n"
            "Select 'Series R' above to overlay the modelled curve.")

    # ── 4. Im(Y)/ω vs 1/ω² (Parallel L linearisation) ───────────────────────
    with st.expander("📊 Open — Im(Y)/ω vs 1/ω²  (Parallel L linearisation)", expanded=False):
        fig_l = go.Figure()
        one_over_omega2 = 1.0 / (omega**2 + 1e-60)
        for key, lbl, col in [("Cpbe","Cpbe","#1f77b4"),
                               ("Cpce","Cpce","#ff7f0e"),
                               ("Cpbc","Cpbc","#2ca02c")]:
            Ceff = open_arr[key]  # Im(Y)/ω already stored per-frequency
            fig_l.add_trace(go.Scatter(
                x=one_over_omega2*1e-18, y=Ceff*1e15, name=lbl,
                line=dict(color=col, width=2), mode="lines",
                hovertemplate="1/ω²=%{x:.4f}×10¹⁸<br>Im(Y)/ω=%{y:.3f} fF<extra></extra>"))
        fig_l.update_layout(
            title="Im(Y)/ω vs 1/ω² — slope = −1/L if Parallel L present",
            xaxis_title="1/ω² (× 10¹⁸ rad⁻²s²)", yaxis_title="Im(Y)/ω  (fF equivalent)",
            plot_bgcolor="white", paper_bgcolor="white", height=320,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=9)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_l.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_l.update_yaxes(showgrid=True, gridcolor="#ebebeb", range=[0, 50])
        plotly_with_dl(fig_l, key=f"step1_lind_{fname}", filename=f"open_lind_{fname}")
        st.caption(
            "Parallel L model: Im(Y)/ω = C − 1/(ω²L).  "
            "A straight line with negative slope → L = −1/slope (SI).")

    # ── 5. Smith chart: measured vs modelled Open ─────────────────────────────
    with st.expander("📡 Open — Measured vs Modelled Smith Chart", expanded=False):
        _p_open = {}
        for cap in ["Cpbe", "Cpce", "Cpbc"]:
            mode, extra = _get_mode_extra(cap)
            _p_open[cap]            = para_caps[cap]
            _p_open[f"{cap}_mode"]  = mode
            _p_open[f"{cap}_extra"] = extra
        # Forward-simulate open (ssm_s2p.simulate_open)
        S_open_sim = simulate_open(_p_open, f_o, z0_o)
        err_open   = ssm_residual(S_o, S_open_sim)
        render_smith_chart(S_o, S_open_sim,
                           f"Open dummy",
                           err_open,
                           scales={"S11":1.0,"S12":1.0,"S21":1.0,"S22":1.0},
                           key=f"smith_open_{fname}")
        st.caption("Adjust the extra element controls above — the modelled curve updates live.")

    return {cap: _get_mode_extra(cap) for cap in ["Cpbe", "Cpce", "Cpbc"]}


# ════════════════════════════════════════════════════════════════════════════════
# Short dummy plots
# ════════════════════════════════════════════════════════════════════════════════

def render_short_plots(short_arr, para_short, fname="", freq=None):
    """
    Render Lead inductances vs frequency [0–150 pH, fixed] for Short dummy diagnostics.

    Parameters
    ----------
    short_arr  : dict from step_short — must contain Lb, Lc, Le arrays.
    para_short : dict of scalar fitted values (used as horizontal reference lines).
    fname      : Streamlit widget-key suffix.
    freq       : Hz array matching the Lb/Lc/Le sample positions. When provided
                 the x-axis is plotted in GHz (downloads as L vs freq, not point
                 index). When None the legacy point-index axis is used.
    """

    with st.expander("📊 Short — Lead Inductances vs Frequency", expanded=True):
        fig_ind = go.Figure(); any_neg = False
        for key, lbl, col in [("Lb","Lb","#8e44ad"),
                               ("Lc","Lc","#e67e22"),
                               ("Le","Le","#16a085")]:
            arr_pH = short_arr[key] * 1e12
            val_pH = para_short[key] * 1e12
            if val_pH < 0: any_neg = True
            if freq is not None:
                xv = np.asarray(freq) * 1e-9
                hov = (f"{lbl}=%{{y:.4f}} pH<br>"
                       f"f=%{{x:.3f}} GHz<extra></extra>")
            else:
                xv = np.arange(len(arr_pH))
                hov = f"{lbl}=%{{y:.4f}} pH<extra></extra>"
            fig_ind.add_trace(go.Scatter(x=xv, y=arr_pH,
                name=f"{lbl} (per-freq)",
                line=dict(color=col, width=2), mode="lines",
                hovertemplate=hov))
            fig_ind.add_trace(go.Scatter(x=[xv[0], xv[-1]], y=[val_pH, val_pH],
                name=f"{lbl}={val_pH:.2f} pH",
                line=dict(color=col, width=1.8, dash="dash"), mode="lines"))
        fig_ind.add_hline(y=0, line_color="#333", line_width=1.2,
                           annotation_text="0 pH", annotation_position="left",
                           annotation_font=dict(size=9, color="#333"))
        x_title = "Frequency (GHz)" if freq is not None else "Point index"
        fig_ind.update_layout(title="Lead Inductances",
            xaxis_title=x_title, yaxis_title="Inductance (pH)",
            plot_bgcolor="white", paper_bgcolor="white", height=360,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=9)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_ind.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_ind.update_yaxes(showgrid=True, gridcolor="#ebebeb", range=[0, 150])
        plotly_with_dl(fig_ind, key=f"step1_ind_{fname}", filename=f"short_inductances_{fname}")
        if any_neg:
            st.warning("One or more lead inductances are negative. Use Short Override to correct.")
        st.caption("Range fixed 0–150 pH.")



# ════════════════════════════════════════════════════════════════════════════════
# Helper: S-parameter comparison plot
# ════════════════════════════════════════════════════════════════════════════════

def _rlc_params_summary(p, fname):
    """Build header param dict for a de-embedded S2P file."""
    d = {"DUT_file": Path(fname).stem}
    for k in ["Cpbe","Cpce","Cpbc"]:
        d[k] = f"{p.get(k,0)*1e15:.4f} fF"
        mode = p.get(f"{k}_mode","None")
        if mode != "None":
            extra = p.get(f"{k}_extra",0.0)
            unit_e = "pH" if "L" in mode else "Ω"
            sc_e   = 1e12 if "L" in mode else 1.0
            d[f"{k}_extra"] = f"{mode}: {extra*sc_e:.4f} {unit_e}"
    for k, lbl, sc_v, unit in [("Lb","Lb",1e12,"pH"),("Lc","Lc",1e12,"pH"),
                                ("Le","Le",1e12,"pH"),("Rpb","Rb",1,"Ω"),
                                ("Rpc","Rc",1,"Ω"),("Rpe","Re",1,"Ω")]:
        d[lbl] = f"{p.get(k,0)*sc_v:.4f} {unit}"
    for ck in ["Cpar_Lb","Cpar_Lc","Cpar_Le"]:
        v = p.get(ck,0.0)
        if v > 0: d[ck] = f"{v*1e15:.4f} fF"
    return d


def _compare_bode_smith(*, S_a, S_b, freq, fname, key_suffix,
                        label_a, label_b, color_a, color_b,
                        smith_meas_label, smith_sim_label, gain_title):
    """Render a side-by-side bode (h21² + Mason U) and Smith chart comparing
    two S-parameter datasets.  Used by both the OS-deembedded and intrinsic
    preview sections."""
    from .models.base_ui import smith_scale_controls

    f_ghz = freq * 1e-9
    h21_a, U_a = compute_h21_U(S_a)
    h21_b, U_b = compute_h21_U(S_b)

    fT_a, fmax_a = find_ft_fmax(f_ghz, h21_a, U_a)
    fT_b, fmax_b = find_ft_fmax(f_ghz, h21_b, U_b)

    def _meas_lbl(name, in_val, ext_val):
        if in_val is not None:
            return f"{name}={in_val:.2f} GHz"
        if ext_val is not None:
            return f"{name}≈{ext_val:.2f} GHz (extrap)"
        return f"{name}=n/a"

    fig = go.Figure()
    f_high_track = float(f_ghz[-1])
    extrap_used = False

    def _add(y_arr, name_fmt, color, symbol, kind, in_val):
        """Plot one trace and append the (in-band or extrap) value to its
        legend name.  ``name_fmt`` must include a ``{lbl}`` placeholder.
        """
        nonlocal f_high_track, extrap_used
        f_ext, g_ext, f0 = extrap_20dbdec(f_ghz, y_arr)
        ext_val = f0 if f_ext is not None else None
        legend_name = name_fmt.format(lbl=_meas_lbl(kind, in_val, ext_val))
        fig.add_trace(go.Scatter(
            x=f_ghz, y=y_arr, mode="lines+markers", name=legend_name,
            line=dict(color=color, width=1.4),
            marker=dict(symbol=symbol, size=6, color=color)))
        if f_ext is not None:
            extrap_used = True
            f_high_track = max(f_high_track, f0)
            fig.add_trace(go.Scatter(
                x=f_ext, y=g_ext, mode="lines", name=f"{legend_name} extrap",
                line=dict(color=color, width=1.6, dash="dot"),
                showlegend=False))

    _add(h21_a, f"|h21|² {label_a}  [{{lbl}}]", color_a, "circle", "fT",   fT_a)
    _add(U_a,   f"Mason U {label_a}  [{{lbl}}]", color_a, "square", "fmax", fmax_a)
    _add(h21_b, f"|h21|² {label_b}  [{{lbl}}]", color_b, "circle", "fT",   fT_b)
    _add(U_b,   f"Mason U {label_b}  [{{lbl}}]", color_b, "square", "fmax", fmax_b)

    fig.add_hline(y=0, line_color="#333", line_width=1.2,
                  annotation_text="0 dB", annotation_position="right",
                  annotation_font=dict(size=9))

    # fT/fmax values are reported in trace legend entries above; vertical
    # markers on the bode plot are intentionally omitted.

    x_min = max(float(f_ghz[0]), 1e-2)
    x_max = float(f_high_track) * 1.25 if extrap_used else float(f_ghz[-1])
    fig.update_layout(
        xaxis=dict(title="Frequency (GHz)", type="log",
                   range=[np.log10(x_min), np.log10(x_max)],
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Gain (dB)", range=[0, 50],
                   showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=560,
        legend=dict(orientation="h", x=0.5, y=-0.22,
                    xanchor="center", yanchor="top",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=9)),
        hovermode="x unified", margin=dict(l=55, r=20, t=40, b=160))

    extrap_note = ("   Dotted = 20 dB/dec extrapolation past the measured band."
                   if extrap_used else "")
    st.markdown(
        f"○ = |h21|² (→ fT).   □ = Mason U (→ fmax).   Y-axis fixed 0–50 dB.{extrap_note}")

    sc = smith_scale_controls(fname, key_suffix)
    col_gain, col_smith = st.columns([1, 1])
    with col_gain:
        st.markdown(f"**{gain_title}**")
        plotly_with_dl(fig, key=f"bode_{key_suffix}_{fname}",
                       filename=f"bode_{key_suffix}_{fname}")
    with col_smith:
        st.markdown(f"**S-Parameters: {smith_meas_label} vs {smith_sim_label}**")
        err = ssm_residual(S_a, S_b)
        render_smith_chart(S_a, S_b, f"{smith_meas_label} vs {smith_sim_label}",
                           err, sc, key=f"smith_{key_suffix}_{fname}",
                           show_title=False,
                           meas_label=smith_meas_label,
                           sim_label=smith_sim_label)

    return fT_b, fmax_b   # return the right-hand trace's fT/fmax for download header


def render_os_deemb_preview(S_raw, freq, z0, para_step1, fname):
    """
    Step 2 — De-embedded Preview.

    Shows raw vs Open/Short de-embedded (pad caps + lead L,R from Steps 1a/1b
    only; no access-resistance correction).  Provides bode+smith comparison
    and a download of the OS de-embedded S2P file.
    """
    from .helpers import y_to_s_batch, write_s2p

    st.markdown(
        "<div style='background:linear-gradient(90deg,#e8f5e9 0%,transparent 100%);"
        "border-left:4px solid #2e7d32;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>📊 Raw vs Open/Short De-embedded</strong></div>",
        unsafe_allow_html=True)

    Y_step1 = peel_parasitics(S_raw, freq, z0, para_step1)
    S_step1 = y_to_s_batch(Y_step1, z0)

    with st.expander("📐 Open/Short de-embedding formulas", expanded=False):
        st.markdown(
            "**Open + Short de-embedding chain** *(Gao §4.2)*  \n"
            "Applied at every frequency point independently.")
        st.markdown("**Step 1 — Open (parallel pad subtraction):**  \n"
                    "S → Y,  then Y − Y_open:")
        st.latex(
            r"Y_1 = Y_{DUT} - Y_{pad},\quad "
            r"Y_{pad}=\begin{bmatrix}Y_{pbe}+Y_{pbc}&-Y_{pbc}\\-Y_{pbc}&Y_{pce}+Y_{pbc}\end{bmatrix}")
        st.markdown("where $Y_{pXX} = j\\omega C_{pXX}$ (extended: Parallel L, Series L, or Series R).")
        st.markdown("**Step 2 — Short (series lead subtraction):**  \n"
                    "Y → Z,  then Z − Z_short,  then Z → Y:")
        st.latex(
            r"Z_2 = Z_1 - Z_{ser},\quad "
            r"Z_{ser}=\begin{bmatrix}Z_b+Z_e&Z_e\\Z_e&Z_c+Z_e\end{bmatrix}")
        st.markdown(
            r"where $Z_b = R_b + j\omega L_b$, etc. (Rb/Rc/Re here come from the Short dummy only — "
            r"access-resistance correction is applied later in Step 3).")

    st.markdown(
        "**Raw** = measured DUT (no de-embedding).  \n"
        "**OS de-embedded** = Open+Short parasitics removed using Step 1a/1b "
        "extracted values (Cpbe/Cpce/Cpbc + Lb/Lc/Le + short-dummy Rs).")

    fT_os, fmax_os = _compare_bode_smith(
        S_a=S_raw, S_b=S_step1, freq=freq, fname=fname, key_suffix="osdeemb",
        label_a="Raw", label_b="OS de-embedded",
        color_a="#2ca02c", color_b="#1f77b4",
        smith_meas_label="Raw", smith_sim_label="OS de-embedded",
        gain_title="Gain vs Frequency — Raw vs OS De-embedded")

    st.markdown(
        "<div style='background:linear-gradient(90deg,#e8f5e9 0%,transparent 100%);"
        "border-left:4px solid #2e7d32;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin:8px 0 2px 0'><strong>📥 Download Open/Short De-embedded S2P</strong></div>",
        unsafe_allow_html=True)
    p_hdr = _rlc_params_summary(para_step1, fname)
    p_hdr["fT"]   = f"{fT_os:.3f} GHz"   if fT_os   is not None else "n/a"
    p_hdr["fmax"] = f"{fmax_os:.3f} GHz" if fmax_os is not None else "n/a"
    st.download_button(
        "📥 OSdeembedded_*.s2p",
        data=write_s2p(freq, S_step1,
                       title=f"OS de-embedded — {Path(fname).stem}",
                       params=p_hdr),
        file_name=f"OSdeembedded_{Path(fname).stem}.s2p",
        mime="text/plain",
        key=f"dl_osdeemb_{fname}",
        width="stretch")

    return S_step1


def render_intrinsic_preview(S_raw, freq, z0, para_step1, para_eff, fname,
                              S_step1=None):
    """
    Step 3 footer — Intrinsic preview.

    Shows Open/Short de-embedded (Step 2) vs Intrinsic (Open/Short + access
    resistance Rb/Rc/Re removed via para_eff).  Provides bode+smith
    comparison and a download of the intrinsic S2P file.
    """
    from .helpers import y_to_s_batch, write_s2p

    st.markdown(
        "<div style='background:linear-gradient(90deg,#0d737722 0%,transparent 100%);"
        "border-left:4px solid #0d7377;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>📊 OS De-embedded vs Intrinsic</strong></div>",
        unsafe_allow_html=True)

    if S_step1 is None:
        Y_step1 = peel_parasitics(S_raw, freq, z0, para_step1)
        S_step1 = y_to_s_batch(Y_step1, z0)

    Y_pareff = peel_parasitics(S_raw, freq, z0, para_eff)
    S_pareff = y_to_s_batch(Y_pareff, z0)

    st.markdown(
        "**OS de-embedded** = Open/Short calibration only (Step 2).  \n"
        "**Intrinsic** = OS calibration + access-resistance Rb/Rc/Re removal "
        "using the values selected in Pre-Extraction Review above.")

    fT_in, fmax_in = _compare_bode_smith(
        S_a=S_step1, S_b=S_pareff, freq=freq, fname=fname, key_suffix="intrinsic",
        label_a="OS de-embedded", label_b="Intrinsic",
        color_a="#1f77b4", color_b="#e67e22",
        smith_meas_label="OS de-embedded", smith_sim_label="Intrinsic",
        gain_title="Gain vs Frequency — OS De-embedded vs Intrinsic")

    st.markdown(
        "<div style='background:linear-gradient(90deg,#0d737722 0%,transparent 100%);"
        "border-left:4px solid #0d7377;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin:8px 0 2px 0'><strong>📥 Download Intrinsic S2P</strong></div>",
        unsafe_allow_html=True)
    p_hdr = _rlc_params_summary(para_eff, fname)
    p_hdr["fT"]   = f"{fT_in:.3f} GHz"   if fT_in   is not None else "n/a"
    p_hdr["fmax"] = f"{fmax_in:.3f} GHz" if fmax_in is not None else "n/a"
    st.download_button(
        "📥 intrinsic_*.s2p",
        data=write_s2p(freq, S_pareff,
                       title=f"Intrinsic — {Path(fname).stem}",
                       params=p_hdr),
        file_name=f"intrinsic_{Path(fname).stem}.s2p",
        mime="text/plain",
        key=f"dl_intrinsic_{fname}",
        width="stretch")


# ════════════════════════════════════════════════════════════════════════════════
# Helper: fT / fmax Bode plot
# ════════════════════════════════════════════════════════════════════════════════

# compute_h21_U, find_ft_fmax, extrap_20dbdec moved to tools/SSM/helpers/metrics.py
# and imported at the top of this file via `from .helpers import …`.


def render_ft_fmax_card(S_mea, S_sim, freq, *, model_name: str,
                        key: str, height: int = 560):
    """
    Compact two-trace fT/fmax mini-plot for a single model.

    Designed to sit beside the per-model Smith chart in a 2-column layout
    (see `render_smith_with_ftfmax` in models/base_ui.py).  Shows |h21|² and
    Mason U for both measured and modeled, draws markers on measurement,
    dashed/long-dashed on the model, and adds a 20 dB/dec dotted extrapolation
    when either trace is still above 0 dB at the highest measured frequency.
    Auto-extends the x-axis past the projected fT / fmax.

    The legend reports fT / fmax for measured and modeled (with "extrap"
    annotation if those values came from the 20 dB/dec projection).
    """
    f_ghz = np.asarray(freq) * 1e-9
    h21_m, U_m = compute_h21_U(S_mea)
    h21_s, U_s = compute_h21_U(S_sim)

    # In-band 0-dB crossings (None if the trace doesn't cross within the band)
    fT_m_in,  fmax_m_in  = find_ft_fmax(f_ghz, h21_m, U_m)
    fT_s_in,  fmax_s_in  = find_ft_fmax(f_ghz, h21_s, U_s)

    fig = go.Figure()
    f_high_track = float(f_ghz[-1])
    extrap_used  = False

    def _meas_label(name, in_val, ext_val):
        if in_val is not None:
            return f"{name} = {in_val:.1f} GHz"
        if ext_val is not None:
            return f"{name} ≈ {ext_val:.1f} GHz (extrap)"
        return f"{name} = n/a"

    # ── Measured |h21|² ───────────────────────────────────────────────────
    f_ext, g_ext, fT_m_ext = extrap_20dbdec(f_ghz, h21_m)
    if f_ext is not None:
        extrap_used = True
        f_high_track = max(f_high_track, fT_m_ext)
    fig.add_trace(go.Scatter(
        x=f_ghz, y=h21_m, mode="lines+markers",
        name=f"|h21|² Meas. ({_meas_label('fT', fT_m_in, fT_m_ext)})",
        line=dict(color="#1f77b4", width=1.4),
        marker=dict(symbol="circle", size=6, color="#1f77b4")))
    if f_ext is not None:
        fig.add_trace(go.Scatter(
            x=f_ext, y=g_ext, mode="lines",
            name="|h21|² Meas. extrap",
            line=dict(color="#1f77b4", width=1.4, dash="dot"),
            showlegend=False))

    # ── Measured Mason U ──────────────────────────────────────────────────
    f_ext, g_ext, fmax_m_ext = extrap_20dbdec(f_ghz, U_m)
    if f_ext is not None:
        extrap_used = True
        f_high_track = max(f_high_track, fmax_m_ext)
    fig.add_trace(go.Scatter(
        x=f_ghz, y=U_m, mode="lines+markers",
        name=f"Mason U Meas. ({_meas_label('fmax', fmax_m_in, fmax_m_ext)})",
        line=dict(color="#1f77b4", width=1.4),
        marker=dict(symbol="square", size=6, color="#1f77b4")))
    if f_ext is not None:
        fig.add_trace(go.Scatter(
            x=f_ext, y=g_ext, mode="lines",
            name="Mason U Meas. extrap",
            line=dict(color="#1f77b4", width=1.4, dash="dot"),
            showlegend=False))

    # ── Modeled |h21|² ────────────────────────────────────────────────────
    f_ext, g_ext, fT_s_ext = extrap_20dbdec(f_ghz, h21_s)
    if f_ext is not None:
        extrap_used = True
        f_high_track = max(f_high_track, fT_s_ext)
    fig.add_trace(go.Scatter(
        x=f_ghz, y=h21_s, mode="lines",
        name=f"|h21|² Model ({_meas_label('fT', fT_s_in, fT_s_ext)})",
        line=dict(color="#d62728", width=2.0, dash="dash")))
    if f_ext is not None:
        fig.add_trace(go.Scatter(
            x=f_ext, y=g_ext, mode="lines",
            name="|h21|² Model extrap",
            line=dict(color="#d62728", width=2.0, dash="dot"),
            showlegend=False))

    # ── Modeled Mason U ───────────────────────────────────────────────────
    f_ext, g_ext, fmax_s_ext = extrap_20dbdec(f_ghz, U_s)
    if f_ext is not None:
        extrap_used = True
        f_high_track = max(f_high_track, fmax_s_ext)
    fig.add_trace(go.Scatter(
        x=f_ghz, y=U_s, mode="lines",
        name=f"Mason U Model ({_meas_label('fmax', fmax_s_in, fmax_s_ext)})",
        line=dict(color="#d62728", width=2.0, dash="longdash")))
    if f_ext is not None:
        fig.add_trace(go.Scatter(
            x=f_ext, y=g_ext, mode="lines",
            name="Mason U Model extrap",
            line=dict(color="#d62728", width=2.0, dash="dot"),
            showlegend=False))

    fig.add_hline(y=0, line_color="#333", line_width=1.2,
                  annotation_text="0 dB", annotation_position="right",
                  annotation_font=dict(size=9))

    x_min = max(float(f_ghz[0]), 1e-2)
    x_max = float(f_high_track) * 1.25 if extrap_used else float(f_ghz[-1])
    fig.update_layout(
        title=dict(text=f"fT / fmax — {model_name}", font=dict(size=12)),
        xaxis=dict(title="Frequency (GHz)", type="log",
                   range=[np.log10(x_min), np.log10(x_max)],
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Gain (dB)", range=[0, 50],
                   showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=height,
        legend=dict(x=0.0, y=-0.15, xanchor="left", yanchor="top",
                    orientation="v",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=9)),
        hovermode="x unified",
        margin=dict(l=55, r=20, t=40, b=180))
    plotly_with_dl(fig, key=key, filename=key)


# ════════════════════════════════════════════════════════════════════════════════
# Matplotlib Smith chart (publication-style, no in-figure legend)
# ════════════════════════════════════════════════════════════════════════════════

_MPL_SMITH_COLORS = {"S11": "#1f77b4", "S12": "#d62728",
                     "S21": "#2ca02c", "S22": "#ff7f0e"}

_MPL_LINE_STYLES   = {"solid": "-", "dashed": "--", "dotted": ":"}
_MPL_MARKER_STYLES = {"circle": "o", "square": "s", "star": "*", "triangle": "^"}


def _draw_mpl_smith_background(ax, line_lw: float, grid_lw: float):
    """Draw the constant-R / constant-X grid for a unit Smith chart on ``ax``.

    Grid (constant-R / constant-X arcs) is drawn first so the black outer-unit
    circle and real-axis line render on top of any grid intersections.
    """
    t = np.linspace(0.0, 2.0*np.pi, 500)
    for r in [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]:
        cx  = r/(r+1.0); rad = 1.0/(r+1.0)
        xc  = cx + rad*np.cos(t); yc = rad*np.sin(t)
        out = (xc**2 + yc**2) > 1.0
        xc[out] = np.nan; yc[out] = np.nan
        ax.plot(xc, yc, color="gray", linewidth=grid_lw, alpha=0.7, zorder=1)
    for x in [0.2, 0.5, 1.0, 2.0, 5.0]:
        for sign in (1, -1):
            xv  = sign*x; rad = 1.0/abs(xv)
            xc  = 1.0 + rad*np.cos(t); yc = (1.0/xv) + rad*np.sin(t)
            out = (xc**2 + yc**2) > 1.0
            xc[out] = np.nan; yc[out] = np.nan
            ax.plot(xc, yc, color="gray", linewidth=grid_lw, alpha=0.7, zorder=1)
    ax.plot(np.cos(t), np.sin(t), color="black", linewidth=line_lw, zorder=3)
    ax.plot([-1.0, 1.0], [0.0, 0.0], color="black",
            linewidth=line_lw*0.8, zorder=3)


_SPARAM_DEFAULT_POS = {
    "S11": ( 0.60, -0.45),
    "S12": ( 0.25,  0.10),
    "S21": (-0.40,  0.45),
    "S22": ( 0.25, -0.20),
}


def render_matplotlib_smith(S_mea=None, S_sim=None, fname: str = "",
                            topo_key: str = "", *,
                            sets=None, default_multiplier=1.0):
    """
    Publication-style Smith chart drawn with matplotlib.

    Two ways to call:
      Backward-compatible (used by SSM extraction):
        render_matplotlib_smith(S_mea, S_sim, fname, topo_key)
            → 2 sets: Measured (dashed line) + Modeled (circle markers).

      Extensible:
        render_matplotlib_smith(fname=..., topo_key=..., sets=[
            {"S": ndarray(N,2,2), "label": "Sim",
             "kind": "line"|"marker", "style": "solid"/"dashed"/"dotted"
                                            or "circle"/"square"/"star"/"triangle"},
            ...
        ], default_multiplier=1.0)

    The four S-param rows have a multiplier, text, x/y position, and color
    that always apply to the corresponding trace in every set (and to the
    text annotation drawn at the chosen x/y).  ``default_multiplier`` may be
    a scalar applied to all four traces, or a dict ``{"S11":…,"S12":…}``.
    """
    import matplotlib.pyplot as plt

    # ── Build sets list when called the legacy way ───────────────────────────
    if sets is None:
        _sets = []
        if S_mea is not None:
            _sets.append({"S": S_mea, "label": "Measured",
                          "kind": "line", "style": "dashed"})
        if S_sim is not None:
            _sets.append({"S": S_sim, "label": "Modeled",
                          "kind": "marker", "style": "circle"})
        sets = _sets
    sets = [s for s in sets if s.get("S") is not None]
    if not sets:
        st.info("No S-parameter data available to plot.")
        return

    if isinstance(default_multiplier, dict):
        default_mults = {sp: float(default_multiplier.get(sp, 1.0))
                         for sp in ("S11", "S12", "S21", "S22")}
    else:
        default_mults = {sp: float(default_multiplier)
                         for sp in ("S11", "S12", "S21", "S22")}

    skey       = f"smith_mpl_{topo_key}_{fname}"
    extra_key  = f"{skey}_n_extra_texts"
    sparams    = ("S11", "S12", "S21", "S22")

    # ── First-time defaults for the 4 fixed S-param rows ─────────────────────
    for sp in sparams:
        x0, y0 = _SPARAM_DEFAULT_POS[sp]
        if f"{skey}_text_{sp}" not in st.session_state:
            st.session_state[f"{skey}_text_{sp}"]  = sp
        if f"{skey}_x_{sp}" not in st.session_state:
            st.session_state[f"{skey}_x_{sp}"]     = float(x0)
        if f"{skey}_y_{sp}" not in st.session_state:
            st.session_state[f"{skey}_y_{sp}"]     = float(y0)
        if f"{skey}_mult_{sp}" not in st.session_state:
            st.session_state[f"{skey}_mult_{sp}"]  = float(default_mults[sp])
        if f"{skey}_color_{sp}" not in st.session_state:
            st.session_state[f"{skey}_color_{sp}"] = _MPL_SMITH_COLORS[sp]

    if extra_key not in st.session_state:
        st.session_state[extra_key] = 0

    # ── Smith chart background thickness ─────────────────────────────────────
    c_smith, c_grid = st.columns(2)
    smith_lw = c_smith.number_input("Smith chart line thickness",
                                    min_value=0.1, max_value=5.0, value=3.0,
                                    step=0.1, format="%.2f",
                                    key=f"{skey}_smith_lw")
    grid_lw  = c_grid.number_input("Smith chart grid thickness",
                                   min_value=0.1, max_value=5.0, value=1.0,
                                   step=0.1, format="%.2f",
                                   key=f"{skey}_grid_lw")

    # ── Per-set: kind / style / size (contextual: line thickness or marker) ──
    st.markdown("**S-parameter set styles** — applies the same symbol or "
                "line to all four S-params within a set")
    for si, s in enumerate(sets):
        kind_sk  = f"{skey}_set{si}_kind"
        style_sk = f"{skey}_set{si}_style"
        size_sk  = f"{skey}_set{si}_size"
        if kind_sk not in st.session_state:
            st.session_state[kind_sk] = ("Line" if s.get("kind") == "line"
                                         else "Markers")
        if style_sk not in st.session_state:
            st.session_state[style_sk] = s.get(
                "style", "dashed" if s.get("kind") == "line" else "circle")

        cc1, cc2, cc3, cc_lbl, cc_in = st.columns([1.2, 1.2, 0.7, 0.7, 0.7])
        cc1.markdown(f"**{s.get('label', f'Set {si+1}')}**")
        kind = cc2.radio(f"Kind {si+1}", ["Markers", "Line"],
                         horizontal=True, key=kind_sk,
                         label_visibility="collapsed")
        if kind == "Line":
            opts = list(_MPL_LINE_STYLES.keys())
            if st.session_state[style_sk] not in opts:
                st.session_state[style_sk] = "solid"
            size_default, size_max = 2.0, 10.0
            size_label = "Line thickness"
        else:
            opts = list(_MPL_MARKER_STYLES.keys())
            if st.session_state[style_sk] not in opts:
                st.session_state[style_sk] = "circle"
            size_default, size_max = 6.0, 30.0
            size_label = "Marker size"
        if size_sk not in st.session_state:
            st.session_state[size_sk] = float(size_default)
        cc3.selectbox(f"Style {si+1}", opts, key=style_sk,
                      label_visibility="collapsed")
        cc_lbl.markdown(f"<div style='padding-top:6px;text-align:right'>"
                        f"{size_label}:</div>",
                        unsafe_allow_html=True)
        cc_in.number_input(f"{size_label} {si+1}",
                           min_value=0.1, max_value=size_max,
                           step=0.1, format="%.2f", key=size_sk,
                           label_visibility="collapsed")

    # ── Per-S-param row: multiplier / text / x / y / color ───────────────────
    st.markdown("**Per-trace controls** — multiplier and color are linked "
                "to both the trace and the text annotation")
    for sp in sparams:
        c_mult, c_text, c_x, c_y, c_color = st.columns([1, 2, 1, 1, 1])
        c_mult.number_input(f"Multiplier {sp}", step=0.1, format="%.3f",
                            key=f"{skey}_mult_{sp}")
        c_text.text_input(f"{sp} text", key=f"{skey}_text_{sp}")
        c_x.number_input(f"{sp} x position", step=0.05, format="%.3f",
                         key=f"{skey}_x_{sp}")
        c_y.number_input(f"{sp} y position", step=0.05, format="%.3f",
                         key=f"{skey}_y_{sp}")
        c_color.color_picker(f"{sp} color", key=f"{skey}_color_{sp}")

    # ── Free text annotations + ➕ button ─────────────────────────────────
    n_extra = int(st.session_state.get(extra_key, 0))
    if n_extra > 0:
        st.markdown("**Extra text annotations**")
    for i in range(n_extra):
        st.session_state.setdefault(f"{skey}_etext_{i}",  "")
        st.session_state.setdefault(f"{skey}_ex_{i}",     0.0)
        st.session_state.setdefault(f"{skey}_ey_{i}",     0.0)
        st.session_state.setdefault(f"{skey}_ecolor_{i}", "#000000")
        ec1, ec2, ec3, ec4 = st.columns([2, 1, 1, 1])
        ec1.text_input(f"Text {i+1}",     key=f"{skey}_etext_{i}")
        ec2.number_input(f"x {i+1}", step=0.05, format="%.3f",
                         key=f"{skey}_ex_{i}")
        ec3.number_input(f"y {i+1}", step=0.05, format="%.3f",
                         key=f"{skey}_ey_{i}")
        ec4.color_picker(f"Color {i+1}", key=f"{skey}_ecolor_{i}")

    def _add_text_slot():
        st.session_state[extra_key] = int(st.session_state[extra_key]) + 1
    st.button("➕ Add text", key=f"{skey}_add_btn", on_click=_add_text_slot)

    # ── Resolve per-S-param config (linked color/mult/text/position) ─────────
    sparam_cfg = {}
    for sp in sparams:
        sparam_cfg[sp] = {
            "mult":  float(st.session_state[f"{skey}_mult_{sp}"]),
            "color": str(st.session_state[f"{skey}_color_{sp}"]),
            "text":  str(st.session_state[f"{skey}_text_{sp}"] or ""),
            "x":     float(st.session_state[f"{skey}_x_{sp}"]),
            "y":     float(st.session_state[f"{skey}_y_{sp}"]),
        }

    # ── Build the matplotlib figure ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    _draw_mpl_smith_background(ax, smith_lw, grid_lw)

    for si, s in enumerate(sets):
        S = s["S"]
        kind  = st.session_state.get(f"{skey}_set{si}_kind",
                                      "Line" if s.get("kind") == "line"
                                      else "Markers")
        style = st.session_state.get(f"{skey}_set{si}_style",
                                      s.get("style", "dashed"))
        size_val = float(st.session_state.get(
            f"{skey}_set{si}_size", 2.0 if kind == "Line" else 6.0))
        for sp, (r, c) in [("S11", (0, 0)), ("S12", (0, 1)),
                           ("S21", (1, 0)), ("S22", (1, 1))]:
            cfg = sparam_cfg[sp]
            sv  = S[:, r, c] * cfg["mult"]
            color = cfg["color"]
            if kind == "Line":
                ls = _MPL_LINE_STYLES.get(style, "-")
                ax.plot(sv.real, sv.imag, linestyle=ls, color=color,
                        linewidth=size_val)
            else:
                mk = _MPL_MARKER_STYLES.get(style, "o")
                ax.plot(sv.real, sv.imag, marker=mk, color=color,
                        markersize=size_val, markeredgewidth=0,
                        linestyle="None", alpha=0.85)

    # S-param text annotations (color linked to trace)
    for sp in sparams:
        cfg = sparam_cfg[sp]
        if cfg["text"]:
            ax.text(cfg["x"], cfg["y"], cfg["text"],
                    ha="center", va="center", fontsize=12,
                    color=cfg["color"], fontweight="bold", zorder=5)

    # Free text annotations
    for i in range(n_extra):
        txt = str(st.session_state.get(f"{skey}_etext_{i}", "") or "")
        if not txt:
            continue
        ax.text(float(st.session_state.get(f"{skey}_ex_{i}", 0.0)),
                float(st.session_state.get(f"{skey}_ey_{i}", 0.0)),
                txt, ha="center", va="center", fontsize=12,
                color=str(st.session_state.get(f"{skey}_ecolor_{i}", "#000000")),
                fontweight="bold", zorder=5)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_parts = []
    for si, s in enumerate(sets):
        kind  = st.session_state.get(f"{skey}_set{si}_kind",
                                     "Line" if s.get("kind") == "line"
                                     else "Markers")
        style = st.session_state.get(f"{skey}_set{si}_style",
                                     s.get("style", "dashed"))
        legend_parts.append(f"**{s.get('label', f'Set {si+1}')}** "
                            f"({kind.lower()} · {style})")
    color_chips = " · ".join(
        f"<span style='color:{sparam_cfg[sp]['color']}'>**{sp}**</span>"
        for sp in sparams)
    st.markdown(
        "**Legend** — " + " ; ".join(legend_parts) +
        "  |  Trace colors: " + color_chips,
        unsafe_allow_html=True,
    )


def render_ft_fmax_overlay(S_raw, sim_results: dict[str, np.ndarray], freq, fname):
    """
    Bode plot: |h21|² and Mason U for measured + all simulated models.

    Measured traces are drawn as markers (○ = h21, □ = Mason U), modeled traces
    as dashed lines.  When |h21|² and/or U are still above 0 dB at the highest
    measured frequency, a 20 dB/dec extrapolation (dotted, same colour) is added
    so that fT / fmax remain visible.  The x-axis is auto-extended to fit the
    farthest extrapolated 0-dB crossing.

    sim_results: {model_SHORT: S_sim_array}  (None values are skipped)
    """
    f_ghz = freq * 1e-9
    h21_mea, U_mea = compute_h21_U(S_raw)

    fig = go.Figure()
    f_high_track = float(f_ghz[-1])  # Tracks the furthest x we need to show
    extrap_used  = False             # Whether any trace required extrapolation

    # ── Measured traces (markers + line) ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=f_ghz, y=h21_mea, mode="lines+markers",
        name="|h21|² Meas.",
        line=dict(color="#1f77b4", width=1.4),
        marker=dict(symbol="circle", size=6, color="#1f77b4")))
    f_ext, g_ext, f0 = extrap_20dbdec(f_ghz, h21_mea)
    if f_ext is not None:
        extrap_used = True
        f_high_track = max(f_high_track, f0)
        fig.add_trace(go.Scatter(
            x=f_ext, y=g_ext, mode="lines",
            name=f"|h21|² Meas. extrap (fT≈{f0:.1f} GHz)",
            line=dict(color="#1f77b4", width=1.6, dash="dot")))

    fig.add_trace(go.Scatter(
        x=f_ghz, y=U_mea, mode="lines+markers",
        name="Mason U Meas.",
        line=dict(color="#1f77b4", width=1.4),
        marker=dict(symbol="square", size=6, color="#1f77b4")))
    f_ext, g_ext, f0 = extrap_20dbdec(f_ghz, U_mea)
    if f_ext is not None:
        extrap_used = True
        f_high_track = max(f_high_track, f0)
        fig.add_trace(go.Scatter(
            x=f_ext, y=g_ext, mode="lines",
            name=f"Mason U Meas. extrap (fmax≈{f0:.1f} GHz)",
            line=dict(color="#1f77b4", width=1.6, dash="dot")))

    # ── Modeled traces (dashed lines, dotted continuation when extrapolated) ──
    palette = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for (short, S_sim), col in zip(sim_results.items(), palette):
        if S_sim is None:
            continue
        h21_s, U_s = compute_h21_U(S_sim)

        fig.add_trace(go.Scatter(
            x=f_ghz, y=h21_s, mode="lines",
            name=f"|h21|² {short}",
            line=dict(color=col, width=2.0, dash="dash")))
        f_ext, g_ext, f0 = extrap_20dbdec(f_ghz, h21_s)
        if f_ext is not None:
            extrap_used = True
            f_high_track = max(f_high_track, f0)
            fig.add_trace(go.Scatter(
                x=f_ext, y=g_ext, mode="lines",
                name=f"|h21|² {short} extrap (fT≈{f0:.1f} GHz)",
                line=dict(color=col, width=2.0, dash="dot"),
                showlegend=False))

        fig.add_trace(go.Scatter(
            x=f_ghz, y=U_s, mode="lines",
            name=f"Mason U {short}",
            line=dict(color=col, width=2.0, dash="longdash")))
        f_ext, g_ext, f0 = extrap_20dbdec(f_ghz, U_s)
        if f_ext is not None:
            extrap_used = True
            f_high_track = max(f_high_track, f0)
            fig.add_trace(go.Scatter(
                x=f_ext, y=g_ext, mode="lines",
                name=f"Mason U {short} extrap (fmax≈{f0:.1f} GHz)",
                line=dict(color=col, width=2.0, dash="dot"),
                showlegend=False))

    fig.add_hline(y=0, line_color="#333", line_width=1.2,
                  annotation_text="0 dB", annotation_position="right",
                  annotation_font=dict(size=9))

    x_min = max(float(f_ghz[0]), 1e-2)
    x_max = float(f_high_track) * 1.25 if extrap_used else float(f_ghz[-1])
    fig.update_layout(
        title="Gain vs Frequency — Measured vs Modeled",
        xaxis=dict(title="Frequency (GHz)", type="log",
                   range=[np.log10(x_min), np.log10(x_max)],
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Gain (dB)", range=[0, 50],
                   showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=450,
        legend=dict(x=1.01, y=1.0, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=9)),
        hovermode="x unified", margin=dict(l=55, r=20, t=50, b=50))
    plotly_with_dl(fig, key=f"ftfmax_{fname}", filename=f"ftfmax_{fname}")
    cap = ("Measured: ○ = |h21|², □ = Mason U.   Modeled: dashed lines.   "
           "Y-axis fixed 0–50 dB.")
    if extrap_used:
        cap += "   Dotted = 20 dB/dec extrapolation past the measured band."
    st.caption(cap)


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
