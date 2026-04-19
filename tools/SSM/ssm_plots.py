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

from .ssm_core        import open_elem_Y, s_to_y, y_to_z
from .ssm_deembedding  import peel_parasitics
from .ssm_s2p          import simulate_open
from .models.base_ui   import render_smith_chart, ssm_residual
from .ssm_chart_utils  import plotly_with_dl


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

def render_short_plots(short_arr, para_short, fname=""):
    """
    Render three expanders for Short dummy diagnostics:
      1. Parallel C controls per lead
      2. Lead inductances vs frequency  [0–150 pH, fixed]
      3. Lead series resistances vs frequency

    Returns
    -------
    dict  {Cpar_Lb, Cpar_Lc, Cpar_Le}  in SI Farads.
    """

    # ── 1. Parallel C controls ────────────────────────────────────────────────
    with st.expander("🔧 Short Lead Model — optional parallel capacitance per lead",
                     expanded=False):
        st.markdown(
            "Adds a capacitance **in parallel** with each lead's R+jωL impedance.  \n"
            "Z_lead_eff = (R+jωL) ∥ (1/jωC_par) = (R+jωL) / (1 + jωC_par(R+jωL))  \n"
            "Causes extracted L to appear frequency-dependent (decreasing at high freq).  \n"
            "Default = 0 (disabled)."
        )
        cpar_cols = st.columns(3)
        for col_w, (key, lbl) in zip(cpar_cols, [("Cpar_Lb","Lb"),
                                                   ("Cpar_Lc","Lc"),
                                                   ("Cpar_Le","Le")]):
            ks = f"short_{key}_{fname}"
            if ks not in st.session_state: st.session_state[ks] = 0.0
            col_w.number_input(f"C_par_{lbl} (fF)", min_value=0.0,
                                step=0.1, format="%.3f", key=ks)

    def _get_cpar(lead_key):
        return float(st.session_state.get(f"short_{lead_key}_{fname}", 0.0)) * 1e-15

    cpar_Lb = _get_cpar("Cpar_Lb")
    cpar_Lc = _get_cpar("Cpar_Lc")
    cpar_Le = _get_cpar("Cpar_Le")

    # ── 2. Inductance plot ────────────────────────────────────────────────────
    with st.expander("📊 Short — Lead Inductances vs Frequency", expanded=True):
        fig_ind = go.Figure(); any_neg = False
        for key, lbl, col in [("Lb","Lb","#8e44ad"),
                               ("Lc","Lc","#e67e22"),
                               ("Le","Le","#16a085")]:
            arr_pH = short_arr[key] * 1e12
            val_pH = para_short[key] * 1e12
            if val_pH < 0: any_neg = True
            idx = np.arange(len(arr_pH))
            fig_ind.add_trace(go.Scatter(x=idx, y=arr_pH,
                name=f"{lbl} (per-freq)", line=dict(color=col, width=2), mode="lines"))
            fig_ind.add_trace(go.Scatter(x=[0, len(arr_pH)-1], y=[val_pH, val_pH],
                name=f"{lbl}={val_pH:.2f} pH",
                line=dict(color=col, width=1.8, dash="dash"), mode="lines"))
        fig_ind.add_hline(y=0, line_color="#333", line_width=1.2,
                           annotation_text="0 pH", annotation_position="left",
                           annotation_font=dict(size=9, color="#333"))
        fig_ind.update_layout(title="Lead Inductances",
            xaxis_title="Point index", yaxis_title="Inductance (pH)",
            plot_bgcolor="white", paper_bgcolor="white", height=360,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=9)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_ind.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_ind.update_yaxes(showgrid=True, gridcolor="#ebebeb", range=[0, 150])
        plotly_with_dl(fig_ind, key=f"step1_ind_{fname}", filename=f"short_inductances_{fname}")
        if any_neg:
            st.warning("One or more lead inductances are negative. Use Short Override to correct.")
        st.caption("Range fixed 0–150 pH.")

    # ── 3. Series resistance plot ─────────────────────────────────────────────
    with st.expander("📊 Short — Lead Series Resistances vs Frequency", expanded=False):
        fig_r = go.Figure()
        for key, lbl, col in [("Rpb","Rb","#8e44ad"),
                               ("Rpc","Rc","#e67e22"),
                               ("Rpe","Re","#16a085")]:
            arr_O = short_arr[key]
            val_O = para_short[key]
            fig_r.add_trace(go.Scatter(x=np.arange(len(arr_O)), y=arr_O,
                name=f"{lbl} (per-freq)", line=dict(color=col, width=2), mode="lines"))
            fig_r.add_trace(go.Scatter(x=[0, len(arr_O)-1], y=[val_O, val_O],
                name=f"{lbl}={val_O:.4f} Ω",
                line=dict(color=col, width=1.8, dash="dash"), mode="lines"))
        fig_r.add_hline(y=0, line_color="#aaa", line_width=1)
        fig_r.update_layout(
            title="Lead Series Resistances — Re(Z terms from Short)",
            xaxis_title="Point index", yaxis_title="Resistance (Ω)",
            plot_bgcolor="white", paper_bgcolor="white", height=320,
            legend=dict(x=1.02, y=1.0, xanchor="left", font=dict(size=9)),
            margin=dict(l=55,r=10,t=40,b=45), hovermode="x unified")
        fig_r.update_xaxes(showgrid=True, gridcolor="#ebebeb")
        fig_r.update_yaxes(showgrid=True, gridcolor="#ebebeb")
        plotly_with_dl(fig_r, key=f"step1_res_{fname}", filename=f"short_resistances_{fname}")
        st.caption("Flat curve = clean extraction. Rising with frequency = skin effect or artefact.")

    return {"Cpar_Lb": cpar_Lb, "Cpar_Lc": cpar_Lc, "Cpar_Le": cpar_Le}


# ════════════════════════════════════════════════════════════════════════════════
# Helper: S-parameter comparison plot
# ════════════════════════════════════════════════════════════════════════════════

def render_deemb_preview(S_raw, freq, z0, para_step1, para_eff, fname):
    """
    'De-embedded DUT preview' section — shown before model selection.

    Shows:
      1. Smith chart: raw (markers) vs Step-1 de-embedded (dashed), all 4 S-params
      2. Formula expander: de-embedding transform chain
      3. Bode plot: |h21|² and Mason U for two de-embedding levels
      4. fT / fmax per trace (vertical lines + annotation)
      5. S2P download buttons for each de-embedded level
    """
    from pathlib import Path
    from .ssm_core import y_to_s_batch
    from .ssm_s2p import write_s2p
    from .models.base_ui import smith_scale_controls

    # st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#e8f5e9 0%,transparent 100%);"
        "border-left:4px solid #2e7d32;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>📊 De-embedded DUT Preview</strong></div>",
        unsafe_allow_html=True)

    f_ghz = freq * 1e-9

    # ── Compute both de-embedded S-parameter sets ─────────────────────────────
    Y_step1  = peel_parasitics(S_raw, freq, z0, para_step1)
    S_step1  = y_to_s_batch(Y_step1, z0)

    Y_pareff = peel_parasitics(S_raw, freq, z0, para_eff)
    S_pareff = y_to_s_batch(Y_pareff, z0)

    # ── Formula expander (collapsible) ────────────────────────────────────────
    with st.expander("📐 De-embedding formulas", expanded=False):
        st.markdown(
            "**Full Open + Short de-embedding chain** *(Gao §4.2)*  \n"
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
            r"where $Z_b = R_b + j\omega L_b$, etc. (optional: $C_{par}$ in parallel with each lead).")
        st.markdown(
            "**Output:** $Y_{ex1} = (Z_2)^{-1}$ — model input.  \n"
            "**Smith chart (right):** $Y_{ex1} \\to S_{deemb}$ plotted vs raw S.")

    # ── Bode plot: h21² and Mason U for raw + both de-embedding levels ───────
    h21_raw, U_raw = _compute_h21_U(S_raw)
    h21_s1,  U_s1  = _compute_h21_U(S_step1)
    h21_pe,  U_pe  = _compute_h21_U(S_pareff)
    
    # st.write("diff raw vs step1:", float(np.max(np.abs(h21_raw - h21_s1))), "dB")
    # st.write("para_step1:", para_step1)

    fT_raw,  fmax_raw  = _find_ft_fmax(f_ghz, h21_raw, U_raw)
    fT_s1,   fmax_s1   = _find_ft_fmax(f_ghz, h21_s1, U_s1)
    fT_pe,   fmax_pe   = _find_ft_fmax(f_ghz, h21_pe, U_pe)

    def _ft_lbl(ft, fm):
        parts = []
        if ft   is not None: parts.append(f"fT={ft:.2f} GHz")
        if fm   is not None: parts.append(f"fmax={fm:.2f} GHz")
        return "  |  ".join(parts) if parts else "(0 dB not crossed)"

    fig = go.Figure()
    f_high_track = float(f_ghz[-1])
    extrap_used  = False

    def _add_meas(y_arr, name, color, symbol, gain_kind):
        """Add a measured trace (markers+line) and a 20 dB/dec extrapolation if needed."""
        nonlocal f_high_track, extrap_used
        fig.add_trace(go.Scatter(
            x=f_ghz, y=y_arr, mode="lines+markers", name=name,
            line=dict(color=color, width=1.4),
            marker=dict(symbol=symbol, size=6, color=color)))
        f_ext, g_ext, f0 = extrap_20dbdec(f_ghz, y_arr)
        if f_ext is not None:
            extrap_used = True
            f_high_track = max(f_high_track, f0)
            tag = f"{gain_kind}≈{f0:.1f} GHz"
            fig.add_trace(go.Scatter(
                x=f_ext, y=g_ext, mode="lines",
                name=f"{name} extrap ({tag})",
                line=dict(color=color, width=1.6, dash="dot"),
                showlegend=False))
            return f0
        return None

    # Raw traces (green family) — no de-embedding
    fT_raw_x = _add_meas(h21_raw, f"|h21|² Raw  [{_ft_lbl(fT_raw, None)}]",
                         "#2ca02c", "circle", "fT")
    fm_raw_x = _add_meas(U_raw,   f"Mason U Raw  [{_ft_lbl(None, fmax_raw)}]",
                         "#2ca02c", "square", "fmax")
    # Open/Short de-embedding traces (blue family)
    fT_s1_x = _add_meas(h21_s1, f"|h21|² Open/Short deembedding  [{_ft_lbl(fT_s1, None)}]",
                        "#1f77b4", "circle", "fT")
    fm_s1_x = _add_meas(U_s1,   f"Mason U Open/Short deembedding  [{_ft_lbl(None, fmax_s1)}]",
                        "#1f77b4", "square", "fmax")
    # Full de-embedding traces (orange family) — pre-ext override applied
    fT_pe_x = _add_meas(h21_pe, f"|h21|² Full deembedding  [{_ft_lbl(fT_pe, None)}]",
                        "#e67e22", "circle", "fT")
    fm_pe_x = _add_meas(U_pe,   f"Mason U Full deembedding  [{_ft_lbl(None, fmax_pe)}]",
                        "#e67e22", "square", "fmax")

    # 0 dB line
    fig.add_hline(y=0, line_color="#333", line_width=1.2,
                  annotation_text="0 dB", annotation_position="right",
                  annotation_font=dict(size=9))

    # fT / fmax vertical markers — prefer the in-band crossing, fall back to extrap result
    for lbl, val_meas, val_ext, col, dash in [
        ("fT (Raw)",      fT_raw,  fT_raw_x, "#2ca02c", "dot"),
        ("fmax (Raw)",    fmax_raw, fm_raw_x, "#2ca02c", "dashdot"),
        ("fT (O/S)",      fT_s1,   fT_s1_x, "#1f77b4", "dot"),
        ("fmax (O/S)",    fmax_s1, fm_s1_x, "#1f77b4", "dashdot"),
        ("fT (Full)",     fT_pe,   fT_pe_x, "#e67e22", "dot"),
        ("fmax (Full)",   fmax_pe, fm_pe_x, "#e67e22", "dashdot"),
    ]:
        val = val_meas if val_meas is not None else val_ext
        if val is not None:
            note = lbl if val_meas is not None else f"{lbl} (extrap)"
            fig.add_vline(x=val, line_color=col, line_dash=dash, line_width=1.5,
                          annotation_text=f"{note}={val:.2f}",
                          annotation_position="top left",
                          annotation_font=dict(size=8, color=col))

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

    # ── Shared explanation on top of both plots ──────────────────────────────
    st.markdown(
        "**Raw** = measured DUT (no de-embedding).  \n"
        "**Open/Short deembedding** = Open+Short parasitics removed using Step 1a/1b "
        "extracted values (Cpbe/Cpce/Cpbc + Lb/Lc/Le + Rb/Rc/Re).  \n"
        "**Full deembedding** = same Open+Short chain with Pre-Extraction Review "
        "overrides applied on top.  \n"
        "○ = |h21|² (→ fT).   □ = Mason U (→ fmax).   Y-axis fixed 0–50 dB."
        + ("   Dotted = 20 dB/dec extrapolation past the measured band."
           if extrap_used else ""))

    # ── Two-column layout: Gain (left) | Smith chart (right) ─────────────────
    sc = smith_scale_controls(fname, "deemb")
    col_gain, col_smith = st.columns([1, 1])
    with col_gain:
        st.markdown("**Gain vs Frequency — Raw, Open/Short, Full De-embedding**")
        plotly_with_dl(fig, key=f"bode_deemb_{fname}", filename=f"deemb_gain_{fname}")
    with col_smith:
        st.markdown("**S-Parameters: Raw vs Full Deembedding**")
        err = ssm_residual(S_raw, S_pareff)
        render_smith_chart(S_raw, S_pareff,
                           "Raw vs Full Deembedding",
                           err, sc,
                           key=f"smith_deemb_{fname}",
                           show_title=False,
                           meas_label="Measured",
                           sim_label="Deembedded")

    # ── S2P downloads ─────────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:linear-gradient(90deg,#e8f5e9 0%,transparent 100%);"
        "border-left:4px solid #2e7d32;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin:8px 0 2px 0'><strong>📥 Download De-embedded S2P</strong></div>",
        unsafe_allow_html=True)

    def _rlc_params(p):
        d = {}
        d["DUT_file"] = Path(fname).stem
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

    dl_c1, dl_c2 = st.columns(2)
    with dl_c1:
        st.markdown("**Step-1 de-embedded** *(para_step1)*")
        fT_s1_str   = f"{fT_s1:.3f} GHz"   if fT_s1   is not None else "n/a"
        fmax_s1_str = f"{fmax_s1:.3f} GHz" if fmax_s1 is not None else "n/a"
        p1 = _rlc_params(para_step1)
        p1["fT"]   = fT_s1_str
        p1["fmax"] = fmax_s1_str
        st.download_button("📥 Step-1 de-embedded.s2p",
            data=write_s2p(freq, S_step1,
                           title=f"Step-1 de-embedded — {Path(fname).stem}",
                           params=p1),
            file_name=f"deemb_step1_{Path(fname).stem}.s2p",
            mime="text/plain",
            key=f"dl_deemb_s1_{fname}",
            width="stretch")

    with dl_c2:
        st.markdown("**Pre-extraction override** *(para_eff)*")
        fT_pe_str   = f"{fT_pe:.3f} GHz"   if fT_pe   is not None else "n/a"
        fmax_pe_str = f"{fmax_pe:.3f} GHz" if fmax_pe is not None else "n/a"
        p2 = _rlc_params(para_eff)
        p2["fT"]   = fT_pe_str
        p2["fmax"] = fmax_pe_str
        st.download_button("📥 Pre-ext override.s2p",
            data=write_s2p(freq, S_pareff,
                           title=f"Pre-ext override — {Path(fname).stem}",
                           params=p2),
            file_name=f"deemb_preext_{Path(fname).stem}.s2p",
            mime="text/plain",
            key=f"dl_deemb_pe_{fname}",
            width="stretch")


# ════════════════════════════════════════════════════════════════════════════════
# Helper: fT / fmax Bode plot
# ════════════════════════════════════════════════════════════════════════════════

def _compute_h21_U(S):
    """Compute |h21|² (dB) and Mason's U (dB) from S-parameters."""
    Y = s_to_y(S, 50.0)
    y11, y12, y21, y22 = Y[:,0,0], Y[:,0,1], Y[:,1,0], Y[:,1,1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h21     = -y21 / (y11 + 1e-30)
        h21_db  = 10.0*np.log10(np.abs(h21)**2 + 1e-30)
        num_u   = np.abs(y21 - y12)**2
        den_u   = 4.0*(y11.real*y22.real - y12.real*y21.real)
        U       = np.where(den_u > 0, num_u/den_u, np.nan)
        U_db    = 10.0*np.log10(np.abs(U) + 1e-30)
    return h21_db, U_db

def _find_ft_fmax(f_ghz, h21_db, U_db):
    """Linear interpolation to find 0 dB crossing (fT from h21², fmax from U)."""
    def _zero_cross(f, arr):
        arr = np.asarray(arr, dtype=float)
        for i in range(len(arr) - 1):
            if np.isfinite(arr[i]) and np.isfinite(arr[i+1]) and arr[i] > 0 >= arr[i+1]:
                slope = arr[i+1] - arr[i]
                return float(f[i] - arr[i] * (f[i+1] - f[i]) / slope)
        return None
    return _zero_cross(f_ghz, h21_db), _zero_cross(f_ghz, U_db)


def extrap_20dbdec(f_ghz, gain_db, n_pts: int = 60):
    """
    20 dB/decade extrapolation of a gain trace beyond its highest measured frequency.

    If `gain_db` is still above 0 at the last finite point, project the trace forward
    along a -20 dB/dec slope (anchored at that last point) until it crosses 0 dB.

    Returns
    -------
    (f_ext, g_ext, f_zero) :
        f_ext   : ndarray  Frequencies (GHz) of the extrapolated segment, starting at
                           the last measured point and ending where g_ext == 0.
        g_ext   : ndarray  Corresponding gain values (dB).
        f_zero  : float    The 0-dB crossing frequency (GHz) — i.e. fT or fmax.
    or  (None, None, None) if the trace already crosses 0 dB inside the measured band
        or if the data is unusable.
    """
    g = np.asarray(gain_db, dtype=float)
    f = np.asarray(f_ghz, dtype=float)
    m = np.isfinite(g) & np.isfinite(f) & (f > 0)
    if not np.any(m):
        return None, None, None
    fv = f[m]; gv = g[m]
    if gv[-1] <= 0:
        return None, None, None
    f_high = float(fv[-1]); g_high = float(gv[-1])
    f_zero = f_high * 10.0 ** (g_high / 20.0)
    if not np.isfinite(f_zero) or f_zero <= f_high:
        return None, None, None
    f_ext = np.logspace(np.log10(f_high), np.log10(f_zero), n_pts)
    g_ext = g_high - 20.0 * np.log10(f_ext / f_high)
    return f_ext, g_ext, f_zero



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
    h21_m, U_m = _compute_h21_U(S_mea)
    h21_s, U_s = _compute_h21_U(S_sim)

    # In-band 0-dB crossings (None if the trace doesn't cross within the band)
    fT_m_in,  fmax_m_in  = _find_ft_fmax(f_ghz, h21_m, U_m)
    fT_s_in,  fmax_s_in  = _find_ft_fmax(f_ghz, h21_s, U_s)

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
    h21_mea, U_mea = _compute_h21_U(S_raw)

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
        h21_s, U_s = _compute_h21_U(S_sim)

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
