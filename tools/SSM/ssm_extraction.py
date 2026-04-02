"""
ssm_extraction.py — Main entry point for HBT SSM parameter extraction.

This file is intentionally thin: it orchestrates the UI flow and delegates
all maths to the dedicated modules.  To add a new model:
  1. Create models/mymodel.py implementing AbstractSSMModel.
  2. Register it in models/__init__.py → REGISTRY.
  Nothing else changes here.

Call from your app tab:
    from tools.SSM.ssm_extraction import render_ssm_tab
    render_ssm_tab(fname, S_raw, freq, z0, open_data, short_data, all_data)
"""
from __future__ import annotations
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Internal modules ──────────────────────────────────────────────────────────
from .ssm_core        import strict_freq_check, s_to_y
from .ssm_deembedding  import step1a_open, step1b_short, peel_parasitics
from .ssm_s2p          import (parse_s2p_bytes, interpolate_s2f,
                                write_s2p, simulate_open, simulate_short)
from .ssm_plots        import (render_open_plots, render_short_plots,
                                render_deemb_preview, render_ft_fmax_overlay,
                                render_rz12_section)
from .ssm_override     import render_unified_pre_override, make_topology_fig
from .models           import REGISTRY, DEFAULT_SELECTION   # model registry
from .models.base_ui   import render_interactive_param_groups
import matplotlib.pyplot as plt
import plotly.graph_objects as go



# ════════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════════

def _agg(arr, n0, n1, method, trim_pct=20):
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


def _extract_ui(fname, key, freq, default_frac_lo=0.0, default_frac_hi=0.2):
    """Freq-range + method selector. Returns (n0, n1, method, trim_pct)."""
    N = len(freq)
    f_ghz = freq * 1e-9
    lo_idx = min(int(N * default_frac_lo), N - 2)
    hi_idx = min(int(N * default_frac_hi), N - 1)
    c1, c2, c3 = st.columns([3, 1, 1])
    f_range = c1.slider(
        "Freq range (GHz)", float(f_ghz[0]), float(f_ghz[-1]),
        (float(f_ghz[lo_idx]), float(f_ghz[hi_idx])),
        format="%.2f", key=f"frange_{key}_{fname}")
    method = c2.radio("Method", ["Median", "Trimmed mean"],
                      key=f"method_{key}_{fname}", horizontal=False)
    trim_pct = int(c3.number_input("Trim %", 0, 49, 20,
                                    key=f"trim_{key}_{fname}")) \
               if method == "Trimmed mean" else 20
    n0 = max(0, int(np.searchsorted(f_ghz, f_range[0])))
    n1 = min(N, int(np.searchsorted(f_ghz, f_range[1], side="right")))
    return n0, max(n0 + 1, n1), method, trim_pct


def render_ssm_tab(fname, S_raw, freq, z0, open_data, short_data, all_data=None):
    """
    Render the complete SSM extraction tab for one DUT file.

    Parameters
    ----------
    fname      : str       Unique file identifier (used as session-state namespace).
    S_raw      : ndarray   DUT S-parameters  (N,2,2).
    freq       : ndarray   Frequency array in Hz  (N,).
    z0         : float     Reference impedance (Ω).
    open_data  : tuple     (freq, S, z0) for Open dummy.
    short_data : tuple     (freq, S, z0) for Short dummy.
    all_data   : dict|None {fname: {freq,S_raw,z0}} for multi-bias Z-param section.
    """
    if open_data is None or short_data is None:
        st.warning("⚠️ SSM Extraction requires Device Open & Short dummy files.")
        return
    try:
        strict_freq_check(freq, open_data[0],  "Device Open")
        strict_freq_check(freq, short_data[0], "Device Short")
    except ValueError as e:
        st.error(f"Frequency grid mismatch: {e}"); return

    # ── Decimation ────────────────────────────────────────────────────────────
    original_points = len(freq)
    freq_original   = freq.copy()
    col_info, col_dec = st.columns([2, 1])
    col_info.markdown(f"**Total data points:** {original_points}")
    decimate_factor = col_dec.selectbox("Decimate by:", [1,2,4,8,16,32],
                                         index=0, key=f"decimate_{fname}")
    if decimate_factor > 1:
        freq       = freq[::decimate_factor]
        S_raw      = S_raw[::decimate_factor]
        f_o, S_o, z0_o = open_data
        f_s, S_s, z0_s = short_data
        open_data  = (f_o[::decimate_factor], S_o[::decimate_factor], z0_o)
        short_data = (f_s[::decimate_factor], S_s[::decimate_factor], z0_s)
        st.success(f"✓ Using {len(freq)} points (every {decimate_factor}th from {original_points})")

    st.markdown("---")
    st.markdown("## 🔬 Small-Signal Model (SSM) Parameter Extraction")
    with st.expander("🖼️ Illustration", expanded=False):
        st.image(image="tools/SSM/de_embedding_illus.png")

    with st.expander("ℹ️ Model notation", expanded=False):
        st.markdown(
            "Rb=Rpb, Rc=Rpc, Re=Rpe — same pad resistance, different naming contexts.  \n"
            "Open extra elements (Parallel L / Series L / Series R) affect "
            "de-embedding and all forward simulations.")

    col_nl, _ = st.columns([1, 3])
    n_low = col_nl.slider("Model low-freq pts", 3, 40, 10, key=f"nlow_{fname}",
                           help="Low-frequency points for Step 2/3 model extractions.")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1a — Open dummy
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📌 Step 1a — Open Dummy: Pad Capacitances")
    st.caption("Gao [3] §4.2.  Used to extract pad parasitic capacitances from open dummy. Bias-independent.")
    # Formulas — shown here, implementation is in ssm_deembedding.step1a_open
    c1, c2, c3 = st.columns(3)
    with c1: st.latex(r"C_{pbe}=\mathrm{Im}(Y_{11}^{open}+Y_{12}^{open})/\omega")
    with c2: st.latex(r"C_{pce}=\mathrm{Im}(Y_{22}^{open}+Y_{12}^{open})/\omega")
    with c3: st.latex(r"C_{pbc}=-\mathrm{Im}(Y_{12}^{open})/\omega")

    # ── extraction range ─────────────────────────────────────────────────────
    open_n0, open_n1, open_method, open_trim = _extract_ui(
        fname, "open", freq, default_frac_lo=0.5, default_frac_hi=1.0)

    # ── calculation ──────────────────────────────────────────────────────────
    para_open_calc, open_arr = step1a_open(
        open_data, open_n0, open_n1, open_method, open_trim)
    st.dataframe(pd.DataFrame([
        {"Parameter": k, "Value": f"{para_open_calc[k]*1e15:.4f}", "Unit": "fF", "Description": d}
        for k, d in [("Cpbe","Pad B-E shunt cap"),
                     ("Cpce","Pad C-E shunt cap"),
                     ("Cpbc","Pad B-C shunt cap")]
    ]), use_container_width=True, hide_index=True)

    # Override
    _OPEN_OV = [("Cpbe",1e15),("Cpce",1e15),("Cpbc",1e15)]
    for dk, sc in _OPEN_OV:
        sk = f"ov_{dk}_{fname}"
        if sk not in st.session_state: st.session_state[sk] = para_open_calc[dk]*sc
    with st.expander("✏️ Override Open Capacitances", expanded=False):
        if st.button("↩️ Reset Caps", key=f"rst_caps_{fname}"):
            for dk, sc in _OPEN_OV: st.session_state[f"ov_{dk}_{fname}"] = para_open_calc[dk]*sc
            st.rerun()
        for col_w, (dk, sc) in zip(st.columns(3), _OPEN_OV):
            col_w.number_input(f"{dk} (fF)", key=f"ov_{dk}_{fname}", format="%.4f", step=0.1)
    para_caps_ov = {dk: st.session_state[f"ov_{dk}_{fname}"]/sc for dk, sc in _OPEN_OV}

    # Enhanced open plots — returns {cap: (mode, extra_SI)}
    open_mode_extra = render_open_plots(open_data, para_caps_ov, open_arr, fname)
    for cap, (mode, extra) in open_mode_extra.items():
        para_caps_ov[f"{cap}_mode"]  = mode
        para_caps_ov[f"{cap}_extra"] = extra

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1b — Short dummy
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📌 Step 1b — Short Dummy: Lead Inductances & Series Resistances")
    st.caption("Gao [3] §4.2. Used to extract lead inductances and series resistance. However, series resistance is more accurately modeled by other methods (Cold, Z-parameter, open-collector).")
    col_m2, _ = st.columns([1, 1])
    open_sel    = col_m2.radio("Use open from:", ["measured","modelled"],
                                horizontal=True, key=f"osl_{fname}")
    do_measured = (open_sel == "measured")
    # Formulas — implementation is in ssm_deembedding.step1b_short
    c1, c2, c3 = st.columns(3)
    with c1: st.latex(r"R_e=\mathrm{Re}(Z_{12}^{corr})")
    with c2: st.latex(r"R_b=\mathrm{Re}(Z_{11}^{corr}-Z_{12}^{corr})")
    with c3: st.latex(r"R_c=\mathrm{Re}(Z_{22}^{corr}-Z_{21}^{corr})")

    c1, c2, c3 = st.columns(3)
    with c1: st.latex(r"L_e=\mathrm{Im}(Z_{12}^{corr})/\omega")
    with c2: st.latex(r"L_b=\mathrm{Im}(Z_{11}^{corr}-Z_{12}^{corr})/\omega")
    with c3: st.latex(r"L_c=\mathrm{Im}(Z_{22}^{corr}-Z_{21}^{corr})/\omega")

    # ── extraction range ─────────────────────────────────────────────────────
    short_n0, short_n1, short_method, short_trim = _extract_ui(
        fname, "short", freq, default_frac_lo=0.0, default_frac_hi=0.2)

    # ── calculation ──────────────────────────────────────────────────────────
    para_short_calc, short_arr = step1b_short(
        short_data, open_data[0],
        para_caps_ov["Cpbe"], para_caps_ov["Cpce"], para_caps_ov["Cpbc"],
        open_data, n0=short_n0, n1=short_n1, method=short_method,
        trim_pct=short_trim, measured_open=do_measured,
        Cpbe_mode=para_caps_ov.get("Cpbe_mode","None"),
        Cpbe_extra=para_caps_ov.get("Cpbe_extra",0.0),
        Cpce_mode=para_caps_ov.get("Cpce_mode","None"),
        Cpce_extra=para_caps_ov.get("Cpce_extra",0.0),
        Cpbc_mode=para_caps_ov.get("Cpbc_mode","None"),
        Cpbc_extra=para_caps_ov.get("Cpbc_extra",0.0),
    )
    for w in short_arr.get("warnings", []):
        st.warning(w) if w.startswith("⚠️") else st.info(w)
    st.dataframe(pd.DataFrame([
        {"Parameter": lbl, "Value": f"{para_short_calc[dk]*sc:.4f}", "Unit": unit}
        for dk, lbl, sc, unit in [
            ("Lb","Lb",1e12,"pH"),("Lc","Lc",1e12,"pH"),("Le","Le",1e12,"pH"),
            ("Rpb","Rb (Short)",1.0,"Ω"),("Rpc","Rc (Short)",1.0,"Ω"),("Rpe","Re (Short)",1.0,"Ω"),
        ]
    ]), use_container_width=True, hide_index=True)

    _SHORT_OV = [("Lb",1e12),("Lc",1e12),("Le",1e12),
                 ("Rpb",1.0),("Rpc",1.0),("Rpe",1.0)]
    # Re-init overrides if caps changed (would change Short extraction)
    cap_hash = tuple(round(para_caps_ov[k]*1e18) for k in ["Cpbe","Cpce","Cpbc"])
    if st.session_state.get(f"ov_cap_hash_{fname}") != cap_hash:
        for dk, sc in _SHORT_OV:
            st.session_state[f"ov_{dk}_{fname}"] = para_short_calc[dk]*sc
        st.session_state[f"ov_cap_hash_{fname}"] = cap_hash
    for dk, sc in _SHORT_OV:
        sk = f"ov_{dk}_{fname}"
        if sk not in st.session_state: st.session_state[sk] = para_short_calc[dk]*sc
    with st.expander("✏️ Override Short Lead Values", expanded=False):
        if st.button("↩️ Reset Short", key=f"rst_short_{fname}"):
            for dk, sc in _SHORT_OV: st.session_state[f"ov_{dk}_{fname}"] = para_short_calc[dk]*sc
            st.rerun()
        for row_items in [_SHORT_OV[:3], _SHORT_OV[3:]]:
            for col_w, (dk, sc) in zip(st.columns(3), row_items):
                unit = "pH" if sc==1e12 else "Ω"
                fmt  = "%.3f" if sc==1e12 else "%.4f"
                col_w.number_input(f"{dk} ({unit})", key=f"ov_{dk}_{fname}",
                                   format=fmt, step=0.1 if sc==1e12 else 0.01)
    para_short_ov = {dk: st.session_state[f"ov_{dk}_{fname}"]/sc for dk, sc in _SHORT_OV}

    # Enhanced short plots — returns {Cpar_Lb, Cpar_Lc, Cpar_Le}
    short_cpar = render_short_plots(short_arr, para_short_ov, fname)
    para_short_ov.update(short_cpar)

    para_step1 = {**para_caps_ov, **para_short_ov}

    # ══════════════════════════════════════════════════════════════════════════
    # Z-parameter method (multi-bias)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    with st.expander("📈 Z Parameter Method — Extract Rbe and Re  *(Gao [3] Ch. 5.5.1)*",
                     expanded=False):
        render_rz12_section(all_data or {}, para_step1, fname)
    rz12_Re  = st.session_state.get(f"rz12_Re_{fname}")
    rz12_Rbe = st.session_state.get(f"rz12_Rbe_{fname}")

    # ══════════════════════════════════════════════════════════════════════════
    # Cold-HBT
    # ══════════════════════════════════════════════════════════════════════════
    cold_res = _render_cold_hbt(fname, open_data, para_caps_ov, do_measured, freq)

    # ══════════════════════════════════════════════════════════════════════════
    # S-parameter comparison
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # Unified pre-extraction override
    # ══════════════════════════════════════════════════════════════════════════
    para_eff = render_unified_pre_override(fname, para_step1, cold_res, rz12_Re)
    # Propagate extended open/short params
    for cap in ["Cpbe","Cpce","Cpbc"]:
        para_eff[f"{cap}_mode"]  = para_caps_ov.get(f"{cap}_mode",  "None")
        para_eff[f"{cap}_extra"] = para_caps_ov.get(f"{cap}_extra", 0.0)
    for ck in ["Cpar_Lb","Cpar_Lc","Cpar_Le"]:
        para_eff[ck] = para_short_ov.get(ck, 0.0)

    # ══════════════════════════════════════════════════════════════════════════
    # De-embedded DUT preview (Smith chart + Bode + S2P downloads)
    # ══════════════════════════════════════════════════════════════════════════
    render_deemb_preview(S_raw, freq, z0, para_step1, para_eff, fname)

    # ══════════════════════════════════════════════════════════════════════════
    # Model selection
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔘 Model Selection")
    # One checkbox per registered model, default from DEFAULT_SELECTION
    model_cols = st.columns(len(REGISTRY))
    selected_models: list[str] = []
    for col_w, (short, ModelClass) in zip(model_cols, REGISTRY.items()):
        default = short in DEFAULT_SELECTION
        if col_w.checkbox(ModelClass.NAME, value=default, key=f"sel_{short}_{fname}"):
            selected_models.append(short)

    if not selected_models:
        st.info("Select at least one model above.")
        return

    # De-embed DUT once
    Y_ex1 = peel_parasitics(S_raw, freq, z0, para_eff)

    # ══════════════════════════════════════════════════════════════════════════
    # Extraction loop — each selected model
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📌 Steps 2 & 3 — Model Extraction")

    extract_results: dict[str, tuple] = {}   # short → (params, arrays)

    for short in selected_models:
        ModelClass = REGISTRY[short]
        st.markdown(f"#### {ModelClass.NAME}")

        # Show formulas then run extraction (co-located)
        ModelClass.render_step_formulas()
        params, arrays = ModelClass.extract(Y_ex1, freq, n_low)

        # Rbe override from Z-param method (if available)
        if rz12_Rbe is not None and "Rbe" in params:
            params["Rbe"] = rz12_Rbe
            st.info(f"Rbe overridden from Re(Z₁₂): **{rz12_Rbe:.4f} Ω**")

        # Interactive parameter-vs-frequency plots — slider updates medians, inputs allow override
        # Runs BEFORE the table so the table reflects the current overridden values
        if hasattr(ModelClass, "PARAM_GROUPS"):
            _reextract_fn = None
            if hasattr(ModelClass, "reextract"):
                def _make_fn(_cls, _Y, _f, _n):
                    def _fn(curr_params, changed_group_idx, curr_arrays):
                        return _cls.reextract(_Y, _f, _n,
                                              curr_params, changed_group_idx, curr_arrays)
                    return _fn
                _reextract_fn = _make_fn(ModelClass, Y_ex1, freq, n_low)
            _cold_map = {
                "Cbex": "Cex_cold",
                "Rbi":  "Rbi_cold",
                "Cbc":  "Cbc_cold",
                "Cbe":  "Cbe_cold",
            } if cold_res is not None else None
            params = render_interactive_param_groups(
                params, arrays, freq, fname, short, ModelClass.PARAM_GROUPS,
                cold_res=cold_res, cold_param_map=_cold_map,
                reextract_fn=_reextract_fn)

        # Results table — shows values after interactive slider/override
        ModelClass.render_results_table(params)

        # Degachi-specific diagnostic plots
        if hasattr(ModelClass, "render_diagnostic_plots"):
            ModelClass.render_diagnostic_plots(params, arrays, freq, fname)


        # Full formula trace (collapsible)
        ModelClass.render_formula_trace()

        extract_results[short] = (params, arrays)

    # ══════════════════════════════════════════════════════════════════════════
    # Cold-HBT cross-check
    # ══════════════════════════════════════════════════════════════════════════
    if cold_res is not None and extract_results:
        _render_cold_crosscheck(cold_res, extract_results, REGISTRY)

    # ══════════════════════════════════════════════════════════════════════════
    # Circuit schematics
    # ══════════════════════════════════════════════════════════════════════════
    # st.markdown("---")
    # with st.expander("### 🔌 Circuit Topology Diagrams", expanded=False):
    #     # st.markdown("")
    #     for short in selected_models:
    #         ModelClass = REGISTRY[short]
    #         params, _  = extract_results[short]
    #         st.markdown(f"**{ModelClass.NAME}**")
    #         try:
    #             fig_s = make_topology_fig({**para_eff, **params}, ModelClass.TOPOLOGY_CHAR)
    #             st.pyplot(fig_s, use_container_width=True)
    #             plt.close(fig_s)
    #         except Exception as e:
    #             st.error(f"Schematic error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Smith charts (per model — override + residual)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📡 Measured vs Modeled S-Parameters")
    st.caption("Pad params auto-synced from pre-extraction override. "
               "Use expanders to fine-tune intrinsic/extrinsic values.")

    sim_results: dict[str, np.ndarray | None] = {}
    for short in selected_models:
        ModelClass = REGISTRY[short]
        st.markdown(f"#### {ModelClass.NAME}")
        S_sim = ModelClass.render_override_and_smith(
            fname, S_raw, freq, z0, para_eff, extract_results[short])
        sim_results[short] = S_sim

    # ══════════════════════════════════════════════════════════════════════════
    # fT / fmax overlay
    # ══════════════════════════════════════════════════════════════════════════
    if any(v is not None for v in sim_results.values()):
        st.markdown("---")
        st.markdown("### 📊 fT and fmax — Measured vs Modeled")
        render_ft_fmax_overlay(S_raw, sim_results, freq, fname)

    # ══════════════════════════════════════════════════════════════════════════
    # S2P downloads
    # ══════════════════════════════════════════════════════════════════════════
    _render_s2p_downloads(fname, freq, z0, para_eff, sim_results)

    # ══════════════════════════════════════════════════════════════════════════
    # Parameter summary table
    # ══════════════════════════════════════════════════════════════════════════
    _render_summary_table(fname, para_eff, cold_res, extract_results, REGISTRY)


# ════════════════════════════════════════════════════════════════════════════════
# Private helpers (keep the main function readable)
# ════════════════════════════════════════════════════════════════════════════════

def _render_cold_hbt(fname, open_data, para_caps_ov, do_measured, freq):
    """Cold-HBT extraction UI. Returns cold_res dict or None."""
    st.markdown("---")
    st.markdown("#### 🧊 Cold-HBT Extraction  *(Gao [3] Ch. 5.5.2)*")
    st.caption("Used to extract series/access resistances Re, Rb, Rc. Upload cut-off bias (Vce=0, Vbe≤0) S2P.")
    cold_file = st.file_uploader("Cold HBT S2P", type=["s2p"], key=f"cold_upload_{fname}")
    if cold_file is None:
        return None
    try:
        from .ssm_core import y_to_z, z_to_y
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
            from .ssm_deembedding import build_Y_pad
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


                col_w.plotly_chart(fig, use_container_width=True,
                                   key=f"cold_pfp_{res_key}_{fname}")

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
            st.markdown("---")
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
            st.markdown("---")
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
            st.markdown("---")
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



def _render_cold_crosscheck(cold_res, extract_results, registry):
    st.markdown("---"); st.markdown("**Cold-HBT cross-check:**")
    # Use first available model's Rbi/Cbe/Cbc
    first_params = next(iter(extract_results.values()))[0]
    rows = []
    for sym, hot_key, cv_key, unit, sc in [
        ("Rbi", "Rbi", "Rbi_cold", "Ω",  1.0),
        ("Cbe", "Cbe", "Cbe_cold", "fF", 1e15),
        ("Cbc", "Cbc", "Cbc_cold", "fF", 1e15),
    ]:
        hot = first_params.get(hot_key)
        cv  = cold_res.get(cv_key)
        hs  = f"{hot*sc:.4f}" if hot is not None else "—"
        cs  = f"{cv*sc:.4f}"  if cv  is not None else "—"
        try:    delta = f"{(cv-hot)*sc:+.4f}" if (hot and cv) else "—"
        except: delta = "—"
        rows.append({"Symbol":sym,"Hot (RF)":hs,"Cold-HBT":cs,"Δ":delta,"Unit":unit})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_s2p_downloads(fname, freq, z0, para_eff, sim_results):
    st.markdown("---")
    st.markdown("### 📥 Download Modeled S2P Files")
    st.caption(
        "Forward-simulate Open, Short, and final DUT model.  \n"
        "Files use Touchstone format: `# Hz S DB R 50`.  \n"
        "Header `!` comment lines list all parameter values used.")
    col_d1, col_d2, col_d3 = st.columns(3)

    # ── Open ─────────────────────────────────────────────────────────────────
    with col_d1:
        st.markdown("**Modeled Open dummy**")
        S_open_sim  = simulate_open(para_eff, freq, z0)
        open_params = {}
        for k in ["Cpbe","Cpce","Cpbc"]:
            open_params[k] = f"{para_eff.get(k,0)*1e15:.4f} fF"
            mode = para_eff.get(f"{k}_mode","None")
            if mode != "None":
                extra = para_eff.get(f"{k}_extra", 0.0)
                unit_e = "pH" if "L" in mode else "Ω"
                sc_e   = 1e12 if "L" in mode else 1.0
                open_params[f"{k}_extra"] = f"{mode}: {extra*sc_e:.4f} {unit_e}"
        st.download_button("📥 Open.s2p",
            data=write_s2p(freq, S_open_sim,
                           title=f"Open dummy — {Path(fname).stem}",
                           params=open_params),
            file_name=f"model_open_{Path(fname).stem}.s2p", mime="text/plain",
            key=f"dl_open_{fname}", use_container_width=True)
        st.caption("Y_pad only — no series leads.")

    # ── Short ────────────────────────────────────────────────────────────────
    with col_d2:
        st.markdown("**Modeled Short dummy**")
        S_short_sim  = simulate_short(para_eff, freq, z0)
        short_params = {}
        for k in ["Rpb","Rpc","Rpe"]:
            short_params[{"Rpb":"Rb","Rpc":"Rc","Rpe":"Re"}[k]] = f"{para_eff.get(k,0):.4f} Ω"
        for k in ["Lb","Lc","Le"]:
            short_params[k] = f"{para_eff.get(k,0)*1e12:.4f} pH"
        for k in ["Cpar_Lb","Cpar_Lc","Cpar_Le"]:
            v = para_eff.get(k, 0.0)
            if v > 0: short_params[k] = f"{v*1e15:.4f} fF"
        st.download_button("📥 Short.s2p",
            data=write_s2p(freq, S_short_sim,
                           title=f"Short dummy — {Path(fname).stem}",
                           params=short_params),
            file_name=f"model_short_{Path(fname).stem}.s2p", mime="text/plain",
            key=f"dl_short_{fname}", use_container_width=True)
        st.caption("Y_pad + inv(Z_ser) — terminals shorted.")

    # ── DUT ──────────────────────────────────────────────────────────────────
    with col_d3:
        st.markdown("**Modeled DUT S-parameters**")
        avail = {short: S for short, S in sim_results.items() if S is not None}
        if avail:
            from .models import REGISTRY
            choices = {REGISTRY[s].NAME: s for s in avail}
            chosen_name  = st.selectbox("Model to download:", list(choices.keys()),
                                         key=f"dl_model_sel_{fname}")
            chosen_short = choices[chosen_name]
            S_dut_sim    = avail[chosen_short]
            dut_params   = {}
            for k in ["Cpbe","Cpce","Cpbc"]: dut_params[k] = f"{para_eff.get(k,0)*1e15:.4f} fF"
            for k in ["Rpb","Rpc","Rpe"]:
                dut_params[{"Rpb":"Rb","Rpc":"Rc","Rpe":"Re"}[k]] = f"{para_eff.get(k,0):.4f} Ω"
            for k in ["Lb","Lc","Le"]: dut_params[k] = f"{para_eff.get(k,0)*1e12:.4f} pH"
            st.download_button("📥 DUT.s2p",
                data=write_s2p(freq, S_dut_sim,
                               title=f"DUT {chosen_name} — {Path(fname).stem}",
                               params=dut_params),
                file_name=f"model_dut_{Path(fname).stem}.s2p", mime="text/plain",
                key=f"dl_dut_{fname}", use_container_width=True)
            st.caption("Uses Smith chart fine-tune override values.")
        else:
            st.info("Run at least one model above to enable DUT download.")


def _render_summary_table(fname, para_eff, cold_res, extract_results, registry):
    st.markdown("---"); st.markdown("### 📋 Complete Parameter Summary")
    rows = []
    # Pad
    for sym, key, sc, unit in [
        ("Cpbe","Cpbe",1e15,"fF"),("Cpce","Cpce",1e15,"fF"),("Cpbc","Cpbc",1e15,"fF"),
        ("Lb","Lb",1e12,"pH"),("Lc","Lc",1e12,"pH"),("Le","Le",1e12,"pH"),
        ("Rb (=Rpb)","Rpb",1,"Ω"),("Rc (=Rpc)","Rpc",1,"Ω"),("Re (=Rpe)","Rpe",1,"Ω"),
    ]:
        rows.append({"Layer":"Pad","Symbol":sym,"Value":f"{para_eff[key]*sc:.4f}","Unit":unit})
    # Open extra elements
    for cap in ["Cpbe","Cpce","Cpbc"]:
        mode = para_eff.get(f"{cap}_mode","None")
        if mode != "None":
            extra = para_eff.get(f"{cap}_extra",0.0)
            unit_e = "pH" if "L" in mode else "Ω"; sc_e = 1e12 if "L" in mode else 1.0
            rows.append({"Layer":"Open Extra","Symbol":f"{cap} {mode}","Value":f"{extra*sc_e:.4f}","Unit":unit_e})
    for cap, ck in [("Lb","Cpar_Lb"),("Lc","Cpar_Lc"),("Le","Cpar_Le")]:
        v = para_eff.get(ck,0.0)
        if v > 0: rows.append({"Layer":"Short Extra","Symbol":f"Cpar_{cap}","Value":f"{v*1e15:.4f}","Unit":"fF"})
    # Cold-HBT
    if cold_res:
        for sym, key, sc, unit in [
            ("Rb (Cold)","Rb_cold",1,"Ω"),("Rc (Cold)","Rc_cold",1,"Ω"),("Re (Cold)","Re_cold",1,"Ω"),
            ("Rbi (cold)","Rbi_cold",1,"Ω"),("Cbe (cold)","Cbe_cold",1e15,"fF"),
            ("Cbc (cold)","Cbc_cold",1e15,"fF"),("Cex","Cex_cold",1e15,"fF"),
        ]:
            rows.append({"Layer":"Cold-HBT","Symbol":sym,"Value":f"{cold_res[key]*sc:.4f}","Unit":unit})
    # Per-model
    for short, (params, _) in extract_results.items():
        ModelClass = registry[short]
        layer = ModelClass.NAME
        for k, v in params.items():
            if k.startswith("_") or not isinstance(v, (int,float)): continue
            if not np.isfinite(float(v)): continue
            # choose sensible display scale
            av = abs(float(v))
            if av < 1e-12:  sc_d, unit_d = 1e15, "fF"
            elif av < 1e-9: sc_d, unit_d = 1e12, "pH"
            elif av > 1e2:  sc_d, unit_d = 1e-3, "k-unit"
            else:           sc_d, unit_d = 1.0,  ""
            rows.append({"Layer":layer,"Symbol":k,"Value":f"{float(v)*sc_d:.4f}","Unit":unit_d})
    if rows:
        df_sum = pd.DataFrame(rows)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
        buf = io.BytesIO(); df_sum.to_csv(buf, index=False)
        st.download_button("📥 Download SSM parameters (CSV)", data=buf.getvalue(),
            file_name=f"SSM_{Path(fname).stem}.csv", mime="text/csv",
            key=f"dl_ssm_{fname}")
