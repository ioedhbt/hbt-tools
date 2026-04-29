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
from .ssm_deembedding  import (step_open, step_short, peel_parasitics,_render_cold_hbt,
                                render_rz12_section,
                                render_open_collector_section)
from .ssm_s2p          import (parse_s2p_bytes, interpolate_s2f,
                                write_s2p, simulate_open, simulate_short)
from .ssm_plots        import (render_open_plots, render_short_plots,
                                render_os_deemb_preview,
                                render_intrinsic_preview,
                                render_ft_fmax_overlay)
from .ssm_chart_utils  import plotly_with_dl
from .ssm_override     import render_unified_pre_override
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
    has_open  = open_data  is not None
    has_short = short_data is not None
    if not has_open or not has_short:
        st.info("ℹ️ No Open/Short dummy files — pad parasitics defaulted to zero.")
    if has_open:
        try:
            strict_freq_check(freq, open_data[0], "Device Open")
        except ValueError as e:
            st.error(f"Frequency grid mismatch: {e}"); return
    if has_short:
        try:
            strict_freq_check(freq, short_data[0], "Device Short")
        except ValueError as e:
            st.error(f"Frequency grid mismatch: {e}"); return


    # ── Decimation ────────────────────────────────────────────────────────────
    original_points = len(freq)
    freq_original   = freq.copy()
    with st.container(border=True):
        st.caption("⚙️ Data Decimation")
        col_info, col_dec = st.columns([2, 1])
        col_info.markdown(f"**Total data points:** {original_points}")
        decimate_factor = col_dec.selectbox("Decimate by:", [1,2,4,8,16,32],
                                             index=0, key=f"decimate_{fname}")
        if decimate_factor > 1:
            freq       = freq[::decimate_factor]
            S_raw      = S_raw[::decimate_factor]
            if open_data is not None:
                f_o, S_o, z0_o = open_data
                open_data  = (f_o[::decimate_factor], S_o[::decimate_factor], z0_o)
            if short_data is not None:
                f_s, S_s, z0_s = short_data
                short_data = (f_s[::decimate_factor], S_s[::decimate_factor], z0_s)
            st.success(f"✓ Using {len(freq)} points (every {decimate_factor}th from {original_points})")

    st.divider()
    st.markdown("## 🔬 Small-Signal Model (SSM) Parameter Extraction")
    with st.expander("🖼️ Illustration", expanded=False):
        st.image(image="tools/SSM/de_embedding_illus.png")

    col_nl, _ = st.columns([1, 3])
    n_low = col_nl.slider("Model low-freq pts", 3, 40, 10, key=f"nlow_{fname}",
                           help="Low-frequency points for Step 2/3 model extractions.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Pad capacitance and series inductance
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#3d52a022 0%,transparent 100%);"
        "border-left:5px solid #3d52a0;padding:10px 16px;border-radius:0 8px 8px 0;"
        "margin:4px 0'><span style='font-size:1.1em;font-weight:700'>"
        "1 — Pad Capacitance &amp; Series Inductance</span></div>",
        unsafe_allow_html=True)

    with st.expander("📌 Open & Short Dummy De-embedding", expanded=False):
        # ── Step 1a — Open dummy ──────────────────────────────────────────────
        st.markdown(
            "<div style='background:linear-gradient(90deg,#eef0f8 0%,transparent 100%);"
            "border-left:4px solid #3d52a0;padding:8px 14px;border-radius:0 6px 6px 0;"
            "margin-bottom:2px'><strong>📌 Open Dummy: Pad Capacitances</strong></div>",
            unsafe_allow_html=True)
        st.caption("Gao [3] §4.2.  Used to extract pad parasitic capacitances from open dummy. Bias-independent.")
        # Formulas — shown here, implementation is in ssm_deembedding.step_open
        c1, c2, c3 = st.columns(3)
        with c1: st.latex(r"C_{pbe}=\mathrm{Im}(Y_{11}^{open}+Y_{12}^{open})/\omega")
        with c2: st.latex(r"C_{pce}=\mathrm{Im}(Y_{22}^{open}+Y_{12}^{open})/\omega")
        with c3: st.latex(r"C_{pbc}=-\mathrm{Im}(Y_{12}^{open})/\omega")

        if has_open:
            # ── extraction range ─────────────────────────────────────────────
            open_n0, open_n1, open_method, open_trim = _extract_ui(
                fname, "open", freq, default_frac_lo=0.5, default_frac_hi=1.0)

            # ── calculation ──────────────────────────────────────────────────
            para_open_calc, open_arr = step_open(
                open_data, open_n0, open_n1, open_method, open_trim)
            st.dataframe(pd.DataFrame([
                {"Parameter": k, "Value": f"{para_open_calc[k]*1e15:.4f}", "Unit": "fF", "Description": d}
                for k, d in [("Cpbe","Pad B-E shunt cap"),
                             ("Cpce","Pad C-E shunt cap"),
                             ("Cpbc","Pad B-C shunt cap")]
            ]), width="stretch", hide_index=True)

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

            # open_mode_extra = render_open_plots(open_data, para_caps_ov, open_arr, fname)
            # for cap, (mode, extra) in open_mode_extra.items():
            #     para_caps_ov[f"{cap}_mode"]  = mode
            #     para_caps_ov[f"{cap}_extra"] = extra

            # ── Modeled Open dummy S2P download ──────────────────────────────
            S_open_sim_s1 = simulate_open(para_caps_ov, freq, z0)
            _open_hdr = {}
            for k in ["Cpbe","Cpce","Cpbc"]:
                _open_hdr[k] = f"{para_caps_ov.get(k,0)*1e15:.4f} fF"
                _mode = para_caps_ov.get(f"{k}_mode","None")
                if _mode != "None":
                    _extra = para_caps_ov.get(f"{k}_extra", 0.0)
                    _u = "pH" if "L" in _mode else "Ω"
                    _s = 1e12 if "L" in _mode else 1.0
                    _open_hdr[f"{k}_extra"] = f"{_mode}: {_extra*_s:.4f} {_u}"
            st.download_button(
                "📥 model_open_*.s2p",
                data=write_s2p(freq, S_open_sim_s1,
                               title=f"Open dummy — {Path(fname).stem}",
                               params=_open_hdr),
                file_name=f"model_open_{Path(fname).stem}.s2p",
                mime="text/plain",
                key=f"dl_open_s1_{fname}", width="stretch")
            st.caption("Forward-simulated Open dummy from extracted/overridden Cpbe/Cpce/Cpbc.")
        else:
            st.info("No Open dummy — Cpbe, Cpce, Cpbc defaulted to 0 fF.")
            para_caps_ov = {
                "Cpbe": 0.0, "Cpce": 0.0, "Cpbc": 0.0,
                "Cpbe_mode": "None", "Cpbe_extra": 0.0,
                "Cpce_mode": "None", "Cpce_extra": 0.0,
                "Cpbc_mode": "None", "Cpbc_extra": 0.0,
            }

        # ── Step 1b — Short dummy ─────────────────────────────────────────────
        st.divider()
        st.markdown(
            "<div style='background:linear-gradient(90deg,#eef0f8 0%,transparent 100%);"
            "border-left:4px solid #3d52a0;padding:8px 14px;border-radius:0 6px 6px 0;"
            "margin-bottom:2px'><strong>📌 Short Dummy: Lead Inductances &amp; Series Resistances</strong></div>",
            unsafe_allow_html=True)
        st.caption("Gao [3] §4.2. Used to extract lead inductances and series resistance. However, series resistance is more accurately modeled by other methods (Cold, Z-parameter, open-collector).")
        col_m2, _ = st.columns([1, 1])
        if has_open:
            open_sel    = col_m2.radio("Use open from:", ["measured","modelled"],
                                        horizontal=True, key=f"osl_{fname}")
            do_measured = (open_sel == "measured")
        else:
            col_m2.markdown("*Use open from:* ~~measured~~ / **modelled** *(no Open file)*")
            do_measured = False
        # Formulas — implementation is in ssm_deembedding.step_short
        c1, c2, c3 = st.columns(3)
        with c1: st.latex(r"R_e=\mathrm{Re}(Z_{12}^{corr})")
        with c2: st.latex(r"R_b=\mathrm{Re}(Z_{11}^{corr}-Z_{12}^{corr})")
        with c3: st.latex(r"R_c=\mathrm{Re}(Z_{22}^{corr}-Z_{21}^{corr})")

        c1, c2, c3 = st.columns(3)
        with c1: st.latex(r"L_e=\mathrm{Im}(Z_{12}^{corr})/\omega")
        with c2: st.latex(r"L_b=\mathrm{Im}(Z_{11}^{corr}-Z_{12}^{corr})/\omega")
        with c3: st.latex(r"L_c=\mathrm{Im}(Z_{22}^{corr}-Z_{21}^{corr})/\omega")

        if has_short:
            # ── extraction range ──────────────────────────────────────────────
            short_n0, short_n1, short_method, short_trim = _extract_ui(
                fname, "short", freq, default_frac_lo=0.0, default_frac_hi=0.2)

            # ── calculation ───────────────────────────────────────────────────
            para_short_calc, short_arr = step_short(
                short_data, freq if open_data is None else open_data[0],
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
            ]), width="stretch", hide_index=True)

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
            render_short_plots(short_arr, para_short_ov, fname)

            # ── Modeled Short dummy S2P download ─────────────────────────────
            _p_short = {**para_caps_ov, **para_short_ov}
            S_short_sim_s1 = simulate_short(_p_short, freq, z0)
            _short_hdr = {}
            for k in ["Rpb","Rpc","Rpe"]:
                _short_hdr[{"Rpb":"Rb","Rpc":"Rc","Rpe":"Re"}[k]] = f"{_p_short.get(k,0):.4f} Ω"
            for k in ["Lb","Lc","Le"]:
                _short_hdr[k] = f"{_p_short.get(k,0)*1e12:.4f} pH"
            for k in ["Cpar_Lb","Cpar_Lc","Cpar_Le"]:
                v = _p_short.get(k, 0.0)
                if v > 0: _short_hdr[k] = f"{v*1e15:.4f} fF"
            st.download_button(
                "📥 model_short_*.s2p",
                data=write_s2p(freq, S_short_sim_s1,
                               title=f"Short dummy — {Path(fname).stem}",
                               params=_short_hdr),
                file_name=f"model_short_{Path(fname).stem}.s2p",
                mime="text/plain",
                key=f"dl_short_s1_{fname}", width="stretch")
            st.caption("Forward-simulated Short dummy: Y_pad + inv(Z_ser) — terminals shorted.")
        else:
            st.info("No Short dummy — Lb, Lc, Le, Rb, Rc, Re defaulted to 0.")
            para_short_ov = {
                "Lb": 0.0, "Lc": 0.0, "Le": 0.0,
                "Rpb": 0.0, "Rpc": 0.0, "Rpe": 0.0,
                "Cpar_Lb": 0.0, "Cpar_Lc": 0.0, "Cpar_Le": 0.0,
            }
    para_step1 = {**para_caps_ov, **para_short_ov}

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — De-embedded Preview (Open/Short calibration only)
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#2e7d3222 0%,transparent 100%);"
        "border-left:5px solid #2e7d32;padding:10px 16px;border-radius:0 8px 8px 0;"
        "margin:4px 0'><span style='font-size:1.1em;font-weight:700'>"
        "2 — De-embedded Preview</span></div>",
        unsafe_allow_html=True)

    S_step1 = render_os_deemb_preview(S_raw, freq, z0, para_step1, fname)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Series or Access Resistance Extraction (+ Intrinsic preview)
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#0d737722 0%,transparent 100%);"
        "border-left:5px solid #0d7377;padding:10px 16px;border-radius:0 8px 8px 0;"
        "margin:4px 0'><span style='font-size:1.1em;font-weight:700'>"
        "3 — Series / Access Resistance Extraction</span></div>",
        unsafe_allow_html=True)

    # ── Z-parameter method ────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:linear-gradient(90deg,#e0f2f1 0%,transparent 100%);"
        "border-left:4px solid #0d7377;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>📈 Z-Parameter Method</strong> "
        "<span style='font-weight:normal;font-size:0.9em'>*(Gao [3] Ch. 5.5.1)*</span>"
        " for Re </div>",
        unsafe_allow_html=True)
    with st.expander("Z-Parameter Method — Re(Z₁₂) vs 1/IE", expanded=False):
        render_rz12_section(all_data or {}, para_step1, fname)
    rz12_Re  = st.session_state.get(f"rz12_Re_{fname}")
    rz12_Rbe = st.session_state.get(f"rz12_Rbe_{fname}")

    # ── Open-collector method ─────────────────────────────────────────────────
    st.markdown(
        "<div style='background:linear-gradient(90deg,#e0f2f1 0%,transparent 100%);"
        "border-left:4px solid #0d7377;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>📈 Open-Collector Method</strong> "
        "<span style='font-weight:normal;font-size:0.9em'>*(Gao [3] Ch. 5.5.3)*</span>"
        " for Rb, Re, Rc </div>",
        unsafe_allow_html=True)
    with st.expander("Open-Collector Method — Re(Zij) vs 1/IB", expanded=False):
        render_open_collector_section(all_data or {}, para_step1, fname)

    # ── Cold-HBT ──────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:linear-gradient(90deg,#e0f2f1 0%,transparent 100%);"
        "border-left:4px solid #0d7377;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>🧊 Cold-HBT Extraction</strong> "
        "<span style='font-weight:normal;font-size:0.9em'>*(Gao [3] Ch. 5.5.2)*</span>"
        " for Rb and Rc </div>",
        unsafe_allow_html=True)
    with st.expander("Cold-HBT Extraction", expanded=False):
        cold_res = _render_cold_hbt(fname, open_data, para_step1, do_measured, freq,
                                     re_zparam=rz12_Re)

    # ── Unified pre-extraction override (resolves Rb/Rc/Re sources) ──────────
    para_eff = render_unified_pre_override(fname, para_step1, cold_res, rz12_Re)
    # Propagate extended open/short params
    for cap in ["Cpbe","Cpce","Cpbc"]:
        para_eff[f"{cap}_mode"]  = para_caps_ov.get(f"{cap}_mode",  "None")
        para_eff[f"{cap}_extra"] = para_caps_ov.get(f"{cap}_extra", 0.0)
    for ck in ["Cpar_Lb","Cpar_Lc","Cpar_Le"]:
        para_eff[ck] = para_short_ov.get(ck, 0.0)

    # ── Intrinsic preview (OS de-embedded vs Intrinsic + s2p download) ───────
    render_intrinsic_preview(S_raw, freq, z0, para_step1, para_eff, fname,
                              S_step1=S_step1)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Intrinsic Model
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#6a1b9a22 0%,transparent 100%);"
        "border-left:5px solid #6a1b9a;padding:10px 16px;border-radius:0 8px 8px 0;"
        "margin:4px 0'><span style='font-size:1.1em;font-weight:700'>"
        "4 — Intrinsic Model</span></div>",
        unsafe_allow_html=True)

    # ── Model selection ───────────────────────────────────────────────────────
    # st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#f3e5f5 0%,transparent 100%);"
        "border-left:4px solid #6a1b9a;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>🔘 Model Selection</strong></div>",
        unsafe_allow_html=True)
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
    st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#f3e5f5 0%,transparent 100%);"
        "border-left:4px solid #6a1b9a;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>📌 Steps 2 &amp; 3 — Model Extraction</strong></div>",
        unsafe_allow_html=True)

    extract_results: dict[str, tuple] = {}   # short → (params, arrays)

    for i, short in enumerate(selected_models):
        ModelClass = REGISTRY[short]
        if i > 0:
            st.divider()
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
            _cbex_sweep_fn = None
            if hasattr(ModelClass, "sweep_cbex"):
                def _make_sweep_fn(_cls, _Y, _f):
                    def _fn(cbex_SI_array, mask):
                        return _cls.sweep_cbex(_Y, _f, cbex_SI_array, mask)
                    return _fn
                _cbex_sweep_fn = _make_sweep_fn(ModelClass, Y_ex1, freq)
            _cold_map = {
                "Cbex": "Cex_cold",
                "Rbi":  "Rbi_cold",
                "Cbc":  "Cbc_cold",
                "Cbe":  "Cbe_cold",
            } if cold_res is not None else None
            params = render_interactive_param_groups(
                params, arrays, freq, fname, short, ModelClass.PARAM_GROUPS,
                cold_res=cold_res, cold_param_map=_cold_map,
                reextract_fn=_reextract_fn,
                cbex_sweep_fn=_cbex_sweep_fn)
            # st.write("DEBUG params after interactive:", {k: v for k, v in params.items() if k in ["Rbi","Rbe","Cbe","Rbc","Cbc","alpha0","tauB","tauC"]})

        # Results table — shows values after interactive slider/override
        ModelClass.render_results_table(params)

        # Degachi-specific diagnostic plots
        # if hasattr(ModelClass, "render_diagnostic_plots"):
        #     ModelClass.render_diagnostic_plots(params, arrays, freq, fname)


        # Full formula trace (collapsible)
        ModelClass.render_formula_trace()

        extract_results[short] = (params, arrays)

    # ══════════════════════════════════════════════════════════════════════════
    # Smith charts (per model — override + residual)
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#f3e5f5 0%,transparent 100%);"
        "border-left:4px solid #6a1b9a;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>📡 Measured vs Modeled S-Parameters</strong></div>",
        unsafe_allow_html=True)
    st.caption("Pad params auto-synced from pre-extraction override. "
               "Use expanders to fine-tune intrinsic/extrinsic values.")

    sim_results: dict[str, np.ndarray | None] = {}
    for j, short in enumerate(selected_models):
        ModelClass = REGISTRY[short]
        if j > 0:
            st.divider()
        st.markdown(f"#### {ModelClass.NAME}")
        S_sim = ModelClass.render_override_and_smith(
            fname, S_raw, freq, z0, para_eff, extract_results[short])
        sim_results[short] = S_sim

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Review
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#bf360c22 0%,transparent 100%);"
        "border-left:5px solid #bf360c;padding:10px 16px;border-radius:0 8px 8px 0;"
        "margin:4px 0'><span style='font-size:1.1em;font-weight:700'>"
        "5 — Review</span></div>",
        unsafe_allow_html=True)

    # ── fT / fmax overlay (≥2 models) ─────────────────────────────────────────
    _n_with_sim = sum(1 for v in sim_results.values() if v is not None)
    if _n_with_sim >= 2:
        st.divider()
        st.markdown(
            "<div style='background:linear-gradient(90deg,#fbe9e7 0%,transparent 100%);"
            "border-left:4px solid #bf360c;padding:8px 14px;border-radius:0 6px 6px 0;"
            "margin-bottom:2px'><strong>📊 fT and fmax — Measured vs Modeled</strong></div>",
            unsafe_allow_html=True)
        render_ft_fmax_overlay(S_raw, sim_results, freq, fname)

    # ── S2P downloads ─────────────────────────────────────────────────────────
    _render_s2p_downloads(fname, freq, z0, para_eff, sim_results)

    # ── Parameter summary table ───────────────────────────────────────────────
    _render_summary_table(fname, para_eff, cold_res, extract_results, REGISTRY)


# ════════════════════════════════════════════════════════════════════════════════
# Private helpers (keep the main function readable)
# ════════════════════════════════════════════════════════════════════════════════

def _render_s2p_downloads(fname, freq, z0, para_eff, sim_results):
    st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#fbe9e7 0%,transparent 100%);"
        "border-left:4px solid #bf360c;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>📥 Download Modeled DUT S2P</strong></div>",
        unsafe_allow_html=True)
    st.caption(
        "Forward-simulate the final DUT model.  \n"
        "Files use Touchstone format: `# Hz S DB R 50`.  \n"
        "Header `!` comment lines list all parameter values used.  \n"
        "(Open and Short dummy downloads have moved to Section 1.)")

    # ── DUT ──────────────────────────────────────────────────────────────────
    with st.container():
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
                key=f"dl_dut_{fname}", width="stretch")
            st.caption("Uses Smith chart fine-tune override values.")
        else:
            st.info("Run at least one model above to enable DUT download.")


def _render_summary_table(fname, para_eff, cold_res, extract_results, registry):
    st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#fbe9e7 0%,transparent 100%);"
        "border-left:4px solid #bf360c;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>📋 Complete Parameter Summary</strong></div>",
        unsafe_allow_html=True)
    st.caption("Values reflect the **current** state after any pre-extraction "
               "overrides, fine-tune Smith chart edits, and tuning sweeps.")

    # Source of truth for the *current* per-model param dict.  Each model
    # writes this in its own render_override_and_smith() right after the
    # Smith chart is rendered, so it always reflects the latest fine-tune /
    # tuning state.  Falls back to extraction-time params on first run.
    def _live_params(short):
        live = st.session_state.get(f"current_p_{short}_{fname}")
        if live is not None:
            return live
        return extract_results.get(short, ({}, {}))[0]

    # Pad: prefer the *first* model's live param dict (it shares pad keys with
    # all other models via the pre-extraction sync), then fall back to para_eff.
    pad_src = dict(para_eff)
    for short in extract_results:
        live = st.session_state.get(f"current_p_{short}_{fname}")
        if live is not None:
            for pk in ("Cpbe","Cpce","Cpbc","Lb","Lc","Le","Rpb","Rpc","Rpe"):
                if pk in live:
                    pad_src[pk] = live[pk]
            break

    rows = []
    # Pad
    for sym, key, sc, unit in [
        ("Cpbe","Cpbe",1e15,"fF"),("Cpce","Cpce",1e15,"fF"),("Cpbc","Cpbc",1e15,"fF"),
        ("Lb","Lb",1e12,"pH"),("Lc","Lc",1e12,"pH"),("Le","Le",1e12,"pH"),
        ("Rb (=Rpb)","Rpb",1,"Ω"),("Rc (=Rpc)","Rpc",1,"Ω"),("Re (=Rpe)","Rpe",1,"Ω"),
    ]:
        rows.append({"Layer":"Pad","Symbol":sym,
                     "Value":f"{pad_src.get(key, para_eff.get(key, 0.0))*sc:.4f}",
                     "Unit":unit})
    # Open extra elements
    for cap in ["Cpbe","Cpce","Cpbc"]:
        mode = pad_src.get(f"{cap}_mode", para_eff.get(f"{cap}_mode","None"))
        if mode != "None":
            extra = pad_src.get(f"{cap}_extra", para_eff.get(f"{cap}_extra",0.0))
            unit_e = "pH" if "L" in mode else "Ω"; sc_e = 1e12 if "L" in mode else 1.0
            rows.append({"Layer":"Open Extra","Symbol":f"{cap} {mode}","Value":f"{extra*sc_e:.4f}","Unit":unit_e})
    for cap, ck in [("Lb","Cpar_Lb"),("Lc","Cpar_Lc"),("Le","Cpar_Le")]:
        v = pad_src.get(ck, para_eff.get(ck,0.0))
        if v > 0: rows.append({"Layer":"Short Extra","Symbol":f"Cpar_{cap}","Value":f"{v*1e15:.4f}","Unit":"fF"})
    # Cold-HBT
    if cold_res:
        for sym, key, sc, unit in [
            ("Rb (Cold)","Rb_cold",1,"Ω"),("Rc (Cold)","Rc_cold",1,"Ω"),
            ("Rbi (cold)","Rbi_cold",1,"Ω"),("Cbe (cold)","Cbe_cold",1e15,"fF"),
            ("Cbc (cold)","Cbc_cold",1e15,"fF"),("Cex","Cex_cold",1e15,"fF"),
        ]:
            rows.append({"Layer":"Cold-HBT","Symbol":sym,"Value":f"{cold_res[key]*sc:.4f}","Unit":unit})
    # Per-model — pull *current* (post-override / post-tuning) values when
    # available, fall back to extraction-time results otherwise.
    _PAD_KEYS_SET = {"Cpbe","Cpce","Cpbc","Lb","Lc","Le","Rpb","Rpc","Rpe"}
    for short, (extr_params, _) in extract_results.items():
        ModelClass = registry[short]
        layer = ModelClass.NAME
        live_p = _live_params(short)
        # Iterate the union of extracted-param keys and live keys, preferring
        # live values whenever present.
        all_keys = list(extr_params.keys())
        for lk in live_p:
            if lk not in all_keys:
                all_keys.append(lk)
        for k in all_keys:
            if k.startswith("_") or k in _PAD_KEYS_SET:
                continue
            v = live_p.get(k, extr_params.get(k))
            if not isinstance(v, (int, float)):
                continue
            v = float(v)
            if not np.isfinite(v):
                continue
            av = abs(v)
            if av < 1e-12:  sc_d, unit_d = 1e15, "fF"
            elif av < 1e-9: sc_d, unit_d = 1e12, "pH"
            elif av > 1e2:  sc_d, unit_d = 1e-3, "k-unit"
            else:           sc_d, unit_d = 1.0,  ""
            rows.append({"Layer":layer,"Symbol":k,"Value":f"{v*sc_d:.4f}","Unit":unit_d})
    if rows:
        df_sum = pd.DataFrame(rows)
        st.dataframe(df_sum, width="stretch", hide_index=True)
        buf = io.BytesIO(); df_sum.to_csv(buf, index=False)
        st.download_button("📥 Download SSM parameters (CSV)", data=buf.getvalue(),
            file_name=f"SSM_{Path(fname).stem}.csv", mime="text/csv",
            key=f"dl_ssm_{fname}")
