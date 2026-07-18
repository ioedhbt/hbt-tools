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
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Internal modules ──────────────────────────────────────────────────────────
from .helpers          import (strict_freq_check, s_to_y,
                                step_open, step_short, peel_parasitics,
                                parse_s2p_bytes, interpolate_s2f,
                                write_s2p, simulate_open, simulate_short,
                                plotly_with_dl,
                                quickset_buttons, apply_pending,
                                info_icon_html,
                                load_cache, cache_path_str, list_fits,
                                export_cache_bytes, import_cache_bytes,
                                delete_fit)
from .ssm_access_resistance  import (_render_cold_hbt,
                                render_rz12_section,
                                render_open_collector_section)
from .ssm_plots        import (render_open_plots, render_short_plots,
                                render_os_deemb_preview,
                                render_intrinsic_preview,
                                render_ft_fmax_overlay)
from .ssm_override     import render_unified_pre_override
from .models           import REGISTRY, DEFAULT_SELECTION   # model registry
from .models.base_ui   import render_interactive_param_groups
import matplotlib.pyplot as plt
import plotly.graph_objects as go



# ════════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════════

# Models exposed in the built-in analytic-extraction flow.  Xu ("XuT") and
# Kun-Yang ("KY") live under the Custom-model section as forward-simulation
# views instead (render_builtin_forward_sim).
BUILTIN_SHORTS = ("T", "pi")


def render_builtin_forward_sim(short, S_raw, freq, z0, fname,
                                show_header: bool = True,
                                show_cache_banner: bool = True):
    """Forward-simulation-only view for a built-in model (Xu / Kun-Yang),
    surfaced under the Custom-model section.

    Unlike the guided built-in extraction, this skips the Open/Short
    de-embedding entirely: parameters are *seeded* from a one-shot guess on
    THIS device (pads forced to zero), then the shared override → simulate →
    Smith/residual/fT-fmax → grid-sweep tuning UI takes over so the user can
    forward-simulate and fit by hand.

    ``show_header=False`` suppresses the "### 🧩 {NAME} — forward simulation"
    markdown — used by RF_simulator.py's fit-mode call, which already shows
    its own "### 🎯 Fit — {NAME}" header immediately above, so the model name
    doesn't render twice back-to-back.

    ``show_cache_banner=False`` is threaded through to
    ``render_override_and_smith`` — RF_simulator.py's fit-mode call renders
    its own compact cache pill in the header row instead of the full banner.
    """
    from .models.base_ui import PAD_SPECS
    ModelClass = REGISTRY[short]
    if show_header:
        st.markdown(
            f"### 🧩 {ModelClass.NAME} — forward simulation"
            " <span class='hbt-help' title='Starting values are guessed from this"
            " device (no Open/Short de-embedding). Edit any parameter, read the"
            " residual, and use the Auto-tuning expander (same grid sweep as the"
            " built-in models) to fit this DUT.'>?</span>",
            unsafe_allow_html=True)
    para_eff = {k: 0.0 for k, *_ in PAD_SPECS}     # no pad parasitics
    try:
        Y_seed = peel_parasitics(S_raw, freq, z0, para_eff)
        params, arrays = ModelClass.extract(Y_seed, freq, 10)
    except Exception as exc:                                  # noqa: BLE001
        st.warning(f"Could not auto-seed from the device ({exc}); "
                   "starting from zeros — set the values manually below.")
        params, arrays = {}, {}
    ModelClass.render_override_and_smith(
        fname, S_raw, freq, z0, para_eff, (params, arrays),
        show_cache_banner=show_cache_banner)


# `_agg` was a dead duplicate of `helpers/deembed_math.py::_agg_arr` —
# defined here but never called within this module.  Removed.


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

    # Default low-freq fit width for Step 2/3 extractions.  Used to be a
    # user-facing slider ("Model low-freq pts") but nobody touched it in
    # practice; 10 is the sweet spot for typical 100 MHz – 110 GHz sweeps.
    # If a future user wants to tweak it, restore the slider here.
    n_low = 10

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
            "margin-bottom:2px'><strong>📌 Open Dummy: Pad Capacitances</strong>"
            f"{info_icon_html('Gao [3] §4.2. Used to extract pad parasitic capacitances from open dummy. Bias-independent.')}"
            "</div>",
            unsafe_allow_html=True)
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
                    sk = f"ov_{dk}_{fname}"
                    apply_pending(sk)
                    col_w.number_input(f"{dk} (fF)", key=sk, format="%.4f", step=0.1)
                    arr_disp = np.asarray(open_arr[dk]) * sc
                    quickset_buttons(container=col_w,
                                      key_prefix=sk,
                                      target_key=sk,
                                      arr_disp=arr_disp,
                                      default_disp=para_open_calc[dk]*sc,
                                      unit="fF",
                                      fmt="%.4g", layout="below")
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
            st.info("No Open dummy uploaded — enter pad capacitances manually (defaults to 0 fF).")
            _OPEN_OV = [("Cpbe",1e15),("Cpce",1e15),("Cpbc",1e15)]
            for dk, _ in _OPEN_OV:
                sk = f"ov_{dk}_{fname}"
                if sk not in st.session_state:
                    st.session_state[sk] = 0.0
            for col_w, (dk, _) in zip(st.columns(3), _OPEN_OV):
                sk = f"ov_{dk}_{fname}"
                apply_pending(sk)
                col_w.number_input(f"{dk} (fF)", key=sk, format="%.4f", step=0.1)
            para_caps_ov = {dk: st.session_state[f"ov_{dk}_{fname}"]/sc
                            for dk, sc in _OPEN_OV}
            para_caps_ov.update({
                "Cpbe_mode": "None", "Cpbe_extra": 0.0,
                "Cpce_mode": "None", "Cpce_extra": 0.0,
                "Cpbc_mode": "None", "Cpbc_extra": 0.0,
            })

        # ── Step 1b — Short dummy ─────────────────────────────────────────────
        st.divider()
        st.markdown(
            "<div style='background:linear-gradient(90deg,#eef0f8 0%,transparent 100%);"
            "border-left:4px solid #3d52a0;padding:8px 14px;border-radius:0 6px 6px 0;"
            "margin-bottom:2px'><strong>📌 Short Dummy: Lead Inductances &amp; Series Resistances</strong>"
            f"{info_icon_html('Gao [3] §4.2. Used to extract lead inductances and series resistance. However, series resistance is more accurately modeled by other methods (Cold, Z-parameter, open-collector).')}"
            "</div>",
            unsafe_allow_html=True)
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
                        sk   = f"ov_{dk}_{fname}"
                        apply_pending(sk)
                        col_w.number_input(f"{dk} ({unit})", key=sk,
                                            format=fmt, step=0.1 if sc==1e12 else 0.01)
                        arr_disp = np.asarray(short_arr[dk]) * sc
                        quickset_buttons(container=col_w,
                                          key_prefix=sk,
                                          target_key=sk,
                                          arr_disp=arr_disp,
                                          default_disp=para_short_calc[dk]*sc,
                                          unit=unit,
                                          fmt="%.4g", layout="below")
            para_short_ov = {dk: st.session_state[f"ov_{dk}_{fname}"]/sc for dk, sc in _SHORT_OV}

            # Enhanced short plots — returns {Cpar_Lb, Cpar_Lc, Cpar_Le}
            render_short_plots(short_arr, para_short_ov, fname, freq=freq)

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
            st.info("No Short dummy uploaded — enter lead inductances manually (defaults to 0 pH). "
                    "Rb/Rc/Re default to 0; use Cold-HBT / Z-parameter / Open-collector / Custom in Section 3 to set them.")
            _LEAD_OV = [("Lb",1e12),("Lc",1e12),("Le",1e12)]
            for dk, _ in _LEAD_OV:
                sk = f"ov_{dk}_{fname}"
                if sk not in st.session_state:
                    st.session_state[sk] = 0.0
            for col_w, (dk, _) in zip(st.columns(3), _LEAD_OV):
                sk = f"ov_{dk}_{fname}"
                apply_pending(sk)
                col_w.number_input(f"{dk} (pH)", key=sk, format="%.3f", step=0.1)
            para_short_ov = {dk: st.session_state[f"ov_{dk}_{fname}"]/sc
                             for dk, sc in _LEAD_OV}
            para_short_ov.update({
                "Rpb": 0.0, "Rpc": 0.0, "Rpe": 0.0,
                "Cpar_Lb": 0.0, "Cpar_Lc": 0.0, "Cpar_Le": 0.0,
            })
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


    with st.expander("🍊 Access Resistance Extraction", expanded=False):

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
                                        re_zparam=rz12_Re,
                                        open_arr=(open_arr if has_open else None),
                                        short_arr=(short_arr if has_short else None),
                                        all_data=all_data)

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
    # One checkbox per built-in model, default from DEFAULT_SELECTION.  The
    # built-in analytic extraction now covers only Cheng's T and π — Xu and
    # Kun-Yang are reached through the Custom-model section's forward-simulation
    # views (see render_builtin_forward_sim).
    builtin = {s: REGISTRY[s] for s in BUILTIN_SHORTS if s in REGISTRY}
    model_cols = st.columns(len(builtin))
    selected_models: list[str] = []
    for col_w, (short, ModelClass) in zip(model_cols, builtin.items()):
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

    from .helpers.fit_cache import get_fit as _get_fit
    from .helpers.fit_cache import get_fit_timestamp as _get_fit_ts

    for i, short in enumerate(selected_models):
        ModelClass = REGISTRY[short]
        if i > 0:
            st.divider()
        st.markdown(f"#### {ModelClass.NAME}")

        # ── Cache-first flow ─────────────────────────────────────────────
        # If a cached fit exists for (file, model) and the user hasn't
        # asked to "Re-extract" this session, SKIP the extraction +
        # interactive section entirely.  The fine-tune section below will
        # then see calc_vals == cached values (no sync overwrite), which
        # is the reliable way to keep the cache visible in the UI.
        cached_fit    = _get_fit(fname, short)
        cached_ts     = _get_fit_ts(fname, short)
        reextract_key = f"reextract_session_{short}_{fname}"
        # Latch the cache-first decision at SESSION ENTRY.  The fine-tune
        # section (base_ui) auto-saves to the cache on every edit; without this
        # latch a cache written mid-session would flip `use_cache` on and hide
        # the interactive section while the user is still adjusting values.  We
        # only take the cache-first (skip-interactive) path when a cache already
        # existed when this Run-SSM session began — i.e. on first extract of the
        # file, not when changing values.  Cleared by Run-SSM / Clear so a fresh
        # session re-evaluates against the on-disk cache.
        entry_key = f"cache_use_on_entry_{short}_{fname}"
        if entry_key not in st.session_state:
            st.session_state[entry_key] = cached_fit is not None
        use_cache     = (st.session_state[entry_key]
                         and cached_fit is not None
                         and not st.session_state.get(reextract_key, False))

        ModelClass.render_step_formulas()

        if use_cache:
            snap_key = f"ssm_cache_params_snap_{short}_{fname}"
            col_msg, col_btn = st.columns([3, 1])
            col_msg.success(f"📌 Using cached fit (saved {cached_ts}). "
                            "Extraction + interactive section skipped — "
                            "values come straight from the cache.  Edit "
                            "them in the fine-tune section below to update "
                            "the cache.")
            if col_btn.button("🔄 Re-extract",
                              key=f"reextract_btn_{short}_{fname}",
                              width="stretch",
                              help="Run extraction and interactive section "
                                   "for this session.  The on-disk cache "
                                   "stays put and will reload next time the "
                                   "file is opened."):
                st.session_state[reextract_key] = True
                st.session_state.pop(snap_key, None)
                st.rerun()
            # Pad keys are owned by Step 1's para_eff — never let a cached
            # pad value (which may legitimately be zero or be stale from a
            # previous-session Step 1 extraction) override the LIVE pad
            # values that flow through render_override_and_smith.  See
            # also the matching `_PAD_KEYS` exclusion in the cache restore
            # and the auto-save block in base_ui.py.
            from .models.base_ui import _PAD_KEYS as _PAD_KEYS_FILTER
            # Snapshot the cached params ONCE per Run-SSM session.  The
            # fine-tune section's auto-save (base_ui) rewrites the on-disk
            # cache on every keystroke; if we re-read `cached_fit` here each
            # render, `calc_vals` (= para_eff + params) would follow those
            # saves with a one-render lag, shifting `_override_ui`'s sync
            # hash and re-syncing — i.e. clobbering the user's NEXT edit
            # back to the just-saved value (the "type every value twice"
            # bug).  A stable per-session snapshot keeps calc_vals constant,
            # so the sync fires once on load and never re-clobbers; live
            # edits live in the fine-tune widgets' own session_state.
            if snap_key not in st.session_state:
                st.session_state[snap_key] = {
                    k: float(v) for k, v in cached_fit.items()
                    if isinstance(v, (int, float))
                    and k not in _PAD_KEYS_FILTER}
            params = dict(st.session_state[snap_key])
            arrays = {}
        else:
            params, arrays = ModelClass.extract(Y_ex1, freq, n_low)

            # Rbe override from Z-param method (if available)
            if rz12_Rbe is not None and "Rbe" in params:
                params["Rbe"] = rz12_Rbe
                st.info(f"Rbe overridden from Re(Z₁₂): **{rz12_Rbe:.4f} Ω**")

            # Interactive parameter-vs-frequency plots — slider updates
            # medians, inputs allow override.  Runs BEFORE the table so
            # the table reflects current overridden values.
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
                # Gao §5.5.2 cold-HBT "Cex" is the extrinsic base–collector
                # cap (between the external base and collector contacts) —
                # i.e. Cbcx in our intrinsic-model nomenclature, NOT Cbex
                # (extrinsic base–emitter).  Earlier versions of this map
                # mis-routed Cex_cold into the Cbex slot; corrected here.
                _cold_map = {
                    "Cbcx": "Cex_cold",
                    "Rbi":  "Rbi_cold",
                    "Cbc":  "Cbc_cold",
                    "Cbe":  "Cbe_cold",
                } if cold_res is not None else None
                params = render_interactive_param_groups(
                    params, arrays, freq, fname, short, ModelClass.PARAM_GROUPS,
                    cold_res=cold_res, cold_param_map=_cold_map,
                    reextract_fn=_reextract_fn,
                    cbex_sweep_fn=_cbex_sweep_fn,
                    all_data=all_data, para_eff=para_eff)

        # Lay the extracted-parameters table side-by-side with the
        # 📐 Full formula trace expander when the model provides one.
        # Each lives inside its own expander so the user can collapse
        # either independently.  Models without a formula trace (e.g.
        # Degachi) render the table at full width to avoid leaving a
        # visually empty right column.
        _has_trace = getattr(ModelClass, "has_formula_trace",
                             lambda: False)()
        if _has_trace:
            _col_tbl, _col_trace = st.columns(2, gap="medium")
            with _col_tbl:
                with st.expander("📁 Extracted Parameters Table", expanded=False):
                    ModelClass.render_results_table(params)
            with _col_trace:
                ModelClass.render_formula_trace()
        else:
            with st.expander("📁 Extracted Parameters Table", expanded=False):
                ModelClass.render_results_table(params)
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
            fname, S_raw, freq, z0, para_eff, extract_results[short],
            show_tuning=False)
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

    # The standalone "Download Modeled DUT S2P" section used to live here;
    # the same download button is now under each model's Measured-vs-Modeled
    # Smith chart, beside the xlsx button (see
    # SSMModelTemplate.render_override_and_smith).

    # ── Parameter summary table ───────────────────────────────────────────────
    _render_summary_table(fname, para_eff, cold_res, extract_results, REGISTRY)

    # ── Persistent fit cache (export / import / inspect) ──────────────────────
    _render_fit_cache_panel(fname)


# ════════════════════════════════════════════════════════════════════════════════
# Private helpers (keep the main function readable)
# ════════════════════════════════════════════════════════════════════════════════

# `_render_s2p_downloads` was deleted — its functionality lives under
# each model's Measured-vs-Modeled Smith chart now (📥 modeled S2P
# button next to ⬇ xlsx, wired in `models/base_ui.py::render_smith_with_ftfmax`).


def _render_summary_table(fname, para_eff, cold_res, extract_results, registry):
    st.divider()

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
            # Key-prefix dispatch FIRST so resistances (R*) never get
            # rescaled to "k-unit" or stripped of their Ω suffix — the
            # purely magnitude-based fallback below mishandled Rbe (no
            # unit), Rbi / Rbc (k-unit) before this fix.
            kl = k.lower()
            if kl.startswith("r"):
                sc_d, unit_d = 1.0, "Ω"
            elif kl.startswith("c"):
                sc_d, unit_d = (1e15, "fF") if av < 1e-12 else (1e12, "pF")
            elif kl.startswith("l"):
                sc_d, unit_d = (1e12, "pH") if av < 1e-9 else (1e9, "nH")
            elif kl.startswith("gm"):
                sc_d, unit_d = 1e3, "mS"
            elif kl.startswith("tau") or kl in ("tbi", "tbe", "tau_b",
                                                "tau_c", "taub", "tauc"):
                sc_d, unit_d = 1e12, "ps"
            elif kl.startswith("alpha") or kl.startswith("beta"):
                sc_d, unit_d = 1.0, ""
            else:
                # Last-resort magnitude heuristic for keys we don't
                # recognise from the prefix alone.
                if av < 1e-12:  sc_d, unit_d = 1e15, "fF"
                elif av < 1e-9: sc_d, unit_d = 1e12, "pH"
                else:           sc_d, unit_d = 1.0,  ""
            rows.append({"Layer":layer,"Symbol":k,"Value":f"{v*sc_d:.4f}","Unit":unit_d})
    with st.expander("📋 Complete Parameter Summary", expanded=False):
        st.caption("Values reflect the **current** state after any pre-extraction "
                   "overrides, fine-tune Smith chart edits, and tuning sweeps.")
        if rows:
            df_sum = pd.DataFrame(rows)
            st.dataframe(df_sum, width="stretch", hide_index=True)
            buf = io.BytesIO(); df_sum.to_csv(buf, index=False)
            st.download_button("📥 Download SSM parameters (CSV)", data=buf.getvalue(),
                file_name=f"SSM_{Path(fname).stem}.csv", mime="text/csv",
                key=f"dl_ssm_{fname}")
        else:
            st.caption("No parameters to summarize yet.")


def _render_fit_cache_panel(fname):
    """Persistent fit-cache UI: inspect entries for this file, export the
    whole cache as JSON, or import a JSON cache snapshot (e.g. to carry
    fits from a local install onto the Streamlit Cloud version)."""
    # Hide the entire cache panel on hosts where caching is disabled
    # (Streamlit Community Cloud ephemeral VMs).  Showing it there is
    # misleading because every write is lost on the next cold start
    # and reads always return empty.  See helpers/fit_cache.py for the
    # detection rule (`/mount/src` presence or `HBT_DISABLE_FIT_CACHE`).
    from .helpers.fit_cache import is_cache_disabled
    if is_cache_disabled():
        return

    with st.expander("💾 Fit cache (persists fine-tuned values across sessions)",
                     expanded=False):
        _dut_stem = Path(fname).stem
        st.caption(
            f"Stored under: `{cache_path_str()}` → "
            f"`fits/{_dut_stem}/{_dut_stem}_<model>.json`. "
            "Set `HBT_FIT_CACHE_DIR` to override. "
            "Each (file, model) pair is one isolated JSON file — a corrupt or "
            "all-zero save for one model can no longer overwrite a sibling.  "
            "Fine-tuned values auto-save here and auto-restore on next open.")

        # ── Per-file inspector ──────────────────────────────────────────────
        fits = list_fits(fname)
        if fits:
            st.markdown(f"**Cached fits for `{Path(fname).name}`:**")
            for short, ts in sorted(fits.items()):
                row1, row2 = st.columns([5, 1])
                row1.text(f"• {short} — saved {ts}  "
                          f"({_dut_stem}_{short}.json)")
                if row2.container(
                        key=f"hbt_danger_del_{short}_"
                            + re.sub(r"[^0-9A-Za-z_-]", "-", fname)
                ).button("🗑️ Delete", key=f"cache_del_{short}_{fname}"):
                    delete_fit(fname, short)
                    st.session_state.pop(f"cache_applied_{short}_{fname}",   None)
                    st.session_state.pop(f"cache_dismissed_{short}_{fname}", None)
                    st.rerun()
        else:
            st.caption(f"No cached fits for `{Path(fname).name}` yet — "
                       "fine-tune any model parameter and it'll be saved here.")

        # ── Global export / import ──────────────────────────────────────────
        st.divider()
        st.markdown("**Whole-cache export / import** "
                    "*(useful for syncing local ↔ Streamlit Cloud, "
                    "or backing up before a config change)*")

        try:
            n_files = len(load_cache())
        except Exception:
            n_files = 0

        col_x, col_i = st.columns(2)
        col_x.download_button(
            f"📤 Export cache ({n_files} file{'s' if n_files != 1 else ''})",
            data=export_cache_bytes(),
            file_name="hbt_fit_cache.json", mime="application/json",
            key=f"cache_export_{fname}", width="stretch",
            disabled=(n_files == 0))

        uploaded = col_i.file_uploader(
            "📥 Import cache (.json)", type=["json"],
            key=f"cache_import_{fname}",
            label_visibility="collapsed")
        if uploaded is not None:
            apply_key = f"cache_imp_applied_{uploaded.name}_{uploaded.size}_{fname}"
            if not st.session_state.get(apply_key):
                try:
                    n_f, n_fit = import_cache_bytes(uploaded.getvalue(),
                                                     merge=True)
                    st.success(f"Imported {n_fit} fit(s) across {n_f} file(s) "
                               "(merged into existing cache).")
                    st.session_state[apply_key] = True
                except ValueError as e:
                    st.error(str(e))
