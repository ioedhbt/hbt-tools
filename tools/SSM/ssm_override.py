"""
ssm_override.py — Unified pre-extraction parameter override UI.

"""
from __future__ import annotations
import numpy as np
import streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from .models.base_ui import PAD_SPECS


# ════════════════════════════════════════════════════════════════════════════════
# Unified pre-extraction override
# ════════════════════════════════════════════════════════════════════════════════

def render_unified_pre_override(fname, para_step1, cold_res, rz12_Re):
    """
    Show and return the unified pad parameter set used by all models.

    Priority for Rb/Rc/Re: highest value among available sources
    (Short Step 1b, Cold-HBT, Z-parameter method) is pre-selected.

    Returns
    -------
    para_eff : dict   Effective pad parameters (SI units).
    """
    # st.divider()
    st.markdown(
        "<div style='background:linear-gradient(90deg,#e0f2f1 0%,transparent 100%);"
        "border-left:4px solid #0d7377;padding:8px 14px;border-radius:0 6px 6px 0;"
        "margin-bottom:2px'><strong>⚙️ Choose Series Resistance</strong></div>",
        unsafe_allow_html=True)
    # st.caption(
    #     "All pad parameters feeding every model extraction.  \n"
    #     "For Rb/Rc/Re the source with the highest value is pre-selected.")
    para_eff = para_step1.copy()

    # Collect all available resistance sources
    src_Rb = {"Short (Step 1b)": para_step1.get("Rpb", 0.0)}
    src_Rc = {"Short (Step 1b)": para_step1.get("Rpc", 0.0)}
    src_Re = {"Short (Step 1b)": para_step1.get("Rpe", 0.0)}
    if cold_res is not None:
        src_Rb["Cold-HBT"] = float(cold_res.get("Rb_cold", 0.0))
        src_Rc["Cold-HBT"] = float(cold_res.get("Rc_cold", 0.0))
    if rz12_Re is not None:
        src_Re["Z-parameter method"] = float(rz12_Re)
    ocm_Rb = st.session_state.get(f"ocm_Rb_{fname}")
    ocm_Rc = st.session_state.get(f"ocm_Rc_{fname}")
    ocm_Re = st.session_state.get(f"ocm_Re_{fname}")
    if ocm_Rb is not None: src_Rb["Open-collector"] = float(ocm_Rb)
    if ocm_Rc is not None: src_Rc["Open-collector"] = float(ocm_Rc)
    if ocm_Re is not None: src_Re["Open-collector"] = float(ocm_Re)


    def _best(d): return max(d, key=lambda k: d[k])
    def _src_lbl(k, v): return f"{k}: {v:.4f} Ω"

    _cap_keys = ["Cpbe","Cpce","Cpbc"]
    _ind_keys = ["Lb","Lc","Le"]

# Sync preov_ caps/inds whenever upstream ov_ (Open/Short) values change
    _step1_hash = tuple(
        round(para_step1.get(k, 0.0) * 1e18)
        for k in _cap_keys + _ind_keys
    )
    if st.session_state.get(f"preov_step1_hash_{fname}") != _step1_hash:
        for k in _cap_keys:
            st.session_state[f"preov_{k}_{fname}"] = para_step1.get(k, 0.0) * 1e15
        for k in _ind_keys:
            st.session_state[f"preov_{k}_{fname}"] = para_step1.get(k, 0.0) * 1e12
        st.session_state[f"preov_step1_hash_{fname}"] = _step1_hash

    for var, sources in [("Rb",src_Rb),("Rc",src_Rc),("Re",src_Re)]:
        sk_src = f"preov_src_{var}_{fname}"
        sk_val = f"preov_{var}_{fname}"
        if sk_src not in st.session_state:
            st.session_state[sk_src] = _best(sources)
        if sk_val not in st.session_state:
            st.session_state[sk_val] = sources.get(st.session_state[sk_src],
                                                    list(sources.values())[0])

    with st.expander("✏️ Choose series resistance", expanded=False):
        if st.button("↩️ Reset all to defaults (highest source)",
                     key=f"preov_reset_{fname}"):
            for k in _cap_keys:
                st.session_state[f"preov_{k}_{fname}"] = para_step1.get(k,0.0) * 1e15
            for k in _ind_keys:
                st.session_state[f"preov_{k}_{fname}"] = para_step1.get(k,0.0) * 1e12
            for var, sources in [("Rb",src_Rb),("Rc",src_Rc),("Re",src_Re)]:
                bk = _best(sources)
                st.session_state[f"preov_src_{var}_{fname}"] = bk
                st.session_state[f"preov_{var}_{fname}"]     = sources[bk]
            st.rerun()

        # st.markdown("**Pad capacitances** *(from Open, single source)*")
        # for col_w, (k, lbl) in zip(st.columns(3),
        #                             [("Cpbe","Cpbe (fF)"),
        #                              ("Cpce","Cpce (fF)"),
        #                              ("Cpbc","Cpbc (fF)")]):
        #     col_w.number_input(lbl, key=f"preov_{k}_{fname}", format="%.4f", step=0.1)

        # st.markdown("**Lead inductances** *(from Short, single source)*")
        # for col_w, (k, lbl) in zip(st.columns(3),
        #                             [("Lb","Lb (pH)"),
        #                              ("Lc","Lc (pH)"),
        #                              ("Le","Le (pH)")]):
        #     col_w.number_input(lbl, key=f"preov_{k}_{fname}", format="%.3f", step=0.1)

        st.markdown("**Series resistances** *(choose source — default = highest)*")
        for var, sources, label in [
            ("Rb", src_Rb, "**Rb = Rpb** — base"),
            ("Rc", src_Rc, "**Rc = Rpc** — collector"),
            ("Re", src_Re, "**Re = Rpe** — emitter"),
        ]:
            src_opts  = list(sources.keys()) + ["Custom"]
            sk_src    = f"preov_src_{var}_{fname}"
            sk_val    = f"preov_{var}_{fname}"
            cur_src   = st.session_state.get(sk_src, _best(sources))
            if cur_src not in src_opts: cur_src = src_opts[0]
            radio_lbls = [_src_lbl(k, v) for k, v in sources.items()] + ["Custom"]
            cur_idx    = src_opts.index(cur_src)
            st.markdown(label)
            sc1, sc2 = st.columns([3, 1])
            chosen_lbl = sc1.radio(f"Source for {var}", radio_lbls,
                                    index=cur_idx,
                                    key=f"preov_radio_{var}_{fname}",
                                    horizontal=True,
                                    label_visibility="collapsed")
            chosen_src = src_opts[radio_lbls.index(chosen_lbl)]
            st.session_state[sk_src] = chosen_src
            if chosen_src != "Custom":
                resolved = sources[chosen_src]
                st.session_state[sk_val] = resolved
                sc2.metric(var, f"{resolved:.4f} Ω")
            else:
                sc2.number_input(f"{var} (Ω)", key=sk_val, format="%.4f", step=0.01)

    # Assemble para_eff from session state
    for k in _cap_keys:
        para_eff[k] = st.session_state.get(f"preov_{k}_{fname}",
                                            para_step1.get(k,0.0)*1e15) / 1e15
    for k in _ind_keys:
        para_eff[k] = st.session_state.get(f"preov_{k}_{fname}",
                                            para_step1.get(k,0.0)*1e12) / 1e12
    para_eff["Rpb"] = st.session_state.get(f"preov_Rb_{fname}", para_step1.get("Rpb",0.0))
    para_eff["Rpc"] = st.session_state.get(f"preov_Rc_{fname}", para_step1.get("Rpc",0.0))
    para_eff["Rpe"] = st.session_state.get(f"preov_Re_{fname}", para_step1.get("Rpe",0.0))
    return para_eff
