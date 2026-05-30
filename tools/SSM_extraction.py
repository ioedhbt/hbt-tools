"""
SSM_extraction.py — Standalone HBT Small-Signal Model (SSM) extraction page.

Previously this lived inside the RF S-Parameter Extraction tool's
"Individual → 🔬 SSM Extraction" sub-tab.  It now has its own portal page so
the extraction workflow is front-and-centre instead of buried under the RF
metrics tabs.

Upload one or more DUT ``.s2p`` / ``.csv`` bias files (and, optionally, the
device-dummy Open/Short ``.s2p`` pair for pad de-embedding), pick a file, and
run the full SSM extraction (Cheng T/π, Xu T, Kun-Yang HEMT).

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.
"""
from __future__ import annotations

__version__ = "6.3"

from pathlib import Path

import streamlit as st

from tools.SSM.main_ssm_extraction import render_ssm_tab
from tools.SSM.helpers import parse_s2p, parse_csv, load_cal


# ─────────────────────────────────────────────────────────────────────────────
st.title(f"🔬 HBT SSM Extraction (v{__version__})")
st.caption("Small-signal model parameter extraction from one or more DUT bias files.")

with st.expander(f"What's new in v{__version__}", expanded=False):
    st.markdown(
        "- 🪧 **SSM extraction is now its own page** — split out of the RF "
        "S-Parameter Extraction tool's Individual tab.  Upload DUT files "
        "directly here\n"
        "- 🧮 **\"Calculated Tau_total and fmax\" expander** under the "
        "measured-vs-modeled fT/fmax card (HBT T/π models): τ_total = "
        "1/(2π f_T) with the τ_B+τ_C (or τ) split and charge-storage "
        "residual, plus calculated f_max = √(f_T/(8π·C_BC·R_bb)) vs the "
        "real f_max — measured **and** modeled — with an Extracted/Custom "
        "C_BC / R_bb toggle\n"
        "- 📈 **Bode extrapolation control** beneath the fT/fmax plot: "
        "pick −20 dB/dec or single-pole (window slider, default = final "
        "5 GHz); τ_total/f_max numbers track the chosen method.  Legend "
        "moved bottom-left inside the plot\n"
        "- ⏱️ **Auto Tuning**: ETA now rolls into m/h past 60 s/60 min, "
        "and a total **\"Evaluated in …\"** run time shows above the best "
        "residual; tuning buttons relabeled (Brute force / Optimized / "
        "Prioritized with CPU/CUDA, backend shown on hover)\n"
        "- 🐞 **Fine-tune cache fix**: values no longer need to be entered "
        "twice — auto-restore runs once and the override widgets keep "
        "ownership of session state thereafter"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — device-dummy Open/Short (optional, for pad de-embedding)
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Device Dummy (Open-Short)")
    st.caption("Optional — supplies the pad parasitics that SSM de-embeds "
               "before extraction.  Without it, pad caps/leads default to 0.")
    sw2 = st.toggle("Enable device-dummy de-embedding", value=False)
    f2o = st.file_uploader("Dev Open",  type=["s2p"], key="ssm_d2o") if sw2 else None
    f2s = st.file_uploader("Dev Short", type=["s2p"], key="ssm_d2s") if sw2 else None
    if sw2 and f2o is not None and f2s is not None:
        st.caption("✅ Pad de-embedding active.")

s2o = load_cal(f2o) if sw2 else None   # (freq, S, z0) or None
s2s = load_cal(f2s) if sw2 else None


# ═════════════════════════════════════════════════════════════════════════════
#  DUT FILE UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
if "ssm_uploader_key" not in st.session_state:
    st.session_state["ssm_uploader_key"] = 0

col_up1, col_up2 = st.columns([4, 1])
with col_up1:
    dut_files = st.file_uploader(
        "Upload DUT .s2p / .csv bias files",
        type=["s2p", "csv"], accept_multiple_files=True,
        key=f"ssm_dut_up_{st.session_state['ssm_uploader_key']}")
with col_up2:
    st.write(""); st.write("")
    if st.button("🗑️ Clear uploads", width="stretch"):
        st.session_state["ssm_uploader_key"] += 1
        st.rerun()

if not dut_files:
    st.info("⬆️ Upload at least one DUT `.s2p` / `.csv` file to begin. "
            "Multiple bias files enable the multi-file Z-parameter and "
            "τ_total reference fits.")
    st.stop()


# ── Parse every uploaded DUT into the minimal all_data shape render_ssm_tab
#    consumes ({fname: {freq, S_raw, z0}}). ──────────────────────────────────
all_data: dict = {}
errors: dict = {}
for f in dut_files:
    try:
        raw = f.getvalue()
        if f.name.lower().endswith(".csv"):
            freq, S_raw, z0 = parse_csv(raw.decode("utf-8", errors="ignore"))
        else:
            freq, S_raw, z0 = parse_s2p(raw)
        all_data[f.name] = {"freq": freq, "S_raw": S_raw, "z0": z0}
    except Exception as e:                       # noqa: BLE001 — surface to UI
        errors[f.name] = str(e)

for fn, msg in errors.items():
    st.error(f"❌ {fn}: {msg}")

if not all_data:
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  FILE SELECTOR + RUN GATE
# ═════════════════════════════════════════════════════════════════════════════
file_names = list(all_data.keys())
n = st.selectbox("Active DUT file", file_names,
                 format_func=lambda s: Path(s).stem, key="ssm_active_file")
d = all_data[n]

run_key = f"ssm_run_{n}"
if not st.session_state.get(run_key, False):
    col_ctr, _, _ = st.columns([1, 2, 2])
    if col_ctr.button("▶ Run SSM Extraction", key=f"ssm_btn_{n}",
                      width="stretch", type="primary"):
        st.session_state[run_key] = True
        # Force the fit-cache restore to re-run on this Run-SSM cycle.
        for _k in list(st.session_state.keys()):
            if (_k.startswith("cache_applied_")
                    or _k.startswith("cache_dismissed_")) and _k.endswith(f"_{n}"):
                del st.session_state[_k]
        st.rerun()
    st.caption("SSM extraction is skipped until activated to keep the page "
               "fast.  Click above to run it for the selected file.")
else:
    if st.button("✕ Clear SSM results", key=f"ssm_clear_{n}",
                 help="Frees cached computation for this file."):
        st.session_state[run_key] = False
        for k in list(st.session_state.keys()):
            if k.endswith(f"_{n}") and k != run_key:
                del st.session_state[k]
        st.rerun()

    render_ssm_tab(n, d["S_raw"], d["freq"], d["z0"],
                   s2o, s2s, all_data=all_data)
