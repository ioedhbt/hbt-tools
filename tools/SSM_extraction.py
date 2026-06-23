"""
SSM_extraction.py — Standalone HBT Small-Signal Model (SSM) extraction page.

Previously this lived inside the RF S-Parameter Extraction tool's
"Individual → 🔬 SSM Extraction" sub-tab.  It now has its own portal page so
the extraction workflow is front-and-centre instead of buried under the RF
metrics tabs.

Upload one or more DUT ``.s2p`` / ``.csv`` bias files (and, optionally, the
device-dummy Open/Short ``.s2p`` pair for pad de-embedding), pick a file, and
run the built-in analytic extraction (Cheng T/π).  Xu T and the Kun-Yang HEMT
are available as forward-simulation views under the Custom-model section.

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.
"""
from __future__ import annotations

__version__ = "7.0"

from pathlib import Path

import streamlit as st

from tools.SSM.main_ssm_extraction import (render_ssm_tab,
                                            render_builtin_forward_sim)
from tools.SSM.helpers import parse_s2p, parse_csv, load_cal
from tools import i18n


# ─────────────────────────────────────────────────────────────────────────────
st.title(i18n.title("ssm"))
st.caption(i18n.tool_desc("ssm"))

with st.expander(f"{i18n.t('whats_new')} · v{__version__}", expanded=False):
    st.markdown(
        "- 🧩 **Custom model** — build *any* small-signal topology, fit it to the "
        "measured device on the same Smith / fT-fmax / residual UI as the built-in "
        "models, and use the grid-sweep Auto Tuning. Import/export `.json`.\n"
        "- 🛠 Generic **netlist→Y→S solver** reproduces Cheng-π/T and Xu to machine "
        "precision.\n\n"
        "See [`CHANGELOG.md`](CHANGELOG.md) for full history."
    )

with st.expander(i18n.t("how_it_works"), expanded=False):
    from tools.diagrams import pipeline_png
    st.image(pipeline_png((
        ("Upload",   "DUT s2p / csv"),
        ("De-embed", "pads"),
        ("Extract",  "Cheng T / π"),
        ("Tune",     "fine + sweep"),
        ("Compare",  "Smith · residual"),
    ), accent="#d62728"), width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — device-dummy Open/Short (optional, for pad de-embedding)
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(i18n.t("dev_dummy_head"))
    sw2 = st.toggle(i18n.t("dev_dummy_toggle"), value=False,
                    help=i18n.t("dev_dummy_help"))
    f2o = st.file_uploader(i18n.t("dev_open"),  type=["s2p"], key="ssm_d2o") if sw2 else None
    f2s = st.file_uploader(i18n.t("dev_short"), type=["s2p"], key="ssm_d2s") if sw2 else None
    if sw2 and f2o is not None and f2s is not None:
        st.caption(i18n.t("pad_deembed_on"))

s2o = load_cal(f2o) if sw2 else None   # (freq, S, z0) or None
s2s = load_cal(f2s) if sw2 else None


# ═════════════════════════════════════════════════════════════════════════════
#  DUT FILE UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
if "ssm_uploader_key" not in st.session_state:
    st.session_state["ssm_uploader_key"] = 0

st.subheader(i18n.t("ssm_step1"))
col_up1, col_up2 = st.columns([4, 1])
with col_up1:
    dut_files = st.file_uploader(
        i18n.t("ssm_upload_label"),
        type=["s2p", "csv"], accept_multiple_files=True,
        key=f"ssm_dut_up_{st.session_state['ssm_uploader_key']}")
with col_up2:
    st.write(""); st.write("")
    if st.button(i18n.t("clear_uploads"), width="stretch"):
        st.session_state["ssm_uploader_key"] += 1
        st.session_state.pop("ssm_use_examples", None)
        st.rerun()

# Uploading your own files always takes precedence over the example mode.
if dut_files:
    st.session_state.pop("ssm_use_examples", None)
use_examples = st.session_state.get("ssm_use_examples", False)

if not dut_files and not use_examples:
    c_info, c_ex = st.columns([4, 1])
    c_info.info(i18n.t("upload_or_example"))
    with c_ex:
        st.write(""); st.write("")
        if st.button(i18n.t("load_example"), width="stretch"):
            st.session_state["ssm_use_examples"] = True
            st.rerun()
    st.stop()


# ── Collect (name, bytes) sources from the uploads or the bundled examples. ──
sources: list[tuple[str, bytes]] = []
if dut_files:
    sources = [(f.name, f.getvalue()) for f in dut_files]
elif use_examples:
    ex_dir = Path(__file__).resolve().parent.parent / "dummy_data_practice"
    sources = [(p.name, p.read_bytes()) for p in sorted(ex_dir.glob("*.s2p"))]
    st.caption(i18n.t("using_example"))

# ── Parse every source into the minimal all_data shape render_ssm_tab
#    consumes ({fname: {freq, S_raw, z0}}). ──────────────────────────────────
all_data: dict = {}
errors: dict = {}
for name, raw in sources:
    try:
        if name.lower().endswith(".csv"):
            freq, S_raw, z0 = parse_csv(raw.decode("utf-8", errors="ignore"))
        else:
            freq, S_raw, z0 = parse_s2p(raw)
        all_data[name] = {"freq": freq, "S_raw": S_raw, "z0": z0}
    except Exception as e:                       # noqa: BLE001 — surface to UI
        errors[name] = str(e)

for fn, msg in errors.items():
    st.error(f"❌ {fn}: {msg}")

if not all_data:
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  FILE SELECTOR + RUN GATE
# ═════════════════════════════════════════════════════════════════════════════
st.subheader(i18n.t("ssm_step2"))
file_names = list(all_data.keys())
n = st.selectbox(i18n.t("active_dut"), file_names,
                 format_func=lambda s: Path(s).stem, key="ssm_active_file")
d = all_data[n]

# ── Model selection — a single flat picker over the built-in extraction and
#    the four custom-model sub-modes (replaces the old two nested radios).  A
#    breadcrumb caption spells out the active mode. ───────────────────────────
_M_BUILTIN = "📐 Built-in fit"
_M_CFIT    = "🔬 Custom · fit"
_M_CBUILD  = "🛠 Custom · build"
_M_XU      = "🧪 Xu's T sim"
_M_KY      = "🧪 Kun-Yang sim"
_MODES = [_M_BUILTIN, _M_CFIT, _M_CBUILD, _M_XU, _M_KY]

# Drop stale values from the old two-radio scheme, seed a default, then honour
# the "Send to Load / Fit" hook fired by the custom-model build view.  All must
# run before the widget is instantiated to set its session value.
for _old in ("ssm_model_kind", "ssm_custom_mode"):
    st.session_state.pop(_old, None)
if st.session_state.get("ssm_mode") not in _MODES:
    st.session_state["ssm_mode"] = _M_BUILTIN
if st.session_state.pop("cm_nav_to_loadfit", False):
    st.session_state["ssm_mode"] = _M_CFIT

mode = st.segmented_control(i18n.t("model_label"), _MODES, key="ssm_mode") \
    or st.session_state["ssm_mode"]

_CRUMB = {
    _M_BUILTIN: i18n.t("crumb_builtin"),
    _M_CFIT:    i18n.t("crumb_cfit"),
    _M_CBUILD:  i18n.t("crumb_cbuild"),
    _M_XU:      i18n.t("crumb_xu"),
    _M_KY:      i18n.t("crumb_ky"),
}
st.caption(_CRUMB[mode])

if mode != _M_BUILTIN:
    if mode == _M_CBUILD:
        from tools.SSM.custom_model.ui_build import render_build_ui
        render_build_ui()
    elif mode == _M_XU:
        render_builtin_forward_sim("XuT", d["S_raw"], d["freq"], d["z0"], n)
    elif mode == _M_KY:
        render_builtin_forward_sim("KY", d["S_raw"], d["freq"], d["z0"], n)
    else:  # _M_CFIT
        from tools.SSM.custom_model.ui_fit import render_custom_fit
        render_custom_fit(n, d["S_raw"], d["freq"], d["z0"])
    st.stop()

st.subheader(i18n.t("ssm_step3"))
run_key = f"ssm_run_{n}"
if not st.session_state.get(run_key, False):
    col_ctr, _, _ = st.columns([1, 2, 2])
    if col_ctr.button(i18n.t("run_ssm"), key=f"ssm_btn_{n}",
                      width="stretch", type="primary"):
        st.session_state[run_key] = True
        # Force the fit-cache restore to re-run on this Run-SSM cycle.
        for _k in list(st.session_state.keys()):
            if (_k.startswith("cache_applied_")
                    or _k.startswith("cache_dismissed_")) and _k.endswith(f"_{n}"):
                del st.session_state[_k]
        st.rerun()
    st.caption(i18n.t("ssm_run_hint"))
else:
    if st.button(i18n.t("clear_ssm"), key=f"ssm_clear_{n}",
                 help=i18n.t("ssm_clear_help")):
        st.session_state[run_key] = False
        for k in list(st.session_state.keys()):
            if k.endswith(f"_{n}") and k != run_key:
                del st.session_state[k]
        st.rerun()

    render_ssm_tab(n, d["S_raw"], d["freq"], d["z0"],
                   s2o, s2s, all_data=all_data)
