"""
SSM_extraction.py — Standalone HBT Small-Signal Model (SSM) extraction page.

Previously this lived inside the RF S-Parameter Extraction tool's
"Individual → 🔬 SSM Extraction" sub-tab.  It now has its own portal page so
the extraction workflow is front-and-centre instead of buried under the RF
metrics tabs.

Upload one or more DUT ``.s2p`` / ``.csv`` bias files (and, optionally, the
device-dummy Open/Short ``.s2p`` pair for pad de-embedding), pick a file, and
run the analytic peeling extraction (Cheng T/π).

This page is **extraction only**.  Forward simulation, custom-model building,
Xu / Kun-Yang topologies and *all* tuning (visual + auto) now live on the
**SSM Simulation & Fitting** page — use the "→ Simulation & Fitting" button
below the extracted result to hand the device + fitted values straight over.

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.
"""
from __future__ import annotations

__version__ = "7.1"

import re
from pathlib import Path

import streamlit as st

from tools.rf.ssm.main_ssm_extraction import render_ssm_tab
from tools.rf.ssm.helpers import (parse_s2p, parse_csv, load_cal,
                               dedupe_upload_names)
from tools.common import handoff
from tools.common import i18n
from tools.common.paths import EXAMPLES_DIR


# ─────────────────────────────────────────────────────────────────────────────
st.title(i18n.title("ssm"))
st.caption(i18n.tool_desc("ssm"))

with st.expander(f"{i18n.t('whats_new')} · v{__version__}", expanded=False):
    st.markdown(i18n.tr(
        "- ✂️ **Extraction only.** This page now does just the analytic peeling "
        "extraction (Cheng T/π).  Forward simulation, custom models, Xu / "
        "Kun-Yang, and all tuning moved to **SSM Simulation & Fitting**.\n"
        "- 🔁 **Seamless handoff.** Send a de-embedded device here from *RF At a "
        "Glance*, and send the extracted model + values onward to *Simulation & "
        "Fitting* with one button — no re-uploading.\n\n"
        "See [`CHANGELOG.md`](CHANGELOG.md) for full history.",

        "- ✂️ **只做萃取。** 此頁現在只執行解析剝離萃取（Cheng T/π）。"
        "正向模擬、自訂模型、Xu / Kun-Yang 與所有調諧功能都已移至"
        "**小訊號模型模擬與擬合**。\n"
        "- 🔁 **無縫交接。** 可從 *RF 一覽* 將去嵌入後的元件送到這裡，"
        "再一鍵將萃取出的模型與數值送往 *模擬與擬合* — 完全不需重新上傳。\n\n"
        "完整紀錄請見 [`CHANGELOG.md`](CHANGELOG.md)。"
    ))

# with st.expander(i18n.t("how_it_works"), expanded=False):
#     from tools.common.diagrams import pipeline_png
#     st.image(pipeline_png((
#         ("Upload",   "DUT s2p / csv"),
#         ("De-embed", "pads"),
#         ("Extract",  "Cheng T / π"),
#         ("Review",   "Smith · residual"),
#         ("Hand off", "→ Sim & Fitting"),
#     ), accent="#d62728"), width="stretch")


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
#  HANDOFF — a device sent over from "RF At a Glance" (no re-upload needed)
# ═════════════════════════════════════════════════════════════════════════════
_inc = handoff.take(handoff.TARGET_EXTRACTION)
if _inc is not None:
    inj = st.session_state.setdefault("ssm_injected", {})

    def _norm_lbl(s):
        s = s or "handoff"
        return s if s.lower().endswith((".s2p", ".csv")) else f"{s}.s2p"

    lbl = _norm_lbl(_inc.get("label"))
    inj[lbl] = {"freq": _inc["freq"], "S_raw": _inc["S"], "z0": _inc["z0"],
                "stage": _inc.get("stage", "raw")}
    # Extra devices (other bias files) — needed by the Z-parameter, Cold-HBT and
    # τ_total methods.  Don't clobber the primary if a label collides.
    _n_extra = 0
    for _elbl, _ed in (_inc.get("extras") or {}).items():
        _k = _norm_lbl(_elbl)
        if _k == lbl:
            continue
        inj[_k] = {"freq": _ed["freq"], "S_raw": _ed["S"], "z0": _ed["z0"],
                   "stage": _inc.get("stage", "raw")}
        _n_extra += 1
    st.session_state["ssm_active_file"] = lbl
    _extra_msg = f" (+{_n_extra} more file{'s' if _n_extra != 1 else ''})" if _n_extra else ""
    st.markdown(
        f"<span class='hbt-chip-ok'>📥 {lbl} · {_inc.get('stage', 'raw')}"
        f"{_extra_msg}</span>"
        "<span class='hbt-help' title='Forwarded from another RF page. The "
        "processing stage (raw or de-embedded) is preserved.'>?</span>",
        unsafe_allow_html=True)
_injected: dict = st.session_state.get("ssm_injected", {})


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

if not dut_files and not use_examples and not _injected:
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
    # Disambiguated names: all_data is keyed on them, so two selected files
    # with the same name would otherwise drop one device silently.
    sources = [(name, f.getvalue()) for f, name in dedupe_upload_names(dut_files)]
elif use_examples:
    ex_dir = EXAMPLES_DIR
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

# Merge any handoff-injected devices (they already carry parsed arrays).
for _lbl, _dd in _injected.items():
    all_data[_lbl] = {"freq": _dd["freq"], "S_raw": _dd["S_raw"],
                      "z0": _dd["z0"]}

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

st.subheader(i18n.t("ssm_step3"))
run_key = f"ssm_run_{n}"
if not st.session_state.get(run_key, False):
    col_ctr, _, _ = st.columns([1, 2, 2])
    if col_ctr.button(i18n.t("run_ssm"), key=f"ssm_btn_{n}",
                      width="stretch", type="primary",
                      help=i18n.t("ssm_run_hint")):
        st.session_state[run_key] = True
        # Force the fit-cache restore to re-run on this Run-SSM cycle.
        for _k in list(st.session_state.keys()):
            if (_k.startswith("cache_applied_")
                    or _k.startswith("cache_dismissed_")
                    or _k.startswith("cache_use_on_entry_")
                    or _k.startswith("reextract_session_")
                    or _k.startswith("ssm_cache_params_snap_")) and _k.endswith(f"_{n}"):
                del st.session_state[_k]
        st.rerun()
else:
    if st.container(
            key=f"hbt_danger_clear_{re.sub(r'[^0-9A-Za-z_-]', '-', n)}"
    ).button(i18n.t("clear_ssm"), key=f"ssm_clear_{n}",
             help=i18n.t("ssm_clear_help")):
        st.session_state[run_key] = False
        for k in list(st.session_state.keys()):
            if k.endswith(f"_{n}") and k != run_key:
                del st.session_state[k]
        st.rerun()

    render_ssm_tab(n, d["S_raw"], d["freq"], d["z0"],
                   s2o, s2s, all_data=all_data)

    # ── Hand the extracted model + values over to Simulation & Fitting ───────
    _SHORT_LABEL = {"T": "Cheng's T", "pi": "Cheng's π"}
    _available = [(sh, st.session_state.get(f"current_p_{sh}_{n}"))
                  for sh in ("T", "pi")]
    _available = [(sh, cp) for sh, cp in _available if isinstance(cp, dict)]
    if _available:
        st.divider()
        st.markdown(
            i18n.tr("#### 🛠️ Continue in Simulation & Fitting",
                    "#### 🛠️ 前往模擬與擬合繼續")
            + "<span class='hbt-help' title='"
            + i18n.tr("Carry this device and its extracted values straight to "
                      "the Simulation & Fitting page for final tuning - no "
                      "re-upload. The S-parameters loaded here (de-embedded if "
                      "this device came from RF At a Glance) are forwarded "
                      "unchanged.",
                      "將此元件與其萃取值直接帶到模擬與擬合頁做最後調諧，"
                      "不需重新上傳。此處載入的 S 參數（若元件來自 RF 一覽則為"
                      "去嵌入後的資料）會原樣轉送。")
            + "'>?</span>",
            unsafe_allow_html=True)
        # Preserve the provenance of the active device: an injected file keeps
        # the stage it arrived with (e.g. "deembedded"); a directly-uploaded
        # file is "raw".  d["S_raw"] holds exactly those arrays either way, so
        # the device is forwarded unchanged.
        _stage = st.session_state.get("ssm_injected", {}).get(n, {}).get(
            "stage", "raw")
        hc = st.columns(len(_available))
        for col, (sh, cp) in zip(hc, _available):
            if col.button(
                    i18n.tr(f"→ Send {_SHORT_LABEL[sh]} to Simulation & Fitting",
                            f"→ 將 {_SHORT_LABEL[sh]} 送至模擬與擬合"),
                          key=f"ssm_send_simfit_{sh}_{n}", width="stretch",
                          type="primary"):
                handoff.send(handoff.TARGET_SIMFIT,
                             S=d["S_raw"], freq=d["freq"], z0=d["z0"],
                             label=Path(n).stem, stage=_stage,
                             params=dict(cp), model_short=sh)
                st.switch_page(handoff.PAGE_SIMFIT)
