"""
hbt_rf_tool.py
============================
Main Streamlit application for HBT RF extraction.

SSM extraction has moved to its own portal page,
``tools/SSM_extraction.py`` (sidebar → "HBT SSM Extraction").  The
Individual tab's "🔬 SSM Extraction" sub-tab here is now just a pointer
with a "Go there!" button.  This file covers the RF metrics workflow:
overlay / individual Bode·Plateau·Smith, summary, bulk upload, and the
3-step de-embedding + batch de-embed tabs.

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.
"""
__version__ = "1.1"

import hashlib, io, re, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from tools import i18n

from tools.SSM.ssm_plots      import render_matplotlib_smith
from tools.SSM             import handoff
from tools.batch_deembedding  import render_batch_deembedding_tab
from tools.SSM.helpers        import (
    parse_s2p, parse_csv,
    s_to_y, y_to_s_batch as y_to_s,
    strict_freq_check,
    deembed_open_short, deembed_thru_half,
    compute_metrics, extract_limit,
    extrap_20dbdec, single_pole_extrap,
    PALETTE, FT_FMAX_SYMBOLS, FT_FMAX_COLORS, darken, bode_layout, make_smith, make_bode, make_plateau,
    add_overlay_trace_with_markers,
    metric_card, build_excel, load_cal,
    xlsx_bytes_to_tsv, copy_button, frames_to_tsv,
    rust_parse_and_compute_batch,
    segmented_radio,
)

# ─────────────────────────────────────────────────────────────────────────────
if "rf_uploader_key" not in st.session_state:
    st.session_state["rf_uploader_key"] = 0

st.title(i18n.title("rf_extract"))
st.caption(i18n.tool_desc("rf_extract"))

with st.expander(f"{i18n.t('whats_new')} · v{__version__}", expanded=False):
    st.markdown(i18n.tr(
        "- 🪧 **SSM extraction moved to its own page** — find **HBT SSM "
        "Extraction** in the sidebar (RF group).\n"
        "- This tool keeps the RF metrics workflow: overlay / individual "
        "Bode · Plateau · Smith, summary table, bulk upload, and the "
        "3-step + batch de-embedding tabs.\n\n"
        "See [`CHANGELOG.md`](CHANGELOG.md) for full history.",
        "- 🪧 **SSM 萃取已移至獨立頁面** — 請至側邊欄（RF 群組）尋找"
        "**HBT SSM 萃取**。\n"
        "- 本工具保留 RF 指標工作流程：疊圖 / 單一檔案 Bode · Plateau · "
        "Smith、彙總表、批次上傳，以及三步驟 + 批次去嵌入分頁。\n\n"
        "完整歷史請見 [`CHANGELOG.md`](CHANGELOG.md)。"
    ))

# with st.expander(i18n.t("how_it_works"), expanded=False):
    # from tools.diagrams import pipeline_png
    # # Labels stay English: pipeline_png() rasterizes them with matplotlib
    # # (tools/diagrams.py, not this file) using a Latin-only default font, so
    # # CJK text would render as missing-glyph boxes.
    # st.image(pipeline_png((
    #     ("Upload",      "S2P / CSV"),
    #     ("De-embed",    "Open · Short · Thru"),
    #     ("Metrics",     "|h21|² · U · K"),
    #     ("Extrapolate", "fT · fmax"),
    #     ("Charts",      "Smith · Bode"),
    # ), accent="#d62728"), width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
#  CORE RF UTILITIES — moved to tools/SSM/helpers/ (rf_math, s2p_io,
#  deembed_math, metrics). See helpers/INDEX.md for the catalog.
# ═════════════════════════════════════════════════════════════════════════════

def _df_from_rust_entry(prepared: dict) -> pd.DataFrame:
    """Build a metrics DataFrame matching `compute_metrics()`'s schema from a
    single dict returned by `rust_parse_and_compute_batch`.  Lets the bulk
    loop hand the parsed result back into the existing process_dut path
    without re-doing parse + s_to_y + compute_metrics in Python."""
    return pd.DataFrame({
        "Freq (GHz)":             prepared["freq"] * 1e-9,
        "|h21|² (dB)":            prepared["h21_db"],
        "Mason U (dB)":           prepared["u_db"],
        "MAG/MSG (dB)":           prepared["mag_db"],
        "K Factor":               prepared["k"],
        "fT Plateau (GHz)":       prepared["ft_plat"],
        "fmax U Plateau (GHz)":   prepared["fmax_u_plat"],
        "fmax MAG Plateau (GHz)": prepared["fmax_mag_plat"],
    })


def process_dut(content, filename, s1_o, s1_s, s2_o, s2_s, s3_t,
                n_pts, f_min, f_max, *, prepared=None):
    """Parse, de-embed, compute metrics, extract fT/fmax for one DUT file.

    When `prepared` is given (a dict from `rust_parse_and_compute_batch`
    plus a `df_raw` field), the parse + s_to_y + compute_metrics phase is
    skipped — we use the pre-parsed `freq`/`S`/`z0`/`df_raw` directly.
    `Y_raw` becomes lazy: it's only materialised when a de-embed branch
    actually fires, saving ~0.2 ms on the no-cal bulk-upload path.
    """
    if prepared is not None:
        freq   = prepared["freq"]
        S_raw  = prepared["S"]
        z0     = float(prepared["z0"])
        df_raw = prepared["df_raw"]
    elif filename.lower().endswith(".csv"):
        freq, S_raw, z0 = parse_csv(content)
        df_raw = compute_metrics(s_to_y(S_raw, z0), freq)
    else:
        freq, S_raw, z0 = parse_s2p(content)
        df_raw = compute_metrics(s_to_y(S_raw, z0), freq)

    _y_raw_cache = [None]
    def _y_raw():
        if _y_raw_cache[0] is None:
            _y_raw_cache[0] = s_to_y(S_raw, z0)
        return _y_raw_cache[0]

    Y_fin, stages, d1_o, d1_s = None, [], None, None
    if s1_o and s1_s:
        f1o,S1o,z1o=s1_o; f1s,S1s,z1s=s1_s
        strict_freq_check(freq,f1o,"Probe Open")
        d1_o,d1_s=s_to_y(S1o,z1o),s_to_y(S1s,z1s)
        Y_fin=deembed_open_short(_y_raw(),d1_o,d1_s); stages.append("Probe")
    Y2o=Y2s=None
    if s2_o and s2_s:
        f2o,S2o,z2o=s2_o; f2s,S2s,z2s=s2_s
        strict_freq_check(freq,f2o,"Dev Open")
        Y2o,Y2s=s_to_y(S2o,z2o),s_to_y(S2s,z2s)
        if d1_o is not None:
            Y2o=deembed_open_short(Y2o,d1_o,d1_s)
            Y2s=deembed_open_short(Y2s,d1_o,d1_s)
        if Y_fin is None: Y_fin = _y_raw()
        Y_fin=deembed_open_short(Y_fin,Y2o,Y2s); stages.append("Dev(O/S)")
    if s3_t:
        f3t,S3t,z3t=s3_t; strict_freq_check(freq,f3t,"Dev Thru")
        Y3t=s_to_y(S3t,z3t)
        if d1_o is not None: Y3t=deembed_open_short(Y3t,d1_o,d1_s)
        if "Dev(O/S)" in stages:
            Y3t_r=s_to_y(S3t,z3t)
            if d1_o is not None: Y3t_r=deembed_open_short(Y3t_r,d1_o,d1_s)
            Y3t=deembed_open_short(Y3t_r,Y2o,Y2s)
        if Y_fin is None: Y_fin = _y_raw()
        Y_fin=deembed_thru_half(Y_fin,Y3t); stages.append("Dev(Thru)")
    note=" + ".join(stages) if stages else "None"
    df_fin=compute_metrics(Y_fin,freq) if stages else None
    S_fin=y_to_s(Y_fin,z0) if stages else S_raw
    df_e=df_fin if df_fin is not None else df_raw
    f_arr=df_e["Freq (GHz)"].values
    # When the Rust batch kernel pre-computed extract_limit for df_raw
    # AND no de-embed stages fired (so df_e == df_raw), reuse those
    # values directly — saves 3 Python extract_limit calls per file.
    # When stages did fire, df_fin is different from df_raw, so we must
    # re-extract from df_fin (Python path).
    if prepared is not None and not stages \
            and "ft_cross" in prepared and "ft_method" in prepared:
        fT_cr,  fT_pl,  ft_m  = prepared["ft_cross"],     prepared["ft_plateau"],     prepared["ft_method"]
        fmU_cr, fmU_pl, fmU_m = prepared["fmax_u_cross"], prepared["fmax_u_plateau"], prepared["fmax_u_method"]
        fmM_cr, fmM_pl, fmM_m = prepared["fmax_mag_cross"], prepared["fmax_mag_plateau"], prepared["fmax_mag_method"]
    else:
        fT_cr,fT_pl,ft_m    = extract_limit(f_arr,df_e["|h21|² (dB)"].values,  df_e["fT Plateau (GHz)"].values,  n_pts,f_min,f_max)
        fmU_cr,fmU_pl,fmU_m = extract_limit(f_arr,df_e["Mason U (dB)"].values,  df_e["fmax U Plateau (GHz)"].values,n_pts,f_min,f_max)
        fmM_cr,fmM_pl,fmM_m = extract_limit(f_arr,df_e["MAG/MSG (dB)"].values,  df_e["fmax MAG Plateau (GHz)"].values,n_pts,f_min,f_max)
    stem=re.sub(r"\.(s2p|csv)$","",filename,flags=re.IGNORECASE)
    m=re.search(r"[Vv][Cc][Ee][_\-]?([\d]+(?:p\d+)?)\s*[Vv]",stem)
    vce=float(m.group(1).replace("p",".")) if m else None
    m=re.search(r"[Ii][Bb][_\-]?([\d]+(?:p\d+)?)\s*([pnuUmM]?)[Aa]?",stem)
    ib=float(m.group(1).replace("p","."))*{"p":1e-12,"n":1e-9,"u":1e-6,"U":1e-6,"m":1e-3,"M":1e-3,"":1.0}.get(m.group(2) if m else "",1.0) if m else None
    return df_raw,df_fin,S_fin,S_raw,freq,z0,{
        "Label":stem,"Vce (V)":vce,"Ib (A)":ib,"De-embedding":note,
        "fT Cross/Extrap (GHz)":fT_cr,"fT Plateau (GHz)":fT_pl,"fT Method":ft_m,
        "fmax U Cross/Extrap (GHz)":fmU_cr,"fmax U Plateau (GHz)":fmU_pl,"fmax U Method":fmU_m,
        "fmax MAG Cross/Extrap (GHz)":fmM_cr,"fmax MAG Plateau (GHz)":fmM_pl,"fmax MAG Method":fmM_m,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING UTILITIES — moved to tools/SSM/helpers/plotly_plots.py
#  (PALETTE, darken, bode_layout, make_smith, make_bode, make_plateau)
#  and tools/SSM/helpers/chart_export.py (build_excel, metric_card, load_cal).
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"## {i18n.tr('Settings', '設定')}")
    st.markdown(f"#### {i18n.tr('3-Step De-embedding', '三步驟去嵌入')}")
    sw1=st.toggle(i18n.tr("① Probe (Open-Short)", "① 探針（Open-Short）"),value=False)
    f1o=st.file_uploader(i18n.tr("Probe Open","探針 Open"),  type=["s2p"],key="p1o") if sw1 else None
    f1s=st.file_uploader(i18n.tr("Probe Short","探針 Short"), type=["s2p"],key="p1s") if sw1 else None
    st.divider()
    sw2=st.toggle(i18n.tr("② Device Dummy (Open-Short)", "② 元件 Dummy（Open-Short）"),value=False,
                  help=i18n.tr("Open-short de-embedding of the device pad parasitics",
                               "以 open-short 去嵌入元件 pad 寄生參數"))
    f2o=st.file_uploader(i18n.tr("Dev Open","元件 Open"),  type=["s2p"],key="d2o") if sw2 else None
    f2s=st.file_uploader(i18n.tr("Dev Short","元件 Short"), type=["s2p"],key="d2s") if sw2 else None
    if sw2: st.caption(i18n.tr("✅ Device-dummy de-embedding enabled.", "✅ 元件 dummy 去嵌入已啟用。"))
    st.divider()
    sw3=st.toggle(i18n.tr("③ Device Thru (Half-Z)", "③ 元件 Thru（Half-Z）"),value=False)
    f3t=st.file_uploader(i18n.tr("Dev Thru","元件 Thru"),  type=["s2p"],key="d3t") if sw3 else None
    st.divider()
    st.markdown(f"#### {i18n.tr('Chart Control', '圖表控制')}")
    freq_min=st.number_input(i18n.tr("Freq Min (GHz)","頻率下限 (GHz)"),value=0.01,min_value=0.01,format="%.4f")
    freq_max=st.number_input(i18n.tr("Freq Max (GHz)","頻率上限 (GHz)"),value=50.0,min_value=1.0)
    db_min=st.number_input(i18n.tr("Bode Y Min (dB)","Bode Y 軸下限 (dB)"),value=0.0)
    db_max=st.number_input(i18n.tr("Bode Y Max (dB)","Bode Y 軸上限 (dB)"),value=50.0)
    n_pts=st.slider(i18n.tr("Interpolation pts","內插點數"),2,20,2)
    show_raw=st.checkbox(i18n.tr("Overlay Raw","疊加原始值"),value=True,disabled=not(sw1 or sw2 or sw3))
    st.markdown(f"##### {i18n.tr('Trace Selection', '曲線選擇')}")
    sh21=st.checkbox("|h21|² → fT",value=True,key="sh21")
    su=st.checkbox("Mason U → fmax(U)",value=True,key="su")
    smag=st.checkbox("MAG/MSG → fmax",value=True,key="smag")
    skk=st.checkbox(i18n.tr("K factor (stability)", "K 因子（穩定性）"),value=False,key="sk_factor",
                     help=i18n.tr(
                         "Rollett's stability factor K = (1−|S11|²−|S22|²+|Δ|²)/(2|S12·S21|).  "
                         "Network is unconditionally stable when K > 1 *and* |Δ| < 1.  "
                         "Rendered on a separate axis below the bode plot.",
                         "Rollett 穩定性因子 K = (1−|S11|²−|S22|²+|Δ|²)/(2|S12·S21|)。"
                         "當 K > 1 *且* |Δ| < 1 時，網路為絕對穩定。"
                         "繪製於 Bode 圖下方的獨立座標軸。"))
    st.divider()
    st.markdown(f"#### {i18n.tr('Smith Chart', 'Smith 圖')}")
    smith_f_min=st.number_input(i18n.tr("Smith Freq Min (GHz)","Smith 頻率下限 (GHz)"),value=freq_min,min_value=0.01,format="%.4f")
    smith_f_max=st.number_input(i18n.tr("Smith Freq Max (GHz)","Smith 頻率上限 (GHz)"),value=freq_max,min_value=1.0)
    smith_max_r=st.slider(i18n.tr("|Γ| Max Radius","|Γ| 最大半徑"),1.0,5.0,1.0,0.5)
    st.markdown(f"##### {i18n.tr('Display & Scale', '顯示與縮放')}")
    ca,cb=st.columns(2); show_s11=ca.checkbox("S11",value=True,key="ss11"); scale_s11=cb.number_input("S11 ×",value=1.0,step=0.1,key="sc11")
    ca,cb=st.columns(2); show_s22=ca.checkbox("S22",value=True,key="ss22"); scale_s22=cb.number_input("S22 ×",value=1.0,step=0.1,key="sc22")
    ca,cb=st.columns(2); show_s21=ca.checkbox("S21",value=True,key="ss21"); scale_s21=cb.number_input("S21 ×",value=1.0,step=0.1,key="sc21")
    ca,cb=st.columns(2); show_s12=ca.checkbox("S12",value=True,key="ss12"); scale_s12=cb.number_input("S12 ×",value=1.0,step=0.1,key="sc12")

# ═════════════════════════════════════════════════════════════════════════════
#  FILE UPLOADER & PROCESSING
# ═════════════════════════════════════════════════════════════════════════════
# Uploader + "Clear uploads" share one row; bottom-aligning the columns lines
# the button up with the base of the dropzone (no manual st.write() spacers).
col_up1,col_up2=st.columns([4,1],vertical_alignment="bottom")
with col_up1:
    dut_files=st.file_uploader(i18n.tr("Upload DUT .s2p / .csv files","上傳 DUT .s2p / .csv 檔案"),type=["s2p","csv"],
                               accept_multiple_files=True,
                               key=st.session_state["rf_uploader_key"])
with col_up2:
    if st.button(i18n.t("clear_uploads"),width="stretch"):
        st.session_state["rf_uploader_key"]+=1
        st.session_state.pop("rf_ms_files",None)
        st.session_state.pop("rf_prev_uploaded",None)
        st.rerun()

s1o=load_cal(f1o) if sw1 else None
s1s=load_cal(f1s) if sw1 else None
s2o=load_cal(f2o) if sw2 else None   # device-dummy open (de-embedding)
s2s=load_cal(f2s) if sw2 else None   # device-dummy short (de-embedding)
s3t=load_cal(f3t) if sw3 else None

all_data,errors={},{}

# Per-file process_dut cache.  Streamlit re-runs the whole script on every
# interaction (slider drag, checkbox toggle), so without a cache the bulk
# loop re-parses every .s2p file on every rerun — that's the ~10 s
# unresponsiveness when uploading 30+ files.  Profiling shows the math is
# only ~8 ms/file; the gain here is from skipping it entirely on reruns
# where nothing the file depends on has changed.
#
# Cache key = file bytes + cal signatures + chart-window params.  Cal
# signatures only need to cover the inputs that actually feed
# `process_dut` (the four de-embed S-arrays + thru); the Smith / display
# controls don't.  Anything else changing leaves the cache hot.
def _cal_sig(cal):
    if cal is None:
        return b""
    _, S, _ = cal
    return S.tobytes()

def _dut_cache_key(content_bytes, s1o, s1s, s2o, s2s, s3t,
                   n_pts, freq_min, freq_max):
    h = hashlib.blake2b(digest_size=16)
    h.update(content_bytes)
    for cal in (s1o, s1s, s2o, s2s, s3t):
        h.update(_cal_sig(cal))
    h.update(repr((int(n_pts), float(freq_min), float(freq_max))).encode())
    return h.digest()

_dut_cache = st.session_state.setdefault("rf_dut_cache", {})
_dut_keys: dict[str, bytes] = {}

if dut_files:
    fresh_keys = set()

    # ── Pass 1: classify each file as (cache hit | s2p-miss | csv-miss) ──
    # `.s2p` cache misses get batched through the Rust kernel below — one
    # call across all of them, Rayon-parallel with the GIL released — so
    # parse + s_to_y + compute_metrics for 30 files completes in roughly
    # one-Nth the wall-clock of the old per-file Python loop.  CSV files
    # stay on the Python path (pandas dependency, low usage).
    s2p_misses: list[tuple] = []   # (file_obj, key, bytes)
    csv_misses: list[tuple] = []   # (file_obj, key, bytes)
    for f in dut_files:
        content = f.getvalue()
        try:
            key = _dut_cache_key(content, s1o, s1s, s2o, s2s, s3t,
                                 n_pts, freq_min, freq_max)
        except Exception as e:
            errors[f.name] = f"cache-key error: {e}"
            continue
        fresh_keys.add(key)
        cached = _dut_cache.get(key)
        if cached is not None:
            all_data[f.name] = cached
            _dut_keys[f.name] = key
            continue
        if f.name.lower().endswith(".s2p"):
            s2p_misses.append((f, key, content))
        else:
            csv_misses.append((f, key, content))

    # ── Pass 2: Rust batch parse + metrics + extract for .s2p misses ──
    # We hand n_pts/freq_min/freq_max in so the kernel can run extract_limit
    # inside the parallel section.  For files with no de-embed (which is
    # the common bulk-upload case), this skips the 3 Python extract_limit
    # calls per file — saves ~1.6 ms/file × 30 = ~48 ms cold first-upload.
    if s2p_misses:
        batch_bytes = [c for _, _, c in s2p_misses]
        try:
            rust_results = rust_parse_and_compute_batch(
                batch_bytes, n_pts, freq_min, freq_max)
        except Exception as e:
            rust_results = None
            for f, _key, _c in s2p_misses:
                errors[f.name] = f"batch parse failed: {e}"
        if rust_results is not None:
            for (f, key, _content), prepared in zip(s2p_misses, rust_results):
                if "error" in prepared:
                    errors[f.name] = prepared["error"]
                    continue
                try:
                    prepared["df_raw"] = _df_from_rust_entry(prepared)
                    df_raw,df_fin,S_fin,S_raw,freq,z0_dut,res = process_dut(
                        None, f.name, s1o, s1s, s2o, s2s, s3t,
                        n_pts, freq_min, freq_max, prepared=prepared)
                    entry = {
                        "df_raw":df_raw,"df_fin":df_fin,
                        "S_fin":S_fin,"S_raw":S_raw,
                        "freq":freq,"z0":z0_dut,**res}
                    _dut_cache[key] = entry
                    all_data[f.name] = entry
                    _dut_keys[f.name] = key
                except Exception as e:
                    errors[f.name] = str(e)

    # ── Pass 3: Python path for .csv files ──
    for f, key, content in csv_misses:
        try:
            df_raw,df_fin,S_fin,S_raw,freq,z0_dut,res = process_dut(
                content.decode("utf-8",errors="ignore"),
                f.name,s1o,s1s,s2o,s2s,s3t,n_pts,freq_min,freq_max)
            entry = {
                "df_raw":df_raw,"df_fin":df_fin,
                "S_fin":S_fin,"S_raw":S_raw,
                "freq":freq,"z0":z0_dut,**res}
            _dut_cache[key] = entry
            all_data[f.name] = entry
            _dut_keys[f.name] = key
        except Exception as e:
            errors[f.name] = str(e)

    # Evict entries for files no longer in the uploader (or with stale
    # cal/chart params) to bound memory across long sessions.
    for stale in list(_dut_cache.keys() - fresh_keys):
        del _dut_cache[stale]
else:
    _dut_cache.clear()

# ── Plotly figure cache (overlay tab) ─────────────────────────────────────────
# Streamlit's st.tabs always executes every tab body on every rerun, so any
# checkbox toggle or slider drag in the sidebar currently rebuilds the heavy
# Overlay figures from scratch.  We memoize them on a key derived from the
# per-DUT cache keys (proxy for "input data identity") plus the settings each
# figure reads.  First render still pays the construction cost; every rerun
# afterwards that leaves these inputs unchanged returns the cached figure.
_fig_cache: dict = st.session_state.setdefault("rf_fig_cache", {})

def _selected_dut_keys(names):
    """Tuple of cache keys for the currently-selected files, preserving order."""
    return tuple(_dut_keys.get(n, n.encode()) for n in names)

def _cached_fig(key, build_fn):
    fig = _fig_cache.get(key)
    if fig is None:
        fig = build_fn()
        _fig_cache[key] = fig
    return fig

def _evict_stale_figs():
    """Drop cached figures whose referenced DUT keys are no longer current.

    Each cached key has its DUT reference at position [1]; overlay figures
    store a tuple of keys (one per selected file), individual figures
    store a single bytes key.
    """
    valid = set(_dut_keys.values())
    for k in list(_fig_cache.keys()):
        try:
            ref = k[1]
            if isinstance(ref, bytes):
                if ref not in valid and valid:
                    del _fig_cache[k]
            elif isinstance(ref, tuple):
                refs = [r for r in ref if isinstance(r, bytes)]
                if refs and any(r not in valid for r in refs):
                    del _fig_cache[k]
        except Exception:
            pass
    # If the user cleared all uploads, drop everything so memory doesn't
    # linger across sessions where the same cache key is reused.
    if not _dut_keys:
        _fig_cache.clear()

_evict_stale_figs()

for fname,err in errors.items():
    st.error(f"**{fname}**: {err}")

if all_data:
    file_options=list(all_data.keys())
    # Every uploaded file is analysed by default — no per-file selector.
    selected_files=file_options
else:
    selected_files=[]
    st.info(i18n.tr("Upload DUT .s2p files above to begin.","請於上方上傳 DUT .s2p 檔案以開始。"))

xr,yr=(freq_min,freq_max),(db_min,db_max)

# ═════════════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_ov,tab_ind,tab_sum,tab_bd=st.tabs([
    i18n.tr("📊 Overlay","📊 疊圖"),
    i18n.tr("📁 Individual","📁 單一檔案"),
    i18n.tr("📋 Summary","📋 摘要"),
    i18n.tr("🧰 Batch De-embed","🧰 批次去嵌入"),
])

with tab_ov:
    st.markdown(f"### {i18n.tr('Bode Plot Overlay', 'Bode 圖疊圖')}")

    def _build_overlay_bode():
        # Standardised colours: fT trace (|h21|²) = blue, fmax traces
        # (Mason U, MAG/MSG) = red.  All files share the same fT/fmax
        # colour — files are distinguished by name in the legend rather
        # than by hue.  PALETTE is still passed (i, c, ...) so future
        # per-file tinting can re-use it without changing this loop.
        c_fT, c_fmax = FT_FMAX_COLORS["fT"], FT_FMAX_COLORS["fmax"]
        fig = go.Figure()
        if all_data and selected_files:
            hov = "Freq:%{x:.4f}GHz<br>%{y:.4f}dB<extra></extra>"
            for i, n in enumerate(selected_files):
                d, _c, lbl = all_data[n], PALETTE[i % len(PALETTE)], Path(n).stem
                df_p = d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
                fx = df_p["Freq (GHz)"]
                if show_raw and d["df_fin"] is not None and sh21:
                    fig.add_trace(go.Scattergl(x=d["df_raw"]["Freq (GHz)"], y=d["df_raw"]["|h21|² (dB)"],
                                                name=f"|h21|² raw–{lbl}", line=dict(color=c_fT, width=1.2, dash="dot"),
                                                opacity=0.35, hovertemplate=hov))
                if sh21:
                    add_overlay_trace_with_markers(
                        fig, fx, df_p["|h21|² (dB)"], name=f"|h21|²–{lbl}",
                        color=c_fT, symbol=FT_FMAX_SYMBOLS["h21"], line_width=2.5,
                        marker_size=5, hovertemplate=hov)
                if su:
                    add_overlay_trace_with_markers(
                        fig, fx, df_p["Mason U (dB)"], name=f"U–{lbl}",
                        color=c_fmax, symbol=FT_FMAX_SYMBOLS["U"], dash="dash",
                        line_width=2.5, marker_size=5, hovertemplate=hov)
                if smag:
                    add_overlay_trace_with_markers(
                        fig, fx, df_p["MAG/MSG (dB)"], name=f"MAG–{lbl}",
                        color=c_fmax, symbol=FT_FMAX_SYMBOLS["MAG"], dash="dot",
                        line_width=2, marker_size=5, opacity=0.7,
                        hovertemplate=hov)
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(**bode_layout(i18n.tr("Overlay — Bode Plot", "疊圖 — Bode 圖"),
                                         i18n.tr("Gain (dB)", "增益 (dB)"), yr, xr))
        fig.update_layout(height=550)
        return fig

    f_bode = _cached_fig(("ov_bode", _selected_dut_keys(selected_files),
                          bool(show_raw), bool(sh21), bool(su), bool(smag),
                          xr, yr), _build_overlay_bode)
    st.plotly_chart(f_bode, width="stretch")

    if skk:
        st.markdown(f"### {i18n.tr('K-Factor Overlay (Rollett stability)', 'K 因子疊圖（Rollett 穩定性）')}")

        def _build_overlay_kfactor():
            fig = go.Figure()
            if all_data and selected_files:
                for i, n in enumerate(selected_files):
                    d, c, lbl = all_data[n], PALETTE[i % len(PALETTE)], Path(n).stem
                    df_p = d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
                    if "K Factor" in df_p.columns:
                        fig.add_trace(go.Scattergl(
                            x=df_p["Freq (GHz)"], y=df_p["K Factor"],
                            name=f"K — {lbl}",
                            line=dict(color=c, width=2.5),
                            hovertemplate="Freq:%{x:.4f} GHz<br>K=%{y:.4f}"
                                           "<extra></extra>"))
            fig.add_hline(y=1.0, line_dash="dash", line_color="#888",
                          annotation_text="K = 1", annotation_position="right",
                          annotation_font=dict(size=9, color="#888"))
            fig.update_layout(**bode_layout(i18n.tr("Overlay — K Factor", "疊圖 — K 因子"),
                                              i18n.tr("K (dimensionless)", "K（無因次）"),
                                              [0, 6], xr))
            fig.update_layout(height=350)
            return fig

        f_k = _cached_fig(("ov_kfactor", _selected_dut_keys(selected_files), xr),
                          _build_overlay_kfactor)
        st.plotly_chart(f_k, width="stretch")

    st.markdown(f"### {i18n.tr('Plateau Plot Overlay', 'Plateau 圖疊圖')}")

    def _build_overlay_plateau():
        # Standardised colours — same convention as the Bode overlay above.
        c_fT, c_fmax = FT_FMAX_COLORS["fT"], FT_FMAX_COLORS["fmax"]
        fig = go.Figure()
        all_v: list[float] = []
        if all_data and selected_files:
            hov = "Freq:%{x:.4f}GHz<br>GBP:%{y:.4f}GHz<extra></extra>"
            for i, n in enumerate(selected_files):
                d, _c, lbl = all_data[n], PALETTE[i % len(PALETTE)], Path(n).stem
                df_p = d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
                fx = df_p["Freq (GHz)"]
                if sh21:
                    add_overlay_trace_with_markers(
                        fig, fx, df_p["fT Plateau (GHz)"], name=f"fT–{lbl}",
                        color=c_fT, symbol=FT_FMAX_SYMBOLS["h21"], line_width=2.5,
                        marker_size=5, hovertemplate=hov)
                    all_v += df_p["fT Plateau (GHz)"].dropna().tolist()
                if su:
                    add_overlay_trace_with_markers(
                        fig, fx, df_p["fmax U Plateau (GHz)"], name=f"fmax(U)–{lbl}",
                        color=c_fmax, symbol=FT_FMAX_SYMBOLS["U"], dash="dash",
                        line_width=2.5, marker_size=5, hovertemplate=hov)
                    all_v += df_p["fmax U Plateau (GHz)"].dropna().tolist()
        arr = np.array([v for v in all_v if np.isfinite(v) and v > 0])
        ym = float(np.quantile(arr, 0.97)) * 1.3 if len(arr) else 100
        fig.update_layout(**bode_layout(i18n.tr("Overlay — Plateau", "疊圖 — Plateau"),
                                         "GBP (GHz)", [0, ym], xr))
        fig.update_layout(height=550)
        return fig

    f_plat = _cached_fig(("ov_plateau", _selected_dut_keys(selected_files),
                          bool(sh21), bool(su), xr), _build_overlay_plateau)
    st.plotly_chart(f_plat, width="stretch")

with tab_ind:
    if not all_data or not selected_files:
        st.info(i18n.tr("Upload and select files to view individual analysis.",
                        "請上傳並選取檔案以檢視單一檔案分析。"))
    else:
        # Single-active-file selector — replaces st.tabs(...) over selected_files.
        # Streamlit executes the body of every st.tabs branch on every rerun (only
        # the visibility is toggled), so with N files a single slider drag would
        # re-run all N pipelines. A selectbox conditionally renders only the
        # selected file's body, so cost no longer scales with N.
        # The Summary tab's nav button may have queued a file to open here.
        # Apply it *before* the selectbox is instantiated — Streamlit only lets
        # you seed a widget's session_state key pre-instantiation. tab_ind runs
        # earlier in the script than tab_sum, so the queue is read on the rerun
        # the button triggers.
        _pending=st.session_state.pop("_pending_active_file",None)
        if _pending in selected_files:
            st.session_state["active_file_n"]=_pending
        n=st.selectbox(
            i18n.tr("📁 Active file", "📁 目前檔案"),
            options=selected_files,
            format_func=lambda fn: Path(fn).stem,
            key="active_file_n",
            help=i18n.tr("Only the selected file is rendered. Use the dropdown or arrow keys to switch.",
                         "僅渲染目前選取的檔案。可用下拉選單或方向鍵切換。"),
        )
        file_names=list(all_data.keys())
        c=PALETTE[file_names.index(n)%len(PALETTE)]
        d=all_data[n]
        df_p=d["df_fin"] if d["df_fin"] is not None else d["df_raw"]

        # extract_limit()'s method tags — comparisons stay against the raw
        # English tag; only the displayed text localizes.
        _METHOD_ZH = {"No Gain": "無增益", "No Data": "無資料",
                     "0dB Cross": "0dB 交越", "Extrap & Plat.": "外插與平台"}
        def _method_disp(m):
            return i18n.tr(m, _METHOD_ZH.get(m, m))

        def _fc(v_cr,v_pl,method):
            if method in ["No Gain","No Data"]: return _method_disp(method)
            if method=="0dB Cross":      return f"{v_cr:.3f} GHz" if np.isfinite(v_cr) else "N/A"
            if method=="Extrap & Plat.": return f"{v_pl:.3f} GHz" if np.isfinite(v_pl) else "N/A"
            return "N/A"

        _f_card = df_p["Freq (GHz)"].values
        _nf_card = len(_f_card)

        # Extrapolation only makes sense while the gain trace is still above
        # 0 dB at the last measured point — that's exactly the case the
        # −20 dB/dec and single-pole overlays project toward the 0-dB crossing.
        # Once a trace has already dropped to/below 0 dB inside the band the
        # overlays draw nothing, so default their checkboxes off.  Evaluated on
        # the fT (|h21|²) and fmax (Mason U) traces the overlays actually apply
        # to; ticked when either still sits above 0 dB at the final frequency.
        def _last_gain_above_0db(col_name):
            if col_name not in df_p.columns or _nf_card == 0:
                return False
            y = df_p[col_name].values
            m = np.isfinite(y)
            return bool(m.any() and float(y[m][-1]) > 0.0)

        _extrap_default = (_last_gain_above_0db("|h21|² (dB)")
                           or _last_gain_above_0db("Mason U (dB)"))

        # Mirror the Bode tab's extrapolation controls on the fT/fmax cards.
        # The Bode tab's widgets render later in this rerun, but their
        # session_state keys persist from the previous rerun, so reading
        # them here matches what the tab is about to display.
        _show20_card = st.session_state.get(f"bode_show20_{n}", _extrap_default)
        _showsp_card = st.session_state.get(f"bode_showsp_{n}", _extrap_default)
        _sp_win_card = st.session_state.get(f"bode_spwin_{n}", None)

        def _extrap_f0(col_name):
            f0_20 = f0_sp = None
            if col_name not in df_p.columns or _nf_card == 0:
                return f0_20, f0_sp
            y = df_p[col_name].values
            if _show20_card:
                _, _, f0_20 = extrap_20dbdec(_f_card, y)
            if (_showsp_card and _sp_win_card is not None
                    and _nf_card >= 4):
                f_lo, f_hi = float(_f_card[0]), float(_f_card[-1])
                lo = max(float(_sp_win_card[0]), f_lo)
                hi = min(float(_sp_win_card[1]), f_hi)
                _il = int(np.searchsorted(_f_card, lo, side="left"))
                _ih = int(np.searchsorted(_f_card, hi, side="right")) - 1
                _il = max(0, min(_il, _nf_card - 2))
                _ih = max(_il + 1, min(_ih, _nf_card - 1))
                _, _, f0_sp, _, _ = single_pole_extrap(_f_card, y, _il, _ih)
            return f0_20, f0_sp

        _fT_20, _fT_sp     = _extrap_f0("|h21|² (dB)")
        _fmU_20, _fmU_sp   = _extrap_f0("Mason U (dB)")

        def _sub(method, f20, fsp):
            parts = [_method_disp(method)]
            if f20 is not None and np.isfinite(f20):
                parts.append(f"−20dB: {f20:.2f} GHz")
            if fsp is not None and np.isfinite(fsp):
                parts.append(f"SP: {fsp:.2f} GHz")
            return " | ".join(parts)

        # K-factor summary — minimum K over the swept band + the
        # frequency where it occurs, plus a stability indicator.
        # Rollett K is dimensionless and varies with frequency, so the
        # most useful single-number summary is the worst-case (min).
        def _k_summary(_df):
            if "K Factor" not in _df.columns:
                return None, None, "—"
            k_series = _df["K Factor"]
            mask = np.isfinite(k_series.values)
            if not mask.any():
                return None, None, "—"
            k_arr = k_series.values[mask]
            f_arr = _df["Freq (GHz)"].values[mask]
            i_min = int(np.argmin(k_arr))
            k_min = float(k_arr[i_min])
            f_at_min = float(f_arr[i_min])
            # Stability classification (Rollett's K alone — full
            # criterion also requires |Δ| < 1; for the card we report
            # the K side and let users dig into Δ in the plot below).
            if (k_arr >= 1.0).all():
                tag = i18n.tr("🟢 K≥1 across band", "🟢 全頻段 K≥1")
            elif k_min < 1.0 < float(k_arr.max()):
                tag = i18n.tr("🟡 K crosses 1", "🟡 K 值跨越 1")
            else:
                tag = i18n.tr("🔴 K<1 across band", "🔴 全頻段 K<1")
            return k_min, f_at_min, tag

        _k_min, _k_fmin, _k_tag = _k_summary(df_p)

        c1,c2,c3,c4,c5,c6=st.columns(6)
        metric_card(c1,i18n.tr("De-embedding","去嵌入"),d["De-embedding"],i18n.tr("mode","模式"),"#888")
        metric_card(c2,"fT (GHz)",_fc(d["fT Cross/Extrap (GHz)"],d["fT Plateau (GHz)"],d["fT Method"]),_sub(d["fT Method"],_fT_20,_fT_sp))
        metric_card(c3,"fmax U",_fc(d["fmax U Cross/Extrap (GHz)"],d["fmax U Plateau (GHz)"],d["fmax U Method"]),_sub(d["fmax U Method"],_fmU_20,_fmU_sp),"#d62728")
        metric_card(c4,"fmax MAG",_fc(d["fmax MAG Cross/Extrap (GHz)"],d["fmax MAG Plateau (GHz)"],d["fmax MAG Method"]),_method_disp(d["fmax MAG Method"]),"#2ca02c")
        if _k_min is not None:
            metric_card(c5,"K min",
                         f"{_k_min:.3f}",
                         f"{_k_tag} @ {_k_fmin:.2f} GHz",
                         "#0d7377")
        else:
            metric_card(c5,"K min","—",i18n.tr("not available","無法計算"),"#888")
        if d["Vce (V)"] is not None:  metric_card(c6,"Vce",f"{d['Vce (V)']} V",i18n.tr("bias","偏壓"),"#9467bd")
        elif d["Ib (A)"] is not None: metric_card(c6,"Ib",f"{d['Ib (A)']*1e6:.1f} µA",i18n.tr("bias","偏壓"),"#9467bd")

        toggles={"S11":show_s11,"S22":show_s22,"S21":show_s21,"S12":show_s12}
        scales ={"S11":scale_s11,"S22":scale_s22,"S21":scale_s21,"S12":scale_s12}

        ta,tb,tc=st.tabs([i18n.tr("Bode Plot","Bode 圖"),
                          i18n.tr("Plateau Plot","Plateau 圖"),
                          i18n.tr("Smith Chart","Smith 圖")])
        with ta:
            f_arr = df_p["Freq (GHz)"].values
            n_freq = len(f_arr)

            # ── Extrapolation controls ───────────────────────────────────────
            bc1, bc2 = st.columns([1, 1])
            show_20db = bc1.checkbox(
                i18n.tr("Show −20 dB/dec extrapolation", "顯示 −20 dB/dec 外插線"),
                value=_extrap_default, key=f"bode_show20_{n}",
                help=i18n.tr("Anchors a line of slope −20 dB/dec at the last "
                             "measured point (textbook fT/fmax extraction).",
                             "於最後一個量測點錨定斜率 −20 dB/dec 的直線"
                             "（教科書式 fT/fmax 萃取法）。"))
            show_sp = bc2.checkbox(
                i18n.tr("Show single-pole fit", "顯示單極點擬合"),
                value=_extrap_default, key=f"bode_showsp_{n}",
                help=i18n.tr("Log-linear (single-pole) least-squares fit on a "
                             "user-chosen window — slope is determined by the data.",
                             "於使用者選定的區間內進行對數線性（單極點）最小平方擬合"
                             "— 斜率由資料本身決定。"))

            sp_window_idx = None
            if show_sp and n_freq >= 4:
                f_lo, f_hi = float(f_arr[0]), float(f_arr[-1])
                _spkey = f"bode_spwin_{n}"
                _dflt  = (max(f_lo, f_hi * 0.5), f_hi)
                sp_win = st.slider(
                    i18n.tr("Single-pole fit window (GHz)", "單極點擬合區間 (GHz)"),
                    min_value=f_lo, max_value=f_hi,
                    value=st.session_state.get(_spkey, _dflt),
                    step=max((f_hi - f_lo) / 400.0, 1e-3),
                    key=_spkey)
                _il = int(np.searchsorted(f_arr, sp_win[0], side="left"))
                _ih = int(np.searchsorted(f_arr, sp_win[1], side="right")) - 1
                _il = max(0, min(_il, n_freq - 2))
                _ih = max(_il + 1, min(_ih, n_freq - 1))
                sp_window_idx = (_il, _ih)

            # f_max_target so extrap curves extend past the largest fT/fmax
            _ft_now  = d.get("fT Cross/Extrap (GHz)")
            _fmU_now = d.get("fmax U Cross/Extrap (GHz)")
            _maxes = [v for v in (_ft_now, _fmU_now, f_arr[-1] if n_freq else 0)
                      if v is not None and np.isfinite(v)]
            f_max_target = max(_maxes) * 1.2 if _maxes else None

            _bode_key = ("ind_bode", _dut_keys.get(n, n.encode()),
                          bool(show_20db), bool(show_sp), sp_window_idx,
                          xr, yr, bool(sh21), bool(su), bool(smag), c,
                          f_max_target)
            fig_bode, bode_xl = _cached_fig(_bode_key, lambda: make_bode(
                df_p, Path(n).stem, xr, yr, sh21, su, smag, c,
                show_20db=show_20db, show_sp=show_sp,
                sp_window_idx=sp_window_idx,
                extrap_f_max=f_max_target,
                return_excel_bytes=True))
            st.plotly_chart(fig_bode, width="stretch")

            # ── K-factor sub-plot (controlled by the global "K factor"
            #    sidebar checkbox `skk`).  Plotted on its own axis below
            #    the Bode plot because K is dimensionless and shares no
            #    natural scale with dB gain.  Dashed line at K=1 marks
            #    the unconditional-stability threshold.
            if skk and "K Factor" in df_p.columns:
                def _build_ind_kfactor():
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=df_p["Freq (GHz)"], y=df_p["K Factor"],
                        name="K", line=dict(color=c, width=2.5),
                        hovertemplate="Freq: %{x:.4f} GHz<br>"
                                       "K = %{y:.4f}<extra></extra>"))
                    fig.add_hline(y=1.0, line_dash="dash", line_color="#888",
                                  annotation_text="K = 1",
                                  annotation_position="right",
                                  annotation_font=dict(size=9, color="#888"))
                    _k_arr = df_p["K Factor"].values
                    _k_finite = _k_arr[np.isfinite(_k_arr)]
                    _y_top = max(float(_k_finite.max()) * 1.1, 2.0) if len(_k_finite) else 5.0
                    _y_bot = min(float(_k_finite.min()) * 1.1, 0.0) if len(_k_finite) else 0.0
                    fig.update_layout(
                        title=dict(text=f"{i18n.tr('K Factor (stability)', 'K 因子（穩定性）')} — {Path(n).stem}",
                                    font=dict(size=12)),
                        xaxis=dict(title=i18n.tr("Frequency (GHz)", "頻率 (GHz)"), type="log",
                                    range=[np.log10(max(xr[0], 1e-2)),
                                           np.log10(xr[1])],
                                    showgrid=True, gridcolor="#ebebeb"),
                        yaxis=dict(title="K", range=[_y_bot, _y_top],
                                    showgrid=True, gridcolor="#ebebeb"),
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=320, margin=dict(l=55, r=20, t=40, b=50),
                        hovermode="x unified")
                    return fig
                _k_fig = _cached_fig(
                    ("ind_kfactor", _dut_keys.get(n, n.encode()), xr, c),
                    _build_ind_kfactor)
                st.plotly_chart(_k_fig, width="stretch")

            # ── Excel download (standardised fT/fmax export) ─────────────────
            # One xlsx button: simulated columns (freq + each gain trace) and,
            # when extrapolation is in play, a side-by-side extrapolated block.
            if bode_xl is not None:
                _bode_tsv = xlsx_bytes_to_tsv(bode_xl)
                _cx, _cc = st.columns(2)
                _cx.download_button(
                    i18n.tr("⬇ xlsx", "⬇ xlsx 檔"),
                    data=bode_xl,
                    file_name=f"{Path(n).stem}_bode.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"bode_dl_{n}", width="stretch")
                if _bode_tsv:
                    copy_button(_bode_tsv, key=f"bode_{n}", container=_cc)
        with tb:
            _plat_fig = _cached_fig(
                ("ind_plateau", _dut_keys.get(n, n.encode()),
                 bool(sh21), bool(su), bool(smag), xr, c),
                lambda: make_plateau(df_p, d, Path(n).stem, xr, sh21, su, smag, c))
            st.plotly_chart(_plat_fig, width="stretch")
        with tc:
            smith_sub_plotly, smith_sub_mpl = st.tabs(
                ["Plotly", "Matplotlib"])
            with smith_sub_plotly:
                _smith_key = ("ind_smith", _dut_keys.get(n, n.encode()),
                               smith_f_min, smith_f_max, smith_max_r,
                               tuple(sorted(toggles.items())),
                               tuple(sorted(scales.items())))
                _smith_fig = _cached_fig(_smith_key, lambda: make_smith(
                    d["S_fin"], df_p["Freq (GHz)"].values,
                    smith_f_min, smith_f_max, toggles, scales,
                    Path(n).stem, max_r=smith_max_r))
                st.plotly_chart(_smith_fig, width="stretch")
            with smith_sub_mpl:
                # Two-column split (chart left, controls right) + auto
                # frequency-range annotation — same pattern as the SSM
                # extraction tab and RF simulator.  freq_hz is in Hz.
                _mpl_freq_hz = df_p["Freq (GHz)"].values * 1e9
                # Pool of the *other* uploaded files, so the user can overlay
                # them on this chart via the styling table's "➕ Add a file
                # trace" button (only offered when >1 file is loaded).
                _mpl_pool = [
                    {"label": Path(k).stem, "S": all_data[k]["S_fin"]}
                    for k in selected_files
                    if k != n and all_data.get(k, {}).get("S_fin") is not None]
                col_mpl_left, col_mpl_right = st.columns([1.2, 1])
                with col_mpl_right:
                    render_matplotlib_smith(
                        fname=n, topo_key="meas",
                        sets=[{"S": d["S_fin"], "label": Path(n).stem,
                               "kind": "line", "style": "solid"}],
                        default_multiplier=scales,
                        phase="controls", freq_hz=_mpl_freq_hz,
                        add_pool=_mpl_pool,
                    )
                with col_mpl_left:
                    render_matplotlib_smith(
                        fname=n, topo_key="meas",
                        sets=[{"S": d["S_fin"], "label": Path(n).stem,
                               "kind": "line", "style": "solid"}],
                        default_multiplier=scales,
                        phase="chart", freq_hz=_mpl_freq_hz,
                        add_pool=_mpl_pool,
                    )
        # ── Send this device to the SSM pages (no re-upload) ─────────────────
        with st.container(border=True):
            st.markdown(f"**🔁 {i18n.tr('Send this device to an SSM page', '將此元件傳送至 SSM 頁面')}**")
            # `De-embedding` is the note "" / "None" when no de-embed stage
            # fired (and S_fin then just aliases S_raw), so test the note itself
            # rather than S_fin (which is always non-None).
            _has_deemb = str(d.get("De-embedding") or "None") not in ("", "None")
            # Only offer the De-embedded / Raw choice when a de-embedding file
            # was actually uploaded — otherwise only raw S-parameters exist, so
            # the selector would be a pointless single "Raw" chip.  Hide it and
            # just send raw.
            if _has_deemb:
                hs1, hs2, hs3 = st.columns([1.4, 1, 1])
                # Values stay English ("De-embedded"/"Raw") — stage_lbl is
                # compared against them below; only the label localizes.
                _stage_labels_zh = {"De-embedded": "去嵌入後", "Raw": "原始"}
                with hs1:
                    stage_lbl = segmented_radio(
                        i18n.tr("S-parameters to send", "傳送的 S 參數"),
                        ["De-embedded", "Raw"], key=f"hand_stage_seg_{n}",
                        format_func=lambda s: i18n.tr(s, _stage_labels_zh[s]),
                        help=i18n.tr("De-embedded = pads/leads removed (fit intrinsic only). "
                                     "Raw = probe-level (fit Cpxx / Lx parasitics too).",
                                     "去嵌入後 = 已移除 pad/引線（僅擬合本質元件）。"
                                     "原始 = 探針層級（同時擬合 Cpxx / Lx 寄生參數）。"))
                _use_deemb = (stage_lbl == "De-embedded")
            else:
                hs2, hs3 = st.columns(2)
                _use_deemb = False
            _S_send = d["S_fin"] if _use_deemb else d["S_raw"]
            _stage = "deembedded" if _use_deemb else "raw"
            if hs2.button(i18n.tr("→ SSM Extraction", "→ 小訊號模型萃取"), key=f"hand_ext_{n}",
                          width="stretch", type="primary"):
                handoff.send(handoff.TARGET_EXTRACTION,
                             S=_S_send, freq=d["freq"], z0=d["z0"],
                             label=Path(n).stem, stage=_stage)
                st.switch_page(handoff.PAGE_EXTRACTION)
            if hs3.button(i18n.tr("→ Simulation & Fitting", "→ 模擬與擬合"), key=f"hand_sim_{n}",
                          width="stretch"):
                handoff.send(handoff.TARGET_SIMFIT,
                             S=_S_send, freq=d["freq"], z0=d["z0"],
                             label=Path(n).stem, stage=_stage)
                st.switch_page(handoff.PAGE_SIMFIT)

        with st.expander(i18n.tr("📋 Data Table", "📋 資料表")):
            if d["df_fin"] is not None:
                ta2,tb2=st.tabs([i18n.tr("De-embedded","去嵌入後"), i18n.tr("Raw","原始")])
                with ta2:
                    st.dataframe(df_p.round(4),width="stretch",hide_index=True)
                    copy_button(frames_to_tsv([("De-embedded",df_p.round(4))]),key=f"dtab_de_{n}")
                with tb2:
                    st.dataframe(d["df_raw"].round(4),width="stretch",hide_index=True)
                    copy_button(frames_to_tsv([("Raw",d["df_raw"].round(4))]),key=f"dtab_raw_{n}")
            else:
                st.dataframe(df_p.round(4),width="stretch",hide_index=True)
                copy_button(frames_to_tsv([("Data",df_p.round(4))]),key=f"dtab_{n}")

with tab_sum:
    if not all_data:
        st.info(i18n.tr("Upload files to generate summary.", "請上傳檔案以產生摘要。"))
    else:
        rows=[{"File":k,"De-embedding":d["De-embedding"],"Vce (V)":d["Vce (V)"],
               "Ib (µA)":round(d["Ib (A)"]*1e6,1) if d["Ib (A)"] else None,
               "fT Cross":d["fT Cross/Extrap (GHz)"],"fT Plat":d["fT Plateau (GHz)"],
               "fmax U Cross":d["fmax U Cross/Extrap (GHz)"],"fmax U Plat":d["fmax U Plateau (GHz)"]}
              for k,d in all_data.items()]
        sum_df=pd.DataFrame(rows)
        # Drop columns that carry no information — every value null / None / blank
        # (e.g. no de-embedding on any file, or no Ib recorded). "File" is always
        # kept so the nav button below can resolve a selected row to its device.
        _nullish=sum_df.replace({"None":pd.NA,"":pd.NA})
        sum_df=sum_df[[c for c in sum_df.columns
                       if c=="File" or _nullish[c].notna().any()]]
        fmt={c:"{:.4f}" for c in sum_df.columns if "Cross" in c or "Plat" in c}
        if "Vce (V)" in sum_df.columns: fmt["Vce (V)"]="{:.3f}"
        if "Ib (µA)" in sum_df.columns: fmt["Ib (µA)"]="{:.1f}"
        # Sortable table (click the fT / fmax headers to rank) with single-row
        # select. Picking a row reveals a button that jumps to that device's
        # Individual tab — so the user can rank by fT/fmax here, then dive
        # straight into the highest/lowest one.
        _sum_evt=st.dataframe(sum_df.style.format(fmt,na_rep="—"),
                              width="stretch",hide_index=True,
                              on_select="rerun",selection_mode="single-row",
                              key="sum_sel_table")
        _sel=_sum_evt.selection.rows if _sum_evt.selection else []
        if _sel:
            _sel_file=sum_df.iloc[_sel[0]]["File"]; _stem=Path(_sel_file).stem
            if st.button(i18n.tr(f"📁 Open “{_stem}” in the Individual tab →",
                                 f"📁 於單一檔案分頁開啟「{_stem}」→"),
                         key="sum_goto_ind",type="primary",width="stretch"):
                st.session_state["_pending_active_file"]=_sel_file
                st.session_state["_goto_individual"]=True
                st.rerun()
        else:
            st.caption(i18n.tr("Tip: sort by clicking a header, then select a row "
                               "to open that file in the Individual tab.",
                               "提示：點欄位標題排序，再選取一列即可於單一檔案分頁開啟該檔。"))
        date=datetime.now().strftime("%Y-%m-%d")
        d1,d2,d3=st.columns(3)
        with d1:
            st.download_button(i18n.tr("📥 Excel","📥 下載 Excel"),data=build_excel(sum_df,all_data),
                               file_name=f"RF_Extraction_{date}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               width="stretch")
        with d2:
            zbuf=io.BytesIO()
            with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("Summary.csv",sum_df.to_csv(index=False).encode())
                for k,d in all_data.items():
                    dp=d["df_fin"] if d["df_fin"] is not None else d["df_raw"]
                    zf.writestr(f"{Path(k).stem}.csv",dp.to_csv(index=False).encode())
            st.download_button(i18n.tr("📦 ZIP (CSV)","📦 下載 ZIP (CSV)"),data=zbuf.getvalue(),
                               file_name=f"RF_Extraction_{date}.zip",
                               mime="application/zip",width="stretch")
        with d3:
            copy_button(frames_to_tsv([("Summary",sum_df)]),key="sum_copy")

        # The nav button queued the file (read by tab_ind, which runs earlier)
        # and reran. Streamlit has no API to switch st.tabs, so nudge the DOM
        # from the component iframe: click the top-level Individual tab. It's the
        # only tab whose label carries 📁 (inner Bode/Plateau/Smith tabs don't)
        # so the match is language-agnostic; index-1 is a fallback. We POLL
        # because the tab-list may not be mounted yet the instant the iframe
        # loads, and stamp a nonce so Streamlit remounts the iframe (re-runs the
        # script) on every navigation. st.iframe embeds a raw HTML string in a
        # same-origin srcdoc iframe, so window.parent.document stays reachable.
        if st.session_state.pop("_goto_individual",False):
            st.iframe(
                """<script>
                (function(){
                  var pdoc;
                  try { pdoc = window.parent.document; } catch(e){ return; }
                  var tries = 0;
                  var timer = setInterval(function(){
                    tries++;
                    var tabs = pdoc.querySelectorAll('[role="tab"]');
                    var target = Array.prototype.find.call(tabs, function(b){
                      return b.innerText.indexOf('📁') !== -1;
                    });
                    if (!target) {
                      var lists = pdoc.querySelectorAll('[role="tablist"]');
                      if (lists.length) target = lists[0].querySelectorAll('[role="tab"]')[1];
                    }
                    if (target && target.getAttribute('aria-selected') !== 'true') {
                      target.click();
                    }
                    if ((target && target.getAttribute('aria-selected') === 'true') || tries > 60) {
                      clearInterval(timer);
                    }
                  }, 50);
                })();
                </script>
                <!-- nonce %s -->""" % (datetime.now().timestamp(),), height=0)

with tab_bd:
    render_batch_deembedding_tab(
        all_data=all_data,
        open_data=s2o,
        short_data=s2s,
        ui=dict(
            freq_min=freq_min, freq_max=freq_max,
            db_min=db_min, db_max=db_max, n_pts=n_pts,
            sh21=sh21, su=su, smag=smag,
            smith_f_min=smith_f_min, smith_f_max=smith_f_max, smith_max_r=smith_max_r,
            toggles={"S11":show_s11,"S22":show_s22,"S21":show_s21,"S12":show_s12},
            scales ={"S11":scale_s11,"S22":scale_s22,"S21":scale_s21,"S12":scale_s12},
        ),
        helpers=dict(
            y_to_s=y_to_s, compute_metrics=compute_metrics, extract_limit=extract_limit,
            make_bode=make_bode, make_smith=make_smith, card=metric_card, PALETTE=PALETTE,
        ),
    )