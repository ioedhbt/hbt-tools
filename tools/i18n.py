"""
i18n.py — Lightweight bilingual (English / 中文) registry for the IOED Lab
Portal.

This is the single source of truth for:
  * tool names + icons (used by both the sidebar navigation and each tool's
    in-page ``st.title`` so the two can never drift apart again), and
  * portal-level UI strings (group headers, home page, language toggle).

The active language lives in ``st.session_state["ui_lang"]`` and is driven by
the language radio rendered in ``IOED_Tool_Web.py``.  ``get_lang()`` defaults
to English when the widget has not been rendered yet (e.g. a tool launched
standalone outside the portal).

Phase 1 scope: the portal shell + every tool *title* is bilingual.  Individual
tool *bodies* still default to English and can adopt ``t()`` incrementally.
"""
from __future__ import annotations

import streamlit as st

EN = "English"
ZH = "中文"
LANGS = [EN, ZH]


def get_lang() -> str:
    """Return the active UI language, defaulting to English."""
    lang = st.session_state.get("ui_lang", EN)
    return lang if lang in LANGS else EN


def is_zh() -> bool:
    return get_lang() == ZH


# ─────────────────────────────────────────────────────────────────────────────
#  Tool registry — key → metadata.  ``path`` matches the st.Page file so the
#  home page's st.page_link and the navigation stay in sync.
# ─────────────────────────────────────────────────────────────────────────────
TOOLS: dict[str, dict] = {
    "rf_extract": {
        "icon": "📡", "group": "rf", "path": "tools/IOED_HBT_RF_extract.py",
        "en": "RF At a Glance", "zh": "RF 一覽",
        "desc_en": "De-embed S-parameters and extract fT / fmax with Smith & Bode charts.",
        "desc_zh": "去嵌入 S 參數並萃取 fT / fmax，產生 Smith 與 Bode 圖。",
        "features_en": [
            "Bulk-upload many DUT .s2p / .csv bias files",
            "3-step + batch Open/Short/Thru de-embedding",
            "fT / fmax extraction (crossing · extrapolation · plateau)",
            "Overlay / per-file Bode · Plateau · Smith (1 or many files)",
            "Summary table + Excel / ZIP export",
            "→ Hand a device to Extraction or Simulation & Fitting",
        ],
        "features_zh": [
            "批次上傳多個元件 .s2p / .csv 偏壓檔",
            "三步驟 + 批次 Open/Short/Thru 去嵌入",
            "fT / fmax 萃取（交越 · 外插 · 平台）",
            "疊圖 / 單檔 Bode · Plateau · Smith（單檔或多檔）",
            "彙總表 + Excel / ZIP 匯出",
            "→ 將元件傳送至萃取或模擬與擬合頁",
        ],
    },
    "ssm": {
        "icon": "🔬", "group": "rf", "path": "tools/SSM_extraction.py",
        "en": "Small Signal Model Extraction by Peeling",
        "zh": "剝離法小訊號模型萃取",
        "desc_en": "Extract Cheng T/π small-signal parameters from a DUT by peeling.",
        "desc_zh": "以剝離法從元件萃取 Cheng T/π 小訊號參數。",
        "features_en": [
            "Analytic peeling extraction — Cheng T and π",
            "Pad / lead de-embedding + access-R (Z-param · open-collector · cold-HBT)",
            "τ_total multi-file reference fit",
            "Extracted-parameter summary + persistent fit cache",
            "Receives a de-embedded device from RF At a Glance",
            "→ Send the extracted model onward to Simulation & Fitting",
        ],
        "features_zh": [
            "解析剝離萃取 — Cheng T 與 π",
            "焊墊 / 引線去嵌入 + 接觸電阻（Z 參數 · 開路集極 · 冷 HBT）",
            "τ_total 多檔參考擬合",
            "萃取參數彙總 + 永久擬合快取",
            "可接收 RF 一覽傳來的去嵌入元件",
            "→ 將萃取模型傳送至模擬與擬合頁",
        ],
    },
    "rf_sim": {
        "icon": "🛠️", "group": "rf", "path": "tools/RF_simulator.py",
        "en": "SSM Simulation & Fitting",
        "zh": "小訊號模型模擬與擬合",
        "desc_en": "Forward-simulate any SSM topology, or fit one to a measured file.",
        "desc_zh": "正向模擬任意小訊號模型，或對量測檔進行擬合。",
        "features_en": [
            "Forward-simulate Cheng T/π · Xu T · Kun-Yang HEMT · Open/Short pad",
            "🧩 Build & simulate any custom topology (netlist→Y→S solver)",
            "Optional: upload a measured .s2p to compare & fit",
            "Residual readout + visual and grid-sweep auto tuning",
            "fT / fmax (with extrapolation), τ_total, topology & Smith charts",
            "Receives devices / extracted values from the other RF pages",
        ],
        "features_zh": [
            "正向模擬 Cheng T/π · Xu T · Kun-Yang HEMT · 開路/短路焊墊",
            "🧩 建立並模擬任意自訂拓樸（netlist→Y→S 求解器）",
            "選用：上傳量測 .s2p 進行比較與擬合",
            "殘差顯示 + 視覺化與網格掃描自動調諧",
            "fT / fmax（含外插）、τ_total、拓樸圖與 Smith 圖",
            "可接收其他 RF 頁面傳來的元件 / 萃取值",
        ],
    },
    "ebl": {
        "icon": "🧮", "group": "process", "path": "tools/ebeam_calculator.py",
        "en": "EBL Calculator", "zh": "電子束微影計算機",
        "desc_en": "Compute JEOL ELS-7000 chip positions and exposure workflow.",
        "desc_zh": "計算 JEOL ELS-7000 晶片位置與曝光流程。",
    },
    "gummel": {
        "icon": "📈", "group": "tcad", "path": "tools/IOED_Gummel_Analyzer.py",
        "en": "Gummel Plot Analyzer", "zh": "Gummel 圖分析器",
        "desc_en": "Compare simulated Gummel plots against the UIUC reference.",
        "desc_zh": "比對模擬 Gummel 圖與 UIUC 參考資料。",
    },
    "b1500a": {
        "icon": "📊", "group": "dc", "path": "tools/B1500A_Plot.py",
        "en": "B1500A Plot & TLM", "zh": "B1500A 繪圖與 TLM",
        "desc_en": "View B1500A Excel data, extract parameters, and run TLM analysis.",
        "desc_zh": "檢視 B1500A Excel 資料、萃取參數並進行 TLM 分析。",
    },
    "hp4155a": {
        "icon": "📉", "group": "dc", "path": "tools/HP4155A_plot.py",
        "en": "HP4155A Quick Plot", "zh": "HP4155A 快速繪圖",
        "desc_en": "Quick interactive SMU plots from HP4155A data.",
        "desc_zh": "由 HP4155A 資料快速繪製 SMU 互動圖。",
    },
    "csv": {
        "icon": "🗂️", "group": "data", "path": "tools/csv_process.py",
        "en": "Measurement Data Multi-Process", "zh": "量測資料批次處理",
        "desc_en": "Batch-process and convert measurement CSV / CITI files.",
        "desc_zh": "批次處理並轉換量測 CSV / CITI 檔案。",
    },
}

# Domain groups, in sidebar display order.
GROUP_ORDER = ["rf", "process", "tcad", "dc", "data"]
GROUPS: dict[str, dict] = {
    "rf":      {"en": "RF Measurement",        "zh": "高頻量測"},
    "process": {"en": "Process",               "zh": "製程"},
    "tcad":    {"en": "Device Simulation",     "zh": "元件模擬"},
    "dc":      {"en": "DC Measurement",        "zh": "直流量測"},
    "data":    {"en": "Data Processing",       "zh": "資料處理"},
}

# Portal-level UI strings.
_UI: dict[str, dict] = {
    "portal_title":   {"en": "IOED Lab Unified Portal",    "zh": "IOED 實驗室整合平台"},
    "portal_caption": {"en": "Integrated device analysis & extraction platform",
                       "zh": "整合式元件分析與萃取平台"},
    "language_label": {"en": "🌐 Language",                "zh": "🌐 語言"},
    "start_group":    {"en": "Start",                      "zh": "開始"},
    "home_nav":       {"en": "Home",                       "zh": "首頁"},
    "home_title":     {"en": "Welcome to the IOED Lab Portal",
                       "zh": "歡迎使用 IOED 實驗室平台"},
    "home_intro":     {"en": "Pick a tool to get started. Tools are grouped by "
                             "measurement domain — open one from a card below or "
                             "from the sidebar.",
                       "zh": "請選擇工具開始使用。工具依量測領域分類 — 可從下方卡片或側邊欄開啟。"},
    "open_tool":      {"en": "Open",                       "zh": "開啟"},
    "how_it_works":   {"en": "ℹ️ How it works",            "zh": "ℹ️ 運作方式"},
    "load_example":   {"en": "📂 Load example files",       "zh": "📂 載入範例檔"},
    "using_example":  {"en": "Using bundled example files — upload your own to replace them.",
                       "zh": "正在使用內建範例檔 — 上傳自己的檔案即可取代。"},
    # SSM page – common controls
    "ssm_upload_label": {"en": "Upload DUT .s2p / .csv bias files",
                         "zh": "上傳 DUT .s2p / .csv 偏壓檔"},
    "clear_uploads":   {"en": "🗑️ Clear uploads",           "zh": "🗑️ 清除上傳"},
    "active_dut":      {"en": "Active DUT file",            "zh": "目前 DUT 檔案"},
    "run_ssm":         {"en": "▶ Run SSM Extraction",       "zh": "▶ 執行小訊號模型萃取"},
    "clear_ssm":       {"en": "✕ Clear SSM results",        "zh": "✕ 清除萃取結果"},
    "ssm_run_hint":    {"en": "Skipped until you run it, to keep the page fast.",
                        "zh": "在你執行前會略過，以維持頁面流暢。"},
    "ssm_clear_help":  {"en": "Frees cached computation for this file.",
                        "zh": "釋放此檔案的快取運算。"},
    "dev_dummy_head":  {"en": "## Device Dummy (Open-Short)",
                        "zh": "## 元件 Dummy（Open-Short）"},
    "dev_dummy_toggle": {"en": "Enable device-dummy de-embedding",
                         "zh": "啟用元件 dummy 去嵌入"},
    "dev_dummy_help":  {"en": "Optional — supplies the pad parasitics that SSM "
                              "de-embeds before extraction. Without it, pad "
                              "caps/leads default to 0.",
                        "zh": "選用 — 提供萃取前 SSM 去嵌入所需的 pad 寄生參數。"
                              "未提供時，pad 電容/引線預設為 0。"},
    "dev_open":        {"en": "Dev Open",                   "zh": "元件 Open"},
    "dev_short":       {"en": "Dev Short",                  "zh": "元件 Short"},
    "pad_deembed_on":  {"en": "✅ Pad de-embedding active.", "zh": "✅ Pad 去嵌入已啟用。"},
    "upload_or_example": {
        "en": "Upload at least one DUT `.s2p` / `.csv` file to begin, or load "
              "the bundled examples. Multiple bias files enable the multi-file "
              "Z-parameter and τ_total reference fits.",
        "zh": "請上傳至少一個 DUT `.s2p` / `.csv` 檔，或載入內建範例。多個偏壓檔可"
              "啟用多檔 Z 參數與 τ_total 參考擬合。"},
    "whats_new":      {"en": "What's new",                 "zh": "更新內容"},
    "ebl_corner_guide": {"en": "ℹ️ How corners are labeled",
                         "zh": "ℹ️ 角落標示說明"},
    "ebl_corner_note":  {"en": "Holder frame: x increases →, y increases ↑. In "
                               "Rectangular mode you edit one diagonal pair "
                               "(BL–TR or BR–TL); the other pair is computed.",
                         "zh": "夾具座標：x 向右增加，y 向上增加。矩形模式下只需編輯一組"
                               "對角（BL–TR 或 BR–TL），另一組會自動計算。"},
    # B1500A internal sub-views
    "b1500a_viewer":  {"en": "B1500A Excel Viewer",        "zh": "B1500A Excel 檢視器"},
    "b1500a_tlm":     {"en": "TLM Analysis",               "zh": "TLM 分析"},
    # ── Numbered workflow steps (consistent ①②③ guidance across tools) ──
    "ssm_step1":      {"en": "① Upload bias files",        "zh": "① 上傳偏壓檔"},
    "ssm_step2":      {"en": "② Select device & model",    "zh": "② 選擇元件與模型"},
    "ssm_step3":      {"en": "③ Run extraction",           "zh": "③ 執行萃取"},
    "b1500a_step1":   {"en": "① Upload Excel file",        "zh": "① 上傳 Excel 檔"},
    "b1500a_step2":   {"en": "② Choose sheet & data type", "zh": "② 選擇工作表與資料類型"},
    "gm_step1":       {"en": "① Upload simulation CSV",    "zh": "① 上傳模擬 CSV"},
    "gm_step2":       {"en": "② Compare with the UIUC target", "zh": "② 與 UIUC 目標比對"},
    # ── Gummel Plot Analyzer — body strings (migrated off hardcoded 中文) ──
    "gm_settings":      {"en": "Settings",                 "zh": "參數設定"},
    "gm_noise_head":    {"en": "Noise floor",              "zh": "量測底噪"},
    "gm_inject_noise":  {"en": "Inject noise",             "zh": "啟用模擬底噪"},
    "gm_ic_noise":      {"en": "Ic noise limit (A)",       "zh": "Ic 底噪上限 (A)"},
    "gm_ib_noise":      {"en": "Ib noise limit (A)",       "zh": "Ib 底噪上限 (A)"},
    "gm_scale_head":    {"en": "Plot scale & axis",        "zh": "圖表範圍控制"},
    "gm_y_scale":       {"en": "Current Y-axis scale",     "zh": "電流軸刻度"},
    "gm_xmin":          {"en": "X min (V)",                "zh": "X 最小值 (V)"},
    "gm_xmax":          {"en": "X max (V)",                "zh": "X 最大值 (V)"},
    "gm_auto_y":        {"en": "Auto-scale Y axes",        "zh": "Y 軸自動縮放"},
    "gm_cur_min":       {"en": "Current min (A)",          "zh": "電流最小值 (A)"},
    "gm_cur_max":       {"en": "Current max (A)",          "zh": "電流最大值 (A)"},
    "gm_beta_min":      {"en": "Beta min",                 "zh": "Beta 最小值"},
    "gm_beta_max":      {"en": "Beta max",                 "zh": "Beta 最大值"},
    "gm_physics_head":  {"en": "Physics params",           "zh": "物理參數"},
    "gm_n_imin":        {"en": "n calc I_min (A)",         "zh": "理想因子下限 (A)"},
    "gm_n_imax":        {"en": "n calc I_max (A)",         "zh": "理想因子上限 (A)"},
    "gm_vt":            {"en": "Thermal voltage Vt (V)",   "zh": "熱電壓 Vt (V)"},
    "gm_export_head":   {"en": "Export",                   "zh": "匯出設定"},
    "gm_export_name":   {"en": "Export file name",         "zh": "輸出檔名"},
    "gm_upload":        {"en": "Upload TonyPlot Gummel CSV", "zh": "上傳模擬 Gummel CSV"},
    "gm_tab_overlay":   {"en": "📊 Overlay vs UIUC",       "zh": "📊 疊加 UIUC"},
    "gm_tab_single":    {"en": "🔍 Single Check",          "zh": "🔍 單一比對"},
    "gm_tab_summary":   {"en": "📋 Summary",               "zh": "📋 摘要"},
    "gm_overlay_head":  {"en": "Overlay simulated plots with the UIUC target",
                         "zh": "與 UIUC 目標疊加模擬曲線"},
    "gm_select_all":    {"en": "Select all",               "zh": "全部選取"},
    "gm_clear_sel":     {"en": "Clear selection",          "zh": "清除選取"},
    "gm_multiselect":   {"en": "Files to overlay (type to search)",
                         "zh": "選擇要疊加的模擬檔案（可輸入關鍵字搜尋）"},
    "gm_waiting":       {"en": "Waiting for simulation data — the built-in UIUC "
                               "reference curve is shown below.",
                         "zh": "等待上傳模擬資料中… 下方為內建的 UIUC 基準曲線。"},
    "gm_single_head":   {"en": "Single-check calibration (dual-axis)",
                         "zh": "單一比對校正（雙軸）"},
    "gm_need_single":   {"en": "Upload a simulation CSV to run the single comparison.",
                         "zh": "請先上傳模擬 CSV 檔案才能進行單一比對分析。"},
    "gm_choose_single": {"en": "Choose a single simulation to analyze",
                         "zh": "選擇要分析的單一模擬"},
    "gm_col_metric":    {"en": "Metric",                   "zh": "指標"},
    "gm_col_sim":       {"en": "Simulated",                "zh": "模擬值"},
    "gm_col_target":    {"en": "UIUC target",              "zh": "目標值"},
    "gm_col_err":       {"en": "Error (%)",                "zh": "誤差 (%)"},
    "gm_err_table_head": {"en": "Calibration error table", "zh": "對位誤差表"},
    "gm_summary_head":  {"en": "Summary",                  "zh": "摘要"},
    "gm_need_summary":  {"en": "Upload a simulation CSV to generate the summary report.",
                         "zh": "請先上傳模擬 CSV 檔案產生摘要報表。"},
    "gm_dl_excel":      {"en": "📥 Download Excel report",  "zh": "📥 下載完整報告"},
    "gm_dl_zip":        {"en": "📦 Download ZIP (CSV)",     "zh": "📦 下載 CSV 壓縮包"},
}


def _pick(d: dict) -> str:
    """Select the ``en``/``zh`` field of a bilingual dict for the active lang."""
    return d["zh"] if is_zh() else d["en"]


def t(key: str) -> str:
    """Return a localized portal-level UI string."""
    return _pick(_UI[key])


def group_label(group_key: str) -> str:
    """Bilingual sidebar group header, e.g. ``RF Measurement / 高頻量測``."""
    g = GROUPS[group_key]
    return f"{g['en']} / {g['zh']}"


def tool_name(tool_key: str) -> str:
    """Localized tool name (no icon)."""
    return _pick(TOOLS[tool_key])


def tool_icon(tool_key: str) -> str:
    return TOOLS[tool_key]["icon"]


def title(tool_key: str) -> str:
    """Localized tool name for a tool's in-page ``st.title``.

    The icon is intentionally omitted here: the sidebar navigation already
    shows it next to the active page, so repeating a large decorative emoji
    in the H1 only adds visual noise.  Use :func:`tool_icon` if a caller
    genuinely needs the glyph.
    """
    return tool_name(tool_key)


def tool_desc(tool_key: str) -> str:
    m = TOOLS[tool_key]
    return m["desc_zh"] if is_zh() else m["desc_en"]


def tool_features(tool_key: str) -> list[str]:
    """Localized bullet list of a tool's key functionalities (empty if none)."""
    m = TOOLS[tool_key]
    return m.get("features_zh" if is_zh() else "features_en", [])
