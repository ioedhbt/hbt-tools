"""
csv_process.py — B1500A CSV → preset template (Family / Transfer / Gummel)
batch converter.

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.
"""
__version__ = "1.0"

import streamlit as st

from tools.common import i18n
from tools.common import handoff
from tools.SSM.helpers import unique_sheet_name
import pandas as pd
import numpy as np
import io
import re
import os
import zipfile

st.title(i18n.title("csv"))
st.caption(i18n.tool_desc("csv"))


# --- Utility functions ---
def read_smu_table(raw):
    """Parse an SMU dump that may be whitespace- or comma-delimited.

    ``sep=r"\\s+"`` (not the old ``delim_whitespace=True``, removed in
    pandas 2.2/3.x — it raised ``TypeError`` on every modern install, and
    the bare ``except`` around it silently swallowed that into a comma
    parse that produced one garbage column, no matching SMU columns, and
    an empty result with no error shown).

    Whether the file was actually whitespace-delimited is now decided by
    the column count, not by an exception — so a genuine parse failure
    propagates instead of being masked.
    """
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", engine="python")
    if df.shape[1] < 2:                     # not whitespace-delimited
        df = pd.read_csv(io.StringIO(raw), engine="python")
    df.columns = df.columns.str.strip()
    return df


def detect_header_row(text):
    """Detect header row by searching for 'DataName' in the first column (case-insensitive)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        first_cell = line.split(",")[0].strip()
        if re.fullmatch(r"dataname", first_cell, flags=re.IGNORECASE):
            return i
    return 0


def process_b1500a_file(file, template_cols, header_idx, preset):
    text = file.getvalue().decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(text), header=header_idx, engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    if preset == "Ic-Vc Family preset":
        if not all(c in df.columns for c in ["Vc", "Ic", "Ib"]):
            raise ValueError("Missing Vc/Ic/Ib for Family curve")

        vc_axis = np.sort(df["Vc"].unique())
        out = pd.DataFrame({"Vc": vc_axis})

        for ib_val, grp in df.groupby("Ib"):
            series = (
                grp.set_index("Vc")["Ic"]
                .reindex(vc_axis)
                .reset_index(drop=True)
            )
            out[f"Ic at Ib={format_ib_label(ib_val)}"] = series

        return out

    return df[[c for c in template_cols if c in df.columns]]


def process_file(file, selected_cols, template_header_idx):
    text = file.getvalue().decode("utf-8", errors="ignore")
    header_idx = template_header_idx if template_header_idx is not None else detect_header_row(text)
    df = pd.read_csv(io.StringIO(text), header=header_idx, engine="python", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df = df[[c for c in selected_cols if c in df.columns]]
    return df


def sanitize_sheet_name(name):
    name = name.split(";")[0].strip()
    name = re.sub(r'[:\\/*?\[\]]', '_', name)
    return name[:31]


def detect_preset_from_filename(filename):
    prefix = filename.strip().lower()[:2]
    if prefix == "tl":
        return "TLM preset"
    elif prefix == "bc":
        return "BC diode preset"
    elif prefix == "be":
        return "BE diode preset"
    elif prefix == "gu":
        return "Gummel preset"
    elif prefix == "fa":
        return "Ic-Vc Family preset"
    return "none"


def preset_to_type(preset):
    return {
        "Ic-Vc Family preset": "Family",
        "BE diode preset": "BE",
        "BC diode preset": "BC",
        "Gummel preset": "Gummel",
        "TLM preset": "TLM",
    }.get(preset, "Other")


def get_preset_columns(preset, available_cols):
    available_cols = [c.strip() for c in available_cols]
    presets = {
        "BC diode preset": ["Vb", "Ic_abs"],
        "BE diode preset": ["Vb", "Ie_abs"],
        "Gummel preset": ["Vb", "Ib", "Ic", "Beta"],
        "Ic-Vc Family preset": ["Vc", "Ib", "Ic"],
        "TLM preset": ["I1", "Rsa", "Vsa"],
    }
    wanted = presets.get(preset, [])
    return [c for c in available_cols if c in wanted]


def format_output_filename(preset):
    if preset == "BC diode preset":
        return "BC batch output.xlsx"
    elif preset == "BE diode preset":
        return "BE batch output.xlsx"
    elif preset == "Gummel preset":
        return "Gummel batch output.xlsx"
    elif preset == "Ic-Vc Family preset":
        return "Family batch output.xlsx"
    elif preset == "TLM preset":
        return "TLM batch output.xlsx"
    return "batch_output.xlsx"


def convert_units(val):
    units = [("A", 1), ("mA", 1e-3), ("uA", 1e-6), ("nA", 1e-9)]
    for unit, scale in units:
        if 1 <= val / scale < 1000:
            return val / scale, unit
    return val, "A"


def _lines(content):
    return [ln.rstrip() for ln in content.splitlines()]


def parse_var_headers(content):
    matches = re.findall(r'(?mi)^\s*VAR\s+(\S+)\s+MAG\s+(\d+)\s*$', content)
    return [(m[0], int(m[1])) for m in matches]


def parse_data_header(content):
    m = re.search(r'(?mi)^\s*DATA\s+(\S+)\s+MAG', content)
    return m.group(1) if m else None


def extract_var_list_blocks(lines):
    blocks = []
    i = 0
    while i < len(lines):
        tokens = lines[i].strip().split()
        if len(tokens) > 0 and tokens[0].upper() == "VAR_LIST_BEGIN":
            vals = []
            j = i + 1
            while j < len(lines):
                tks = lines[j].strip().split()
                if tks and tks[0].upper() == "VAR_LIST_END":
                    break
                for tok in tks:
                    try:
                        vals.append(float(tok.replace('E', 'e')))
                    except Exception:
                        continue
                j += 1
            blocks.append(vals)
            i = j + 1
        else:
            i += 1
    return blocks


def extract_data_blocks(lines):
    blocks = []
    i = 0
    while i < len(lines):
        tokens = lines[i].strip().split()
        if len(tokens) > 0 and tokens[0].upper() == "BEGIN":
            vals = []
            j = i + 1
            while j < len(lines):
                tks = lines[j].strip().split()
                if tks and tks[0].upper() == "END":
                    break
                for tok in tks:
                    try:
                        vals.append(float(tok.replace('E', 'e')))
                    except Exception:
                        continue
                j += 1
            blocks.append(vals)
            i = j + 1
        else:
            i += 1
    return blocks


def parse_gummel_from_text(content):
    lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    rows = []
    for ln in lines:
        toks = ln.split()
        if len(toks) < 3: continue
        try:
            vce = float(toks[0].replace('E', 'e'))
            ib = float(toks[1].replace('E', 'e'))
            ic = float(toks[2].replace('E', 'e'))
            rows.append((vce, ib, ic))
        except Exception:
            continue
    if not rows: raise ValueError("No numeric rows found for Gummel format.")
    df = pd.DataFrame(rows, columns=["Vce", "Ib", "Ic"])
    df["Beta"] = df["Ic"] / df["Ib"].replace({0: np.nan})
    return df


def parse_diode_by_header(content):
    lines = _lines(content)
    var_hdrs = parse_var_headers(content)
    data_name = parse_data_header(content) or "DATA"
    var_blocks = extract_var_list_blocks(lines)
    data_blocks = extract_data_blocks(lines)
    if not var_hdrs or not var_blocks or not data_blocks:
        raise ValueError("Missing header or data for diode.")

    var_name, mag_count = var_hdrs[0]
    x = var_blocks[0][:mag_count]

    if any("#Base-Emitter Diode" in ln for ln in lines):
        diode_type = "BE"
    elif any("#Base-Collector Diode" in ln for ln in lines):
        diode_type = "BC"
    else:
        diode_type = "Diode"

    y_raw = data_blocks[0][:mag_count]
    y = []
    for val in y_raw:
        try:
            f = float(val)
            if diode_type in ["BE", "BC"]: f = abs(f)
            y.append(f)
        except:
            continue
    x = x[:len(y)]
    df = pd.DataFrame({var_name: x, data_name: y})
    return df, diode_type


def parse_family_by_header(content):
    lines = _lines(content)
    var_hdrs = parse_var_headers(content)
    data_names = []
    for ln in lines:
        t = ln.strip().split()
        if len(t) >= 3 and t[0].upper() == "DATA":
            data_names.append(t[1])
    var_blocks = extract_var_list_blocks(lines)
    data_blocks = extract_data_blocks(lines)
    if len(var_hdrs) < 2 or not data_blocks:
        raise ValueError("Missing header or data for Family format.")

    (var1_name, n1), (var2_name, n2) = var_hdrs[0], var_hdrs[1]
    var1_list = var_blocks[0][:n1]
    var2_list = var_blocks[1][:n2]
    df = pd.DataFrame({var1_name: var1_list})
    expected = n1 * n2

    ordered_pairs = []
    if "L" in data_names:
        idx = data_names.index("L")
        ordered_pairs.append(("L", data_blocks[idx]))
    if "Ic" in data_names:
        idx = data_names.index("Ic")
        ordered_pairs.append(("Ic", data_blocks[idx]))
    for name, block in zip(data_names, data_blocks):
        if name not in ["L", "Ic"]:
            ordered_pairs.append((name, block))

    for data_name, data in ordered_pairs:
        padded = np.full(expected, np.nan)
        padded[: min(len(data), expected)] = data[: min(len(data), expected)]
        arr = padded.reshape((n2, n1))
        for i in range(n2):
            ib_val = var2_list[i]
            col_name = f"{data_name} at {var2_name}={ib_val}"
            df[col_name] = arr[i, :]
    return df


def parse_gummel_vb_ib_ic(content):
    rows = []
    for ln in content.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        toks = ln.split()
        if len(toks) < 3: continue
        try:
            vb = float(toks[0].replace("E", "e"))
            ib = float(toks[1].replace("E", "e"))
            ic = float(toks[2].replace("E", "e"))
            rows.append((vb, ib, ic))
        except Exception:
            continue
    if not rows: raise ValueError("No numeric rows found for Vb-Ib-Ic Gummel format.")
    df = pd.DataFrame(rows, columns=["Vb", "Ib", "Ic"])
    df["Beta"] = df["Ic"] / df["Ib"].replace({0: np.nan})
    return df


def parse_citi_file(content):
    var_hdrs = parse_var_headers(content)
    if not var_hdrs:
        try:
            return parse_gummel_vb_ib_ic(content), "Gummel"
        except Exception:
            return parse_gummel_from_text(content), "Gummel"
    elif len(var_hdrs) == 1:
        return parse_diode_by_header(content)
    else:
        return parse_family_by_header(content), "Family"


def parse_special_csv(content):
    lines = content.splitlines()
    numeric_rows = []
    for ln in lines:
        parts = ln.split(",")
        try:
            nums = [float(p) for p in parts]
            numeric_rows.append(nums)
        except Exception:
            continue
    if not numeric_rows: raise ValueError("No numeric data found in CSV.")
    header = ["Ib", "Vbe", "Ic", "L", "dL/dI", "dV/dI", "beta"]
    df = pd.DataFrame(numeric_rows, columns=header[: len(numeric_rows[0])])
    return df


def parse_smu_table(df, smu_map):
    out = {}
    for role, (vcol, icol) in smu_map.items():
        if vcol in df.columns and icol in df.columns:
            out[role] = (df[vcol].astype(float), df[icol].astype(float))
    return out


def group_family(vc, ic, ib):
    vc_axis = np.sort(vc.unique())
    grouped = {}
    for ib_val, grp in pd.DataFrame({"Vc": vc, "Ic": ic, "Ib": ib}).groupby("Ib"):
        series = grp.set_index("Vc")["Ic"].reindex(vc_axis).reset_index(drop=True)
        grouped[ib_val] = series
    return vc_axis, grouped


def format_ib_label(ib):
    val, unit = convert_units(abs(ib))
    sign = "-" if ib < 0 else ""
    return f"{sign}{val:.3g}{unit}"


_PAGE_LABELS_ZH = {
    "B1500A Smart Batch Tool": "B1500A 智慧批次工具",
    "TLM Resistance Avg": "TLM 電阻平均",
    "E5270B citi File Tool": "E5270B CITI 檔案工具",
    "HP4155A Data Processing Tool": "HP4155A 資料處理工具",
    "B1500A Column Selection & Batch": "B1500A 欄位選取與批次",
}
# Values stay English — `page` is compared against literal strings throughout
# this file (if/elif branches below); only the on-screen label localizes.
page = st.sidebar.radio(
    i18n.tr("Choose page:", "選擇頁面:"),
    list(_PAGE_LABELS_ZH.keys()),
    format_func=lambda p: i18n.tr(p, _PAGE_LABELS_ZH[p]),
)

if page == "B1500A Smart Batch Tool":
    st.header(i18n.tr("B1500A Smart Batch Tool", "B1500A 智慧批次工具"))
    st.caption(i18n.tr(
        "Upload multiple B1500A CSV files. The tool auto-detects measurement type and prepares batch downloads.",
        "上傳多個 B1500A CSV 檔案。工具會自動偵測量測類型並準備批次下載。"))

    uploaded_files = st.file_uploader(
        i18n.tr("Upload B1500A CSV files", "上傳 B1500A CSV 檔案"),
        type=["csv"], accept_multiple_files=True)
    if not uploaded_files: st.stop()

    processed = []
    for f in uploaded_files:
        try:
            raw = f.getvalue().decode("utf-8", errors="ignore")
            header_idx = detect_header_row(raw)
            preset = detect_preset_from_filename(f.name)
            group_type = preset_to_type(preset)

            df_tmp = pd.read_csv(io.StringIO(raw), header=header_idx, engine="python", on_bad_lines="skip")
            df_tmp.columns = df_tmp.columns.str.strip()
            cols = get_preset_columns(preset, df_tmp.columns.tolist())

            df_out = process_b1500a_file(f, cols, header_idx, preset)
            processed.append((f.name, preset, group_type, df_out))
        except Exception as e:
            st.error(f"{f.name}: {e}")

    if processed:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            seen_zip = {}
            for fname, _, _, df in processed:
                buf = io.BytesIO()
                # One sheet per workbook here, so the sheet name can't
                # collide — but the *zip entry* can, which would drop a
                # file just as silently.  Dedupe on the entry name.
                sheet = unique_sheet_name(fname.rsplit(".", 1)[0], seen_zip)
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name=sheet, index=False)
                zf.writestr(f"{sheet}.xlsx", buf.getvalue())

        st.download_button(
            i18n.tr("📦 Download each data as an Excel file (ZIP)",
                    "📦 下載各檔案的 Excel 檔（ZIP）"),
            data=zip_buf.getvalue(),
            file_name="B1500A_Individual_Files.zip", mime="application/zip")

        type_files = {}
        for fname, preset, gtype, df in processed:
            type_files.setdefault(gtype, []).append((fname, df))

        zip_buf2 = io.BytesIO()
        with zipfile.ZipFile(zip_buf2, "w", zipfile.ZIP_DEFLATED) as zf:
            for gtype, items in type_files.items():
                if not items: continue
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    seen = {}
                    for fname, df in items:
                        sheet = unique_sheet_name(fname.rsplit(".", 1)[0], seen)
                        df.to_excel(writer, sheet_name=sheet, index=False)
                out_name = {"Family": "IcVc_Family.xlsx", "BE": "BE_Diode.xlsx", "BC": "BC_Diode.xlsx",
                            "Gummel": "Gummel.xlsx", "TLM": "TLM.xlsx", "Other": "Other.xlsx"}.get(gtype,
                                                                                                   f"{gtype}.xlsx")
                zf.writestr(out_name, buf.getvalue())

        st.download_button(
            i18n.tr("📊 Download grouped files by measurement type", "📊 依量測類型下載分組檔案"),
            data=zip_buf2.getvalue(),
            file_name="B1500A_Grouped_By_Type.zip", mime="application/zip")

        # --- Hand off to DC Analysis, GROUPED BY MEASUREMENT TYPE ---------------
        # Send one workbook per type (mirrors the "grouped by measurement type"
        # ZIP above): each payload entry is a multi-sheet file where every sheet
        # is one measurement.  In DC Analysis the file dropdown then lists the
        # types and the sheet picker lists the measurements within each type.
        _VIEWER_DTYPE = {"Family": "Family", "Gummel": "Gummel", "BE": "Diode", "BC": "Diode"}
        _GROUP_NAME = {"Family": "IcVc_Family.xlsx", "BE": "BE_Diode.xlsx",
                       "BC": "BC_Diode.xlsx", "Gummel": "Gummel.xlsx"}
        payload = []
        for gtype, items in type_files.items():
            if gtype not in _VIEWER_DTYPE or not items:
                continue
            sheets = {}
            for fname, df in items:
                base = sanitize_sheet_name(fname.rsplit(".", 1)[0]) or "Sheet"
                sheet, _k = base, 2
                while sheet in sheets:          # keep every measurement's sheet
                    sheet = f"{base[:28]}_{_k}"  # (dict would silently drop dups)
                    _k += 1
                sheets[sheet] = df
            payload.append({"name": _GROUP_NAME[gtype], "sheets": sheets,
                            "dtype": _VIEWER_DTYPE[gtype]})
        skipped = sum(len(v) for k, v in type_files.items()
                      if k not in _VIEWER_DTYPE)
        if st.button(i18n.tr("🔬 Analyze Data in DC Analysis", "🔬 在 DC 分析中檢視"), type="primary"):
            if not payload:
                st.warning(i18n.tr(
                    "No Family/Gummel/BE/BC files to analyze (TLM/Other are skipped).",
                    "沒有可分析的 Family/Gummel/BE/BC 檔案（TLM/Other 已略過）。"))
            else:
                handoff.send_dc(payload)
                st.switch_page(handoff.PAGE_DC_ANALYSIS)
        if skipped:
            st.caption(i18n.tr(
                f"{skipped} TLM/Other file(s) skipped — not viewer-compatible.",
                f"已略過 {skipped} 個 TLM/Other 檔案 — 檢視器不支援此類型。"))

elif page == "B1500A Column Selection & Batch":
    st.header(i18n.tr("R307B B1500A CSV Batch Processing Tool", "R307B B1500A CSV 批次處理工具"))
    st.header(i18n.tr("Step 1: Upload a sample CSV", "步驟一：上傳範例 CSV"))

    sample_file = st.file_uploader(
        i18n.tr("Upload sample CSV", "上傳範例 CSV"), type=["csv"], key="sample")
    if sample_file:
        text = sample_file.getvalue().decode("utf-8", errors="ignore")
        header_idx = detect_header_row(text)
        df = pd.read_csv(io.StringIO(text), header=header_idx, engine="python", on_bad_lines="skip")
        df.columns = df.columns.str.strip()

        detected_preset = detect_preset_from_filename(sample_file.name)
        # Values stay English — preset_choice flows into get_preset_columns(),
        # format_output_filename() and preset_to_type() as a literal key;
        # only the displayed label localizes via format_func.
        _preset_opts = ["none", "BC diode preset", "BE diode preset", "Gummel preset",
                        "Ic-Vc Family preset", "TLM preset"]
        _preset_labels_zh = {
            "none": "無",
            "BC diode preset": "BC 二極體預設",
            "BE diode preset": "BE 二極體預設",
            "Gummel preset": "Gummel 預設",
            "Ic-Vc Family preset": "Ic-Vc 族群預設",
            "TLM preset": "TLM 預設",
        }
        preset_choice = st.selectbox(
            i18n.tr("Choose a preset:", "選擇預設:"),
            _preset_opts,
            index=_preset_opts.index(detected_preset) if detected_preset else 0,
            format_func=lambda p: i18n.tr(p, _preset_labels_zh[p]),
        )

        if preset_choice != "none":
            preset_cols = get_preset_columns(preset_choice, df.columns.tolist())
            selected_cols = preset_cols
        else:
            preset_cols = []
            selected_cols = df.columns.tolist()

        st.write(i18n.tr("Detected header columns:", "偵測到的欄位:"))
        selected_cols = st.multiselect(
            i18n.tr("Select columns to keep", "選擇要保留的欄位"),
            df.columns.tolist(), default=selected_cols,
            key="col_select_widget",
            placeholder=i18n.tr("Choose options", "請選擇項目"))

        if preset_choice != "none" and set(selected_cols) != set(preset_cols):
            preset_choice = "none"

        if st.button(i18n.tr("💾 Save template", "💾 儲存範本")):
            st.session_state.batch_template_cols = selected_cols
            st.session_state.batch_template_header_idx = header_idx
            st.session_state.batch_preset_choice = preset_choice
            st.success(i18n.tr(f"Template saved with {len(selected_cols)} columns.",
                               f"範本已儲存，共 {len(selected_cols)} 個欄位。"))

    if "batch_template_cols" in st.session_state:
        st.subheader(i18n.tr("Step 2: Upload batch CSV files", "步驟二：上傳批次 CSV 檔案"))
        batch_files = st.file_uploader(
            i18n.tr("Upload multiple CSV files", "上傳多個 CSV 檔案"),
            type=["csv"], accept_multiple_files=True,
            key="batch")

        if batch_files and st.button(i18n.tr("⚡ Process Batch", "⚡ 批次處理")):
            all_sheets = {}
            seen = {}
            for f in batch_files:
                df_proc = process_file(f, st.session_state.batch_template_cols,
                                       st.session_state.batch_template_header_idx)
                raw_name = os.path.splitext(f.name)[0]
                # Keyed by sheet name, so two uploads whose names match after
                # the 31-char Excel truncation used to silently drop one here,
                # before the workbook was even built.
                all_sheets[unique_sheet_name(raw_name, seen)] = df_proc

            out_buffer = io.BytesIO()
            out_filename = format_output_filename(st.session_state.get("batch_preset_choice", "none"))
            with pd.ExcelWriter(out_buffer, engine="openpyxl") as writer:
                for sheet, df in all_sheets.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)

            st.session_state.batch_ready = {"data": out_buffer.getvalue(), "filename": out_filename}

        if "batch_ready" in st.session_state:
            st.download_button(
                i18n.tr("📥 Download processed Excel", "📥 下載處理後的 Excel"),
                data=st.session_state.batch_ready["data"],
                file_name=st.session_state.batch_ready["filename"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            if st.button(i18n.tr("🔄 Refresh", "🔄 重新整理")):
                for key in ["batch_template_cols", "batch_template_header_idx", "batch_preset_choice", "batch_ready"]:
                    if key in st.session_state: del st.session_state[key]
                st.rerun()

elif page == "TLM Resistance Avg":
    st.header(i18n.tr("TLM Resistance Average Calculator", "TLM 電阻平均計算機"))
    excel_file = st.file_uploader(
        i18n.tr("Upload TLM Excel file. The file name should be 'TLM batch output.xlsx'.",
                "上傳 TLM Excel 檔案。檔名應為 'TLM batch output.xlsx'。"),
        type=["xlsx"], key="tlm_batch")
    if excel_file:
        xls = pd.ExcelFile(excel_file)
        sheet_names = xls.sheet_names
        st.write(f"{i18n.tr('Found sheets:', '找到的工作表:')} {sheet_names}")
        resistance_avgs = []
        chosen_col = st.text_input(
            i18n.tr("Enter column name for resistance (case-sensitive):", "輸入電阻欄位名稱（區分大小寫）:"),
            value="Rsa")

        if st.button(i18n.tr("⚡ Process Averages", "⚡ 計算平均值")):
            for sheet in sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet)
                avg_val = df[chosen_col].mean() if chosen_col in df.columns else None
                resistance_avgs.append((sheet, avg_val))
            result_df = pd.DataFrame(resistance_avgs, columns=["Sheet", "AvgResistance"])
            out_buffer = io.BytesIO()
            result_df.to_excel(out_buffer, index=False)
            st.dataframe(result_df)
            st.download_button(
                i18n.tr("📥 Download averages Excel", "📥 下載平均值 Excel"),
                data=out_buffer.getvalue(), file_name="TLM_avg_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif page == "E5270B citi File Tool":
    st.header(i18n.tr("CITI File → Excel Converter", "CITI 檔案 → Excel 轉換器"))
    uploaded_files = st.file_uploader(
        i18n.tr("Upload .citi or .txt files", "上傳 .citi 或 .txt 檔案"),
        accept_multiple_files=True,
        type=["citi", "txt", "csv"])
    if "citi_preview_visible" not in st.session_state:
        st.session_state.citi_preview_visible = False

    if uploaded_files:
        parsed_sheets = []
        for f in uploaded_files:
            try:
                raw = f.read().decode("utf-8", errors="ignore")
                name = f.name.rsplit(".", 1)[0]
                if f.name.lower().endswith(".csv"):
                    df = parse_special_csv(raw)
                    ftype = "CSV"
                else:
                    df, ftype = parse_citi_file(raw)
                parsed_sheets.append((name, df, ftype))

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    # One sheet per workbook — can't collide, but `name` may
                    # still carry characters Excel rejects.
                    df.to_excel(writer, index=False,
                                sheet_name=unique_sheet_name(name, {}))
                st.download_button(
                    f"📥 {i18n.tr('Download', '下載')} {name}_excel.xlsx",
                    data=buf.getvalue(), file_name=f"{name}_excel.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_{name}")
            except Exception as e:
                st.error(f"❌ {f.name}: {e}")

        if parsed_sheets:
            if st.button(i18n.tr("Show/Hide Preview", "顯示/隱藏預覽")):
                st.session_state.citi_preview_visible = not st.session_state.citi_preview_visible

            if st.session_state.citi_preview_visible:
                st.subheader(i18n.tr("File Preview", "檔案預覽"))
                for fname, df, ftype in parsed_sheets:
                    st.write(f"{fname} ({ftype})")
                    st.dataframe(df.head(), height=200)

            st.subheader(i18n.tr("Grouped Downloads", "分組下載"))
            groups = ["BE", "BC", "Gummel", "Family"]
            for g in groups:
                group_sheets = [(n, df) for n, df, ftype in parsed_sheets if ftype == g]
                if group_sheets:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        seen = {}
                        for n, df in group_sheets:
                            df.to_excel(writer, index=False,
                                        sheet_name=unique_sheet_name(
                                            n.rsplit(".", 1)[0], seen))
                    buf.seek(0)
                    st.download_button(
                        f"📥 {i18n.tr('Download', '下載')} {g} Excel "
                        f"({len(group_sheets)} {i18n.tr('files', '個檔案')})",
                        data=buf,
                        file_name=f"{g.upper()} Batch Output.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.subheader(i18n.tr("Download All", "下載全部"))
            all_buf = io.BytesIO()
            with pd.ExcelWriter(all_buf, engine="openpyxl") as writer:
                seen = {}
                for n, df, _ in parsed_sheets:
                    df.to_excel(writer, index=False,
                                sheet_name=unique_sheet_name(
                                    n.rsplit(".", 1)[0], seen))
            all_buf.seek(0)
            st.download_button(
                i18n.tr("📦 Download ALL files", "📦 下載全部檔案"),
                data=all_buf, file_name="All_CITI_Converted.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info(i18n.tr("Upload one or more .citi files (BE/BC/Family/Gummel).",
                        "請上傳一個以上的 .citi 檔案（BE/BC/Family/Gummel）。"))

elif page == "HP4155A Data Processing Tool":
    st.header(i18n.tr("HP4155A SMU Table Processor", "HP4155A SMU 表格處理器"))
    uploaded_files = st.file_uploader(
        i18n.tr("Upload raw data file(s)", "上傳原始資料檔"), accept_multiple_files=True)
    if not uploaded_files: st.stop()

    results = []
    for uploaded in uploaded_files:
        raw = uploaded.getvalue().decode("utf-8", errors="ignore")
        df = read_smu_table(raw)
        results.append((uploaded.name or "data", df))

    st.subheader(i18n.tr("SMU role assignment", "SMU 角色指定"))
    # Values stay English — lower-cased into smu_role_map's dict keys
    # ("collector"/"base"/…) which downstream parse_smu_table() looks up
    # by literal name; only the on-screen label localizes.
    roles = ["Collector", "Base", "Emitter", "PD", "None"]
    _role_labels_zh = {"Collector": "集極 (Collector)", "Base": "基極 (Base)",
                       "Emitter": "射極 (Emitter)", "PD": "PD", "None": "無"}
    smu_cols = [("V1", "I1"), ("V2", "I2"), ("V3", "I3"), ("V4", "I4")]
    smu_role_map = {}
    for vcol, icol in smu_cols:
        choice = st.selectbox(f"{vcol}/{icol}", roles, key=f"hp_batch_{vcol}_{icol}",
                              format_func=lambda r: i18n.tr(r, _role_labels_zh[r]))
        if choice != "None": smu_role_map[choice.lower()] = (vcol, icol)

    st.subheader(i18n.tr("Measurement type", "量測類型"))
    # Values stay English — `meas` is compared against these literals
    # throughout the processing block below; only the label localizes.
    _meas_opts = ["family Ic-Vc", "family L-Ic-Vc", "gummel", "BE diode", "BC diode"]
    _meas_labels_zh = {
        "family Ic-Vc": "族群 Ic-Vc", "family L-Ic-Vc": "族群 L-Ic-Vc",
        "gummel": "Gummel", "BE diode": "BE 二極體", "BC diode": "BC 二極體",
    }
    meas = st.selectbox(i18n.tr("Select measurement", "選擇量測"), _meas_opts,
                        format_func=lambda m: i18n.tr(m, _meas_labels_zh[m]))
    responsivity = st.number_input(
        i18n.tr("Responsivity (A/W)", "響應率 (A/W)"), value=0.35,
        format="%.3f") if meas == "family L-Ic-Vc" else None

    if st.button(i18n.tr("⚡ Process", "⚡ 處理")):
        per_file_outputs = []
        for uploaded in uploaded_files:
            raw = uploaded.getvalue().decode("utf-8", errors="ignore")
            df = read_smu_table(raw)
            fname = sanitize_sheet_name(uploaded.name or "data")
            parsed = parse_smu_table(df, smu_role_map)

            if meas in ["family Ic-Vc", "family L-Ic-Vc", "gummel"] and any(
                k not in parsed for k in ["collector", "base", "emitter"]): continue
            if meas in ["BE diode", "BC diode"] and any(k not in parsed for k in (
            ["base", "emitter"] if meas == "BE diode" else ["base", "collector"])): continue

            buf = io.BytesIO()
            if meas.startswith("family"):
                vc, ic = parsed["collector"]
                _, ib = parsed["base"]
                vc_axis, ic_groups = group_family(vc, ic, ib)
                out_elec = pd.DataFrame({"Vc": vc_axis})
                for ib_val, series in ic_groups.items():
                    out_elec[f"Ic at Ib={format_ib_label(ib_val)}"] = series

                if meas == "family Ic-Vc":
                    out_elec.to_excel(buf, index=False)
                else:
                    if "pd" not in parsed:
                        st.error(f"{uploaded.name}: "
                                + i18n.tr("PD SMU required for L-Ic-Vc",
                                          "L-Ic-Vc 模式需要 PD SMU"))
                        continue
                    _, ipd = parsed["pd"]
                    L = -ipd / responsivity
                    vc_axis, L_groups = group_family(vc, L, ib)
                    out_opt = pd.DataFrame({"Vc": vc_axis})
                    for ib_val, series in L_groups.items():
                        out_opt[f"L at Ib={format_ib_label(ib_val)}"] = series
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        out_elec.to_excel(writer, sheet_name="Electrical", index=False)
                        out_opt.to_excel(writer, sheet_name=f"Optical_responsivity_{responsivity}", index=False)

            elif meas == "gummel":
                vb, ib = parsed["base"]
                _, ic = parsed["collector"]
                out = pd.DataFrame({"Vb": vb, "Ib": ib, "Ic": ic, "Beta": ic / ib.replace({0: np.nan})})
                out.to_excel(buf, index=False)

            elif meas in ["BE diode", "BC diode"]:
                v, i = parsed["base"] if meas == "BE diode" else parsed["collector"]
                out = pd.DataFrame({"V": v, "I": abs(i)})
                out.to_excel(buf, index=False)

            per_file_outputs.append((fname, buf.getvalue()))

        st.subheader(i18n.tr("Per-file downloads", "個別檔案下載"))
        for fname, data in per_file_outputs:
            st.download_button(f"📥 {fname}.xlsx", data=data, file_name=f"{fname}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key=f"dl_hp_{fname}")

        if per_file_outputs:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, data in per_file_outputs: zf.writestr(f"{fname}.xlsx", data)
            st.download_button(
                i18n.tr("📦 Download ALL (ZIP)", "📦 下載全部（ZIP）"),
                data=zip_buf.getvalue(), file_name="SMU_Table_ALL.zip",
                mime="application/zip")