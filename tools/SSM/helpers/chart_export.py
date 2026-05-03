"""
helpers/chart_export.py — Excel export, Streamlit chart wrapper, metric tile.

Consolidates:
  - ssm_chart_utils.py        (fig_to_excel_bytes, plotly_with_dl, _EXCEL_MIME,
                               and the internal _axis_text/_is_smith/etc.)
  - IOED_HBT_RF_extract.py    (build_excel, _card → renamed to metric_card)
"""
from __future__ import annotations
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ════════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════════

def _axis_text(axis_obj) -> str:
    try:
        return axis_obj.title.text or ""
    except Exception:
        return ""


def _is_smith(fig) -> bool:
    """True when the figure is a Smith chart (Re(Γ) / Im(Γ) axes)."""
    return "Re(" in _axis_text(fig.layout.xaxis)


def _smith_col_name(trace_name: str) -> str:
    """
    Convert a Smith chart trace name to a compact column-prefix.

    Examples
    --------
    "S11 Meas."          → "S11_meas"
    "S12 Model"          → "S12_mod"
    "S22 ×2 Meas."       → "S22_meas"
    "anything else"      → "anything_else"  (fallback)
    """
    m = re.match(r"(S\d{2})", trace_name)
    if m:
        sparam = m.group(1)
        low = trace_name.lower()
        if "meas" in low:
            return f"{sparam}_meas"
        if "mod" in low:
            return f"{sparam}_mod"
        return sparam
    return re.sub(r"\s+", "_", trace_name.strip(" .")).lower()


def _freq_sheet_name(x_lbl: str, x_arr: np.ndarray) -> str:
    """
    Build an Excel sheet name from the frequency range of x_arr.

    The axis title is used to infer the unit (GHz assumed if "GHz" appears;
    MHz if "MHz"; raw Hz otherwise).  The result is formatted as "10m_to_5g",
    "100m_to_67g", etc.  Falls back to "Data" if x_arr is empty or non-numeric.

    Rules
    -----
    - Values that land below 1 GHz are expressed in MHz with an "m" suffix.
    - Values ≥ 1 GHz are expressed in GHz with a "g" suffix.
    - Trailing zeros after the decimal are stripped ("5.0g" → "5g").
    - Maximum 31 chars to fit Excel's sheet-name limit.
    """
    try:
        fin = x_arr[np.isfinite(x_arr)]
        if len(fin) == 0:
            return "Data"
        lo, hi = float(fin.min()), float(fin.max())
    except Exception:
        return "Data"

    lbl_low = x_lbl.lower()
    if "mhz" in lbl_low:
        lo_ghz, hi_ghz = lo / 1e3, hi / 1e3
    elif "ghz" in lbl_low:
        lo_ghz, hi_ghz = lo, hi
    elif "freq" in lbl_low or "hz" in lbl_low:
        lo_ghz, hi_ghz = lo / 1e9, hi / 1e9
    else:
        return "Data"

    def _lbl(v: float) -> str:
        if v < 1.0:
            mhz = v * 1e3
            s = f"{mhz:.4g}"
        else:
            s = f"{v:.4g}"
            s = s + "g"
            return s
        return s + "m"

    sheet = f"{_lbl(lo_ghz)}_to_{_lbl(hi_ghz)}"
    return sheet[:31]


def _collect_traces(fig):
    """
    Return list of (name, x_array, y_array) for all exportable traces.

    Filters out:
    - Traces with showlegend=False  (Smith chart grid lines, hlines, etc.)
    - Traces with fewer than 3 points  (constant reference / zero lines)
    """
    out = []
    for trace in fig.data:
        if getattr(trace, "showlegend", True) is False:
            continue
        x = getattr(trace, "x", None)
        y = getattr(trace, "y", None)
        if x is None or y is None:
            continue
        try:
            xa = np.asarray(x, dtype=float)
            ya = np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            continue
        if len(xa) < 3 or len(ya) < 3:
            continue
        name = (getattr(trace, "name", None) or f"Trace{len(out)}")
        name = re.sub(r"<[^>]+>", "", name)
        out.append((name, xa, ya))
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Excel export — single Plotly figure
# ════════════════════════════════════════════════════════════════════════════════

def fig_to_excel_bytes(fig) -> bytes | None:
    """
    Extract trace data from a Plotly figure and return Excel (.xlsx) bytes.

    Layout
    ------
    Smith charts
        All traces in one "Smith" sheet.  Each trace occupies two columns:
        ``S11_meas (re)`` and ``S11_meas (im)`` (Re(Γ) and Im(Γ)).

    Other plots — wide format (same shared x-axis)
        One "Data" sheet: one x column followed by one column per trace.

    Other plots — mixed x-axes
        One sheet per trace (name capped at 31 chars for Excel).

    Grid lines (showlegend=False) and constant reference lines (≤2 points)
    are always excluded.

    Returns None when no exportable data is found.
    """
    traces = _collect_traces(fig)
    if not traces:
        return None

    x_lbl = _axis_text(fig.layout.xaxis)
    y_lbl = _axis_text(fig.layout.yaxis)

    buf = io.BytesIO()

    if _is_smith(fig):
        max_len = max(len(x) for _, x, _ in traces)

        def _pad(arr: np.ndarray) -> np.ndarray:
            if len(arr) < max_len:
                return np.concatenate([arr, np.full(max_len - len(arr), np.nan)])
            return arr

        data: dict[str, np.ndarray] = {}
        for raw_name, x, y in traces:
            prefix = _smith_col_name(raw_name)
            data[f"{prefix} (re)"] = _pad(x)
            data[f"{prefix} (im)"] = _pad(y)

        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pd.DataFrame(data).to_excel(writer, sheet_name="Smith", index=False)
        return buf.getvalue()

    lengths = [len(x) for _, x, _ in traces]
    all_same_len = len(set(lengths)) == 1

    use_wide = False
    if all_same_len and len(traces) > 0:
        x_ref = traces[0][1]
        use_wide = all(np.allclose(x_ref, x, equal_nan=True) for _, x, _ in traces)

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if use_wide:
            x_ref = traces[0][1]
            sheet_name = _freq_sheet_name(x_lbl, x_ref)
            data = {x_lbl or "x": x_ref}
            for name, _, y in traces:
                col_hdr = name if not y_lbl else f"{name} ({y_lbl})"
                data[col_hdr] = y
            pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            seen: dict[str, int] = {}
            for name, x, y in traces:
                base = name[:28]
                count = seen.get(base, 0)
                seen[base] = count + 1
                sheet = base if count == 0 else f"{base}_{count}"
                pd.DataFrame({
                    x_lbl or "x": x,
                    y_lbl or "y": y,
                }).to_excel(writer, sheet_name=sheet, index=False)

    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════════
# Drop-in plotly_chart wrapper
# ════════════════════════════════════════════════════════════════════════════════

def plotly_with_dl(
    fig,
    key: str,
    filename: str = "",
    width: str = "stretch",
    container=None,
    **kwargs,
):
    """
    Render a Plotly chart and place a compact Excel download button below it.

    Parameters
    ----------
    fig       : plotly.graph_objects.Figure
    key       : Streamlit widget key (must be unique on the page).
    filename  : Base filename for the .xlsx download (no extension).
                Defaults to ``key`` when empty.
    width     : Passed through to st.plotly_chart (default "stretch").
    container : Optional Streamlit column / container object.
                When None the function renders into the current st context.
    **kwargs  : Extra keyword arguments forwarded to st.plotly_chart.
    """
    ctx = container if container is not None else st
    ctx.plotly_chart(fig, width=width, key=key, **kwargs)

    xl = fig_to_excel_bytes(fig)
    if xl is None:
        return

    ctx.download_button(
        label="⬇ xlsx",
        data=xl,
        file_name=f"{filename or key}.xlsx",
        mime=EXCEL_MIME,
        key=f"dl_xl_{key}",
    )


# ════════════════════════════════════════════════════════════════════════════════
# Multi-DUT workbook export (was IOED's `build_excel`)
# ════════════════════════════════════════════════════════════════════════════════

def build_excel(summary_df: pd.DataFrame, all_data: dict) -> bytes:
    """Build a multi-sheet workbook: a 'Summary' sheet plus one sheet per DUT.

    `all_data` is the IOED dict-of-dicts keyed by filename. Each entry must
    contain at least `df_raw` and optionally `df_fin` (de-embedded). The
    de-embedded DataFrame is used when present, otherwise the raw one.

    Sheet names are derived from the file stem with characters illegal in
    Excel sheet names replaced by underscores, capped at 28 characters.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Summary", index=False)
        for k, v in all_data.items():
            df_p = v["df_fin"] if v["df_fin"] is not None else v["df_raw"]
            base = re.sub(r"[:\\/*?\[\]]", "_", Path(k).stem)[:28]
            df_p.to_excel(w, sheet_name=base, index=False)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════════
# Streamlit metric tile (was IOED's `_card`)
# ════════════════════════════════════════════════════════════════════════════════

def metric_card(col, title: str, val, sub: str, color: str = "#4A90D9"):
    """Render a styled HTML metric tile inside a Streamlit column.

    Used by the IOED Individual tab and by the batch de-embedding tab to
    show fT/fmax cards in a uniform style.
    """
    col.markdown(
        f'<div style="padding:10px 14px;border-radius:8px;border-left:4px solid {color};'
        f'background:#f7f9fc;min-height:70px;margin-bottom:10px;">'
        f'<div style="font-size:0.74rem;color:#666;">{title}</div>'
        f'<div style="font-size:1.15rem;font-weight:700;color:#1a2e4a;">{val}</div>'
        f'<div style="font-size:0.70rem;color:#888;margin-top:1px;">{sub}</div></div>',
        unsafe_allow_html=True)
