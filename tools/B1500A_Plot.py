"""
B1500A_Plot.py — Keysight B1500A output/transfer curve viewer + TLM analysis.

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.

Changelog (short):
- v1.1: Parameter extraction for Diode/Gummel/Family — ideality (η) window with
  dashed markers, β(max) at Ic, family hover gain + per-curve offset voltage,
  saturation/output resistance, Early & knee voltage, diode Is / forward /
  leakage / rectification readouts; controls grouped into containers.
- v1.0: Initial version tracking.
"""
__version__ = "1.1"

import re

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools import i18n


# =================================================
# Utilities
# =================================================
def find_col_like(df, keywords):
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in keywords):
            return c
    return None


def find_cols_starting(df, prefix):
    return [c for c in df.columns if c.lower().startswith(prefix.lower())]


def ideality_factor(v, i, vmin, vmax):
    mask = (v >= vmin) & (v <= vmax) & (i > 0)
    if mask.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(v[mask], np.log(i[mask]), 1)
    q = 1.602e-19
    k = 1.381e-23
    T = 300
    return q / (slope * k * T)


def parse_ib_current(label):
    """Extract the Ib drive current (in amps) from a column label such as
    'Ic at Ib=200uA'. Returns None if no current can be parsed."""
    m = re.search(r"ib\s*=?\s*([\d.]+)\s*([pnumµμ]?)\s*a", label, re.IGNORECASE)
    if not m:
        return None
    scale = {"": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "μ": 1e-6,
             "n": 1e-9, "p": 1e-12}
    return float(m.group(1)) * scale.get(m.group(2).lower(), 1.0)


def fmt_current(a):
    """Human-readable current with engineering units."""
    if a is None or not np.isfinite(a):
        return "—"
    a = float(a)
    for unit, scale in [("A", 1), ("mA", 1e-3), ("µA", 1e-6),
                        ("nA", 1e-9), ("pA", 1e-12)]:
        if abs(a) >= scale:
            return f"{a / scale:.3g} {unit}"
    return f"{a:.3g} A"


def zero_crossing_x(x, y):
    """Linearly interpolate the x where y rises through zero (negative →
    positive). Returns NaN if no such crossing exists."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    for k in range(len(y) - 1):
        if y[k] <= 0 <= y[k + 1] and y[k + 1] != y[k]:
            return x[k] + (0 - y[k]) * (x[k + 1] - x[k]) / (y[k + 1] - y[k])
    return np.nan


def linear_fit_resistance(vce, ic, vmin, vmax):
    """Fit Ic = m·Vce + b over [vmin, vmax]. Returns (R_out, V_early, slope)
    where R_out = dVce/dIc (output resistance) and V_early is the magnitude of
    the extrapolated x-intercept (Early voltage)."""
    vce = np.asarray(vce, float)
    ic = np.asarray(ic, float)
    mask = (vce >= vmin) & (vce <= vmax)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    m, b = np.polyfit(vce[mask], ic[mask], 1)
    if m == 0:
        return np.inf, np.nan, 0.0
    return 1.0 / m, abs(-b / m), m


def knee_voltage(vce, ic, frac=0.9):
    """Vce at which Ic first reaches `frac` of its saturation value (mean Ic
    over the top 20 % of the Vce sweep)."""
    vce = np.asarray(vce, float)
    ic = np.asarray(ic, float)
    order = np.argsort(vce)
    vce, ic = vce[order], ic[order]
    sat = ic[vce >= 0.8 * vce.max()].mean()
    if not np.isfinite(sat) or sat <= 0:
        return np.nan
    hit = np.where(ic >= frac * sat)[0]
    return vce[hit[0]] if len(hit) else np.nan


def diode_knee_voltage(v, i):
    """Turn-on (knee) voltage from linear extrapolation of the steep forward
    region back to I = 0 on a linear I–V scale."""
    v = np.asarray(v, float)
    i = np.asarray(i, float)
    fwd = (v > 0) & (i > 0)
    if fwd.sum() < 2:
        return np.nan
    imax = i[fwd].max()
    mask = fwd & (i >= 0.2 * imax)
    if mask.sum() < 2:
        return np.nan
    m, b = np.polyfit(v[mask], i[mask], 1)
    return -b / m if m != 0 else np.nan


def saturation_current(v, i, vmin, vmax):
    """Reverse-saturation current Is from the y-intercept of the semilog fit."""
    v = np.asarray(v, float)
    i = np.asarray(i, float)
    mask = (v >= vmin) & (v <= vmax) & (i > 0)
    if mask.sum() < 2:
        return np.nan
    _, intercept = np.polyfit(v[mask], np.log(i[mask]), 1)
    return np.exp(intercept)


def window_resistance(v, i, vmin, vmax):
    """Chord resistance R = ΔV/ΔI between the lowest- and highest-V samples
    inside the η window."""
    v = np.asarray(v, float)
    i = np.asarray(i, float)
    mask = (v >= vmin) & (v <= vmax) & (i > 0)
    if mask.sum() < 2:
        return np.nan
    v_lo, v_hi = v[mask].min(), v[mask].max()
    i_lo = i[mask][v[mask].argmin()]
    i_hi = i[mask][v[mask].argmax()]
    di = i_hi - i_lo
    return (v_hi - v_lo) / di if di != 0 else np.nan


def box_x_range(key):
    """Raw (x0, x1) sorted tuple from the box selection on the chart
    registered under `key`, or None when there is no usable selection."""
    event = st.session_state.get(key) or {}
    boxes = (event.get("selection") or {}).get("box") or []
    if not boxes:
        return None
    xs = boxes[0].get("x") or []
    if len(xs) < 2:
        return None
    return tuple(sorted((float(xs[0]), float(xs[1]))))


def range_inputs(container, label, key, default, lo, hi):
    """Numeric start/end inputs for a plot x-window. Last-edited-wins: a new
    box selection on the chart registered under `key` overwrites the inputs
    before they're instantiated; otherwise typed values persist."""
    lo, hi = float(lo), float(hi)
    lo_key, hi_key = f"{key}_lo", f"{key}_hi"
    st.session_state.setdefault(lo_key, min(max(float(default[0]), lo), hi))
    st.session_state.setdefault(hi_key, min(max(float(default[1]), lo), hi))

    box = box_x_range(key)
    if box is not None and box != st.session_state.get(f"{key}_box"):
        st.session_state[f"{key}_box"] = box
        st.session_state[lo_key] = min(max(box[0], lo), hi)
        st.session_state[hi_key] = min(max(box[1], lo), hi)

    step = max((hi - lo) / 100, 1e-6)
    c1, c2 = container.columns(2)
    if i18n.is_zh():
        lo_label, hi_label = f"{label}起始 (V)", f"{label}結束 (V)"
    else:
        lo_label, hi_label = f"{label} start (V)", f"{label} end (V)"
    v_lo = c1.number_input(
        lo_label, min_value=lo, max_value=hi,
        step=step, format="%.3f", key=lo_key
    )
    v_hi = c2.number_input(
        hi_label, min_value=lo, max_value=hi,
        step=step, format="%.3f", key=hi_key
    )

    if v_lo >= v_hi:
        container.warning(i18n.tr(
            f"Start must be below end — falling back to "
            f"{default[0]:.3f}–{default[1]:.3f} V.",
            f"起始值必須小於結束值 — 已還原為 "
            f"{default[0]:.3f}–{default[1]:.3f} V。"
        ))
        return default
    return v_lo, v_hi


# =================================================
# Sidebar
# =================================================
st.sidebar.title(i18n.tr("Navigation", "導覽"))
# Values stay English — `page` is compared against these literals below;
# only the on-screen label localizes (reusing the same titles as the
# in-page st.title calls for consistency).
_PAGE_OPTS = ["B1500A Viewer", "TLM Analysis"]
_PAGE_LABELS = {"B1500A Viewer": i18n.t("b1500a_viewer"), "TLM Analysis": i18n.t("b1500a_tlm")}
page = st.sidebar.selectbox(
    i18n.tr("Select Page", "選擇頁面"),
    _PAGE_OPTS,
    format_func=lambda p: _PAGE_LABELS[p]
)

# =================================================
# B1500A VIEWER (EXCEL ONLY)
# =================================================
if page == "B1500A Viewer":

    st.title(i18n.t("b1500a_viewer"))
    st.caption(i18n.tool_desc("b1500a"))

    with st.expander(i18n.t("how_it_works"), expanded=False):
        from tools.diagrams import pipeline_png
        st.image(pipeline_png((
            ("Upload",  "xlsx"),
            ("Select",  "sheet · type"),
            ("Extract", "η · β · V_early"),
            ("Plot",    "I–V"),
        ), accent="#1f77b4"), width="stretch")

    st.subheader(i18n.t("b1500a_step1"))
    uploaded = st.file_uploader(i18n.tr("Upload Excel file (.xlsx)", "上傳 Excel 檔 (.xlsx)"), type=["xlsx"])
    if not uploaded:
        st.stop()

    st.subheader(i18n.t("b1500a_step2"))
    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox(i18n.tr("Select sheet", "選擇工作表"), xls.sheet_names)
    df = xls.parse(sheet)

    name = f"{uploaded.name} — {sheet}".lower()

    # Auto data type
    if "gummel" in name:
        default = "Gummel"
    elif "family" in name:
        default = "Family"
    else:
        default = "Diode"

    # Values stay English — `dtype` is compared against these literals
    # throughout the blocks below; only the on-screen label localizes.
    _DTYPE_OPTS = ["Diode", "Gummel", "Family"]
    _DTYPE_ZH = {"Diode": "二極體", "Gummel": "Gummel", "Family": "族群 Ic-Vc"}
    dtype = st.selectbox(
        i18n.tr("Data type", "資料類型"),
        _DTYPE_OPTS,
        index=_DTYPE_OPTS.index(default),
        format_func=lambda d: i18n.tr(d, _DTYPE_ZH[d])
    )

    # =================================================
    # DIODE
    # =================================================
    if dtype == "Diode":
        vcol = df.columns[0]
        icol = df.columns[1]

        v = df[vcol].astype(float)
        i_signed = df[icol].astype(float)
        i = i_signed.abs()

        # reserve the plot's slot so it stays on top while its controls
        # live in the results container below
        plot_area = st.container()

        diode_key = f"diode_sel_{uploaded.name}_{sheet}"

        with st.container(border=True):
            st.subheader(i18n.tr("Extracted parameters", "萃取參數"))

            vmin, vmax = range_inputs(
                st, i18n.tr("η window", "η 窗口"), diode_key, (0.35, 0.4),
                float(v.min()), float(v.max())
            )
            st.caption(i18n.tr("Dragging a band across the plot also sets this window.",
                               "在圖上拖曳選取範圍也可以設定此窗口。"))

            cv1, cv2 = st.columns(2)
            v_fwd_default = 1.0 if float(v.max()) >= 1.0 else float(v.max())
            v_rev_default = -1.0 if float(v.min()) <= -1.0 else float(v.min())
            v_rev = cv1.number_input(
                i18n.tr("Leakage-current readout voltage (V)", "洩漏電流讀值電壓 (V)"),
                float(v.min()), float(v.max()), v_rev_default
            )
            v_fwd = cv2.number_input(
                i18n.tr("Forward-current readout voltage (V)", "順向電流讀值電壓 (V)"),
                float(v.min()), float(v.max()), v_fwd_default
            )

            # --- extractions ---
            n = ideality_factor(v, i, vmin, vmax)
            i_s = saturation_current(v, i, vmin, vmax)
            knee = diode_knee_voltage(v, i_signed)
            r_series = window_resistance(v, i, vmin, vmax)
            i_fwd = i[(v - v_fwd).abs().idxmin()]
            i_leak = i[(v - v_rev).abs().idxmin()]
            vmag = min(abs(float(v.min())), abs(float(v.max())))
            rect = np.nan
            if vmag > 0:
                f_pt = i[(v - vmag).abs().idxmin()]
                r_pt = i[(v + vmag).abs().idxmin()]
                rect = f_pt / r_pt if r_pt else np.nan

            if re.search(r"(?<![a-z])be(?![a-z])", name):
                r_label = i18n.tr("BE resistance", "BE 電阻")
            elif re.search(r"(?<![a-z])bc(?![a-z])", name):
                r_label = i18n.tr("BC resistance", "BC 電阻")
            else:
                r_label = i18n.tr("Series resistance", "串聯電阻")

            c1, c2, c3 = st.columns(3)
            c1.metric(i18n.tr("Ideality factor, η", "理想因子 η"), f"{n:.2f}")
            c1.metric(i18n.tr("Knee / turn-on voltage", "膝點 / 開啟電壓"),
                      f"{knee:.3f} V" if np.isfinite(knee) else "—")
            c1.metric(r_label,
                      f"{r_series:.3g} Ω" if np.isfinite(r_series) else "—")
            c2.metric(i18n.tr("Saturation current, Is", "飽和電流 Is"), fmt_current(i_s))
            c2.metric(i18n.tr(f"Forward current @ {v_fwd:g} V", f"順向電流 @ {v_fwd:g} V"), fmt_current(i_fwd))
            c3.metric(i18n.tr(f"Leakage current @ {v_rev:g} V", f"洩漏電流 @ {v_rev:g} V"), fmt_current(i_leak))
            c3.metric(i18n.tr("Rectification ratio", "整流比"),
                      f"{rect:.2e}" if np.isfinite(rect) else "—")
            st.caption(i18n.tr(
                "η ≈ 1 → diffusion-dominated · η → 2 → recombination-dominated. "
                "Knee from linear extrapolation of the steep forward region; "
                "Is from the semilog-fit intercept over the η window. "
                "R is ΔV/ΔI between the endpoints of the η window.",
                "η ≈ 1 → 擴散主導 · η → 2 → 復合主導。"
                "膝點由順向陡峭區線性外插求得；"
                "Is 取自 η 窗口內半對數擬合的截距。"
                "R 為 η 窗口兩端點間的 ΔV/ΔI。"
            ))

        # --- plot (rendered into the reserved area above) ---
        fig = go.Figure(go.Scatter(x=v, y=i, mode="lines+markers", name="|I|"))
        fig.add_vline(x=vmin, line_dash="dash", line_color="gray",
                      annotation_text=i18n.tr("η range", "η 範圍"), annotation_position="top")
        fig.add_vline(x=vmax, line_dash="dash", line_color="gray")
        fig.update_yaxes(type="log", title=i18n.tr("Current (A)", "電流 (A)"))
        fig.update_xaxes(title=i18n.tr("Voltage (V)", "電壓 (V)"))
        fig.update_layout(title=i18n.tr("Diode I–V", "二極體 I–V"), dragmode="select",
                          selectdirection="h")
        plot_area.plotly_chart(
            fig, width="stretch", key=diode_key,
            on_select="rerun", selection_mode="box"
        )

    # =================================================
    # GUMMEL
    # =================================================
    elif dtype == "Gummel":
        vb_col = find_col_like(df, ["vb"])
        ic_col = find_col_like(df, ["ic"])
        ib_col = find_col_like(df, ["ib"])

        if not all([vb_col, ic_col, ib_col]):
            st.error(i18n.tr("Could not identify Vb / Ic / Ib columns",
                             "無法辨識 Vb / Ic / Ib 欄位"))
            st.stop()

        vb = df[vb_col].astype(float)
        ic = df[ic_col].astype(float)
        ib = df[ib_col].astype(float)

        beta = ic / ib.replace(0, np.nan)
        beta_max = beta.max()
        beta_v = vb.loc[beta.idxmax()]
        ic_at_betamax = abs(ic).loc[beta.idxmax()]

        # reserve the plot's slot; its η window selection lives in the
        # ideality container below
        plot_area = st.container()

        gummel_key = f"gummel_sel_{uploaded.name}_{sheet}"

        with st.container(border=True):
            st.subheader(i18n.tr("Ideality factors (η)", "理想因子 (η)"))
            vmin, vmax = range_inputs(
                st, i18n.tr("η window", "η 窗口"), gummel_key, (0.35, 0.4), 0.0, float(vb.max())
            )
            n_ic = ideality_factor(vb, abs(ic), vmin, vmax)
            n_ib = ideality_factor(vb, abs(ib), vmin, vmax)
            c1, c2 = st.columns(2)
            c1.metric(i18n.tr("Collector ideality, η(Ic)", "集極理想因子 η(Ic)"), f"{n_ic:.2f}")
            c2.metric(i18n.tr("Base ideality, η(Ib)", "基極理想因子 η(Ib)"), f"{n_ib:.2f}")
            st.caption(i18n.tr("Dragging a band across the plot also sets this window.",
                               "在圖上拖曳選取範圍也可以設定此窗口。"))

        with st.container(border=True):
            st.subheader(i18n.tr("Current gain (β)", "電流增益 (β)"))
            c1, c2, c3 = st.columns(3)
            c1.metric(i18n.tr("Maximum gain, β(max)", "最大增益 β(max)"), f"{beta_max:.1f}")
            c2.metric(i18n.tr("β(max) occurs at Vb", "β(max) 發生於 Vb"), f"{beta_v:.3f} V")
            c3.metric(i18n.tr("β(max) occurs at Ic", "β(max) 發生於 Ic"), fmt_current(ic_at_betamax))

        # --- plot (rendered into the reserved area above) ---
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )

        fig.add_trace(go.Scatter(x=vb, y=abs(ic), name="Ic"),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=vb, y=abs(ib), name="Ib"),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=vb, y=beta, name="β"),
                      row=1, col=1, secondary_y=True)

        fig.add_vline(x=vmin, line_dash="dash", line_color="gray",
                      annotation_text=i18n.tr("η range", "η 範圍"), annotation_position="top")
        fig.add_vline(x=vmax, line_dash="dash", line_color="gray")

        fig.update_yaxes(type="log", secondary_y=False, title=i18n.tr("Current (A)", "電流 (A)"))
        fig.update_yaxes(
            type="linear",
            range=[0.0, 1.5 * beta_max],
            secondary_y=True,
            title="β"
        )
        fig.update_xaxes(title="Vb (V)")
        fig.update_layout(title=i18n.tr("Gummel Plot", "Gummel 圖"), dragmode="select",
                          selectdirection="h")

        plot_area.plotly_chart(
            fig, width="stretch", key=gummel_key,
            on_select="rerun", selection_mode="box"
        )

    # =================================================
    # FAMILY
    # =================================================
    else:
        vc_col = find_col_like(df, ["vc"])
        if not vc_col:
            st.error(i18n.tr("Vc column not found", "找不到 Vc 欄位"))
            st.stop()

        vc = df[vc_col].astype(float)
        ic_cols = [c for c in df.columns if c.lower().startswith("ic")]

        if not ic_cols:
            st.error(i18n.tr("No Ic curves found (expected 'Ic at Ib=...')",
                             "找不到 Ic 曲線（預期格式為 'Ic at Ib=...'）"))
            st.stop()

        # default Rsat window = upper 40 % of the Vc sweep (flat region)
        vc_lo, vc_hi = float(vc.min()), float(vc.max())
        default_lo = round(vc_lo + 0.6 * (vc_hi - vc_lo), 3)

        # parse Ib per curve; the largest is the "highest Ib" line
        ib_vals = {c: parse_ib_current(c) for c in ic_cols}
        top_col = max(
            ic_cols,
            key=lambda c: (ib_vals[c] is not None, ib_vals[c] or 0.0)
        )

        # reserve the plot's slot; the Rsat-window selection lives in the
        # output-region container below
        plot_area = st.container()

        family_key = f"family_sel_{uploaded.name}_{sheet}"

        with st.container(border=True):
            st.subheader(i18n.tr("Output-region parameters", "輸出區參數"))
            r_min, r_max = range_inputs(
                st, i18n.tr("Rsat window", "Rsat 窗口"), family_key,
                (default_lo, round(vc_hi, 3)), vc_lo, vc_hi
            )
            st.caption(i18n.tr(f"Fitted on the highest-Ib curve: **{top_col}**",
                               f"擬合曲線：Ib 最高的曲線 **{top_col}**"))
            st.caption(i18n.tr("Dragging a band across the plot also sets this window.",
                               "在圖上拖曳選取範圍也可以設定此窗口。"))
            ic_top = df[top_col].astype(float)
            r_out, v_early, _ = linear_fit_resistance(vc, ic_top, r_min, r_max)
            knee = knee_voltage(vc, ic_top)
            c1, c2, c3 = st.columns(3)
            c1.metric(i18n.tr("Saturation (output) resistance", "飽和（輸出）電阻"),
                      f"{r_out:,.0f} Ω" if np.isfinite(r_out) else "—")
            c2.metric(i18n.tr("Early voltage, |VA|", "厄利電壓 |VA|"),
                      f"{v_early:.2f} V" if np.isfinite(v_early) else "—")
            c3.metric(i18n.tr("Knee voltage", "膝點電壓"),
                      f"{knee:.3f} V" if np.isfinite(knee) else "—")

        offsets = {}
        fig = go.Figure()
        for c in ic_cols:
            ib_a = ib_vals[c]
            ic_line = df[c].astype(float)
            v_off = zero_crossing_x(vc, ic_line)
            offsets[c] = v_off
            gain = (ic_line / ib_a) if ib_a else pd.Series(
                np.nan, index=ic_line.index)
            customdata = np.column_stack([
                gain.to_numpy(),
                np.full(len(ic_line), v_off),
            ])
            ht = ("Vc=%{x:.3f} V<br>Ic=%{y:.3e} A"
                  "<br>β=%{customdata[0]:.1f}"
                  "<br>V(offset)=%{customdata[1]:.3f} V"
                  "<extra>" + c + "</extra>")
            fig.add_trace(go.Scatter(
                x=vc, y=ic_line, name=c,
                customdata=customdata, hovertemplate=ht
            ))

        all_max = max(float(df[c].max()) for c in ic_cols)
        all_min = min(float(df[c].min()) for c in ic_cols)
        fig.add_vrect(
            x0=r_min, x1=r_max, fillcolor="LightGray", opacity=0.25,
            line_width=0, annotation_text=i18n.tr("Rsat fit", "Rsat 擬合"),
            annotation_position="top left"
        )
        fig.update_xaxes(title="Vc (V)")
        fig.update_yaxes(
            title="Ic (A)",
            range=[min(0.0, all_min) * 1.2, all_max * 1.2]
        )
        fig.update_layout(title=i18n.tr("Family I–V", "族群 I–V"), hovermode="closest",
                          dragmode="select", selectdirection="h")
        plot_area.plotly_chart(
            fig, width="stretch", key=family_key,
            on_select="rerun", selection_mode="box"
        )

        valid_off = [o for o in offsets.values() if np.isfinite(o)]
        off_avg = float(np.mean(valid_off)) if valid_off else np.nan

        with st.container(border=True):
            st.subheader(i18n.tr("Offset voltage", "偏移電壓"))
            st.metric(
                i18n.tr("Average offset voltage", "平均偏移電壓"),
                f"{off_avg * 1e3:.1f} mV" if np.isfinite(off_avg) else "—"
            )
            st.caption(i18n.tr(
                "Hover any curve to read its own offset (Ic zero-crossing) and "
                "DC current gain β. The average above is over curves that "
                "actually cross zero.",
                "將滑鼠移到任一曲線上，可讀取其偏移電壓（Ic 過零點）與"
                "直流電流增益 β。上方平均值僅計入實際穿越零點的曲線。"
            ))

# =================================================
# TLM ANALYSIS
# =================================================
else:
    st.title(i18n.t("b1500a_tlm"))

    Z = st.number_input(i18n.tr("Pad width Z (µm)", "焊墊寬度 Z (µm)"), value=80.0)

    R = {
        4: st.number_input("R @ 4 µm (Ω)", value=np.nan),
        8: st.number_input("R @ 8 µm (Ω)", value=np.nan),
        16: st.number_input("R @ 16 µm (Ω)", value=np.nan),
        32: st.number_input("R @ 32 µm (Ω)", value=np.nan),
    }

    x = np.array([k for k, v in R.items() if not np.isnan(v)])
    y = np.array([v for v in R.values() if not np.isnan(v)])

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        Rc = 0.5 * intercept
        Rsh = slope * Z
        LT = Rc / slope
        rho_c = Rc * LT * Z * 1e-8
        r2 = np.corrcoef(x, y)[0, 1] ** 2

        c1, c2, c3 = st.columns(3)
        c1.metric(i18n.tr("Intercept (Ω)", "截距 (Ω)"), f"{intercept:.2f}")
        c1.metric(i18n.tr("Slope (Ω/µm)", "斜率 (Ω/µm)"), f"{slope:.3f}")
        c2.metric(i18n.tr("Contact Resistance, Rc (Ω)", "接觸電阻 Rc (Ω)"), f"{Rc:.2f}")
        c2.metric(i18n.tr("Sheet Resistance, Rsh (Ω/□)", "片電阻 Rsh (Ω/□)"), f"{Rsh:.1f}")
        c3.metric(i18n.tr("Transfer Length, LT (µm)", "傳輸長度 LT (µm)"), f"{LT:.2f}")
        c3.metric(i18n.tr("Specific Contact Resistivity, ρc (Ω·cm²)", "比接觸電阻率 ρc (Ω·cm²)"), f"{rho_c:.2e}")
        st.metric(i18n.tr("Goodness, R²", "擬合優度 R²"), f"{r2:.4f}")

        xs = np.linspace(0, 40, 200)
        ys = slope * xs + intercept

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=i18n.tr("Measured", "量測值")))
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(dash="dash"),
            name=i18n.tr("Linear fit", "線性擬合")
        ))

        fig.update_xaxes(title=i18n.tr("Spacing (µm)", "間距 (µm)"), range=[0, 40])
        fig.update_yaxes(title=i18n.tr("Resistance (Ω)", "電阻 (Ω)"), range=[0, max(y) * 1.2])
        st.plotly_chart(fig, width="stretch")