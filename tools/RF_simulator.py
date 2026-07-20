"""
RF_simulator.py — Forward RF S-parameter simulator.

Allows the user to pick an SSM model (Cheng's T / π) or "Open and Short Pad",
key in all extrinsic / intrinsic parameters from scratch, and inspect the
resulting Smith chart and (for SSM models) fT/fmax bode plot.  S2P and Excel
exports are provided for every chart.

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.
"""
from __future__ import annotations

__version__ = "1.2"

from pathlib import Path

import numpy as np
import streamlit as st

from tools import i18n
from tools.SSM import handoff
import plotly.graph_objects as go

from tools.SSM.models.cheng    import (ChengT, ChengPi,
                                        _render_topology_illustration,
                                        _EXT_T_SPECS, _INT_T_SPECS,
                                        _EXT_PI_SPECS, _INT_PI_SPECS)
from tools.SSM.models.xu       import (XuModel,
                                        _render_topology_illustration as _render_xu_illustration,
                                        _EXT_T_SPECS as _XU_EXT_SPECS,
                                        _INT_T_SPECS as _XU_INT_SPECS,
                                        _XU_PAD_SPECS)
from tools.SSM.models.kunyang  import (KunYangHEMT,
                                        _render_topology_illustration as _render_ky_illustration,
                                        _EXT_KY_SPECS, _INT_KY_SPECS,
                                        _KY_PAD_SPECS, _DEFAULT_PARAMS as _KY_DEFAULT_PARAMS)
from tools.SSM.models.base_ui  import (PAD_SPECS, render_finetune_diagram,
                                        render_smith_with_ftfmax)
from tools.SSM.ssm_plots       import (render_matplotlib_smith,
                                        render_tau_fmax_expander)
from tools.SSM.helpers         import (extended_smith_grid, parse_s2p, parse_csv,
                                        write_s2p, simulate_open, simulate_short,
                                        compute_h21_U, find_ft_fmax,
                                        extrap_20dbdec, single_pole_extrap,
                                        FT_FMAX_SYMBOLS, FT_FMAX_COLORS,
                                        plotly_with_dl, fig_to_excel_bytes,
                                        bode_excel_bytes,
                                        fig_to_tsv, copy_button,
                                        segmented_radio,
                                        make_smith_bode_slider_fig)

try:
    import cupy as _cp
    _HAS_CUDA = True
    _v = _cp.cuda.runtime.runtimeGetVersion()
    _CUDA_VER = f"{_v // 1000}.{(_v % 1000) // 10}"
except Exception:
    _cp = None  # type: ignore[assignment]
    _HAS_CUDA = False
    _CUDA_VER = ""

_EXCEL_MIME = ("application/vnd.openxmlformats-officedocument."
               "spreadsheetml.sheet")


# ─────────────────────────────────────────────────────────────────────────────

st.title(i18n.title("rf_sim"))
st.caption(i18n.tool_desc("rf_sim"))


# ─── Handoff from the other RF pages (device / extracted values) ─────────────
_MS_TO_OPTION = {"T": "Cheng's T", "pi": "Cheng's π", "XuT": "Xu T",
                 "KY": "Kun-Yang HEMT", "custom": "🧩 Custom model"}
_inc = handoff.take(handoff.TARGET_SIMFIT)
if _inc is not None:
    st.session_state["_simfit_meas"] = {
        "S": _inc["S"], "freq": _inc["freq"], "z0": _inc["z0"],
        "label": _inc.get("label", "device"), "stage": _inc.get("stage", "raw")}
    # A fresh handover is a new device — drop any seeds left from a previous one
    # so models other than the handed-over one auto-guess from THIS device.
    for _k in [k for k in st.session_state if k.startswith("_simfit_seed_")]:
        st.session_state.pop(_k, None)
    _ms = _inc.get("model_short")
    if _ms and _MS_TO_OPTION.get(_ms):
        st.session_state["rfsim_model_choice"] = _MS_TO_OPTION[_ms]
    if _inc.get("params") and _ms:
        st.session_state[f"_simfit_seed_{_ms}"] = dict(_inc["params"])
        # Flag consumed by the fit branch below: a just-arrived handoff must
        # override stale widget state AND any cached fit for this device.
        st.session_state[f"_simfit_seed_pending_{_ms}"] = True


def _resolve_fit_target():
    """Optional measured-device target for fitting.  Returns a dict
    ``{S, freq, z0, label, stage}`` or ``None``.  Sourced from a handoff
    (preferred) or an inline uploader; persisted in session_state."""
    def _drop_seeds():
        # Extraction handoffs seed the override fields per model
        # (`_simfit_seed_<short>`).  Drop them when the device changes by hand so
        # a stale seed from a previous extraction can't pre-fill an unrelated
        # uploaded device.
        for _k in [k for k in st.session_state if k.startswith("_simfit_seed_")]:
            st.session_state.pop(_k, None)

    meas = st.session_state.get("_simfit_meas")
    with st.container(key="hbt_exp_edit_fitdev"), \
         st.expander(i18n.tr("📂 Fit to a measured device (optional)",
                             "📂 擬合至量測元件（選用）"),
                     expanded=meas is not None):
        if meas is not None:
            cc, cdl, cclr = st.columns([3, 1, 1])
            _pts_word = i18n.tr("pts", "點")
            _tt = i18n.tr(
                "The simulation frequency axis follows the measured grid "
                "point-for-point.",
                "模擬頻率軸會逐點對齊量測資料的網格。")
            cc.markdown(
                f"<span class='hbt-chip-ok'>🎯 {meas['label']} · {meas['stage']} · "
                f"{len(meas['freq'])} {_pts_word}</span>"
                f"<span class='hbt-help' title='{_tt}'>?</span>",
                unsafe_allow_html=True)
            # Download the exact measured device being fitted (de-embedded if it
            # was forwarded that way) as an .s2p.
            try:
                _s2p = write_s2p(
                    np.asarray(meas["freq"], dtype=float), meas["S"],
                    title=f"Measured device — {meas['label']} ({meas['stage']})",
                    params={"stage": meas["stage"]})
                cdl.download_button(
                    "📥 .s2p", data=_s2p,
                    file_name=f"{meas['label']}_{meas['stage']}.s2p",
                    mime="text/plain", key="simfit_dl", width="stretch",
                    help=i18n.tr(
                        "Download the exact S-parameters loaded for fitting.",
                        "下載擬合所用的完整 S 參數。"))
            except Exception:                                  # noqa: BLE001
                cdl.caption("—")
            if cclr.container(key="hbt_danger_simfit_clear").button(
                    i18n.tr("✕ Clear", "✕ 清除"), key="simfit_clear",
                    width="stretch"):
                st.session_state.pop("_simfit_meas", None)
                _drop_seeds()
                st.rerun()
        up = st.file_uploader(
            i18n.tr("Upload a measured .s2p / .csv to compare & fit",
                    "上傳量測 .s2p / .csv 以進行比較與擬合"),
            type=["s2p", "csv"],
            key="simfit_up",
            help=i18n.tr(
                "With a file loaded the page switches to fit mode: residual "
                "readout + visual / auto tuning against this device.  "
                "Send a de-embedded device (no Cpxx/Lx) to fit intrinsic-only.",
                "載入檔案後頁面會切換為擬合模式：顯示殘差，並提供視覺化 / 自動"
                "調諧功能。傳入去嵌入元件（不含 Cpxx/Lx）可僅擬合本質參數。"))
        if up is not None:
            data = up.getvalue()
            sig = (up.name, len(data), hash(data))
            if st.session_state.get("simfit_up_sig") != sig:
                try:
                    if up.name.lower().endswith(".csv"):
                        fr, S, z0 = parse_csv(data.decode("utf-8", "ignore"))
                    else:
                        fr, S, z0 = parse_s2p(data)
                    st.session_state["_simfit_meas"] = {
                        "S": S, "freq": fr, "z0": z0,
                        "label": Path(up.name).stem, "stage": "raw"}
                    st.session_state["simfit_up_sig"] = sig
                    _drop_seeds()       # uploaded device has no extracted seed
                    st.rerun()
                except Exception as exc:                       # noqa: BLE001
                    st.error(i18n.tr(f"Could not read that file: {exc}",
                                     f"無法讀取該檔案：{exc}"))
    return st.session_state.get("_simfit_meas")


# ─── Model selector ──────────────────────────────────────────────────────────
# (Declared before the frequency axis so the "Custom" builder can take over the
#  page without showing the global frequency controls it doesn't use.)
MODEL_OPTIONS = ["Cheng's T", "Cheng's π", "Xu T", "Kun-Yang HEMT",
                 "🧩 Custom model", "Open and Short Pad"]
# Display-only translations — the underlying option VALUE stays the canonical
# English string (matched by _MS_TO_OPTION above and every `model_choice ==`
# branch below), so switching UI language can't desync the handoff mapping.
_MODEL_LABELS = {
    "🧩 Custom model":    i18n.tr("🧩 Custom model", "🧩 自訂模型"),
    "Open and Short Pad": i18n.tr("Open and Short Pad", "開路與短路焊墊"),
}
model_choice  = segmented_radio(i18n.tr("Model", "模型"), MODEL_OPTIONS,
                                index=0, key="rfsim_model_choice",
                                format_func=lambda o: _MODEL_LABELS.get(o, o))

measured = _resolve_fit_target()

if model_choice == "🧩 Custom model":
    from tools.SSM.custom_model import render_custom_section
    render_custom_section(measured)
    st.stop()


# ─── Frequency axis ──────────────────────────────────────────────────────────
# In fit mode the simulation must share the measured device's frequency grid so
# the residual lines up point-for-point; the manual axis is then hidden.

if measured is not None:
    freq  = np.asarray(measured["freq"], dtype=float)
    f_ghz = freq * 1e-9
else:
    c_f1, c_f2, c_f3 = st.columns(3)
    f_start = c_f1.number_input(i18n.tr("Start Frequency (GHz)", "起始頻率 (GHz)"),
                                min_value=0.0, value=0.01,
                                format="%.4f", step=0.01)
    n_pts   = c_f2.number_input(i18n.tr("Data Points", "資料點數"),
                                min_value=2, value=1001, step=1)
    f_end   = c_f3.number_input(i18n.tr("Final Frequency (GHz)", "終止頻率 (GHz)"),
                                min_value=0.001, value=50.0,
                                format="%.4f", step=1.0)

    if f_end <= f_start:
        st.error(i18n.tr("Final frequency must be greater than start frequency.",
                         "終止頻率必須大於起始頻率。"))
        st.stop()

    freq  = np.linspace(float(f_start) * 1e9, float(f_end) * 1e9, int(n_pts))
    f_ghz = freq * 1e-9


# ─── Helpers to render and collect spec lists ────────────────────────────────

def _render_spec_inputs(specs, prefix: str, label: str, cols_per_row: int = 4):
    """Render number_input widgets in rows of ``cols_per_row``.

    Each spec is (key, lbl, sc, unit, fmt, step). Defaults to 0.
    """
    st.markdown(f"**{label}**")
    for row_start in range(0, len(specs), cols_per_row):
        row = specs[row_start:row_start + cols_per_row]
        cs  = st.columns(len(row))
        for col_w, (key, lbl, sc, unit, fmt, step) in zip(cs, row):
            sk = f"rfsim_{prefix}_{key}"
            if sk not in st.session_state:
                st.session_state[sk] = 0.0
            col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                               key=sk, format=fmt, step=step)


def _collect_specs(specs, prefix: str) -> dict:
    """Collect inputs from session_state and convert back to SI units."""
    out = {}
    for key, lbl, sc, unit, fmt, step in specs:
        sk = f"rfsim_{prefix}_{key}"
        out[key] = float(st.session_state.get(sk, 0.0)) / sc
    return out


# ─── Plotly Smith / Bode helpers ─────────────────────────────────────────────

_SMITH_COLORS = {"S11": "#1f77b4", "S22": "#ff7f0e",
                 "S21": "#2ca02c", "S12": "#d62728"}


def _build_smith(S, freq_hz, mults: dict, title: str):
    """``mults`` is a dict ``{"S11":..., "S12":..., "S21":..., "S22":...}``."""
    fig = go.Figure()
    for tr in extended_smith_grid(1.0):
        fig.add_trace(tr)
    f_ghz_local = freq_hz * 1e-9
    for name, (r, c) in [("S11", (0, 0)), ("S22", (1, 1)),
                         ("S21", (1, 0)), ("S12", (0, 1))]:
        col  = _SMITH_COLORS[name]
        m    = float(mults.get(name, 1.0))
        sv   = S[:, r, c] * m
        sc_lbl = "" if abs(m - 1.0) < 1e-9 else (
            f"  ×{m:.3g}" if m >= 1 else f"  ÷{1.0 / m:.3g}")
        hov  = [f"f={fv:.3f} GHz<br>Re={rv:.4f}<br>Im={iv:.4f}"
                for fv, rv, iv in zip(f_ghz_local, sv.real, sv.imag)]
        fig.add_trace(go.Scatter(x=sv.real, y=sv.imag, mode="lines",
                                 name=f"{name}{sc_lbl}",
                                 line=dict(color=col, width=2.0),
                                 text=hov, hoverinfo="text"))
    fig.update_layout(
        title=dict(text=f"{i18n.tr('Smith Chart', 'Smith 圖')} — {title}",
                  font=dict(size=12)),
        xaxis=dict(title="Re(Γ)", range=[-1.1, 1.1], scaleanchor="y",
                   scaleratio=1, showgrid=False, zeroline=False),
        yaxis=dict(title="Im(Γ)", range=[-1.1, 1.1],
                   showgrid=False, zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white", height=560,
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(x=1.02, y=1.0, xanchor="left"),
        hovermode="closest",
    )
    return fig


def _smith_chart_with_dl(fig, key: str, filename: str,
                         s2p_data: bytes, s2p_filename: str):
    """Render a smith chart and put xlsx + s2p download buttons side by side."""
    st.plotly_chart(fig, width="stretch", key=key)
    xl = fig_to_excel_bytes(fig)
    tsv = fig_to_tsv(fig)
    col_xl, col_copy, col_s2p = st.columns(3)
    if xl is not None:
        col_xl.download_button(
            i18n.tr("⬇ xlsx", "⬇ xlsx 檔"),
            data=xl,
            file_name=f"{filename}.xlsx",
            mime=_EXCEL_MIME,
            key=f"dl_xl_{key}",
            width="stretch",
        )
    if tsv:
        copy_button(tsv, key=key, container=col_copy)
    col_s2p.download_button(
        "📥 .s2p",
        data=s2p_data,
        file_name=s2p_filename,
        mime="text/plain",
        key=f"dl_s2p_{key}",
        width="stretch",
    )


def _smith_multiplier_inputs(prefix: str, label: str | None = None) -> dict:
    """Render 4 per-trace multipliers (S11/S12/S21/S22) and return a dict.

    The multipliers are stored under a **model-independent** key so the user's
    choice persists when switching models (previously the per-``prefix`` key
    reset the value on every model change).
    """
    if label is None:
        label = i18n.tr("Smith multipliers", "Smith 倍率")
    st.markdown(f"**{label}** — "
               + i18n.tr("× when ≥ 1, ÷ when < 1, per trace",
                         "× 表示 ≥ 1，÷ 表示 < 1，逐軌跡設定"))
    cols = st.columns(4)
    out = {}
    for col_w, sp in zip(cols, ("S11", "S12", "S21", "S22")):
        sk = f"rfsim_smithmult_{sp}"
        if sk not in st.session_state:
            st.session_state[sk] = 1.0
        out[sp] = col_w.number_input(f"{sp} ×",
                                     min_value=0.001,
                                     step=0.1, format="%.3f",
                                     key=sk)
    return out


def _render_slider_preview(model_cls, all_p, freq, mults, prefix: str,
                           pad_specs, ext_specs, int_specs):
    """RF simulator slider preview block — mode-toggled.

    🎯 All sweep: drag any number of sliders, every tick reruns Streamlit + sim.
    ⚡ Smooth sweep: pre-compute the cartesian sweep once, scrub client-side.
    """
    mode_key = f"rfsim_slpreview_mode_{prefix}"
    mode = segmented_radio(
        i18n.tr("Preview mode", "預覽模式"),
        [i18n.tr("🎯 All sweep", "🎯 全參數掃描"),
         i18n.tr("⚡ Smooth sweep", "⚡ 平滑掃描")],
        index=0,
        key=mode_key,
        help=i18n.tr(
            "🎯 All sweep — best for a few small changes: the plots "
            "re-compute on every drag.  ⚡ Smooth sweep — best for exploring a "
            "large range: pre-computes the whole range once so dragging is "
            "instant afterwards.",
            "🎯 全參數掃描 — 適合少量微調：每次拖曳都會重新計算圖表。"
            "⚡ 平滑掃描 — 適合探索大範圍：一次預先計算整個範圍，之後拖曳即時反應。"))
    if mode.startswith("⚡"):
        _render_rfsim_plotly_slider_preview(model_cls, all_p, freq, prefix,
                                             pad_specs, ext_specs, int_specs)
    else:
        _render_rfsim_live_slider_preview(model_cls, all_p, freq, mults, prefix,
                                           pad_specs, ext_specs, int_specs)


_FRAGMENT = (getattr(st, "fragment", None)
             or getattr(st, "experimental_fragment", None)
             or (lambda f: f))


def _slider_default_range(current_disp):
    if abs(current_disp) < 1e-30:
        return -1.0, 1.0, 0.01
    lo = current_disp * 0.1 if current_disp > 0 else current_disp * 10
    hi = current_disp * 10  if current_disp > 0 else current_disp * 0.1
    d_min, d_max = min(lo, hi), max(lo, hi)
    d_step = max((d_max - d_min) / 100, 1e-9)
    return d_min, d_max, d_step


@_FRAGMENT
def _render_rfsim_live_slider_preview(model_cls, all_p, freq, mults, prefix: str,
                                       pad_specs, ext_specs, int_specs):
    """Streamlit-rerun-per-drag implementation.

    Layout
    ------
    [ multiselect of params                                                  ]
    [ one column per selected param — slider + delta caption                 ]
    [ ▼ Slider ranges (min / step / max) expander, each row has 3 columns   ]
    [ Smith chart  |  fT/fmax bode  (side by side)                          ]
    [ ✅ Use these values | ↩️ Reset preview                                 ]
    """
    cat_of: dict[str, str] = {}
    for s in pad_specs: cat_of[s[0]] = "pad"
    for s in ext_specs: cat_of[s[0]] = "ext"
    for s in int_specs: cat_of[s[0]] = "int"
    tuning_specs = list(pad_specs) + list(ext_specs) + list(int_specs)
    label_for = {s[0]: s[1] for s in tuning_specs}

    sel_key  = f"rfsim_slpreview_sel_{prefix}"
    selected = st.multiselect(
        i18n.tr("Parameters to slide", "要拖曳的參數"),
        options=[s[0] for s in tuning_specs],
        default=st.session_state.get(sel_key, []),
        format_func=lambda k: label_for.get(k, k),
        key=sel_key,
        placeholder=i18n.tr("Choose options", "請選擇項目"),
        help=i18n.tr(
            "Pick parameter(s) to drag.  Plots below show the slider-"
            "substituted model in real time; the main Smith / fT-fmax "
            "plots above stay frozen until you click ✅ Use these values.",
            "選擇要拖曳的參數。下方圖表會即時顯示套用滑桿值後的模型；"
            "上方主要的 Smith / fT-fmax 圖會保持不變，直到按下"
            "「✅ 套用這些數值」為止。"))

    selected_specs = [s for s in tuning_specs if s[0] in selected]
    preview_overrides: dict[str, float] = {}

    for spec in selected_specs:
        key, _, scale = spec[0], spec[1], spec[2]
        current_disp  = float(all_p.get(key, 0.0)) * scale
        kp = f"rfsim_slpreview_{prefix}_{key}"
        if f"{kp}_min" not in st.session_state:
            d_min, d_max, d_step = _slider_default_range(current_disp)
            st.session_state[f"{kp}_min"]  = float(d_min)
            st.session_state[f"{kp}_max"]  = float(d_max)
            st.session_state[f"{kp}_step"] = float(d_step)
        if kp not in st.session_state:
            st.session_state[kp] = float(current_disp)

    if selected_specs:
        with st.expander(i18n.tr("📏 Slider ranges (min / step / max)",
                                 "📏 滑桿範圍（最小 / 步進 / 最大）"),
                         expanded=False):
            _min_lbl  = i18n.tr("Min", "最小值")
            _step_lbl = i18n.tr("Step", "步進")
            _max_lbl  = i18n.tr("Max", "最大值")
            for row_start in range(0, len(selected_specs), 2):
                row_specs = selected_specs[row_start:row_start + 2]
                row_cols  = st.columns(len(row_specs))
                for col_w, spec in zip(row_cols, row_specs):
                    key, label, scale = spec[0], spec[1], spec[2]
                    unit = spec[3] if len(spec) > 3 else ""
                    fmt  = spec[4] if len(spec) > 4 else "%.4g"
                    kp   = f"rfsim_slpreview_{prefix}_{key}"
                    with col_w:
                        st.markdown(f"**{label}** ({unit})" if unit
                                    else f"**{label}**")
                        mc = st.columns(3)
                        mc[0].number_input(f"{_min_lbl} ({unit})" if unit
                                           else _min_lbl,
                                           format=fmt, key=f"{kp}_min")
                        mc[1].number_input(_step_lbl, format=fmt,
                                           key=f"{kp}_step", min_value=0.0)
                        mc[2].number_input(f"{_max_lbl} ({unit})" if unit
                                           else _max_lbl,
                                           format=fmt, key=f"{kp}_max")

    if not selected_specs:
        st.caption(i18n.tr("Select one or more parameters above to begin.",
                           "請先在上方選擇一或多個參數。"))
    else:
        slider_cols = st.columns(len(selected_specs))
        for col, spec in zip(slider_cols, selected_specs):
            key, label, scale = spec[0], spec[1], spec[2]
            unit = spec[3] if len(spec) > 3 else ""
            fmt  = spec[4] if len(spec) > 4 else "%.4g"
            current_disp = float(all_p.get(key, 0.0)) * scale
            kp = f"rfsim_slpreview_{prefix}_{key}"
            mn = float(st.session_state[f"{kp}_min"])
            mx = float(st.session_state[f"{kp}_max"])
            sp = float(st.session_state[f"{kp}_step"])
            if mx <= mn:
                mx = mn + max(sp, abs(mn) * 1e-6 + 1e-9)
            sp_safe = sp if sp > 0 else max((mx - mn) / 100, 1e-12)
            cur_v = min(max(float(st.session_state.get(kp, current_disp)),
                            mn), mx)
            with col:
                v = st.slider(f"{label} ({unit})" if unit else label,
                              min_value=mn, max_value=mx, step=sp_safe,
                              value=cur_v, format=fmt, key=kp)
                st.caption(
                    f"{i18n.tr('main', '主圖')}: **{current_disp:.4g}**  →  "
                    f"{i18n.tr('preview', '預覽')}: **{v:.4g}** {unit}".rstrip())
            preview_overrides[key] = float(v) / scale

    # ── Preview plots: Smith | Bode side by side ──────────────────────
    all_p_prev = dict(all_p)
    all_p_prev.update(preview_overrides)
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            S_prev = model_cls.simulate(all_p_prev, freq)
    except Exception as e:
        st.error(i18n.tr(f"Preview simulation failed: {e}",
                         f"預覽模擬失敗：{e}"))
        S_prev = None
    if S_prev is not None and not np.all(np.isfinite(S_prev)):
        st.warning(i18n.tr(
            "Preview S-parameters contain non-finite values — "
            "adjust slider ranges.",
            "預覽 S 參數含有非有限值 — 請調整滑桿範圍。"))
        S_prev = None
    if S_prev is not None:
        col_s, col_b = st.columns([1.05, 1])
        with col_s:
            st.markdown(f"**{i18n.tr('Preview Smith chart', '預覽 Smith 圖')}**")
            st.plotly_chart(_build_smith(S_prev, freq, mults, model_cls.NAME),
                            width="stretch",
                            key=f"rfsim_slpreview_smith_{prefix}")
        with col_b:
            st.markdown(f"**{i18n.tr('Preview fT / fmax', '預覽 fT / fmax')}**")
            st.plotly_chart(_build_bode(S_prev, freq, model_cls.NAME)[0],
                            width="stretch",
                            key=f"rfsim_slpreview_bode_{prefix}")

    bc1, bc2 = st.columns(2)
    commit_clicked = bc1.container(key=f"hbt_amber_slcommit_rfsim_{prefix}").button(
        i18n.tr("✅ Use these values", "✅ 套用這些數值"),
        key=f"rfsim_slpreview_commit_{prefix}",
        disabled=(len(preview_overrides) == 0),
        help=i18n.tr("Copy slider values into the fine-tune number_inputs above.",
                     "將滑桿數值複製到上方的微調數字輸入欄。"),
        width="stretch")
    reset_clicked = bc2.button(
        i18n.tr("↩️ Reset preview", "↩️ 重設預覽"),
        key=f"rfsim_slpreview_reset_{prefix}",
        help=i18n.tr("Discard slider drags and clear remembered min/step/max.",
                     "捨棄滑桿拖曳並清除已記住的最小值/步進/最大值。"),
        width="stretch")

    if commit_clicked:
        for k, v_si in preview_overrides.items():
            sc  = next(s[2] for s in tuning_specs if s[0] == k)
            cat = cat_of.get(k)
            if cat is None:
                continue
            st.session_state[f"rfsim_{prefix}_{cat}_{k}"] = float(v_si) * sc
        st.rerun()

    if reset_clicked:
        for s in tuning_specs:
            kp = f"rfsim_slpreview_{prefix}_{s[0]}"
            for suf in ("", "_min", "_step", "_max"):
                st.session_state.pop(kp + suf, None)
        st.rerun()


@_FRAGMENT
def _render_rfsim_plotly_slider_preview(model_cls, all_p, freq, prefix: str,
                                         pad_specs, ext_specs, int_specs):
    """Pre-computed Plotly slider — joint (cartesian) multi-param scan.

    Same UX as the SSM version: each selected param gets its own slider,
    frames are the full cartesian product, JS coordinates the sliders so
    dragging one reflects the *current position* of all the others.
    """
    from tools.SSM.helpers import make_smith_bode_joint_slider_html

    tuning_specs = list(pad_specs) + list(ext_specs) + list(int_specs)
    label_for = {s[0]: s[1] for s in tuning_specs}
    options   = [s[0] for s in tuning_specs]

    # Shared label for the "🧮 Build animation" button — reused by every
    # caption / info message below that references it, so the two always match.
    _build_anim_lbl = i18n.tr("🧮 Build animation", "🧮 建立動畫")

    sel_key  = f"rfsim_slprev_pl_sel_{prefix}"
    selected = st.multiselect(
        i18n.tr("Sweep parameters", "掃描參數"),
        options=options,
        default=st.session_state.get(sel_key, [options[0]] if options else []),
        format_func=lambda k: label_for.get(k, k),
        key=sel_key,
        placeholder=i18n.tr("Choose options", "請選擇項目"),
        help=i18n.tr(
            "Each selected param gets its own Plotly slider in the figure. "
            "Frames are the FULL cartesian product — dragging one slider "
            "reflects the model at the current position of every other "
            "slider.",
            "每個選取的參數都會在圖中取得自己的 Plotly 滑桿。影格為所有參數的"
            "完整笛卡兒乘積 — 拖曳任一滑桿都會反映其餘滑桿目前位置下的模型。"))

    selected_specs = [s for s in tuning_specs if s[0] in selected]
    if not selected_specs:
        st.caption(i18n.tr(
            f"Select one or more parameters above and click **{_build_anim_lbl}**.",
            f"請先在上方選擇一或多個參數，再點擊 **{_build_anim_lbl}**。"))
        return

    # Default frames-per-axis shrinks as more params are selected so the
    # cartesian product (and thus the embedded payload) stays manageable
    # at full frequency fidelity: 11 for 1-2 params, 7 for 3, 5 for 4+.
    n_sel = len(selected_specs)
    default_frames = 11 if n_sel <= 2 else (7 if n_sel == 3 else 5)
    for spec in selected_specs:
        key, _, scale = spec[0], spec[1], spec[2]
        current_disp  = float(all_p.get(key, 0.0)) * scale
        kp = f"rfsim_slprev_pl_{prefix}_{key}"
        if f"{kp}_min" not in st.session_state:
            d_min, d_max, _ = _slider_default_range(current_disp)
            st.session_state[f"{kp}_min"]    = float(d_min)
            st.session_state[f"{kp}_max"]    = float(d_max)
            st.session_state[f"{kp}_frames"] = default_frames

    with st.expander(i18n.tr("📏 Slider ranges (min / max / frames)",
                             "📏 滑桿範圍（最小 / 最大 / 影格）"),
                     expanded=False):
        _min_lbl = i18n.tr("Min", "最小值")
        _max_lbl = i18n.tr("Max", "最大值")
        for row_start in range(0, len(selected_specs), 2):
            row_specs = selected_specs[row_start:row_start + 2]
            row_cols  = st.columns(len(row_specs))
            for col_w, spec in zip(row_cols, row_specs):
                key, label, scale = spec[0], spec[1], spec[2]
                unit = spec[3] if len(spec) > 3 else ""
                fmt  = spec[4] if len(spec) > 4 else "%.4g"
                kp   = f"rfsim_slprev_pl_{prefix}_{key}"
                with col_w:
                    st.markdown(f"**{label}** ({unit})" if unit
                                else f"**{label}**")
                    mmf = st.columns(3)
                    mmf[0].number_input(f"{_min_lbl} ({unit})" if unit
                                        else _min_lbl,
                                        format=fmt, key=f"{kp}_min")
                    mmf[1].number_input(f"{_max_lbl} ({unit})" if unit
                                        else _max_lbl,
                                        format=fmt, key=f"{kp}_max")
                    mmf[2].number_input(i18n.tr("Frames", "影格數"),
                                        min_value=2, max_value=100,
                                        step=1, key=f"{kp}_frames",
                                        help=i18n.tr(
                                            "Frames per axis (2–100). "
                                            "Total = product across params.",
                                            "每軸的影格數（2–100）。"
                                            "總數 = 各參數影格數之乘積。"))

    n_freq_full = int(len(freq))
    decim_default = min(120, n_freq_full)
    decim_key   = f"rfsim_slprev_pl_decim_{prefix}"
    if decim_key not in st.session_state:
        st.session_state[decim_key] = decim_default

    dims_preview = []
    for spec in selected_specs:
        kp = f"rfsim_slprev_pl_{prefix}_{spec[0]}"
        dims_preview.append(int(st.session_state.get(f"{kp}_frames",
                                                     default_frames)))
    total_frames = int(np.prod(dims_preview)) if dims_preview else 0

    decim_n = int(st.session_state.get(decim_key, decim_default))
    decim_n = min(decim_n, n_freq_full)
    est_mb  = total_frames * decim_n * 10 * 7 / 1024 / 1024

    fd_col1, fd_col2 = st.columns([1, 2])
    with fd_col1:
        st.number_input(
            i18n.tr(f"Freq points (max: {n_freq_full})",
                    f"頻率點數（上限：{n_freq_full}）"),
            min_value=20, max_value=n_freq_full, step=10,
            key=decim_key,
            help=i18n.tr(
                f"Frequency points kept per trace "
                f"(max = {n_freq_full} = full fidelity).  "
                "Lower this (~120 still looks smooth) when "
                "the payload estimate grows large.",
                f"每條軌跡保留的頻率點數（上限 = {n_freq_full} = 完整解析度）。"
                "當預估負載偏大時可調低此值（約 120 點看起來仍平滑）。"))
    with fd_col2:
        st.caption(
            i18n.tr("Cartesian sweep: ", "笛卡兒掃描：")
            + " × ".join(str(d) for d in dims_preview)
            + i18n.tr(
                f" = **{total_frames}** frames · {decim_n} freq pts · "
                f"estimated payload ≈ **{est_mb:.0f} MB**",
                f" = **{total_frames}** 影格 · {decim_n} 個頻率點 · "
                f"預估負載 ≈ **{est_mb:.0f} MB**"))
    if est_mb > 180:
        st.error(i18n.tr(
            f"❌ Estimated payload ≈ {est_mb:.0f} MB will exceed "
            "Streamlit's 200 MB browser-message limit.  Lower the "
            "**Freq points** value, reduce per-axis frame counts, or "
            "raise the limit via `.streamlit/config.toml` → "
            "`[server] maxMessageSize = 500`.",
            f"❌ 預估負載 ≈ {est_mb:.0f} MB 將超過 Streamlit 的 200 MB "
            "瀏覽器訊息上限。請降低 **頻率點數** 數值、減少每軸影格數，"
            "或透過 `.streamlit/config.toml` → "
            "`[server] maxMessageSize = 500` 調高上限。"))
    elif est_mb > 120:
        st.warning(i18n.tr(
            f"⚠️ Estimated payload ≈ {est_mb:.0f} MB is close "
            "to Streamlit's 200 MB limit.",
            f"⚠️ 預估負載 ≈ {est_mb:.0f} MB 已接近 Streamlit 的 200 MB 上限。"))
    elif total_frames > 2000:
        st.warning(i18n.tr(
            f"⚠️ {total_frames} frames may stutter on slider drag.",
            f"⚠️ {total_frames} 個影格在拖曳滑桿時可能會卡頓。"))

    _build_anim_help = i18n.tr(
        "Pre-compute the cartesian joint sweep and embed with "
        "JS-coordinated multi-sliders.",
        "預先計算笛卡兒聯合掃描，並嵌入以 JS 協調的多滑桿介面。")
    cuda_toggle_key = f"rfsim_slprev_pl_cuda_{prefix}"
    if _HAS_CUDA:
        cuda_col, btn_col = st.columns([1.6, 1])
        with cuda_col:
            use_cuda = st.checkbox(
                i18n.tr(f"⚡ Use CUDA (cupy {_CUDA_VER}) for batched simulation",
                        f"⚡ 使用 CUDA（cupy {_CUDA_VER}）進行批次模擬"),
                value=st.session_state.get(cuda_toggle_key, True),
                key=cuda_toggle_key,
                help=i18n.tr(
                    "Off-load joint cartesian batched simulation to GPU.  "
                    "Result is brought back to host as fp64.",
                    "將笛卡兒聯合批次模擬卸載至 GPU 執行，結果以 fp64 傳回主機。"))
        with btn_col:
            build_clicked = st.button(_build_anim_lbl,
                                      key=f"rfsim_slprev_pl_build_{prefix}",
                                      width="stretch",
                                      help=_build_anim_help)
    else:
        use_cuda = False
        build_clicked = st.button(_build_anim_lbl,
                                  key=f"rfsim_slprev_pl_build_{prefix}",
                                  width="stretch",
                                  help=_build_anim_help)

    state_key = f"rfsim_slprev_pl_state_{prefix}"

    if build_clicked:
        import time as _time
        xp = _cp if (use_cuda and _HAS_CUDA) else np
        device_label = (f"GPU (cupy {_CUDA_VER})"
                        if xp is not np else "CPU (numpy)")
        sweep_disps : list[np.ndarray] = []
        sweep_sis   : list[np.ndarray] = []
        slider_specs_out: list[dict] = []
        for spec in selected_specs:
            key, label, scale = spec[0], spec[1], spec[2]
            unit = spec[3] if len(spec) > 3 else ""
            fmt  = spec[4] if len(spec) > 4 else "%.4g"
            kp   = f"rfsim_slprev_pl_{prefix}_{key}"
            mn = float(st.session_state[f"{kp}_min"])
            mx = float(st.session_state[f"{kp}_max"])
            nf = int(st.session_state[f"{kp}_frames"])
            if mx <= mn:
                mx = mn + abs(mn) * 1e-6 + 1e-9
            sd = np.linspace(mn, mx, nf)
            sweep_disps.append(sd)
            sweep_sis.append(sd / scale)
            slider_specs_out.append(dict(
                label=label, unit=unit, fmt=fmt,
                values_disp=sd.tolist()))
        meshes = np.meshgrid(*sweep_sis, indexing="ij")
        flats  = [m.ravel() for m in meshes]
        n_total = int(flats[0].size) if flats else 0

        from tools.SSM.models.base_ui import _chunked_simulate_batch_to_host
        t0 = _time.perf_counter()
        with st.spinner(i18n.tr(f"Computing {n_total} frames on {device_label}…",
                                f"正在 {device_label} 上計算 {n_total} 個影格…")):
            p_batch = dict(all_p)
            for spec, flat in zip(selected_specs, flats):
                p_batch[spec[0]] = xp.asarray(flat, dtype=float)
            try:
                S_b = _chunked_simulate_batch_to_host(
                    model_cls, p_batch, freq, 50.0, xp=xp)
            except Exception as e:
                st.error(i18n.tr(f"Batched preview simulation failed: {e}",
                                 f"批次預覽模擬失敗：{e}"))
                return
        elapsed = _time.perf_counter() - t0

        if not np.all(np.isfinite(S_b)):
            st.warning(i18n.tr(
                "Some frames contain non-finite S-parameters — "
                "narrow the ranges to avoid singular combinations.",
                "部分影格含有非有限值的 S 參數 — 請縮小範圍以避免奇異組合。"))

        st.session_state[state_key] = {
            "slider_specs":  slider_specs_out,
            "S_batch":       S_b,
            "selected_keys": [s[0] for s in selected_specs],
            "elapsed_s":     elapsed,
            "device":        device_label,
            "n_total":       n_total,
        }

    state = st.session_state.get(state_key)
    if state is None:
        st.info(i18n.tr("Click **🧮 Build animation** to compute frames for "
                        "the Plotly slider(s).",
                        "點選 **🧮 Build animation** 以計算 Plotly 滑桿的影格。"))
        return

    cached_keys  = state.get("selected_keys", [])
    current_keys = [s[0] for s in selected_specs]
    if cached_keys != current_keys:
        st.warning(i18n.tr(
            "Selection changed since last build "
            f"(cached: {cached_keys}, current: {current_keys}).  "
            "Click **🧮 Build animation** to refresh.",
            "選取的參數自上次建立後已變更"
            f"（快取：{cached_keys}，目前：{current_keys}）。"
            "請點選 **🧮 Build animation** 重新整理。"))
        return

    elapsed = float(state.get("elapsed_s", 0.0))
    n_total = int(state.get("n_total", 0)) or len(state["S_batch"])
    device  = str(state.get("device", "?"))
    ms_each = (elapsed / max(1, n_total)) * 1000.0
    st.caption(i18n.tr(
        f"✅ Built **{n_total}** frames on **{device}** in "
        f"**{elapsed:.2f} s** ({ms_each:.1f} ms/frame).  "
        "Drag any slider below to scrub the joint sweep.",
        f"✅ 已在 **{device}** 上建立 **{n_total}** 個影格，耗時 "
        f"**{elapsed:.2f} 秒**（每影格 {ms_each:.1f} 毫秒）。"
        "拖曳下方任一滑桿即可瀏覽聯合掃描結果。"))

    html = make_smith_bode_joint_slider_html(
        S_batch_joint=state["S_batch"],
        freq=freq,
        slider_specs=state["slider_specs"],
        model_name=model_cls.NAME,
        S_meas=None,
        decimate_points=int(st.session_state.get(decim_key, decim_default)),
        # Mirror the page's (model-independent) Smith multipliers so the
        # Wide-sweep view scales in lock-step with the static Smith chart.
        smith_mults={nm: float(st.session_state.get(
            f"rfsim_smithmult_{nm}", 1.0))
            for nm in ("S11", "S12", "S21", "S22")},
    )
    n_sl = len(state["slider_specs"])
    iframe_height = 500 + 26 + 36 * n_sl + 30
    # st.iframe replaced components.v1.html (deprecated 2026-06-01).
    # When src is a raw HTML string (no http(s) / file / Path prefix)
    # Streamlit embeds it directly in an iframe — same behaviour as
    # the old components.html call.  No `scrolling` parameter; the
    # `height=` integer is interpreted in pixels just like before.
    st.iframe(html, height=iframe_height)


def _build_bode(S, freq_hz, title: str, *,
                extrap_method: str = "−20 dB/dec", sp_window=None):
    """Build the fT/fmax Bode figure.

    Returns ``(fig, any_needs)`` where ``any_needs`` is True when at least one
    trace still has positive gain at the top of the band (i.e. extrapolation is
    required to project a 0-dB crossing).  ``extrap_method`` selects the
    projection: ``"−20 dB/dec"`` (slope-locked) or ``"Single-pole"`` (log-linear
    least-squares fit over ``sp_window=(f_lo, f_hi)`` in GHz).
    """
    f_ghz_local = freq_hz * 1e-9
    h21_db, U_db = compute_h21_U(S)
    fT, fmax = find_ft_fmax(f_ghz_local, h21_db, U_db)

    def _needs(in_val, gain):
        if in_val is not None:
            return False
        if not np.any(np.isfinite(gain)):     # all-NaN slice → nothing to extrapolate
            return False
        with np.errstate(invalid="ignore"):
            return bool(np.nanmax(gain) > 0)
    any_needs = _needs(fT, h21_db) or _needs(fmax, U_db)

    def _extrap(y):
        if (extrap_method == "Single-pole" and sp_window is not None
                and len(f_ghz_local) >= 4):
            il = int(np.searchsorted(f_ghz_local, sp_window[0], side="left"))
            ih = int(np.searchsorted(f_ghz_local, sp_window[1], side="right")) - 1
            il = max(0, min(il, len(f_ghz_local) - 2))
            ih = max(il + 1, min(ih, len(f_ghz_local) - 1))
            r = single_pole_extrap(f_ghz_local, y, il, ih)
            return r[0], r[1], r[2]
        return extrap_20dbdec(f_ghz_local, y)

    fig = go.Figure()
    f_high_track = float(f_ghz_local[-1])
    extrap_used  = False
    # Collected for the standardised fT/fmax xlsx export (simulated +
    # extrapolated columns).  Labels match across the two lists so the
    # workbook reads "<trace>" / "<trace> (extrap)".
    sim_traces:    list[tuple[str, np.ndarray]] = []
    extrap_traces: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _meas_lbl(name, in_val, ext_val):
        if in_val is not None:
            return f"{name}={in_val:.2f} GHz"
        if ext_val is not None:
            return f"{name}≈{ext_val:.2f} GHz (extrap)"
        return f"{name}=n/a"

    def _add_trace(y, base_name, color, dash, kind, in_val, symbol):
        """Plot trace + extrapolation; bake the (in-band or extrap) value
        into the legend entry so fT/fmax always show.
        """
        nonlocal f_high_track, extrap_used
        f_ext, g_ext, f0 = _extrap(y)
        ext_val = f0 if f_ext is not None else None
        legend_name = f"{base_name}  [{_meas_lbl(kind, in_val, ext_val)}]"
        fig.add_trace(go.Scatter(x=f_ghz_local, y=y, mode="lines+markers",
                                 name=legend_name,
                                 line=dict(color=color, width=2, dash=dash),
                                 marker=dict(symbol=symbol, size=6, color=color)))
        sim_traces.append((f"{base_name} (dB)", np.asarray(y)))
        if f_ext is not None:
            extrap_used  = True
            f_high_track = max(f_high_track, f0)
            extrap_traces.append((f"{base_name} (dB)", f_ext, g_ext))
            fig.add_trace(go.Scatter(x=f_ext, y=g_ext, mode="lines",
                                     name=f"{legend_name} extrap",
                                     line=dict(color=color, width=2,
                                               dash="dot"),
                                     showlegend=False))

    # Standard colour scheme (FT_FMAX_COLORS): fT in blue, fmax in red.
    # Both traces are "measured" sims (no model comparison in this view),
    # so both are solid; extrap fall-throughs in `_add_trace` are dotted.
    _add_trace(h21_db, "|h21|²",  FT_FMAX_COLORS["fT"],   "solid", "fT",   fT,   FT_FMAX_SYMBOLS["h21"])
    _add_trace(U_db,   "Mason U", FT_FMAX_COLORS["fmax"], "solid", "fmax", fmax, FT_FMAX_SYMBOLS["U"])

    bode_xl = bode_excel_bytes(f_ghz_local, sim_traces, extrap_traces)

    fig.add_hline(y=0, line_color="#333", line_width=1.2,
                  annotation_text="0 dB", annotation_position="right",
                  annotation_font=dict(size=9))

    x_min = max(float(f_ghz_local[0]), 1e-2)
    x_max = (float(f_high_track) * 1.25 if extrap_used
             else float(f_ghz_local[-1]))
    fig.update_layout(
        title=dict(text=f"fT / fmax — {title}", font=dict(size=12)),
        xaxis=dict(title="Frequency (GHz)", type="log",
                   range=[np.log10(x_min), np.log10(x_max)],
                   showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(title="Gain (dB)", range=[0, 50],
                   showgrid=True, gridcolor="#ebebeb"),
        plot_bgcolor="white", paper_bgcolor="white", height=560,
        # Legend pinned bottom-left INSIDE the plot area (paper coords,
        # anchored bottom-left) instead of below the chart.
        legend=dict(orientation="v", x=0.01, y=0.01,
                    xanchor="left", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#ccc", borderwidth=1, font=dict(size=13)),
        hovermode="x unified", margin=dict(l=55, r=20, t=40, b=50),
    )
    return fig, any_needs, bode_xl


def _render_bode_block(S, freq_hz, title: str, key: str):
    """Render the fT/fmax Bode plot, then (if extrapolation is needed) an
    extrapolation-method radio (left) and single-pole window slider (right)
    UNDERNEATH the chart.  The figure reads the current selection from
    session_state, so a Streamlit rerun on widget change feeds it back here.
    """
    f_ghz_local = freq_hz * 1e-9
    method = st.session_state.get(f"{key}_extrap_method", "−20 dB/dec")
    sp_window = None
    if method == "Single-pole" and len(f_ghz_local) >= 4:
        f_lo, f_hi = float(f_ghz_local[0]), float(f_ghz_local[-1])
        sp_window = st.session_state.get(f"{key}_sp_window",
                                         (max(f_lo, f_hi - 5.0), f_hi))
    fig, any_needs, bode_xl = _build_bode(S, freq_hz, title,
                                          extrap_method=method, sp_window=sp_window)
    plotly_with_dl(fig, key=key, filename=key, excel_bytes=bode_xl)

    if any_needs and len(f_ghz_local) >= 2:
        ec1, ec2 = st.columns([1, 2])
        if f"{key}_extrap_method" not in st.session_state:
            st.session_state[f"{key}_extrap_method"] = "−20 dB/dec"
        with ec1:
            # Values stay canonical English — compared just below and stored
            # in session_state; only the chip label localizes.
            segmented_radio(
                i18n.tr("Extrap. method", "外插方法"),
                ["−20 dB/dec", "Single-pole"],
                key=f"{key}_extrap_method",
                format_func=lambda m: i18n.tr(m, {"−20 dB/dec": "−20 dB/dec",
                                                  "Single-pole": "單極點"}[m]),
                help=i18n.tr(
                    "−20 dB/dec anchors a slope-locked line at the last data "
                    "point.  Single-pole fits a log-linear line over the chosen "
                    "window (default = final 5 GHz).",
                    "−20 dB/dec 以最後一個資料點為錨點，畫出固定斜率的直線。"
                    "單極點則在所選視窗內（預設為最後 5 GHz）擬合對數線性直線。"))
        if (st.session_state[f"{key}_extrap_method"] == "Single-pole"
                and len(f_ghz_local) >= 4):
            f_lo, f_hi = float(f_ghz_local[0]), float(f_ghz_local[-1])
            sp_default = (max(f_lo, f_hi - 5.0), f_hi)
            ec2.slider(
                i18n.tr("Single-pole fit window (GHz)", "單極點擬合視窗 (GHz)"),
                min_value=f_lo, max_value=f_hi,
                value=st.session_state.get(f"{key}_sp_window", sp_default),
                step=max((f_hi - f_lo) / 400.0, 1e-3),
                key=f"{key}_sp_window")


# ═════════════════════════════════════════════════════════════════════════════
# Branch by model choice
# ═════════════════════════════════════════════════════════════════════════════

# Open/Short use a small subset of the pad/lead specs
_OPEN_SPECS = [
    ("Cpbe", "Cpbe", 1e15, "fF", "%.4f", 0.1),
    ("Cpbc", "Cpbc", 1e15, "fF", "%.4f", 0.01),
    ("Cpce", "Cpce", 1e15, "fF", "%.4f", 0.1),
]
_SHORT_SPECS = [
    ("Lb", "Lb", 1e12, "pH", "%.3f", 0.1),
    ("Lc", "Lc", 1e12, "pH", "%.3f", 0.1),
    ("Le", "Le", 1e12, "pH", "%.3f", 0.01),
]


if model_choice == "Open and Short Pad":
    st.markdown(i18n.tr("### Inputs", "### 輸入值"))
    col_in_o, col_in_s = st.columns(2)
    with col_in_o:
        _render_spec_inputs(_OPEN_SPECS, "open",  "Open Pad Capacitances",
                            cols_per_row=3)
    with col_in_s:
        _render_spec_inputs(_SHORT_SPECS, "short", "Short Pad Inductances",
                            cols_per_row=3)

    p_open  = _collect_specs(_OPEN_SPECS, "open")
    p_short = {**_collect_specs(_OPEN_SPECS, "open"),
               **_collect_specs(_SHORT_SPECS, "short"),
               "Rpb": 0.0, "Rpc": 0.0, "Rpe": 0.0}

    try:
        S_open = simulate_open(p_open, freq)
    except Exception as e:
        st.error(i18n.tr(f"Open simulation failed: {e}",
                         f"開路模擬失敗：{e}"))
        S_open = np.full((len(freq), 2, 2), np.nan + 0j)
    try:
        S_short = simulate_short(p_short, freq)
    except Exception as e:
        st.error(i18n.tr(f"Short simulation failed: {e}",
                         f"短路模擬失敗：{e}"))
        S_short = np.full((len(freq), 2, 2), np.nan + 0j)

    mults = _smith_multiplier_inputs(
        "os", label=i18n.tr("Smith multipliers (apply to both charts)",
                            "Smith 倍率（同時套用於兩張圖）"))

    col_chart_o, col_chart_s = st.columns(2)
    with col_chart_o:
        st.markdown(i18n.tr("**Open Pad**", "**開路焊墊**"))
        _smith_chart_with_dl(
            _build_smith(S_open, freq, mults, "Open Pad"),
            key="rfsim_smith_open",
            filename="rfsim_open_smith",
            s2p_data=write_s2p(freq, S_open,
                               title="RF simulator — Open pad",
                               params={k: f"{v:g}" for k, v in p_open.items()
                                       if isinstance(v, (int, float))}),
            s2p_filename="rf_sim_open.s2p",
        )
    with col_chart_s:
        st.markdown(i18n.tr("**Short Pad**", "**短路焊墊**"))
        _smith_chart_with_dl(
            _build_smith(S_short, freq, mults, "Short Pad"),
            key="rfsim_smith_short",
            filename="rfsim_short_smith",
            s2p_data=write_s2p(freq, S_short,
                               title="RF simulator — Short pad",
                               params={k: f"{v:g}" for k, v in p_short.items()
                                       if isinstance(v, (int, float))}),
            s2p_filename="rf_sim_short.s2p",
        )

    # Chart on the LEFT, controls on the RIGHT — same split-call pattern the
    # Cheng/Xu models use above.  The "controls" phase must run before the
    # "chart" phase (it writes the session state the chart reads), so the
    # right column is invoked first in code even though it sits on the right.
    with st.container(key="hbt_exp_view_mplsmith_open"), \
         st.expander(i18n.tr("📐 Smith chart (Matplotlib) — Open",
                             "📐 Smith 圖 (Matplotlib) — 開路"),
                     expanded=False):
        col_o_left, col_o_right = st.columns([1.2, 1])
        with col_o_right:
            render_matplotlib_smith(
                fname="rfsim", topo_key="open",
                sets=[{"S": S_open, "label": "Open",
                       "kind": "line", "style": "solid"}],
                default_multiplier=mults, phase="controls", freq_hz=freq,
            )
        with col_o_left:
            render_matplotlib_smith(
                fname="rfsim", topo_key="open",
                sets=[{"S": S_open, "label": "Open",
                       "kind": "line", "style": "solid"}],
                default_multiplier=mults, phase="chart", freq_hz=freq,
            )
    with st.container(key="hbt_exp_view_mplsmith_short"), \
         st.expander(i18n.tr("📐 Smith chart (Matplotlib) — Short",
                             "📐 Smith 圖 (Matplotlib) — 短路"),
                     expanded=False):
        col_s_left, col_s_right = st.columns([1.2, 1])
        with col_s_right:
            render_matplotlib_smith(
                fname="rfsim", topo_key="short",
                sets=[{"S": S_short, "label": "Short",
                       "kind": "line", "style": "solid"}],
                default_multiplier=mults, phase="controls", freq_hz=freq,
            )
        with col_s_left:
            render_matplotlib_smith(
                fname="rfsim", topo_key="short",
                sets=[{"S": S_short, "label": "Short",
                       "kind": "line", "style": "solid"}],
                default_multiplier=mults, phase="chart", freq_hz=freq,
            )

else:
    # ─── Cheng T / Pi / Xu T ──────────────────────────────────────────────
    if model_choice == "Cheng's T":
        model_cls = ChengT
        ext_specs = _EXT_T_SPECS
        int_specs = _INT_T_SPECS
        topo_char = "T"
        prefix    = "ssm_T"
    elif model_choice == "Xu T":
        model_cls = XuModel
        ext_specs = _XU_EXT_SPECS
        int_specs = _XU_INT_SPECS
        topo_char = "T"
        prefix    = "ssm_XuT"
        # Rbcx defaults to 285 kΩ — pre-init so default sim doesn't see Rbcx=0 → Ybcx=∞
        _rbcx_sk = f"rfsim_{prefix}_ext_Rbcx"
        if _rbcx_sk not in st.session_state:
            st.session_state[_rbcx_sk] = 285.0
    elif model_choice == "Kun-Yang HEMT":
        model_cls = KunYangHEMT
        ext_specs = _EXT_KY_SPECS
        int_specs = _INT_KY_SPECS
        topo_char = "pi"
        prefix    = "ssm_KY"
        # Pre-init the intrinsic + custom-pad inputs with the model defaults so
        # the first render produces a finite Smith chart (Rds=0 would explode).
        for _key, _, _sc, *_ in _EXT_KY_SPECS + _INT_KY_SPECS:
            _sk = (f"rfsim_{prefix}_ext_{_key}"
                   if _key in {k for k, *_ in _EXT_KY_SPECS}
                   else f"rfsim_{prefix}_int_{_key}")
            if _sk not in st.session_state:
                st.session_state[_sk] = float(
                    _KY_DEFAULT_PARAMS.get(_key, 0.0)) * float(_sc)
    else:
        model_cls = ChengPi
        ext_specs = _EXT_PI_SPECS
        int_specs = _INT_PI_SPECS
        topo_char = "pi"
        prefix    = "ssm_pi"

    # ── Fit mode — a measured device is loaded.  Reuse the extraction-grade
    #    override → residual → Visual/Auto tuning UI instead of the forward-
    #    sim-only inputs below.  When a handoff carried extracted values we
    #    seed from those; otherwise we auto-guess from the device (the same
    #    one-shot seed render_builtin_forward_sim uses), so the first sim is
    #    never singular.
    if measured is not None:
        _short = model_cls.SHORT
        _seed  = st.session_state.get(f"_simfit_seed_{_short}")
        _fit_fname = f"simfit_{_short}_{measured['label']}"
        st.markdown(i18n.tr(f"### 🎯 Fit — {model_cls.NAME}",
                            f"### 🎯 擬合 — {model_cls.NAME}"))

        from tools.SSM.helpers.fit_cache import get_fit_timestamp
        _cache_ts = get_fit_timestamp(_fit_fname, _short)
        _tooltip = (f"{measured['stage']} · {f_ghz[0]:.3g}–{f_ghz[-1]:.3g} GHz · "
                    f"{len(freq)} pts — the simulation follows the measured grid "
                    "point-for-point. Edit any parameter, read the residual, and "
                    "use the Visual / Auto tuning expanders to fit.")
        _pill_row = (
            "<span style='color:#808495'><small>Fitting</small></span> "
            f"<span class='hbt-chip-file' title='{_tooltip}'>{measured['label']}</span>"
        )
        if _cache_ts:
            from datetime import datetime
            try:
                _dt = datetime.fromisoformat(_cache_ts)
                if _dt.date() == datetime.now().date():
                    _short_ts = _dt.strftime("%H:%M")
                else:
                    _short_ts = _dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                _short_ts = _cache_ts
            _pill_row += (
                " <span class='hbt-chip-cache' title='A fit for this device+model was "
                "saved earlier and auto-applies on entry (a fresh handoff from "
                "Extraction takes priority). Load it anytime with “Use cache” inside "
                f"the Fine-tune expander.'>📌 cache from {_short_ts}</span>"
            )
        _pill_row += (
            " <span style='color:#808495'><small>→</small></span> "
            f"<span class='hbt-chip-model'>{model_cls.NAME}</span>"
        )
        st.markdown(_pill_row, unsafe_allow_html=True)

        if _seed:
            _fresh = st.session_state.pop(f"_simfit_seed_pending_{_short}", False)
            para_eff = {k: float(_seed.get(k, 0.0)) for k, *_ in PAD_SPECS}
            model_cls.render_override_and_smith(
                _fit_fname, measured["S"], freq, measured["z0"],
                para_eff, (dict(_seed), {}), show_tuning=True,
                prefer_calc_vals=_fresh, show_cache_banner=False)
        else:
            from tools.SSM.main_ssm_extraction import render_builtin_forward_sim
            render_builtin_forward_sim(_short, measured["S"], freq,
                                       measured["z0"], _fit_fname,
                                       show_header=False, show_cache_banner=False)
        st.stop()

    # Split pad specs by group for the requested layout.  Xu uses its own
    # pad-label aliases (Rb→Rbx, Re→Rex, Cpce→Cpad) but identical keys.
    _pad_open_keys  = {"Cpbe", "Cpce", "Cpbc"}
    _pad_short_keys = {"Lb", "Lc", "Le"}
    _pad_r_order    = ["Rpe", "Rpb", "Rpc"]    # Re, Rb, Rc

    if model_cls is XuModel:
        _pad_specs_for_model = _XU_PAD_SPECS
    elif model_cls is KunYangHEMT:
        _pad_specs_for_model = _KY_PAD_SPECS
    else:
        _pad_specs_for_model = PAD_SPECS
    pad_open_specs  = [s for s in _pad_specs_for_model if s[0] in _pad_open_keys]
    pad_short_specs = [s for s in _pad_specs_for_model if s[0] in _pad_short_keys]
    pad_r_specs     = sorted(
        [s for s in _pad_specs_for_model if s[0] in _pad_r_order],
        key=lambda s: _pad_r_order.index(s[0]))

    def _render_pad_row(specs):
        for col_w, spec in zip(st.columns(3), specs):
            key, lbl, sc, unit, fmt, step = spec
            sk = f"rfsim_{prefix}_pad_{key}"
            if sk not in st.session_state:
                st.session_state[sk] = 0.0
            col_w.number_input(f"{lbl} ({unit})", key=sk,
                               format=fmt, step=step)

    st.markdown(i18n.tr("### Inputs", "### 輸入值"))
    with st.container(key="hbt_exp_edit_rfsim_" + prefix), \
         st.expander(i18n.tr(f"✏️ {model_cls.NAME} parameters",
                            f"✏️ {model_cls.NAME} 參數"), expanded=True):
        # Values stay canonical English — `_mode == "Diagram"` below and the
        # stored session_state value must not shift with the UI language.
        _mode = segmented_radio(
            i18n.tr("Editor mode", "編輯模式"), ["List", "Diagram"],
            key=f"rfsim_mode_{prefix}",
            format_func=lambda m: i18n.tr(m, {"List": "清單",
                                              "Diagram": "示意圖"}[m]),
            help=i18n.tr(
                "List: grouped number inputs.  Diagram: set values on the "
                "model schematic — the component you edit is highlighted.",
                "清單：分組數值輸入。示意圖：直接在模型電路圖上設定數值 — "
                "正在編輯的元件會被標示出來。"))

        if _mode == "Diagram":
            # Each param maps to its own widget-key scheme: pad keys live under
            # the "_pad_" sub-prefix, extrinsic under "_ext_", intrinsic under
            # "_int_" — the same keys the List view + _collect_specs use.
            _pad_keys = {s[0] for s in _pad_specs_for_model}
            _ext_keys = {s[0] for s in ext_specs}

            def _rf_state_key(k, _pk=_pad_keys, _ek=_ext_keys, _px=prefix):
                if k in _pk:
                    return f"rfsim_{_px}_pad_{k}"
                if k in _ek:
                    return f"rfsim_{_px}_ext_{k}"
                return f"rfsim_{_px}_int_{k}"

            if model_cls is XuModel:
                _ill = lambda pp, hl: _render_xu_illustration(
                    pp, f"rfsim_{prefix}", highlight_key=hl)
            elif model_cls is KunYangHEMT:
                _ill = lambda pp, hl: _render_ky_illustration(
                    pp, f"rfsim_{prefix}", highlight_key=hl)
            else:
                _ill = lambda pp, hl: _render_topology_illustration(
                    pp, topo_char, f"rfsim_{prefix}", highlight_key=hl)

            render_finetune_diagram(
                all_specs=list(_pad_specs_for_model) + list(ext_specs) + list(int_specs),
                state_key_for=_rf_state_key,
                active_state=f"rfsim_dia_active_{prefix}",
                calc_vals={}, render_illustration=_ill)
        elif model_cls is KunYangHEMT:
            # KY: substrate / custom pad FIRST (these caps ARE the pad layer),
            # then access resistance + lead inductances.  No Cpg / Cpd / Cpgd
            # row — those entries are not used by the Kun-Yang model.
            _render_spec_inputs(
                ext_specs, prefix + "_ext",
                i18n.tr("Kun-Yang Custom Pad / Substrate Network",
                        "Kun-Yang 自訂焊墊 / 基板網路"))
            st.markdown(i18n.tr("**Access Resistance & Lead Inductance**",
                                "**存取電阻與引線電感**"))
            _render_pad_row(pad_short_specs)
            _render_pad_row(pad_r_specs)
            _render_spec_inputs(int_specs, prefix + "_int",
                                i18n.tr("Intrinsic π-Model", "本質 π 模型"))
        else:
            st.markdown(i18n.tr("**Pad Parasitics**", "**焊墊寄生參數**"))
            _render_pad_row(pad_open_specs)
            _render_pad_row(pad_short_specs)
            st.markdown(i18n.tr("**Access Resistance**", "**存取電阻**"))
            _render_pad_row(pad_r_specs)
            _render_spec_inputs(ext_specs, prefix + "_ext",
                                i18n.tr("Extrinsic Caps", "外質電容"))
            _render_spec_inputs(int_specs, prefix + "_int",
                                i18n.tr("Intrinsic", "本質參數"))

    p = {**_collect_specs(PAD_SPECS, prefix + "_pad"),
         **_collect_specs(ext_specs, prefix + "_ext"),
         **_collect_specs(int_specs, prefix + "_int")}

    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            S_sim = model_cls.simulate(p, freq)
    except Exception as e:
        st.error(i18n.tr(f"Simulation failed: {e}", f"模擬失敗：{e}"))
        S_sim = np.full((len(freq), 2, 2), np.nan + 0j)

    if not np.all(np.isfinite(S_sim)):
        st.warning(i18n.tr(
            "Simulated S-parameters contain non-finite values "
            "(some intrinsic parameters are zero or singular). "
            "Adjust inputs above to see a meaningful trace.",
            "模擬出的 S 參數含有非有限值（部分本質參數為 0 或矩陣奇異）。"
            "請調整上方輸入值以得到有意義的曲線。"))

    mults = _smith_multiplier_inputs(prefix)

    col_smith, col_bode = st.columns(2)
    with col_smith:
        st.markdown(i18n.tr(f"**Smith Chart — {model_cls.NAME}**",
                            f"**Smith 圖 — {model_cls.NAME}**"))
        _smith_chart_with_dl(
            _build_smith(S_sim, freq, mults, model_cls.NAME),
            key=f"rfsim_smith_{prefix}",
            filename=f"rfsim_smith_{prefix}",
            s2p_data=write_s2p(freq, S_sim,
                               title=f"RF simulator — {model_cls.NAME}",
                               params={k: f"{v:g}" for k, v in p.items()
                                       if isinstance(v, (int, float))}),
            s2p_filename=f"rf_sim_{prefix}.s2p",
        )
    with col_bode:
        st.markdown(i18n.tr(f"**fT / fmax — {model_cls.NAME}**",
                            f"**fT / fmax — {model_cls.NAME}**"))
        _render_bode_block(S_sim, freq, model_cls.NAME,
                           key=f"rfsim_bode_{prefix}")

    # τ_total + calculated fmax expander — every SSM model except Kun-Yang
    # HEMT (which lacks Cbcx/Cbc/Rbi/Rb for the fmax formula).
    if model_cls is not KunYangHEMT:
        _CBC = float(p.get("Cbcx", 0.0)) + float(p.get("Cbc", 0.0))
        _Rbb = float(p.get("Rbi", 0.0)) + float(p.get("Rpb", 0.0))
        if model_cls is ChengPi:
            _tau_sum = float(p.get("tau", 0.0))
            _tau_lbl, _tau_tex = "τ", r"\tau"
        else:
            _tau_sum = float(p.get("tauB", 0.0)) + float(p.get("tauC", 0.0))
            _tau_lbl, _tau_tex = "τB + τC", r"\tau_B+\tau_C"
        render_tau_fmax_expander(key=f"rfsim_taufmax_{prefix}", freq=freq,
                                 S_meas=S_sim, CBC=_CBC, Rbb=_Rbb,
                                 tau_sum=_tau_sum, tau_sum_label=_tau_lbl,
                                 tau_sum_tex=_tau_tex,
                                 extrap_key=f"rfsim_bode_{prefix}")

    from tools.SSM.ssm_plots import render_matplotlib_smith

    # Topology illustration and matplotlib Smith chart go in their own
    # expanders so users can collapse each independently — mirrors how
    # the SSM extraction tab keeps these on separate axes.
    # Rebuild the user's customized Smith chart (same fname/topo_key the
    # "🍩 Smith Chart (Matplotlib)" expander below uses) as PNG bytes so the
    # no-parasitics topology view can overlay it bottom-right.  phase="chart"
    # reads session_state and creates no widgets.
    _smith_png = None
    try:
        _smith_png = render_matplotlib_smith(
            fname=f"rfsim_{prefix}", topo_key=topo_char,
            sets=[{"S": S_sim, "label": "Simulated",
                   "kind": "line", "style": "solid"}],
            default_multiplier=mults, phase="chart", freq_hz=freq,
            return_png=True)
    except Exception:                                    # noqa: BLE001
        _smith_png = None

    with st.container(key="hbt_exp_view_topo_rfsim"), \
         st.expander(i18n.tr("🖼️ Topology Illustration", "🖼️ 拓樸示意圖"),
                     expanded=False):
        try:
            if model_cls is XuModel:
                _render_xu_illustration(p, f"rfsim_{prefix}", smith_png=_smith_png)
            elif model_cls is KunYangHEMT:
                _render_ky_illustration(p, f"rfsim_{prefix}")
            else:
                _render_topology_illustration(p, topo_char,
                                               f"rfsim_{prefix}", smith_png=_smith_png)
        except Exception as e:
            st.warning(i18n.tr(f"Topology illustration unavailable: {e}",
                               f"無法顯示拓樸示意圖：{e}"))

    with st.container(key="hbt_exp_view_mplsmith_rfsim"), \
         st.expander(i18n.tr("🍩 Smith Chart (Matplotlib)",
                             "🍩 Smith 圖 (Matplotlib)"), expanded=False):
        # Controls on the right column, chart on the left — same
        # split-call pattern the SSM tab uses inside its expander.
        col_mpl_left, col_mpl_right = st.columns([1.2, 1])
        with col_mpl_right:
            render_matplotlib_smith(
                fname=f"rfsim_{prefix}", topo_key=topo_char,
                sets=[{"S": S_sim, "label": "Simulated",
                       "kind": "line", "style": "solid"}],
                default_multiplier=mults,
                phase="controls", freq_hz=freq,
            )
        with col_mpl_left:
            render_matplotlib_smith(
                fname=f"rfsim_{prefix}", topo_key=topo_char,
                sets=[{"S": S_sim, "label": "Simulated",
                       "kind": "line", "style": "solid"}],
                default_multiplier=mults,
                phase="chart", freq_hz=freq,
            )

    with st.container(key="hbt_exp_tune_rfsim_" + prefix), \
         st.expander(i18n.tr("🔧 Tuning — Interactive slider preview",
                             "🔧 調諧 — 互動式滑桿預覽"), expanded=False):
        _render_slider_preview(model_cls, p, freq, mults, prefix,
                                _pad_specs_for_model, ext_specs, int_specs)
