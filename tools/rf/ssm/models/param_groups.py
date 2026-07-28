"""
models/param_groups.py — Diagram-mode fine-tune editor + the Interactive
Parameter Extraction expander (per-group frequency-range sliders, plots,
and per-parameter overrides).

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).
"""
from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from ..helpers import plotly_with_dl, quickset_buttons, apply_pending
from tools.common.i18n import tr

from .fit_sections import _render_cbex_sweep_tool, _render_tau_total_fit_section


# Component grouping for the diagram-mode fine-tune editor (the "✏️ Fine-tune"
# override expander).  Keys are matched against each model's spec list; any spec
# key not named here lands in a trailing "Other" group, so nothing is hidden.
_FINETUNE_DIAGRAM_GROUPS_EN = [
    ("Pad parasitics",    ["Cpbe", "Cpbc", "Cpce", "Cgsp", "Cdsp", "Cgdp",
                           "Rsub1", "Rsub2"]),
    ("Lead inductance",   ["Lb", "Lc", "Le"]),
    ("Access resistance", ["Rpb", "Rpc", "Rpe"]),
    ("Extrinsic C",       ["Cbex", "Cbcx", "Rbcx"]),
    ("Delay",             ["tauB", "tauC", "tau", "R_delay", "C_delay"]),
    ("Intrinsic",         ["Rbi", "Rbe", "Cbe", "Cbc", "Rbc", "alpha0", "Gm0"]),
]
# label → zh translation — looked up at render time (via _finetune_diagram_groups())
# so a language switch mid-session relabels the groups immediately, instead of
# baking in whatever language was active when the module was first imported.
_FINETUNE_DIAGRAM_GROUP_ZH = {
    "Pad parasitics":    "焊墊寄生",
    "Lead inductance":   "引線電感",
    "Access resistance": "存取電阻",
    "Extrinsic C":        "外徵電容",
    "Delay":              "延遲",
    "Intrinsic":           "本徵",
}


def _finetune_diagram_groups():
    """Localized ``(label, keys)`` pairs — call fresh each render (see module
    docstring above ``_FINETUNE_DIAGRAM_GROUP_ZH``); never cache at import
    time since :func:`tr` depends on the current session's language."""
    return [(tr(lbl, _FINETUNE_DIAGRAM_GROUP_ZH[lbl]), keys)
            for lbl, keys in _FINETUNE_DIAGRAM_GROUPS_EN]


def render_finetune_diagram(*, all_specs, state_key_for, active_state,
                            calc_vals, render_illustration):
    """Diagram-mode alternative for a model's "✏️ Fine-tune" override expander.

    Two columns:
      • Left  — the model topology schematic, with the component the user last
        edited ringed in red (via ``render_illustration(preview, highlight)``).
      • Right — every override value, grouped (parasitics / lead L / access R /
        extrinsic C / delay / intrinsic).  Each ``number_input`` writes the SAME
        session-state key the list view uses, so the two modes stay in lock-step
        and the caller's downstream parameter assembly is unchanged.  Editing a
        field also moves the highlight to that component.

    Parameters
    ----------
    all_specs        : list of ``(key, label, scale, unit, fmt, step)``.
    state_key_for    : ``f(param_key) -> session_state key`` — lets each caller
                       map a param to its own widget-key scheme (SSM extraction
                       uses ``sim_{topo}_{key}_{fname}``; the RF simulator uses
                       ``rfsim_{prefix}_{pad|ext|int}_{key}``).
    active_state     : session_state key holding the highlighted param.
    calc_vals        : SI defaults used to seed an unset widget (``{}`` ⇒ 0).
    render_illustration : ``f(preview_all_p_SI, highlight_key) -> None``.
    """
    _diagram_groups = _finetune_diagram_groups()
    spec_lookup = {s[0]: s for s in all_specs}
    ordered: list[str] = []
    for _lbl, keys in _diagram_groups:
        ordered += [k for k in keys if k in spec_lookup]
    groups = [(lbl, [k for k in keys if k in spec_lookup])
              for lbl, keys in _diagram_groups]
    other = [s[0] for s in all_specs if s[0] not in ordered]
    if other:
        groups.append((tr("Other", "其他"), other))

    col_diag, col_inp = st.columns([1.1, 1], gap="medium")

    with col_inp:
        st.caption(tr("Edit any value — the diagram highlights the component you "
                      "last changed.",
                      "編輯任一數值 — 示意圖會標示你最後修改的元件。"))
        for g_label, keys in groups:
            if not keys:
                continue
            st.markdown(f"**{g_label}**")
            for i in range(0, len(keys), 2):
                cols = st.columns(2)
                for col_w, key in zip(cols, keys[i:i + 2]):
                    _k, lbl, sc, unit, fmt, step = spec_lookup[key]
                    skey = state_key_for(key)
                    if skey not in st.session_state:
                        st.session_state[skey] = float(calc_vals.get(key, 0.0)) * sc

                    def _mk(_key=key):
                        def _cb():
                            st.session_state[active_state] = _key
                        return _cb

                    col_w.number_input(
                        f"{lbl} ({unit})" if unit else lbl,
                        key=skey, format=fmt, step=step, on_change=_mk())

    with col_diag:
        active = st.session_state.get(active_state)
        preview = {
            s[0]: float(st.session_state.get(
                state_key_for(s[0]),
                float(calc_vals.get(s[0], 0.0)) * s[2])) / s[2]
            for s in all_specs}
        render_illustration(preview, active)
        if active:
            st.caption(f"{tr('Editing', '編輯中')} **{active}**")


def render_interactive_param_groups(params, arrays, freq, fname, model_short, param_groups,
                                    cold_res=None, cold_param_map=None, reextract_fn=None,
                                    cbex_sweep_fn=None,
                                    all_data=None, para_eff=None):

    """
    For each parameter group, render:
      - A labelled section heading with dependency info
      - A "same range as previous" button (for dependent groups)
      - A frequency range slider
      - Per-frequency line plots with a dashed line at the current median
      - A number_input per parameter for manual override
      - (Groups flagged `cbex_sweep_group` only) a Cbex sweep tool that
        searches for the Cbex value that minimises std(Cbcx_arr) across
        the Cbcx-group's selected frequency window.
      - (Groups flagged `tau_total_fit_group` only) the multi-file
        1/(2πfT) vs 1/IC reference fit — auto-hidden when only one s2p
        file is loaded.  Needs ``all_data`` and ``para_eff`` to be passed
        through; otherwise the group is silently skipped.
    Returns a copy of params with all overrides applied (SI units).
    """

    f_ghz   = freq * 1e-9
    f_min_v = float(f_ghz[0])
    f_max_v = float(f_ghz[-1])
    step_v  = max(round((f_max_v - f_min_v) / 100, 3), 0.001)

    params_out = dict(params)
    live_arrays = dict(arrays)   # updated mid-loop after re-extraction
    live_params = dict(params)   # updated mid-loop; used for change detection
    prev_range = (f_min_v, f_max_v)

    with st.expander(tr("📊 Interactive Parameter Extraction", "📊 互動式參數萃取"), expanded=False):
        # Thick outline on the large per-group boxes so each parameter group
        # reads as a bold card, set apart from the thin per-parameter
        # sub-containers inside it.  Streamlit puts the user `key` class on the
        # inner `stVerticalBlock` element (NOT the border wrapper), so we draw
        # the border directly on that keyed element and disable Streamlit's own
        # `border=True` wrapper (below) to avoid a double frame.  Scoped to the
        # `st-key-pfp_groupbox_*` prefix, so no other container is affected.
        st.markdown(
            """<style>
            div[class*="st-key-pfp_groupbox_"] {
                border: 3px solid #9aa0a6 !important;
                border-radius: 0.5rem !important;
                padding: 0.75rem !important;
            }
            </style>""",
            unsafe_allow_html=True)
        for g_idx, group in enumerate(param_groups):
            g_label  = group["label"]
            g_params = group["params"]
            g_deps   = group.get("depends_on", [])

            # ── tau_total_fit_group: multi-file 1/(2πfT) vs 1/IC reference ──
            # Rendered as its own nested expander, so no leading heading.
            # Silently skips when called from a single-file context or
            # when the caller didn't pass all_data/para_eff.
            if group.get("tau_total_fit_group"):
                _tau_rendered = False
                if all_data is not None and para_eff is not None:
                    _tau_rendered = bool(_render_tau_total_fit_section(
                        all_data=all_data, fname=fname,
                        model_short=model_short,
                        params=live_params, para_eff=para_eff))
                # Only emit the trailing separator when the section actually
                # rendered.  With a single s2p file the fit is hidden, so
                # skipping the rule avoids a double "---" (the previous group
                # already drew one) showing as two empty lines.
                if _tau_rendered and g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # Heading: special fit groups (Z-plots / Fbi / F1) print it here;
            # normal parameter groups print it inside their bordered box below.
            _is_special_group = any(group.get(_k) for _k in
                                    ("z_plots_group", "fbi_fit_group",
                                     "f1_fit_group"))
            if _is_special_group:
                st.markdown(f"**{g_label}**")

            # ── z_plots_group: Z1, Z3, Z4 Re/Im plots ───────────────────────────
            if group.get("z_plots_group"):
                for formula_type, formula_content in group.get("formulas", []):
                    if formula_type == "latex":
                        st.latex(formula_content)
                    else:
                        st.markdown(formula_content)

                sl_key = f"pfp_sl_{model_short}_{g_idx}_{fname}"
                if sl_key not in st.session_state:
                    st.session_state[sl_key] = (f_min_v, f_max_v)
                f_lo, f_hi = st.slider(
                    tr("Frequency range (GHz)", "頻率範圍 (GHz)"),
                    min_value=f_min_v, max_value=f_max_v,
                    value=st.session_state[sl_key],
                    step=step_v, format="%.2f", key=sl_key)
                mask   = (f_ghz >= f_lo) & (f_ghz <= f_hi)
                f_plot = f_ghz[mask]

                for zlabel, zarr in [("Z1", live_arrays.get("Z1")),
                                      ("Z3", live_arrays.get("Z3")),
                                      ("Z4", live_arrays.get("Z4"))]:
                    if zarr is None:
                        continue
                    c1, c2 = st.columns(2)
                    for col_w, part_fn, part_lbl in [(c1, np.real, "Re"),
                                                      (c2, np.imag, "Im")]:
                        fig = go.Figure()
                        fig.add_trace(go.Scattergl(
                            x=f_plot, y=part_fn(zarr[mask]), mode="lines",
                            line=dict(color="#1f77b4", width=2)))
                        fig.update_layout(
                            title=dict(text=f"{part_lbl}({zlabel})", font=dict(size=12)),
                            xaxis_title=tr("Frequency (GHz)", "頻率 (GHz)"),
                            yaxis_title=f"{part_lbl}({zlabel}) (Ω)",
                            plot_bgcolor="white", paper_bgcolor="white", height=220,
                            margin=dict(l=50, r=20, t=35, b=40), showlegend=False)
                        fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                        fig.update_yaxes(showgrid=True, gridcolor="#ebebeb")
                        plotly_with_dl(fig,
                                       key=f"pfp_z_{zlabel}_{part_lbl}_{model_short}_{fname}",
                                       filename=f"pfp_z_{zlabel}_{part_lbl}_{model_short}_{fname}",
                                       container=col_w)

                prev_range = (f_lo, f_hi)
                if g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # ── fbi_fit_group: Fbi vs ω², single fit-window slider ───────────────
            if group.get("fbi_fit_group"):
                for formula_type, formula_content in group.get("formulas", []):
                    if formula_type == "latex":
                        st.latex(formula_content)
                    else:
                        st.markdown(formula_content)

                sl_key = f"pfp_fbi_sl_{model_short}_{fname}"
                if sl_key not in st.session_state:
                    st.session_state[sl_key] = f_max_v
                f_hi_fbi = st.slider(
                    tr("Fbi linear fit upper frequency (GHz)",
                       "Fbi 線性擬合上限頻率 (GHz)"),
                    min_value=f_min_v, max_value=f_max_v,
                    value=float(st.session_state[sl_key]),
                    step=step_v, format="%.2f", key=sl_key)

                n_fit_new  = int(np.sum(f_ghz <= f_hi_fbi))
                n_fit_key  = f"pfp_nfit_{model_short}_{fname}"
                n_fit_prev = st.session_state.get(n_fit_key, n_fit_new)
                st.session_state[n_fit_key] = n_fit_new

                if n_fit_new != n_fit_prev and reextract_fn is not None:
                    params_out["_n_fit"] = n_fit_new
                    try:
                        _new_p, _new_a = reextract_fn(params_out, 1, live_arrays)
                        live_params.update(_new_p)
                        params_out.update(_new_p)
                        live_arrays.update(_new_a)
                    except Exception:
                        pass
                    params_out.pop("_n_fit", None)

                omega2  = live_arrays.get("omega2")
                Fbi_arr = live_arrays.get("Fbi")
                A0      = live_params.get("A0", 0.0)
                B0      = live_params.get("B0", 0.0)

                if Fbi_arr is not None and omega2 is not None:
                    valid_mask = (np.isfinite(Fbi_arr) & (np.abs(Fbi_arr) < 1e15)
                                  & (Fbi_arr > 0))
                    win_mask   = valid_mask & (f_ghz <= f_hi_fbi)
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=omega2[valid_mask], y=Fbi_arr[valid_mask], mode="markers",
                        name=tr("All data", "全部資料"), marker=dict(size=4, color="#aec7e8")))
                    fig.add_trace(go.Scattergl(
                        x=omega2[win_mask], y=Fbi_arr[win_mask], mode="markers",
                        name=tr("Fit window", "擬合視窗"), marker=dict(size=6, color="#1f77b4")))
                    if A0 > 1e-30 and win_mask.any():
                        xf = np.linspace(0, float(omega2[win_mask].max()) * 1.1, 200)
                        fig.add_trace(go.Scattergl(
                            x=xf, y=A0 + B0 * xf, mode="lines",
                            name=f"{tr('Fit', '擬合')}  A₀={A0:.3e}  B₀={B0:.3e}",
                            line=dict(color="#d62728", dash="dash", width=2)))
                    fig.update_layout(
                        title=dict(text="Fbi vs ω²  [Eq. 8]", font=dict(size=12)),
                        xaxis_title="ω² (rad²/s²)", yaxis_title="Fbi (rad/s)",
                        plot_bgcolor="white", paper_bgcolor="white", height=300,
                        margin=dict(l=55, r=10, t=40, b=42), showlegend=True,
                        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                                    font=dict(size=9)))
                    fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb")
                    plotly_with_dl(fig, key=f"pfp_fbi_{model_short}_{fname}",
                                   filename=f"pfp_fbi_{model_short}_{fname}")

                    Tbi_fit = float(np.sqrt(max(B0 / A0, 0.0))) if A0 > 1e-30 else 0.0
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("A₀", f"{A0:.4e}", help=tr("Intercept of Fbi vs ω²", "Fbi 對 ω² 的截距"))
                    mc2.metric("B₀", f"{B0:.4e}", help=tr("Slope of Fbi vs ω²", "Fbi 對 ω² 的斜率"))
                    mc3.metric("Tbi = √(B₀/A₀)", f"{Tbi_fit*1e12:.4f} ps",
                               help=tr("Intrinsic base time constant from fit",
                                       "由擬合求得的本徵基極時間常數"))

                prev_range = (f_hi_fbi, f_hi_fbi)
                if g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # ── f1_fit_group: F1 vs ω² fit + Tbe number_input ───────────────────
            if group.get("f1_fit_group"):
                if g_deps:
                    st.caption(f"{tr('Depends on', '依存於')}: {', '.join(g_deps)}")
                for formula_type, formula_content in group.get("formulas", []):
                    if formula_type == "latex":
                        st.latex(formula_content)
                    else:
                        st.markdown(formula_content)

                sl_key = f"pfp_f1_sl_{model_short}_{fname}"
                if sl_key not in st.session_state:
                    st.session_state[sl_key] = f_max_v
                f_hi_f1 = st.slider(
                    tr("F1 linear fit upper frequency (GHz)",
                       "F1 線性擬合上限頻率 (GHz)"),
                    min_value=f_min_v, max_value=f_max_v,
                    value=float(st.session_state[sl_key]),
                    step=step_v, format="%.2f", key=sl_key)

                n_fit_f1_new = int(np.sum(f_ghz <= f_hi_f1))
                nf1_key      = f"pfp_nfit_f1_{model_short}_{fname}"
                nf1_prev     = st.session_state.get(nf1_key, n_fit_f1_new)
                st.session_state[nf1_key] = n_fit_f1_new

                if n_fit_f1_new != nf1_prev and reextract_fn is not None:
                    params_out["_n_fit_f1"] = n_fit_f1_new
                    try:
                        _new_p, _new_a = reextract_fn(params_out, 4, live_arrays)
                        live_params.update(_new_p)
                        params_out.update(_new_p)
                        live_arrays.update(_new_a)
                    except Exception:
                        pass
                    params_out.pop("_n_fit_f1", None)

                omega2 = live_arrays.get("omega2")
                F1_arr = live_arrays.get("F1")
                A1     = live_params.get("A1", 0.0)
                B1     = live_params.get("B1", 0.0)

                if F1_arr is not None and omega2 is not None:
                    valid_mask = (np.isfinite(F1_arr) & (np.abs(F1_arr) < 1e15)
                                  & (F1_arr > 0))
                    win_mask   = valid_mask & (f_ghz <= f_hi_f1)
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=omega2[valid_mask], y=F1_arr[valid_mask], mode="markers",
                        name=tr("All data", "全部資料"), marker=dict(size=4, color="#aec7e8")))
                    fig.add_trace(go.Scattergl(
                        x=omega2[win_mask], y=F1_arr[win_mask], mode="markers",
                        name=tr("Fit window", "擬合視窗"), marker=dict(size=6, color="#1f77b4")))
                    if A1 > 1e-30 and win_mask.any():
                        xf = np.linspace(0, float(omega2[win_mask].max()) * 1.1, 200)
                        fig.add_trace(go.Scattergl(
                            x=xf, y=A1 + B1 * xf, mode="lines",
                            name=f"{tr('Fit', '擬合')}  A={A1:.3e}  B={B1:.3e}",
                            line=dict(color="#d62728", dash="dash", width=2)))
                    fig.update_layout(
                        title=dict(text="F1 vs ω²  [Eq. 19]", font=dict(size=12)),
                        xaxis_title="ω² (rad²/s²)", yaxis_title="F1 (rad/s)",
                        plot_bgcolor="white", paper_bgcolor="white", height=300,
                        margin=dict(l=55, r=10, t=40, b=42), showlegend=True,
                        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                                    font=dict(size=9)))
                    fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb")
                    plotly_with_dl(fig, key=f"pfp_f1_{model_short}_{fname}",
                                   filename=f"pfp_f1_{model_short}_{fname}")

                    alpha   = 1.0 / A1 if A1 > 1e-30 else 0.0
                    Tbe_fit = float(np.sqrt(max(B1 / A1, 0.0))) if A1 > 1e-30 else 0.0
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("A", f"{A1:.4e}", help=tr("Intercept of F1 vs ω²", "F1 對 ω² 的截距"))
                    mc2.metric("B", f"{B1:.4e}", help=tr("Slope of F1 vs ω²", "F1 對 ω² 的斜率"))
                    mc3.metric("α = 1/A", f"{alpha:.4e}", help="α = R(T − Tbe)")
                    mc4.metric("Tbe = √(B/A)", f"{Tbe_fit*1e12:.4f} ps",
                               help=tr("Emitter time constant from fit",
                                       "由擬合求得的射極時間常數"))

                # Tbe number_input (user-overridable)
                _upstream_vals = tuple(
                    round(params_out.get(spec[1], 0.0) * 1e15)
                    for gi in range(g_idx)
                    for spec in param_groups[gi]["params"]
                )
                upstream_tag = str(hash(_upstream_vals) % (10 ** 9))
                rng_tag      = f"{f_hi_f1:.3f}"

                for arr_key, param_key, label, scale, unit in group.get("params", []):
                    if arr_key not in live_arrays:
                        continue
                    raw     = live_arrays[arr_key]
                    arr_num = np.abs(raw) if np.iscomplexobj(raw) else np.real(raw)
                    fin     = arr_num[np.isfinite(arr_num)]
                    auto_SI   = float(np.median(fin)) if len(fin) > 0 \
                                else float(params_out.get(param_key, 0.0))
                    auto_disp = auto_SI * scale
                    inp_key   = (f"pfp_inp_{model_short}_{param_key}_{fname}"
                                 f"_{rng_tag}_{upstream_tag}")
                    actual_val = st.number_input(
                        f"{label} ({unit})" if unit else label,
                        value=float(auto_disp), format="%.5g", key=inp_key)
                    params_out[param_key] = actual_val / scale

                # Re-extract downstream if Tbe changed
                if reextract_fn is not None:
                    _grp_pk  = {spec[1] for spec in group.get("params", [])}
                    _grp_ak  = {spec[0] for spec in group.get("params", [])}
                    _changed = any(
                        abs(params_out.get(pk, 0.0) - live_params.get(pk, 0.0))
                        > 1e-9 * (abs(live_params.get(pk, 0.0)) + 1e-30)
                        for pk in _grp_pk
                    )
                    if _changed:
                        try:
                            _new_p, _new_a = reextract_fn(params_out, g_idx, live_arrays)
                            _processed_pk = {
                                spec[1]
                                for gi in range(g_idx + 1)
                                for spec in param_groups[gi]["params"]
                            }
                            _processed_ak = {
                                spec[0]
                                for gi in range(g_idx + 1)
                                for spec in param_groups[gi]["params"]
                            }
                            for _k, _v in _new_p.items():
                                if _k not in _processed_pk:
                                    live_params[_k] = _v
                                    params_out[_k]  = _v
                            for _k, _v in _new_a.items():
                                if _k not in _processed_ak:
                                    live_arrays[_k] = _v
                        except Exception:
                            pass

                prev_range = (f_hi_f1, f_hi_f1)
                if g_idx < len(param_groups) - 1:
                    st.markdown("---")
                continue

            # Wrap the whole parameter group (title + slider + per-parameter
            # plots) in one large card, so each group (Cbex, Cbcx, intrinsic,
            # τB, τC, …) reads as distinct.  `border=False` here — the thick
            # outline is drawn by the scoped `st-key-pfp_groupbox_*` CSS above
            # (Streamlit's own border wrapper would otherwise add a second,
            # thin frame).  All of this group's top-level widgets render into
            # `box`; nested per-parameter plots inherit it via `box.columns()`.
            box = st.container(
                border=False,
                key=f"pfp_groupbox_{model_short}_{g_idx}_{fname}")
            box.markdown(f"**{g_label}**")
            if g_deps:
                box.caption(f"{tr('Depends on', '依存於')}: {', '.join(g_deps)}")

            slider_key = f"pfp_sl_{model_short}_{g_idx}_{fname}"

            # "Use previous range" button for dependent groups
            if g_deps:
                if box.button(tr("↩ Same range as previous group",
                                 "↩ 與上一組相同範圍"),
                             key=f"pfp_useprev_{model_short}_{g_idx}_{fname}"):
                    st.session_state[slider_key] = prev_range
                    st.rerun()

            # Render per-group formulas before the slider
            for formula_type, formula_content in group.get("formulas", []):
                if formula_type == "markdown":
                    box.markdown(formula_content)
                elif formula_type == "latex":
                    box.latex(formula_content)


            # Initialize slider — pre-seed session_state and DO NOT
            # pass `value=` alongside `key=`, otherwise Streamlit logs
            # `check_session_state_rules` warnings about a widget being
            # created with both a default value and a Session-State entry.
            # Also clamp any pre-existing value to the current [min, max]
            # bounds in case a previous file had a different freq range.
            if slider_key not in st.session_state:
                st.session_state[slider_key] = (f_min_v, f_max_v)
            else:
                _cur = st.session_state[slider_key]
                try:
                    _lo = max(f_min_v, min(f_max_v, float(_cur[0])))
                    _hi = max(f_min_v, min(f_max_v, float(_cur[1])))
                    if _lo > _hi:
                        _lo, _hi = f_min_v, f_max_v
                    st.session_state[slider_key] = (_lo, _hi)
                except (TypeError, ValueError, IndexError):
                    st.session_state[slider_key] = (f_min_v, f_max_v)

            f_lo, f_hi = box.slider(
                tr("Frequency range (GHz)", "頻率範圍 (GHz)"),
                min_value=f_min_v, max_value=f_max_v,
                step=step_v, format="%.2f",
                key=slider_key)

            mask   = (f_ghz >= f_lo) & (f_ghz <= f_hi)
            f_plot = f_ghz[mask]
            # Tag used in widget keys — changing it recreates inputs fresh on slider move
            rng_tag = f"{f_lo:.3f}_{f_hi:.3f}"

            valid_specs = [
                s for s in g_params
                if s[0] in live_arrays and isinstance(live_arrays[s[0]], np.ndarray)
            ]


            _is_cbex_sweep_group = (group.get("cbex_sweep_group")
                                    and cbex_sweep_fn is not None
                                    and "Cbex_arr" in live_arrays)
            _cbex_sweep_rendered = False

            for row_start in range(0, len(valid_specs), 2):
                row  = valid_specs[row_start:row_start + 2]
                cols = box.columns(2)
                for _col_outer, (arr_key, param_key, label, scale, unit) in zip(cols, row):
                    # Wrap each parameter's plot + input + quickset buttons in
                    # a bordered container so individual extracted parameters
                    # are visually distinct from each other within a group.
                    col_w = _col_outer.container(border=True)
                    raw_masked = live_arrays[arr_key][mask]
                    arr_plot   = (np.abs(raw_masked) if np.iscomplexobj(raw_masked)
                                  else np.real(raw_masked)) * scale

                    # Recompute median from current slider range
                    arr_num = np.abs(raw_masked) if np.iscomplexobj(raw_masked) else np.real(raw_masked)
                    fin     = arr_num[np.isfinite(arr_num)]
                    _use_first = param_key in group.get("use_first_params", set())
                    if _use_first:
                        auto_SI = float(fin[0]) if len(fin) > 0 else float(params.get(param_key, 0.0))
                    else:
                        auto_SI = float(np.median(fin)) if len(fin) > 0 else float(params.get(param_key, 0.0))

                    auto_disp = auto_SI * scale

                    # Pre-read session state so plot can use it before number_input renders
                    # Hash of all upstream groups' current values — changes when any
                    # upstream param is overridden, forcing downstream inputs to reset.
                    _upstream_vals = tuple(
                        round(params_out.get(spec[1], 0.0) * 1e15)
                        for gi in range(g_idx)
                        for spec in param_groups[gi]["params"]
                    )
                    upstream_tag = str(hash(_upstream_vals) % (10 ** 9))
                    inp_key   = f"pfp_inp_{model_short}_{param_key}_{fname}_{rng_tag}_{upstream_tag}"

                    # Transfer any pending override (e.g. from the Cbex
                    # sweep tool) into the widget key.  Must happen BEFORE
                    # the number_input is instantiated, or Streamlit raises
                    # "session_state ... cannot be modified after widget".
                    pending_key = f"pfp_pending_{model_short}_{param_key}_{fname}"
                    if pending_key in st.session_state:
                        st.session_state[inp_key] = st.session_state.pop(pending_key)
                    # Same for quickset-button writes
                    apply_pending(inp_key)

                    user_disp = float(st.session_state.get(inp_key, auto_disp))
                    user_SI   = user_disp / scale

                    ylabel = f"{label} ({unit})" if unit else label
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(
                        x=f_plot, y=arr_plot, mode="lines", name=label,
                        line=dict(color="#1f77b4", width=2)))
                    if np.isfinite(user_disp):
                        fig.add_hline(
                            y=user_disp,
                            line=dict(color="#d62728", width=1.8, dash="dash"),
                            annotation_text=f"{user_disp:.4g} {unit}",
                            annotation_position="right",
                            annotation_font=dict(size=9, color="#d62728"))
                    # Cold-section value (computed once — used for both the
                    # plot annotation and the "cold = …" quickset button).
                    _cold_disp = None
                    if cold_res is not None and cold_param_map is not None:
                        _ck = cold_param_map.get(param_key)
                        if _ck and _ck in cold_res:
                            _v = float(cold_res[_ck]) * scale
                            if np.isfinite(_v):
                                _cold_disp = _v
                    if _cold_disp is not None:
                        fig.add_hline(
                            y=_cold_disp,
                            line=dict(color="#2ca02c", width=1.5, dash="dot"),
                            annotation_text=f"{tr('Cold', '冷測')}: {_cold_disp:.4g} {unit}",
                            annotation_position="left",
                            annotation_font=dict(size=9, color="#2ca02c"))
                    fig.update_layout(
                        title=dict(text=label, font=dict(size=12)),
                        xaxis_title=tr("Frequency (GHz)", "頻率 (GHz)"), yaxis_title=ylabel,
                        plot_bgcolor="white", paper_bgcolor="white", height=240,
                        margin=dict(l=50, r=60, t=35, b=40),
                        showlegend=False, hovermode="x unified")

                    fig.update_xaxes(showgrid=True, gridcolor="#ebebeb")
                    _y_range = None
                    if np.isfinite(user_disp) and abs(user_disp) > 1e-30:
                        _v5  = 5.0 * abs(user_disp)
                        _fin = arr_plot[np.isfinite(arr_plot)]
                        if len(_fin) > 0 and (_fin.max() > _v5 or _fin.min() < -_v5):
                            _y_range = [-_v5, _v5]
                    fig.update_yaxes(showgrid=True, gridcolor="#ebebeb",
                                     **({"range": _y_range} if _y_range is not None else {}))

                    plotly_with_dl(fig,
                                   key=f"pfp_{model_short}_{arr_key}_{fname}",
                                   filename=f"pfp_{model_short}_{arr_key}_{fname}",
                                   container=col_w)

                    # Number input — key includes rng_tag so it resets to new median on slider move
                    actual_val = col_w.number_input(
                        f"{label} ({unit})" if unit else label,
                        value=float(auto_disp),
                        format="%.5g",
                        key=inp_key)

                    # Quickset buttons row beneath the input (Step 3 layout).
                    # cold_disp surfaces beside "default" when the parameter
                    # has a value extracted in the Cold-HBT section.
                    quickset_buttons(container=col_w,
                                      key_prefix=inp_key,
                                      target_key=inp_key,
                                      arr_disp=arr_plot,
                                      default_disp=auto_disp,
                                      cold_disp=_cold_disp,
                                      unit=unit,
                                      fmt="%.4g", layout="below")

                    # Extra quickset buttons sourced from elsewhere in the UI:
                    #   - τB / τC : v_c-method values published by the
                    #     tau_total_fit_group section (T-models, ≥2 files).
                    #   - Rbe     : Z-parameter method result (when run and
                    #     this DUT is part of the fit).
                    _extra_qs = []
                    if param_key in ("tauB", "tauC"):
                        _pub = st.session_state.get(
                            f"taut_pub_{param_key}_{model_short}_{fname}")
                        if _pub is not None and np.isfinite(_pub):
                            _extra_qs.append(("v_c", float(_pub) * scale))
                    elif param_key == "Rbe":
                        _rz_rbe = st.session_state.get(f"rz12_Rbe_{fname}")
                        if _rz_rbe is not None and np.isfinite(_rz_rbe) \
                                and abs(_rz_rbe) > 0:
                            _extra_qs.append(("Z-param", float(_rz_rbe) * scale))

                    if _extra_qs:
                        _bcols = col_w.columns(len(_extra_qs))
                        for _bc, (_lbl, _val) in zip(_bcols, _extra_qs):
                            _btn_text = (f"{_lbl} = {_val:.4g} {unit}"
                                         if unit else f"{_lbl} = {_val:.4g}")
                            if _bc.button(_btn_text,
                                          key=f"{inp_key}_qs_extra_{_lbl}",
                                          width="stretch",
                                          help=tr(f"Set {label} to the {_lbl} reference value",
                                                  f"將 {label} 設為 {_lbl} 參考值")):
                                st.session_state[inp_key + "_pending"] = float(_val)
                                st.rerun()

                    params_out[param_key] = actual_val / scale

                # Render the Cbex sweep tool in the unused 2nd column,
                # immediately beside the Cbex plot.
                if (_is_cbex_sweep_group
                        and not _cbex_sweep_rendered
                        and len(row) < 2):
                    with cols[1]:
                        _render_cbex_sweep_tool(
                            cbex_arr=live_arrays["Cbex_arr"],
                            freq=freq,
                            f_ghz=f_ghz,
                            f_min_v=f_min_v,
                            f_max_v=f_max_v,
                            cbex_scale=g_params[0][3],   # 1e15 (fF)
                            cbex_unit=g_params[0][4],    # "fF"
                            cbex_param_key=g_params[0][1],  # "Cbex"
                            model_short=model_short,
                            fname=fname,
                            g_idx=g_idx,
                            rng_tag=rng_tag,
                            cbex_sweep_fn=cbex_sweep_fn,
                            param_groups=param_groups,
                        )
                    _cbex_sweep_rendered = True

            # ── Re-extract downstream groups if any param in this group changed ──
            if reextract_fn is not None and g_idx < len(param_groups) - 1:
                _grp_param_keys = {spec[1] for spec in g_params}
                _grp_arr_keys   = {spec[0] for spec in g_params}
                _any_changed = any(
                    abs(params_out.get(pk, 0.0) - live_params.get(pk, 0.0))
                    > 1e-9 * (abs(live_params.get(pk, 0.0)) + 1e-30)
                    for pk in _grp_param_keys
                    if pk in params_out
                )
                if _any_changed:
                    try:
                        _new_p, _new_a = reextract_fn(params_out, g_idx, live_arrays)
                        # Only overwrite params/arrays from groups not yet processed.
                        # Protecting all groups 0..g_idx prevents a downstream
                        # re-extraction (e.g. tauB change) from clobbering user
                        # overrides set in earlier groups (e.g. Rbi, Rbe, …).
                        _processed_param_keys = {
                            spec[1]
                            for gi in range(g_idx + 1)
                            for spec in param_groups[gi]["params"]
                        }
                        _processed_arr_keys = {
                            spec[0]
                            for gi in range(g_idx + 1)
                            for spec in param_groups[gi]["params"]
                        }
                        for _k, _v in _new_p.items():
                            if _k not in _processed_param_keys:
                                live_params[_k] = _v
                                params_out[_k]  = _v
                        for _k, _v in _new_a.items():
                            if _k not in _processed_arr_keys:
                                live_arrays[_k] = _v
                    except Exception:
                        pass  # silently ignore re-extraction failures

            prev_range = (f_lo, f_hi)
            if g_idx < len(param_groups) - 1:
                st.markdown("---")

    return params_out
