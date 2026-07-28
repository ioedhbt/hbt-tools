"""
models/tuning/sweep.py — The Auto Tuning expander: UI-card renderers +
the ``render_tuning_expander`` orchestrator that assembles them.

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).

``render_tuning_expander`` renders a sequence of distinct UI "cards"
(compute-backend badge, Full Auto Tune, Semi-Auto Tune's sweep table,
Brute-force / Optimized / Prioritized / Minimize-deviation, Parameter
sensitivity, Results).  Each card is broken out below into its own
private ``_render_*`` helper that takes explicit parameters and returns
explicit values (button-click flags, settings, or ``param_rows``).

The three heavy numerical drivers this orchestrator calls —
``_run_one_sweep`` (brute-force / optimized / prioritized / minimize-
deviation), ``_run_nelder_mead`` (currently UI-disabled, kept wired up),
and ``_run_progressive`` (🪜 Full Auto Tune) — live in their own modules
(drivers_sweep.py / drivers_nelder_mead.py / drivers_progressive.py)
rather than here: each is 250-1250 lines of GPU/CPU batched-sweep
bookkeeping (chunked OOM retry, top-K device buffers, the Streamlit
Stop-button cancellation path, empirical VRAM calibration...).  They were
hoisted to module-level functions rather than kept as nested closures —
an AST free-variable scan showed each one's only real dependencies on
render_tuning_expander's scope were its own parameters (model_cls, all_p,
S_raw, freq, z0, param_rows, tuning_specs, topo_key, fname) plus the
module-level helpers already imported here, so passing that surface
explicitly was a safe, name-for-name mechanical hoist — not a rewrite.
"""
from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from io import BytesIO

from ...helpers import plotly_with_dl
from tools.common.i18n import tr

from ..residuals import _port_residuals, _port_residuals_batch
from ._cuda_env import _HAS_CUDA, _CUDA_VER
from .ranges import (_canonical_tune_key, tune_hard_limits,
                      informed_default_range, _detect_low_perf_device,
                      _PARASITIC_KEYS)
from .preview import _make_sweep_values, _fmt_eval_time
from .helpers import (_action_cols, _clamp_row_session, _row_entry,
                       _best_summary_md, _col_name)
from .drivers_sweep import _run_one_sweep
from .drivers_nelder_mead import _run_nelder_mead  # noqa: F401  (currently UI-disabled; see the commented dispatch branch below)
from .drivers_progressive import _run_progressive


def _render_compute_backend_badge(model_cls, fname):
    """Rust/CUDA-active detection + the "Compute backend: ..." badge.
    Returns (cpu_tag, cpu_help_suffix) — both consumed by the card
    helpers rendered later in render_tuning_expander."""
    # ── Compute backend badge — cached in session_state so the
    #    label stays stable across reruns.  Without the cache the
    #    badge sometimes flipped Rust→NumPy after a Stop/cancel
    #    rerun (re-import races on `hbt_rust_kernels` after the
    #    sweep loop unwinds); caching guarantees the badge reflects
    #    a one-time, definitive detection.
    _BADGE_KEY = f"_rust_active_cached_{model_cls.SHORT}_{fname}"

    def _detect_rust_active() -> bool:
        try:
            from ...helpers.rust_kernels import (
                HAS_RUST as _HR,
                _phase2_dispatch_enabled as _phase2_on,
                SIM_FOR_TOPOLOGY as _SIM_TOPO,
            )
            from ...helpers import rust_kernels as _RKMOD
        except Exception:                              # pragma: no cover
            return False
        if not (_HR and _phase2_on()):
            return False
        # Built-in fixed-topology end-to-end kernels.
        if _SIM_TOPO.get(model_cls.SHORT) is not None:
            return True
        # Custom model — the data-driven `sim_custom_batch` kernel isn't in
        # SIM_FOR_TOPOLOGY; it's active when the adapter opts in *and* the
        # compiled symbol is present in the loaded binary.
        if getattr(model_cls, "USES_RUST_BATCH", False):
            return getattr(getattr(_RKMOD, "_rk", None),
                           "sim_custom_batch", None) is not None
        return False

    # Sticky-True caching: re-evaluate every render, but never flip
    # an active=True back to False within the session.  This keeps
    # the original "no Rust→NumPy flicker after Stop" guarantee
    # *and* allows the badge to flip from stale-False → True the
    # moment the Rust binary becomes importable (e.g. user just
    # finished building it without restarting Streamlit).
    _now_active = _detect_rust_active()
    _prev = bool(st.session_state.get(_BADGE_KEY, False))
    st.session_state[_BADGE_KEY] = bool(_now_active or _prev)
    _rust_active_here = st.session_state[_BADGE_KEY]
    if _HAS_CUDA and _rust_active_here:
        _bk = ("<span style='background:#e3f2fd;color:#0d47a1;"
               "padding:2px 8px;border-radius:4px;font-size:0.8em;"
               "font-weight:600'"
               f" title='{tr('CUDA buttons use cupy; CPU buttons use the Rust kernel.',
                             'CUDA 按鈕使用 cupy；CPU 按鈕使用 Rust 核心。')}'"
               f">⚡ {tr('CUDA available', 'CUDA 可用')} + 🦀 {tr('Rust active', 'Rust 已啟用')}</span>")
    elif _HAS_CUDA:
        _bk = ("<span style='background:#e3f2fd;color:#0d47a1;"
               "padding:2px 8px;border-radius:4px;font-size:0.8em;"
               "font-weight:600'"
               f" title='{tr('CUDA buttons use cupy; CPU buttons use NumPy.'
                             ' Set HBT_USE_RUST_SIM_BATCH=1 to enable Rust on the'
                             ' CPU path for ~25x faster sweeps.',
                             'CUDA 按鈕使用 cupy；CPU 按鈕使用 NumPy。'
                             '設定 HBT_USE_RUST_SIM_BATCH=1 可在 CPU 路徑'
                             '啟用 Rust，掃描速度提升約 25 倍。')}'"
               f">⚡ {tr('CUDA available', 'CUDA 可用')}</span>")
    elif _rust_active_here:
        _bk = ("<span style='background:#fff3e0;color:#e65100;"
               "padding:2px 8px;border-radius:4px;font-size:0.8em;"
               "font-weight:600'"
               f" title='{tr('CPU buttons use the Rust kernel.',
                             'CPU 按鈕使用 Rust 核心。')}'"
               f">🦀 {tr('Rust active', 'Rust 已啟用')}</span>")
    else:
        _bk = ("<span style='background:#eceff1;color:#37474f;"
               "padding:2px 8px;border-radius:4px;font-size:0.8em;"
               "font-weight:600'"
               f" title='{tr('CPU buttons use NumPy. Build the Rust crate'
                             ' (python dev/build_rust_kernels.py) and set'
                             ' HBT_USE_RUST_SIM_BATCH=1 for ~25x speedup.',
                             'CPU 按鈕使用 NumPy。建置 Rust crate'
                             '（python dev/build_rust_kernels.py）'
                             '並設定 HBT_USE_RUST_SIM_BATCH=1 可提升約 25 倍速度。')}'"
               ">🐢 NumPy</span>")
    st.markdown(f"{tr('Compute backend', '運算後端')}: {_bk}", unsafe_allow_html=True)

    # ── Mode-card shared bits — used by the Full Auto Tune card here
    #    AND the Semi-Auto cards further below, so they live above both.
    _cpu_tag = "🦀 Rust" if _rust_active_here else "🐢 NumPy"
    # The backend (Rust vs NumPy) is no longer shown on the CPU button
    # face — it's surfaced in the hover tooltip instead.
    _cpu_help_suffix = f"\n\n{tr('CPU backend', 'CPU 後端')}: {_cpu_tag}."
    return _cpu_tag, _cpu_help_suffix


def _render_full_auto_card(tuning_specs, default_fit_keys, topo_key, fname,
                           cpu_help_suffix):
    """The "🪜 Full Auto Tune (recommended)" card: parameter multiselect +
    Evaluate-with-CPU/CUDA buttons.  Returns
    (prog_fit_keys, prog_cpu_clicked, prog_cuda_clicked)."""
    # ── 🪜 Full Auto Tune (recommended) — first card under the badge ──
    # Global coarse scan from physics-informed ranges, then pair-wise
    # (Zbe / Zbc / …) refinement with shrinking steps.  Never
    # re-evaluates a combo; keeps the best result on Stop.
    _prog_label_map = {}
    for _s in tuning_specs:
        _u = _s[3] if len(_s) > 3 else ""
        _prog_label_map[_s[0]] = f"{_s[1]} ({_u})" if _u else _s[1]
    _prog_opt_keys = [s[0] for s in tuning_specs]

    # Default selection: caller-supplied keys, else all non-parasitic
    # (canonical) keys.  Intersect with the actual option keys so
    # st.multiselect never sees an unknown default.
    if default_fit_keys is None:
        _prog_default = [
            s[0] for s in tuning_specs
            if _canonical_tune_key(s[0], s[1]) not in _PARASITIC_KEYS]
    else:
        _prog_default = list(default_fit_keys)
    _prog_default = [k for k in _prog_default if k in _prog_opt_keys]

    with st.container(border=True):
        st.markdown(
            f"**🪜 {tr('Full Auto Tune', '全自動調諧')}** "
            f"<span style='color:#666;font-size:0.85em'>({tr('recommended', '建議')})</span>"
            f" <span class='hbt-help' title='{tr(
                'Coarse-to-fine: global scan'
                ' from physics-informed ranges, then pair-wise refinement with'
                ' shrinking steps. Best result is kept when you press Stop.'
                ' Deselect a parameter to pin it at its current value'
                ' (including 0).',
                '由粗到細：先依物理範圍進行全域掃描，'
                '再以逐漸縮小的步進進行成對細化。'
                '按下停止時會保留最佳結果。'
                '取消勾選某參數會將其固定在目前值（含 0）。')}'>?</span>",
            unsafe_allow_html=True)
        _prog_scope_key = f"tune_prog_scope_{topo_key}_{fname}"
        # Once the widget owns its state, use that as the default (matches
        # the slider-preview multiselect pattern) so Streamlit doesn't warn
        # about a keyed widget also carrying a fixed default.
        _prog_seed = st.session_state.get(_prog_scope_key, _prog_default)
        _prog_seed = [k for k in _prog_seed if k in _prog_opt_keys]
        prog_fit_keys = st.multiselect(
            tr("Parameters to fit", "要擬合的參數"),
            options=_prog_opt_keys,
            default=_prog_seed,
            format_func=lambda k: _prog_label_map.get(k, k),
            key=_prog_scope_key,
            placeholder=tr("Choose options", "請選擇項目"))
        _prog_help = tr(
            "Coarse global scan from the physics-informed ranges, then "
            "pair-wise refinement with shrinking steps. Memoises every "
            "combo so no work is repeated; keeps the best result on "
            "Stop. Fitted parameters are floored above 0 — deselect one "
            "to pin it at its current value instead.",
            "先依物理範圍進行粗略全域掃描，再以逐漸縮小的步進進行成對細化。"
            "每個組合都會被記錄，不會重複計算；按下停止時保留最佳結果。"
            "擬合參數下限為 0 — 取消勾選可改為固定在目前值。")
        _c_cpu, _c_cuda = _action_cols()
        prog_cpu_clicked = _c_cpu.button(
            tr("Evaluate with CPU", "以 CPU 評估"),
            key=f"tune_calc_prog_{topo_key}_{fname}",
            help=_prog_help + cpu_help_suffix,
            width="stretch")
        prog_cuda_clicked = (
            _HAS_CUDA
            and _c_cuda.button(
                tr("⚡ Evaluate with CUDA", "⚡ 以 CUDA 評估"),
                key=f"tune_calc_prog_cuda_{topo_key}_{fname}",
                help=_prog_help,
                width="stretch"))
    return prog_fit_keys, prog_cpu_clicked, prog_cuda_clicked


def _render_semi_auto_table(tuning_specs, all_p, topo_key, fname, low_perf):
    """Semi-Auto Tune sweep-grid setup: toolbar (defaults / select-all /
    de-select) + the per-parameter Sweep/Min/Step/Max/#Calc table.
    Returns param_rows — the single source of truth every driver and
    the sensitivity section reads."""
    param_rows = []
    # ── Toolbar: Use default values + Select all + De-select ──
    # All three buttons live above the table so users hit the
    # bulk actions before scanning per-row.  Each writes to
    # session_state and reruns so the table picks up the new
    # values on the next render pass.
    _tb1, _tb2, _tb3, _ = st.columns([1.0, 1.0, 1.0, 3.0])
    if _tb1.button(tr("↩️ Use default values", "↩️ 使用預設值"),
                    key=f"tune_defaults_{topo_key}_{fname}"):
        for spec in tuning_specs:
            key, label, scale = spec[0], spec[1], spec[2]
            spec_step = spec[5] if len(spec) > 5 else None
            current_disp = float(all_p.get(key, 0.0)) * scale
            kp = f"tune_{topo_key}_{key}_{fname}"
            d_min, d_step, d_max = informed_default_range(
                key, label, current_disp,
                low_perf=low_perf, spec_step=spec_step)
            st.session_state[f"{kp}_min"] = d_min
            st.session_state[f"{kp}_step"] = d_step
            st.session_state[f"{kp}_max"] = d_max
        st.rerun()
    if _tb2.button(tr("✅ Select all", "✅ 全選"),
                    key=f"tune_select_all_{topo_key}_{fname}"):
        for spec in tuning_specs:
            key = spec[0]
            st.session_state[
                f"tune_{topo_key}_{key}_{fname}_chk"] = True
        st.rerun()
    if _tb3.button(tr("❌ De-select all", "❌ 全部取消"),
                    key=f"tune_deselect_all_{topo_key}_{fname}"):
        for spec in tuning_specs:
            key = spec[0]
            st.session_state[
                f"tune_{topo_key}_{key}_{fname}_chk"] = False
        st.rerun()

    # Column headers
    hdr = st.columns([0.5, 1.5, 1.2, 1.2, 1.2, 1.0])
    hdr[0].markdown(f"**{tr('Sweep', '掃描')}**")
    hdr[1].markdown(f"**{tr('Parameter', '參數')}**")
    hdr[2].markdown(f"**{tr('Min', '最小值')}**")
    hdr[3].markdown(f"**{tr('Step', '步進')}**")
    hdr[4].markdown(f"**{tr('Max', '最大值')}**")
    hdr[5].markdown(f"**{tr('# Calc', '計算數')}**")

    # Per-parameter rows — widget path
    for spec in tuning_specs:
        key, label, scale = spec[0], spec[1], spec[2]
        unit = spec[3] if len(spec) > 3 else ""
        current_disp = float(all_p.get(key, 0.0)) * scale
        kp = f"tune_{topo_key}_{key}_{fname}"

        cols = st.columns([0.5, 1.5, 1.2, 1.2, 1.2, 1.0])
        enabled = cols[0].checkbox(
            f"sweep_{key}", value=False, key=f"{kp}_chk",
            label_visibility="collapsed")
        cur_str = f"{current_disp:.2f} {unit}".strip()
        cols[1].markdown(
            f"**{label}** ({unit}) ({cur_str})" if unit
            else f"**{label}** ({cur_str})")

        if enabled:
            # Hard physical limits: floor Min/Max at (h_lo,
            # h_hi) — pads can't go negative, alpha0 is boxed
            # to [0.95, 0.99], etc.
            h_lo, h_hi = tune_hard_limits(key, label)
            h_lo = float(h_lo)
            _clamp_row_session(kp, h_lo, h_hi)
            _min_kwargs = {"min_value": h_lo}
            _max_kwargs = {"min_value": h_lo}
            if h_hi is not None:
                _min_kwargs["max_value"] = float(h_hi)
                _max_kwargs["max_value"] = float(h_hi)
            min_val = cols[2].number_input(
                tr("Min", "最小值"), format="%.5g", key=f"{kp}_min",
                label_visibility="collapsed", **_min_kwargs)
            step_val = cols[3].number_input(
                tr("Step", "步進"), format="%.5g", key=f"{kp}_step",
                min_value=0.0, label_visibility="collapsed")
            max_val = cols[4].number_input(
                tr("Max", "最大值"), format="%.5g", key=f"{kp}_max",
                label_visibility="collapsed", **_max_kwargs)
        else:
            min_val = current_disp
            step_val = 0.0
            max_val = current_disp
            cols[2].markdown(f"{current_disp:.2f}")
            cols[3].markdown("0")
            cols[4].markdown(f"{current_disp:.2f}")

        row = _row_entry(spec, enabled,
                         min_val, step_val, max_val, all_p)
        cols[5].markdown(f"**{row['n_calc']}**")
        param_rows.append(row)

    # Total calculation count
    total_calcs = 1
    for row in param_rows:
        if row["enabled"]:
            total_calcs *= row["n_calc"]
        # unchecked params contribute 1 (single current value)

    st.markdown(f"**{tr('Total calculations', '總計算數')}: {total_calcs:,}**")

    if total_calcs > 500_000:
        st.warning(tr("More than 500,000 combinations — this may "
                      "take a long time.",
                      "超過 500,000 種組合 — 可能需要很長時間。"))

    # ── CUDA availability note ──────────────────────────────
    if _HAS_CUDA:
        st.caption(tr(f"CUDA {_CUDA_VER} detected — GPU "
                      "acceleration available",
                      f"偵測到 CUDA {_CUDA_VER} — GPU 加速可用"))

    return param_rows


def _render_bruteforce_card(topo_key, fname, cpu_help_suffix):
    """The "🧮 Brute force — all combos" card. Returns (cpu_clicked, cuda_clicked)."""
    # ── 🧮 Brute-force all combos ───────────────────────────────────
    _calc_all_help = tr(
        "Evaluates EVERY combination in the sweep grid above and ranks "
        "the top-100 results by lowest TOTAL residual.",
        "評估上方掃描表格中的每一種組合，並依最低總殘差排出前 100 名。")
    with st.container(border=True):
        st.markdown(
            f"**🧮 {tr('Brute force — all combos', '暴力法 — 所有組合')}**"
            f" <span class='hbt-help' title='{tr(
                'Evaluate every'
                ' combination in the grid, rank by total residual.',
                '評估表格中每一種組合，並依總殘差排序。')}'"
            ">?</span>",
            unsafe_allow_html=True)
        _c_cpu, _c_cuda = _action_cols()
        cpu_clicked = _c_cpu.button(
            tr("Brute force with CPU", "以 CPU 執行暴力法"),
            key=f"tune_calc_{topo_key}_{fname}",
            help=_calc_all_help + cpu_help_suffix,
            width="stretch")
        cuda_clicked = (
            _HAS_CUDA
            and _c_cuda.button(
                tr("⚡ Brute force with CUDA", "⚡ 以 CUDA 執行暴力法"),
                key=f"tune_calc_cuda_{topo_key}_{fname}",
                help=_calc_all_help,
                width="stretch"))
    return cpu_clicked, cuda_clicked


def _render_optimized_card(topo_key, fname, cpu_help_suffix):
    """The "🎯 Optimized — recursive bisection" card. Returns
    (opt_cpu_clicked, opt_cuda_clicked)."""
    # ── 🎯 Optimized recursive bisection ────────────────────────────
    _opt_help = tr(
        "Repeatedly subsamples 5 evenly-spaced values per swept parameter "
        "(e.g. min=1, step=1, max=100 → 1, 25, 50, 75, 100), picks the 2 "
        "combos with the lowest total residual, then narrows the search "
        "box to those two values and recurses.  Iteration stops once a "
        "brute-force sweep at the user's chosen step would fit ≤ 5,000,000 "
        "combos, and that final refinement is run as the closing pass.",
        "對每個掃描參數重複取 5 個等間距值進行子取樣"
        "（例如 min=1, step=1, max=100 → 1, 25, 50, 75, 100），"
        "挑出總殘差最低的 2 個組合，將搜尋範圍縮小至該兩值後遞迴。"
        "當以使用者所選步進進行暴力掃描的組合數 ≤ 5,000,000 時停止疊代，"
        "並以該次細化作為最終回合。")
    with st.container(border=True):
        st.markdown(
            f"**🎯 {tr('Optimized — recursive bisection', '最佳化 — 遞迴二分法')}**"
            f" <span class='hbt-help' title='{tr(
                'Iteratively narrows the'
                ' search box: 5-point subsample, keep top-2, recurse'
                ' until the final pass fits at most 5 million combos.',
                '逐步縮小搜尋範圍：5 點子取樣，保留前 2 名，'
                '遞迴直到最終回合組合數不超過 500 萬。')}'"
            ">?</span>",
            unsafe_allow_html=True)
        _c_cpu, _c_cuda = _action_cols()
        opt_cpu_clicked = _c_cpu.button(
            tr("Optimized with CPU", "以 CPU 執行最佳化"),
            key=f"tune_calc_opt_{topo_key}_{fname}",
            help=_opt_help + cpu_help_suffix,
            width="stretch")
        opt_cuda_clicked = (
            _HAS_CUDA
            and _c_cuda.button(
                tr("⚡ Optimized with CUDA", "⚡ 以 CUDA 執行最佳化"),
                key=f"tune_calc_opt_cuda_{topo_key}_{fname}",
                help=_opt_help,
                width="stretch"))
    return opt_cpu_clicked, opt_cuda_clicked


def _render_prioritized_card(topo_key, fname, cpu_help_suffix):
    """The "🥇 Prioritized — rank by one S-parameter" card. Returns
    (prio_cpu_clicked, prio_cuda_clicked, prio_metric)."""
    # ── 🥇 Prioritized by single S-parameter ────────────────────────
    _prio_help = tr(
        "Runs the same full-grid sweep but ranks results by the residual "
        "of a single chosen S-parameter instead of the total.",
        "執行相同的全網格掃描，但改依單一選定 S 參數的殘差排序，"
        "而非總殘差。")
    with st.container(border=True):
        st.markdown(
            f"**🥇 {tr('Prioritized — rank by one S-parameter', '優先排序 — 依單一 S 參數排序')}**"
            f" <span class='hbt-help' title='{tr(
                'Same full-grid sweep,'
                ' but the top-100 selection is sorted by a single'
                ' S-parameter residual rather than the total.',
                '相同的全網格掃描，但前 100 名的排序改依'
                '單一 S 參數殘差，而非總殘差。')}'"
            ">?</span>",
            unsafe_allow_html=True)
        _prio_row = st.columns([0.6, 2])
        _prio_row[0].markdown(
            f"<div style='padding-top:0.4em'>"
            f"<small><b>{tr('Sort by', '排序依據')}</b></small></div>",
            unsafe_allow_html=True)
        prio_metric = _prio_row[1].radio(
            tr("Prioritize sort metric", "優先排序指標"),
            ["S11", "S12", "S21", "S22"],
            horizontal=True,
            label_visibility="collapsed",
            key=f"tune_prio_metric_{topo_key}_{fname}")
        _c_cpu, _c_cuda = _action_cols()
        prio_cpu_clicked = _c_cpu.button(
            tr("Prioritized with CPU", "以 CPU 執行優先排序"),
            key=f"tune_calc_prio_{topo_key}_{fname}",
            help=_prio_help + cpu_help_suffix,
            width="stretch")
        prio_cuda_clicked = (
            _HAS_CUDA
            and _c_cuda.button(
                tr("⚡ Prioritized with CUDA", "⚡ 以 CUDA 執行優先排序"),
                key=f"tune_calc_prio_cuda_{topo_key}_{fname}",
                help=_prio_help,
                width="stretch"))
    return prio_cpu_clicked, prio_cuda_clicked, prio_metric


def _render_min_dev_card(topo_key, fname, cpu_tag):
    """The hidden "🎚️ Minimize deviation" card (behind _SHOW_MIN_DEV in
    render_tuning_expander — kept wired up, not currently reachable from
    the UI, exactly as in the original file).  Returns
    (bal_cpu_clicked, bal_cuda_clicked, bal_dev_threshold, bal_use_res,
    bal_res_threshold)."""
    _bal_help = tr(
        "Runs the full-grid sweep but discards combos whose per-port "
        "residuals are unbalanced (peak-to-peak spread > threshold).  "
        "Surviving balanced combos are then ranked by lowest total "
        "residual.  Optionally also drop combos where any single port "
        "residual exceeds a quality floor.",
        "執行全網格掃描，但捨棄各埠殘差不平衡的組合"
        "（峰對峰差 > 閾值）。倖存的平衡組合再依最低總殘差排序。"
        "亦可選擇捨棄任一埠殘差超過品質下限的組合。")
    with st.container(border=True):
        st.markdown(
            f"**🎚️ {tr('Minimize deviation — peak-to-peak filter', '最小化偏差 — 峰對峰篩選')}**  "
            f"<span style='color:#666;font-size:0.85em'>"
            f"{tr('Discard unbalanced combos, then rank survivors by '
                 'total residual.',
                 '捨棄不平衡的組合，再依總殘差排序倖存者。')}</span>",
            unsafe_allow_html=True)
        _bal_inputs = st.columns([1, 1, 2])
        bal_dev_threshold = _bal_inputs[0].number_input(
            tr("Max per-port deviation (%)", "每埠最大偏差 (%)"),
            min_value=0.0, max_value=100.0,
            value=float(st.session_state.get(
                f"tune_bal_dev_{topo_key}_{fname}", 3.0)),
            step=0.5, format="%.2f",
            key=f"tune_bal_dev_{topo_key}_{fname}",
            help=tr("Max allowed |max(S11,S12,S21,S22) − min(...)| residual "
                    "spread (in %).  Smaller = more balanced.",
                    "允許的最大 |max(S11,S12,S21,S22) − min(...)| 殘差"
                    "分佈（%）。數值越小越平衡。"))
        _bal_use_res = _bal_inputs[1].checkbox(
            tr("Use residual cap", "使用殘差上限"),
            value=False,
            key=f"tune_bal_use_res_{topo_key}_{fname}")
        bal_res_threshold = _bal_inputs[2].number_input(
            tr("Max per-port residual (%)", "每埠最大殘差 (%)"),
            min_value=0.0, max_value=100.0,
            value=float(st.session_state.get(
                f"tune_bal_res_{topo_key}_{fname}", 5.0)),
            step=0.5, format="%.2f",
            key=f"tune_bal_res_{topo_key}_{fname}",
            disabled=(not _bal_use_res),
            help=tr("Quality floor — drop combos where any port's residual "
                    "exceeds this.",
                    "品質下限 — 捨棄任一埠殘差超過此值的組合。"))
        _c_cpu, _c_cuda = _action_cols()
        bal_cpu_clicked = _c_cpu.button(
            f"CPU — {cpu_tag}",
            key=f"tune_calc_bal_{topo_key}_{fname}",
            help=_bal_help,
            width="stretch")
        bal_cuda_clicked = (
            _HAS_CUDA
            and _c_cuda.button(
                "⚡ CUDA",
                key=f"tune_calc_bal_cuda_{topo_key}_{fname}",
                help=_bal_help,
                width="stretch"))
    return (bal_cpu_clicked, bal_cuda_clicked, bal_dev_threshold,
            _bal_use_res, bal_res_threshold)


def _render_semi_auto_closed_rows(tuning_specs, all_p, topo_key, fname):
    """Semi-Auto Tune collapsed: read back the same keyed session state
    the (unrendered) table widgets would return, so param_rows is
    identical whether or not the section is open."""
    param_rows = []
    # Semi-Auto collapsed — the row widgets aren't rendered, but every
    # driver (Full Auto included) still needs param_rows.  Read the
    # same keyed session state the widgets would return; _row_entry is
    # the shared source of truth for sweep/n_calc, so the two paths
    # cannot diverge.  (The pre-init seeding loop above guarantees the
    # _chk/_min/_step/_max keys exist.)
    for spec in tuning_specs:
        key, label, scale = spec[0], spec[1], spec[2]
        kp = f"tune_{topo_key}_{key}_{fname}"
        h_lo, h_hi = tune_hard_limits(key, label)
        _clamp_row_session(kp, float(h_lo), h_hi)
        enabled = bool(st.session_state.get(f"{kp}_chk", False))
        current_disp = float(all_p.get(key, 0.0)) * scale
        if enabled:
            min_val  = float(st.session_state.get(
                f"{kp}_min", current_disp))
            step_val = float(st.session_state.get(f"{kp}_step", 0.0))
            max_val  = float(st.session_state.get(
                f"{kp}_max", current_disp))
        else:
            min_val, step_val, max_val = (current_disp, 0.0,
                                          current_disp)
        param_rows.append(
            _row_entry(spec, enabled, min_val, step_val, max_val, all_p))

    return param_rows
def _render_sensitivity_section(model_cls, all_p, S_raw, freq, z0,
                                tuning_specs, topo_key, fname):
    """The "📈 Parameter sensitivity" toggle-collapsed section: per-
    parameter residual curves swept around the best result."""
    # ── 📈 Parameter sensitivity (toggle-collapsed section) ────────
    # Same toggle-not-expander workaround as Semi-Auto Tune (Streamlit
    # forbids nested expanders).  Reads the persisted results df, so a
    # fresh run's charts appear on the rerun after it finishes.
    sens_open = st.toggle(
        tr("📈 Parameter sensitivity", "📈 參數敏感度"),
        key=f"tune_sens_open_{topo_key}_{fname}",
        help=tr(
            "Per-parameter residual curves swept around the best "
            "result, other parameters held at their best values.  "
            "Curves are plotted for parameters with a ticked Sweep "
            "checkbox in Semi-Auto Tune.",
            "在最佳結果附近，逐一掃描每個參數繪出殘差曲線，"
            "其餘參數固定在其最佳值。僅繪製半自動調諧中"
            "已勾選「掃描」的參數。"))
    if sens_open:
        with st.container(border=True):
            df_sens = st.session_state.get(f"tune_df_{topo_key}_{fname}")
            ticked_specs = [
                spec for spec in tuning_specs
                if st.session_state.get(
                    f"tune_{topo_key}_{spec[0]}_{fname}_chk", False)
            ]
            if df_sens is None:
                st.caption(tr(
                    "Run a tune first — the sensitivity plots "
                    "sweep each parameter around the best result.",
                    "請先執行一次調諧 — 敏感度圖會在最佳結果附近"
                    "掃描每個參數。"))
            elif not ticked_specs:
                st.caption(tr(
                    "Tick at least one **Sweep** checkbox in "
                    "Semi-Auto Tune to choose which parameters "
                    "to plot.",
                    "請在半自動調諧中至少勾選一個**掃描**核取方塊，"
                    "以選擇要繪製的參數。"))
            else:
                best = df_sens.iloc[0]
                st.markdown(tr(
                    "**Parameter sensitivity** *(other params held at best values)*",
                    "**參數敏感度** *（其餘參數固定於最佳值）*"))
                # Build baseline param dict from best row (in SI).
                # Missing column (stale parameter-less table) → keep
                # the all_p value.
                best_p = dict(all_p)
                for spec in tuning_specs:
                    key, label, scale = spec[0], spec[1], spec[2]
                    unit = spec[3] if len(spec) > 3 else ""
                    col_name = f"{label} ({unit})" if unit else label
                    if col_name in best.index:
                        best_p[key] = float(best[col_name]) / scale

                _sp_colors = {"S11": "#1f77b4", "S12": "#d62728",
                              "S21": "#2ca02c", "S22": "#ff7f0e"}

                # Lay sensitivity charts out in a 2-column grid.  Even
                # index → left column, odd index → right column.  An
                # odd total leaves the final chart alone in the left
                # column (right column stays empty).
                _sens_col_pair = None  # current (left_col, right_col)
                _sens_rendered = 0

                for spec in ticked_specs:
                    key, label, scale = spec[0], spec[1], spec[2]
                    unit = spec[3] if len(spec) > 3 else ""
                    kp = f"tune_{topo_key}_{key}_{fname}"
                    sweep_min = float(st.session_state.get(f"{kp}_min", 0))
                    sweep_step = float(st.session_state.get(f"{kp}_step", 0))
                    sweep_max = float(st.session_state.get(f"{kp}_max", 0))
                    sweep_vals = _make_sweep_values(sweep_min, sweep_max, sweep_step)

                    # Cap the sensitivity chart at 50 points — when the
                    # user-defined sweep is denser, drop to 50 evenly-spaced
                    # samples (always keeping the endpoints).
                    SENS_MAX_PTS = 50
                    if len(sweep_vals) > SENS_MAX_PTS:
                        idxs = np.linspace(0, len(sweep_vals) - 1,
                                            SENS_MAX_PTS).round().astype(int)
                        idxs = np.unique(idxs)
                        sweep_vals = sweep_vals[idxs]

                    if len(sweep_vals) < 2:
                        continue  # nothing to plot for a single point

                    # Residuals for the whole sweep in ONE batched call (the
                    # swept param as a length-M array) instead of M sequential
                    # simulate_vec calls — keeps the sensitivity panel cheap even
                    # when many parameters are ticked (matters most for the custom
                    # model, whose per-call path crosses the Rust/PyO3 boundary).
                    res_s11 = res_s12 = res_s21 = res_s22 = None
                    if hasattr(model_cls, "simulate_batch"):
                        try:
                            p_b = dict(best_p)
                            p_b[key] = np.asarray(sweep_vals, dtype=float) / scale
                            S_b = np.asarray(
                                model_cls.simulate_batch(p_b, freq, z0, xp=np)
                            ).reshape(len(sweep_vals), len(freq), 2, 2)
                            rb = _port_residuals_batch(S_raw, S_b, np)
                            res_s11 = np.asarray(rb["S11"], dtype=float)
                            res_s12 = np.asarray(rb["S12"], dtype=float)
                            res_s21 = np.asarray(rb["S21"], dtype=float)
                            res_s22 = np.asarray(rb["S22"], dtype=float)
                            for _a in (res_s11, res_s12, res_s21, res_s22):
                                _a[~np.isfinite(_a)] = float("inf")
                        except Exception:
                            res_s11 = None       # fall back to the per-point loop

                    if res_s11 is None:
                        res_s11 = np.zeros(len(sweep_vals))
                        res_s12 = np.zeros(len(sweep_vals))
                        res_s21 = np.zeros(len(sweep_vals))
                        res_s22 = np.zeros(len(sweep_vals))
                        _sens_has_vec = hasattr(model_cls, "simulate_vec")
                        for vi, sv in enumerate(sweep_vals):
                            p = dict(best_p)
                            p[key] = sv / scale  # convert display -> SI
                            try:
                                if _sens_has_vec:
                                    S_sim = model_cls.simulate_vec(p, freq, z0, xp=np)
                                else:
                                    S_sim = model_cls.simulate(p, freq, z0)
                            except Exception:
                                S_sim = None
                            if S_sim is None:
                                res_s11[vi] = res_s12[vi] = float("inf")
                                res_s21[vi] = res_s22[vi] = float("inf")
                            else:
                                r = _port_residuals(S_raw, S_sim)
                                res_s11[vi] = r["S11"]
                                res_s12[vi] = r["S12"]
                                res_s21[vi] = r["S21"]
                                res_s22[vi] = r["S22"]

                    x_label = f"{label} ({unit})" if unit else label
                    fig = go.Figure()
                    for sp_name, y_data in [("S11", res_s11), ("S12", res_s12),
                                            ("S21", res_s21), ("S22", res_s22)]:
                        fig.add_trace(go.Scattergl(
                            x=sweep_vals, y=y_data, mode="lines+markers",
                            name=sp_name,
                            line=dict(color=_sp_colors[sp_name], width=2),
                            marker=dict(size=4),
                        ))
                    fig.update_layout(
                        title=tr(f"Sensitivity -- {x_label}",
                                 f"敏感度 -- {x_label}"),
                        xaxis_title=x_label,
                        yaxis_title=tr("Residual (%)", "殘差 (%)"),
                        height=350,
                        margin=dict(l=50, r=30, t=40, b=50),
                        legend=dict(orientation="h", y=1.12),
                        hovermode="x unified",
                    )
                    # Open a fresh 2-column pair every even-indexed chart.
                    if _sens_rendered % 2 == 0:
                        _sens_col_pair = st.columns(2)
                    _target = _sens_col_pair[_sens_rendered % 2]
                    with _target:
                        plotly_with_dl(
                            fig,
                            key=f"tune_sens_{topo_key}_{key}_{fname}",
                            filename=f"tune_sens_{topo_key}_{key}_{fname}")
                    _sens_rendered += 1



def _render_results_section(tuning_specs, topo_key, fname, low_perf):
    """The "Results" block: evaluated-in line, best-residual summary,
    "🏆 Use best values" button, the top-K table, and the xlsx download."""
    # ── Results — evaluated-in line + best residual + top-K table ──
    df = st.session_state.get(f"tune_df_{topo_key}_{fname}")
    if df is not None:
        best = df.iloc[0]
        _elapsed = st.session_state.get(f"tune_elapsed_{topo_key}_{fname}")
        if _elapsed is not None:
            st.markdown(
                f"**{tr('Evaluated in', '評估耗時')} "
                f"{_fmt_eval_time(_elapsed)}**")
        st.markdown(
            _best_summary_md(best, tuning_specs, topo_key, fname,
                                 label=tr("Best residual", "最佳殘差")),
            unsafe_allow_html=True,
        )

        # "Use best values" — push best-row params into the fine-tune widgets.
        # Uses on_click callback so writes happen *before* widgets re-render.
        def _apply_best(best_row, specs, topo, fn, low_perf):
            for spec in specs:
                key, label, scale = spec[0], spec[1], spec[2]
                unit = spec[3] if len(spec) > 3 else ""
                spec_step = spec[5] if len(spec) > 5 else None
                col_name = f"{label} ({unit})" if unit else label
                # A stale results table (e.g. persisted before the
                # closed-path param_rows fix) may lack parameter
                # columns — skip those instead of KeyError'ing.
                if col_name not in best_row.index:
                    continue
                disp_val = float(best_row[col_name])
                st.session_state[f"sim_{topo}_{key}_{fn}"] = disp_val
                kp = f"tune_{topo}_{key}_{fn}"
                # Re-seed the sweep box around the newly-applied best value
                # from the physics-informed ranges (widened to include it).
                d_min, d_step, d_max = informed_default_range(
                    key, label, disp_val,
                    low_perf=low_perf, spec_step=spec_step)
                st.session_state[f"{kp}_min"] = d_min
                st.session_state[f"{kp}_step"] = d_step
                st.session_state[f"{kp}_max"] = d_max

        st.container(key=f"hbt_amber_best_{topo_key}").button(
            tr("🏆 Use best values", "🏆 使用最佳值"),
            key=f"tune_best_{topo_key}_{fname}",
            on_click=_apply_best,
            args=(best, tuning_specs, topo_key, fname, low_perf))

        st.dataframe(df, width="stretch", hide_index=True)

        # Excel download
        buf = BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        st.download_button(
            tr("📥 Download as Excel", "📥 下載為 Excel"),
            data=buf,
            file_name=f"tuning_{topo_key}_{fname}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"tune_dl_{topo_key}_{fname}",
        )
def render_tuning_expander(model_cls, all_p, S_raw, freq, z0,
                           tuning_specs, fname, topo_key,
                           *, default_fit_keys=None):
    """🔧 Auto Tuning for Minimum Residuals — grid sweep + residual ranking.

    Renders the full sweep-grid / brute-force / optimized / prioritize /
    minimize-deviation toolset, plus a 🪜 Progressive auto-fit card (coarse →
    fine, physics-informed ranges).  Kept as ``render_tuning_expander`` for
    call-site stability (still imported under that name); the visible
    label is "Auto Tuning for Minimum Residuals".

    Parameters
    ----------
    model_cls : AbstractSSMModel subclass with .simulate(params, freq, z0)
    all_p     : dict — current parameter values in SI units
    S_raw     : np.ndarray — measured S-parameters (N, 2, 2)
    freq      : np.ndarray — frequency array
    z0        : float
    tuning_specs : list of (key, label, scale, unit, ...) tuples
                   scale converts SI → display unit (e.g. 1e15 for fF)
    fname     : str — filename for unique widget keys
    topo_key  : str — topology short name for unique widget keys
    default_fit_keys : iterable of spec keys, optional
        Spec keys pre-selected in the 🪜 Progressive auto-fit "Parameters to
        fit" multiselect.  When None, defaults to every spec whose canonical
        key is NOT a parasitic (pads / leads) — so intrinsic + extrinsic +
        access params (and unknown custom params) are on by default.
    """
    # Physics-informed default sweep ranges (and the low-perf flag they key on)
    # are used at every seeding site below; cache the low-perf detection once
    # per (topo, file) so we don't recompute fT/fmax on every rerun.
    _lp_key = f"tune_lowperf_{topo_key}_{fname}"
    if _lp_key not in st.session_state:
        st.session_state[_lp_key] = _detect_low_perf_device(S_raw, freq)
    _low_perf = bool(st.session_state[_lp_key])

    with st.container(key="hbt_exp_tune_auto_" + topo_key), \
         st.expander(tr("🔧 Auto Tuning for Minimum Residuals",
                        "🔧 自動調諧至最小殘差"), expanded=False):

        cpu_tag, cpu_help_suffix = _render_compute_backend_badge(model_cls, fname)

        prog_fit_keys, prog_cpu_clicked, prog_cuda_clicked = _render_full_auto_card(
            tuning_specs, default_fit_keys, topo_key, fname, cpu_help_suffix)

        # ── Pre-initialize session state defaults (only on first render of
        #    each key).  MUST stay outside the Semi-Auto toggle: every driver
        #    (Full Auto included) and the closed-path row build below read
        #    these keys.
        for spec in tuning_specs:
            key, label, scale = spec[0], spec[1], spec[2]
            spec_step = spec[5] if len(spec) > 5 else None
            current_si = float(all_p.get(key, 0.0))
            current_disp = current_si * scale
            is_zero = abs(current_si) < 1e-30
            kp = f"tune_{topo_key}_{key}_{fname}"
            if f"{kp}_chk" not in st.session_state:
                d_min, d_step, d_max = informed_default_range(
                    key, label, current_disp,
                    low_perf=_low_perf, spec_step=spec_step)
                st.session_state[f"{kp}_chk"]  = not is_zero
                st.session_state[f"{kp}_min"]  = d_min
                st.session_state[f"{kp}_step"] = d_step
                st.session_state[f"{kp}_max"]  = d_max

        # ── Per-parameter sweep-row data — ONE source of truth ──────────
        # param_rows feeds every driver (_run_one_sweep / _run_nelder_mead /
        # _run_progressive col-names + constant fill) and the sensitivity
        # section, so it must exist on every run whether or not the
        # Semi-Auto section below is open.  The row widgets are keyed, so
        # the closed path reads the very same session keys the widgets
        # would return — identical values by construction.
        #
        #    _row_entry / _clamp_row_session / _best_summary_md / _col_name and
        #    every _render_*_card helper below are module-level functions in this
        #    file (see the top) — hoisted out of this closure so they take
        #    explicit parameters instead of reading render_tuning_expander's
        #    locals by closure.

        # Mode-click flags must exist even when the Semi-Auto section is
        # closed (the dispatch below reads all of them) — default everything
        # False / neutral and let the open branch's widgets overwrite.
        cpu_clicked = cuda_clicked = False
        opt_cpu_clicked = opt_cuda_clicked = False
        prio_cpu_clicked = prio_cuda_clicked = False
        bal_cpu_clicked = bal_cuda_clicked = False
        bal_dev_threshold = 3.0
        _bal_use_res = False
        bal_res_threshold = 5.0
        prio_metric = st.session_state.get(
            f"tune_prio_metric_{topo_key}_{fname}", "S11")

        # ── 🔬 Semi-Auto Tune (toggle-collapsed section) ────────────────
        # Streamlit forbids nesting st.expander, so the collapsible section
        # is a toggle-gated bordered container instead (toggle off =
        # collapsed).
        param_rows = []
        semi_open = st.toggle(
            tr("🔬 Semi-Auto Tune", "🔬 半自動調諧"),
            key=f"tune_semi_open_{topo_key}_{fname}",
            help=tr("Manual sweep grid: pick parameters, set Min/Step/Max, "
                    "then run Brute force / Optimized / Prioritized passes.",
                    "手動掃描表格：選擇參數、設定最小值/步進/最大值，"
                    "再執行暴力法／最佳化／優先化計算。"))
        if semi_open:
            with st.container(border=True):
                param_rows = _render_semi_auto_table(
                    tuning_specs, all_p, topo_key, fname, _low_perf)
                cpu_clicked, cuda_clicked = _render_bruteforce_card(
                    topo_key, fname, cpu_help_suffix)
                opt_cpu_clicked, opt_cuda_clicked = _render_optimized_card(
                    topo_key, fname, cpu_help_suffix)
                prio_cpu_clicked, prio_cuda_clicked, prio_metric = (
                    _render_prioritized_card(topo_key, fname, cpu_help_suffix))

                # ── 🎚️  Minimize deviation (HIDDEN for now — to re-enable, flip
                #    `_SHOW_MIN_DEV` to True.  All the underlying logic at the
                #    `bal_cpu_clicked or bal_cuda_clicked` dispatch site below
                #    stays wired up; the only thing the toggle gates is the
                #    UI section.).
                _SHOW_MIN_DEV = False
                if _SHOW_MIN_DEV:
                    (bal_cpu_clicked, bal_cuda_clicked, bal_dev_threshold,
                     _bal_use_res, bal_res_threshold) = _render_min_dev_card(
                        topo_key, fname, cpu_tag)
        else:
            # Semi-Auto collapsed — the row widgets aren't rendered, but every
            # driver (Full Auto included) still needs param_rows.  Read the
            # same keyed session state the widgets would return; _row_entry is
            # the shared source of truth for sweep/n_calc, so the two paths
            # cannot diverge.  (The pre-init seeding loop above guarantees the
            # _chk/_min/_step/_max keys exist.)
            param_rows = _render_semi_auto_closed_rows(
                tuning_specs, all_p, topo_key, fname)

        # # ── Auto (Nelder-Mead) ──────────────────────────────────────────
        # st.caption(
        #     "🤖 **Auto (Nelder-Mead)** seeds the simplex from the current "
        #     "parameter values, expands/contracts adaptively (large moves when "
        #     "residual is high, small moves when low), and converges to a "
        #     "local minimum within the **Min/Max** boxes above.  Cheap, "
        #     "gradient-free, and respects the Stop button.  Most useful as a "
        #     "polish step after a coarse grid sweep.")
        # _auto_inputs = st.columns([1, 1, 2])
        # auto_max_iter = int(_auto_inputs[0].number_input(
        #     "Max iterations",
        #     min_value=10, max_value=10_000,
        #     value=int(st.session_state.get(
        #         f"tune_auto_iter_{topo_key}_{fname}", 200)),
        #     step=10,
        #     key=f"tune_auto_iter_{topo_key}_{fname}",
        #     help="Nelder-Mead iteration cap (function evals ≈ iters × (n+1))."))
        # auto_restart = _auto_inputs[1].checkbox(
        #     "Random restart on collapse",
        #     value=True,
        #     key=f"tune_auto_restart_{topo_key}_{fname}",
        #     help="If the simplex shrinks below tol before convergence, "
        #          "perturb x0 and run a second pass.")
        # _auto_btn = _auto_inputs[2].columns([1, 1] if _HAS_CUDA else [1])
        # auto_cpu_clicked = _auto_btn[0].button(
        #     "🤖 Auto (Nelder-Mead)",
        #     key=f"tune_calc_auto_{topo_key}_{fname}")
        # auto_cuda_clicked = (
        #     _HAS_CUDA
        #     and _auto_btn[1].button(
        #         "⚡ Auto with CUDA",
        #         key=f"tune_calc_auto_cuda_{topo_key}_{fname}"))


        # _run_one_sweep / _run_nelder_mead / _run_progressive were hoisted to
        # module-level functions in drivers_sweep.py / drivers_nelder_mead.py /
        # drivers_progressive.py (imported at the top of this file) instead of
        # staying nested here.  Every one of their real inputs was already an
        # explicit closure read of render_tuning_expander's own parameters plus
        # the freshly-built `param_rows` — i.e. they were never coupled to
        # widget-instantiation ORDER, just to a large shared-state surface —
        # so threading that surface through as explicit parameters was a safe,
        # mechanical hoist (verified name-for-name with an AST free-variable
        # scan).  See each driver module's docstring for details.

        # ── Dispatch ────────────────────────────────────────────────────
        if cpu_clicked or cuda_clicked:
            _run_one_sweep(model_cls, all_p, S_raw, freq, z0, param_rows,
                           tuning_specs, topo_key, fname,
                           use_cuda=bool(cuda_clicked), sort_metric="Total")
        elif opt_cpu_clicked or opt_cuda_clicked:
            # Recursive bisection: each iteration subsamples N_SUB evenly-
            # spaced points per swept parameter inside the current range, runs
            # the sweep, picks the top-2, and narrows the range to between
            # those two values.  Recursion stops once a final refinement at
            # the user's chosen step fits inside MAX_FINAL combos — that
            # final refinement is then run as the closing pass.
            #
            # N_SUB adapts to the dimensionality so per-iter cost stays
            # tractable: 5^13 ≈ 1.2 B combos is unreasonable, but 3^13 ≈
            # 1.6 M is fine and the box still narrows (just by ½ per iter
            # instead of ¼).  We pick the largest N_SUB ∈ {3, 4, 5} whose
            # full grid fits MAX_PER_ITER.
            _use_cuda      = bool(opt_cuda_clicked)
            MAX_FINAL      = 5_000_000
            # Per-iter cap: GPU can absorb ~100M combos in seconds, CPU
            # struggles past ~10M.  Both still drop to N_SUB=3 for very
            # high-dim sweeps — that's expected and still cheap.
            MAX_PER_ITER   = 100_000_000 if _use_cuda else 10_000_000
            # 3-point bisection halves the box per iter; 5-point quarters
            # it.  Going from a width of ~1 to ~1e-4 takes 14 iters at 3
            # points, 7 iters at 5 points — bumped to 30 to leave headroom.
            MAX_ITERS      = 30

            current_ranges = {}   # key -> (lo, hi) — search box this iter
            user_steps     = {}   # key -> float    — step from the row UI
            rows_by_key    = {}
            for row in param_rows:
                if not row["enabled"]:
                    continue
                full = np.asarray(row["sweep"], dtype=np.float64)
                if len(full) == 0:
                    continue
                current_ranges[row["key"]] = (float(full[0]), float(full[-1]))
                kp = f"tune_{topo_key}_{row['key']}_{fname}"
                user_steps[row["key"]] = (
                    float(st.session_state.get(f"{kp}_step", 0.0)) or 0.0)
                rows_by_key[row["key"]] = row

            if not current_ranges:
                st.warning(tr(
                    "Optimized mode needs at least one parameter with "
                    "the **Sweep** checkbox enabled.",
                    "最佳化模式至少需要一個已啟用**掃描**核取方塊的參數。"))
            else:
                # Adaptive N_SUB — largest in {3,4,5} that fits MAX_PER_ITER.
                # If even 3^N exceeds the budget we still use 3 and warn.
                _n_swept = len(current_ranges)
                N_SUB = 3
                for _cand in (5, 4, 3):
                    if (_cand ** _n_swept) <= MAX_PER_ITER:
                        N_SUB = _cand
                        break
                _per_iter = N_SUB ** _n_swept
                if N_SUB == 3 and _per_iter > MAX_PER_ITER:
                    st.warning(tr(
                        f"3-point subsample of {_n_swept} swept params is "
                        f"{_per_iter:,} combos — over the {MAX_PER_ITER:,} "
                        "per-iter target. The run will still proceed but "
                        "each iteration will be slow. Consider sweeping "
                        "fewer parameters at once.",
                        f"對 {_n_swept} 個掃描參數進行 3 點子取樣即為 "
                        f"{_per_iter:,} 種組合 — 超過每次疊代 "
                        f"{MAX_PER_ITER:,} 的目標。仍會繼續執行，但每次疊代"
                        "會較慢。建議一次掃描較少的參數。"))
                else:
                    st.caption(tr(
                        f"🎯 N_SUB = **{N_SUB}** points/param "
                        f"({_per_iter:,} combos per iter for {_n_swept} "
                        "swept params)",
                        f"🎯 N_SUB = **{N_SUB}** 點/參數"
                        f"（{_n_swept} 個掃描參數，每次疊代 "
                        f"{_per_iter:,} 種組合）"))
                def _estimate_final_combos(ranges, steps):
                    """How many combos would a brute-force sweep of these
                    ranges at the user's steps generate?"""
                    n = 1
                    for k, (lo, hi) in ranges.items():
                        step = steps.get(k, 0.0)
                        if step <= 0 or abs(hi - lo) < 1e-15:
                            pts = 1
                        else:
                            pts = max(1, int(round(abs(hi - lo) / step)) + 1)
                        n *= pts
                    return n

                # Short-circuit: if the user's full grid already fits the
                # final-refine budget, skip the subsample iterations and run
                # the brute force directly. No point doing 5^N subsampling
                # when we could just sweep everything.
                _initial_full = _estimate_final_combos(current_ranges, user_steps)
                _stop_recursion = False
                _final_iter_idx = None
                _skip_recursion = (_initial_full <= MAX_FINAL)
                if _skip_recursion:
                    st.caption(tr(
                        f"🎯 Full sweep is already **{_initial_full:,}** "
                        f"combos ≤ {MAX_FINAL:,} — skipping subsample, "
                        "running brute force at the user step directly.",
                        f"🎯 全掃描已為 **{_initial_full:,}** 組合 "
                        f"≤ {MAX_FINAL:,} — 略過子取樣，"
                        "直接以使用者步進執行暴力法。"))
                    _final_iter_idx = 0
                    iter_idx = 0
                # The recursion loop only runs when we actually need it.
                # Wrapped in `if not _skip_recursion` rather than `else:` so
                # the existing for-else block keeps working at its original
                # indentation.
                _do_loop = not _skip_recursion
                _loop_converged = False  # set True on any clean break
                for iter_idx in range(1, MAX_ITERS + 1) if _do_loop else range(0):
                    # Build 5-sample lists within the current ranges
                    sub = {}
                    for k, (lo, hi) in current_ranges.items():
                        if abs(hi - lo) < 1e-15:
                            sub[k] = np.array([lo], dtype=np.float64)
                        else:
                            sub[k] = np.linspace(lo, hi, N_SUB)

                    est_final = _estimate_final_combos(current_ranges, user_steps)
                    st.caption(tr(
                        f"🎯 Optimized iter {iter_idx} — current final-refine "
                        f"estimate: **{est_final:,}** combos "
                        f"(target ≤ {MAX_FINAL:,})",
                        f"🎯 最佳化疊代 {iter_idx} — 目前最終細化"
                        f"預估：**{est_final:,}** 組合"
                        f"（目標 ≤ {MAX_FINAL:,}）"))

                    _run_one_sweep(
                        model_cls, all_p, S_raw, freq, z0, param_rows,
                        tuning_specs, topo_key, fname,
                        use_cuda=_use_cuda, sweep_lists_override=sub,
                        sort_metric="Total",
                        phase_label=tr(
                            f"Iter {iter_idx} (subsample {N_SUB})",
                            f"疊代 {iter_idx}（子取樣 {N_SUB}）"),
                        phase_suffix=f"_optP{iter_idx}")

                    _df = st.session_state.get(f"tune_df_{topo_key}_{fname}")
                    if _df is None or len(_df) == 0:
                        # No finite residuals — abandon bisection but
                        # still run the closing pass on current_ranges so
                        # the user gets a brute-force result.
                        st.warning(tr(
                            f"Iteration {iter_idx} produced no finite "
                            "residuals — running the final refinement on "
                            "the current range without further narrowing.",
                            f"第 {iter_idx} 次疊代未產生有限殘差 — "
                            "將在目前範圍上執行最終細化，不再進一步縮小。"))
                        _final_iter_idx = iter_idx
                        _loop_converged = True
                        break

                    # Narrow to the box between the top-2 combos.  With
                    # only one survivor, centre a quarter-width box on it
                    # so the optimised pass keeps making progress.
                    single = len(_df) < 2
                    new_ranges = {}
                    for k, (lo_old, hi_old) in current_ranges.items():
                        col = _col_name(rows_by_key[k])
                        if col not in _df.columns:
                            new_ranges[k] = (lo_old, hi_old)
                            continue
                        v0 = float(_df.iloc[0][col])
                        if single:
                            half_w = abs(hi_old - lo_old) * 0.25
                            new_ranges[k] = (max(lo_old, v0 - half_w),
                                             min(hi_old, v0 + half_w))
                        else:
                            v1 = float(_df.iloc[1][col])
                            new_ranges[k] = (min(v0, v1), max(v0, v1))

                    # Bail if the box can't shrink anymore (guards against
                    # the corner case where the top-2 are identical).
                    if all(abs(new_ranges[k][1] - new_ranges[k][0]) < 1e-15
                           for k in new_ranges):
                        current_ranges = new_ranges
                        _final_iter_idx = iter_idx
                        _loop_converged = True
                        break

                    current_ranges = new_ranges

                    # Done once the final brute-force fits the budget
                    if _estimate_final_combos(current_ranges, user_steps) \
                            <= MAX_FINAL:
                        _final_iter_idx = iter_idx
                        _loop_converged = True
                        break
                # for-else replaced by explicit flag so the warning only
                # fires when the loop actually ran *and* didn't converge.
                if _do_loop and not _loop_converged:
                    st.warning(tr(
                        f"Reached MAX_ITERS={MAX_ITERS} without converging "
                        f"under {MAX_FINAL:,} combos — running the final "
                        "refinement on the last narrowed range anyway.",
                        f"已達 MAX_ITERS={MAX_ITERS} 仍未收斂至 "
                        f"{MAX_FINAL:,} 組合以下 — 仍會在最後縮小的範圍上"
                        "執行最終細化。"))

                # ── Closing pass: brute-force at the user's step ─────────
                if not _stop_recursion:
                    final_lists = {}
                    for k, (lo, hi) in current_ranges.items():
                        step = user_steps.get(k, 0.0)
                        if step <= 0 or abs(hi - lo) < 1e-15:
                            final_lists[k] = np.array([lo], dtype=np.float64)
                        else:
                            final_lists[k] = _make_sweep_values(lo, hi, step)
                    final_total = 1
                    for arr in final_lists.values():
                        final_total *= len(arr)
                    st.caption(tr(
                        f"🎯 Final refine — running **{final_total:,}** "
                        "combos at user-defined step.",
                        f"🎯 最終細化 — 正在以使用者定義步進執行 "
                        f"**{final_total:,}** 種組合。"))
                    _run_one_sweep(
                        model_cls, all_p, S_raw, freq, z0, param_rows,
                        tuning_specs, topo_key, fname,
                        use_cuda=_use_cuda, sweep_lists_override=final_lists,
                        sort_metric="Total",
                        phase_label=tr(
                            f"Final refine ({final_total:,} combos)",
                            f"最終細化（{final_total:,} 組合）"),
                        phase_suffix="_optFinal")
        elif prio_cpu_clicked or prio_cuda_clicked:
            _run_one_sweep(model_cls, all_p, S_raw, freq, z0, param_rows,
                            tuning_specs, topo_key, fname,
                            use_cuda=bool(prio_cuda_clicked),
                            sort_metric=str(prio_metric),
                            phase_label=tr(f"Prioritize {prio_metric}",
                                           f"優先排序 {prio_metric}"),
                            phase_suffix=f"_prio{prio_metric}")
        elif bal_cpu_clicked or bal_cuda_clicked:
            _res_thr = float(bal_res_threshold) if _bal_use_res else None
            _run_one_sweep(
                model_cls, all_p, S_raw, freq, z0, param_rows,
                tuning_specs, topo_key, fname,
                use_cuda=bool(bal_cuda_clicked),
                sort_metric="Total",
                phase_label=tr(f"Balance ≤{bal_dev_threshold:.2f}%",
                               f"平衡 ≤{bal_dev_threshold:.2f}%"),
                phase_suffix="_bal",
                dev_threshold=float(bal_dev_threshold),
                res_threshold=_res_thr,
            )
        elif prog_cpu_clicked or prog_cuda_clicked:
            _run_progressive(
                model_cls, all_p, S_raw, freq, z0, param_rows,
                tuning_specs, topo_key, fname, _low_perf,
                use_cuda=bool(prog_cuda_clicked),
                fit_keys=list(prog_fit_keys),
                target_pct=0.0)
        # elif auto_cpu_clicked or auto_cuda_clicked:
        #     _run_nelder_mead(
        #         max_iter=int(auto_max_iter),
        #         restart=bool(auto_restart),
        #         use_cuda=bool(auto_cuda_clicked),
        #     )


        _render_sensitivity_section(model_cls, all_p, S_raw, freq, z0,
                                    tuning_specs, topo_key, fname)

        _render_results_section(tuning_specs, topo_key, fname, _low_perf)
