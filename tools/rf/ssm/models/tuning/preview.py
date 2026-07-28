"""
models/tuning/preview.py — Visual Tuning: slider previews (Live-rerun mode
and pre-computed Plotly-slider mode) shown above the Auto Tuning expander.

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).
"""
from __future__ import annotations
import numpy as np
import streamlit as st

from ...helpers import params_hash, segmented_radio
from tools.common.mem_budget import ram_available_bytes
from tools.common.i18n import tr

from ..residuals import ssm_residual
from ..smith_ui import render_smith_chart
from .ranges import tune_hard_limits, _clamp_to_hard

# ── CUDA detection (runtime, zero-cost when CuPy is absent) ─────────────────

try:
    import cupy as _cp
    _HAS_CUDA = True
    _v = _cp.cuda.runtime.runtimeGetVersion()      # e.g. 13000
    _CUDA_VER = f"{_v // 1000}.{(_v % 1000) // 10}"
except Exception:
    _cp = None          # type: ignore[assignment]
    _HAS_CUDA = False
    _CUDA_VER = ""


# ── Streamlit fragment decorator (1.36 → st.experimental_fragment;
#    1.37+ → st.fragment).  Wrapping the slider-preview render functions
#    in a fragment confines slider-drag reruns to JUST the fragment —
#    the rest of the SSM script (Sections 1-5, all other models) does
#    NOT re-execute.  That's the order-of-magnitude speedup for Live
#    mode (1-2 s per drag → ~150 ms).
_FRAGMENT = (getattr(st, "fragment", None)
             or getattr(st, "experimental_fragment", None)
             or (lambda f: f))


# ── Tuning (parameter sweep + residual table) ───────────────────────────────

def _make_sweep_values(min_val, max_val, step):
    """Generate sweep values, always including max_val as the last point."""
    if step <= 0 or abs(max_val - min_val) < 1e-15:
        return np.array([min_val])
    if max_val < min_val:
        return np.array([min_val])
    values = np.arange(min_val, max_val + step * 0.5, step)
    if len(values) == 0:
        return np.array([min_val])
    # Ensure max is included
    if abs(values[-1] - max_val) > 1e-12:
        values = np.append(values, max_val)
    return values


def _fmt_eta(seconds) -> str:
    """Human-readable ETA: seconds under a minute, ``Xm Ys`` under an hour,
    ``Xh Ym Zs`` beyond an hour."""
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{s:.1f}s"
    total = int(round(s))
    if total < 3600:
        m, sec = divmod(total, 60)
        return f"{m}m {sec}s"
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m {sec}s"


def _fmt_eval_time(seconds) -> str:
    """Total run time as ``xx s (xx h: xx m: xx s)`` — raw seconds plus an
    hours:minutes:seconds breakdown."""
    s = max(0.0, float(seconds))
    total = int(s)
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{s:.1f} s ({h} h: {m:02d} m: {sec:02d} s)"


def _render_slider_preview(model_cls, all_p, S_raw, freq, z0,
                           tuning_specs, fname, topo_key):
    """Sandbox-style slider preview at the top of the Tuning expander.

    Two flavors selectable via the mode selector:

      🎯 All sweep           — drag any number of sliders; every drag-tick
                                triggers a Streamlit rerun + a full sim.
                                Slow with many params or many freq points,
                                but supports multi-param sliding.

      ⚡ Smooth sweep        — click ``🧮 Build animation`` once, then the
                                embedded Plotly figure scrubs through
                                pre-computed frames entirely client-side
                                (no Streamlit rerun per drag-tick).
                                Multi-param: frames are the cartesian
                                product of every selected sweep axis.

    Sliders in either mode write into ``slpreview_*`` session keys.
    ✅ "Use these values" (live mode) copies them into the main ``sim_*``
    keys; the auto-save gate in ``render_override_and_smith`` then picks
    that change up and persists it to the fit cache.
    """
    mode_key = f"slpreview_mode_{topo_key}_{fname}"
    mode = segmented_radio(
        tr("Preview mode", "預覽模式"),
        [tr("🎯 All sweep", "🎯 全參數掃描"),
         tr("⚡ Smooth sweep", "⚡ 平滑掃描")],
        index=0,
        key=mode_key,
        help=tr(
            "🎯 All sweep — best for a few small changes: the plots "
            "re-compute on every drag.  ⚡ Smooth sweep — best for exploring a "
            "large range: pre-computes the whole range once so dragging is "
            "instant afterwards.",
            "🎯 全參數掃描 — 適合少量微調：每次拖曳都會重新計算圖表。"
            "⚡ 平滑掃描 — 適合探索大範圍：一次預先計算整個範圍，之後拖曳即時反應。"))
    if mode.startswith("⚡"):
        _render_plotly_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                       tuning_specs, fname, topo_key)
    else:
        _render_live_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                     tuning_specs, fname, topo_key)


def _slider_default_range(current_disp, key="", label=""):
    """Sane default (min, max, step) for one slider given the current value.

    The (min, max) are clamped into ``tune_hard_limits(key, label)`` so R / L / C
    sliders floor at 0 and alpha0 stays inside [0.95, 0.99].  A zero current
    value defaults to (0.0, 1.0, 0.01) rather than a symmetric ±1 span.
    """
    hard_lo, hard_hi = tune_hard_limits(key, label)
    if abs(current_disp) < 1e-30:
        d_min, d_max = 0.0, 1.0
    else:
        lo = current_disp * 0.1 if current_disp > 0 else current_disp * 10
        hi = current_disp * 10  if current_disp > 0 else current_disp * 0.1
        d_min, d_max = min(lo, hi), max(lo, hi)
    d_min, d_max = _clamp_to_hard(d_min, d_max, hard_lo, hard_hi)
    d_step = max((d_max - d_min) / 100, 1e-9)
    return d_min, d_max, d_step


def _multi_metric_top_n(arr, per_metric: int = 10):
    """Return the union of top-``per_metric`` rows by each of the first five
    columns (Total, S11, S12, S21, S22), deduped, sorted by Total Residual.

    ``arr`` is an (N, n_cols) float ndarray.  Rows whose Total Residual is
    non-finite or >= 1e30 (the BIG sentinel used to mark filtered combos)
    are dropped first.  Result row count: 10 (all five metrics' bests are
    the same row) ≤ R ≤ 50 (all distinct).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or arr.shape[0] == 0:
        return arr
    finite_mask = np.isfinite(arr[:, 0]) & (arr[:, 0] < 1e30)
    arr = arr[finite_mask]
    if arr.shape[0] == 0:
        return arr
    keep: set[int] = set()
    for c in range(min(5, arr.shape[1])):
        col_vals = arr[:, c]
        col_safe = np.where(np.isfinite(col_vals), col_vals, np.inf)
        n_take   = min(per_metric, arr.shape[0])
        idx      = np.argpartition(col_safe, n_take - 1)[:n_take]
        keep.update(idx.tolist())
    sub = arr[sorted(keep)]
    unique = np.unique(sub, axis=0)
    return unique[np.argsort(unique[:, 0])]


@_FRAGMENT
def _render_live_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                tuning_specs, fname, topo_key):
    """Streamlit-rerun-per-drag preview.

    Layout (v4 — fixed-height scroll containers, same primitive
    Streamlit's own sidebar uses for independent scrolling)
    --------------------------------------------------------
    Two top-level columns, each wrapped in ``st.container(height=…)``
    so they get their own scrollbar.  The page itself does not grow
    when many variable cards are added — the LEFT container scrolls
    internally, the RIGHT container stays put.

      ┌─ Left scroll-box ──────────┐ ┌─ Right scroll-box ─────────────┐
      │ [Parameters to slide]      │ │   Smith         |     Bode     │
      │  (narrow multiselect)      │ │   (compact)     |   (compact)   │
      │ ─────────                  │ │   legend below  |  legend below │
      │ variable card 1            │ │                                │
      │ variable card 2            │ │                                │
      │ variable card 3            │ │                                │
      │ ... (1 per row, scrolls)   │ │                                │
      └────────────────────────────┘ └────────────────────────────────┘
      ┌─ Below the columns (always visible) ─────────────────────────┐
      │  ✅ Use these values     ↩️ Reset preview                     │
      └───────────────────────────────────────────────────────────────┘

    Buttons live OUTSIDE the scroll boxes so the user doesn't have to
    scroll the left panel to find them.
    """
    from ...ssm_plots import render_ft_fmax_card

    # Fixed height for the two scroll boxes — picks a value comfortable
    # on a typical 1080p laptop, deliberately taller than the plots so
    # the right box never shows its own scrollbar (the plots fit) while
    # the left box gets scrollbars when the user adds many variables.
    _BOX_HEIGHT = 680

    label_for = {s[0]: s[1] for s in tuning_specs}

    # ── Top-level split — keep the slider column compact so the plots get
    #    most of the width.
    controls_col, plots_col = st.columns([0.6, 1.4])

    # Pre-declare BOTH scroll containers so we can append into them in
    # any order (simulation happens after slider values are read).
    with controls_col:
        sliders_box = st.container(height=_BOX_HEIGHT, border=False)
    with plots_col:
        plots_box = st.container(height=_BOX_HEIGHT, border=False)

    # ── Controls box: multiselect (narrowed) + variable cards ─────────
    with sliders_box:
        # Constrain multiselect width with a sub-column so the chips
        # don't wrap awkwardly across the full panel.
        sel_key  = f"slpreview_sel_{topo_key}_{fname}"
        ms_col, _ms_pad = st.columns([3, 1])
        with ms_col:
            selected = st.multiselect(
                tr("Parameters to slide", "可拖曳參數"),
                options=[s[0] for s in tuning_specs],
                default=st.session_state.get(sel_key, []),
                format_func=lambda k: label_for.get(k, k),
                key=sel_key,
                placeholder=tr("Choose options", "請選擇項目"),
                help=tr(
                    "Pick parameter(s) to drag.  Plots on the right "
                    "show the slider-substituted model in real time.  "
                    "The main Smith / fT-fmax plots above stay frozen "
                    "until you click ✅ Use these values.",
                    "選擇要拖曳的參數。右側圖表即時顯示套用滑桿值後的模型。"
                    "上方主要的 Smith / fT-fmax 圖會維持不變，"
                    "直到你點擊「✅ 使用這些數值」為止。"))

        selected_specs = [s for s in tuning_specs if s[0] in selected]
        preview_overrides: dict[str, float] = {}

        # Initialize ranges + sliders
        for spec in selected_specs:
            key, _label, scale = spec[0], spec[1], spec[2]
            current_disp  = float(all_p.get(key, 0.0)) * scale
            kp = f"slpreview_{topo_key}_{key}_{fname}"
            if f"{kp}_min" not in st.session_state:
                d_min, d_max, d_step = _slider_default_range(
                    current_disp, key, _label)
                st.session_state[f"{kp}_min"]  = float(d_min)
                st.session_state[f"{kp}_max"]  = float(d_max)
                st.session_state[f"{kp}_step"] = float(d_step)
            if kp not in st.session_state:
                st.session_state[kp] = float(current_disp)

        # Variable cards — ONE per row.  The fixed-height scroll box
        # handles overflow internally.
        if not selected_specs:
            st.caption(tr("Select one or more parameters above to begin.",
                          "請先在上方選擇一個或多個參數以開始。"))
        else:
            for spec in selected_specs:
                key, label, scale = spec[0], spec[1], spec[2]
                unit = spec[3] if len(spec) > 3 else ""
                fmt  = spec[4] if len(spec) > 4 else "%.4g"
                current_disp = float(all_p.get(key, 0.0)) * scale
                kp = f"slpreview_{topo_key}_{key}_{fname}"
                mn = float(st.session_state[f"{kp}_min"])
                mx = float(st.session_state[f"{kp}_max"])
                sp = float(st.session_state[f"{kp}_step"])
                if mx <= mn:
                    mx = mn + max(sp, abs(mn) * 1e-6 + 1e-9)
                sp_safe = sp if sp > 0 else max((mx - mn) / 100, 1e-12)
                # Clamp the persisted slider value into the CURRENT
                # min/max bounds and write it back to session_state
                # BEFORE the widget renders.  Passing `value=` to a
                # widget that also has `key=` (where the key is in
                # session_state) triggers Streamlit's
                # check_session_state_rules warning every render —
                # the supported pattern is "set the key in
                # session_state, then omit value=".
                cur_v = min(max(float(st.session_state.get(kp, current_disp)),
                                mn), mx)
                st.session_state[kp] = cur_v
                label_unit = f"{label} ({unit})" if unit else label
                with st.container(border=True):
                    head = st.columns([1.4, 1, 1, 1])
                    head[0].markdown(
                        f"<div style='padding-top:1.6em;font-weight:600'>"
                        f"{label_unit}</div>",
                        unsafe_allow_html=True)
                    head[1].number_input(tr("Min", "最小值"), format=fmt,
                                         key=f"{kp}_min")
                    head[2].number_input(tr("Step", "步進"), format=fmt,
                                         key=f"{kp}_step",
                                         min_value=0.0)
                    head[3].number_input(tr("Max", "最大值"), format=fmt,
                                         key=f"{kp}_max")
                    v = st.slider(label_unit, min_value=mn, max_value=mx,
                                  step=sp_safe, format=fmt,
                                  key=kp, label_visibility="collapsed")
                    st.caption(f"{tr('main', '目前')}: **{current_disp:.4g}**  →  "
                               f"{tr('preview', '預覽')}: **{v:.4g}** {unit}".rstrip())
                preview_overrides[key] = float(v) / scale

    # ── Preview simulation (same logic as before, narrowed plot heights
    #    because each plot now occupies a half-width column).
    if preview_overrides:
        all_p_prev = dict(all_p)
        all_p_prev.update(preview_overrides)
        swept_keys = list(preview_overrides.keys())
        xp_live = _cp if _HAS_CUDA else np

        static_cache_key  = f"slpreview_static_cache_{topo_key}_{fname}"
        static_hash_input = {k: float(v) for k, v in all_p.items()
                              if k not in swept_keys
                              and isinstance(v, (int, float))}
        static_hash_input["__swept"] = tuple(sorted(swept_keys))
        static_hash_input["__nf"]    = int(len(freq))
        static_hash_input["__xp"]    = "cuda" if xp_live is not np else "cpu"
        static_hash = params_hash({k: str(v) for k, v in static_hash_input.items()})
        cached_static = st.session_state.get(static_cache_key)
        if cached_static is None or cached_static.get("hash") != static_hash:
            try:
                static_cache = model_cls.build_static_cache(
                    all_p, freq, xp=xp_live, swept_keys=swept_keys)
            except Exception:
                static_cache = None
            st.session_state[static_cache_key] = {"hash": static_hash,
                                                  "cache": static_cache}
            cached_static = st.session_state[static_cache_key]
        static_cache = cached_static.get("cache")

        p_batch = dict(all_p)
        for k, v in preview_overrides.items():
            p_batch[k] = xp_live.asarray([v], dtype=float)
        S_prev = None
        try:
            S_b = model_cls.simulate_batch(p_batch, freq, z0,
                                            xp=xp_live, cache=static_cache)
            if xp_live is not np:
                S_b = _cp.asnumpy(S_b)
            S_prev = np.asarray(S_b)[0]
        except Exception:
            try:
                S_prev = model_cls.simulate_vec(all_p_prev, freq, z0)
            except Exception as e:
                st.error(tr(f"Preview simulation failed: {e}",
                            f"預覽模擬失敗：{e}"))
                S_prev = None
        if S_prev is not None and not np.all(np.isfinite(S_prev)):
            st.warning(tr("Preview S-parameters contain non-finite values — "
                          "adjust slider ranges to avoid singular combinations.",
                          "預覽 S 參數包含非有限值 — 請調整滑桿範圍以避免奇異組合。"))
            S_prev = None
        if S_prev is not None:
            # Render INTO the right scroll box.  Smith + Bode go in two
            # sub-columns so they sit side-by-side, with legends below
            # each plot (compact mode on the smith chart).
            with plots_box:
                st.markdown(
                    f"<div style='font-size:0.85em;color:#555;"
                    f"margin-bottom:4px'>{tr('Preview', '預覽')} — {model_cls.NAME} "
                    f"({tr('residual', '殘差')} {ssm_residual(S_raw, S_prev):.2f}%)"
                    f"</div>",
                    unsafe_allow_html=True)
                smith_col, bode_col = st.columns(2)
                with smith_col:
                    # Default to the per-trace display multipliers set above the
                    # main Smith chart (smith_scale_controls) so the preview
                    # matches it; inline Sxx labels keep the legend off the plot.
                    _scales = {
                        nm: float(st.session_state.get(
                            f"smith_scale_{topo_key}_{nm}_{fname}", 1.0))
                        for nm in ("S11", "S12", "S21", "S22")}
                    render_smith_chart(
                        S_raw, S_prev, model_cls.NAME,
                        ssm_residual(S_raw, S_prev),
                        scales=_scales,
                        key=f"slpreview_smith_{topo_key}_{fname}",
                        show_title=False,
                        compact=True, height=540, inline_labels=True)
                with bode_col:
                    render_ft_fmax_card(
                        S_raw, S_prev, freq,
                        model_name=model_cls.NAME,
                        key=f"slpreview_bode_{topo_key}_{fname}",
                        height=540, compact=True)

    # ── Commit / Reset buttons — appended to the left column BELOW the
    #    sliders_box scroll container, so they're always visible
    #    without scrolling the slider list.
    with controls_col:
        bc1, bc2 = st.columns(2)
        commit_clicked = bc1.container(key=f"hbt_amber_slcommit_{topo_key}").button(
            tr("✅ Use these values", "✅ 使用這些數值"),
            key=f"slpreview_commit_{topo_key}_{fname}",
            disabled=(len(preview_overrides) == 0),
            help=tr(
                "Copy slider values into the fine-tune Smith-chart override "
                "fields above.  Does NOT auto-save to the persistent fit "
                "cache — only direct edits in the fine-tune number_inputs do.",
                "將滑桿數值複製到上方的微調 Smith 圖覆寫欄位。"
                "不會自動儲存到永久擬合快取 — 只有直接編輯微調數字輸入框才會。"),
            width="stretch")
        reset_clicked = bc2.button(
            tr("↩️ Reset preview", "↩️ 重設預覽"),
            key=f"slpreview_reset_{topo_key}_{fname}",
            help=tr("Discard slider drags and clear remembered min/step/max.",
                    "捨棄滑桿拖曳並清除記住的最小值/步進/最大值。"),
            width="stretch")

    if commit_clicked:
        for k, v_si in preview_overrides.items():
            sc = next(s[2] for s in tuning_specs if s[0] == k)
            st.session_state[f"sim_{topo_key}_{k}_{fname}"] = float(v_si) * sc
        st.rerun()

    if reset_clicked:
        for s in tuning_specs:
            kp = f"slpreview_{topo_key}_{s[0]}_{fname}"
            for suf in ("", "_min", "_step", "_max"):
                st.session_state.pop(kp + suf, None)
        st.rerun()


def _chunked_simulate_batch_to_host(model_cls, p_batch, freq, z0, *,
                                     xp, chunk_size: int = 5000,
                                     dtype=np.complex64):
    """Run ``simulate_batch`` in chunks of ``chunk_size`` so OOM doesn't bite
    on million-frame sweeps.  Returns one host-side ``np.ndarray`` of the
    requested complex ``dtype``.

    The default storage dtype is **complex64** (2× memory savings vs.
    complex128 with imperceptible visual difference on Smith + Bode).
    Pass ``dtype=np.complex128`` for full fp64 storage if you need it
    for downstream residual calculations.
    """
    if xp is np:
        # Shrink the effective chunk size to whatever CPU RAM is actually
        # available right now — same per-row working-set model as the
        # Full Auto Tune driver (_run_progressive) and _run_one_sweep's
        # per_combo_bytes. Prevents a caller-supplied (or default) chunk_size
        # from allocating more than the Streamlit Cloud cgroup limit in one
        # simulate_batch call, which would get SIGKILLed before any
        # except MemoryError recovery path could run.
        per_row_bytes = 16 * 4 * len(freq) * 12
        cap = max(256, int(ram_available_bytes() * 0.25 // max(per_row_bytes, 1)))
        chunk_size = min(chunk_size, cap)

    sweep_keys = []
    B = None
    for k, v in p_batch.items():
        if isinstance(v, np.ndarray) or (_HAS_CUDA and isinstance(v, _cp.ndarray)):
            sweep_keys.append(k)
            if B is None:
                B = int(v.shape[0])
    if B is None or B <= chunk_size:
        out = model_cls.simulate_batch(p_batch, freq, z0, xp=xp)
        if xp is not np:
            out = _cp.asnumpy(out)
        return np.asarray(out).astype(dtype, copy=False)

    chunks = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        sub_p = {k: (v[start:end] if k in sweep_keys else v)
                 for k, v in p_batch.items()}
        S_c = model_cls.simulate_batch(sub_p, freq, z0, xp=xp)
        if xp is not np:
            S_c = _cp.asnumpy(S_c)
            try:
                _cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        chunks.append(np.asarray(S_c).astype(dtype, copy=False))
    return np.concatenate(chunks, axis=0)


@_FRAGMENT
def _render_plotly_slider_preview(model_cls, all_p, S_raw, freq, z0,
                                  tuning_specs, fname, topo_key):
    """Pre-computed Plotly slider — joint (cartesian) multi-param scan.

    Each selected param becomes one Plotly slider.  Frames are the *full
    cartesian product* of all sliders' sweep values, so dragging slider
    B reflects the model at the *current position of every other slider*
    (true M × N × K joint behaviour).  Coordination between sliders is
    done by injected JS listening to ``plotly_sliderchange``.

    Performance notes
    -----------------
    • All traces use ``Scattergl`` (WebGL).
    • Frequency axis defaults to full fidelity (≤ 1001 points per trace);
      the **Freq points** input decimates it when the payload grows large.
    • Single ``simulate_batch`` call runs the full cartesian product —
      one GPU pass when CUDA is available.

    Cache
    -----
    Built batch + slider specs stash in session_state until the user
    changes the selection / ranges and clicks ``🧮 Build`` again.
    """
    from ...helpers.plotly_plots import build_smith_bode_slider_payload
    from ...components import smith_bode_slider

    label_for = {s[0]: s[1] for s in tuning_specs}
    options   = [s[0] for s in tuning_specs]

    sel_key = f"slprev_pl_sel_{topo_key}_{fname}"
    selected = st.multiselect(
        tr("Sweep parameters", "掃描參數"),
        options=options,
        default=st.session_state.get(sel_key, [options[0]] if options else []),
        format_func=lambda k: label_for.get(k, k),
        key=sel_key,
        placeholder=tr("Choose options", "請選擇項目"),
        help=tr(
            "Each selected param gets its own Plotly slider in the figure. "
            "Frames are the FULL cartesian product, so dragging slider B "
            "reflects the model at the current position of every other "
            "slider (true joint scan).  Watch the total frame count below "
            "— it grows multiplicatively.",
            "每個選取的參數在圖表中都有各自的 Plotly 滑桿。"
            "各幀是所有滑桿值的完整笛卡兒積，因此拖曳滑桿 B 時，"
            "會反映其他每個滑桿目前位置下的模型（真正的聯合掃描）。"
            "請留意下方的總幀數 — 它會以乘法方式增長。"))

    selected_specs = [s for s in tuning_specs if s[0] in selected]
    if not selected_specs:
        st.caption(tr("Select one or more parameters above and click "
                      "**🧮 Build animation**.",
                      "請先在上方選擇一個或多個參數，再點擊"
                      "**🧮 建立動畫**。"))
        return

    # ── Initialize per-param ranges ────────────────────────────────────
    # Default frames-per-axis shrinks as more params are selected so the
    # cartesian product (and thus the embedded payload) stays manageable
    # at full frequency fidelity: 11 for 1-2 params, 7 for 3, 5 for 4+.
    n_sel = len(selected_specs)
    default_frames = 11 if n_sel <= 2 else (7 if n_sel == 3 else 5)
    for spec in selected_specs:
        key, _label, scale = spec[0], spec[1], spec[2]
        current_disp  = float(all_p.get(key, 0.0)) * scale
        kp = f"slprev_pl_{topo_key}_{key}_{fname}"
        if f"{kp}_min" not in st.session_state:
            d_min, d_max, _ = _slider_default_range(current_disp, key, _label)
            st.session_state[f"{kp}_min"]    = float(d_min)
            st.session_state[f"{kp}_max"]    = float(d_max)
            st.session_state[f"{kp}_frames"] = default_frames

    # ── 2-column variable-card grid (min / max / frames per card) ──────
    def _render_one_range_card(spec):
        key, label, scale = spec[0], spec[1], spec[2]
        unit = spec[3] if len(spec) > 3 else ""
        fmt  = spec[4] if len(spec) > 4 else "%.4g"
        kp   = f"slprev_pl_{topo_key}_{key}_{fname}"
        label_unit = f"{label} ({unit})" if unit else label
        with st.container(border=True):
            head = st.columns([1.4, 1, 1, 1])
            # Top-pad the variable name so it sits at the same vertical
            # level as the input boxes (whose own "Min"/"Max"/"Frames"
            # labels add ~1.6 em of header height above them).
            head[0].markdown(
                f"<div style='padding-top:1.6em;font-weight:600'>"
                f"{label_unit}</div>",
                unsafe_allow_html=True)
            head[1].number_input(tr("Min", "最小值"), format=fmt, key=f"{kp}_min")
            head[2].number_input(tr("Max", "最大值"), format=fmt, key=f"{kp}_max")
            head[3].number_input(tr("Frames", "幀數"), min_value=2, max_value=100, step=1,
                                 key=f"{kp}_frames",
                                 help=tr("Frames per axis (2–100). "
                                         "Total = product across params.",
                                         "每軸幀數（2–100）。"
                                         "總數 = 各參數幀數的乘積。"))

    for row_start in range(0, len(selected_specs), 2):
        row_specs = selected_specs[row_start:row_start + 2]
        l_col, r_col = st.columns(2)
        with l_col:
            _render_one_range_card(row_specs[0])
        with r_col:
            if len(row_specs) > 1:
                _render_one_range_card(row_specs[1])
            else:
                st.empty()

    # ── Decimation (fidelity) control + payload estimate ───────────────
    n_freq_full = int(len(freq))
    decim_default = min(1001, n_freq_full)
    decim_key   = f"slprev_pl_decim_{topo_key}_{fname}"
    if decim_key not in st.session_state:
        st.session_state[decim_key] = decim_default

    dims_preview = []
    for spec in selected_specs:
        kp = f"slprev_pl_{topo_key}_{spec[0]}_{fname}"
        dims_preview.append(int(st.session_state.get(f"{kp}_frames",
                                                     default_frames)))
    total_frames = int(np.prod(dims_preview)) if dims_preview else 0

    # Estimated payload: (5-sig-fig ≈ 7 chars / number) × 10 numbers / sample.
    decim_n = int(st.session_state.get(decim_key, decim_default))
    decim_n = min(decim_n, n_freq_full)
    est_mb  = total_frames * decim_n * 10 * 7 / 1024 / 1024

    fd_col1, fd_col2 = st.columns([1, 2])
    with fd_col1:
        st.number_input(
            f"{tr('Freq points', '頻率點數')} (max: {n_freq_full})",
            min_value=20, max_value=n_freq_full, step=10,
            key=decim_key,
            help=tr(
                f"Frequency points kept per trace (max = {n_freq_full} = "
                "full fidelity).  Lower this (~120 still looks smooth on "
                "Smith / Bode) when the payload estimate grows large.",
                f"每條曲線保留的頻率點數（最大 = {n_freq_full} = 完整精度）。"
                "當估計負載變大時可調低此值（約 120 在 Smith / Bode 圖上"
                "仍相當平滑）。"))
    with fd_col2:
        st.caption(
            f"{tr('Cartesian sweep', '笛卡兒掃描')}: "
            + " × ".join(str(d) for d in dims_preview)
            + f" = **{total_frames}** {tr('frames', '幀')} · {decim_n} "
            f"{tr('freq pts', '頻率點')} · "
            f"{tr('estimated payload', '估計負載')} ≈ **{est_mb:.0f} MB**")

    if est_mb > 180:
        st.error(tr(
            f"❌ Estimated payload ≈ {est_mb:.0f} MB will exceed "
            "Streamlit's 200 MB browser-message limit.  Lower the "
            "**Freq points** value, reduce per-axis frame counts, or "
            "raise the limit via `.streamlit/config.toml` → "
            "`[server] maxMessageSize = 500`.",
            f"❌ 估計負載 ≈ {est_mb:.0f} MB 將超過 Streamlit 的 200 MB "
            "瀏覽器訊息上限。請降低 **頻率點數**、減少每軸幀數，"
            "或透過 `.streamlit/config.toml` → `[server] maxMessageSize = 500` "
            "提高上限。"))
    elif est_mb > 120:
        st.warning(tr(f"⚠️ Estimated payload ≈ {est_mb:.0f} MB is close "
                      "to Streamlit's 200 MB limit.",
                      f"⚠️ 估計負載 ≈ {est_mb:.0f} MB 已接近 Streamlit 的 "
                      "200 MB 上限。"))
    elif total_frames > 2000:
        st.warning(tr(f"⚠️ {total_frames} frames may stutter on "
                      "slider drag.",
                      f"⚠️ {total_frames} 幀可能會導致拖曳滑桿時卡頓。"))

    # ── CUDA checkbox + Build button (button next to checkbox when CUDA available)
    cuda_toggle_key = f"slprev_pl_cuda_{topo_key}_{fname}"
    if _HAS_CUDA:
        cuda_col, btn_col = st.columns([1.6, 1])
        with cuda_col:
            use_cuda = st.checkbox(tr(f"⚡ Use CUDA (cupy {_CUDA_VER}) for "
                                      "batched simulation",
                                      f"⚡ 使用 CUDA（cupy {_CUDA_VER}）"
                                      "進行批次模擬"),
                                    value=st.session_state.get(cuda_toggle_key, True),
                                    key=cuda_toggle_key,
                                    help=tr("Off-load the joint cartesian "
                                            "batched simulation to the GPU.  "
                                            "Result is brought back to host as "
                                            "fp64 for Plotly embedding.",
                                            "將聯合笛卡兒批次模擬卸載至 GPU 運算。"
                                            "結果會以 fp64 帶回主機供 Plotly 嵌入。"))
        with btn_col:
            build_clicked = st.button(tr("🧮 Build animation", "🧮 建立動畫"),
                                      key=f"slprev_pl_build_{topo_key}_{fname}",
                                      width="stretch",
                                      help=tr("Pre-compute the cartesian joint "
                                              "sweep and embed with JS-"
                                              "coordinated multi-sliders.",
                                              "預先計算笛卡兒聯合掃描，"
                                              "並以 JS 協調的多重滑桿嵌入。"))
    else:
        use_cuda = False
        build_clicked = st.button(tr("🧮 Build animation", "🧮 建立動畫"),
                                  key=f"slprev_pl_build_{topo_key}_{fname}",
                                  width="stretch",
                                  help=tr("Pre-compute the cartesian joint "
                                          "sweep and embed with JS-coordinated "
                                          "multi-sliders.",
                                          "預先計算笛卡兒聯合掃描，"
                                          "並以 JS 協調的多重滑桿嵌入。"))

    state_key = f"slprev_pl_state_{topo_key}_{fname}"

    if build_clicked:
        import time as _time
        xp = _cp if (use_cuda and _HAS_CUDA) else np
        if xp is not np:
            device_label = f"GPU (cupy {_CUDA_VER})"
        else:
            # CPU path routes through simulate_batch → the Rust end-to-end
            # kernel whenever it's available (mirrors _detect_rust_active),
            # so the badge must reflect what actually ran, not a hardcoded
            # "numpy".  Only true NumPy composition gets the numpy label.
            try:
                from ...helpers.rust_kernels import (
                    HAS_RUST as _HR,
                    _phase2_dispatch_enabled as _p2on,
                    SIM_FOR_TOPOLOGY as _SIMTOPO,
                )
                from ...helpers import rust_kernels as _RKMOD
                _rust_used = bool(
                    _HR and _p2on() and (
                        _SIMTOPO.get(model_cls.SHORT) is not None
                        or (getattr(model_cls, "USES_RUST_BATCH", False)
                            and getattr(getattr(_RKMOD, "_rk", None),
                                        "sim_custom_batch", None) is not None)))
            except Exception:
                _rust_used = False
            device_label = (tr("CPU (🦀 Rust)", "CPU（🦀 Rust）") if _rust_used
                            else tr("CPU (numpy)", "CPU（numpy）"))
        # Build per-axis sweeps then meshgrid → cartesian product
        sweep_disps  : list[np.ndarray] = []
        sweep_sis    : list[np.ndarray] = []
        slider_specs_out: list[dict] = []
        for spec in selected_specs:
            key, label, scale = spec[0], spec[1], spec[2]
            unit = spec[3] if len(spec) > 3 else ""
            fmt  = spec[4] if len(spec) > 4 else "%.4g"
            kp   = f"slprev_pl_{topo_key}_{key}_{fname}"
            mn = float(st.session_state[f"{kp}_min"])
            mx = float(st.session_state[f"{kp}_max"])
            nf = int(st.session_state[f"{kp}_frames"])
            if mx <= mn:
                mx = mn + abs(mn) * 1e-6 + 1e-9
            sd = np.linspace(mn, mx, nf)
            sweep_disps.append(sd)
            sweep_sis.append(sd / scale)
            slider_specs_out.append(dict(
                key=key, label=label, unit=unit, fmt=fmt,
                values_disp=sd.tolist()))
        meshes = np.meshgrid(*sweep_sis, indexing="ij")
        flats  = [m.ravel() for m in meshes]
        n_total = int(flats[0].size) if flats else 0

        # Spinner so the user sees that compute is happening (the Plotly
        # figure below stays "stale" — the prior build — until the new
        # batch finishes and we replace it).  For very large sweeps the
        # chunked path avoids GPU OOM by simulating in slabs of 5 000.
        t0 = _time.perf_counter()
        with st.spinner(tr(f"Computing {n_total} frames on {device_label}…",
                           f"正在 {device_label} 上計算 {n_total} 幀…")):
            p_batch = dict(all_p)
            for spec, flat in zip(selected_specs, flats):
                p_batch[spec[0]] = xp.asarray(flat, dtype=float)
            try:
                S_b = _chunked_simulate_batch_to_host(
                    model_cls, p_batch, freq, z0, xp=xp)
            except Exception as e:
                st.error(tr(f"Batched preview simulation failed: {e}",
                            f"批次預覽模擬失敗：{e}"))
                return
        elapsed = _time.perf_counter() - t0

        if not np.all(np.isfinite(S_b)):
            st.warning(tr("Some frames contain non-finite S-parameters — "
                          "narrow the ranges to avoid singular combinations.",
                          "部分幀包含非有限的 S 參數 — 請縮小範圍以避免奇異組合。"))

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
        st.info(tr("Click **🧮 Build animation** to compute frames for the "
                   "Plotly slider(s).",
                   "點擊 **🧮 建立動畫** 以計算 Plotly 滑桿的幀。"))
        return

    cached_keys  = state.get("selected_keys", [])
    current_keys = [s[0] for s in selected_specs]
    if cached_keys != current_keys:
        st.warning(tr("Selection changed since last build "
                      f"(cached: {cached_keys}, current: {current_keys}).  "
                      "Click **🧮 Build animation** to refresh.",
                      f"自上次建立以來選擇已變更"
                      f"（快取：{cached_keys}，目前：{current_keys}）。"
                      "請點擊 **🧮 建立動畫** 以更新。"))
        return

    if state.get("S_batch") is None:
        # e.g. a stale build from the removed server-cached/int16 path.
        st.warning(tr("Cached build is unusable — click **🧮 Build animation** "
                      "to refresh.",
                      "快取的建構結果無法使用 — 請點擊 **🧮 建立動畫** 以更新。"))
        return

    # Device + elapsed banner so the user can confirm GPU vs CPU.
    elapsed = float(state.get("elapsed_s", 0.0))
    n_total = int(state.get("n_total", 0)) or len(state["S_batch"])
    device  = str(state.get("device", "?"))
    ms_each = (elapsed / max(1, n_total)) * 1000.0
    st.caption(tr(
        f"✅ Built **{n_total}** frames on **{device}** in "
        f"**{elapsed:.2f} s** ({ms_each:.1f} ms/frame).  "
        "Drag any slider below to scrub.",
        f"✅ 已在 **{device}** 上建立 **{n_total}** 幀，耗時 "
        f"**{elapsed:.2f} 秒**（{ms_each:.1f} 毫秒/幀）。"
        "拖曳下方任一滑桿即可瀏覽。"))

    payload = build_smith_bode_slider_payload(
        S_batch_joint=state["S_batch"],
        freq=freq,
        slider_specs=state["slider_specs"],
        model_name=model_cls.NAME,
        S_meas=S_raw,
        decimate_points=int(st.session_state.get(decim_key, decim_default)),
        # Mirror the Plotly Smith chart's per-trace display scale (set via
        # smith_scale_controls) so the slider's Smith view matches it.
        smith_mults={
            nm: float(st.session_state.get(
                f"smith_scale_{topo_key}_{nm}_{fname}", 1.0))
            for nm in ("S11", "S12", "S21", "S22")
        },
    )
    # Bidirectional component: renders the Smith+Bode figure with client-side
    # scrub sliders (the per-frame data is inflated in the browser from a
    # gzip+base64 blob via native DecompressionStream, so no multi-MB HTML
    # string is re-embedded on every rerun) plus a "✅ Use these values" button
    # that posts the chosen slider indices back.  Height is self-measured by
    # the component so the button is never clipped.
    n_sl   = len(state["slider_specs"])
    height = 500 + 30 + 36 * n_sl + 52
    ret = smith_bode_slider(
        payload=payload, height=height,
        use_label=tr("✅ Use these values", "✅ 使用這些數值"),
        key=f"sbslider_{topo_key}_{fname}")

    # Commit chosen slider values into the fine-tune sim_* override fields —
    # same target keys + display-unit convention as the Live-mode commit
    # above (`sim_{topo_key}_{k}_{fname}` = value × scale).  values_disp is
    # already in display units, so it's written straight through.  A nonce
    # guard stops the persisted component value from re-committing on every
    # rerun.
    if isinstance(ret, dict) and ret.get("nonce") is not None:
        _nonce_key = f"_sbslider_nonce_{topo_key}_{fname}"
        if ret["nonce"] != st.session_state.get(_nonce_key):
            st.session_state[_nonce_key] = ret["nonce"]
            idxs = ret.get("indices") or []
            for i, sp in enumerate(state["slider_specs"]):
                if i < len(idxs):
                    vals = sp.get("values_disp", [])
                    ii = int(idxs[i])
                    if 0 <= ii < len(vals):
                        st.session_state[
                            f"sim_{topo_key}_{sp['key']}_{fname}"] = float(vals[ii])
            st.rerun()


def render_visual_tuning_expander(model_cls, all_p, S_raw, freq, z0,
                                   tuning_specs, fname, topo_key):
    """🎚️ Visual Tuning expander — drag sliders to see the model react.

    Strictly UI-only: writes into the same fine-tune ``sim_*`` session
    keys via the ✅ commit button.  Auto-save is suppressed for slider
    commits (see ``render_override_and_smith``).
    """
    with st.container(key="hbt_exp_tune_vis_" + topo_key), \
         st.expander(tr("🎚️ Visual Tuning", "🎚️ 視覺化調諧"), expanded=False):
        # ── Backend status badges — show ALL active accelerators.
        #    When both CUDA and Rust are available, the Visual Tuning
        #    Live mode picks CUDA via `xp=cupy`, but the Plotly slider
        #    "Build animation" path (cache-less batched sim) can route
        #    through Rust.  Showing both lets the user know what's
        #    available, not just which one wins per click.
        from ...helpers.rust_kernels import (
            HAS_RUST as _HAS_RUST_BACKEND,
            rust_diagnostic as _rust_diag,
        )
        _chips = []
        if _HAS_CUDA:
            _chips.append(
                "<span style='background:#e3f2fd;color:#0d47a1;"
                f"padding:2px 8px;border-radius:4px;font-size:0.8em;"
                f"font-weight:600'>⚡ CUDA (cupy {_CUDA_VER})</span>")
        if _HAS_RUST_BACKEND:
            _chips.append(
                "<span style='background:#fff3e0;color:#e65100;"
                "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                "font-weight:600'>🦀 Rust kernels</span>")
        if not _chips:
            _chips.append(
                "<span style='background:#eceff1;color:#37474f;"
                "padding:2px 8px;border-radius:4px;font-size:0.8em;"
                "font-weight:600'"
                f" title='{tr('Build the Rust crate for ~10x speedup.'
                             ' See tools/rf/ssm/rust_kernels/README.md.',
                             '建置 Rust crate 可加速約 10 倍。'
                             '詳見 tools/rf/ssm/rust_kernels/README.md。')}'"
                f">🐢 {tr('NumPy fallback', 'NumPy 備援')}</span>")
        st.markdown(
            tr("Backends", "運算後端") + ": " + "  ".join(_chips),
            unsafe_allow_html=True)

        # ── Diagnostic: when the binary exists on disk but Rust didn't
        #    load, surface the actual import error inside an expander so
        #    the user doesn't have to dig through the terminal.
        if not _HAS_RUST_BACKEND:
            _d = _rust_diag()
            if _d["binary_files"] and not _d["force_numpy"]:
                with st.expander(tr("🛠️ Why is Rust not active?",
                                    "🛠️ 為什麼 Rust 未啟用？"), expanded=False):
                    st.code(
                        f"arch_tag       : {_d['arch_tag']}\n"
                        f"bin_dir        : {_d['bin_dir']}\n"
                        f"bin_dir_exists : {_d['bin_dir_exists']}\n"
                        f"binary_files   : {_d['binary_files']}\n"
                        f"import_error   : {_d['import_error']}\n"
                        f"force_numpy    : {_d['force_numpy']}\n",
                        language="text")
                    st.caption(tr(
                        "A binary exists on disk but couldn't be imported. "
                        "Most common cause: the Streamlit process was started "
                        "*before* the binary was placed in this folder — "
                        "restart the launcher (`python LAUNCH_Tool.py`) to "
                        "pick it up.  If the import error persists after a "
                        "fresh restart, the .pyd may be from a different ABI; "
                        "delete it and run `python dev/build_rust_kernels.py`.",
                        "磁碟上存在二進位檔，但無法匯入。"
                        "最常見的原因：Streamlit 程序是在二進位檔放入此資料夾"
                        "*之前*啟動的 — 請重新啟動啟動器"
                        "（`python LAUNCH_Tool.py`）以載入。"
                        "若重新啟動後匯入錯誤仍持續發生，"
                        "該 .pyd 可能來自不同的 ABI；"
                        "請刪除後執行 `python dev/build_rust_kernels.py`。"))
        _render_slider_preview(model_cls, all_p, S_raw, freq, z0,
                               tuning_specs, fname, topo_key)
