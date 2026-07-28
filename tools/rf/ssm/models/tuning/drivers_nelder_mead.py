"""
models/tuning/drivers_nelder_mead.py — The (currently UI-disabled, kept
wired up) Nelder-Mead local-search driver (_run_nelder_mead).

Hoisted out of render_tuning_expander for the same reason as
drivers_sweep.py — see that module's docstring.

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).
"""
from __future__ import annotations
import gc
import numpy as np
import streamlit as st

from tools.common.i18n import tr

from ..residuals import _port_residuals
from ._cuda_env import _cp, _HAS_CUDA
from .ranges import tune_hard_limits, _clamp_to_hard
from .preview import _multi_metric_top_n
from .helpers import _best_summary_md

try:
    from streamlit.runtime.scriptrunner.script_runner import RerunException as _RerunException
except Exception:
    try:
        from streamlit.runtime.scriptrunner import RerunException as _RerunException
    except Exception:
        _RerunException = None  # type: ignore[assignment]
try:
    from streamlit.runtime.scriptrunner.script_runner import StopException as _StopException
except Exception:
    try:
        from streamlit.runtime.scriptrunner_utils.exceptions import StopException as _StopException
    except Exception:
        _StopException = None  # type: ignore[assignment]


def _run_nelder_mead(model_cls, all_p, S_raw, freq, z0, param_rows, tuning_specs, topo_key, fname, *, max_iter: int, restart: bool,
                     use_cuda: bool = False):
    """Local-search optimisation using scipy's adaptive Nelder-Mead.

    x is in *display units* so the simplex moves at comparable
    magnitudes across mixed parameters (fF / pH / Ω).  Bounds come
    from the per-row Min/Max boxes; the seed is the current
    parameter value clipped into bounds.

    When ``use_cuda`` is True and CuPy is available, each objective
    evaluation uses ``simulate_vec(xp=cupy)`` and computes residuals
    on the GPU, with a single host sync per eval to read the scalar
    total back.  Helps when N_freq is large enough that the GPU
    outruns CPU even at batch=1.

    Top-K accumulator and session-state layout match _run_one_sweep
    so the existing display / "Use best values" button work
    unchanged.
    """
    try:
        from scipy.optimize import minimize as _nm_minimize
    except ImportError:
        st.error(tr(
            "scipy is required for Nelder-Mead Auto tuning. "
            "Install with `pip install scipy`.",
            "Nelder-Mead 自動調諧需要 scipy。"
            "請以 `pip install scipy` 安裝。"))
        return

    import pandas as pd

    # Only enabled rows participate in optimisation; constants stay put.
    swept_rows = [r for r in param_rows if r["enabled"]]
    if not swept_rows:
        st.warning(tr(
            "Auto needs at least one parameter with the **Sweep** "
            "checkbox enabled.",
            "自動調諧至少需要一個已啟用**掃描**核取方塊的參數。"))
        return

    sweep_keys_nm   = [r["key"]   for r in swept_rows]
    sweep_scales_nm = [r["scale"] for r in swept_rows]

    # Seed + bounds in display units.
    x0_disp = []
    bounds  = []
    for r in swept_rows:
        cur  = float(all_p.get(r["key"], 0.0)) * r["scale"]
        arr  = np.asarray(r["sweep"], dtype=np.float64)
        lo   = float(np.min(arr))
        hi   = float(np.max(arr))
        if hi <= lo:
            hi = lo + max(abs(lo), 1.0) * 0.5  # avoid degenerate box
        # Clip the box into the hard physical limits so the simplex
        # can never wander into negative components / out-of-range α₀.
        _h_lo, _h_hi = tune_hard_limits(r["key"], r["label"])
        lo, hi = _clamp_to_hard(lo, hi, _h_lo, _h_hi)
        if hi <= lo:
            hi = lo + max(abs(lo), 1.0) * 0.5
        cur  = max(lo, min(hi, cur))
        x0_disp.append(cur)
        bounds.append((lo, hi))
    x0      = np.asarray(x0_disp, dtype=np.float64)
    lo_arr  = np.asarray([b[0] for b in bounds], dtype=np.float64)
    hi_arr  = np.asarray([b[1] for b in bounds], dtype=np.float64)

    TOP_K = 100
    col_names = [
        "Total Residual (%)", "S11 (%)", "S12 (%)", "S21 (%)", "S22 (%)"]
    for r in param_rows:
        u   = r["unit"]
        lbl = r["label"]
        col_names.append(f"{lbl} ({u})" if u else lbl)
    sess_key = f"tune_df_{topo_key}_{fname}"

    # Top-K kept on host as a small max-heap-like sorted list.
    top_rows: list[list[float]] = []
    n_eval   = [0]

    def _push_topk(row: list[float]):
        # Keep top-K by total residual.  Linear insert is fine — TOP_K
        # is 100 and Nelder-Mead does <~10k evals total.
        if len(top_rows) < TOP_K:
            top_rows.append(row)
            top_rows.sort(key=lambda r: r[0])
            return
        if row[0] < top_rows[-1][0]:
            top_rows[-1] = row
            top_rows.sort(key=lambda r: r[0])

    use_vec = hasattr(model_cls, "simulate_vec")

    # ── CUDA hot path setup ───────────────────────────────────
    # When use_cuda is requested we move S_raw to device once and
    # run residuals there, doing a single .item() per eval to pull
    # the scalar total back to host for scipy.
    xp_dev = _cp if (use_cuda and _HAS_CUDA and use_vec) else np
    if use_cuda and not _HAS_CUDA:
        st.warning(tr("CUDA not available — falling back to CPU.",
                      "CUDA 不可用 — 已回退至 CPU。"))
    elif use_cuda and not use_vec:
        st.warning(tr(
            f"{model_cls.__name__} has no simulate_vec — "
            "CUDA path needs it; falling back to CPU.",
            f"{model_cls.__name__} 沒有 simulate_vec — "
            "CUDA 路徑需要它；已回退至 CPU。"))
    S_mea_dev = xp_dev.asarray(S_raw)
    _den_dev  = xp_dev.sum(xp_dev.abs(S_mea_dev) ** 2, axis=0)  # (2,2)

    def _residuals_dev(S_sim_dev):
        """Return (total, s11, s12, s21, s22) as Python floats.
        One host sync per call when xp_dev is cupy."""
        diff = S_mea_dev - S_sim_dev                            # (N,2,2)
        num  = xp_dev.sum(xp_dev.abs(diff) ** 2, axis=0)        # (2,2)
        den_s = xp_dev.where(_den_dev > 0, _den_dev, 1.0)
        val   = xp_dev.sqrt(num / den_s) * 100.0
        val   = xp_dev.where(_den_dev > 0, val, 0.0)
        s11 = val[0, 0]; s12 = val[0, 1]
        s21 = val[1, 0]; s22 = val[1, 1]
        tot = (s11 + s12 + s21 + s22) * 0.25
        if xp_dev is np:
            return (float(tot), float(s11), float(s12),
                    float(s21), float(s22))
        # Single fused host sync — pulls all five scalars in one shot.
        arr = xp_dev.stack([tot, s11, s12, s21, s22])
        arr_h = _cp.asnumpy(arr)
        return (float(arr_h[0]), float(arr_h[1]), float(arr_h[2]),
                float(arr_h[3]), float(arr_h[4]))

    def _objective(x_disp):
        # Hard-clip into bounds — scipy NM with bounds usually stays
        # inside but the simplex initialisation can poke out by a hair.
        x_disp = np.minimum(np.maximum(x_disp, lo_arr), hi_arr)
        p = dict(all_p)
        for k, sc, v in zip(sweep_keys_nm, sweep_scales_nm, x_disp):
            p[k] = float(v) / float(sc)
        try:
            if use_vec:
                S_sim = model_cls.simulate_vec(p, freq, z0, xp=xp_dev)
            else:
                S_sim = model_cls.simulate(p, freq, z0)
        except Exception:
            return 1.0e8
        if S_sim is None:
            return 1.0e8
        if xp_dev is np:
            r = _port_residuals(S_raw, S_sim)
            r_tot, r_s11, r_s12, r_s21, r_s22 = (
                r["Total"], r["S11"], r["S12"], r["S21"], r["S22"])
        else:
            r_tot, r_s11, r_s12, r_s21, r_s22 = _residuals_dev(S_sim)
        if not np.isfinite(r_tot):
            return 1.0e8
        # Repack to dict shape used downstream (top-K row build)
        r = {"Total": r_tot, "S11": r_s11, "S12": r_s12,
             "S21": r_s21, "S22": r_s22}
        # Build the (5 + n_params) row in the same column order as the
        # full-sweep dataframe.
        row = [r["Total"], r["S11"], r["S12"], r["S21"], r["S22"]]
        x_dict = dict(zip(sweep_keys_nm, x_disp))
        for pr in param_rows:
            k = pr["key"]
            if k in x_dict:
                row.append(float(x_dict[k]))
            else:
                row.append(float(all_p.get(k, 0.0)) * pr["scale"])
        _push_topk(row)
        n_eval[0] += 1
        return float(r["Total"])

    # ── UI placeholders (mirror the sweep UI) ─────────────────
    ui_cols = st.columns([5, 1])
    with ui_cols[0]:
        progress = st.progress(
            0, text=tr("Auto (Nelder-Mead)…", "自動（Nelder-Mead）…"))
    with ui_cols[1]:
        stop_box = st.empty()
    best_box = st.empty()
    stop_key = f"tune_stop_{topo_key}_{fname}_nm"
    stop_box.button(
        tr("⏹ Stop", "⏹ 停止"), key=stop_key, type="secondary",
        help=tr("Stop the calculation. Best results so far are kept.",
                "停止計算。目前的最佳結果將被保留。"))

    # NM does roughly maxiter * (n+1) evals worst case.
    ev_budget = max(1, max_iter * (len(x0) + 1))

    def _persist_now():
        if not top_rows:
            return
        df = pd.DataFrame(
            _multi_metric_top_n(np.asarray(top_rows, dtype=float),
                                 per_metric=10),
            columns=col_names)
        st.session_state[sess_key] = df

    def _callback(xk, *args, **kwargs):
        # Periodic progress + best-so-far card.  Streamlit's Stop
        # button works by raising RerunException on any st.* call,
        # so calling progress.progress() here also gives us a clean
        # cancellation point.
        if top_rows:
            best_series = pd.Series(top_rows[0], index=col_names)
            best_box.markdown(
                _best_summary_md(best_series, tuning_specs, topo_key, fname,
                                     label=tr("Best so far", "目前最佳")),
                unsafe_allow_html=True,
            )
        _auto_nm_word = tr("Auto (Nelder-Mead)", "自動（Nelder-Mead）")
        _evals_word = tr("evals", "次評估")
        if top_rows:
            _txt = (f"{_auto_nm_word}… {n_eval[0]} {_evals_word}  "
                    f"({tr('best Total', '最佳總計')}: "
                    f"{top_rows[0][0]:.3f}%)")
        else:
            _txt = f"{_auto_nm_word}… {n_eval[0]} {_evals_word}"
        progress.progress(min(1.0, n_eval[0] / ev_budget), text=_txt)
        _persist_now()

    cancelled = False
    options = {
        "maxiter":  int(max_iter),
        "maxfev":   int(max_iter) * (len(x0) + 1),
        "xatol":    1e-6,
        "fatol":    1e-4,
        "adaptive": True,
    }

    try:
        _nm_minimize(
            _objective, x0,
            method="Nelder-Mead",
            bounds=bounds,
            callback=_callback,
            options=options,
        )
        if restart and len(top_rows) > 0:
            # Random perturbation around current best (10% of box width
            # per axis) for a single second pass — catches premature
            # simplex collapse without doubling cost.
            best_disp = np.asarray(top_rows[0][5:5 + len(param_rows)],
                                   dtype=np.float64)
            # Pull just the swept components out of best_disp
            swept_idx = [i for i, pr in enumerate(param_rows)
                         if pr["enabled"]]
            x_seed = np.asarray(
                [best_disp[i] for i in swept_idx], dtype=np.float64)
            rng = np.random.default_rng(0)
            span = hi_arr - lo_arr
            x_seed = np.clip(
                x_seed + rng.uniform(-0.1, 0.1, size=x_seed.shape) * span,
                lo_arr, hi_arr)
            _nm_minimize(
                _objective, x_seed,
                method="Nelder-Mead",
                bounds=bounds,
                callback=_callback,
                options=options,
            )
    except BaseException as exc:
        _is_rerun = (_RerunException is not None
                     and isinstance(exc, _RerunException))
        _is_stop  = (_StopException is not None
                     and isinstance(exc, _StopException))
        _name = type(exc).__name__
        if _is_rerun or _is_stop or _name in (
                "RerunException", "StopException"):
            cancelled = True
            _persist_now()
            # Drop device buffers before re-raising so the next
            # rerun starts from a clean pool.
            if xp_dev is not np:
                try:
                    S_mea_dev = None
                    _cp.cuda.runtime.deviceSynchronize()
                    gc.collect()
                    _cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
            raise
        st.error(tr(f"Nelder-Mead failed: {exc!r}",
                    f"Nelder-Mead 執行失敗：{exc!r}"))
    finally:
        _persist_now()
        if xp_dev is not np:
            try:
                S_mea_dev = None
                gc.collect()
                _cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    progress.empty()
    best_box.empty()
    stop_box.empty()
    if not cancelled and top_rows:
        _mode = "CUDA" if xp_dev is not np else "CPU"
        st.success(tr(
            f"Auto tuning done ({_mode}) — {n_eval[0]} evals, "
            f"best Total = {top_rows[0][0]:.3f}%",
            f"自動調諧完成（{_mode}）— {n_eval[0]} 次評估，"
            f"最佳總計 = {top_rows[0][0]:.3f}%"))
