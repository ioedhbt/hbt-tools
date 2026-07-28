"""
models/tuning/drivers_progressive.py — The 🪜 Full Auto Tune driver
(_run_progressive): coarse global scan from physics-informed ranges, then
pair-wise (Zbe / Zbc / …) refinement with shrinking steps, plus an escape
phase that re-widens box-pinned parameters when refinement stalls.

Hoisted out of render_tuning_expander for the same reason as
drivers_sweep.py — see that module's docstring.

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).
"""
from __future__ import annotations
import gc
import time
import numpy as np
import streamlit as st

from tools.common.mem_budget import ram_available_bytes
from tools.common.i18n import tr

from ..residuals import _port_residuals_batch
from ._cuda_env import _cp, _HAS_CUDA
from .ranges import (_canonical_tune_key, tune_hard_limits,
                      informed_default_range, TUNE_DEFAULT_RANGES)
from .preview import _multi_metric_top_n
from .helpers import _best_summary_md
from .progressive_geometry import (
    _PROG_GLOBAL_BUDGET_CPU, _PROG_GLOBAL_BUDGET_GPU,
    _PROG_GROUP_BUDGET_CPU, _PROG_GROUP_BUDGET_GPU,
    _PROG_CHUNK_CPU, _PROG_CHUNK_GPU, _PROG_EDGE_FRAC,
    _PROG_MAX_CYCLES, _PROG_MIN_STEP,
    _PROG_PORT_GOAL, _PROG_STALL_CYCLES, _PROG_STALL_REL, _PROG_ESCAPE_PTS,
    _PROG_GROUPS, _prog_axis_values, _prog_next_bounds,
)

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


def _run_progressive(model_cls, all_p, S_raw, freq, z0, param_rows, tuning_specs, topo_key, fname, low_perf, *, use_cuda: bool, fit_keys: list,
                     target_pct: float):
    """Coarse → fine Full Auto Tune (progressive refinement).

    A global coarse scan over physics-informed ranges seeds a set of
    pair-wise (Zbe / Zbc / …) refinement cycles with shrinking boxes.
    Every candidate is memoised (quantised to floor/2) so no combo is
    ever re-evaluated; the best result is kept regardless of how the
    run ends (step floor / target residual / ⏹ Stop).  Selected
    parameters carry a strictly-positive lower limit (max(hard_lo,
    step floor)) so the fit can never propose 0 — deselecting a
    parameter pins it at its current value (incl. 0) instead.

    ``target_pct`` is kept in the signature for programmatic use; the
    UI card no longer exposes it and dispatches with 0.0 (= refine
    until the step floor or ⏹ Stop).

    Closes over param_rows / all_p / tuning_specs / S_raw / freq / z0 /
    model_cls / topo_key / fname / _low_perf.  Structured around the
    module-level pure helpers (_prog_axis_values / _prog_signature /
    _prog_next_bounds) so the geometry is unit-testable.
    """
    import pandas as pd

    xp = _cp if (use_cuda and _HAS_CUDA) else np
    if use_cuda and not _HAS_CUDA:
        st.warning(tr("CUDA not available — falling back to CPU.",
                      "CUDA 不可用 — 已回退至 CPU。"))
    if not hasattr(model_cls, "simulate_batch"):
        st.error(tr(
            f"{model_cls.__name__} has no simulate_batch — "
            "Full Auto Tune needs it.",
            f"{model_cls.__name__} 沒有 simulate_batch — "
            "全自動調諧需要它。"))
        return
    fit_keys = [k for k in fit_keys if k in
                {s[0] for s in tuning_specs}]
    if not fit_keys:
        st.warning(tr(
            "Full Auto Tune needs at least one parameter "
            "selected under **Parameters to fit**.",
            "全自動調諧至少需要在**要擬合的參數**下選取一個參數。"))
        return

    _global_budget = (_PROG_GLOBAL_BUDGET_GPU if xp is not np
                      else _PROG_GLOBAL_BUDGET_CPU)
    _group_budget  = (_PROG_GROUP_BUDGET_GPU if xp is not np
                      else _PROG_GROUP_BUDGET_CPU)
    if xp is not np:
        _chunk_max = _PROG_CHUNK_GPU
    else:
        # CPU: derive the chunk size from RAM actually available right
        # now instead of the fixed _PROG_CHUNK_CPU constant — same
        # per-row working-set model as _run_one_sweep's
        # per_combo_bytes. At high freq-point counts the fixed constant
        # could allocate several GB in one simulate_batch call and
        # get SIGKILLed by the Streamlit Cloud cgroup OOM-killer
        # before the try/except MemoryError path below ever runs.
        per_row_bytes = 16 * 4 * len(freq) * 12
        _chunk_max = max(256, min(_PROG_CHUNK_CPU,
                          int(ram_available_bytes() * 0.25 // per_row_bytes)))

    # Per-fit-key metadata: label / scale / hard limits / initial box /
    # step floor.  Order = fit_keys order.
    spec_by_key = {s[0]: s for s in tuning_specs}
    n_fit  = len(fit_keys)
    labels, scales = [], []
    bounds  = {}    # key -> [lo, hi] (display units), mutated per cycle
    floors  = []    # per fit key step floor
    pos_floors = []  # per fit key strictly-positive lower limit
    hard_lims = {}
    for k in fit_keys:
        s = spec_by_key[k]
        lbl   = s[1]
        scale = s[2]
        sstep = s[5] if len(s) > 5 else None
        labels.append(lbl)
        scales.append(scale)
        cur_disp = float(all_p.get(k, 0.0)) * scale
        lo, _st, hi = informed_default_range(
            k, lbl, cur_disp, low_perf=low_perf, spec_step=sstep)
        # A parasitic pinned to (0,0,0) can't be fit — give it a tiny
        # informed box anyway so the axis has room to move.
        if hi <= lo:
            _dlo, _dhi = TUNE_DEFAULT_RANGES.get(
                _canonical_tune_key(k, lbl) or "", (0.0, 1.0))
            lo, hi = float(_dlo), float(_dhi)
        floor_val = (float(sstep) if sstep and sstep > 0
                     else _PROG_MIN_STEP)
        floors.append(floor_val)
        h_lo, h_hi = tune_hard_limits(k, lbl)
        hard_lims[k] = (h_lo, h_hi)
        # No-zero floor: a *selected* parameter is never proposed at 0
        # (a 0-valued Cbe/tau just compensates degenerately elsewhere).
        # Keys whose hard lower limit is 0 get one step floor (>0) as
        # the effective lower limit; alpha0's 0.95 is already above.
        # Deselected params stay pinned at their current value —
        # including 0 — that's the user's escape hatch.
        pos_floor = max(float(h_lo) if h_lo is not None else 0.0,
                        floor_val)
        pos_floors.append(pos_floor)
        lo = max(float(lo), pos_floor)
        if hi <= lo:
            # Floor swallowed the box (tiny decade range) — keep a
            # minimal viable span above the floor.
            hi = lo + 2.0 * floor_val
            if h_hi is not None:
                hi = min(hi, float(h_hi))
        bounds[k] = [float(lo), float(hi)]
    floors_arr = np.asarray(floors, dtype=np.float64)

    # ── Result table plumbing (mirrors _run_one_sweep / _run_nm) ──
    TOP_K = 100
    col_names = [
        "Total Residual (%)", "S11 (%)", "S12 (%)", "S21 (%)", "S22 (%)"]
    for r in param_rows:
        u = r["unit"]; lbl = r["label"]
        col_names.append(f"{lbl} ({u})" if u else lbl)
    sess_key = f"tune_df_{topo_key}_{fname}"
    top_rows: list = []

    def _push_topk(row: list):
        if len(top_rows) < TOP_K:
            top_rows.append(row)
            top_rows.sort(key=lambda r: r[0])
            return
        if row[0] < top_rows[-1][0]:
            top_rows[-1] = row
            top_rows.sort(key=lambda r: r[0])

    def _persist_now():
        if not top_rows:
            return
        try:
            st.session_state[sess_key] = pd.DataFrame(
                _multi_metric_top_n(np.asarray(top_rows, dtype=float),
                                     per_metric=10),
                columns=col_names)
        except Exception:
            pass

    # ── Refinement groups: map canonical groups onto fit keys, then
    #    chunk leftovers into pairs (spec order). ──────────────────
    _fit_set = list(fit_keys)
    _assigned = set()
    groups: list = []
    for _gname, _gkeys in _PROG_GROUPS:
        members = [k for k in _fit_set
                   if _canonical_tune_key(k, spec_by_key[k][1]) in _gkeys
                   and k not in _assigned]
        if members:
            groups.append((_gname, members))
            _assigned.update(members)
    _leftover = [k for k in _fit_set if k not in _assigned]
    for _i in range(0, len(_leftover), 2):
        groups.append(("misc", _leftover[_i:_i + 2]))

    # ── State / memo / UI ─────────────────────────────────────────
    memo: set = set()
    best_disp = {k: float(all_p.get(k, 0.0)) * scales[i]
                 for i, k in enumerate(fit_keys)}
    best_tot = [float("inf")]
    n_eval = [0]
    n_skip = [0]

    ui_cols = st.columns([5, 1])
    with ui_cols[0]:
        progress = st.progress(
            0, text=tr("Full Auto Tune…", "全自動調諧…"))
    with ui_cols[1]:
        stop_box = st.empty()
    best_box = st.empty()
    stop_key = f"tune_stop_{topo_key}_{fname}_prog"
    stop_box.button(
        tr("⏹ Stop", "⏹ 停止"), key=stop_key, type="secondary",
        help=tr("Stop the calculation. Best result so far is kept.",
                "停止計算。目前的最佳結果將被保留。"))

    last_ui = [0.0]
    _t_start = time.time()
    S_mea_dev = xp.asarray(S_raw)

    def _release():
        if xp is np:
            gc.collect(); gc.collect(); return
        try:
            _cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass
        gc.collect(); gc.collect()
        try:
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    _PORT_NAMES = ("S11", "S12", "S21", "S22")

    def _worst_port():
        """(name, value) of the worst per-port residual on the current
        best row — inf before any result exists."""
        if not top_rows:
            return _PORT_NAMES[0], float("inf")
        vals = top_rows[0][1:5]
        i = int(np.argmax(vals))
        return _PORT_NAMES[i], float(vals[i])

    def _ui_tick(phase, cyc, frac):
        if time.time() - last_ui[0] < 0.5:
            return
        last_ui[0] = time.time()
        _txt = (f"Full Auto… {phase} | cycle {cyc} | "
                f"{n_eval[0]:,} evals ({n_skip[0]:,} memo-skipped) | "
                f"best Total {best_tot[0]:.2f}%")
        if top_rows:
            _wn, _wv = _worst_port()
            _txt += f" | worst {_wn} {_wv:.2f}%"
        # Any st.* call is the Stop-button cancellation point.
        progress.progress(min(1.0, max(0.0, float(frac))), text=_txt)
        if top_rows:
            best_box.markdown(
                _best_summary_md(pd.Series(top_rows[0], index=col_names),
                                 tuning_specs, topo_key, fname,
                                 label=tr("Best so far", "目前最佳")),
                unsafe_allow_html=True)
        _persist_now()

    def _row_for(cand_disp):
        """Assemble a (5 + n_params) result-table row from a fitted
        candidate (display units) + the constant params."""
        cand = {k: float(v) for k, v in zip(fit_keys, cand_disp)}
        row = [0.0, 0.0, 0.0, 0.0, 0.0]
        for pr in param_rows:
            k = pr["key"]
            if k in cand:
                row.append(cand[k])
            else:
                row.append(float(all_p.get(k, 0.0)) * pr["scale"])
        return row

    def _eval_block(cands, phase, cyc):
        """Evaluate a (B, n_fit) host array of display-unit candidates.

        Filters memoised rows, chunks the survivors through
        simulate_batch, scores residuals, and pushes into the top-K.
        Returns the number of *new* (non-skipped) rows evaluated.
        """
        cands = np.asarray(cands, dtype=np.float64)
        if cands.ndim != 2 or cands.shape[0] == 0:
            return 0
        B = cands.shape[0]
        # Vectorised signature buckets, then a python loop to filter.
        buckets = np.round(cands / (floors_arr * 0.5)).astype(np.int64)
        survivors = []
        for bi in range(B):
            sig = tuple(int(x) for x in buckets[bi])
            if sig in memo:
                n_skip[0] += 1
                continue
            memo.add(sig)
            survivors.append(cands[bi])
        if not survivors:
            return 0
        surv = np.asarray(survivors, dtype=np.float64)
        n_new = surv.shape[0]

        start = 0
        chunk = _chunk_max
        while start < surv.shape[0]:
            sub = surv[start:start + chunk]
            Bc = sub.shape[0]
            try:
                p = dict(all_p)
                for j, k in enumerate(fit_keys):
                    p[k] = xp.asarray(sub[:, j] / scales[j],
                                      dtype=np.float64)
                S = model_cls.simulate_batch(p, freq, z0, xp=xp)
                S_flat = xp.asarray(S).reshape(Bc, len(freq), 2, 2)
                res = _port_residuals_batch(S_mea_dev, S_flat, xp)
                if xp is np:
                    tot = np.asarray(res["Total"], dtype=float)
                    s11 = np.asarray(res["S11"], dtype=float)
                    s12 = np.asarray(res["S12"], dtype=float)
                    s21 = np.asarray(res["S21"], dtype=float)
                    s22 = np.asarray(res["S22"], dtype=float)
                else:
                    tot = _cp.asnumpy(res["Total"])
                    s11 = _cp.asnumpy(res["S11"])
                    s12 = _cp.asnumpy(res["S12"])
                    s21 = _cp.asnumpy(res["S21"])
                    s22 = _cp.asnumpy(res["S22"])
            except Exception as exc:
                _is_oom = (isinstance(exc, MemoryError)
                           or "out of memory" in str(exc).lower()
                           or "OutOfMemoryError" in type(exc).__name__)
                if _is_oom and chunk > 1:
                    # Halve the chunk and retry from the same offset.
                    chunk = max(1, chunk // 2)
                    _release()
                    continue
                if _is_oom:
                    st.error(tr(
                        "Out of memory — a single candidate row "
                        "won't fit. Aborting Full Auto Tune.",
                        "記憶體不足 — 單一候選列都無法容納。"
                        "正在中止全自動調諧。"))
                    return n_new
                raise
            for _a in (tot, s11, s12, s21, s22):
                _a[~np.isfinite(_a)] = np.inf
            # Only the chunk's TOP_K lowest totals can possibly enter
            # the (Total-sorted) global top-K, so prefilter with one
            # argsort instead of building a Python row per candidate —
            # keeps the host loop negligible even for 262k-row GPU
            # chunks.
            n_eval[0] += Bc
            for ri in np.argsort(tot)[:TOP_K]:
                if not np.isfinite(tot[ri]):
                    break        # sorted → the rest are inf too
                row = _row_for(sub[ri])
                row[0] = float(tot[ri]); row[1] = float(s11[ri])
                row[2] = float(s12[ri]); row[3] = float(s21[ri])
                row[4] = float(s22[ri])
                _push_topk(row)
                if row[0] < best_tot[0]:
                    best_tot[0] = row[0]
                    for j, k in enumerate(fit_keys):
                        best_disp[k] = float(sub[ri, j])
            start += Bc
            _ui_tick(phase, cyc, min(1.0, start / max(1, surv.shape[0])))
        return n_new

    # ── Cartesian grid builder within current bounds ───────────────
    def _grid(keys, n_pts_per):
        """Cartesian product over ``keys`` axis value-lists (current
        bounds, best value inserted, floor-limited).  Non-listed fit
        keys are pinned at their current best.  Returns (B, n_fit)."""
        axis_vals = []
        for k in fit_keys:
            if k in keys:
                i = fit_keys.index(k)
                lo, hi = bounds[k]
                axis_vals.append(_prog_axis_values(
                    lo, hi, n_pts_per, floors[i],
                    include=best_disp[k]))
            else:
                axis_vals.append(np.array([best_disp[k]],
                                          dtype=np.float64))
        mesh = np.meshgrid(*axis_vals, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)

    # ── Escape pass — wide re-scan of box-pinned params ────────────
    def _escape_pass(cyc):
        """One wide log re-scan for every fitted param pinned at a box
        edge (all fit keys when none are pinned): 1-D over a much
        wider range, then 2-D with the first group partner so
        compensating pairs can move together.  Each scanned param's
        box is re-widened so later cycles keep exploring the
        discovery.  All work goes through _eval_block (memo / top-K /
        UI tick / ⏹ Stop).  Returns True when best_tot improved."""
        _before = best_tot[0]
        pinned = []
        for i, k in enumerate(fit_keys):
            lo, hi = bounds[k]
            span = hi - lo
            if (best_disp[k] <= lo + 3.0 * floors[i]
                    or best_disp[k] >= hi - _PROG_EDGE_FRAC * span):
                pinned.append(k)
        if not pinned:
            pinned = list(fit_keys)
        for k in pinned:
            i = fit_keys.index(k)
            s = spec_by_key[k]
            sstep = s[5] if len(s) > 5 else None
            _ilo, _istep, inf_hi = informed_default_range(
                k, s[1], best_disp[k],
                low_perf=low_perf, spec_step=sstep)
            wide_hi = max(float(inf_hi), bounds[k][1],
                          abs(best_disp[k]) * 100.0,
                          100.0 * floors[i])
            _h_hi = hard_lims[k][1]
            if _h_hi is not None:
                wide_hi = min(wide_hi, float(_h_hi))
            # 1-D wide scan — log spacing kicks in automatically for
            # the many-decade span; others pinned at the current best.
            axis_w = _prog_axis_values(
                pos_floors[i], wide_hi, _PROG_ESCAPE_PTS,
                floors[i], include=best_disp[k])
            base = np.asarray([best_disp[kk] for kk in fit_keys],
                              dtype=np.float64)
            cands = np.tile(base, (len(axis_w), 1))
            cands[:, i] = axis_w
            _eval_block(cands, f"escape {k}", cyc)
            # 2-D with the FIRST other member of k's group present in
            # the fit scope — compensating pairs move together.
            partner = None
            for _gn, members in groups:
                if k in members:
                    partner = next(
                        (m for m in members if m != k), None)
                    break
            if partner is not None:
                j = fit_keys.index(partner)
                ax_k = _prog_axis_values(
                    pos_floors[i], wide_hi, 24, floors[i],
                    include=best_disp[k])
                ax_p = _prog_axis_values(
                    bounds[partner][0], bounds[partner][1], 12,
                    floors[j], include=best_disp[partner])
                mk, mp = np.meshgrid(ax_k, ax_p, indexing="ij")
                cands2 = np.tile(base, (mk.size, 1))
                cands2[:, i] = mk.ravel()
                cands2[:, j] = mp.ravel()
                _eval_block(cands2, f"escape {k}", cyc)
            # Re-widen so later cycles keep exploring the discovery.
            bounds[k] = [pos_floors[i], wide_hi]
        return best_tot[0] < _before - 1e-15

    cancelled = False
    try:
        # 0 — evaluate the current point first (baseline in the table).
        _eval_block(np.asarray([[best_disp[k] for k in fit_keys]],
                               dtype=np.float64), "baseline", 0)

        # 1 — GLOBAL coarse pass.
        _n_pts_global = 3
        for _cand in (5, 4, 3):
            if _cand ** n_fit <= _global_budget:
                _n_pts_global = _cand
                break
        if 3 ** n_fit > _global_budget:
            # Too many dims for even a 3-point full cartesian — draw
            # `budget` random combos from per-axis 3-point lists.
            rng = np.random.default_rng(0)
            axis3 = []
            for i, k in enumerate(fit_keys):
                lo, hi = bounds[k]
                axis3.append(_prog_axis_values(
                    lo, hi, 3, floors[i], include=best_disp[k]))
            cols_rand = [rng.choice(a, size=_global_budget) for a in axis3]
            _eval_block(np.stack(cols_rand, axis=1), "global", 0)
        else:
            _eval_block(_grid(set(fit_keys), _n_pts_global), "global", 0)

        # 2 — cycle loop, with an automatic escape phase.
        #
        # `stall` counts consecutive low-improvement cycles.  While
        # any port residual is above _PROG_PORT_GOAL, a stall (or a
        # fully-memoised cycle) triggers ONE escape: box-pinned
        # params get a wide log re-scan (alone + with a group
        # partner) and re-widened bounds; a fruitless escape falls
        # back to a single global random re-scan.  Only after that
        # may the normal termination paths end the run above the
        # goal.
        _converged = False
        stall = 0
        escape_spent = False
        for cyc in range(1, _PROG_MAX_CYCLES + 1):
            _prev_best = best_tot[0]
            _cycle_new = 0
            for _gname, members in groups:
                _k = len(members)
                if _k == 0:
                    continue
                _pts = max(3, int(round(_group_budget ** (1.0 / _k))))
                _cycle_new += _eval_block(
                    _grid(set(members), _pts), _gname, cyc)

            # Shrink / expand each axis around its best.  The no-zero
            # floor doubles as the effective hard lower limit so an
            # edge-expansion can never re-open the box down to 0.
            for i, k in enumerate(fit_keys):
                lo, hi = bounds[k]
                h_lo, h_hi = hard_lims[k]
                h_lo_eff = max(float(h_lo) if h_lo is not None else 0.0,
                               pos_floors[i])
                bounds[k] = list(_prog_next_bounds(
                    lo, hi, best_disp[k], h_lo_eff, h_hi, floors[i]))

            # Stall tracking + worst-port status.
            _rel_impr = ((_prev_best - best_tot[0])
                         / max(_prev_best, 1e-9))
            stall = 0 if _rel_impr > _PROG_STALL_REL else stall + 1
            _wname, _wval = _worst_port()

            # Termination: explicit target reached.
            if target_pct > 0 and best_tot[0] <= target_pct:
                _converged = True
                break

            # Escape phase — refinement is stuck above the per-port
            # goal (stalled, or the grids collapsed onto memoised
            # points) and the escape hasn't been spent yet.
            if (not escape_spent and _wval > _PROG_PORT_GOAL
                    and (stall >= _PROG_STALL_CYCLES
                         or _cycle_new == 0)):
                if _escape_pass(cyc):
                    stall = 0
                    continue      # keep cycling from the discovery
                # Escape found nothing — one global random re-scan
                # over the (re-widened) bounds, then let the normal
                # termination paths end the run.
                escape_spent = True
                rng_esc = np.random.default_rng(1)
                axis5 = []
                for i, k in enumerate(fit_keys):
                    lo, hi = bounds[k]
                    axis5.append(_prog_axis_values(
                        lo, hi, 5, floors[i]))
                _n_rand = max(1, _global_budget // 4)
                cols_esc = [rng_esc.choice(a, size=_n_rand)
                            for a in axis5]
                _before_r = best_tot[0]
                _eval_block(np.stack(cols_esc, axis=1),
                            "escape rescan", cyc)
                if best_tot[0] < _before_r - 1e-15:
                    stall = 0
                continue

            if _cycle_new == 0:
                # Planned cycle produced no new evaluations and no
                # escape is available (spent, or the per-port goal
                # is met).  Stop.
                _converged = True
                break
            _all_at_floor = all(
                (bounds[k][1] - bounds[k][0]) <= 2.0 * floors[i] + 1e-30
                for i, k in enumerate(fit_keys))
            if (_all_at_floor and _rel_impr < 1e-3
                    and (_wval <= _PROG_PORT_GOAL or escape_spent)):
                _converged = True
                break

        _persist_now()
        progress.empty(); best_box.empty(); stop_box.empty()
        st.session_state[f"tune_elapsed_{topo_key}_{fname}"] = (
            time.time() - _t_start)
        _wname, _wval = _worst_port()
        if _wval <= _PROG_PORT_GOAL:
            st.success(tr(
                f"Full Auto Tune done — {n_eval[0]:,} evals, "
                f"best Total = {best_tot[0]:.2f}% "
                f"(all ports ≤ {_PROG_PORT_GOAL:.0f}%)",
                f"全自動調諧完成 — {n_eval[0]:,} 次評估，"
                f"最佳總計 = {best_tot[0]:.2f}%"
                f"（所有埠 ≤ {_PROG_PORT_GOAL:.0f}%）"))
        else:
            st.info(tr(
                f"Full Auto Tune stopped above the per-port goal — "
                f"best kept (worst {_wname} {_wval:.2f}%, Total "
                f"{best_tot[0]:.2f}%, {n_eval[0]:,} evals). "
                f"Re-running Evaluate continues from the best values "
                f"after 🏆 Use best values.",
                f"全自動調諧已停止於高於各埠目標之處 — "
                f"已保留最佳結果（最差 {_wname} {_wval:.2f}%，總計 "
                f"{best_tot[0]:.2f}%，{n_eval[0]:,} 次評估）。"
                f"按下🏆使用最佳值後，重新執行評估將由最佳值繼續。"))
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
            try:
                S_mea_dev = None
            except Exception:
                pass
            _release()
            raise
        st.error(tr(f"Full Auto Tune failed: {exc!r}",
                    f"全自動調諧失敗：{exc!r}"))
    finally:
        _persist_now()
        try:
            S_mea_dev = None
        except Exception:
            pass
        _release()
