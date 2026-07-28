"""
models/tuning/drivers_sweep.py — The Brute-force / Optimized / Prioritized /
Minimize-deviation grid-sweep driver (_run_one_sweep).

Hoisted out of render_tuning_expander (see models/tuning/sweep.py) as a
module-level function: every one of its real inputs was already an
explicit closure read of render_tuning_expander's own parameters plus the
freshly-built ``param_rows`` — i.e. it was never coupled to widget-
instantiation ORDER, just to a large shared-state surface.  It now takes
that surface as explicit parameters instead.

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import streamlit as st

from ...helpers import build_Y_pad_batch, build_Z_ser_batch
from tools.common.mem_budget import ram_available_bytes
from tools.common.i18n import tr

from ..residuals import _port_residuals_batch
from ._cuda_env import _cp
from .preview import _fmt_eta, _multi_metric_top_n
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


def _run_one_sweep(model_cls, all_p, S_raw, freq, z0, param_rows, tuning_specs, topo_key, fname, *, use_cuda: bool,
                    sweep_lists_override: dict | None = None,
                    sort_metric: str = "Total",
                    phase_label: str = "",
                    phase_suffix: str = "",
                    dev_threshold: float | None = None,
                    res_threshold: float | None = None):
    """Run one full parameter-sweep pass.

    sweep_lists_override : dict[str, np.ndarray] | None
        When provided, overrides the per-row sweep arrays for the listed
        keys. Other rows fall back to their checkbox/min/step/max state.
    sort_metric : "Total" | "S11" | "S12" | "S21" | "S22"
        Selects the residual metric used to rank the top-K results.
        Total residual / per-port residuals are still recorded for
        every kept combo regardless of choice.
    phase_label / phase_suffix
        phase_label is shown in the progress text; phase_suffix is
        appended to the Stop-button key so multi-pass runs (e.g.
        Optimized) don't clash on a duplicate Streamlit widget key.
    dev_threshold : float | None
        Max allowed peak-to-peak (max-min) per-port residual deviation
        in %.  Combos exceeding this are pushed to the bottom of the
        ranking — surviving combos are ranked normally by sort_metric.
        Used by the "Minimize deviation" button to pick the lowest
        total residual *among balanced* combinations.
    res_threshold : float | None
        Max allowed per-port residual in %.  Combos with any port
        exceeding this are pushed to the bottom of the ranking.
    """
    xp = _cp if use_cuda else np
    has_batch = hasattr(model_cls, "simulate_batch")
    _mode_label = "CUDA" if use_cuda else "CPU"

    # Only the top-K combos (lowest residuals) are kept in memory.
    # Saves >99% RAM on huge sweeps and side-steps MemoryError.
    TOP_K = 100

    # Build sweep lists (display units) — short host arrays of unique
    # values per parameter (length 1 for unswept params).
    sweep_keys, sweep_lists = [], []
    sweep_scales, sweep_labels, sweep_units = [], [], []
    for row in param_rows:
        sweep_keys.append(row["key"])
        sweep_scales.append(row["scale"])
        sweep_labels.append(row["label"])
        sweep_units.append(row["unit"])
        if sweep_lists_override and row["key"] in sweep_lists_override:
            sweep_lists.append(np.asarray(
                sweep_lists_override[row["key"]], dtype=np.float64))
        elif row["enabled"]:
            sweep_lists.append(np.asarray(row["sweep"], dtype=np.float64))
        else:
            sweep_lists.append(np.array(
                [float(all_p.get(row["key"], 0.0)) * row["scale"]],
                dtype=np.float64))

    n_params = len(sweep_keys)
    L_list = [int(len(sl)) for sl in sweep_lists]
    n_total = 1
    for L in L_list:
        n_total *= L

    # Strides for "ij"-order linear→multi-index decomposition:
    #   combo k → (i_0, i_1, …, i_{n-1})
    #   where i_p = (k // strides[p]) % L_list[p]
    strides = [1] * n_params
    for i in range(n_params - 2, -1, -1):
        strides[i] = strides[i + 1] * L_list[i + 1]

    # Indices of params that actually vary — only these need to be
    # carried per-row in the top-K state. Constants get filled in at
    # display time from sweep_lists.
    swept_idx_full = [i for i in range(n_params) if L_list[i] > 1]
    n_swept = len(swept_idx_full)

    # ── N-D parameter-sweep layout ──────────────────────────────────
    # Each *swept* parameter gets its own broadcast axis.  Sub-networks
    # whose inputs are not all swept are computed at their *native*
    # dimensionality and broadcast across the rest of the grid — so
    # `h/i` in `g·h/i` is built once with shape `(1,Lh,Li,1)` and
    # reused for every value of `g`, with no recomputation.
    #
    # Layout convention:
    #   axis 0 .. n_swept_dims-1  → one length-L_i axis per swept param
    #   axis n_swept_dims         → frequency (length N_freq)
    # Constants stay as Python scalars (broadcast to anything).
    #
    # Output S has shape  (L_0, L_1, ..., L_{n-1}, N_freq, 2, 2)
    # — flattened to (B_inner, N_freq, 2, 2) for residual scoring.

    swept_pos_to_param_idx = [i for i in range(n_params) if L_list[i] > 1]
    n_swept_dims = len(swept_pos_to_param_idx)
    inner_lens = [int(L_list[i]) for i in swept_pos_to_param_idx]

    # Lookup table of *display* values for index→display mapping
    L_max = max(max(L_list), 1)
    sweep_table_h = np.zeros((n_params, L_max), dtype=np.float64)
    for i, sl in enumerate(sweep_lists):
        sweep_table_h[i, :L_list[i]] = sl
    sweep_table_dev = xp.asarray(sweep_table_h)                            # (n_params, L_max)
    if n_swept > 0:
        swept_indices_dev = xp.asarray(np.asarray(swept_idx_full, dtype=np.int64))
    else:
        swept_indices_dev = None

    # Constant scalars passed straight into simulate_batch
    const_si = {sweep_keys[i]: float(sweep_lists[i][0]) / float(sweep_scales[i])
                for i in range(n_params) if L_list[i] == 1}
    varies_mask = [L_list[i] > 1 for i in range(n_params)]
    swept_set = {sweep_keys[i] for i in range(n_params) if varies_mask[i]}

    # ── FP32 sweep mode (GPU only, model-opt-in) ─────────────────────
    # Cheng T/Pi's batched intrinsic-Y kernels honour cache["_cdtype"]
    # so the main loop can run entirely in complex64.  fp64 rerank
    # below replaces the residuals on the surviving top-K combos so
    # the final ranking and the persisted residuals are double-precision.
    #
    # CPU mode is left at fp64: scalar AVX is the same width for
    # both, and reduced precision saves no wall time on this code.
    use_fp32_sweep = bool(use_cuda and getattr(
        model_cls, "SUPPORTS_FP32_SWEEP", False))
    cdtype_main = np.complex64 if use_fp32_sweep else np.complex128
    rdtype_main = np.float32   if use_fp32_sweep else np.float64

    # Move S_mea onto the compute device once.  Two copies for fp32:
    # the cast (complex64) version drives the main loop, the original
    # complex128 stays alive for the fp64 rerank.
    S_mea_dev_fp64 = xp.asarray(S_raw)
    if use_fp32_sweep:
        S_mea_dev = S_mea_dev_fp64.astype(np.complex64)
    else:
        S_mea_dev = S_mea_dev_fp64

    # ── N-D omega: (1,)*n_swept_dims + (N_freq,) ────────────────────
    # Number of leading 1s = number of swept axes, so omega broadcasts
    # cleanly against any swept-param tensor regardless of which axes
    # it occupies.
    N_freq = len(freq)
    omega_shape = (1,) * n_swept_dims + (N_freq,)
    omega_dev = xp.asarray(
        2.0 * np.pi * np.asarray(freq, dtype=np.float64)
    ).reshape(omega_shape)

    # ── Pre-bake fully-constant sub-networks (lifted out of the loop)
    # If a sub-network has *zero* swept inputs, its tensor is built
    # once here at shape (1,...,1,N_freq) and reused every iteration.
    # If a sub-network has *some* swept inputs, the model code will
    # build it inside simulate_batch — but at its own native dims
    # (smaller than the full inner block), thanks to N-D broadcasting.
    _PAD_CAP_KEYS = ("Cpbe", "Cpce", "Cpbc")
    _SER_LEAD_KEYS = ("Rpb", "Rpc", "Rpe", "Lb", "Lc", "Le")
    _CHENG_EXTR_KEYS = ("Cbex", "Cbcx")
    # Cheng-T intrinsic sub-expression groups: each is pre-bakeable
    # whenever *none* of its inputs are swept.
    _T_ZBE_KEYS   = ("Rbe", "Cbe")
    _T_ZBC_KEYS   = ("Rbc", "Cbc")
    _T_ALPHA_KEYS = ("alpha0", "tauB", "tauC")
    _T_INT_ALL    = ("Rbi",) + _T_ZBE_KEYS + _T_ZBC_KEYS + _T_ALPHA_KEYS
    # Cheng-π intrinsic sub-expression groups
    _PI_YBE_KEYS  = ("Rbe", "Cbe")
    _PI_YBC_KEYS  = ("Rbc", "Cbc")
    _PI_GM_KEYS   = ("Gm0", "tau")
    _PI_INT_ALL   = ("Rbi",) + _PI_YBE_KEYS + _PI_YBC_KEYS + _PI_GM_KEYS

    static_p = dict(all_p)
    for _k, _v in const_si.items():
        static_p[_k] = _v

    # The fp64 cache is the canonical one — built first, then
    # cast to a parallel fp32 cache for the main loop if eligible.
    # The fp64 cache is also kept alive for the fp64 rerank below.
    static_cache_fp64 = {"omega": omega_dev}
    _cached_msgs = []
    if not (set(_PAD_CAP_KEYS) & swept_set):
        static_cache_fp64["Y_pad"] = build_Y_pad_batch(
            static_p, omega_dev, 1, N_freq, xp)
        _cached_msgs.append("Y_pad")
    if not (set(_SER_LEAD_KEYS) & swept_set):
        static_cache_fp64["Z_ser"] = build_Z_ser_batch(
            static_p, omega_dev, 1, N_freq, xp)
        _cached_msgs.append("Z_ser")
    if not (set(_CHENG_EXTR_KEYS) & swept_set):
        _Cbex_c = float(static_p.get("Cbex", 0.0))
        _Cbcx_c = float(static_p.get("Cbcx", 0.0))
        static_cache_fp64["Y_extr"] = (1j * omega_dev * _Cbex_c,
                                       1j * omega_dev * _Cbcx_c)
        _cached_msgs.append("Y_extr")

    # ── Cheng-specific intrinsic sub-expression caches ──────────────
    # These mirror what _Y_int_T_batch / _Y_int_Pi_batch will look up:
    # if a (Rbe,Cbe) / (Rbc,Cbc) / (alpha0,tauB,tauC) pair has all
    # constant inputs we can pre-build the result once and reuse it
    # every chunk.  When *every* intrinsic param is constant we go
    # one step further and pre-build the four intrinsic-Y planes
    # outright, skipping the entire per-chunk inv() of Z_in.
    _model_short = getattr(model_cls, "SHORT", "")
    if _model_short == "T":
        if not (set(_T_ZBE_KEYS) & swept_set):
            _Rbe_c = float(static_p.get("Rbe", 1.0))
            _Cbe_c = float(static_p.get("Cbe", 0.0))
            static_cache_fp64["Zbe"] = (
                _Rbe_c / (1.0 + 1j * omega_dev * _Rbe_c * _Cbe_c))
            _cached_msgs.append("Zbe")
        if not (set(_T_ZBC_KEYS) & swept_set):
            _Rbc_c = float(static_p.get("Rbc", 1.0))
            _Cbc_c = float(static_p.get("Cbc", 0.0))
            static_cache_fp64["Zbc"] = (
                _Rbc_c / (1.0 + 1j * omega_dev * _Rbc_c * _Cbc_c))
            _cached_msgs.append("Zbc")
        if not (set(_T_ALPHA_KEYS) & swept_set):
            _a0 = float(static_p.get("alpha0", 0.0))
            _tC = float(static_p.get("tauC",   0.0))
            _tB = float(static_p.get("tauB",   0.0))
            static_cache_fp64["alpha"] = (
                _a0 * xp.exp(-1j * omega_dev * _tC)
                / (1.0 + 1j * omega_dev * _tB))
            _cached_msgs.append("alpha")
        if not (set(_T_INT_ALL) & swept_set):
            # Inline what _Y_int_T_batch would compute, once.
            _Rbi_c = float(static_p.get("Rbi", 0.0))
            _Zbe_c = static_cache_fp64["Zbe"]
            _Zbc_c = static_cache_fp64["Zbc"]
            _alpha_c = static_cache_fp64["alpha"]
            _z00 = _Rbi_c + _Zbe_c
            _z01 = _Zbe_c
            _z10 = _Zbe_c - _alpha_c * _Zbc_c
            _z11 = (1.0 - _alpha_c) * _Zbc_c + _Zbe_c
            _det = _z00 * _z11 - _z01 * _z10
            _inv_det = 1.0 / _det
            static_cache_fp64["T_int_planes"] = (
                 _z11 * _inv_det,
                -_z01 * _inv_det,
                -_z10 * _inv_det,
                 _z00 * _inv_det,
            )
            _cached_msgs.append("T_int_planes")
    elif _model_short == "pi":
        if not (set(_PI_YBE_KEYS) & swept_set):
            _Rbe_c = float(static_p.get("Rbe", 1.0))
            _Cbe_c = float(static_p.get("Cbe", 0.0))
            static_cache_fp64["Ybe"] = (
                1.0 / _Rbe_c + 1j * omega_dev * _Cbe_c)
            _cached_msgs.append("Ybe")
        if not (set(_PI_YBC_KEYS) & swept_set):
            _Rbc_c = float(static_p.get("Rbc", 1e9))
            _Cbc_c = float(static_p.get("Cbc", 0.0))
            static_cache_fp64["Ybc"] = (
                1.0 / _Rbc_c + 1j * omega_dev * _Cbc_c)
            _cached_msgs.append("Ybc")
        if not (set(_PI_GM_KEYS) & swept_set):
            _Gm0_c = float(static_p.get("Gm0", 0.0))
            _tau_c = float(static_p.get("tau", 0.0))
            static_cache_fp64["gm"] = (
                _Gm0_c * xp.exp(-1j * omega_dev * _tau_c))
            _cached_msgs.append("gm")
        if not (set(_PI_INT_ALL) & swept_set):
            # Inline what _Y_int_Pi_batch would compute, once.
            _Rbi_c   = float(static_p.get("Rbi", 0.0))
            _Ybe_c   = static_cache_fp64["Ybe"]
            _Ybc_c   = static_cache_fp64["Ybc"]
            _gm_c    = static_cache_fp64["gm"]
            _yc00 = _Ybe_c + _Ybc_c
            _yc01 = -_Ybc_c
            _yc10 = _gm_c  - _Ybc_c
            _yc11 = _Ybc_c
            _idc  = 1.0 / (_yc00 * _yc11 - _yc01 * _yc10)
            _zc00 =  _yc11 * _idc + _Rbi_c
            _zc01 = -_yc01 * _idc
            _zc10 = -_yc10 * _idc
            _zc11 =  _yc00 * _idc
            _idi  = 1.0 / (_zc00 * _zc11 - _zc01 * _zc10)
            static_cache_fp64["Pi_int_planes"] = (
                 _zc11 * _idi,
                -_zc01 * _idi,
                -_zc10 * _idi,
                 _zc00 * _idi,
            )
            _cached_msgs.append("Pi_int_planes")

    # ── Build the fp32 cache (cast of fp64 cache) for main loop ─────
    # Everything except `omega` and the dtype tag is a complex tensor;
    # cast each in-place to complex64.  `omega` is real — keep its
    # own copy at float32 so multiply-by-J stays in c64.
    if use_fp32_sweep:
        omega_dev_fp32 = omega_dev.astype(np.float32)
        static_cache = {"omega": omega_dev_fp32, "_cdtype": np.complex64}
        for _k, _v in static_cache_fp64.items():
            if _k in ("omega", "_cdtype"):
                continue
            if isinstance(_v, tuple):
                static_cache[_k] = tuple(
                    _p.astype(np.complex64) if hasattr(_p, "astype") else _p
                    for _p in _v
                )
            elif hasattr(_v, "astype"):
                static_cache[_k] = _v.astype(np.complex64)
            else:
                static_cache[_k] = _v
    else:
        static_cache = static_cache_fp64
        static_cache["_cdtype"] = np.complex128

    if _cached_msgs:
        _prec_lbl = "fp32" if use_fp32_sweep else "fp64"
        st.caption(f"Pre-baked constant networks ({_prec_lbl}): "
                   + ", ".join(_cached_msgs))
        print(f"[tune] pre-baked ({_prec_lbl}): "
              f"{', '.join(_cached_msgs)}", flush=True)

    col_names = ["Total Residual (%)", "S11 (%)", "S12 (%)", "S21 (%)", "S22 (%)"]
    for lbl, u in zip(sweep_labels, sweep_units):
        col_names.append(f"{lbl} ({u})" if u else lbl)

    sess_key = f"tune_df_{topo_key}_{fname}"

    def _topk_to_df(top_arr_host):
        """Display DataFrame is the top-10 per metric across
        {Total, S11, S12, S21, S22}, deduped — 10 ≤ R ≤ 50 rows
        sorted by Total Residual.  See ``_multi_metric_top_n``."""
        return pd.DataFrame(
            _multi_metric_top_n(top_arr_host, per_metric=10),
            columns=col_names)

    # ── Auto slab sizing from device free memory ────────────────────
    # Per-combo working-set estimate.
    #
    # The model uses *scalar plane* representation (4 separate (B,N)
    # complex128 planes per 2×2 matrix, not full (B,N,2,2) tensors)
    # — see _sim_wrap_batch in models/cheng.py.  Realistic peak live
    # planes per combo:
    #   • Y_int (4 planes) + Y_extr (2) + Y_ex (4)       = 10 planes
    #   • Z_ex (4) + Z_ser (4) + Z_tot (4)               = 12 planes
    #   • Y_tot (4) + Y_pad (4) + Y_total (4)            = 12 planes
    #   • Y_norm/M/M_inv/N/S working set                 = 12 planes
    #   • Final stacked (B,N,2,2)                        =  4 planes
    #   • Residual diff/num/val                          =  6 planes
    # Each plane = 16 bytes/element × N_freq elements per combo
    # in complex128, or 8 bytes/element in complex64 (fp32 sweep).
    # Initial guess: ~60 planes × 16 = 960 B/combo per N_freq, with a
    # 1.25× safety margin → 12× N_freq complex128 tensor-equivalents.
    # In fp32 mode the same 60 planes are 8 B each, so the bytes
    # estimate halves.  Replaced after iter 1 by an empirical
    # measurement (see `_calibrated` below) — the initial guess only
    # governs the *first* block size before we have real data.
    _bytes_per_complex = 8 if use_fp32_sweep else 16
    per_combo_bytes = (_bytes_per_complex * 4) * N_freq * 12
    _calibrated = False
    free_label = ""
    if use_cuda:
        try:
            # Free up any cached blocks first so the query reflects
            # what we can *actually* allocate now (not what's been
            # pinned by previous calculations).
            try:
                _cp.get_default_memory_pool().free_all_blocks()
                _cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            free_b, total_b = _cp.cuda.runtime.memGetInfo()
            dev = _cp.cuda.Device(0)
            sm_count = dev.attributes.get("MultiProcessorCount", 0)
            free_label = (f"GPU{dev.id}: {free_b/1024**3:.2f}/"
                          f"{total_b/1024**3:.2f} GiB free  ·  {sm_count} SMs")
            # 0.55 keeps ~45% of free VRAM as headroom for pool
            # 0.95 uses more VRAM
            # fragmentation, top-K scratch, persistent buffers, and
            # the measurement S_mea_dev tensor.
            budget = int(free_b * 0.95)
        except Exception:
            budget = 1 * 1024**3
    else:
        # ram_available_bytes() is cgroup-aware (Streamlit Cloud runs
        # inside a memory-limited container) and never raises, so no
        # try/except is needed here — see helpers/mem_budget.py.
        avail = ram_available_bytes()
        free_label = f"CPU RAM: {avail/1024**3:.2f} GiB free"
        budget = int(avail * 0.25)

    max_inner = max(1, budget // max(per_combo_bytes, 1))
    if not use_cuda:
        # CPU mode: cap inner block size more aggressively
        max_inner = min(max_inner, 65_536)
    else:
        # GPU mode: clamp to a sane upper bound to bound output size
        max_inner = min(max_inner, 16_777_216)  # 16M combos per slab

    # ── Multi-axis block sizing ─────────────────────────────────────
    # Pick a per-axis block_shape (one length per swept axis) such
    # that prod(block_shape) <= max_inner.  We start with the full
    # inner_lens and greedily halve the *largest* axis until the
    # product fits.  This generalises the old "slab one axis" logic
    # to handle the case where multiple axes need slabbing — i.e.
    # when the product of all "other" axes alone exceeds VRAM.
    def _shrink_to_budget(shape, budget):
        """Greedy: halve the largest axis until prod(shape) <= budget."""
        bs = list(shape)
        if not bs:
            return bs
        while True:
            p = 1
            for v in bs:
                p *= v
            if p <= budget or budget < 1:
                return bs
            k_max = 0
            for k in range(1, len(bs)):
                if bs[k] > bs[k_max]:
                    k_max = k
            if bs[k_max] <= 1:
                return bs  # cannot shrink further
            bs[k_max] = max(1, bs[k_max] // 2)

    def _shrink_one_step(shape):
        """Halve the largest axis once. Returns (new_shape, did_shrink)."""
        bs = list(shape)
        if not bs:
            return bs, False
        k_max = 0
        for k in range(1, len(bs)):
            if bs[k] > bs[k_max]:
                k_max = k
        if bs[k_max] <= 1:
            return bs, False
        bs[k_max] = max(1, bs[k_max] // 2)
        return bs, True

    def _grow_to_budget(shape, full_lens, budget):
        """Greedy: double the smallest still-growable axis until either
        doubling again would exceed `budget`, or every axis is at its
        maximum (full_lens[k]).  Returns the new shape (a list)."""
        bs = list(shape)
        if not bs:
            return bs
        while True:
            p = 1
            for v in bs:
                p *= v
            if p * 2 > budget:
                return bs
            # Pick smallest axis that can still grow
            cand_k = -1
            cand_v = None
            for k in range(len(bs)):
                if bs[k] < full_lens[k]:
                    if cand_k < 0 or bs[k] < cand_v:
                        cand_k = k
                        cand_v = bs[k]
            if cand_k < 0:
                return bs  # nothing left to grow
            bs[cand_k] = min(full_lens[cand_k], max(2, bs[cand_k] * 2))

    if n_swept_dims == 0:
        block_shape = []
        inner_block = 1
    else:
        block_shape = _shrink_to_budget(inner_lens, max_inner)
        inner_block = 1
        for v in block_shape:
            inner_block *= v

    CHUNK_MAX = inner_block  # upper bound on per-iter combo count

    import sys as _sys, time as _time
    _t_start = _time.time()
    if free_label:
        st.caption(free_label)
    if n_swept_dims == 0:
        _slab_label = "single combo"
    elif inner_block >= n_total:
        _slab_label = (f"single N-D block, "
                       f"{inner_block:,} combos / iter")
    else:
        # Shape summary: e.g. "block=(8,32,7), 1,792 combos / iter"
        _shape_str = ",".join(str(v) for v in block_shape)
        _full_str  = ",".join(str(v) for v in inner_lens)
        # Conservative iter count from current block (may rise if
        # shrunk later, fall if grown — we don't grow).
        _n_iters_pre = 1
        for k, v in enumerate(block_shape):
            _n_iters_pre *= (inner_lens[k] + v - 1) // v
        _slab_label = (f"block=({_shape_str})/({_full_str}), "
                       f"{inner_block:,} combos / iter, "
                       f"≥{_n_iters_pre} iters")
    print(f"\n[tune] start  mode={_mode_label}  total={n_total:,}  "
          f"{_slab_label}  batched={has_batch}  {free_label}", flush=True)

    # ── UI placeholders ────────────────────────────────────────────
    ui_cols = st.columns([5, 1])
    with ui_cols[0]:
        _tuning_word = tr("Tuning", "調諧")
        _prog_lbl = f"{_tuning_word} {phase_label} ({_mode_label})…" \
                    if phase_label \
                    else f"{_tuning_word} ({_mode_label})…"
        progress = st.progress(0, text=_prog_lbl)
    with ui_cols[1]:
        stop_box = st.empty()
    best_box = st.empty()      # live "best so far" line

    # The Stop button works by triggering a Streamlit re-run on click;
    # the next st.* call inside the loop raises RerunException, which
    # we catch and turn into a clean cancellation.  No on_click needed.
    stop_key = f"tune_stop_{topo_key}_{fname}{phase_suffix}"
    stop_box.button(
        tr("⏹ Stop", "⏹ 停止"),
        key=stop_key,
        help=tr("Stop the calculation. The best results found so far "
                "are kept.",
                "停止計算。目前找到的最佳結果將被保留。"),
        type="secondary",
    )

    cancelled = False

    # ── Persistent device buffers for the top-K accumulator ─────────
    # Inf placeholders ensure new finite values always displace them.
    # top_4 must also start at inf — Prioritize sorts by view_4[:, _smap[sort_metric]],
    # so zero placeholders would otherwise out-rank every real residual and the
    # "Best so far" panel would stay empty for the entire sweep.
    top_res    = xp.full(TOP_K, xp.inf, dtype=xp.float64)
    top_4      = xp.full((TOP_K, 4), xp.inf, dtype=xp.float64)
    top_swept  = xp.zeros((TOP_K, max(n_swept, 1)), dtype=xp.float64)

    # ── Pre-allocated scratch buffers for the merge step ────────────
    # Sized for TOP_K + the *maximum* chunk we'd ever submit.  Reused
    # every iteration → zero per-chunk allocation churn for top-K.
    SCRATCH = TOP_K + CHUNK_MAX
    scratch_res   = xp.empty(SCRATCH, dtype=xp.float64)
    scratch_4     = xp.empty((SCRATCH, 4), dtype=xp.float64)
    scratch_swept = xp.empty((SCRATCH, max(n_swept, 1)), dtype=xp.float64)

    # Threshold used to distinguish "real residual" from the BIG=1e308
    # clamp the inner loop assigns to combos that fail the
    # deviation / residual filters or produced NaN/inf simulations.
    # Real residuals are percentages (typically 0.01–1000); 1e100 is a
    # comfortable separator below BIG and above any plausible value.
    _VALID_RES_MAX = 1.0e100

    def _sync_topk_host():
        """Pull the top-K state to host as a (n, 5+n_params) numpy array.
        Drops inf placeholders and BIG-clamped (filter-rejected) rows.
        Constants are filled from sweep_lists.
        """
        if use_cuda:
            tr = _cp.asnumpy(top_res)
            t4 = _cp.asnumpy(top_4)
            ts = _cp.asnumpy(top_swept) if n_swept > 0 else \
                 np.zeros((TOP_K, 0), dtype=np.float64)
        else:
            tr = np.asarray(top_res)
            t4 = np.asarray(top_4)
            ts = np.asarray(top_swept) if n_swept > 0 else \
                 np.zeros((TOP_K, 0), dtype=np.float64)
        valid = np.isfinite(tr) & (tr < _VALID_RES_MAX)
        n = int(valid.sum())
        if n == 0:
            return None
        out = np.empty((n, 5 + n_params), dtype=np.float64)
        out[:, 0]   = tr[valid]
        out[:, 1:5] = t4[valid]
        j = 0
        for i in range(n_params):
            if L_list[i] == 1:
                out[:, 5 + i] = float(sweep_lists[i][0])
            else:
                out[:, 5 + i] = ts[valid, j]
                j += 1
        return out

    def _persist_topk():
        """Best-effort persist to session_state. Safe to call from
        anywhere (including the finally clause)."""
        try:
            h = _sync_topk_host()
            if h is not None:
                st.session_state[sess_key] = _topk_to_df(h)
        except Exception:
            pass

    def _release_gpu_memory():
        """Best-effort: synchronize the device, run gc twice (to break
        cycles), then return all idle pool blocks to the driver.

        Order matters:
          1. Synchronize first — async kernels may still be holding
             tensor inputs alive on the stream.  Without sync,
             `free_all_blocks()` would skip those blocks.
          2. gc.collect() twice — CuPy ndarrays often participate in
             reference cycles via residual/topk dicts; one pass may
             not break them all.
          3. Free both device and pinned-host memory pools.
        """
        if not use_cuda:
            gc.collect()
            gc.collect()
            return
        try:
            _cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass
        gc.collect()
        gc.collect()
        try:
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    try:
        if not has_batch:
            raise RuntimeError(
                f"Model {model_cls.__name__} has no simulate_batch — "
                f"cannot run batched tuning. Implement simulate_batch.")

        processed = 0
        last_ui = 0.0
        chunks_done = 0
        last_chunk_ms = 0.0
        # axis_offsets[k] = current row index along swept axis k.
        # Advances by block_shape[k] after each successful iteration,
        # carrying over to the next-outer axis at L_k.
        axis_offsets = [0] * n_swept_dims
        # Resize gate — never re-poll the driver more than once per
        # this many seconds.  Polling memGetInfo() is cheap (~µs) but
        # the *real* cost we're avoiding is fragmenting the allocator
        # by trying to resize too aggressively.
        _last_resize_t = 0.0
        _RESIZE_INTERVAL = 2.0
        # Consecutive-OOM counter — drives the escalating recovery
        # strategy: 1st OOM just shrinks (cheap), 2nd consecutive OOM
        # also flushes the pool (expensive but reclaims everything).
        # Reset to 0 after any successful iteration.
        _oom_streak = 0

        while processed < n_total:
            # ── Adaptive multi-axis block sizing (gated). ──────────
            # We re-query free VRAM at most once every _RESIZE_INTERVAL
            # seconds.  Crucially we do NOT call free_all_blocks() —
            # flushing the pool every iteration kills allocator
            # amortization and forces every alloc to fall through to
            # cudaMalloc, which on Windows WDDM saturates the Copy
            # engine with page-table updates and starves Compute.
            #
            # Instead we use the pool's own bookkeeping (free_bytes
            # = bytes already cached and re-allocatable for free) to
            # build an "effective free" estimate without disturbing
            # the pool.  Once `_calibrated`, the budget is also used
            # to decide whether the block can grow.
            _now_t = _time.time()
            if (use_cuda and n_swept_dims > 0
                    and (_now_t - _last_resize_t) >= _RESIZE_INTERVAL):
                _last_resize_t = _now_t
                try:
                    _mp = _cp.get_default_memory_pool()
                    _pool_free = _mp.free_bytes()    # cached, re-allocatable
                    _drv_free, _ = _cp.cuda.runtime.memGetInfo()
                    # Effective free = what driver reports + what's
                    # already cached in the pool (the pool's cached
                    # blocks count against driver-reported free, but
                    # are available to us without a malloc).
                    _eff_free = _drv_free + _pool_free
                    _budget_now = int(_eff_free * 0.55)
                    _max_inner_now = max(
                        1, _budget_now // max(per_combo_bytes, 1))
                    _cur_prod = 1
                    for v in block_shape:
                        _cur_prod *= v
                    # Shrink if we're now over budget
                    _shrunk = _shrink_to_budget(block_shape, _max_inner_now)
                    _shrunk_prod = 1
                    for v in _shrunk:
                        _shrunk_prod *= v
                    if _shrunk_prod < _cur_prod:
                        block_shape = _shrunk
                        print(f"\n[tune] free VRAM dropped → block shrunk to "
                              f"({','.join(str(v) for v in block_shape)}) "
                              f"= {_shrunk_prod:,} combos / iter",
                              flush=True)
                    elif _calibrated:
                        # Try to grow.  Only after calibration — using
                        # the pessimistic initial estimate to compute
                        # a grow target would let us grow into an OOM.
                        _grown = _grow_to_budget(
                            block_shape, inner_lens, _max_inner_now)
                        _grown_prod = 1
                        for v in _grown:
                            _grown_prod *= v
                        # Only act if growth is meaningful (≥ +50%)
                        if _grown_prod >= int(_cur_prod * 1.5):
                            block_shape = _grown
                            print(f"\n[tune] free VRAM ample → block grown to "
                                  f"({','.join(str(v) for v in block_shape)}) "
                                  f"= {_grown_prod:,} combos / iter",
                                  flush=True)
                except Exception:
                    pass

            # ── Compute this iteration's per-axis extent ────────────
            # cur_shape[k] = how many rows of axis k this iteration
            # covers, capped at the remainder of the axis.
            if n_swept_dims == 0:
                cur_shape = ()
                inner_shape = ()
            else:
                cur_shape = tuple(
                    min(block_shape[k], inner_lens[k] - axis_offsets[k])
                    for k in range(n_swept_dims)
                )
                inner_shape = cur_shape
            B_inner = 1
            for L in inner_shape:
                B_inner *= L
            if B_inner == 0:
                break
            _t_chunk = _time.time()

            # ── Build N-D parameter tensors for this slab ───────────
            # Each swept param gets shape (1,..,L,..,1,1) — its own
            # length-L axis at its position, 1s elsewhere, and a
            # trailing 1 for the freq slot.  Constants stay scalar.
            # On host the per-param values are tiny: only the *device*
            # tensors matter for VRAM, and they're (1,..,L,..,1,1).
            p_nd = None
            S_batch_nd = None
            S_flat = None
            res = None
            cur_total = None
            cur_4 = None
            cur_swept = None
            lin = None
            idx_2d = None
            view_res = None
            idx = None
            try:
                p_nd = dict(all_p)
                for j, key in enumerate(sweep_keys):
                    if not varies_mask[j]:
                        p_nd[key] = const_si[key]
                        continue
                    swept_pos = swept_pos_to_param_idx.index(j)
                    o = axis_offsets[swept_pos]
                    ln = cur_shape[swept_pos]
                    vals_disp = sweep_lists[j][o:o + ln]
                    vals_si = vals_disp / float(sweep_scales[j])
                    nd_shape = [1] * (n_swept_dims + 1)
                    nd_shape[swept_pos] = len(vals_si)
                    # rdtype_main is float32 in fp32 sweep mode, so
                    # the swept tensor doesn't get promoted back to
                    # float64 inside the model kernels.
                    p_nd[key] = xp.asarray(
                        vals_si, dtype=rdtype_main).reshape(nd_shape)
            except Exception as exc:
                is_oom = (isinstance(exc, MemoryError) or
                          "out of memory" in str(exc).lower() or
                          "OutOfMemoryError" in type(exc).__name__)
                # Drop any partially-built tensors so the retry has room
                p_nd = None
                if is_oom:
                    new_block, did = _shrink_one_step(block_shape)
                    if did:
                        block_shape = new_block
                        _oom_streak += 1
                        print(f"\n[tune] OOM in param-gen → block "
                              f"({','.join(str(v) for v in block_shape)}) "
                              f"[streak={_oom_streak}]",
                              flush=True)
                        gc.collect()
                        # Only flush the pool on the *second* OOM in a
                        # row.  A single OOM is usually solved by the
                        # shrink alone — flushing every time would
                        # destroy allocator amortization (the same
                        # bug we're fixing in this commit).
                        if use_cuda and _oom_streak >= 2:
                            try:
                                _cp.get_default_memory_pool().free_all_blocks()
                                _cp.get_default_pinned_memory_pool().free_all_blocks()
                                print("[tune]   pool flushed (last resort)",
                                      flush=True)
                            except Exception:
                                pass
                        continue
                raise

            # ── Run simulate + residuals + top-K merge on device ────
            try:
                # Snapshot pool size BEFORE simulate so we can
                # measure the actual per-combo allocation footprint
                # of this iteration and replace the static estimate
                # with an empirical one (only on the first call,
                # gated by `_calibrated`).
                if use_cuda and not _calibrated:
                    try:
                        _bytes_before = _cp.get_default_memory_pool().total_bytes()
                    except Exception:
                        _bytes_before = 0
                S_batch_nd = model_cls.simulate_batch(
                    p_nd, freq, z0, xp=xp, cache=static_cache)
                # Shape: inner_shape + (N_freq, 2, 2)
                S_flat = S_batch_nd.reshape(B_inner, N_freq, 2, 2)
                res = _port_residuals_batch(S_mea_dev, S_flat, xp)
                cur_total = res["Total"]                                  # (B_inner,)
                cur_4 = xp.stack(
                    [res["S11"], res["S12"], res["S21"], res["S22"]],
                    axis=1)                                                # (B_inner, 4)

                # NaN/inf protection — sentinel value sorts to bottom
                # without forcing a host-side .all() check (no sync).
                BIG = 1.0e308
                cur_total = xp.where(xp.isfinite(cur_total), cur_total, BIG)
                # Per-port residuals are also the sort key in
                # "Prioritize" mode, so clamp NaN/inf there too.
                cur_4 = xp.where(xp.isfinite(cur_4), cur_4, BIG)

                # ── Balance / quality filters ───────────────────────
                # Push unbalanced or poor-residual combos to the
                # bottom of the ranking by clamping their sort key
                # to BIG.  Balanced survivors are then sorted normally
                # → "lowest total among balanced".
                if dev_threshold is not None:
                    _dev = cur_4.max(axis=1) - cur_4.min(axis=1)
                    cur_total = xp.where(_dev > dev_threshold, BIG, cur_total)
                    cur_4 = xp.where(
                        _dev[:, None] > dev_threshold, BIG, cur_4)
                if res_threshold is not None:
                    _peak = cur_4.max(axis=1)
                    cur_total = xp.where(_peak > res_threshold, BIG, cur_total)
                    cur_4 = xp.where(
                        _peak[:, None] > res_threshold, BIG, cur_4)

                # ── Build display values for swept params ───────────
                # Decompose flat index → multi-axis index using
                # row-major strides over `inner_shape`, then look up
                # via sweep_table_dev[swept_indices, axis_index].
                if n_swept_dims > 0:
                    local_strides_h = np.ones(n_swept_dims, dtype=np.int64)
                    for k in range(n_swept_dims - 2, -1, -1):
                        local_strides_h[k] = local_strides_h[k + 1] * inner_shape[k + 1]
                    local_strides_dev = xp.asarray(local_strides_h)
                    local_lens_dev    = xp.asarray(np.asarray(inner_shape, dtype=np.int64))

                    lin = xp.arange(B_inner, dtype=xp.int64)
                    idx_2d = (lin[:, None] // local_strides_dev[None, :]) \
                              % local_lens_dev[None, :]                    # (B_inner, n_swept)

                    if any(o > 0 for o in axis_offsets):
                        offset_h = np.asarray(axis_offsets, dtype=np.int64)
                        idx_2d = idx_2d + xp.asarray(offset_h)[None, :]

                    cur_swept = sweep_table_dev[swept_indices_dev[None, :], idx_2d]

                # ── Top-K merge into pre-allocated scratch ──────────
                total_in = TOP_K + B_inner
                scratch_res[:TOP_K]              = top_res
                scratch_res[TOP_K:total_in]      = cur_total
                scratch_4[:TOP_K]                = top_4
                scratch_4[TOP_K:total_in]        = cur_4
                if n_swept_dims > 0:
                    scratch_swept[:TOP_K]         = top_swept
                    scratch_swept[TOP_K:total_in] = cur_swept

                view_res = scratch_res[:total_in]
                view_4   = scratch_4[:total_in]
                # When prioritizing one S-parameter, sort by that
                # column instead of the total residual. We still keep
                # `top_res` = total so the displayed "Total Residual"
                # column stays meaningful.
                if sort_metric == "Total":
                    sort_view = view_res
                else:
                    _smap = {"S11": 0, "S12": 1, "S21": 2, "S22": 3}
                    sort_view = view_4[:, _smap[sort_metric]]
                idx = xp.argpartition(sort_view, TOP_K)[:TOP_K]
                idx = idx[xp.argsort(sort_view[idx])]

                top_res[:] = view_res[idx]
                top_4[:]   = view_4[idx]
                if n_swept_dims > 0:
                    top_swept[:] = scratch_swept[:total_in][idx]

                # ── Empirical calibration (first iteration only) ─
                # Measure how many bytes the pool actually grew by
                # during this iteration, divide by B_inner, and use
                # that as the ground-truth per-combo footprint.
                # Apply a 1.3× safety to absorb spike differences
                # between iterations.  This replaces the (rough)
                # initial estimate so subsequent shrink/grow
                # decisions are made on real data.
                if use_cuda and not _calibrated and B_inner > 0:
                    try:
                        _bytes_after = _cp.get_default_memory_pool().total_bytes()
                        _delta = max(0, _bytes_after - _bytes_before)
                        if _delta > 0:
                            _measured = _delta // B_inner
                            _new_pcb = max(1, int(_measured * 1.3))
                            print(f"\n[tune] calibrated per_combo_bytes "
                                  f"= {_new_pcb:,}  ({_measured:,} "
                                  f"measured × 1.3 safety; was "
                                  f"{per_combo_bytes:,})",
                                  flush=True)
                            per_combo_bytes = _new_pcb
                        _calibrated = True
                        # Force a resize check on the *next* iter so
                        # the new estimate can immediately grow the
                        # block if there's headroom.
                        _last_resize_t = 0.0
                    except Exception:
                        _calibrated = True   # don't keep retrying
            except Exception as exc:
                is_oom = (isinstance(exc, MemoryError) or
                          "out of memory" in str(exc).lower() or
                          "OutOfMemoryError" in type(exc).__name__)
                if is_oom:
                    new_block, did = _shrink_one_step(block_shape)
                    if did:
                        old_str = ",".join(str(v) for v in block_shape)
                        new_str = ",".join(str(v) for v in new_block)
                        _oom_streak += 1
                        print(f"\n[tune] OOM at block=({old_str}) → "
                              f"retry with ({new_str}) "
                              f"[streak={_oom_streak}]",
                              flush=True)
                        block_shape = new_block
                        # Drop intermediates before retrying — finally
                        # block will null these out, but we run gc.collect
                        # immediately to release memory before continue.
                        p_nd = None
                        S_batch_nd = None
                        S_flat = None
                        res = None
                        cur_total = None
                        cur_4 = None
                        cur_swept = None
                        lin = None
                        idx_2d = None
                        view_res = None
                        idx = None
                        gc.collect()
                        # Only flush the pool on the *second*
                        # consecutive OOM.  A single OOM is usually
                        # solved by the shrink alone.  Flushing every
                        # OOM destroys allocator amortization.
                        if use_cuda and _oom_streak >= 2:
                            try:
                                _cp.get_default_memory_pool().free_all_blocks()
                                _cp.get_default_pinned_memory_pool().free_all_blocks()
                                print("[tune]   pool flushed (last resort)",
                                      flush=True)
                            except Exception:
                                pass
                        continue   # do NOT advance counters
                    # Already at all-1s floor — bail out cleanly with
                    # whatever top-K we have, instead of spinning
                    # forever or crashing the Streamlit script.
                    print(f"\n[tune] OOM at block=(1,1,...) — single combo "
                          f"won't fit in VRAM, giving up",
                          flush=True)
                    st.error(tr(
                        f"GPU ran out of memory even at block=(1,1,...): "
                        f"a single combination's working set "
                        f"(~{per_combo_bytes/1024**2:.1f} MiB for {N_freq} freq pts) "
                        f"won't fit in available VRAM. Reduce the number "
                        f"of frequency points or free GPU memory.",
                        f"即使在 block=(1,1,...) 下 GPU 仍記憶體不足："
                        f"單一組合的工作集"
                        f"（約 {per_combo_bytes/1024**2:.1f} MiB，"
                        f"{N_freq} 個頻率點）無法容納於可用 VRAM 中。"
                        f"請減少頻率點數或釋放 GPU 記憶體。"))
                    cancelled = True
                    break
                off_str = ",".join(str(o) for o in axis_offsets)
                print(f"\n[tune] block @offsets ({off_str}) failed: {exc!r}",
                      flush=True)
                # Skip this block on non-OOM errors — advance below
                # via the post-loop counters by treating it as done.
                # Fall through to normal advance.
            finally:
                # Drop per-iteration intermediates so OOM retry has room
                # *before* re-allocating next iteration.  We don't
                # touch `top_*` / `scratch_*` (persistent).
                p_nd = None
                S_batch_nd = None
                S_flat = None
                res = None
                cur_total = None
                cur_4 = None
                cur_swept = None
                lin = None
                idx_2d = None
                view_res = None
                idx = None

            # Successful (or skipped) iteration — advance counters.
            # Reset OOM streak: we got through a full iteration, so
            # any past OOMs are no longer "consecutive".
            _oom_streak = 0
            # Multi-axis carry: advance the *innermost* axis by its
            # current block step; if it overflows L_k, reset to 0
            # and carry into the next-outer axis.
            if n_swept_dims == 0:
                processed = 1
            else:
                processed += B_inner
                k = n_swept_dims - 1
                while k >= 0:
                    axis_offsets[k] += block_shape[k]
                    if axis_offsets[k] < inner_lens[k]:
                        break
                    axis_offsets[k] = 0
                    k -= 1
                # k < 0  →  every axis wrapped, we're done.  The
                # while-loop guard `processed < n_total` will exit.
            chunks_done += 1
            last_chunk_ms = (_time.time() - _t_chunk) * 1000.0

            # ── Throttled UI tick: ~2 Hz, the only host sync point ──
            now = _time.time()
            if now - last_ui > 0.5 or processed >= n_total:
                elapsed = now - _t_start
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (n_total - processed) / rate if rate > 0 else 0.0
                _phase_str = f" {phase_label}" if phase_label else ""
                _tuning_word = tr("Tuning", "調諧")
                _combos_word = tr("combos", "組合")
                progress.progress(
                    min(1.0, processed / max(n_total, 1)),
                    text=(f"{_tuning_word}{_phase_str} ({_mode_label})… "
                          f"{processed:,}/{n_total:,} {_combos_word}  "
                          f"({rate:,.0f}/s, ETA {_fmt_eta(eta)})  "
                          f"slab={B_inner:,}  ({last_chunk_ms:.1f} ms/iter)"))
                top_arr_host = _sync_topk_host()
                if top_arr_host is not None:
                    best_series = pd.Series(top_arr_host[0], index=col_names)
                    best_box.markdown(
                        _best_summary_md(best_series, tuning_specs, topo_key, fname,
                                     label=tr("Best so far", "目前最佳")),
                        unsafe_allow_html=True,
                    )
                    # Persist every tick so a later crash leaves a result
                    st.session_state[sess_key] = _topk_to_df(top_arr_host)
                elif dev_threshold is not None or res_threshold is not None:
                    # All combos so far failed the deviation/residual
                    # filters — surface this so the user can widen
                    # thresholds instead of staring at a blank panel.
                    best_box.markdown(tr(
                        "*No combos have passed the deviation/residual "
                        "filters yet — consider widening the thresholds "
                        "if this persists.*",
                        "*尚無組合通過偏差/殘差篩選 — "
                        "若持續發生，請考慮放寬閾值。*"))
                last_ui = now

            _sys.stdout.write(
                f"\r[tune] {processed:>11,}/{n_total:,}  "
                f"({100.0*processed/max(n_total,1):5.1f}%)  "
                f"{(processed / max(_time.time()-_t_start, 1e-9)):>10,.0f} calc/s  "
                f"slab={B_inner:,}  {last_chunk_ms:6.1f} ms/iter")
            _sys.stdout.flush()
    except (KeyboardInterrupt, SystemExit):
        cancelled = True
        st.warning(tr(
            "Computation cancelled — keeping the best results found so far.",
            "計算已取消 — 已保留目前找到的最佳結果。"))
        print("\n[tune] cancelled by user (KeyboardInterrupt)", flush=True)
    except MemoryError as me:
        st.error(tr(
            f"Out of memory: {me}. Keeping the best results found so far.",
            f"記憶體不足：{me}。已保留目前找到的最佳結果。"))
        print(f"\n[tune] MemoryError: {me}", flush=True)
    except BaseException as exc:
        # Streamlit raises RerunException OR StopException when the
        # user clicks any widget (incl. our Stop button) — which one
        # depends on Streamlit version and the exact widget path.
        # Treat both identically: persist state, free GPU buffers,
        # then re-raise so Streamlit can finish the rerun cleanly.
        _is_rerun = (_RerunException is not None
                     and isinstance(exc, _RerunException))
        _is_stop  = (_StopException  is not None
                     and isinstance(exc, _StopException))
        # Belt-and-suspenders: also match by class name in case the
        # exception class moved between Streamlit versions and our
        # imports above silently fell through to None.
        _name = type(exc).__name__
        _is_streamlit_stop = _is_rerun or _is_stop or (
            _name in ("RerunException", "StopException"))
        if _is_streamlit_stop:
            cancelled = True
            print(f"\n[tune] cancelled (Streamlit {_name}, "
                  f"e.g. Stop button)", flush=True)
            _persist_topk()
            # Free GPU buffers before re-raising so the rerun starts clean
            try:
                S_mea_dev = None
                S_mea_dev_fp64 = None
                top_res = None; top_4 = None; top_swept = None
                scratch_res = None; scratch_4 = None; scratch_swept = None
                sweep_table_dev = None
                swept_indices_dev = None
                omega_dev = None
                # Per-iteration intermediates that may still be alive
                # if the exception was raised mid-iteration.  The
                # inner finally usually nulls these, but we
                # belt-and-suspenders here.
                try:
                    del p_nd, S_batch_nd, S_flat, res, cur_total, cur_4
                    del cur_swept, lin, idx_2d, view_res, idx
                except (NameError, UnboundLocalError):
                    pass
                static_cache.clear()
                if static_cache_fp64 is not static_cache:
                    static_cache_fp64.clear()
            except Exception:
                pass
            _release_gpu_memory()
            raise
        # Anything else: log, persist, re-raise
        print(f"\n[tune] unexpected exception: {exc!r}", flush=True)
        _persist_topk()
        raise
    finally:
        # Always persist whatever we have so a crash never wipes results.
        _persist_topk()

    progress.empty()
    stop_box.empty()

    n_kept = (0 if top_res is None
              else int(xp.sum(xp.isfinite(top_res)
                               & (top_res < _VALID_RES_MAX)).item()))

    # Keep a warning visible if a deviation/residual sweep rejected
    # everything; otherwise clear the live "best so far" line because
    # the persistent results table will render the final ranking.
    if (n_kept == 0
            and (dev_threshold is not None or res_threshold is not None)):
        best_box.warning(tr(
            "No combos passed the deviation/residual filters. "
            "Widen the thresholds and re-run.",
            "沒有組合通過偏差/殘差篩選。請放寬閾值後重新執行。"))
        # Clear any stale prior-sweep results so the table below
        # doesn't misleadingly show data from a different setting.
        st.session_state.pop(sess_key, None)
        st.session_state.pop(f"tune_elapsed_{topo_key}_{fname}", None)
    else:
        best_box.empty()
    print(f"\n[tune] done   processed={'?' if cancelled else f'{n_total:,}'}  "
          f"top={n_kept}  in {_time.time()-_t_start:.2f}s", flush=True)

    # ── FP64 rerank of the surviving top-K ──────────────────────────
    # The main loop ran in complex64 (cache_fp32) so the residuals
    # are fp32-accurate.  Re-evaluate every finite top-K combo at
    # complex128 using `static_cache_fp64`, replace the residuals,
    # and re-sort.  TOP_K is small (~100) so this is one batched
    # simulate_batch call — negligible cost vs the main sweep.
    if (use_fp32_sweep and not cancelled
            and top_res is not None and n_kept > 0):
        try:
            _top_h = _sync_topk_host()
            if _top_h is not None and len(_top_h) > 0:
                K_rk = int(len(_top_h))
                # Build (K,) SI arrays for swept params, scalars for
                # constants — same param dict shape `simulate_batch`
                # already understands.
                _p_rk = dict(all_p)
                for j, key in enumerate(sweep_keys):
                    col_disp = _top_h[:, 5 + j]
                    if L_list[j] == 1:
                        _p_rk[key] = (float(col_disp[0])
                                      / float(sweep_scales[j]))
                    else:
                        _p_rk[key] = xp.asarray(
                            col_disp / float(sweep_scales[j]),
                            dtype=np.float64)
                # Force fp64 cdtype on the rerank cache.
                static_cache_fp64["_cdtype"] = np.complex128
                S_rk = model_cls.simulate_batch(
                    _p_rk, freq, z0, xp=xp, cache=static_cache_fp64)
                S_rk_flat = S_rk.reshape(K_rk, N_freq, 2, 2)
                res_rk = _port_residuals_batch(
                    S_mea_dev_fp64, S_rk_flat, xp)
                tot_rk = xp.where(
                    xp.isfinite(res_rk["Total"]),
                    res_rk["Total"], 1.0e308)
                s4_rk  = xp.stack(
                    [res_rk["S11"], res_rk["S12"],
                     res_rk["S21"], res_rk["S22"]], axis=1)

                # Re-sort by fp64 residuals — match sort_metric chosen
                # for the main loop so Prioritize keeps its ranking.
                if sort_metric == "Total":
                    sort_rk = tot_rk
                else:
                    _smap = {"S11": 0, "S12": 1, "S21": 2, "S22": 3}
                    _col  = _smap[sort_metric]
                    sort_rk = xp.where(xp.isfinite(s4_rk[:, _col]),
                                        s4_rk[:, _col], 1.0e308)
                order_dev = xp.argsort(sort_rk)
                tot_rk_s  = tot_rk[order_dev]
                s4_rk_s   = s4_rk[order_dev]

                order_h = (_cp.asnumpy(order_dev) if use_cuda
                           else np.asarray(order_dev))
                _top_h_s = _top_h[order_h]
                _top_h_s[:, 0]   = (_cp.asnumpy(tot_rk_s) if use_cuda
                                    else np.asarray(tot_rk_s))
                _top_h_s[:, 1:5] = (_cp.asnumpy(s4_rk_s) if use_cuda
                                    else np.asarray(s4_rk_s))

                # Push back into the device top-K accumulators so
                # display and persist see fp64 values.
                top_res[:K_rk] = xp.asarray(_top_h_s[:, 0])
                if K_rk < TOP_K:
                    top_res[K_rk:] = xp.inf
                top_4[:K_rk] = xp.asarray(_top_h_s[:, 1:5])
                if n_swept > 0:
                    _ts = np.empty((K_rk, n_swept), dtype=np.float64)
                    _jswept = 0
                    for i in range(n_params):
                        if L_list[i] > 1:
                            _ts[:, _jswept] = _top_h_s[:, 5 + i]
                            _jswept += 1
                    top_swept[:K_rk] = xp.asarray(_ts)
                _persist_topk()
                print(f"[tune] fp64 rerank: top-{K_rk} "
                      f"re-evaluated and re-sorted", flush=True)
        except Exception as _rk_exc:
            print(f"[tune] fp64 rerank failed: {_rk_exc!r} — "
                  f"keeping fp32 ranking", flush=True)

    # ── Aggressive cleanup: free everything except the persisted
    # top-100 dataframe (already in st.session_state).  Drop refs
    # first so the GC can collect, then return memory pools to the
    # device / OS.  See _release_gpu_memory() for the sync+gc+flush
    # sequence — without it, async kernels in flight would prevent
    # the pool from actually returning blocks to the driver.
    try:
        S_mea_dev = None
        S_mea_dev_fp64 = None
        top_res = None; top_4 = None; top_swept = None
        scratch_res = None; scratch_4 = None; scratch_swept = None
        sweep_table_dev = None
        swept_indices_dev = None
        omega_dev = None
        static_cache.clear()
        if static_cache_fp64 is not static_cache:
            static_cache_fp64.clear()
    except Exception:
        pass
    _release_gpu_memory()

    # Persist total wall-clock run time so the results panel can show
    # "Evaluated in …" above the best-residual line (survives reruns).
    st.session_state[f"tune_elapsed_{topo_key}_{fname}"] = (
        _time.time() - _t_start)
