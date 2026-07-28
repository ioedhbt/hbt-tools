"""
models/base_ui/__init__.py — Package façade for the former (monolithic)
models/base_ui.py, now split across sibling modules:

  models/residuals.py          — S-parameter residual helpers
  models/smith_ui.py           — Smith chart + pad-sync UI
  models/fit_sections.py       — Cbex sweep tool + tau_total fit section
  models/param_groups.py       — Fine-tune diagram + Interactive Parameter
                                  Extraction expander
  models/tuning/ranges.py      — Physics-informed sweep ranges + hard limits
  models/tuning/preview.py     — Visual Tuning slider previews
  models/tuning/sweep.py       — Auto Tuning expander (render_tuning_expander)

This file re-exports every one of those symbols so ``from .base_ui import
X`` keeps working unchanged for every call site in the repo (verified
against every existing import — see the split commit message), and hosts
``SSMModelTemplate`` itself (the shared scaffold every concrete model
class mixes in), which was NOT moved: it is the natural "top" of the
former base_ui.py and stays where the public name ``base_ui`` already
points.
"""
from __future__ import annotations
import numpy as np
import streamlit as st

from ...helpers import params_hash
from tools.common.i18n import tr

# ── Re-exports — keep `from .base_ui import X` working for every X that
#    any module in the repo imports (verified via grep across tools/). ──
from ..residuals import ssm_residual, _port_residuals, _port_residuals_batch  # noqa: F401
from ..tuning.ranges import (PAD_SPECS, _PAD_KEYS, _PARASITIC_KEYS,  # noqa: F401
                              tune_hard_limits, informed_default_range,
                              _canonical_tune_key, _detect_low_perf_device)
from ..smith_ui import (render_smith_chart, render_smith_with_ftfmax,  # noqa: F401
                         smith_scale_controls, sync_pad_from_preov)
from ..param_groups import (render_finetune_diagram,  # noqa: F401
                             render_interactive_param_groups)
from ..tuning.preview import (render_visual_tuning_expander,  # noqa: F401
                               _chunked_simulate_batch_to_host)
from ..tuning.sweep import render_tuning_expander  # noqa: F401

# ════════════════════════════════════════════════════════════════════════════════
# SSMModelTemplate — shared scaffold for concrete model classes
# ════════════════════════════════════════════════════════════════════════════════
#
# This is a *mixin-style* parent: it provides default implementations of
# simulate_vec, simulate_batch, render_override_and_smith, and
# render_results_table that work for any model whose forward simulation
# follows the standard pad→lead→intrinsic chain.
#
# Concrete model classes inherit from BOTH this template AND AbstractSSMModel
# (the ABC contract in models/__init__.py).  Python's MRO combines them:
#
#     class MyModel(SSMModelTemplate, AbstractSSMModel):
#         NAME, SHORT, TOPOLOGY_CHAR = "...", "...", "..."
#         _INT_SPECS = [...]   # internal param specs (key, label, scale, unit, ...)
#         _EXT_SPECS = [...]   # external param specs
#         _Y_INT_VEC_FN     = staticmethod(_Y_int_xxx_vec)
#         _Y_INT_BATCH_FN   = staticmethod(_Y_int_xxx_batch)
#         _SIM_WRAP_VEC_FN  = staticmethod(_sim_wrap_vec)   # model's wrap
#         _SIM_WRAP_BATCH_FN= staticmethod(_sim_wrap_batch) # model's wrap
#
#         @classmethod
#         def _do_override_ui(cls, fname, calc_vals, cache_ctx=None): ...  # call model's _override_ui
#         @classmethod
#         def _render_topology(cls, all_p, fname): ...      # render topology illus
#         @classmethod
#         def _results_rows(cls, params): ...               # list[(sym, val, unit)]
#         @classmethod
#         def _render_results_trace(cls): pass              # optional formula trace
#
#         # Still required by AbstractSSMModel (model-specific math):
#         extract, simulate, reextract
#
# The template intentionally does NOT inherit AbstractSSMModel so it can live
# in base_ui.py without a circular import — concrete classes inherit both.

class SSMModelTemplate:
    """
    Shared scaffold for HBT small-signal model classes.

    Provides default bodies for the UI-side and forward-sim wrapper methods;
    concrete subclasses supply the math kernels and per-model UI bits via
    class attributes and a handful of classmethod hooks.  See the module
    docstring above for the contract.
    """

    # Subclasses must set these:
    _INT_SPECS: list      = []
    _EXT_SPECS: list      = []
    _Y_INT_VEC_FN         = None    # staticmethod or plain function
    _Y_INT_BATCH_FN       = None
    _SIM_WRAP_VEC_FN      = None
    _SIM_WRAP_BATCH_FN    = None
    # Pad-spec list used by the tuning expander.  Default is the shared
    # PAD_SPECS; override to customise displayed pad-parameter labels
    # (e.g. XuModel relabels Cpce → "Cpce / Cpad").
    _TUNING_PAD_SPECS     = None    # falls back to PAD_SPECS in render_override_and_smith

    # Default: no pre-bakeable sub-networks.  Concrete subclasses override
    # ``STATIC_SUBNETWORKS`` with a dict of {subnet_name: frozenset(deps)}.
    STATIC_SUBNETWORKS: dict = {}

    # ── Pre-bake truth table ──────────────────────────────────────────────────

    @classmethod
    def prebake_static_keys(cls, swept_keys):
        """Given the iterable of *swept* (i.e. changing) param keys, return
        the list of pre-bakeable sub-network names for this model.  A sub-
        network is pre-bakeable iff none of the keys it depends on are
        swept.  Caller can use this to decide which entries to populate
        in the ``cache`` argument to ``simulate_batch``."""
        if not cls.STATIC_SUBNETWORKS:
            return []
        swept_set = set(swept_keys)
        return [name for name, deps in cls.STATIC_SUBNETWORKS.items()
                if not (deps & swept_set)]

    @classmethod
    def build_static_cache(cls, all_p, freq, *, xp=None, swept_keys=()):
        """Build the ``cache`` dict (matching what ``_sim_wrap_batch`` reads)
        for the given fixed-param baseline and the set of params being
        swept.  Concrete subclasses override to fill in model-specific
        sub-networks (Zbe / Zbc / alpha / Y_extr / etc.).

        Default implementation only handles the shared sub-networks
        (``Y_pad``, ``Z_ser``) — enough to give the live mode a meaningful
        speed-up even on models without intrinsic pre-bake support.
        """
        from ...helpers.deembed_math import build_Y_pad_batch, build_Z_ser_batch
        if xp is None:
            xp = np
        omega = xp.asarray(2 * np.pi * np.asarray(freq), dtype=float)
        N = int(omega.shape[0])
        cache: dict = {"omega": omega, "_cdtype": np.complex128}
        static_p = dict(all_p)
        pbk = set(cls.prebake_static_keys(swept_keys))
        if "Y_pad" in pbk and not (cls.STATIC_SUBNETWORKS.get("Y_pad", set()) & set(swept_keys)):
            try:
                cache["Y_pad"] = build_Y_pad_batch(static_p, omega, 1, N, xp)
            except Exception:
                pass
        if "Z_ser" in pbk and not (cls.STATIC_SUBNETWORKS.get("Z_ser", set()) & set(swept_keys)):
            try:
                cache["Z_ser"] = build_Z_ser_batch(static_p, omega, 1, N, xp)
            except Exception:
                pass
        # Concrete subclasses extend via _build_intrinsic_static_cache().
        try:
            cls._build_intrinsic_static_cache(static_p, omega, cache, xp, pbk)
        except (AttributeError, NotImplementedError):
            pass
        return cache

    @classmethod
    def _build_intrinsic_static_cache(cls, p, omega, cache, xp, prebakeable):
        """Override in concrete classes to populate intrinsic sub-networks
        (Y_extr / Zbe / Zbc / alpha / Ybe / Ybc / gm / T_int_planes / etc.).
        Default is no-op."""
        return

    # ── Forward simulation ────────────────────────────────────────────────────

    @classmethod
    def simulate_vec(cls, params, freq, z0=50.0, xp=None):
        """Vectorised simulate — no per-freq loop.  Pass xp=cupy for GPU."""
        if xp is None:
            xp = np
        return cls._SIM_WRAP_VEC_FN(cls._Y_INT_VEC_FN, params, freq, z0, xp)

    @classmethod
    def simulate_batch(cls, params, freq, z0=50.0, xp=None, cache=None):
        """Batched simulate over (param_combo × freq).  Pass xp=cupy for GPU.

        params dict values may be scalars or (B,) arrays.
        Returns (B, N_freq, 2, 2) on the *xp* device (no host transfer).
        Optional ``cache`` carries pre-computed constant sub-networks.

        Phase 2 opt-in
        --------------
        When the env var ``HBT_USE_RUST_SIM_BATCH=1`` is set AND the
        Rust crate is loaded AND a Rust kernel exists for this model's
        ``SHORT`` identifier, the CPU path (xp is numpy) routes through
        the Rust end-to-end batched simulator instead of the NumPy
        composition chain.  The CUDA path (xp is cupy) stays on cupy.

        Cache + Rust
        ------------
        When a non-empty ``cache`` is provided, Rust still runs (the
        whole composition is so much faster than NumPy that the cache
        savings can't beat it).  The cache is preserved for the NumPy
        FALLBACK path so a Rust failure (e.g. exotic pad mode raises
        ``NotImplementedError``) still gets the pre-bake speedup.

        The default behaviour (env var unset) is identical to before:
        CPU goes through ``cls._SIM_WRAP_BATCH_FN`` exactly as it does
        today, with no measurable overhead from the opt-in check.
        """
        if xp is None:
            xp = np

        # Phase 2 dispatch — opt-in, CPU-only.  The lazy import keeps
        # this branch zero-cost when Phase 2 isn't activated.
        if xp is np:
            try:
                from ...helpers.rust_kernels import (
                    HAS_RUST as _HAS_RUST,
                    SIM_FOR_TOPOLOGY as _SIM_FOR_TOPOLOGY,
                    _phase2_dispatch_enabled as _phase2_on,
                )
            except Exception:                              # pragma: no cover
                _HAS_RUST = False
                _SIM_FOR_TOPOLOGY = {}

                def _phase2_on() -> bool:
                    return False

            if _HAS_RUST and _phase2_on():
                rust_wrapper = _SIM_FOR_TOPOLOGY.get(cls.SHORT)
                if rust_wrapper is not None:
                    # NumPy fallback closure — preserves the ORIGINAL
                    # cache so the pre-bake speedup isn't lost if Rust
                    # raises (NotImplementedError for exotic pad modes
                    # is the common case).
                    _saved_cache = cache
                    def _np_fallback(_p, _f, _z0):
                        return cls._SIM_WRAP_BATCH_FN(
                            cls._Y_INT_BATCH_FN, _p, _f, _z0, np,
                            _saved_cache)
                    return rust_wrapper(params, freq, z0,
                                         np_fallback=_np_fallback)

        return cls._SIM_WRAP_BATCH_FN(cls._Y_INT_BATCH_FN, params, freq, z0, xp, cache)

    # ── Cached simulate-vec (used by render_override_and_smith) ───────────────

    @classmethod
    def _cached_simulate_vec(cls, all_p, freq, z0, fname):
        """Hash-cached wrapper around ``simulate_vec``.

        Skips re-simulation when the parameter dict + freq length match the
        last invocation.  Errors are caught and reported in-place; the cache
        is populated with a NaN array so the rest of the UI keeps rendering.
        """
        cache_key = f"sim_result_{cls.SHORT}_{fname}"
        hash_key  = f"sim_phash_{cls.SHORT}_{fname}"
        cur_hash  = params_hash({k: str(v) for k, v in
                                  {**all_p, "__nf": len(freq)}.items()})

        def _run():
            with st.spinner(tr(f"Simulating {cls.NAME}…",
                               f"正在模擬 {cls.NAME}…")):
                try:
                    return cls.simulate_vec(all_p, freq, z0)
                except Exception as e:
                    st.error(tr(f"Simulation error ({cls.NAME}): {e}",
                                f"模擬錯誤（{cls.NAME}）：{e}"))
                    return np.full((len(freq), 2, 2), np.nan + 0j)

        if st.session_state.get(hash_key) != cur_hash:
            S_sim = _run()
            st.session_state[cache_key] = S_sim
            st.session_state[hash_key]  = cur_hash
        else:
            S_sim = st.session_state.get(cache_key)
            if S_sim is None or S_sim.shape[0] != len(freq):
                S_sim = _run()
                st.session_state[cache_key] = S_sim
                st.session_state[hash_key]  = cur_hash
        return S_sim

    # ── UI scaffolding ────────────────────────────────────────────────────────

    @classmethod
    def render_results_table(cls, params):
        """Render JUST the scalar-parameter dataframe.

        The formula-trace expander is rendered separately via
        :meth:`render_formula_trace` so the call site can place both
        side-by-side in their own columns / expanders.
        """
        import pandas as pd
        rows = cls._results_rows(params)
        st.dataframe(pd.DataFrame(rows, columns=["Symbol", "Value", "Unit"]),
                     width="stretch", hide_index=True)

    @classmethod
    def render_formula_trace(cls):
        """Render the 📐 Full formula trace expander, if the subclass
        provides one.  Default dispatches to the private template hook
        ``_render_results_trace`` — subclasses (Cheng T / π, Xu T)
        override that hook to supply their own LaTeX dependency chain.
        Models without a trace (e.g. Degachi) silently render nothing.
        """
        cls._render_results_trace()

    @classmethod
    def _render_results_trace(cls):
        """Optional: render an expander with the full extraction/sim formula chain.
        Default is a no-op; override to add a 📐 trace expander."""
        return

    @classmethod
    def has_formula_trace(cls) -> bool:
        """True iff the subclass overrides ``_render_results_trace``.

        Used by callers that want to drop the side-by-side layout when
        no formula trace is available (so the parameter table can use
        full width instead of leaving an empty right column).
        """
        return (cls._render_results_trace.__func__
                is not SSMModelTemplate._render_results_trace.__func__)

    @classmethod
    def render_override_and_smith(cls, fname, S_raw, freq, z0,
                                  para_eff, extract_result, *,
                                  show_tuning: bool = True,
                                  prefer_calc_vals: bool = False,
                                  show_cache_banner: bool = True, **kwargs):
        """
        Standard override-UI → cached sim → Smith chart → topology illustration
        → matplotlib Smith expander → tuning expander.

        ``show_tuning=False`` (used by the Extraction page) suppresses the
        Visual Tuning + Auto Tuning expanders — tuning now lives on the
        Simulation & Fitting page.

        ``prefer_calc_vals=True`` (set by the Simulation & Fitting page when a
        fresh extraction handoff arrives) makes calc_vals win over both stale
        widget state and any cached fit: the sim widget keys and sync hashes
        for this (model, fname) are cleared so ``_override_ui`` reseeds
        everything from calc_vals; the cache auto-restore is skipped (the
        banner + "📌 Use cache" button inside Fine-tune remain available).

        ``show_cache_banner=False`` suppresses the "📌 Cached fit for ..."
        banner entirely — used by RF_simulator.py's fit-mode call, which
        renders its own compact cache pill in the header row instead.  The
        "📌 Use cache" button inside the Fine-tune expander (see cache_ctx
        below) still works regardless of this flag.

        Concrete subclasses supply the per-model UI pieces via:
          cls._do_override_ui(fname, calc_vals, cache_ctx=None)  → all_p
              cache_ctx is {"has_cache": bool, "ts": str|None, "req_key": str}
              — forwarded to the module-level _override_ui so it can render a
              "📌 Use cache" button that sets session_state[req_key] = True
              and reruns; this function applies the cache on the next render
              before any sim widgets instantiate (see _cache_req_key below).
          cls._render_topology(all_p, fname)     → render illustration
          cls._INT_SPECS / _EXT_SPECS            → tuning specs

        Also performs persistent fit-cache restore (on first render after
        upload) and auto-save (when the user has fine-tuned vs. extraction
        defaults).  See helpers/fit_cache.py.
        """
        from ...helpers.fit_cache import (get_fit, get_fit_timestamp,
                                          save_fit, delete_fit, differs_from)

        params, _arrays = extract_result
        calc_vals = {**para_eff, **params}

        pad_specs   = cls._TUNING_PAD_SPECS if cls._TUNING_PAD_SPECS is not None else PAD_SPECS
        all_specs   = pad_specs + cls._EXT_SPECS + cls._INT_SPECS
        int_ext_keys = [k for k, *_ in cls._EXT_SPECS + cls._INT_SPECS]
        scale_for    = {k: sc for k, _, sc, *_ in all_specs}

        # ── Cache restore — runs on the first call after Run SSM is clicked
        #    (applied_key resides in session_state, which IOED's "Clear SSM
        #    results" button wipes for the file; so a fresh Run SSM cycle
        #    re-applies the cache).
        cached        = get_fit(fname, cls.SHORT)
        cached_ts     = get_fit_timestamp(fname, cls.SHORT)
        applied_key   = f"cache_applied_{cls.SHORT}_{fname}"
        dismissed_key = f"cache_dismissed_{cls.SHORT}_{fname}"

        # ── Fresh-handoff priority — the Sim & Fitting page sets this when an
        #    extraction handoff just arrived: the forwarded values must win
        #    over both stale widget state and the cached fit (previously the
        #    cache silently clobbered a fresh extraction in a new session).
        if prefer_calc_vals:
            for _k in list(st.session_state.keys()):
                if (_k.startswith(f"sim_{cls.SHORT}_")
                        and _k.endswith(f"_{fname}")):
                    del st.session_state[_k]
            st.session_state.pop(f"sim_synchash_{cls.SHORT}_{fname}", None)
            st.session_state.pop(f"smith_pad_synced_{cls.SHORT}_{fname}", None)
            st.session_state[applied_key] = True

        # Helper closure — write cached values (including pad) into the
        # fine-tune session_state and align the sync hashes so subsequent
        # renders don't overwrite us.
        #
        # Pad strategy: prefer cached pad if the cache has it (preserves
        # the user's fine-tuned pad, e.g. Rpb/Rpc/Rpe from Cold-HBT that
        # don't get re-extracted from Step 1b alone).  Fall back to
        # `para_eff` only when the cache has no value for that key
        # (older cache files written with the no-pad filter, or never
        # touched).  The matching `main_ssm_extraction` change ensures
        # `calc_vals[pad] = para_eff[pad]` (live Step 1), so the pad
        # sync hash below stays aligned with para_eff and
        # `sync_pad_from_preov` doesn't clobber the cached pad on the
        # next render.
        def _apply_cached_to_simstate(cached_dict):
            if not isinstance(cached_dict, dict):
                return
            # Pad: cached value first, else live para_eff.
            for key, _, sc, *_ in pad_specs:
                cv = cached_dict.get(key)
                if not isinstance(cv, (int, float)):
                    cv = para_eff.get(key, 0.0)
                st.session_state[f"sim_{cls.SHORT}_{key}_{fname}"] = float(cv) * sc
            # Int/ext: from cache.
            for k, v in cached_dict.items():
                if (k in scale_for and k not in _PAD_KEYS
                        and isinstance(v, (int, float))):
                    st.session_state[f"sim_{cls.SHORT}_{k}_{fname}"] = (
                        float(v) * scale_for[k])
            # Intrinsic sync hash: hash calc_vals[int_ext].  Since
            # main_ssm filters pad out of `params`, calc_vals[int_ext]
            # equals cached[int_ext] — so _override_ui's intrinsic sync
            # is a no-op.
            st.session_state[f"sim_synchash_{cls.SHORT}_{fname}"] = params_hash(
                {k: str(round(float(calc_vals.get(k, 0.0)), 15))
                 for k in int_ext_keys})
            # Pad sync hash matches live para_eff (NOT cached pad), so
            # sync_pad_from_preov stays a no-op until Step 1 actually
            # changes.  This is what protects the just-loaded cached pad
            # values from being overwritten by para_eff on the next
            # render.
            st.session_state[f"smith_pad_synced_{cls.SHORT}_{fname}"] = params_hash(
                {k: para_eff.get(k, 0.0) for k in _PAD_KEYS})

        # ── "Use cache" request — set by the "📌 Use cache" button inside the
        #    Fine-tune expander (module-level _override_ui in cheng/xu/kunyang).
        #    Runs before any sim widgets instantiate this run, so writing their
        #    session_state values here is legal (same pattern as the cache
        #    auto-restore below).
        _cache_req_key = f"cache_apply_request_{cls.SHORT}_{fname}"
        if st.session_state.pop(_cache_req_key, False) and cached:
            _apply_cached_to_simstate(cached)
            st.session_state[applied_key] = True

        # ── Cache restore — runs on the first call after Run SSM is clicked
        #    (applied_key resides in session_state, which IOED's "Clear SSM
        #    results" button wipes for the file; so a fresh Run SSM cycle
        #    re-applies the cache).
        #
        #    `applied_key` is set on this first render *whether or not* a cache
        #    existed.  Critically, when NO cache exists yet, the user's first
        #    fine-tune edit triggers the auto-save below, which creates a cache
        #    file.  If `applied_key` were still unset on the next render, this
        #    branch would see the freshly-written cache and re-apply it —
        #    clobbering the user's in-progress edit with the previous value
        #    (the "have to type every value twice" bug).  Setting the flag now
        #    guarantees the override widgets own session_state from here on.
        # Widget keys can be GC'd by a page switch (see the keep-alive in
        # IOED_Tool_Web.py); if that happened, re-apply the cache — thanks to
        # auto-save the cache IS the user's last-seen state — instead of
        # letting every input recreate at 0.
        _widgets_missing = any(
            f"sim_{cls.SHORT}_{k}_{fname}" not in st.session_state
            for k in int_ext_keys)
        if (not prefer_calc_vals
                and (not st.session_state.get(applied_key) or _widgets_missing)
                and not st.session_state.get(dismissed_key)):
            if cached:
                _apply_cached_to_simstate(cached)
            st.session_state[applied_key] = True

        # The container renders unconditionally so the element tree above the
        # fine-tune expander stays stable whether or not the banner shows —
        # conditional siblings above an expander reset its client-side open
        # state (the "expander always closes" bug).
        _banner_slot = st.container()
        if show_cache_banner and cached_ts and not st.session_state.get(dismissed_key):
            with _banner_slot:
                st.markdown(
                    f"<small>📌 Cached fit for <b>{cls.NAME}</b> — saved {cached_ts}"
                    " <span class='hbt-help' title='A fit saved earlier for this"
                    " file and model. It is applied automatically on entry -"
                    " except right after a handoff from Extraction, where the"
                    " freshly extracted values take priority. Load it into the"
                    " fields below with “📌 Use cache” inside the Fine-tune"
                    " expander.'>?</span></small>",
                    unsafe_allow_html=True)

        all_p = cls._do_override_ui(fname, calc_vals, cache_ctx={
            "has_cache": bool(cached), "ts": cached_ts, "req_key": _cache_req_key})

        S_sim = cls._cached_simulate_vec(all_p, freq, z0, fname)

        # Build the modeled S2P bytes once here so render_smith_with_ftfmax
        # can wire a "📥 modeled S2P" button next to the xlsx download under
        # the Smith chart.  This replaces the standalone "Download Modeled
        # DUT S2P" section that used to live at the bottom of the SSM tab.
        from ...helpers import write_s2p
        from pathlib import Path as _Path
        s2p_params = {}
        for _k in ("Cpbe", "Cpce", "Cpbc"):
            s2p_params[_k] = f"{para_eff.get(_k, 0.0) * 1e15:.4f} fF"
        for _k_raw, _label in (("Rpb", "Rb"), ("Rpc", "Rc"), ("Rpe", "Re")):
            s2p_params[_label] = f"{para_eff.get(_k_raw, 0.0):.4f} Ω"
        for _k in ("Lb", "Lc", "Le"):
            s2p_params[_k] = f"{para_eff.get(_k, 0.0) * 1e12:.4f} pH"
        s2p_bytes = write_s2p(
            freq, S_sim,
            title=f"DUT {cls.NAME} — {_Path(fname).stem}",
            params=s2p_params,
        )
        s2p_filename = f"model_dut_{_Path(fname).stem}_{cls.SHORT}.s2p"

        sc = smith_scale_controls(fname, cls.SHORT)
        render_smith_with_ftfmax(S_raw, S_sim, freq,
                                 model_name=cls.NAME, model_short=cls.SHORT,
                                 fname=fname, scales=sc,
                                 s2p_bytes=s2p_bytes,
                                 s2p_filename=s2p_filename)

        # τ_total + calculated fmax expander (HBT T/π models only — Kun-Yang
        # HEMT has no Cbcx/Cbc/Rbi/Rb to evaluate the fmax formula).
        if cls.SHORT in ("T", "pi", "XuT"):
            from ...ssm_plots import render_tau_fmax_expander
            _CBC = float(all_p.get("Cbcx", 0.0)) + float(all_p.get("Cbc", 0.0))
            _Rbb = float(all_p.get("Rbi", 0.0)) + float(all_p.get("Rpb", 0.0))
            if cls.SHORT == "pi":
                _tau_sum = float(all_p.get("tau", 0.0))
                _tau_lbl, _tau_tex = "τ", r"\tau"
            else:
                _tau_sum = float(all_p.get("tauB", 0.0)) + float(all_p.get("tauC", 0.0))
                _tau_lbl, _tau_tex = "τB + τC", r"\tau_B+\tau_C"
            render_tau_fmax_expander(key=f"taufmax_{cls.SHORT}_{fname}",
                                     freq=freq, S_meas=S_raw, S_model=S_sim,
                                     CBC=_CBC, Rbb=_Rbb,
                                     tau_sum=_tau_sum, tau_sum_label=_tau_lbl,
                                     tau_sum_tex=_tau_tex,
                                     extrap_key=f"ftfmax_card_{cls.SHORT}_{fname}")

        # Persist the *current* (post-override) param dict so the Complete
        # Parameter Summary can read live values instead of extraction-time ones.
        st.session_state[f"current_p_{cls.SHORT}_{fname}"] = dict(all_p)

        # ── Auto-save — fires whenever the fine-tune section's `all_p`
        # differs from the current "baseline" snapshot.  The baseline is
        # `calc_vals` (extraction defaults) overlaid with any cached
        # values — so after a cache restore, `all_p == baseline` and no
        # redundant save fires; only genuine user edits beyond the
        # cached state trigger a write.  Pad is included so users can
        # fine-tune Rpb/Rpc/Rpe (etc.) and have those values persist.
        baseline = dict(calc_vals)
        if isinstance(cached, dict):
            for _k, _v in cached.items():
                if _k in scale_for and isinstance(_v, (int, float)):
                    baseline[_k] = float(_v)
        check_keys = _PAD_KEYS + int_ext_keys
        # Refuse to persist a degenerate/uninitialised state: a genuine fit
        # has (nearly) all intrinsic/extrinsic values nonzero, while a
        # GC-wiped widget set is all zeros except whatever the user just
        # touched — exactly the state that poisoned the cache before.
        _n_nonzero = sum(
            1 for k in int_ext_keys
            if isinstance(all_p.get(k), (int, float)) and float(all_p[k]) != 0.0)
        if (_n_nonzero >= max(2, len(int_ext_keys) // 2)
                and not st.session_state.get(dismissed_key)
                and differs_from(all_p, baseline, keys=check_keys)):
            save_fit(fname, cls.SHORT, dict(all_p))

        # Topology illustration in its own expander (collapsed by default).
        # For no-parasitics models we also pass the user's *customized* Smith
        # chart (rebuilt from session_state via phase="chart", return_png=True —
        # creates no widgets) so the topology view can overlay it bottom-right.
        from ...ssm_plots import render_matplotlib_smith
        _smith_png = None
        if S_sim is not None:
            try:
                _smith_png = render_matplotlib_smith(
                    S_raw, S_sim, fname, cls.SHORT,
                    default_multiplier=sc,
                    phase="chart", freq_hz=freq, return_png=True)
            except Exception:                            # noqa: BLE001
                _smith_png = None
        with st.container(key=f"hbt_exp_view_topo_{cls.SHORT}"), \
             st.expander(tr("🖼️ Topology illustration", "🖼️ 拓樸示意圖"),
                         expanded=False):
            # Models registered with a built-in custom-model preset (Cheng
            # T/π, Xu T, Kun-Yang HEMT — see svg_topology._PRESET_LABEL) can
            # switch to a live SVG schematic that omits every zero-valued
            # component.  Degachi (no preset) has no ``_SVG_TOPOLOGY`` attr,
            # so it keeps the plain PNG-template path unchanged.
            svg_mode = False
            if getattr(cls, "_SVG_TOPOLOGY", False):
                svg_mode = st.toggle(
                    tr("Simplified schematic (non-zero components only)",
                       "簡化示意圖（僅顯示非零元件）"),
                    key=f"topo_svg_{cls.SHORT}_{fname}",
                    help=tr("ON: a live SVG schematic built from the model "
                            "topology, showing only components with a "
                            "non-zero current value. OFF: the standard "
                            "labeled template illustration.",
                            "開啟：由模型拓樸即時產生的 SVG 示意圖，"
                            "僅顯示目前為非零值的元件。"
                            "關閉：標準的標籤範本示意圖。"))
            if svg_mode:
                from ..svg_topology import render_svg_topology
                render_svg_topology(cls.SHORT, all_p, fname)
            else:
                cls._render_topology(all_p, fname, smith_png=_smith_png)

        # NOTE: the Open/Short pad-dummy schematics (svg_topology.
        # render_pad_topology) deliberately do NOT appear here.  A device
        # model's own topology illustration above already draws its pad
        # parasitics in place, so a second pad-only drawing was redundant on
        # every model page (and on the custom model).  The pad schematics now
        # live only where they are the subject: the RF simulator's "Open and
        # Short Pad" model (tools/rf/simulator.py).

        # Smith chart (matplotlib) + its controls live in a single
        # expander, rendered side-by-side — matches the RF simulator
        # layout (RF_simulator.py "🍩 Smith Chart (Matplotlib)").  The
        # split-call pattern (controls in the right column, chart in the
        # left) preserves the order-of-operations requirement that
        # widgets render BEFORE the chart so session_state is fresh when
        # the chart half reads it.
        with st.container(key=f"hbt_exp_view_mplsmith_{cls.SHORT}"), \
             st.expander(tr("🍩 Smith Chart (Matplotlib)",
                            "🍩 Smith 圖 (Matplotlib)"), expanded=False):
            col_mpl_left, col_mpl_right = st.columns([1.2, 1])
            with col_mpl_right:
                render_matplotlib_smith(S_raw, S_sim, fname, cls.SHORT,
                                         default_multiplier=sc,
                                         phase="controls", freq_hz=freq)
            with col_mpl_left:
                render_matplotlib_smith(S_raw, S_sim, fname, cls.SHORT,
                                         default_multiplier=sc,
                                         phase="chart", freq_hz=freq)

        if show_tuning:
            render_visual_tuning_expander(cls, all_p, S_raw, freq, z0,
                                           all_specs,
                                           fname, cls.SHORT)
            render_tuning_expander(
                cls, all_p, S_raw, freq, z0, all_specs, fname, cls.SHORT,
                default_fit_keys=(
                    [k for k, *_ in cls._EXT_SPECS + cls._INT_SPECS]
                    + ["Rpb", "Rpc", "Rpe"]))
        return S_sim
