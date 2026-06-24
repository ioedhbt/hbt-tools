"""
ui_fit.py — overlay a custom model on a *measured* device and tune it.

Surfaced inside the HBT SSM Extraction page (as the "🧩 Custom model" entry in
the per-DUT model selection).  It reuses the **same** result UI as the built-in
models:

* :func:`render_smith_with_ftfmax` — Total / per-trace residual above the Smith
  chart + fT/fmax card, with the standard download / copy buttons below;
* the topology illustration (the custom schematic) with PNG download **and a
  copy-image button**;
* the publication matplotlib Smith chart (two-column: chart left, controls
  right);
* :func:`render_tuning_expander` — the *same* grid-sweep auto-tuning the other
  models use (the user picks each parameter's Min/Step/Max), backed by a generic
  ``simulate_batch`` adapter over the netlist solver.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from .core import CustomModel, simulate_custom_model, load_model
from .schematic import (render_schematic, svg_to_png, copy_image_button,
                        svg_pixel_height)
from ..helpers import write_s2p
from ..models.base_ui import (render_smith_with_ftfmax,
                              render_tuning_expander, smith_scale_controls)
from ..ssm_plots import render_matplotlib_smith, render_tau_fmax_expander

# kind → (SI→display scale, sensible non-degenerate start) for value inputs +
# the tuning sweep's display units.
_FIT_SPEC = {
    "C":     (1e15, 10e-15),
    "R":     (1.0, 50.0),
    "L":     (1e12, 10e-12),
    "gm":    (1e3, 0.1),
    "tau":   (1e12, 0.5e-12),
    "alpha": (1.0, 0.98),
}
_UNIT = {"C": "fF", "R": "Ω", "L": "pH", "gm": "mS", "tau": "ps", "alpha": ""}
_TOPO = "custom"


# ════════════════════════════════════════════════════════════════════════════
def _clear_tuning_state(fname: str) -> None:
    """Drop the shared tuning expander's cached state **and** the value-input
    widgets for this DUT when a new model is loaded.  The results DataFrame is
    keyed only by (topo, fname), so a different custom model's leftover sweep —
    with different parameter columns — would otherwise make the sensitivity /
    "use best" lookups KeyError on a column the new model lacks; the previous
    model's ``sim_*`` value keys are likewise stale.  Matching is by exact key
    boundaries (no substring) so one DUT name can't clobber another's state."""
    exact = {f"tune_df_{_TOPO}_{fname}", f"tune_elapsed_{_TOPO}_{fname}",
             f"_rust_active_cached_{_TOPO}_{fname}"}
    sweep_sfx = ("chk", "min", "step", "max")
    for k in list(st.session_state):
        if k in exact:
            st.session_state.pop(k, None)
        elif (k.startswith(f"tune_{_TOPO}_")
              and any(k.endswith(f"_{fname}_{s}") for s in sweep_sfx)):
            st.session_state.pop(k, None)            # per-param sweep widgets
        elif k.startswith(f"sim_{_TOPO}_") and k.endswith(f"_{fname}"):
            st.session_state.pop(k, None)            # stale value inputs


def install_fit_model(model: CustomModel) -> None:
    """Install ``model`` as the active fit model and bump the model token so the
    fit view clears any stale tuning/value state.  Shared by the uploader and
    the build view's "Send to Fit" shortcut."""
    st.session_state["cmf_model"] = model
    st.session_state["cmf_model_token"] = \
        st.session_state.get("cmf_model_token", 0) + 1
    st.session_state.pop("cmf_base", None)


def _load_model(fname: str) -> CustomModel | None:
    up = st.file_uploader("Upload a custom model .json", type=["json"],
                          key=f"cmf_up_{fname}")
    if up is not None:
        data = up.getvalue()
        sig = (up.name, len(data), hash(data))
        if st.session_state.get(f"cmf_sig_{fname}") != sig:
            try:
                install_fit_model(load_model(data))
                st.session_state[f"cmf_sig_{fname}"] = sig
            except Exception as exc:                       # noqa: BLE001
                st.error(f"Could not read that .json: {exc}")
    return st.session_state.get("cmf_model")


def _val_key(cid: str, fname: str) -> str:
    # Matches the key render_tuning_expander's "Use best values" writes to,
    # i.e. f"sim_{topo}_{key}_{fname}", so applied sweeps flow into the inputs.
    return f"sim_{_TOPO}_{cid}_{fname}"


def _value_inputs(model: CustomModel, fname: str) -> dict:
    """Render component value inputs grouped outside→inside (parasitic → lead L
    → access R → extrinsic → port/delay → intrinsic), clean names; return
    {id: SI value}."""
    values: dict = {}
    per_row = 4
    for title, items in model.grouped_value_specs():
        st.markdown(f"**{title}**")
        for i in range(0, len(items), per_row):
            row = items[i:i + per_row]
            cs = st.columns(len(row))
            for col, (cid, kind, name) in zip(cs, row):
                scale, start = _FIT_SPEC.get(kind, (1.0, 1.0))
                unit = _UNIT.get(kind, "")
                k = _val_key(cid, fname)
                if k not in st.session_state:
                    st.session_state[k] = float(start * scale)
                elif st.session_state[k] < 0:     # clamp a negative (e.g. a sweep
                    st.session_state[k] = 0.0     # "best") up to the 0 floor
                disp = col.number_input(f"{name} ({unit})" if unit else name,
                                        key=k, format="%.4f", step=0.0,
                                        min_value=0.0)
                values[cid] = float(disp) / scale
    return values


def _to_host(v):
    """cupy → numpy passthrough (the tuning sweep may hand us device arrays)."""
    if type(v).__module__.split(".")[0] == "cupy":
        import cupy
        return cupy.asnumpy(v)
    return np.asarray(v, dtype=float)


def _make_adapter(model: CustomModel):
    """A minimal AbstractSSMModel-shaped class wrapping ``model`` so the shared
    ``render_tuning_expander`` (grid sweep + residual ranking + "Use best
    values") works unchanged.  Implements ``simulate`` (single param set) and
    ``simulate_batch`` (a generic per-combo loop over the netlist solver)."""

    class _CustomModelAdapter:
        NAME = f"Custom · {model.name}"
        SHORT = _TOPO
        TOPOLOGY_CHAR = "T" if model.intrinsic_type == "T" else "pi"
        _model = model

        @classmethod
        def simulate(cls, params, freq, z0=50.0):
            return simulate_custom_model(cls._model, freq, dict(params), z0)

        @classmethod
        def simulate_batch(cls, params, freq, z0=50.0, xp=None, cache=None):
            xp = np if xp is None else xp
            freq = np.asarray(freq, dtype=float)
            N = freq.size
            arrs = {k: _to_host(v) for k, v in params.items()}
            shapes = [a.shape for a in arrs.values()]
            inner = np.broadcast_shapes(*shapes) if shapes else ()
            if inner == ():                       # all scalars → single sim
                vi = {k: float(a) for k, a in arrs.items()}
                return xp.asarray(simulate_custom_model(cls._model, freq, vi, z0))
            B = int(np.prod(inner))
            bp = {k: np.broadcast_to(a, inner).reshape(B) for k, a in arrs.items()}
            out = np.empty((B, N, 2, 2), dtype=complex)
            for i in range(B):
                vi = {k: float(bp[k][i]) for k in bp}
                out[i] = simulate_custom_model(cls._model, freq, vi, z0)
            return xp.asarray(out.reshape(tuple(inner) + (N, 2, 2)))

    return _CustomModelAdapter


# ════════════════════════════════════════════════════════════════════════════
def render_custom_fit(fname: str, S_meas: np.ndarray, freq: np.ndarray,
                      z0: float) -> None:
    st.markdown("### 🧩 Custom model — overlay on the measured device & tune")
    st.caption("Upload a custom-model topology (built in the RF Forward "
               "Simulator), set starting values, read the residual, then use "
               "the **Auto Tuning** expander (same grid sweep as the other "
               "models) to fit this DUT.")
    from .ui_build import fire_pending_download
    fire_pending_download()           # download a model just "sent" from build
    model = _load_model(fname)
    if model is None:
        st.info("⬆️ Upload a custom model `.json` (or send one from the **Build "
                "/ modify** view) to overlay it on this device.")
        return
    # Whenever the active model changes (upload *or* a "Send to Fit" from the
    # build view), drop the previous model's stale tuning / value state.
    tok = st.session_state.get("cmf_model_token", 0)
    if st.session_state.get(f"cmf_seen_{fname}") != tok:
        _clear_tuning_state(fname)
        st.session_state[f"cmf_seen_{fname}"] = tok
    st.success(f"Loaded **{model.name}** · {model.device} · intrinsic "
               f"{'π' if model.intrinsic_type == 'Pi' else 'T'}")

    with st.expander("⚙️ Component values", expanded=True):
        values = _value_inputs(model, fname)

    try:
        S_sim = simulate_custom_model(model, freq, values, z0)
    except Exception as exc:                               # noqa: BLE001
        st.error(f"Simulation failed: {exc}")
        return
    if not np.all(np.isfinite(S_sim)):
        st.warning("Some simulated points are non-finite — check for missing "
                   "values before tuning.")

    # ── Smith + fT/fmax + Total/per-trace residual (same UI as other models) ──
    s2p = write_s2p(freq, S_sim, title=f"Custom fit {model.name}",
                    params={"intrinsic": model.intrinsic_type})
    sc = smith_scale_controls(fname, _TOPO)     # per-trace Smith multipliers
    render_smith_with_ftfmax(
        S_meas, S_sim, freq,
        model_name=f"Custom · {model.name}", model_short=_TOPO, fname=fname,
        scales=sc,
        s2p_bytes=s2p, s2p_filename=f"{model.name or 'custom'}_sim.s2p")

    # ── Calculated τ_total and fmax (custom: enter C_BC / R_bb manually) ──────
    # The fT/fmax follows the Smith-card's extrapolation selection.
    render_tau_fmax_expander(
        key=f"cmf_taufmax_{fname}", freq=freq, S_meas=S_meas, S_model=S_sim,
        CBC=0.0, Rbb=0.0, tau_sum=None, tau_sum_label="τ", tau_sum_tex=r"\tau",
        extrap_key=f"ftfmax_card_{_TOPO}_{fname}")

    # ── Topology illustration (value-aware layout so values never overlap) ───
    svg = render_schematic(model, values)
    with st.expander("🖼️ Topology (custom schematic)", expanded=False):
        st.iframe(svg, height=svg_pixel_height(svg) + 12)
        png = svg_to_png(svg, zoom=2)
        dc = st.columns(2)
        if png is not None:
            dc[0].download_button("🖼️ Download PNG", data=png,
                                  file_name=f"{model.name or 'custom'}.png",
                                  mime="image/png", key=f"cmf_png_{fname}",
                                  width="stretch")
            copy_image_button(png, container=dc[1], label="📋 copy image")
        else:
            dc[0].caption("PNG export needs `rsvg-convert` / `cairosvg`.")

    # ── Publication matplotlib Smith chart (chart left, controls right) ──────
    with st.expander("🍩 Smith Chart (Matplotlib)", expanded=False):
        c_left, c_right = st.columns([1.2, 1])
        with c_right:
            render_matplotlib_smith(S_meas, S_sim, fname, _TOPO,
                                    default_multiplier=sc,
                                    phase="controls", freq_hz=freq)
        with c_left:
            render_matplotlib_smith(S_meas, S_sim, fname, _TOPO,
                                    default_multiplier=sc,
                                    phase="chart", freq_hz=freq)

    # ── Auto Tuning — the *same* grid-sweep expander the other models use ────
    # Clean names + outside→inside order (matches the value-input grouping).
    tuning_specs = [
        (cid, name, _FIT_SPEC.get(kind, (1.0, 0))[0], _UNIT.get(kind, ""))
        for _title, items in model.grouped_value_specs()
        for cid, kind, name in items]
    render_tuning_expander(_make_adapter(model), values, S_meas, freq, z0,
                           tuning_specs, fname, _TOPO)
