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

from .core import (CustomModel, simulate_custom_model,
                   simulate_custom_model_batch, compile_plan, load_model)
from .schematic import (render_schematic, svg_to_png, copy_image_button,
                        svg_pixel_height)
from ..helpers import write_s2p
from ..helpers.rust_kernels import sim_custom_batch as _rust_sim_custom
from ..models.base_ui import (render_smith_with_ftfmax, render_tuning_expander,
                              render_visual_tuning_expander, smith_scale_controls)
from ..ssm_plots import render_matplotlib_smith, render_tau_fmax_expander
from ._i18n import tr

# Display translations for the grouped value-input section titles, which come
# from CustomModel.grouped_value_specs() in English.
_GROUP_TITLE_ZH = {
    "Parasitic pad capacitances": "寄生焊墊電容",
    "Lead inductance": "引線電感",
    "Access resistance": "接觸電阻",
    "Extrinsic capacitances": "外質電容",
    "Port / delay extras": "埠 / 延遲附加元件",
    "Intrinsic core": "本質核心",
}

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

# The b–c junction's resistance (Rbc / Rgd) sits in the 10²–10³ kΩ range on a
# real device, so that *position* gets a kΩ input instead of the generic Ω one
# — matching the built-in Cheng/Xu specs.  Keyed by element id, not name, so a
# renamed component still gets the right unit.
_KOHM_SCALE, _KOHM_UNIT = 1e-3, "kΩ"


def _kohm_ids(model: CustomModel) -> frozenset[str]:
    """Ids of the components displayed in kΩ — every R in the intrinsic b–c
    junction (``BI→CI``), whatever the user named it."""
    return frozenset(e.id for e in model.intrinsic_bc.elements() if e.kind == "R")


def _disp_spec(cid: str, kind: str, kohm_ids: frozenset[str]) -> tuple:
    """``(SI→display scale, non-degenerate start in SI, unit)`` for one
    component — the kind default, overridden by position where it applies."""
    scale, start = _FIT_SPEC.get(kind, (1.0, 1.0))
    if kind == "R" and cid in kohm_ids:
        return _KOHM_SCALE, start, _KOHM_UNIT
    return scale, start, _UNIT.get(kind, "")


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
    up = st.file_uploader(tr("Upload a custom model .json", "上傳自訂模型 .json"),
                          type=["json"], key=f"cmf_up_{fname}")
    if up is not None:
        data = up.getvalue()
        sig = (up.name, len(data), hash(data))
        if st.session_state.get(f"cmf_sig_{fname}") != sig:
            try:
                install_fit_model(load_model(data))
                st.session_state[f"cmf_sig_{fname}"] = sig
            except Exception as exc:                       # noqa: BLE001
                st.error(tr(f"Could not read that .json: {exc}",
                            f"無法讀取該 .json：{exc}"))
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
    kohm = _kohm_ids(model)
    for title, items in model.grouped_value_specs():
        st.markdown(f"**{tr(title, _GROUP_TITLE_ZH.get(title, title))}**")
        for i in range(0, len(items), per_row):
            row = items[i:i + per_row]
            cs = st.columns(len(row))
            for col, (cid, kind, name) in zip(cs, row):
                scale, start, unit = _disp_spec(cid, kind, kohm)
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


def _make_adapter(model: CustomModel):
    """A minimal AbstractSSMModel-shaped class wrapping ``model`` so the shared
    ``render_tuning_expander`` (grid sweep + residual ranking + "Use best
    values") works unchanged.  The topology is compiled to a :class:`SimPlan`
    once; ``simulate`` / ``simulate_vec`` / ``simulate_batch`` then evaluate it
    vectorised, honouring ``xp`` so a CuPy sweep runs entirely on the GPU."""

    class _CustomModelAdapter:
        NAME = f"Custom · {model.name}"
        SHORT = _TOPO
        TOPOLOGY_CHAR = "T" if model.intrinsic_type == "T" else "pi"
        # The Auto-Tuning backend badge keys off this: the custom model's CPU
        # path runs the data-driven `sim_custom_batch` Rust kernel (not one of
        # the fixed-topology kernels in SIM_FOR_TOPOLOGY).
        USES_RUST_BATCH = True
        _model = model
        _plan = compile_plan(model)

        @classmethod
        def simulate(cls, params, freq, z0=50.0):
            return simulate_custom_model(cls._model, freq, dict(params), z0)

        @classmethod
        def simulate_vec(cls, params, freq, z0=50.0, xp=None):
            """Vectorised single-param-set simulate → (N, 2, 2) on the xp device.
            Used by the live slider preview and the Nelder-Mead auto-tuner.

            CPU (``xp`` is numpy) routes through the Rust ``sim_custom_batch``
            kernel (B=1) just like ``simulate_batch``; CUDA stays on cupy."""
            xp = np if xp is None else xp
            if xp is np:
                S = _rust_sim_custom(
                    cls._plan, dict(params),
                    np.ascontiguousarray(freq, dtype=np.float64), z0,
                    np_fallback=lambda p, f, z:
                        simulate_custom_model_batch(cls._plan, f, p, z, xp=np))
                return S[0]
            return simulate_custom_model_batch(cls._plan, freq, dict(params),
                                               z0, xp=xp)[0]

        @classmethod
        def simulate_batch(cls, params, freq, z0=50.0, xp=None, cache=None):
            """Batched simulate over (param_combo × freq) → ``inner + (N, 2, 2)``
            on the xp device (no host transfer).  ``params`` values may be
            scalars or arrays that broadcast to a common ``inner`` shape (the
            shared sweep uses one axis per swept parameter).

            CPU (``xp`` is numpy) routes through the Rust ``sim_custom_batch``
            kernel when the crate is built (silent NumPy fallback otherwise);
            the CUDA path (``xp`` is cupy) runs the vectorised evaluator on the
            GPU directly."""
            xp = np if xp is None else xp
            N = int(np.asarray(freq).size)
            shapes = [np.shape(v) for v in params.values()]
            inner = np.broadcast_shapes(*shapes) if shapes else ()
            B = int(np.prod(inner)) if inner != () else 1

            # Flatten each swept param to a 1-D length-B array; scalars stay scalar.
            flat = {}
            for k, v in params.items():
                a = xp.asarray(v)
                flat[k] = a if a.ndim == 0 else xp.broadcast_to(a, inner).reshape(B)

            if xp is np:
                S = _rust_sim_custom(
                    cls._plan, flat, np.ascontiguousarray(freq, dtype=np.float64),
                    z0, np_fallback=lambda p, f, z:
                        simulate_custom_model_batch(cls._plan, f, p, z, xp=np))
            else:
                S = simulate_custom_model_batch(cls._plan, freq, flat, z0, xp=xp)

            return S[0] if inner == () else S.reshape(tuple(inner) + (N, 2, 2))

    return _CustomModelAdapter
