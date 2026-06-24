"""
custom_model — user-built ("custom") small-signal model subpackage.

Two workflows, surfaced via :func:`render_custom_section` as a "🧩 Custom
model" section inside both the RF Forward Simulator and HBT SSM Extraction
pages:

* **Make**  — a progressive, visual circuit builder (``ui_build.py``).  The
  user picks a device (Bipolar/Unipolar) and an intrinsic π/T core, edits the
  B/C/E junction Networks (default = Cheng's), then adds R/L/C branches working
  from the inside (intrinsic) outward (parasitic pad caps).  The topology +
  every component *name* is serialised to a JSON file in ``custom_models/``.

* **Use**   — load a saved topology (``ui_use.py``), type a value for every
  component, forward-simulate the S-parameters with the generic nodal solver
  (``core.py``) and inspect the Smith / Bode charts.

``core.py``      — data model (:class:`CustomModel`), JSON save/load, and the
                   netlist→Y→S solver.
``schematic.py`` — live SVG schematic renderer shared by both workflows.
"""
from __future__ import annotations

from .core import (
    Element, Network, CustomModel,
    simulate_custom_model,
    save_model, load_model, list_saved_models, models_dir,
)


def render_custom_section(measured: dict | None = None) -> None:
    """Render the full custom-model section (Build / Simulate / Fit) embedded in
    the SSM Simulation & Fitting page.  The draft model lives in
    ``st.session_state`` so it carries across pages within a session.

    ``measured`` (``{S, freq, z0, label, stage}`` or None) is the optional
    measured device to fit against.  When present a "🎯 Fit to measurement"
    mode appears (overlay + residual + tuning via :func:`render_custom_fit`).
    """
    import streamlit as st
    from .ui_build import render_build_ui
    from .ui_use import render_use_ui
    from ..helpers import segmented_radio

    _M_USE, _M_BUILD, _M_FIT = "📂 Load / simulate", "🛠 Build model", "🎯 Fit to measurement"
    options = [_M_USE, _M_BUILD] + ([_M_FIT] if measured is not None else [])

    # "Send to Load / Fit" from the build view → switch this selector.  Must run
    # *before* the selector is instantiated to set its session value.
    if st.session_state.pop("cm_nav_to_loadfit", False):
        st.session_state["cm_mode"] = _M_FIT if measured is not None else _M_USE
    # Default to Fit mode whenever a measured device is present (so a handed-over
    # / uploaded file is shown straight away, like the built-in models), unless
    # the user has explicitly chosen another *valid* mode.
    if st.session_state.get("cm_mode") not in options:
        st.session_state["cm_mode"] = (_M_FIT if measured is not None else _M_USE)

    mode = segmented_radio("Custom model", options,
                           key="cm_mode", label_visibility="collapsed")
    if mode == _M_BUILD:
        render_build_ui()
    elif mode == _M_FIT and measured is not None:
        from .ui_fit import render_custom_fit
        render_custom_fit(measured["label"], measured["S"],
                          measured["freq"], measured["z0"])
    else:
        render_use_ui()


__all__ = [
    "Element", "Network", "CustomModel",
    "simulate_custom_model",
    "save_model", "load_model", "list_saved_models", "models_dir",
    "render_custom_section",
]
