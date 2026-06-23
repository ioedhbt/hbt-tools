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


def render_custom_section() -> None:
    """Render the full custom-model builder (Make + Use tabs) as a section that
    can be embedded inside any portal page (RF Forward Simulator + HBT SSM
    Extraction both host it).  The draft model lives in ``st.session_state`` so
    it carries across pages within a session.
    """
    import streamlit as st
    from .ui_build import render_build_ui
    from .ui_use import render_use_ui

    # "Send to Load / Fit" from the build view → switch this radio to Load.
    # Must run *before* the radio is instantiated to set its session value.
    if st.session_state.pop("cm_nav_to_loadfit", False):
        st.session_state["cm_mode"] = "📂 Load model"

    mode = st.radio("Custom model", ["📂 Load model", "🛠 Build model"],
                    horizontal=True, key="cm_mode", label_visibility="collapsed")
    if mode == "🛠 Build model":
        render_build_ui()
    else:
        render_use_ui()


__all__ = [
    "Element", "Network", "CustomModel",
    "simulate_custom_model",
    "save_model", "load_model", "list_saved_models", "models_dir",
    "render_custom_section",
]
