"""
handoff.py — cross-page workflow bus for the RF tool group.

Lets the three RF pages pass the **active DUT** between each other without
re-uploading files, so a user can flow:

    📡 RF At a Glance  →  🔬 SSM Extraction  →  🛠️ SSM Simulation & Fitting

The payload travels through ``st.session_state`` (which Streamlit preserves
across ``st.switch_page`` within a session).  Each destination page checks for
a pending handoff at the top of its script and consumes it once.

Payload fields
--------------
``S``           : np.ndarray (N, 2, 2) — the S-parameters to carry.
``freq``        : np.ndarray (N,)      — frequency axis in Hz.
``z0``          : float                — reference impedance.
``label``       : str                  — display name (DUT stem).
``stage``       : str                  — provenance of ``S``:
                  "raw"         (probe-level, parasitics present),
                  "deembedded"  (Open/Short removed, pads ≈ 0),
                  "intrinsic"   (fully peeled).
``params``      : dict | None          — fitted model params (SI) to seed the
                  Simulation & Fitting override fields (set by Extraction).
``model_short`` : str | None           — which model the params belong to
                  ("T" / "pi" / "XuT" / "KY" / "custom").
``extras``      : dict | None           — additional devices to load alongside
                  the primary one, ``{label: {"S","freq","z0"}}``.  Extraction
                  needs the *other* bias files for the Z-parameter, Cold-HBT and
                  τ_total methods, so a batch de-embed sends the whole set here
                  while ``S``/``label`` carry the user's selected (primary) file.
"""
from __future__ import annotations

import streamlit as st

# Page paths (must match the st.Page entries in IOED_Tool_Web.py / i18n).
PAGE_AT_A_GLANCE = "tools/IOED_HBT_RF_extract.py"
PAGE_EXTRACTION  = "tools/SSM_extraction.py"
PAGE_SIMFIT      = "tools/RF_simulator.py"

# Valid handoff targets.
TARGET_EXTRACTION = "extraction"
TARGET_SIMFIT     = "simfit"

_KEY = "_ssm_handoff_{target}"


def send(target: str, *, S, freq, z0, label: str, stage: str = "raw",
         params: dict | None = None, model_short: str | None = None,
         extras: dict | None = None) -> None:
    """Stash a payload for ``target`` ("extraction" | "simfit")."""
    st.session_state[_KEY.format(target=target)] = {
        "S": S, "freq": freq, "z0": z0, "label": label, "stage": stage,
        "params": params, "model_short": model_short, "extras": extras,
    }


def peek(target: str) -> dict | None:
    """Return the pending payload for ``target`` without consuming it."""
    return st.session_state.get(_KEY.format(target=target))


def take(target: str) -> dict | None:
    """Pop and return the pending payload for ``target`` (one-shot consume)."""
    return st.session_state.pop(_KEY.format(target=target), None)
