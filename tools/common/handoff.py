"""
handoff.py — cross-page workflow bus (RF and DC).

Layer:       ui-helper (streamlit session_state only, no math)
Imported by: every page that sends or receives a device / dataset
Gotchas:     PAGE_* strings are resolved by st.switch_page **relative to the
             main script's directory** (the repo root), so they carry a
             "tools/..." prefix and must match i18n.TOOLS paths exactly.

This merges what used to be two files with the same shape and the same
idiom — ``tools/SSM/handoff.py`` (RF) and ``tools/dc_handoff.py`` (DC).
Keeping one module means the page-path registry below has exactly one
definition; ``dev/smoke_test.py`` asserts every entry resolves on disk.

Payloads travel through ``st.session_state``, which Streamlit preserves
across ``st.switch_page`` within a session.  Each destination page checks
for a pending handoff at the top of its script and consumes it once.

RF payload fields
-----------------
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
``extras``      : dict | None          — additional devices to load alongside
                  the primary one, ``{label: {"S","freq","z0"}}``.  Extraction
                  needs the *other* bias files for the Z-parameter, Cold-HBT and
                  τ_total methods, so a batch de-embed sends the whole set here
                  while ``S``/``label`` carry the user's selected (primary) file.
"""
from __future__ import annotations

import streamlit as st

# ── Page path registry ───────────────────────────────────────────────────────
# Must match the st.Page entries in IOED_Tool_Web.py / common.i18n.TOOLS.
PAGE_AT_A_GLANCE = "tools/rf/at_a_glance.py"
PAGE_EXTRACTION  = "tools/rf/extraction.py"
PAGE_SIMFIT      = "tools/rf/simulator.py"
PAGE_DC_ANALYSIS = "tools/dc/b1500a_plot.py"

# ── RF bus ───────────────────────────────────────────────────────────────────
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


# ── DC bus ───────────────────────────────────────────────────────────────────
# Deliberately a separate API rather than another RF "target": the payload is
# a list of workbooks, nothing like the S/freq/z0 shape above, and collapsing
# the two would only hide that.
_DC_KEY = "_dc_handoff_files"


def send_dc(files: list[dict]) -> None:
    """Stash a list of type-group workbooks, one per measurement type:
    {"name": str, "sheets": {sheet_name: DataFrame}, "dtype": "Family"|"Gummel"|"Diode"}.
    In DC Analysis the file dropdown lists ``name`` (the type) and the sheet
    picker lists the measurements (``sheets`` keys) within it."""
    st.session_state[_DC_KEY] = files


def peek_dc():
    return st.session_state.get(_DC_KEY)


def take_dc():
    return st.session_state.pop(_DC_KEY, None)
