"""
helpers/widgets.py — Streamlit widgets shared across the SSM extraction UI.

Currently exposes:
  - quickset_buttons : a row/column of "mean=…", "median=…", "low f=…",
                       "high f=…", "default=…" buttons that one-click
                       overwrite the value of a paired number_input.

Wiring contract (caller side)
-----------------------------
The buttons here cannot mutate a widget's session_state value AFTER the
widget has been instantiated (Streamlit forbids it). Instead they write
``st.session_state[target_key + "_pending"]`` and call ``st.rerun()``. On
the next run the caller MUST, **before** creating the number_input:

    >>> pk = target_key + "_pending"
    >>> if pk in st.session_state:
    ...     st.session_state[target_key] = st.session_state.pop(pk)

The convenience helper :func:`apply_pending` does exactly this.
"""
from __future__ import annotations
import numpy as np
import streamlit as st


def apply_pending(target_key: str) -> None:
    """Promote any pending quickset value into the widget's state key.
    Must be called BEFORE the number_input that reads ``target_key``.
    """
    pk = target_key + "_pending"
    if pk in st.session_state:
        st.session_state[target_key] = st.session_state.pop(pk)


def _candidates(arr_disp, default_disp):
    out = []
    if arr_disp is not None:
        a = np.asarray(arr_disp, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) > 0:
            out.append(("mean",   float(np.mean(a))))
            out.append(("median", float(np.median(a))))
            out.append(("low f",  float(a[0])))
            out.append(("high f", float(a[-1])))
    if default_disp is not None and np.isfinite(default_disp):
        out.append(("default", float(default_disp)))
    return out


def quickset_buttons(*, container, key_prefix: str, target_key: str,
                     arr_disp=None, default_disp=None,
                     unit: str = "",
                     fmt: str = "%.4g", layout: str = "side") -> None:
    """Render quickset buttons that, when clicked, write a pending value
    into ``target_key + '_pending'`` and rerun.

    Parameters
    ----------
    container    : Streamlit container (column / expander / st itself).
    key_prefix   : Unique session_state prefix for this set of buttons —
                   keep it derived from `target_key` for collision-free keys.
    target_key   : The session_state key of the paired number_input.
    arr_disp     : Optional 1-D array of per-frequency values **in display
                   units** (so mean/median/low/high are already correct for
                   the input). When None, no buttons are rendered (skip
                   quickset entirely for scalar-only inputs).
    default_disp : The original auto-extracted scalar in display units.
                   Surfaces as the "default = …" button.
    unit         : Display unit appended to each button label and tooltip.
    fmt          : printf-style format spec for the values shown on the
                   button labels (e.g. ``"%.4g"``).
    layout       : ``"side"`` or ``"below"`` — both produce one row of
                   equal-width buttons; the choice is purely semantic for
                   the caller (where the row is placed in the layout).
    """
    if arr_disp is None:
        # No per-frequency array → skip quickset entirely (per design).
        return
    spec = fmt.lstrip("%")  # "%.4g" → ".4g"
    cands = _candidates(arr_disp, default_disp)
    if not cands:
        return

    suffix = f" {unit}" if unit else ""

    def _render(row_container, items):
        cols = row_container.columns(len(items))
        for col, (lbl, val) in zip(cols, items):
            text = f"{lbl} = {format(val, spec)}{suffix}"
            if col.button(text,
                          key=f"{key_prefix}_qs_{lbl}",
                          use_container_width=True,
                          help=f"Set input to {lbl} of the per-frequency array"):
                st.session_state[target_key + "_pending"] = val
                st.rerun()

    if layout == "below":
        by_lbl = {lbl: (lbl, val) for lbl, val in cands}
        for row_lbls in (("default",), ("mean", "median"), ("low f", "high f")):
            row = [by_lbl[l] for l in row_lbls if l in by_lbl]
            if row:
                _render(container, row)
    else:
        _render(container, cands)
