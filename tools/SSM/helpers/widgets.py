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
import html as _html
import numpy as np
import streamlit as st

from ...i18n import tr


def info_icon_html(text: str, label: str = "ⓘ") -> str:
    """HTML markup for a hover-help icon.

    Use inside an existing HTML markdown block (`unsafe_allow_html=True`)
    or via ``st.markdown(info_icon_html(...), unsafe_allow_html=True)`` to
    embed a small hover-only tooltip without a visible caption.  Browser-
    native tooltip — works on every Streamlit version.
    """
    safe = _html.escape(text)
    return (f'<span title="{safe}" '
            f'style="cursor:help;color:#666;font-size:0.85em;'
            f'border-bottom:1px dotted #aaa;margin-left:6px;'
            f'padding:0 3px">{label}</span>')


def segmented_radio(label, options, *, index: int = 0, key=None,
                    horizontal: bool = True, format_func=None, help=None,
                    label_visibility: str = "visible",
                    disabled: bool = False):
    """Drop-in replacement for :func:`st.radio` that renders as a segmented
    button group (``st.segmented_control``) instead of radio circles — the same
    chip-style selector the custom-model builder uses to pick component groups.

    Returns the *selected option value* (not its label), matching ``st.radio``.
    A selection is always guaranteed: ``st.segmented_control`` can otherwise
    return ``None`` when the user clicks the active chip to deselect it, so we
    fall back to the default value in that case.

    Streamlit forbids assigning to a widget's ``key`` in session_state **after**
    the widget is instantiated, so we never do that.  Instead the value is
    seeded / repaired **before** ``segmented_control`` is created (which is
    allowed) — this also recovers gracefully from a stale value left by a
    previous page whose option set has since changed.

    Falls back to ``st.radio`` on Streamlit builds without ``segmented_control``
    (added in 1.40), so callers can switch unconditionally.  ``horizontal`` is
    accepted for API-compatibility with ``st.radio`` and only used by that
    fallback (segmented controls are always horizontal).
    """
    options = list(options)
    seg = getattr(st, "segmented_control", None)
    if seg is None or not options:
        return st.radio(label, options, index=index, key=key,
                        horizontal=horizontal,
                        format_func=(format_func or str),
                        help=help, label_visibility=label_visibility,
                        disabled=disabled)

    idx = index if 0 <= index < len(options) else 0

    kwargs = dict(options=options, key=key, help=help,
                  label_visibility=label_visibility, disabled=disabled,
                  selection_mode="single")
    if format_func is not None:
        kwargs["format_func"] = format_func

    if key is not None:
        # Seed / repair the value BEFORE the widget exists (allowed).  Covers
        # first render (no value yet) and a stale value from a previous page
        # whose option list has changed; both resolve to the default option.
        if st.session_state.get(key) not in options:
            st.session_state[key] = options[idx]
        # `key` already carries the value → must NOT also pass `default`.
    else:
        kwargs["default"] = options[idx]

    picked = seg(label, **kwargs)
    # Deselect (user clicked the active chip) → segmented_control returns None.
    # Fall back to the default without touching the widget key post-render.
    if picked is None:
        picked = options[idx]
    return picked


def apply_pending(target_key: str) -> None:
    """Promote any pending quickset value into the widget's state key.
    Must be called BEFORE the number_input that reads ``target_key``.
    """
    pk = target_key + "_pending"
    if pk in st.session_state:
        st.session_state[target_key] = st.session_state.pop(pk)


def _candidates(arr_disp, default_disp, cold_disp=None):
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
    if cold_disp is not None and np.isfinite(cold_disp):
        out.append(("cold", float(cold_disp)))
    return out


def quickset_buttons(*, container, key_prefix: str, target_key: str,
                     arr_disp=None, default_disp=None, cold_disp=None,
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
    cold_disp    : Optional cold-section value in display units. When
                   provided, surfaces as a "cold = …" button placed beside
                   "default" (layout="below") for quick adoption.
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
    cands = _candidates(arr_disp, default_disp, cold_disp)
    if not cands:
        return

    suffix = f" {unit}" if unit else ""

    _HELP = {
        "mean":    tr("Set input to the mean of the per-frequency array",
                      "將輸入值設為逐頻率陣列的平均值"),
        "median":  tr("Set input to the median of the per-frequency array",
                      "將輸入值設為逐頻率陣列的中位數"),
        "low f":   tr("Set input to the lowest-frequency point",
                      "將輸入值設為最低頻率點"),
        "high f":  tr("Set input to the highest-frequency point",
                      "將輸入值設為最高頻率點"),
        "default": tr("Restore the auto-extracted (median over the slider range) value",
                      "還原自動萃取值（滑桿範圍內的中位數）"),
        "cold":    tr("Set input to the value extracted in the Cold-HBT section",
                      "將輸入值設為冷 HBT 區段所萃取的值"),
    }
    # Button-label word for each internal candidate id — translated for
    # display only; `lbl` itself (the dict key / session-state suffix /
    # _HELP lookup) stays the English id so nothing downstream breaks.
    _LABEL_ZH = {
        "mean": "平均", "median": "中位數", "low f": "低頻",
        "high f": "高頻", "default": "預設", "cold": "冷量測",
    }

    def _render(row_container, items):
        cols = row_container.columns(len(items))
        for col, (lbl, val) in zip(cols, items):
            disp_lbl = tr(lbl, _LABEL_ZH.get(lbl, lbl))
            text = f"{disp_lbl} = {format(val, spec)}{suffix}"
            if col.button(text,
                          key=f"{key_prefix}_qs_{lbl}",
                          width="stretch",
                          help=_HELP.get(lbl, tr(f"Set input to {lbl} of the per-frequency array",
                                                  f"將輸入值設為逐頻率陣列的 {lbl}"))):
                st.session_state[target_key + "_pending"] = val
                st.rerun()

    if layout == "below":
        by_lbl = {lbl: (lbl, val) for lbl, val in cands}
        # Place "cold" beside "default" when both are present.
        for row_lbls in (("default", "cold"), ("mean", "median"), ("low f", "high f")):
            row = [by_lbl[l] for l in row_lbls if l in by_lbl]
            if row:
                _render(container, row)
    else:
        _render(container, cands)
