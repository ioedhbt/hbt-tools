"""
models/tuning/helpers.py — Small shared building blocks for the Auto Tuning
expander: the CPU/CUDA column layout, the sweep-row builder, the hard-limit
session-state clamp, the "best result" markdown summary, and the results
table's column-name lookup.

Kept in their own module (rather than in sweep.py) so both sweep.py (the
UI-card renderers + orchestrator) and the driver modules
(drivers_sweep.py / drivers_nelder_mead.py / drivers_progressive.py) can
import them without a circular import — the drivers are called BY sweep.py,
so nothing here may import back from sweep.py.

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).
"""
from __future__ import annotations
import numpy as np
import streamlit as st

from tools.common.i18n import tr

from .preview import _make_sweep_values
from ._cuda_env import _HAS_CUDA


def _action_cols():
    return st.columns(2) if _HAS_CUDA else (st.columns(1)[0],
                                             st.empty())


def _clamp_row_session(kp, h_lo, h_hi):
    """Pre-clamp stale session Min/Max into the hard limits (the
    keyed number_inputs would otherwise raise
    StreamlitAPIException) and floor Step at 0.  Runs in BOTH the
    open (widget) and closed (session-read) paths."""
    for _sfx in ("_min", "_max"):
        _sk = f"{kp}{_sfx}"
        if _sk in st.session_state:
            _v = float(st.session_state[_sk])
            _v = max(_v, h_lo)
            if h_hi is not None:
                _v = min(_v, float(h_hi))
            st.session_state[_sk] = _v
    _step_sk = f"{kp}_step"
    if _step_sk in st.session_state:
        st.session_state[_step_sk] = max(
            0.0, float(st.session_state[_step_sk]))


def _row_entry(spec, enabled, min_val, step_val, max_val, all_p):
    """Build one param_rows dict — the single place sweep/n_calc
    come from, so the open and closed paths can never diverge."""
    key, label, scale = spec[0], spec[1], spec[2]
    unit = spec[3] if len(spec) > 3 else ""
    if enabled:
        sweep = _make_sweep_values(min_val, max_val, step_val)
    else:
        sweep = np.array([float(all_p.get(key, 0.0)) * scale])
    return {"key": key, "label": label, "scale": scale, "unit": unit,
            "enabled": enabled, "sweep": sweep, "n_calc": len(sweep)}


_HIDE_IF_ZERO = {"Cpbe", "Cpce", "Cpbc", "Lb", "Lc", "Le"}
def _best_summary_md(best_row, tuning_specs, topo_key, fname, label=None):
    if label is None:
        label = tr("Best so far", "目前最佳")
    head = (f"**{label} — {tr('Total', '總計')}: {best_row['Total Residual (%)']:.2f}%  |  "
            f"S11: {best_row['S11 (%)']:.2f}%  S12: {best_row['S12 (%)']:.2f}%  "
            f"S21: {best_row['S21 (%)']:.2f}%  S22: {best_row['S22 (%)']:.2f}%**")
    parts = []
    for spec in tuning_specs:
        key = spec[0]; label_p = spec[1]
        unit = spec[3] if len(spec) > 3 else ""
        fmt = spec[4] if len(spec) > 4 else "%.4g"
        col = f"{label_p} ({unit})" if unit else label_p
        if col not in best_row.index:
            continue
        v = float(best_row[col])
        if key in _HIDE_IF_ZERO and np.isfinite(v) and abs(v) < 1e-30:
            continue
        is_swept = bool(st.session_state.get(
            f"tune_{topo_key}_{key}_{fname}_chk", False))
        sval = (fmt % v) if np.isfinite(v) else "NaN"
        name_md = f"<b>{label_p}</b>" if is_swept else label_p
        val_md  = (f"<b>{sval} {unit}</b>".rstrip()
                   if is_swept else f"{sval} {unit}".rstrip())
        parts.append(f"{name_md}: {val_md}")
    if parts:
        head += "  \n<small>" + ", ".join(parts) + "</small>"
    return head


def _col_name(row):
    return f"{row['label']} ({row['unit']})" if row["unit"] else row["label"]
