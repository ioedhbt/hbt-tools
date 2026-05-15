"""
tools.SSM.helpers — Single import surface for all SSM helper functions.

See INDEX.md (next to this file) for a human-readable catalog of every
helper, what it does, and which modules use it.

Submodules
----------
rf_math       — S/Y/Z conversions, smith grid, freq check, element math
s2p_io        — Touchstone/CSV parsing + writing, dummy simulators
deembed_math  — Open/Short extraction math (step_open, step_short, peel_parasitics)
metrics       — h21/Mason U/fT/fmax computation, 20 dB/dec extrapolation
plotly_plots  — Plotly Smith / Bode / Plateau builders
chart_export  — Excel export, Streamlit chart wrapper, metric card
"""
from __future__ import annotations

# ── rf_math ──────────────────────────────────────────────────────────────────
from .rf_math import (
    s_to_y,
    y_to_s_single,
    y_to_s_batch,
    y_to_s_vec,
    _inv2,
    inv2x2,
    mm2x2,
    y_to_z,
    z_to_y,
    safe_median,
    strict_freq_check,
    open_elem_Y,
    short_lead_Z,
    extended_smith_grid,
    params_hash,
)

# ── s2p_io ───────────────────────────────────────────────────────────────────
from .s2p_io import (
    parse_s2p,
    parse_s2p_bytes,
    parse_csv,
    write_s2p,
    interpolate_s2f,
    simulate_open,
    simulate_short,
    build_Y_pad,
    build_Z_ser,
    load_cal,
)

# ── deembed_math ─────────────────────────────────────────────────────────────
from .deembed_math import (
    step_open,
    step_short,
    peel_parasitics,
    build_Y_pad_vec,
    build_Z_ser_vec,
    build_Y_pad_batch,
    build_Z_ser_batch,
    deembed_open_short,
    deembed_thru_half,
)

# ── metrics ──────────────────────────────────────────────────────────────────
from .metrics import (
    compute_h21_U,
    find_ft_fmax,
    extrap_20dbdec,
    single_pole_extrap,
    compute_metrics,
    extract_limit,
)

# ── plotly_plots ─────────────────────────────────────────────────────────────
from .plotly_plots import (
    PALETTE,
    darken,
    bode_layout,
    make_smith,
    make_bode,
    make_plateau,
)

# ── chart_export ─────────────────────────────────────────────────────────────
from .chart_export import (
    fig_to_excel_bytes,
    plotly_with_dl,
    build_excel,
    metric_card,
    EXCEL_MIME,
)

# ── widgets ──────────────────────────────────────────────────────────────────
from .widgets import (
    quickset_buttons,
    apply_pending,
)
