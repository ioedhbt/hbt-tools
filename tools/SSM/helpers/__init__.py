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
    FT_FMAX_SYMBOLS,
    FT_FMAX_COLORS,
    darken,
    bode_layout,
    make_smith,
    make_bode,
    make_plateau,
    make_smith_bode_slider_fig,
    make_smith_bode_joint_slider_html,
    add_overlay_trace_with_markers,
    thinned_indices,
)

# ── chart_export ─────────────────────────────────────────────────────────────
from .chart_export import (
    fig_to_excel_bytes,
    bode_excel_bytes,
    plotly_with_dl,
    build_excel,
    metric_card,
    fig_to_tsv,
    frames_to_tsv,
    xlsx_bytes_to_tsv,
    copy_button,
    EXCEL_MIME,
)

# ── widgets ──────────────────────────────────────────────────────────────────
from .widgets import (
    quickset_buttons,
    apply_pending,
    info_icon_html,
    segmented_radio,
)

# ── fit_cache ────────────────────────────────────────────────────────────────
from .fit_cache import (
    get_fit,
    get_fit_timestamp,
    list_fits,
    save_fit,
    delete_fit,
    load_cache,
    export_cache_bytes,
    import_cache_bytes,
    cache_path_str,
    differs_from,
)

# ── rust_kernels (optional; NumPy fallback when not built) ───────────────────
from .rust_kernels import (
    HAS_RUST as RUST_KERNELS_AVAILABLE,
    inv2x2_batch  as rust_inv2x2_batch,
    mm2x2_batch   as rust_mm2x2_batch,
    y_to_s_batch  as rust_y_to_s_batch,
    y_to_s_4d     as rust_y_to_s_4d,
    port_residuals_batch as rust_port_residuals_batch,
    parse_and_compute_batch as rust_parse_and_compute_batch,
)
