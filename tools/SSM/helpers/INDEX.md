# SSM helpers — function index

Single source of truth for every helper in `tools/SSM/helpers/`. Skim this
file to find which submodule owns a function before grep'ing the codebase.

All helpers re-export from `tools.SSM.helpers`, so call sites can simply do:

```python
from tools.SSM.helpers import parse_s2p, make_smith, compute_metrics
```

The submodule paths below are for jump-to-definition only.

---

## `rf_math.py` — Pure RF math (no Streamlit, no plotting)

| Function | Purpose |
|----------|---------|
| `s_to_y(S, z0=50.0)` | S → Y conversion, vectorised across all frequency points |
| `y_to_s_single(Y, z0=50.0)` | Y → S for a single 2×2 matrix |
| `y_to_s_batch(Y, z0=50.0)` | Y → S per-row loop (compatibility shim) |
| `y_to_s_vec(Y, z0, xp=None)` | GPU-aware vectorised Y → S (numpy or cupy) |
| `_inv2(M)` | Per-row 2×2 matrix inverse using `np.linalg.inv` |
| `inv2x2(M, xp=None)` | Analytic 2×2 batched inverse (GPU-tuned, no cuSOLVER) |
| `mm2x2(A, B, xp=None)` | Analytic 2×2 batched matmul (GPU-tuned, no cuBLAS) |
| `y_to_z`, `z_to_y` | Aliases for `_inv2` |
| `safe_median(arr, n=None)` | Median of finite values, returns 0.0 if empty |
| `strict_freq_check(f_dut, f_dummy, label)` | Raise if frequency grids differ |
| `open_elem_Y(C, mode, extra, w)` | Pad capacitor admittance (None / Parallel L / Series L / Series R) |
| `short_lead_Z(R, L, Cpar, w)` | Series-lead impedance with optional parallel cap |
| `extended_smith_grid(max_r=1.0)` | Plotly Smith chart background traces |
| `params_hash(p)` | MD5 of a parameter dict, for caching |

## `s2p_io.py` — Touchstone / CSV I/O and dummy simulators

| Function | Purpose |
|----------|---------|
| `parse_s2p(content)` | Parse a Touchstone .s2p (accepts str or bytes) → `(freq, S, z0)` |
| `parse_s2p_bytes(raw)` | Bytes-only alias of `parse_s2p` (backwards compatibility) |
| `parse_csv(content, z0=50.0)` | Parse VNA CSV export (RI columns) → `(freq, S, z0)` |
| `write_s2p(freq_hz, S, title="", params=None)` | Serialize to Touchstone .s2p (DB format) bytes |
| `interpolate_s2f(f_src, S_src, f_tgt)` | Interpolate S to a new frequency grid |
| `simulate_open(p, freq, z0=50.0)` | Forward-simulate Open dummy from pad params |
| `simulate_short(p, freq, z0=50.0)` | Forward-simulate Short dummy from pad+lead params |
| `build_Y_pad(p, w)` | 2×2 pad admittance matrix at angular freq `w` |
| `build_Z_ser(p, w)` | 2×2 series-lead impedance matrix at angular freq `w` |
| `load_cal(fobj)` | Streamlit-aware S2P loader (handles `UploadedFile`) |

## `deembed_math.py` — Open/Short extraction & de-embedding

| Function | Purpose |
|----------|---------|
| `step_open(open_data, n0, n1, method, trim_pct)` | Extract Cpbe/Cpce/Cpbc from Open dummy |
| `step_short(short_data, freq, ...)` | Extract Lb/Lc/Le/Rpb/Rpc/Rpe from Short dummy |
| `peel_parasitics(S_raw, freq, z0, p)` | Remove pad+lead parasitics → Y_intrinsic |
| `build_Y_pad_vec(p, omega, xp)` | (N,2,2) pad admittance, fully vectorised |
| `build_Z_ser_vec(p, omega, xp)` | (N,2,2) series-lead impedance, fully vectorised |
| `build_Y_pad_batch(p, omega, B, N, xp)` | (B,N) pad admittance planes for sweep tuning |
| `build_Z_ser_batch(p, omega, B, N, xp)` | (B,N) series-lead impedance planes for sweep tuning |
| `deembed_open_short(Y_dut, Y_open, Y_short)` | Standard open-short de-embedding (Gao §4.2) |
| `deembed_thru_half(Y_dut, Y_thru_deemb)` | THRU/2 half-impedance subtraction |

## `metrics.py` — Gain figures of merit

| Function | Purpose |
|----------|---------|
| `compute_h21_U(S)` | Compute |h21|² (dB) and Mason U (dB) from S-parameters |
| `find_ft_fmax(f_ghz, h21_db, U_db)` | Linear interpolation of 0 dB crossings → (fT, fmax) |
| `extrap_20dbdec(f_ghz, gain_db, n_pts=60)` | Project trace along −20 dB/dec to 0 dB crossing |
| `compute_metrics(Y, freq_hz)` | Build DataFrame of h21²/U/MAG/MSG/K/plateau columns |
| `extract_limit(freq_ghz, gain_db, plateau_arr, n_pts, f_min, f_max)` | Genuine-crossing fT/fmax extractor with extrapolation fallback |

## `plotly_plots.py` — Plotly Smith / Bode / Plateau builders

| Function | Purpose |
|----------|---------|
| `PALETTE` | 10-color hex palette for trace cycling |
| `darken(hex_color)` | Darken a `#rrggbb` color by 45 units per channel |
| `bode_layout(title, ytitle, yr, xr)` | Plotly layout dict for Bode/plateau plots |
| `make_smith(S, f_array, f_min, f_max, toggles, scales, title, max_r=1.0)` | Plotly Smith chart with 4 selectable S-params |
| `make_bode(df, title, xr, yr, sh21, su, smag, color)` | Plotly Bode plot with optional 20 dB/dec extrapolation |
| `make_plateau(df, res, title, xr, sh21, su, smag, color)` | Plotly GBP plateau plot |

> **Step 6 (later):** `make_smith` / `make_bode` will absorb the simpler `mults`-based API
> currently in `RF_simulator.py::_build_smith` / `_build_bode`. For now they match the IOED
> signatures verbatim.

## `widgets.py` — Streamlit input widgets

| Function | Purpose |
|----------|---------|
| `quickset_buttons(container, key_prefix, target_key, arr_disp, default_disp, fmt, layout)` | Row of one-click buttons (mean/median/low f/high f/default) that overwrite a paired number_input via a pending session-state key + rerun. |
| `apply_pending(target_key)` | Call **before** the paired `number_input` to promote any pending quickset value into the widget's state. |

## `chart_export.py` — Excel export & Streamlit UI helpers

| Function | Purpose |
|----------|---------|
| `EXCEL_MIME` | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` |
| `fig_to_excel_bytes(fig)` | Extract Plotly traces → .xlsx bytes (smart Smith vs normal layout) |
| `plotly_with_dl(fig, key, filename, ...)` | Render Plotly chart + xlsx download button |
| `build_excel(summary_df, all_data)` | Multi-sheet workbook: Summary + per-DUT DataFrames |
| `metric_card(col, title, val, sub, color)` | Styled HTML metric tile (used in IOED tab_ind, batch tab) |

---

## Migration map (where each helper used to live)

| Helper | Was in | Now in |
|--------|--------|--------|
| `s_to_y`, `y_to_s_*`, `_inv2`, `inv2x2`, `mm2x2`, smith grid, etc. | `ssm_core.py` | `helpers/rf_math.py` |
| `parse_s2p_bytes`, `write_s2p`, `simulate_open/short`, `build_Y_pad/Z_ser` | `ssm_s2p.py` | `helpers/s2p_io.py` |
| `parse_s2p` (str), `parse_csv`, `_load_cal` | `IOED_HBT_RF_extract.py` | `helpers/s2p_io.py` |
| `step_open`, `step_short`, `peel_parasitics`, `build_*_vec/_batch` | `ssm_deembedding.py` | `helpers/deembed_math.py` |
| `deembed_open_short`, `deembed_thru_half` | `IOED_HBT_RF_extract.py` | `helpers/deembed_math.py` |
| `_compute_h21_U`, `_find_ft_fmax`, `extrap_20dbdec` | `ssm_plots.py` | `helpers/metrics.py` (renamed without `_`) |
| `compute_metrics`, `extract_limit` | `IOED_HBT_RF_extract.py` | `helpers/metrics.py` |
| `make_smith`, `make_bode`, `make_plateau`, `_layout`, `_darken`, `PALETTE` | `IOED_HBT_RF_extract.py` | `helpers/plotly_plots.py` |
| `fig_to_excel_bytes`, `plotly_with_dl`, `_EXCEL_MIME` | `ssm_chart_utils.py` | `helpers/chart_export.py` |
| `build_excel`, `_card` | `IOED_HBT_RF_extract.py` | `helpers/chart_export.py` (renamed `_card` → `metric_card`) |

> **Status:** migration complete. `ssm_core.py`, `ssm_s2p.py`, and
> `ssm_chart_utils.py` have been deleted. `ssm_deembedding.py` and
> `ssm_plots.py` retain only their `render_*` Streamlit functions; their
> math halves now live in `helpers/`. Every consumer imports through
> `tools.SSM.helpers`.
