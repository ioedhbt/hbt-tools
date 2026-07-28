# Repository map

One line per module. **Read this first — it is usually enough.** Drill into
[`docs/`](docs/) only when it isn't.

Layer legend: `pure` = no Streamlit, unit-testable · `io` = parsing/caching ·
`ui` = builds or renders Streamlit widgets · `page` = a Streamlit page script,
executed top-to-bottom on every interaction.

---

## Entry points

| path | layer | does | key symbols |
|---|---|---|---|
| [IOED_Tool_Web.py](IOED_Tool_Web.py) | page | Portal entry. The **only** `st.set_page_config` for the multi-page app. Password gate, language toggle, sidebar nav. | `check_password`, `_app_password` |
| [LAUNCH_Tool.py](LAUNCH_Tool.py) | — | Double-click launcher: auto-update (git or GitHub zip overlay), creates `.hbttools/` venv, spawns streamlit. | `main`, `auto_update`, `_overlay_tree` |
| [launch_ebl_calculator.py](launch_ebl_calculator.py) | — | Second, independent launcher that runs **only** the EBL page (own `.ebl_venv/`, no portal, no password). | `_find_app_file`, `_upload_limit_mb` |

## `tools/common/` — shared by every tool group

Nothing here may import `tools.rf` / `tools.dc` / `tools.ebeam`.

| path | layer | does | key symbols |
|---|---|---|---|
| [paths.py](tools/common/paths.py) | pure | The one definition of where the repo root is. Never re-derive it with `parents[N]`. | `REPO_ROOT`, `EXAMPLES_DIR` |
| [i18n.py](tools/common/i18n.py) | pure | Bilingual registry + **the tool registry that drives the sidebar**. Folder name == group key. | `TOOLS`, `GROUPS`, `GROUP_ORDER`, `t`, `tr`, `title` |
| [handoff.py](tools/common/handoff.py) | ui | Cross-page session_state bus (RF + DC) and the single page-path registry. | `PAGE_*`, `send`/`peek`/`take`, `send_dc`/`peek_dc`/`take_dc` |
| [ui_theme.py](tools/common/ui_theme.py) | ui | Every app-wide CSS rule, in one place. Sidebar RAM badge. | `inject_css`, `render_ram_badge` |
| [widgets.py](tools/common/widgets.py) | ui | Small shared widgets. | `segmented_radio`, `quickset_buttons`, `dedupe_upload_names` |
| [chart_export.py](tools/common/chart_export.py) | ui | Chart → xlsx/TSV/clipboard, and the chart wrapper every plot uses. | `plotly_with_dl`, `build_excel`, `unique_sheet_name`, `metric_card` |
| [mem_budget.py](tools/common/mem_budget.py) | pure | cgroup-aware RAM probe (psutil reports the *host*, not the container). | `ram_usage`, `ram_available_bytes` |
| [diagrams.py](tools/common/diagrams.py) | ui | The "ℹ️ How it works" pipeline drawer. | `pipeline_png` |

## `tools/rf/` — RF measurement

| path | layer | does | key symbols |
|---|---|---|---|
| [at_a_glance.py](tools/rf/at_a_glance.py) | page | Bulk-upload DUTs, de-embed, extract fT/fmax, Bode/Smith/plateau, export. | `process_dut` |
| [extraction.py](tools/rf/extraction.py) | page | Thin wrapper: parses uploads, delegates to the SSM engine. | — |
| [simulator.py](tools/rf/simulator.py) | page | Forward-simulate any topology; optionally fit to a measured file. | `_build_smith` |
| [batch_deembedding.py](tools/rf/batch_deembedding.py) | ui | Batch Open/Short de-embed tab, embedded in At a Glance. | `render_batch_deembedding_tab` |

### `tools/rf/ssm/` — the small-signal-model engine

Layered bottom-up; see [docs/SSM_INDEX.md](docs/SSM_INDEX.md) for per-function detail.

| path | layer | does | key symbols |
|---|---|---|---|
| [agent_api.py](tools/rf/ssm/agent_api.py) | pure | **Headless entry point for scripts and AI agents** — load, simulate, fit, no Streamlit needed. Also a CLI. | `load_data`, `fit`, `fit_multistart`, `simulate`, `residuals` |
| [main_ssm_extraction.py](tools/rf/ssm/main_ssm_extraction.py) | ui | The extraction page's Steps 1–3 + summary + fit cache. | `render_ssm_tab`, `render_builtin_forward_sim` |
| [ssm_plots.py](tools/rf/ssm/ssm_plots.py) | ui | Open/Short/de-embed previews, publication Smith charts. | `render_open_plots`, `render_matplotlib_smith` |
| [ssm_access_resistance.py](tools/rf/ssm/ssm_access_resistance.py) | ui | Z-param, Cold-HBT and open-collector access-R extractors. | `render_rz12_section`, `_render_cold_hbt` |
| [ssm_override.py](tools/rf/ssm/ssm_override.py) | ui | Picks the effective pad/access values feeding extraction. | `render_unified_pre_override` |
| helpers/[rf_math.py](tools/rf/ssm/helpers/rf_math.py) | pure | S↔Y↔Z conversions, 2×2 algebra, Smith grid. | `s_to_y`, `y_to_s_batch`, `y_to_z`, `inv2x2` |
| helpers/[s2p_io.py](tools/rf/ssm/helpers/s2p_io.py) | io | Touchstone/CSV parse + write, dummy simulators. **Header format matters** — `agent_api` reads it back. | `parse_s2p`, `parse_csv`, `write_s2p` |
| helpers/[deembed_math.py](tools/rf/ssm/helpers/deembed_math.py) | pure | Open/Short/Thru peel math. | `step_open`, `step_short`, `peel_parasitics` |
| helpers/[metrics.py](tools/rf/ssm/helpers/metrics.py) | pure | h21 / Mason U / fT / fmax, extrapolation. | `compute_h21_U`, `find_ft_fmax`, `extract_limit`, `compute_metrics` |
| helpers/[plotly_plots.py](tools/rf/ssm/helpers/plotly_plots.py) | ui | Smith/Bode/plateau figure builders (return figures, don't render). | `make_smith`, `make_bode`, `FT_FMAX_COLORS` |
| helpers/[fit_cache.py](tools/rf/ssm/helpers/fit_cache.py) | io | Per-(DUT, model) fitted params on disk. | `get_fit`, `save_fit`, `cache_path_str` |
| helpers/[rust_kernels.py](tools/rf/ssm/helpers/rust_kernels.py) | io | Loads the prebuilt Rust binary; silent NumPy fallback. | `HAS_RUST`, `parse_and_compute_batch` |
| models/[__init__.py](tools/rf/ssm/models/__init__.py) | pure | `REGISTRY` — **add a new model here**. | `REGISTRY`, `AbstractSSMModel` |
| models/[cheng.py](tools/rf/ssm/models/cheng.py) | pure+ui | Cheng T and π topologies (the two analytic-extraction models). | `ChengT`, `ChengPi` |
| models/[xu.py](tools/rf/ssm/models/xu.py) · [kunyang.py](tools/rf/ssm/models/kunyang.py) · [degachi.py](tools/rf/ssm/models/degachi.py) | pure+ui | Forward-simulation-only topologies. | `XuModel`, `KunYangHEMT` |
| models/[_shared.py](tools/rf/ssm/models/_shared.py) | pure | Code shared verbatim across models. | `tauC_from_alpha_phase` |
| models/[base_ui/](tools/rf/ssm/models/base_ui/) | ui | `SSMModelTemplate` + the re-export surface. **Import shared model UI from here** — `from .base_ui import X` still resolves for everything below. | `SSMModelTemplate`, `PAD_SPECS` |
| models/[residuals.py](tools/rf/ssm/models/residuals.py) | pure | Residual metrics. | `ssm_residual`, `_port_residuals` |
| models/[smith_ui.py](tools/rf/ssm/models/smith_ui.py) | ui | Smith chart + fT/fmax panel, scale controls. | `render_smith_chart`, `render_smith_with_ftfmax` |
| models/[param_groups.py](tools/rf/ssm/models/param_groups.py) | ui | The interactive per-parameter extraction fields. | `render_interactive_param_groups`, `render_finetune_diagram` |
| models/[fit_sections.py](tools/rf/ssm/models/fit_sections.py) | ui | Cbex sweep tool, τ_total multi-file fit. | `_render_cbex_sweep_tool` |
| models/[tuning/](tools/rf/ssm/models/tuning/) | ui | Auto/Visual tuning: `ranges` (bounds), `preview` (Visual), `sweep` (the expander + card helpers), `drivers_*` (the sweep / Nelder-Mead / progressive engines). | `render_tuning_expander`, `tune_hard_limits` |
| [custom_model/](tools/rf/ssm/custom_model/) | pure+ui | User-built topologies: netlist → Y → S solver, schematic, build/use/fit UI. | `CustomModel`, `render_custom_section` |
| [rust_kernels/](tools/rf/ssm/rust_kernels/) | — | Rust crate + committed per-platform binaries + parity benchmark. | `src/lib.rs` |

## `tools/dc/`, `tools/tcad/`, `tools/data/`, `tools/portal/`

| path | layer | does |
|---|---|---|
| [dc/b1500a_plot.py](tools/dc/b1500a_plot.py) | page | B1500A curve viewer + parameter extraction + TLM analysis. |
| [dc/hp4155a_plot.py](tools/dc/hp4155a_plot.py) | page | HP/Agilent 4155A quick SMU plots. |
| [tcad/gummel_analyzer.py](tools/tcad/gummel_analyzer.py) | page | Compare simulated Gummel plots against the UIUC reference. |
| [data/csv_process.py](tools/data/csv_process.py) | page | Batch-convert measurement CSV / CITI files; hands results to DC Analysis. |
| [portal/home.py](tools/portal/home.py) | page | Landing page; cards built from `i18n.TOOLS`. |

## `tools/ebeam/` — E-beam lithography

**Deliberately self-contained: imports no other repo module.** It keeps its own
`tr()` and `segmented_radio()` on purpose, so it can be run standalone.

Dependencies run one way — `limits → parser → stream → plotting → exposure →
calculator` — and **nothing imports `calculator.py` back**. Streamlit always
executes it as `__main__`, so importing it under its dotted name would run the
whole page a second time (and call `st.set_page_config` twice).

| path | layer | does |
|---|---|---|
| [calculator.py](tools/ebeam/calculator.py) | page | The page script: header, `set_page_config`, chip-position calc, and `render_page()`. |
| [gdsii/limits.py](tools/ebeam/gdsii/limits.py) | pure | RAM-budget sizing — `_limits_for`, `_Limits`. Also the shared `tr()` for this subtree. |
| [gdsii/parser.py](tools/ebeam/gdsii/parser.py) | pure | Single-pass GDSII decoder, `_PolyLayer` / `_InstancedLayer`. |
| [gdsii/stream.py](tools/ebeam/gdsii/stream.py) | io | Compressed upload store + bounded-memory scan/window path for oversized masks. |
| [plotting.py](tools/ebeam/plotting.py) | ui | Plotly traces, coverage rasters, per-cell area binning. |
| [exposure.py](tools/ebeam/exposure.py) | ui | The Time Calculator. |

## `dev/` — not imported by the app

| path | does |
|---|---|
| [smoke_test.py](dev/smoke_test.py) | **The regression net.** Imports everything, asserts every page path resolves, compares ~50 numerical goldens. Run after any structural change. |
| [build_rust_kernels.py](dev/build_rust_kernels.py) · [check_rust_status.py](dev/check_rust_status.py) | Build / report the Rust acceleration binaries. |
| [profile_*.py](dev/) | Hot-path profilers (bulk upload, rust batch, SSM extraction). |
| [gds/](dev/gds/) | Synthetic GDSII generator + memory/time limits sweep behind the EBL page's budgets. |

---

## Invariants worth knowing before you change anything

1. **`tools/<folder>` == `i18n.GROUP_ORDER` entry.** A tool's folder is
   derivable from its group key and vice versa.
2. **Three registries must agree**, and `dev/smoke_test.py` enforces it:
   `i18n.TOOLS[*]["path"]`, `handoff.PAGE_*`, and `st.Page(...)` in the entry
   point. All are **relative to the repo root** (hence the `tools/` prefix),
   because that is how Streamlit resolves them.
3. **Repo root comes from `tools.common.paths.REPO_ROOT`** — never
   `Path(__file__).parents[N]`, which silently breaks on any move.
4. **Cross-package imports are absolute** (`from tools.common.i18n import tr`).
   Relative imports only *within* a package.
5. **`tools/ebeam/` imports nothing from the repo.** Intentional — see above.
6. **`tools/common/` imports nothing from a tool group.** That is the whole
   point of the package.

## Where to start for a given task

| task | start at |
|---|---|
| Add / rename a page, change sidebar groups | `tools/common/i18n.py` (`TOOLS`, `GROUP_ORDER`) + the new page file |
| Password, language, nav behaviour | `IOED_Tool_Web.py` |
| Launcher, venv, auto-update, dependencies | `LAUNCH_Tool.py` |
| S2P/CSV parsing or Touchstone writing | `tools/rf/ssm/helpers/s2p_io.py` |
| De-embedding math | `tools/rf/ssm/helpers/deembed_math.py` |
| fT/fmax, gain metrics | `tools/rf/ssm/helpers/metrics.py` |
| A model's equations | `tools/rf/ssm/models/<model>.py` |
| Add a new SSM model | subclass in `models/<name>.py`, register in `models/__init__.py::REGISTRY` |
| Tuning sweeps, residuals | `tools/rf/ssm/models/base_ui.py` |
| Fit programmatically / from a script | `tools/rf/ssm/agent_api.py` |
| GDS parsing, EBL exposure times | `tools/ebeam/calculator.py` |
| Chart export, clipboard, xlsx | `tools/common/chart_export.py` |
| Rust kernels | `tools/rf/ssm/helpers/rust_kernels.py` + `dev/build_rust_kernels.py` |

**Do not read:** `.hbttools/`, `.hbttools_build/`, `dev/gds/masks*/`,
`__pycache__/`, `docs/CHANGELOG.md` (long, historical).
