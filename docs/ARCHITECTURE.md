# How things work — program outline

Orientation document for humans and AI agents: how the app boots, how a
page is put together, how data flows, and where to look for a given kind
of change. For a per-function catalogue use
[INDEX.md](INDEX.md) (everything outside `tools/rf/ssm/`) and
[docs/SSM_INDEX.md](docs/SSM_INDEX.md) (the SSM tree).

---

## 1. Boot flow

```
LAUNCH_Tool.py                      (double-click / python3)
  ├─ auto_update()                  git pull, or GitHub zip overlay for non-git installs
  ├─ .hbttools/ venv                created on first run; deps installed (CuPy matched to detected CUDA)
  └─ streamlit run IOED_Tool_Web.py   with HBT_LOCAL_LAUNCH=1  → password gate skipped

IOED_Tool_Web.py                    (the ONLY st.set_page_config in the repo)
  ├─ check_password()               st.secrets["APP_PASSWORD"]; skipped on local launch
  ├─ 🌐 language toggle             writes st.session_state["ui_lang"] → tools/i18n
  └─ st.navigation(nav)             sidebar groups built from i18n.TOOLS / GROUP_ORDER
        └─ runs ONE tools/<page>.py per rerun
```

- Deployed on Streamlit Community Cloud the entry point is the same
  `IOED_Tool_Web.py`; no launcher, password gate active, fit cache
  auto-disabled (ephemeral disk).
- The entry point lives at the repo root, so `st.Page` / `st.page_link` /
  `st.switch_page` paths (in `i18n.TOOLS` and `tools/rf/ssm/handoff.py`) are
  written **relative to the root** — i.e. with a `tools/` prefix.
- `launch_ebl_calculator.py` is a second, independent launcher that runs
  **only** `tools/ebeam_calculator.py` (own venv `.ebl_venv/`, no portal,
  no password). That page therefore imports **no repo modules**.

## 2. Anatomy of a portal page (`tools/*.py`)

Every page is a straight-line Streamlit script, re-executed top-to-bottom
on each interaction ("rerun"). Conventions shared by all pages:

- **Title/desc from the registry:** `st.title(i18n.title("<tool_key>"))` —
  never a hard-coded string, so sidebar and page title can't drift.
- **Bilingual UI:** every user-facing string goes through `tools/i18n.py`
  (portal-level `t(key)`) or, in the custom-model builder, the inline
  `tr(en, zh)` helper.
- **"ℹ️ How it works" expander:** `tools/diagrams.pipeline_png(...)`.
- **Uploader reset:** pages keep an integer `*_uploader_key` in session
  state and bump it to programmatically clear `st.file_uploader`.
- **Widget state:** everything the user can tweak lives in
  `st.session_state` under per-file / per-model keys (e.g.
  `sim_{topo}_{param}_{fname}`), so multiple DUTs coexist.
- **Charts:** Plotly via `helpers/plotly_plots.py`, always rendered with
  `plotly_with_dl(...)` (chart + ⬇ xlsx + 📋 copy buttons); publication
  Smith charts via `render_matplotlib_smith` in `tools/rf/ssm/ssm_plots.py`.

## 3. RF data flow

```
.s2p / VNA CSV upload
  → parse_s2p / parse_csv            (helpers/s2p_io.py; Rust bulk path: parse_and_compute_batch)
  → s_to_y                           (helpers/rf_math.py)
  → de-embedding                     (helpers/deembed_math.py: step_open/step_short/peel_parasitics
                                      or deembed_open_short / deembed_thru_half)
  → metrics                          (helpers/metrics.py: |h21|², Mason U, K, fT/fmax extraction)
  → plots                            (helpers/plotly_plots.py, ssm_plots.py)
```

**Cross-page handoff:** `tools/rf/ssm/handoff.py` is a tiny session-state
bus. RF At a Glance → SSM Extraction → RF Simulator pass the active DUT
(`S`, `freq`, `z0`, label, optional seed params + sibling bias files) via
`send(target, ...)` / `take(target)`, surviving `st.switch_page` with no
re-upload.

## 4. SSM extraction engine (`tools/rf/ssm/`)

Layered, bottom-up:

| Layer | Modules | Rule |
|---|---|---|
| Pure math (no Streamlit) | `helpers/rf_math.py`, `deembed_math.py`, `metrics.py`, `_array_utils.py` | Import-safe anywhere, unit-testable |
| I/O + caching | `helpers/s2p_io.py`, `fit_cache.py`, `chart_export.py` | Streamlit-aware but headless-tolerant |
| Plot builders | `helpers/plotly_plots.py` | Return figures, don't render |
| Model classes | `models/*.py` | Math + per-model UI; registered in `models/__init__.py::REGISTRY` |
| Shared model UI | `models/base_ui.py` | `SSMModelTemplate` — override expander, Smith+fT/fmax, Visual/Auto tuning |
| Orchestration | `main_ssm_extraction.py`, `ssm_plots.py`, `ssm_override.py`, `ssm_access_resistance.py` | The extraction page's Steps 1–3 |
| Custom models | `custom_model/` | User-built topologies: `core.py` (SimPlan), `schematic.py`, `ui_build/use/fit.py` |

Extraction sequence rendered by `render_ssm_tab`:
**Step 1** Open/Short dummy → pad caps + lead L (`step_open`/`step_short`) →
**Step 2** de-embed preview → **Step 3** per-model extraction
(`REGISTRY` loop: ChengT, ChengPi; XuT/KY are forward-sim-only) →
summary table + fit cache. Access resistances (Z-param, Cold-HBT,
open-collector) feed a unified pre-override (`ssm_override.py`).

**Adding a new SSM model:** subclass `SSMModelTemplate` +
`AbstractSSMModel` in a new `models/<name>.py` (copy `xu.py` as the
smallest template), implement `extract` / `simulate` / `_Y_int_*_vec` /
`_Y_int_*_batch`, register it in `models/__init__.py::REGISTRY`, and add
its rows to the SSM index.

## 5. Acceleration & persistence

Three compute paths, selected at runtime with silent fallback:

1. **Rust kernels** (`tools/rf/ssm/rust_kernels/` crate → prebuilt binaries
   in `bin/<arch>/`, loaded by `helpers/rust_kernels.py`). Hot paths:
   batched Y→S, end-to-end `sim_*_batch` topology sweeps, and the
   bulk-upload `parse_and_compute_batch`. Build with
   `python dev/build_rust_kernels.py`, check with `python dev/check_rust_status.py`,
   disable with `HBT_DISABLE_RUST=1`.
2. **CuPy / CUDA GPU** for the Auto-Tuning grid sweeps (adaptive VRAM
   block sizing, fp32 sweep + fp64 rerank). NumPy code is written
   `xp`-generically (`xp = numpy or cupy`).
3. **NumPy** — the always-available canonical reference; parity is
   enforced by `tools/rf/ssm/rust_kernels/benchmark.py`.

**Persistence:** fine-tuned SSM params are cached per (DUT, model) as
JSON under `$HBT_FIT_CACHE_DIR` → `~/.hbt-tools/fits/…`
(`helpers/fit_cache.py`; auto-disabled on Streamlit Cloud). Everything
else is `st.session_state` (lost on refresh).

CPU-side Auto Tuning chunk/budget sizes are cgroup-aware via
`helpers/mem_budget.py` (reads `/sys/fs/cgroup/...` before falling back
to `psutil`), so they're sized against Streamlit Cloud's actual
container memory limit instead of the host's — a fixed-size chunk sized
off host RAM was getting SIGKILLed by the cgroup OOM-killer before any
`except MemoryError` path could run. The portal sidebar also shows a
live RAM-usage bar (`tools/ui_theme.py::render_ram_badge`) built on the
same probe.

## 6. EBL calculator (`tools/ebeam_calculator.py`)

Deliberately self-contained (see §1). Page sections:
holder-position calculator → left-computer origin → **GDS mask viewer**
(own single-pass streaming GDSII parser with instanced layers and
RAM-adaptive budgets: `_limits_for()` sizes the parse guards per upload
from the memory actually free — strictly, from the cgroup limit inside a
container, where over-committing is a SIGKILL; more generously on a PC,
where it only pages. Over-budget files are refused with a message, never
a crash. `maxUploadSize` (350 MB in `.streamlit/config.toml`, raised by
`launch_ebl_calculator.py` when run locally) is the outer ceiling; the
numbers behind both come from `gds/_profile_gds_limits.py`) → workflow
modes (dose-time
test / first exposure / second alignment), each with an exposure Time
Calculator (vectorized per-cell area binning; exact polygon clipping for
small N).

## 7. Where to look — task → file

| Task | Start at |
|---|---|
| Add / rename a portal page, change sidebar groups | `tools/i18n.py` (`TOOLS`, `GROUP_ORDER`) + new `tools/<page>.py` |
| Change password / language / nav behaviour | `IOED_Tool_Web.py` |
| Launcher, venv, auto-update, dependency list | `LAUNCH_Tool.py` (portal) / `launch_ebl_calculator.py` (EBL) |
| S2P/CSV parsing, Touchstone writing | `tools/rf/ssm/helpers/s2p_io.py` |
| De-embedding math | `tools/rf/ssm/helpers/deembed_math.py` |
| fT/fmax, gain metrics, extrapolation | `tools/rf/ssm/helpers/metrics.py` |
| Bode/Smith/plateau figure styling | `tools/rf/ssm/helpers/plotly_plots.py` (colors: `FT_FMAX_COLORS`) |
| xlsx / clipboard export of any chart | `tools/rf/ssm/helpers/chart_export.py` |
| SSM extraction page layout / steps | `tools/rf/ssm/main_ssm_extraction.py` + `ssm_plots.py` |
| A specific model's equations or override UI | `tools/rf/ssm/models/<model>.py` |
| Tuning sweeps (Visual / Auto), residuals | `tools/rf/ssm/models/base_ui.py` |
| Custom model builder | `tools/rf/ssm/custom_model/` |
| Rust kernels / dispatch / fallbacks | `tools/rf/ssm/helpers/rust_kernels.py` + `tools/rf/ssm/rust_kernels/` |
| Fit-cache location or format | `tools/rf/ssm/helpers/fit_cache.py` |
| GDS parsing / EBL exposure times | `tools/ebeam_calculator.py` |
| Performance investigation | `dev/profile_*.py`, `tools/rf/ssm/rust_kernels/benchmark.py` |

**Maintenance rule:** structural changes (new page, new model, new
layer/compute path) should update this outline **and** the matching
index in the same change.
