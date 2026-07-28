# Restructure plan — hbt-tools

> **Status: executed.** Branch `restructure-and-bugfixes`, on top of
> `808f1ba`. Every problem below is addressed except **P9** — the EBL
> calculator keeps its duplicated `tr()` / `segmented_radio()` and stays
> self-contained, by the owner's decision. Consequently bug **B4** is closed
> as won't-fix and `i18n._UI["ebl_corner_*"]` stay unreachable on purpose.
>
> Also **not** done, deliberately: `csv_process.py`'s five sub-tool uploaders
> have the same widget-GC issue as B1500A (BZ). They take five different file
> types, so the single-cache fix that suited B1500A does not transfer; left
> alone rather than half-fixed.
>
> The safety net in §7 is `dev/smoke_test.py` — 162 checks: imports, page
> paths, **page rendering**, path resolution, and ~50 numerical goldens.
> Run it after any structural change.

Goal: domain folders, cheaper orientation for an AI agent, fewer
cross-domain imports. Written 2026-07-28 against commit `808f1ba`.

Scope: 106 tracked files, ~34 k lines of app Python (excluding
`.claude/skills/`). No test suite exists — see §7 for the safety net this
refactor needs before step 1.

---

## 1. What is actually wrong today

| # | Problem | Evidence |
|---|---|---|
| P1 | **`tools/` is flat** — 18 files, 5 unrelated domains in one folder | `tools/B1500A_Plot.py` sits next to `tools/ebeam_calculator.py` |
| P2 | **Shared helpers live inside the SSM tree** | `IOED_Tool_Web.py`, `B1500A_Plot.py`, `HP4155A_plot.py`, `IOED_Gummel_Analyzer.py`, `ui_theme.py` all `from tools.SSM.helpers import …`. A DC bug drags an agent into the RF engine. |
| P3 | **Two monster files dominate reading cost** | `models/base_ui.py` 6813 lines, `ebeam_calculator.py` 6198 lines — 38 % of app code in 2 files |
| P4 | **One 3330-line function** | `base_ui.py::render_tuning_expander` spans lines 2916–6246 with no nested `def`. Nothing smaller to grep to. |
| P5 | **~2000 lines of module-level Streamlit script** | `ebeam_calculator.py` 4190–6198 is straight-line UI at module scope (2 top-level `with` blocks) — not callable, not testable, not navigable |
| P6 | **Orientation costs ~1200 lines of prose** | `ARCHITECTURE.md` 171 + `INDEX.md` 490 + `SSM/helpers/INDEX.md` 690. `.claude/CLAUDE.md` tells the agent to read the first two before doing anything. |
| P7 | **Path depth is hard-coded in 6 places** | `parents[3]` in `fit_cache.py` / `custom_model/core.py`, `parents[2]` in `agent_api.py`, `parent.parent` in `rust_kernels.py`. Every one breaks on a move. |
| P8 | **Two parallel handoff buses** | `tools/SSM/handoff.py` (RF) and `tools/dc_handoff.py` (DC) — same pattern, same session-state idiom, different files |
| P9 | **EBL page duplicates portal infrastructure** | `ebeam_calculator.py:44-62` re-implements `tr()` and `segmented_radio()`; `i18n._UI["ebl_corner_guide"/"ebl_corner_note"]` are therefore dead keys |
| P10 | **Dev scripts mixed into the app package** | `tools/_profile_*.py` (3 files) are importable as `tools.*` but never used by the app |

Not wrong, leave alone: 106 tracked files with no `__pycache__`/`.DS_Store`
committed, a real `.gitignore`, prebuilt Rust binaries deliberately
tracked, a working CI workflow.

---

## 2. Target layout

Guiding invariant: **`tools/<folder>/` == `i18n.GROUP_ORDER` entry.**
Given a tool key you can derive its folder, and vice-versa.

```
hbt-tools/
├─ IOED_Tool_Web.py            # entry point — MUST stay at root (streamlit main script)
├─ LAUNCH_Tool.py
├─ launch_ebl_calculator.py
├─ README.md  requirements.txt
├─ MAP.md                      # NEW — ≤80-line routing table (see §4)
├─ docs/
│  ├─ ARCHITECTURE.md          # moved from root
│  ├─ INDEX.md                 # moved from root
│  ├─ SSM_INDEX.md             # moved from tools/SSM/helpers/INDEX.md
│  ├─ CHANGELOG.md             # moved from root
│  └─ MIGRATION.md             # NEW — old path → new path table
├─ tools/
│  ├─ __init__.py
│  ├─ common/                  # ← the "general_helpers" you asked for
│  │  ├─ __init__.py           # single import surface
│  │  ├─ paths.py              # NEW — REPO_ROOT, kills every parents[N]  (P7)
│  │  ├─ i18n.py               # ← tools/i18n.py
│  │  ├─ ui_theme.py           # ← tools/ui_theme.py
│  │  ├─ diagrams.py           # ← tools/diagrams.py
│  │  ├─ widgets.py            # ← tools/SSM/helpers/widgets.py      (P2)
│  │  ├─ chart_export.py       # ← tools/SSM/helpers/chart_export.py (P2)
│  │  ├─ mem_budget.py         # ← tools/SSM/helpers/mem_budget.py   (P2)
│  │  └─ handoff.py            # ← SSM/handoff.py + dc_handoff.py merged (P8)
│  ├─ portal/
│  │  └─ home.py
│  ├─ rf/
│  │  ├─ at_a_glance.py        # ← IOED_HBT_RF_extract.py
│  │  ├─ simulator.py          # ← RF_simulator.py
│  │  ├─ extraction.py         # ← SSM_extraction.py  (thin page wrapper)
│  │  ├─ batch_deembedding.py
│  │  └─ ssm/                  # ← tools/SSM/  (lowercased)
│  │     ├─ agent_api.py
│  │     ├─ extraction_page.py # ← main_ssm_extraction.py
│  │     ├─ ssm_plots.py  ssm_override.py  ssm_access_resistance.py
│  │     ├─ helpers/           # rf_math, s2p_io, deembed_math, metrics,
│  │     │                     # plotly_plots, fit_cache, rust_kernels, _array_utils
│  │     ├─ models/  custom_model/  components/  rust_kernels/
│  ├─ ebeam/                   # i18n group key "process" → "ebeam"
│  │  ├─ calculator.py         # page entry, thin (§5)
│  │  ├─ gdsii/                # parser.py  stream.py  limits.py
│  │  ├─ plotting.py
│  │  └─ exposure.py
│  ├─ dc/
│  │  ├─ b1500a_plot.py        # ← B1500A_Plot.py
│  │  └─ hp4155a_plot.py       # ← HP4155A_plot.py
│  ├─ tcad/
│  │  └─ gummel_analyzer.py    # ← IOED_Gummel_Analyzer.py
│  └─ data/
│     └─ csv_process.py
├─ dev/                        # NEW — nothing here is imported by the app (P10)
│  ├─ profile_bulk_upload.py  profile_rust_batch.py  profile_ssm_extraction.py
│  ├─ build_rust_kernels.py  check_rust_status.py     # ← rust_things/
│  ├─ smoke_test.py            # NEW — see §7
│  └─ gds/                     # ← gds/ (EBL stress-test generator)
└─ examples/                   # ← dummy_data_practice/
```

Two decisions worth confirming before step 1:

- **`csv_process.py` → `tools/data/`**, not `tools/dc/`. It converts
  B1500A **and** CITI **and** HP4155A files; its i18n group is already
  `data`. Keeping folder == group is what makes the layout predictable.
- **i18n group key `process` → `ebeam`.** Display labels
  ("Process" / "製程") do not change; only the dict key and the folder
  name, so the invariant holds. If you'd rather keep `process` as a group
  that may later hold other process tools, use `tools/process/` as the
  folder and put the calculator at `tools/process/ebeam/`.

---

## 3. Two structural fixes that must land with the move

These are the difference between a rename and an actual improvement.

### 3.1 Kill relative-import depth coupling

`tools/SSM/**` has 30 dotted relative imports. 18 of them stay inside the
SSM package (`..helpers`, `..models.base_ui`, `..ssm_plots`) and survive
`git mv tools/SSM tools/rf/ssm` untouched. **12 reach outside it and all
break:**

| Form | Count | Files |
|---|---|---|
| `from ..i18n import tr` | 4 | `ssm_access_resistance.py:21`, `ssm_override.py:13`, `ssm_plots.py:21`, `main_ssm_extraction.py:24` |
| `from ...i18n import …` | 8 | `models/{xu:31, cheng:31, kunyang:48, degachi:53, base_ui:24}`, `custom_model/_i18n.py:14`, `helpers/{widgets:27, chart_export:20}` |

**Replace those 12 with absolute imports** (`from tools.common.i18n import
tr`). Keep relatives only *within* a package (`from .rf_math import
s_to_y` inside `helpers/` is fine). One mechanical pass, and the tree is
permanently decoupled from its own depth.

### 3.2 One `REPO_ROOT`, defined once

Create `tools/common/paths.py`:

```python
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
```

Then replace, in order of breakage risk:

| File | Today | Becomes |
|---|---|---|
| `helpers/fit_cache.py:54` | `parents[3] / ".fit_cache"` | `REPO_ROOT / ".fit_cache"` |
| `custom_model/core.py:397` | `parents[3]` | `REPO_ROOT` |
| `agent_api.py:77` | `parents[2]` | `REPO_ROOT` |
| `helpers/rust_kernels.py:65,162` | `parent.parent / "rust_kernels"` | `Path(__file__).resolve().parents[1] / "rust_kernels"` (stays package-relative — correct, just re-verify) |
| `SSM_extraction.py:161` | `parent.parent / "dummy_data_practice"` | `REPO_ROOT / "examples"` |
| `main_ssm_extraction.py:199` | `st.image("tools/SSM/de_embedding_illus.png")` | `st.image(str(Path(__file__).parent / "de_embedding_illus.png"))` — **this one is already a latent bug** (CWD-relative; breaks the moment anything runs from another directory) |

---

## 4. Making the repo cheap for an agent to read

This is the part that pays off on every prompt. The current problem is not
"too many files" — it is **two files too big to grep into, and 1200 lines
of prose at the top of the funnel.**

### 4.1 `MAP.md` at the root — the new entry point (≤80 lines)

One row per module. Nothing else. Example rows:

```
| path | does | key symbols |
|---|---|---|
| tools/common/i18n.py | bilingual registry; TOOLS drives the sidebar | TOOLS, GROUP_ORDER, t, tr |
| tools/rf/ssm/helpers/deembed_math.py | open/short/thru de-embed math, pure numpy | step_open, step_short, peel_parasitics |
| tools/rf/ssm/models/tuning/sweep.py | auto-tuning grid sweep + residual ranking | render_tuning_expander |
```

Rewrite `.claude/CLAUDE.md` to point at `MAP.md` **first** and make
`ARCHITECTURE.md` / `INDEX.md` explicit drill-downs, not required reading:

```markdown
Read MAP.md first — one line per module, that is usually enough.
Only if MAP.md is insufficient:
  - docs/ARCHITECTURE.md — boot flow, data flow, layering rules
  - docs/INDEX.md        — per-function catalogue outside tools/rf/ssm
  - docs/SSM_INDEX.md    — per-function catalogue for tools/rf/ssm
Do not read: .hbttools/, .hbttools_build/, dev/gds/masks*/, __pycache__/
```

### 4.2 Per-folder `AGENTS.md` (15–30 lines each)

Claude Code picks up nested context files automatically, so an agent
editing `tools/rf/ssm/models/` gets the local rules without reading the
690-line SSM index. One per: `tools/common/`, `tools/rf/`,
`tools/rf/ssm/`, `tools/ebeam/`, `tools/dc/`, `dev/`.

Contents: what's in the folder, the entry point, the layering rule ("pure
math here, no `st.*`"), the top 3 gotchas, and how to add a new thing.

### 4.3 Uniform module headers

Every module opens with the same 5 fields, so `head -12 <file>` answers
"do I need this file?" without loading it:

```python
"""One-line purpose.

Layer:      pure-math | streamlit-page | ui-helper | io
Imported by: tools/rf/simulator.py, tools/rf/ssm/extraction_page.py
Imports:     tools.common.i18n, .rf_math
Gotchas:     freq is Hz; S is [N,2,2]; params dict is SI units.
"""
```

### 4.4 The measurable win

Answering "how does auto-tuning pick its grid?" today means reading a
3330-line function. After §5 it means `head -12` on 4 files plus one
~400-line module.

---

## 5. Splitting the two monsters

### 5.1 `tools/SSM/models/base_ui.py` (6813 → ~6 files)

Seams are already visible at existing line boundaries:

| New file | From lines | Contains |
|---|---|---|
| `models/tuning/ranges.py` | 152–283 | `_canonical_tune_key`, `tune_hard_limits`, `informed_default_range`, `_range_step`, `_clamp_to_hard` |
| `models/residuals.py` | 324–384 | `ssm_residual`, `_port_residuals`, `_port_residuals_batch` |
| `models/smith_ui.py` | 385–600 | `render_smith_chart`, `render_smith_with_ftfmax`, `smith_scale_controls`, `sync_pad_from_preov` |
| `models/fit_sections.py` | 601–1197 | `_render_cbex_sweep_tool`, `_render_tau_total_fit_section` |
| `models/param_groups.py` | 1116–1852 | `render_finetune_diagram`, `render_interactive_param_groups` |
| `models/tuning/sweep.py` | 1853–6246 | the preview/sweep machinery **and** `render_tuning_expander` — split further inside (see below) |
| `models/base_ui.py` (kept) | 6247–6813 | `SSMModelTemplate` only, re-exporting the names above for call-site stability |

`render_tuning_expander` (2916–6246) is the hard part. It is one function
with no internal `def`; split it by the UI cards it renders — sweep-grid
setup, brute-force runner, optimized runner, prioritize / minimize-
deviation, and the 🪜 progressive auto-fit card — into 5 private
`_render_*` functions in `models/tuning/`, keeping the public name and
signature exactly as-is so `RF_simulator.py` and `base_ui` callers do not
change. Do this **last** (step 5), on its own commit, with nothing else in
the diff.

### 5.2 `tools/ebeam_calculator.py` (6198 → ~5 files)

Cleaner seams; the boundaries below are already comment-delimited:

| New file | From lines | Contains |
|---|---|---|
| `ebeam/gdsii/limits.py` | 600–744 | `_free_ram_mb`, `_mask_budget_mb`, `_Limits`, `_limits_for` |
| `ebeam/gdsii/parser.py` | 745–1600 | `_PolyLayer`, `_InstancedLayer`, `_apply_ref_transform`, `_parse_gds`, `_flatten_instanced` |
| `ebeam/gdsii/stream.py` | 1600–3055 | compression store, `_run_pass1/2`, `_stream_scan`, `_stream_window`, `_load_gds` |
| `ebeam/plotting.py` | 3056–3760 | traces, coverage grids, rasterization |
| `ebeam/exposure.py` | 3760–4189 | `_polygon_clip_per_cell_mm`, `_render_time_calculator` |
| `ebeam/calculator.py` | 4190–6198 | **wrap the module-level script in `def render_page():`** and call it once at the bottom (P5) |

Then delete the inline `tr()` / `segmented_radio()` copies (lines 44–62)
in favour of `tools.common` — which also revives the two dead
`ebl_corner_*` i18n keys (P9).

⚠️ One constraint to preserve: `launch_ebl_calculator.py` runs this page
standalone, and `ARCHITECTURE.md §1` states the page imports **no repo
modules** on purpose. Splitting it into `tools/ebeam/*` is fine (the
launcher already resolves the repo layout at `launch_ebl_calculator.py:33`
and must be updated to `tools/ebeam/calculator.py`), but importing
`tools.common` makes the page depend on `tools/common/` + `tools/i18n`
being present. Either accept that (they always are, in both the repo and
the zip install) or keep the page self-contained and just accept the
duplication. **Recommend: accept the dependency, drop the duplication** —
but this is your call, so it is called out rather than assumed.

---

## 6. Execution order

Each phase is one commit and independently revertable.

| Phase | Work | Risk | Breaks if wrong |
|---|---|---|---|
| **0** | `MAP.md`, per-folder `AGENTS.md`, rewrite `.claude/CLAUDE.md`, `dev/smoke_test.py` (§7). **No file moves.** | none | nothing |
| **1** | Create `tools/common/`; `git mv` i18n, ui_theme, diagrams + widgets, chart_export, mem_budget out of `SSM/helpers/`; merge the two handoff buses; add `paths.py`. Convert cross-package relatives → absolute (§3.1). | low | 6 import sites + `helpers/__init__.py` re-exports |
| **2** | `git mv` pages into `rf/ dc/ tcad/ data/ portal/`; update `i18n.TOOLS` paths (8), `handoff.PAGE_*` (3+1), `IOED_Tool_Web.py:126`. | medium | sidebar nav, every `st.switch_page` |
| **3** | `git mv tools/SSM tools/rf/ssm`; fix `parents[N]` (§3.2), the `st.image` CWD bug, `.github/workflows/build-rust-linux.yml` (6 paths), `LAUNCH_Tool.py:447`, `rust_kernels/benchmark.py` sys.path. | **high** | Rust kernel loading, fit cache location, CI |
| **4** | `git mv` dev scripts → `dev/`, `dummy_data_practice/` → `examples/`, `rust_things/` → `dev/`, docs → `docs/`. Update README + CHANGELOG references. | low | profiling scripts, example loader |
| **5** | Split `ebeam_calculator.py` (§5.2). | medium | EBL page + its standalone launcher |
| **6** | Split `base_ui.py` (§5.1). | **high** | every model, `RF_simulator.py` |

Do **not** combine 3 with 4, or 5 with 6. Phase 3 and 6 each deserve a
clean diff you can bisect.

### Migration hazard — zip installs

`LAUNCH_Tool.py::_overlay_tree` is documented "add/replace only — never
delete". After phases 2–4, every non-git install keeps the old files
forever alongside the new ones: two copies of `i18n.py`, two of the SSM
tree. Nothing imports the stale copies, so it is not a crash — but it
doubles the install and will confuse the next person who greps it.

Fix in phase 2: add an explicit orphan list to `LAUNCH_Tool.py`, applied
once after a successful overlay.

```python
_ORPHANED_AFTER_RESTRUCTURE = [
    "tools/i18n.py", "tools/ui_theme.py", "tools/diagrams.py",
    "tools/SSM", "tools/B1500A_Plot.py", ...
]
```

Also widen `_purge_project_pycache()` — it walks `ROOT/__pycache__` and
`ROOT/tools/**` only, which still covers the new tree, but `dev/` and
`docs/` will not be covered after phase 4.

---

## 7. The safety net this refactor needs first (do in phase 0)

There is no test suite. A 6-phase move without one is how a silent
numerical regression gets shipped. `dev/smoke_test.py`, ~60 lines, no test
framework required:

1. **Import every module** under `tools/` and `dev/` — catches every
   broken import, relative-depth error and circular import in phases 1–4.
2. **Assert every `i18n.TOOLS[k]["path"]` exists on disk**, and every
   `handoff.PAGE_*` / `dc_handoff.PAGE_*` constant too — catches the
   phase-2 nav breakage, which is otherwise invisible until you click.
3. **Numerical golden test**: run `agent_api.fit()` on
   `examples/ADSsim_measured_ChengT_1p5V_Ib100u.s2p` and assert the total
   residual matches the pre-refactor value to 1e-6. Record the baseline
   **before** phase 1. This is what makes phase 6 safe.
4. **`Path` resolution check**: assert `fit_cache.cache_path_str()` and
   `rust_kernels._BIN_DIR` point where they did before the move.

Run it at the end of every phase. Wire it into
`.github/workflows/` as a second job so CI catches it too.

---

## 8. Bugs found

Every item below was read and confirmed in the source — nothing is
inferred from a pattern match. Severity is about user-visible wrongness,
not code cleanliness.

| ID | Severity | Where | One line |
|---|---|---|---|
| BX | **critical** | `HP4155A_plot.py`, `csv_process.py` | `delim_whitespace` removed in pandas 3 — parsing silently broken on the installed venv **today** |
| BY | **high** | `helpers/chart_export.py` +4 | truncated Excel sheet names collide and **interleave** two devices' data |
| B0 | **high** | `models/cheng.py`, `models/xu.py` | τC uses a different (uncorrected) formula in `reextract` than in `extract` |
| B0d | **high** | `SSM/agent_api.py` | `fit()` never reads `removed_params`; `"deemb"` substring match is unanchored |
| B0b | medium | `models/cheng.py` | `overrides.get("Cbex") or …` silently discards a user's `0.0` |
| B0c | medium | `models/cheng.py` | π path omits the `abs()` clamp T applies to the identical Cbcx formula |
| B0e | medium | `tools/batch_deembedding.py` | header keys (`Cpbe_fF`) unparseable by `agent_api` — **fix before B0d** |
| B0i | medium | `helpers/metrics.py` | SSM pages' fT extractor lacks the noise filter At-a-Glance's has |
| B1 | medium | `tools/ebeam_calculator.py` | exposure-time result never invalidated when the mask changes |
| B3b | medium | `IOED_Tool_Web.py` | password gate falls back to hardcoded `"IOED"` when secrets are absent |
| BZ | medium | `B1500A_Plot.py`, `csv_process.py` | radio-switching sub-pages garbage-collects uploaded files |
| BW | medium | `tools/batch_deembedding.py` | parasitic overrides never re-seed when a new Open/Short dummy is uploaded |
| BV | low | 3 RF pages, Gummel | `all_data` keyed on filename — duplicate names drop a file |
| BU | low | `LAUNCH_Tool.py` | unhandled first-run setup failure; console closes before the traceback is readable |
| B0f | low | `helpers/fit_cache.py` | fixed `.tmp` path defeats the atomic write |
| B0g | low | `helpers/metrics.py` | one of two `h21` implementations lacks its divide guard |
| B0h | low | `helpers/rf_math.py` | bare `except:` turns real errors into NaN |
| B2 | low | `SSM/main_ssm_extraction.py` | `st.image` path is CWD-relative |
| B3 | low | `models/base_ui.py`, `custom_model/ui_use.py` | `tr` shadowed by a loop variable (latent) |
| B3c | low | `.gitignore` | `.streamlit/secrets.toml` not ignored |
| B4 | low | `tools/ebeam_calculator.py` | duplicated `tr`/`segmented_radio`; 2 dead i18n keys |

Suggested order: **BX first** — it is a three-line fix and the page is
broken for every user on a current pandas. Then BY, then **B0e → B0d**
(that pairing matters — see B0e), then B0, B0b, B0c, then the rest as
convenient.

None of these depend on the restructure, and none of them block it. Fix
BX/BY before phase 1 so the golden-residual baseline in §7 is taken
against correct parsing.

### B1 — Stale exposure-time result survives a mask change (medium, silently wrong output)

`tools/ebeam_calculator.py:3996` writes `{prefix}_time_result` into session
state; `:4027` reads and displays it unconditionally on every rerun. Those
are the **only two** occurrences of `_time_result` in the file — nothing
ever invalidates it, and the file contains no `on_change=` callbacks at
all. "Remove mask" (`:4258`) pops `_ebc_gds_store` only.

Failure: compute exposure time for mask A → upload mask B, or change chip
size / dose / Nx / Ny / selected cell / layer, or remove the mask entirely
→ mask A's breakdown and total stay on screen looking current. A user can
take a dose/time setting computed for the wrong mask to the tool. Affects
all three call sites (`ebc_dt`, `ebc_fe`, `ebc_sa`).

Fix: store an input fingerprint next to the result and drop the result
when it changes.

```python
_fp = (store_digest, cell, layer, chip_mm, dotmap, dose, nx, ny)
if st.session_state.get(f"{prefix}_time_fp") != _fp:
    st.session_state.pop(f"{prefix}_time_result", None)
```

### BX — `delim_whitespace` was removed in pandas 3; HP4155A parsing is silently broken **right now** (**critical**)

Reproduced against this repo's own runtime venv, not inferred:

```
$ .hbttools/bin/python -c "import pandas; print(pandas.__version__)"
pandas 3.0.3
$ .hbttools/bin/python -c "...pd.read_csv(..., delim_whitespace=True, engine='python')"
TypeError: read_csv() got an unexpected keyword argument 'delim_whitespace'
```

Three call sites: `tools/HP4155A_plot.py:47`, `tools/csv_process.py:689`,
`tools/csv_process.py:728`. `requirements.txt` pins `pandas>=2.0.0` with
no ceiling, and the kwarg was removed in pandas 2.2/3.x — so **every fresh
install gets the broken version.** `.hbttools/` already has it.

Each site swallows the `TypeError` and falls back to comma parsing:

```python
try:
    df = pd.read_csv(io.StringIO(raw), delim_whitespace=True, engine="python")
except Exception:                      # bare `except:` at csv_process.py:729
    df = pd.read_csv(io.StringIO(raw), engine="python")
```

Whitespace-delimited HP4155A data parsed as CSV yields **one garbage
column**. `parse_smu_table` then matches no columns and returns `{}`.
`st.success("Loaded: <file>")` still fires. The user sees an empty chart
and **no error of any kind**.

Fix — one-line change at each of the three sites, plus narrow the except
so a genuine parse failure is visible:

```python
df = pd.read_csv(io.StringIO(raw), sep=r"\s+", engine="python")
```

`sep=r"\s+"` is the documented replacement and works on pandas 2.0
through 3.x, so no version pin is needed. Worth adding one anyway
(`pandas>=2.0,<4`) to stop the next silent removal.

### BY — Excel export collides truncated sheet names and **merges** the data (**high**, corrupted output)

`helpers/chart_export.py:569` — `sheet_name` is
`re.sub(...)(Path(k).stem)[:28]` with no de-duplication. Same pattern at
`IOED_Gummel_Analyzer.py:508` and `csv_process.py:629,656,669` (`[:31]`).

Reproduced: writing two DataFrames to one sheet name does not raise, and
the result is not a clean overwrite — it is an **overlay**:

```
sheets: ['LongCommonPrefix_run']
   Unnamed: 0  b
0           0  9      ← second frame's row
1           1  2      ← first frame's leftovers
2           2  3
```

Failure: lab filenames that share a prefix past 28 characters —
`Wafer3_Die12_2x20um_VCE1.0_IB100uA_run1.s2p` and `…_run2.s2p` both
truncate to `Wafer3_Die12_2x20um_VCE1.0_I` — export to one sheet holding
row-wise interleaved data from both devices. Not a missing sheet the user
would notice; plausible-looking numbers that are wrong.

Fix: dedupe before writing.

```python
seen = {}
base = re.sub(r"[:\\/*?\[\]]", "_", Path(k).stem)[:28]
n = seen.get(base, 0); seen[base] = n + 1
if n: base = f"{base[:25]}_{n}"
```

### B0 — τC is computed by two different formulas depending on code path (**high**, silently wrong physics)

Affects `models/cheng.py` (ChengT) **and** `models/xu.py` (XuT), same
duplication in both.

Fresh extraction — `_step3_T` (`cheng.py:200`, `xu.py:117`):

```python
# [Eq. 31 corrected] τC from phase of α
# arg(α) = −ω·τC − arctan(ω·τB)  →  τC = [−arg(α) − arctan(ω·τB)] / ω
tauC_arr = (-np.angle(alpha_arr) - np.arctan(omega * tauB_arr)) / (omega + 1e-40)
```

Interactive re-extraction — `reextract()` (`cheng.py:1185,1200`,
`xu.py:903,918`), fired whenever the user nudges α₀ or τB:

```python
V_arr    = 2.0 * omega * tauB_ov / (U_arr + 1e-30)
tauC_arr = -np.arctan(V_arr / np.sqrt(np.maximum(1.0 - V_arr**2, 1e-30))) / (2.0 * omega)
```

`reextract` calls `_step3_T` first (getting the corrected τC), then
**overwrites** it with the second formula. The comment `[Eq. 31
corrected]` says plainly that the `V`-based form is the one that was
fixed — the fix just never propagated into `reextract`.

Failure: extract a device (τC from the corrected formula) → nudge the α₀
number_input by any amount in "📊 Interactive Parameter Extraction" → τC
silently reverts to the uncorrected formula. The Smith fit, the forward
`simulate_vec`, the summary table and anything cached from that session
now carry a τC no fresh `extract()` would ever produce for the same
α₀/τB. Nothing warns.

Fix: replace the `V_arr` block in all four `reextract` sites with the
phase-based formula.

### B0b — `overrides.get(k) or fallback` discards a user's `0.0` (medium, silently ignored input)

`cheng.py:1152` (T) and `cheng.py:1461` (π):

```python
Cbex = float(overrides.get("Cbex") or abs(safe_median(Cbex_arr, n_low)))
```

`0.0` is falsy, so typing `0` into the Cbex field — a normal editable
Group-0 input, and the natural way to test the model with the extrinsic
base cap removed — makes `reextract` silently substitute the auto-computed
value. The UI shows `0`; the simulated model uses something else.

The correct pattern is already in the same function, 20 lines below:

```python
Cbcx = float(overrides["Cbcx"]) if (changed_group_idx >= 1
                                    and "Cbcx" in overrides) else Cbcx_recomp
```

Fix: use membership, not truthiness, for `Cbex` too.

### B0c — π-topology skips the sign clamp that T applies to the *same* formula (medium)

`_step2_T` clamps both: `Cbex = abs(safe_median(...))` (`cheng.py:90`),
`Cbcx = abs(safe_median(...))` (`:107`). `_step2_pi` clamps neither
(`:131`, `:146`), and `ChengPi.reextract`'s `Cbcx_recomp` (`:1477`) omits
the `abs()` that `ChengT.reextract` applies (`:1167`).

The Cbcx half is the clear defect — `_step2_pi`'s own docstring says
"Cbcx same formula as T-topology [Eq. 22]", and it is the identical
expression, yet only one path clamps it. On noisy or narrow-band
de-embedded data where the median comes out slightly negative, ChengPi
returns a negative Cbcx that flows straight into `Ybcx = 1j*w*Cbcx` — a
negative shunt capacitance in the forward S-parameters and in the seeded
fine-tune value, with no warning. T would have clamped the same input.

(The Cbex half is less clear-cut: π uses a different expression
[Eqs. 26–28], so its sign convention may legitimately differ. Worth a
deliberate decision rather than a copy of T's `abs()`.)

### B0d — `agent_api.fit()` ignores `removed_params`, and "deemb" matches unanchored (**high**, silently wrong fit)

Two defects in one path, and they interact.

**(i) The documented per-parameter freeze does not exist.** `fit()`
(`agent_api.py:664`) does:

```python
if not user_specified and meta.get("deembedded"):
    for k in all_keys:
        if _canonical_tune_key(k, label) in _PARASITIC_KEYS:
            fixed[k] = 0.0
```

It freezes a hardcoded 6-key set off a **boolean**. `removed_params`
appears at exactly three lines in the whole module — 197 (built), 214
(stored), 227 (a docstring claiming `fit()` uses it). `INDEX.md` makes the
same claim. It is never read.

**(ii) The boolean is set by an unanchored substring match**
(`agent_api.py:192`):

```python
deembedded = ("deemb" in name_l or any(tok in header_text for tok in _DEEMBED_TOKENS))
```

`raw_not_deembedded.s2p`, `before_deembed.s2p`, `to_be_deembedded.s2p`
all contain `deemb`.

Failure: point `fit()` at a raw measured file whose name merely contains
that substring → Cpbe/Cpce/Cpbc/Lb/Lc/Le are pinned to 0 → every remaining
parameter absorbs the real parasitics, and the returned residual looks
fine. No error, no warning.

Fix: anchor the token (`re.search(r'(?:^|[_\-.])deemb', name_l)`) **and**
gate per key on `k in meta["removed_params"]`.

### B0e — batch-de-embedded files write header keys `agent_api` cannot parse (medium — and it booby-traps the B0d fix)

The project convention (`models/__init__.py::get_s2p_header_params`) is
`key = "value unit"`:

```python
out[k] = f"{para_eff.get(k,0)*1e15:.4f} fF"      # -> "! Cpbe = 1.2340 fF"
```

`batch_deembedding.py:386-389` instead bakes the unit into the key:

```python
params={"Cpbe_fF": Cpbe*1e15, "Lb_pH": Lb*1e12, "Rb_Ohm": Rb, ...}
```

`_HDR_KV_RE`'s key group `([A-Za-z_][A-Za-z0-9_]*)` swallows `Cpbe_fF`
whole, the unit group captures `""`, and `Cpbe_fF` is not in
`_PAD_HEADER_KEYS` — so every file the Batch De-embedding tab exports
reports `meta["removed_params"] == {}` while `meta["deembedded"]` is
`True` (from the `_deemb.s2p` suffix).

**Fix B0e before or with B0d.** Today B0d's bug masks this one. The
moment `fit()` starts honouring `removed_params`, batch-exported files
report an empty dict and their parasitics would be fitted **free** instead
of frozen — turning a silent metadata bug into a silent numerical
regression on exactly the files the feature exists for.

### B0f — `fit_cache` "atomic" write uses a fixed temp path (low)

`helpers/fit_cache.py:179`:

```python
tmp = path.with_suffix(path.suffix + ".tmp")
```

Deterministic, so two tabs saving the same `(DUT, model)` share one temp
file; one `json.dump` can be interleaved with the other before either
`tmp.replace(path)`. Fix: `tempfile.NamedTemporaryFile(dir=path.parent,
delete=False)`.

### B0g — two h21 implementations, one missing its divide guard (low)

`metrics.py:28` `h21 = -y21 / (y11 + 1e-30)` vs `metrics.py:187`
`h21 = -y21/y11`. The unguarded one feeds the At-a-Glance / bulk-upload /
batch-de-embedding Bode and plateau tables, so an exact `y11 == 0` sample
(reachable with idealized or simulated Open data) puts `inf` into the
`|h21|² (dB)` column and into the exports. Add the same epsilon.

### B0i — the two fT/fmax extractors disagree, and the SSM pages use the unguarded one (medium)

Same file, two implementations of the same physical quantity:

| | `find_ft_fmax` (`metrics.py:37`) | `extract_limit` (`metrics.py:208`) |
|---|---|---|
| which crossing | `idx[0]` — the **first** | `crossings[::-1]` — the **highest-frequency** |
| noise filter | none | gain must stay ≥ 0 dB for **10 consecutive points**, plus a late-band short-run guard |

`extract_limit`'s docstring states the purpose outright: *"filters out
noise crossings."* So the failure mode is known — it just was never
applied to `find_ft_fmax`, which is what feeds the **SSM Extraction** and
**Simulation & Fitting** pages (`ssm_plots.py:337,630,832,998`,
`RF_simulator.py:911`). `extract_limit` feeds At-a-Glance, bulk upload and
batch de-embedding.

Failure: a measured trace with a noise dip below 0 dB anywhere before the
real roll-off. At-a-Glance reports the correct fT; the SSM pages report
the spurious low-frequency crossing — for the same file, in the same
session, after a handoff. Nothing flags the disagreement.

Fix: port `extract_limit`'s run-length filter and reverse scan into
`find_ft_fmax`, or route the SSM callers through `extract_limit`.

### B0h — bare `except:` in the per-frequency S-conversion (low)

`helpers/rf_math.py:56` — `except:` (not even `except Exception`) around
`np.linalg.inv`, returning NaN. Called from every model's `simulate()`
loop, so a genuine `TypeError`/`KeyError` from a malformed params dict
upstream becomes one unremarkable NaN point in a several-hundred-point
sweep. Narrow it to `except np.linalg.LinAlgError:`.

### B2 — `st.image` path is CWD-relative, not file-relative (low, crashes a panel)

`tools/SSM/main_ssm_extraction.py:199`

```python
st.image(image="tools/SSM/de_embedding_illus.png")
```

`LAUNCH_Tool.py:519` spawns `streamlit run <APP_FILE>` **without
`cwd=ROOT`**, and Streamlit does not chdir to the script directory. Launch
from anywhere except the repo root (`python3 ~/hbt-tools/LAUNCH_Tool.py`
from `$HOME` is the obvious case) and expanding "🖼️ Illustration" on the
SSM extraction page raises instead of showing the figure — there is no
`try`/`except` around it.

Fix (and it removes one of the phase-3 hazards in §3.2):

```python
st.image(str(Path(__file__).parent / "de_embedding_illus.png"))
```

Optionally also pass `cwd=str(ROOT)` to the `subprocess.check_call` in
`LAUNCH_Tool.py`.

### B3 — `tr` shadowed by a loop variable in two functions (low, latent)

```python
from ...i18n import tr          # base_ui.py:24 / _i18n.py via ui_use.py:38
...
for tr in extended_smith_grid(1.0):    # base_ui.py:407, ui_use.py:53
    fig.add_trace(tr)
```

Assigning `tr` makes it **local to the whole function**, so the imported
translator is unreachable there — `UnboundLocalError` before the loop, a
plotly trace object (`TypeError: not callable`) after it.

Harmless today only because `render_smith_chart` (base_ui 385–486) and
`_smith_fig` (ui_use 48–75) happen not to call `tr`. It becomes a crash
the first time anyone adds a bilingual string to either function — likely,
since every neighbouring function has them.

Fix: rename the loop variable (`for _grid_tr in …`). The same loop shape
appears at `plotly_plots.py:171,500,797,1210` and `RF_simulator.py:266`,
which are safe (neither module binds a bare `tr`) — rename them too for
consistency.

### B3b — password gate falls back to a hardcoded value on *any* exception (medium, deployment hygiene)

`IOED_Tool_Web.py:72-75`

```python
try:
    correct_pwd = st.secrets["APP_PASSWORD"]
except Exception:
    correct_pwd = "IOED"   # 本機測試預設密碼
```

The comment says "local testing default", but `except Exception` also
catches the *missing secrets file* case — which is exactly what a public
Streamlit Cloud deployment looks like before `APP_PASSWORD` is set in the
dashboard. The gate then silently accepts `IOED`, which is committed in
plaintext in this repo. Since `_LOCAL_LAUNCH` already handles the
developer-machine case (`:66-68`, `:96`), the fallback is not needed for
local use at all.

Fix: fail closed off the local-launch flag instead.

```python
try:
    correct_pwd = st.secrets["APP_PASSWORD"]
except Exception:
    st.error("APP_PASSWORD is not configured for this deployment.")
    st.stop()
```

### B3c — `.streamlit/secrets.toml` is not gitignored (low, latent credential leak)

`.gitignore` covers `.hbttools/`, `__pycache__/`, `s2p/`, `gds/masks*/` —
not Streamlit's secrets file, and `.streamlit/config.toml` **is** tracked,
so the directory is already in the repo. The moment anyone creates
`.streamlit/secrets.toml` to test the gate locally, `git add .` commits
the password. Nothing has leaked yet (no secrets file is tracked today).

Fix: add `.streamlit/secrets.toml` to `.gitignore` now, before it exists.

### BZ — sub-page radio switching garbage-collects uploads (medium, data loss on navigation)

`B1500A_Plot.py:269` puts `st.file_uploader` inside
`if page == "B1500A Viewer":`, where `page` comes from a same-script
radio — **not** `st.tabs`, which always renders every tab body. Streamlit
GCs a widget's session_state when it is skipped for one script run, so
switching to "TLM Analysis" and back wipes the uploaded files. Same shape
across `csv_process.py`'s five sub-tool branches (`:386,488,543,578,607`).

`IOED_Tool_Web.py:139-152` documents this exact mechanism and already
patches it — but only for the `sim_`/`rfsim_`/`smith_scale_` prefixes.
These pages aren't covered.

Fix: `st.tabs` instead of if/elif-on-radio, or extend the keep-alive
prefix list to the uploader keys.

### BW — batch de-embedding overrides never re-seed on a new dummy file (medium, silently wrong parasitics)

`batch_deembedding.py:91-97` (`_ovr_input`, used at `:190-202`) seeds the
Cpbe/Cpce/Cpbc/Lb/Lc/Le `number_input`s from `params_open`/`params_short`
**only when the session_state key is absent**. Upload a different
Open/Short dummy in the same session and those are recomputed, but the
widgets keep the first file's values. De-embedding then applies the old
calibration's parasitics to the new data, with no warning — the only way
out is the "↺ Reset to defaults" button at `:178-185`, which the user has
no reason to suspect they need.

Fix: key the seed on a hash of the dummy data and re-seed when it changes.

### BV — `all_data` keyed on upload filename (low, one file silently dropped)

`IOED_Gummel_Analyzer.py:265`, `IOED_HBT_RF_extract.py:323,360,376`,
`SSM_extraction.py:175` all key by `f.name`. A multi-file selection
containing two same-named files from different folders (`data.csv` twice)
drops one from every downstream analysis and export, no warning. Same
defect class as BY, lower likelihood.

### BU — `LAUNCH_Tool.py` first-run failure is invisible on Windows (low)

`main()` (`:389`, called unguarded at `:526`) has no top-level
`try/except`, and `_pause_if_interactive()` (`:376-386`) is wired only to
the missing-`APP_FILE` check (`:390-394`) — not to venv creation or
`pip_install` (`:422-425`). A `CalledProcessError` during first-time setup
(network drop, disk full, bad wheel) propagates unhandled; on a
double-click launch the console closes before the traceback can be read,
leaving `.hbttools/` half-built. It self-heals via `check_missing()` next
run, but the user gets no diagnostic at the moment it matters.

Fix: wrap `main()` and route the handler through `_pause_if_interactive()`.

### B4 — EBL page duplicates portal code, stranding two i18n keys (low, dead code)

`tools/ebeam_calculator.py:44-62` re-implements `_is_zh()`, `tr()` and
`segmented_radio()` because the page is deliberately import-free
(`ARCHITECTURE.md §1`). Consequence: `i18n._UI["ebl_corner_guide"]` and
`["ebl_corner_note"]` are defined and translated but reachable by nothing.
Resolved by §5.2 if you accept the `tools.common` dependency; otherwise
delete the two dead keys.

### Checked and found clean

Worth recording so nobody re-audits these:

- **Touchstone handling is correct** — the classic bug spot. S11/S21/S12/S22
  column order, z0 normalization, frequency-unit scaling (HZ/KHZ/MHZ/GHZ)
  and RI/MA/DB format handling in `helpers/s2p_io.py` were all checked
  against the spec. So was the open/short peel ordering in
  `helpers/deembed_math.py`. No defect.
- **No missing i18n keys.** Cross-checked all 91 `_UI` keys against every
  `i18n.t("…")` / `t("…")` call site in the repo — zero runtime `KeyError`s.
- **No undefined names anywhere** (pyflakes over `tools/`, `gds/`,
  `rust_things/`, the three root scripts). Only unused locals and
  missing-placeholder f-strings.
- **The uncommitted `ebeam_calculator.py` diff is clean** — it deletes
  `_drain_sref_bucket`, which has zero remaining references.
- **GDSII parser loop bounds are safe** — `_parse_gds`,
  `_iter_stream_events`, `_decode_structure_bytes` and `_walk_local` all
  guard `rlen < 4` and `p + rlen > n` before advancing, so `p` strictly
  increases; no infinite loop on a malformed record.
- **AREF/SREF transform order** (magnification → x-reflection → rotation,
  `:826-841`) and the AREF lattice stepping match the GDSII spec; the
  `cols`/`rows` divisions are guarded upstream.
- **`_gds_real8` excess-64 decode** is correct.
- **The stale-looking Windows Rust binary is fine.** `bin/win_amd64/*.pyd`
  is from `f52623e` (2026-07-07) while `src/lib.rs` last changed in
  `f4958e8` (2026-07-19) — but that change only added
  `#[cfg(target_os = "windows")]` to the mimalloc global allocator, which
  the older Windows build already had unconditionally. Behaviour is
  identical.

  **Process risk, though:** nothing detects the general case.
  `helpers/rust_kernels.py` only feature-sniffs
  (`hasattr(_rk, "parse_and_compute_batch")`, `:787`) — there is no
  version or build-hash guard, so the next `lib.rs` change that lands
  without rebuilding all three binaries will silently ship different math
  per platform. Add a `__build_sha__` to the crate and assert it on load.
