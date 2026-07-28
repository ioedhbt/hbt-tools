# tools/common/

Infrastructure shared by **every** tool group. No domain maths lives here.

## The one rule

**Nothing in this package may import `tools.rf`, `tools.dc`, `tools.tcad`,
`tools.data` or `tools.ebeam`.** That is the entire reason the package exists:
these modules used to live inside the RF/SSM engine, so a DC bug dragged you
into the small-signal-model tree. If something here needs a domain concept, it
belongs in that domain, not here.

## What's here

| file | for |
|---|---|
| `paths.py` | `REPO_ROOT`, `EXAMPLES_DIR` |
| `i18n.py` | `TOOLS` registry (drives the sidebar), `t()`, `tr()` |
| `handoff.py` | cross-page session_state bus + the page-path registry |
| `ui_theme.py` | every app-wide CSS rule |
| `widgets.py` | `segmented_radio`, `dedupe_upload_names`, … |
| `chart_export.py` | `plotly_with_dl`, `unique_sheet_name`, xlsx/TSV export |
| `mem_budget.py` | cgroup-aware RAM probe |
| `diagrams.py` | the "How it works" drawer |

## Gotchas

- **`i18n.TOOLS` paths are relative to the repo root** (so they start with
  `tools/`), because Streamlit resolves `st.Page` / `st.switch_page` against
  the main script's directory. They must match `handoff.PAGE_*` exactly —
  `dev/smoke_test.py` asserts both resolve on disk.
- **Folder name == group key.** Adding a group means adding a folder of the
  same name under `tools/`.
- `widgets.py` and `chart_export.py` are re-exported by
  `tools/rf/ssm/helpers/__init__.py` so the ~20 `from ..helpers import
  copy_button` call sites in the SSM tree keep working. If you rename
  something here, check that re-export list.
- `mem_budget` exists because `psutil` reports the *host's* memory, not the
  container's. On Streamlit Cloud that difference is the OOM-killer.
- `tools/ebeam/` deliberately does NOT use this package — it keeps inline
  copies of `tr()` and `segmented_radio()` so it can run standalone. Don't
  "fix" that.
