# dev/

Nothing here is imported by the app. Scripts only.

## smoke_test.py — run it

```
.hbttools/bin/python dev/smoke_test.py
```

This repo has no test framework. `smoke_test.py` is the entire safety net:

1. **Imports every module** under `tools/` (and every script under `dev/`) —
   catches broken imports, wrong relative depth, circular imports.
2. **Asserts every page path resolves**: `i18n.TOOLS[*]["path"]`,
   `handoff.PAGE_*`, and each `st.Page(...)` in the entry point. Nav breakage
   is otherwise invisible until someone clicks.
3. **Compares ~50 numerical goldens** against `_smoke_baseline.json`: forward
   simulation for every registered model, `.s2p` parsing, S/Y/Z conversions,
   Open/Short peel, fT/fmax, analytic extraction for Cheng T/π, `reextract`
   under user overrides, de-embed detection and freeze policy, residuals.

**A moved golden means you changed physics.** Find out why. Re-baseline only
when the change was intended, and say so in the commit message — the baseline
diff is a reviewable claim, which is the point of committing it.

Page modules are import-skipped (`_SKIP_IMPORT_SUFFIXES`) because importing a
Streamlit page executes it; they are covered by the path check instead. If you
add a page, add it to that list.

## Everything else

| script | for |
|---|---|
| `build_rust_kernels.py` | compile the crate → `tools/rf/ssm/rust_kernels/bin/<arch>/` |
| `check_rust_status.py` | report whether the Rust extension loads here |
| `profile_bulk_upload.py` | bulk `.s2p` upload hot path |
| `profile_rust_batch.py` | Rust batch vs the per-file Python loop |
| `profile_ssm_extraction.py` | the extraction maths pipeline end to end |
| `gds/_gen_test_gds.py` | synthesise GDSII stress-test masks |
| `gds/_profile_gds_limits.py` | the memory/time sweep behind the EBL page's budgets |

## Gotcha

Scripts that rewrap `sys.stdout` for UTF-8 must do it under
`if __name__ == "__main__":`. At module scope it closes the interpreter's real
stdout for anything that imports the package — which is exactly how it was
found.
