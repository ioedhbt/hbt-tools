# hbt-tools

Streamlit portal for HBT device measurement: RF S-parameter de-embedding and
small-signal-model extraction, DC curve analysis, and an e-beam lithography
calculator.

## Read this first

**[MAP.md](../MAP.md)** — one line per module, plus the repo's invariants and a
"task → file" table. That is usually all you need. It also lists what *not*
to read.

Only when MAP.md is insufficient:

- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) — boot flow, page anatomy,
  data flow, the layering rules
- [docs/INDEX.md](../docs/INDEX.md) — per-function catalogue outside `tools/rf/ssm/`
- [docs/SSM_INDEX.md](../docs/SSM_INDEX.md) — per-function catalogue for the SSM engine

Each tool folder also has an `AGENTS.md` with its local rules — you pick it up
automatically when working in that folder, so you rarely need the big indexes.

## Before you finish

Run the regression net. It is fast, and it is the only test coverage:

```
.hbttools/bin/python dev/smoke_test.py
```

It imports every module, asserts every page path resolves, and compares ~50
numerical goldens. **If a golden moves, you changed physics** — find out why
rather than re-baselining. Re-baseline (`--baseline`) only when the change was
intended, and say so explicitly in the commit message.

## House rules

- Use the **caveman** skill for chat responses. Write code, comments and commit
  messages normally.
- For large mechanical work (multi-file splits, sweeping renames), delegate to
  Sonnet agents — especially when parts are parallelizable.
- Keep `MAP.md`, `docs/ARCHITECTURE.md` and the two indexes current in the same
  change that alters structure.
- The venv interpreter is `.hbttools/bin/python`, not the system Python.
