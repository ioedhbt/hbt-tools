# tools/rf/

The three RF pages and the small-signal-model engine underneath them.

## The workflow these pages form

```
at_a_glance.py    upload many bias files → de-embed → fT/fmax → charts
      │ handoff.send(TARGET_EXTRACTION, ...)
      ▼
extraction.py     peel Open/Short → access-R → per-model analytic extraction
      │ handoff.send(TARGET_SIMFIT, params=..., model_short=...)
      ▼
simulator.py      forward-simulate / fit any topology, incl. custom ones
```

`tools/common/handoff.py` is the bus. It carries `S`, `freq`, `z0`, a label, a
`stage` (raw / deembedded / intrinsic), optional seed `params`, and `extras`
(the *other* bias files — Extraction's Z-param, Cold-HBT and τ_total methods
need them). Payloads survive `st.switch_page`; `take()` consumes once.

## Data flow

```
.s2p / VNA CSV
  → parse_s2p / parse_csv          helpers/s2p_io.py   (Rust bulk: parse_and_compute_batch)
  → s_to_y                         helpers/rf_math.py
  → de-embed                       helpers/deembed_math.py  (step_open/step_short/peel_parasitics)
  → metrics                        helpers/metrics.py       (|h21|², Mason U, K, fT/fmax)
  → figures                        helpers/plotly_plots.py, ssm_plots.py
```

## Layering — respect it

| layer | where | rule |
|---|---|---|
| pure maths | `helpers/rf_math, deembed_math, metrics, _array_utils` | no `st.*`, import-safe anywhere, unit-testable |
| I/O + cache | `helpers/s2p_io, fit_cache, rust_kernels` | Streamlit-aware but must work headless |
| figure builders | `helpers/plotly_plots` | return figures, do not render |
| models | `models/*.py` | maths + per-model UI; registered in `models/__init__.py::REGISTRY` |
| orchestration | `main_ssm_extraction, ssm_plots, ssm_override, ssm_access_resistance` | the page's Steps 1–3 |

`agent_api.py` is a headless entry point over the same engine — no `st.*` calls,
usable from a script or a CLI. Keep it that way.

## Gotchas

- **Units are SI everywhere internally** (F, H, Ω, s, Hz). Display scaling
  (fF, pH, GHz) happens only at the UI edge and in `.s2p` headers.
- **`.s2p` headers are a contract.** `write_s2p` emits `Key = value unit`
  (`Cpbe = 12.3400 fF`), and `agent_api._interpret_header` parses it back to
  decide which parasitics `fit()` freezes. A non-zero entry means "this was
  removed"; a listed `0` means it wasn't. Don't change the format on one side.
- **Frequency is Hz** in arrays, GHz only for axes and labels.
- **`S` is `[N, 2, 2]`**, Touchstone order S11 S21 S12 S22 on disk.
- Session-state keys are namespaced per file and model
  (`sim_{topo}_{param}_{fname}`) so several DUTs coexist. Don't drop the
  `fname`.
- The Rust kernels must stay numerically identical to NumPy —
  `rust_kernels/benchmark.py` checks parity. There is no build-hash guard, so
  if you change `src/lib.rs` you must rebuild **all three** platform binaries.

## Adding a model

Subclass `SSMModelTemplate` + `AbstractSSMModel` in `models/<name>.py` (copy
`xu.py`, the smallest), implement `extract` / `simulate` / `_Y_int_*_vec` /
`_Y_int_*_batch`, register it in `models/__init__.py::REGISTRY`, then add rows
to `docs/SSM_INDEX.md`. `dev/smoke_test.py` picks up new registry entries
automatically — re-baseline once and check the numbers look physical.
