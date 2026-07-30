# tools/ebeam/

The JEOL ELS-7000 e-beam lithography calculator: holder positions, a GDS mask
viewer, and exposure-time workflow modes.

## Layout

`calculator.py` used to be one ~6200-line file. It is now the page script
plus a `render_page()` call, with the rest split by seam:

| File | Contains |
|---|---|
| `calculator.py` | Page script Streamlit runs: header/i18n/`segmented_radio`, `st.set_page_config`, `_chip_corner_guide_png`, `_hex_to_rgba`, and `render_page()` (Sections 1-4: chip position, left-computer setup, GDS mask viewer, mode-selector workflows), called once at the bottom. |
| `gdsii/limits.py` | RAM-budget sizing (`_limits_for`, `_mask_budget_mb`, `_Limits`, ...). Lowest layer — nothing else here imports it back. |
| `gdsii/parser.py` | Single-pass streaming GDSII decoder + in-memory layer objects (`_PolyLayer`, `_InstancedLayer`, `_parse_gds`, `_flatten_instanced`). |
| `gdsii/stream.py` | Compressed-upload store, bounded-memory scan/window path for masks too big to fully parse (`_load_gds`, `_load_gds_layers`, `_stream_scan`, `_stream_window`). |
| `plotting.py` | Plotly traces, coverage rasters, per-cell area binning. |
| `exposure.py` | Time Calculator (`_render_time_calculator`, `_polygon_clip_per_cell_mm`). |

Dependency direction is one-way: `limits` -> `parser` -> `stream` ->
`plotting` -> `exposure` -> `calculator.py`. Nothing imports back up that
chain — in particular, no submodule imports `calculator.py`. That is not a
style preference: Streamlit always executes `calculator.py` as `__main__`
(never under the dotted name `tools.ebeam.calculator`), so an import of
`tools.ebeam.calculator` from a module it imports would re-run the whole
page a second time under that name — a second `st.set_page_config()` call,
a second render. `calculator.py` and `gdsii/limits.py` each keep their own
small inline copy of `_is_zh()`/`tr()` for this reason (and `plotting.py`
keeps its own copy of `_hex_to_rgba`); everything downstream of `limits.py`
imports `tr()` from there instead of duplicating it further.

## The one rule

**This tool imports nothing from the rest of the repo.** Not `tools.common`,
not `i18n`, nothing, in any of the files above. It keeps its own inline
`_is_zh()`, `tr()` and `segmented_radio()`.

That duplication is deliberate and the owner has confirmed it: the page is
launched standalone by `launch_ebl_calculator.py` with its own `.ebl_venv/`,
no portal and no password. Do **not** "deduplicate" it against `tools/common/`,
and do not "deduplicate" the split above back into one file either.

The rule covers the files in the table above — the ones the standalone
launcher ships. It is **not** a rule about the folder. `process_flow.py` also
lives here, and imports `tools.common` freely: it is a separate portal page
that shares the directory only because a tool's folder must equal its i18n
group key, and `ebeam` is the group the sidebar labels "Process". It is never
launched standalone. Don't "fix" its imports, and don't let it grow a
dependency on the EBL files (or vice versa) — they are unrelated tools.

Its translations do **not** live in `tools/common/i18n.py`. The illustration is
one self-contained HTML file that has to work when downloaded, so it carries its
own `ZH` table and its own toggle; `process_flow.py` only tells it which
language the portal is on, by rewriting the `HOST` line. Don't move those
strings into `_UI` — that would break the standalone file, which is the whole
point of it.

Consequence to be aware of: `i18n._UI["ebl_corner_guide"]` and
`["ebl_corner_note"]` exist but are unreachable from here. Leave them.

## Memory is the whole design

A GDS mask can be hundreds of MB, and on Streamlit Cloud over-committing is a
SIGKILL, not a swap. So:

- `.streamlit/config.toml`'s `maxUploadSize` is the outer hard ceiling.
- `_limits_for()` sizes the parse guards from the RAM **actually free at that
  moment** — stricter inside a container (cgroup limit), looser on a
  workstation where over-commit only pages.
- Over-budget files are **refused with a message, never a crash**. Preserve
  that property in any change here.
- Past the budget the parser streams instead of building objects.

The numbers behind all of this come from `dev/gds/_profile_gds_limits.py`.
Re-run it before changing any limit.

## Gotchas

- The GDSII parser is hand-written and single-pass. Every record loop guards
  `rlen < 4` and `p + rlen > n` before advancing so `p` strictly increases —
  that is what stops a malformed file from hanging. Keep those guards.
- AREF/SREF transform order is magnification → x-reflection → rotation.
- Coordinates are GDS *database units*; `_gds_unit_to_mm` is the only place
  that should convert.
- Exposure-time results are cached in session state and invalidated by
  `_time_input_fingerprint`. If you add an input that changes the answer, add
  it to the fingerprint — otherwise a stale total from the previous mask stays
  on screen looking current.
- There are three Time Calculator call sites (`ebc_dt`, `ebc_fe`, `ebc_sa`);
  a change to one usually belongs in all three.
