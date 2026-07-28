# tools/ebeam/

The JEOL ELS-7000 e-beam lithography calculator: holder positions, a GDS mask
viewer, and exposure-time workflow modes.

## The one rule

**This tool imports nothing from the rest of the repo.** Not `tools.common`,
not `i18n`, nothing. It keeps its own inline `_is_zh()`, `tr()` and
`segmented_radio()`.

That duplication is deliberate and the owner has confirmed it: the page is
launched standalone by `launch_ebl_calculator.py` with its own `.ebl_venv/`,
no portal and no password. Do **not** "deduplicate" it against `tools/common/`.

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
