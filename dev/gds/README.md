# `gds/` — GDS mask stress tests for the EBL page

Dev-only tooling for [`tools/process/ebeam/calculator.py`](../../tools/process/ebeam/calculator.py)'s
GDS mask viewer. Nothing here is imported by the app.

| File | Purpose |
|---|---|
| [`_gen_test_gds.py`](_gen_test_gds.py) | Writes synthetic GDSII masks: sub-µm rectangles / circles / triangles in a 2 µm unit cell, repeated over at most 1 × 1 cm. Any size up to hundreds of MB. |
| [`_profile_gds_limits.py`](_profile_gds_limits.py) | Sweeps those masks through the app's real pipeline in fresh subprocesses and reports parse time + peak RSS against the host's RAM budget. **Run this before changing `maxUploadSize` or any budget guard.** |
| `masks/` | Generated masks (git-ignored — regenerate rather than commit). |

## The generated masks

`masks/` holds six files, ~690 MB total. The three `demo_*` masks should
load; the three `guard_*` masks are sized to trip a budget guard and be
**refused with a message** — that refusal is the feature, not a bug.

| File | Size | What it is | Expected in the app |
|---|---|---|---|
| `demo_10MB_sref.gds` | 10 MB | 349 k placements, 2.8 M polygons | loads in ~0.04 s |
| `demo_250MB_sref.gds` | 250 MB | 8.7 M placements, 69.9 M polygons | loads in ~0.7 s |
| `demo_300MB_sref.gds` | 300 MB | 10.5 M placements, 83.9 M polygons — at the upload cap | loads in ~0.7 s |
| `guard_aref_1G_places.gds` | **4 KB** | one AREF claiming 1.0 G placements | refused instantly, no RAM spent |
| `guard_flat_120MB.gds` | 120 MB | 11.2 M vertices of *distinct* geometry | refused on a 3 GB container |
| `guard_bigcell.gds` | 11 MB | one 1.3 M-polygon cell stepped 13× | refused on a 3 GB container |

**The last two are host-dependent by design.** The app's budgets are
derived from the RAM free at parse time, so on a workstation with several
GB spare both of those load fine — as does anything else that genuinely
fits. Only `guard_aref_1G_places.gds` is refused everywhere: no machine
has room for a billion placements. To watch the guards fire on a big
machine, shrink the budget instead:

```python
import tools.process.ebeam.calculator as m
m._free_ram_mb = lambda: (250.0, None, "container")   # pretend it's tight
```

Every mask holds two layers: **L1** the pattern, **L2** four 900 nm
alignment squares on the field corners. Each shape is under 1 µm across
and the whole field stays inside 1 × 1 cm.

`guard_aref_1G_places.gds` is the one worth keeping around: 4 KB on disk
claiming a billion placements is what used to be able to OOM-kill the
container (see CHANGELOG v1.5).

## Regenerating

```bash
# the set above
.hbttools/bin/python gds/_gen_test_gds.py gds/masks/demo_250MB_sref.gds --mode sref --mb 250
.hbttools/bin/python gds/_gen_test_gds.py gds/masks/guard_aref_1G_places.gds --mode aref --places 1e9

# worst case per MB — flat geometry and references both at their guards
.hbttools/bin/python gds/_gen_test_gds.py gds/masks/mixed_300MB.gds \
    --mode mixed --mb 300 --vert-budget 9e6
```

Modes: `sref` (N placements — the realistic big file), `sref_norun`
(a rotation on every reference, so no two neighbouring blocks match —
used to take ~13 s for 250 MB, 0.7 s since the group decoder), `aref`
(tiny file, huge expansion), `flat` (no
repetition), `mixed` (both budgets loaded at once), `bigcell` (a cell
too big to stay instanced), `multi` / `multi_grouped` (several *different*
cells — see below).

## Several different cell types (`multi`, `multi_grouped`)

`--cells N` builds `N` distinct unit patterns — square+circle+triangle,
tilted square+isosceles triangle, rectangle+star, … — instead of one
repeated cell. These modes exist because element *ordering* used to
dominate the cost: `_element_run()` compares a block only against the one
directly after it, so `C0, C1, C2, C0, …` cut every run to length 1 even
though the byte stream is perfectly regular. `_decode_copies()` now finds
the repeating **group** instead, which flattened the difference:

| 300 MB / 10.5 M placements, 3 cells | parse | stream scan | zoom window |
|---|---|---|---|
| `multi_grouped` (one run per cell) | 0.73 s | 0.83 s | 17 ms |
| `multi` (interleaved), before | 18.7 s | 73.6 s | 2 769 ms |
| `multi` (interleaved), after | **0.73 s** | **1.42 s** | **27 ms** |

Keep both modes as regression tests: `multi` is the case that breaks a
run-only decoder, `multi_grouped` the case that must not get slower.
Cell-type *count* barely matters either way — 6 cells cost about what 3
do. `gen_chaos`-style files (no periodicity at all, so the group probe
never succeeds) confirm the probe's backoff: they parse at the same speed
with the probe on or off.

The real ceiling on cell types is the viewer's `_POLY_LIMIT`: a layer
stays instanced only while the **total base polygons summed over all of
its cells** is ≤ 50,000. Crossing it flattens the whole layer:

```bash
.hbttools/bin/python gds/_gen_test_gds.py /tmp/k15k.gds --mode multi_grouped --mb 20 --cells 15000
.hbttools/bin/python gds/_gen_test_gds.py /tmp/k20k.gds --mode multi_grouped --mb 20 --cells 20000
```

| | base polygons | result | peak RSS |
|---|---|---|---|
| 15 000 cells, 26 MB | 47 500 | stays instanced | 185 MB |
| 20 000 cells, 28 MB | 63 332 | flattens to a flat layer | **1.31 GB** |

A 28 MB file wanting 1.3 GB is the sharpest cliff in the whole suite —
worth keeping as a regression test. On a 3 GB container the vertex guard
catches it first and refuses at 197 MB peak, then streams.

## Re-measuring the limits

```bash
.hbttools/bin/python gds/_profile_gds_limits.py \
    --sizes 100,200,250,300,400 --idle-mb 1000 --limit-mb 2500 --reupload
```

`--idle-mb` is what the deployed app already occupies doing nothing and
`--limit-mb` the container ceiling, so the difference is what one upload
may use. Add `--enforce` on Linux to make the cap a hard `RLIMIT_AS`
(Darwin ignores it). The numbers this produced are recorded in
[`.streamlit/config.toml`](../.streamlit/config.toml) next to
`maxUploadSize`.
