"""second-exposure.md — SUPERSEDED.

The full Second Alignment walkthrough (layer pick, Custom cross preset,
Existing Pattern, Second Alignment Pattern, Overlayed, Time Calculator) does
not fit run.sh's ~40s ceiling in one pass — the Overlayed section's eager
per-cell polygon clip against the gdstk shim is slow enough that stacking
all of settle()'s waits blows the budget (see git history / commit message
for the run log that diagnosed this).

Split into two scenarios instead, each replaying the same setup:

  e_second_a.py — layer pick, Custom preset, M1/M2, Existing Pattern
                  (shots 00_layer, 01_custom, 02_existing)
  e_second_b.py — replays the above, then Second Alignment Pattern (with
                  the Ny fix), Overlayed, Time Calculator
                  (shots 00_sap_warn, 01_sap_fixed, 02_overlay, 03_time_in,
                  04_result)

Run both; guide/docs/ebl/second-exposure.md's screenshots come from their
combined output. This file is kept only so the name doesn't silently
disappear from the scenarios directory (repo files can't be deleted).
"""

ENTRY = "tools/process/ebeam/calculator.py"


def run(page, shot):
    raise RuntimeError(
        "e_second.py is superseded — run e_second_a.py and e_second_b.py "
        "instead (see this file's module docstring).")
