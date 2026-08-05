# Job number sheet

Every number from [dose test](dose-test.md), [first exposure](first-exposure.md)
and [second exposure](second-exposure.md), organised the way the JEOL ELS-7000
software asks for them: **Job 1**, then **Job 2**, then **Job 3**. Pick the
section for the exposure you're running, each is self-contained. **Default**
means leave the calculator's stock value; **measured** means you supply it
from what you find at the machine; **computed** means the calculator worked
it out, read it off, don't type it in by hand anywhere but the JEOL field
named.

## Dose test

Source: [dose-test.md](dose-test.md), `dose_test.gds`, layer `L1/D1`.

**Job 1**

| Field | Value | Source |
|---|---|---|
| Chip Origin x (mm) | 10.000 | default |
| Chip Origin y (mm) | 10.000 | default |
| Cel Origin x (mm) | 8.610 | measured, positioned so the mask lands inside the write-field tile |
| Cel Origin y (mm) | 9.710 | measured, same |

**Job 2**

| Field | Value | Source |
|---|---|---|
| Chip Size (μm) | 600 | default |
| Dotmap | 60000 | default |
| Resolution (read-only) | 10 nm | computed |

**Job 3**

| Field | Value | Source |
|---|---|---|
| Increment dx (mm) | 0.600 | default (= chip size) |
| Increment dy (mm) | 0.600 | default (= chip size) |
| Grid Count Nx | 5 | default |
| Grid Count Ny | 5 | default |
| Initial Shift x (mm) | 98.800 | computed |
| Initial Shift y (mm) | 109.200 | computed |

**Dose / stage timing** (not a numbered JEOL job field, but set on the writer before running)

| Field | Value | Source |
|---|---|---|
| Initial dose (μs / dot) | 2.000 | default |
| Incremental dose (μs / grid) | 0.200 | default |
| Stage movement time (s / grid) | 15.00 | default |
| Estimated Time | 00:35:35.000 | computed, planning only |

## First exposure

Source: [first-exposure.md](first-exposure.md), `first_exposure.gds`, layer
`L1/D2` (repeat for `L1/D1` and `L1/D3` with the same corners and Cel Origin).

**Job 1**

| Field | Value | Source |
|---|---|---|
| Chip Origin x (mm) | 10.000 | default |
| Chip Origin y (mm) | 10.000 | default |
| Bottom Left (BL) corner x (mm) | 105.000 | measured, chip corner found under the beam |
| Bottom Left (BL) corner y (mm) | 115.000 | measured |
| Top Right (TR) corner x (mm) | 107.000 | measured, alignment mark found under the beam |
| Top Right (TR) corner y (mm) | 117.000 | measured |
| Cel Origin x (mm) | 9.700 | default |
| Cel Origin y (mm) | 9.700 | default |
| Grid Count Nx | 4 | computed, auto-fit to the mask layer |
| Grid Count Ny | 2 | computed, auto-fit to the mask layer |

Top Left and Bottom Right corners are not typed anywhere, they're computed
from BL and TR and shown disabled on the Chip Position plot.

**Job 2**

| Field | Value | Source |
|---|---|---|
| Chip Size (μm) | 600 | default |
| Dotmap | 60000 | default |
| Resolution (read-only) | 10 nm | computed |

**Job 3**

| Field | Value | Source |
|---|---|---|
| Shift x (mm) | 95.100 | computed, centers the grid on the chip |
| Shift y (mm) | 105.700 | computed |

**Dose / stage timing**

| Field | Value | Source |
|---|---|---|
| Dose time (μs / dot) | 2.000 | default |
| Stage movement time (s / grid) | 15.00 | default |
| Estimated Time | 00:00:35.400 | computed, planning only |

## Second exposure

Source: [second-exposure.md](second-exposure.md), `second_exposure.gds`,
layer `L2/D1` (repeat for `L2/D2` with the same corners, marks and Cel
Origin).

**Job 1**

| Field | Value | Source |
|---|---|---|
| Chip Origin x (mm) | 10.000 | default |
| Chip Origin y (mm) | 10.000 | default |
| Bottom Left (BL) / Top Right (TR) corners | same as first exposure | measured, same physical chip, hasn't moved |
| Cel Origin x (mm) | 9.700 | default |
| Cel Origin y (mm) | 9.700 | default |
| Grid Count Nx | 4 | computed, auto-fit to the mask layer |
| Grid Count Ny | 4 | **fixed by hand**, auto-fit gave 2, too short to reach the marks; bumped to 4 |
| Reg-2 Mark, Mark 1 (M1) x (mm) | 9.8000 | computed, mark design position + Cel Origin |
| Reg-2 Mark, Mark 1 (M1) y (mm) | 9.8000 | computed |
| Reg-2 Mark, Mark 2 (M2) x (mm) | 11.6000 | computed |
| Reg-2 Mark, Mark 2 (M2) y (mm) | 11.6000 | computed |

The Reg-2 Mark values depend on the mark position you measured under the
SEM (Target x/y in the calculator's Existing Pattern step), recompute them
there if you're registering to a different chip.

**Job 2**

| Field | Value | Source |
|---|---|---|
| Chip Size (μm) | 600 | default |
| Dotmap | 60000 | default |
| Resolution (read-only) | 10 nm | computed |

**Job 3**

| Field | Value | Source |
|---|---|---|
| Shift x (mm) | 0.0010 | computed |
| Shift y (mm) | −0.0005 | computed |

**Dose / stage timing**

| Field | Value | Source |
|---|---|---|
| Dose time (μs / dot) | 2.000 | default |
| Stage movement time (s / grid) | 15.00 | default |
| Estimated Time | 00:03:48.000 | computed, **excludes** the 1, 2 h typically needed to find the alignment marks |
