# Dose-time test

Expose the same pattern at a range of doses on one chip, then read the
developed linewidths under the SEM to pick a working dose before committing
a real device to it. This page walks the **Dose Time Testing** mode with the
demo file `dose_test.gds`, a 5×5 array of test blocks, 40 μm pitch, each
block holding an 8×8 μm pad and three lines (0.5, 1.0, 2.0 μm wide).

Read [EBL calculator basics](index.md) first if you haven't opened this page
before; this walkthrough picks up after a mask is loaded.

## 1. Load the mask and pick a layer

1. Upload `dose_test.gds`, then set **Layer to expose** to `L1/D1` (25
   polys), the block pads.

![GDS Mask section: dose_test.gds loaded, layer L1/D1 selected](../assets/ebl/dose_layer.png)

This demo file keeps pads (datatype 1) and lines (datatype 2) on separate
datatypes, so you expose one at a time. The walkthrough below uses the pads;
the lines work the same way with `L1/D2` selected instead.

## 2. Position the pattern in the write field

2. Pick **Dose Time Testing** under **Workflow**. Type the **Cel Origin (mm)
   in job1** x and y here. This is the physical stage position that the
   mask's GDS `(0,0)` lands on.

![Cel Origin x/y fields, typed to 8.610 / 9.710](../assets/ebl/dose_cel.png)

The block array in this file sits at GDS coordinates x = 1300, 1480 μm, y =
200, 380 μm (far from the file's own origin; it shares a coordinate system
with the alignment marks used in [second exposure](second-exposure.md)).
Leaving Cel Origin at its auto-computed default would place the array
outside the single write-field tile entirely, so it never gets exposed. The
values above (8.610, 9.710) shift the array to sit centered in the tile
instead, take that as the pattern to follow for your own mask: read the
pattern's bounding box off the GDS Mask viewer, then set Cel Origin so it
lands inside the tile.

## 3. Leave the grid geometry at its defaults

3. **Increment (mm) in job3** (dx, dy) and **Grid Count in job3** (Nx, Ny)
   don't need to change, dx/dy auto-track the chip size so tiles pack
   edge-to-edge, and the default 5×5 count matches this demo's dose steps.
   **Initial Shift (mm) in job3** is computed automatically to center the
   array on the chip; leave it alone too.

![Single Grid and Chip Position with Grids plots, pattern centered in both](../assets/ebl/dose_grids.png)

The left plot is one tile at the Cel Origin you just set, with no
replication, check the mask lands inside the orange box here first. The
right plot is the full 5×5 array over the chip, after the auto-centering
shift.

## 4. Set the dose ramp

4. Under **Time Calculator**, leave **Initial dose (μs / dot)**,
   **Incremental dose (μs / grid)** and **Stage movement time (s / grid)**
   at their defaults (2.000, 0.200, 15.00), then click **Calculate time**.

![Time Calculator dose-ramp inputs before Calculate](../assets/ebl/dose_time_in.png)

Each of the 25 tiles gets its own dose: tile *k* (in grid iteration order,
x outer / y inner) gets `2.000 + k × 0.200` μs/dot. Tile 0 is the lowest
dose, tile 24 the highest, that's the ramp a dose test needs.

## 5. Read the exposure time per dose step

![Breakdown: dose ramp range and Estimated Time](../assets/ebl/dose_result.png)

The **Dose ramp** line is the one to check: `2.000 μs + 0.200 μs/grid → active
range 2.000, 6.800 μs`, the lowest and highest per-tile dose actually
used, in iteration order across the 25 tiles. **Filled resolution boxes**
(400,000,000 here) is just the raw 10 nm-pixel count across all 25 tiles, large because the resolution is fine, not something to act on. **Estimated
Time** at the bottom (`00:35:35.000`) is what goes on the job sheet, see
[job number sheet](job-sheet.md) for where each of these numbers lands in
the JEOL system.
