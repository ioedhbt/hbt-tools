# Second exposure

Register a new layer to marks already written on the chip in a prior
exposure, so it lands where it needs to relative to what's already there.
This page walks the **Second Alignment** mode with `second_exposure.gds`, layer 1 in this file is the *same* four alignment marks at the *same*
coordinates as [first exposure](first-exposure.md)'s `first_exposure.gds`;
layer 2 (interconnect bridges and airbridge posts) is the pattern actually
being written this time.

Read [EBL calculator basics](index.md) first if this is your first time on
this page. **Every field below keeps its default value except the positions
you're told to type**, everything else is auto-computed or left alone;
each screenshot says which.

## 1. Load the mask and pick the pattern layer

1. Upload `second_exposure.gds`, then set **Layer to expose** to `L2/D1` (6
   polys), the interconnect bridges. (The airbridge posts on `L2/D2` are a
   separate pass, same corners and marks.)

![GDS Mask section: second_exposure.gds loaded, layer L2/D1 selected](../assets/ebl/second_layer.png)

## 2. Set the mark design positions

2. Pick **Second Alignment** under **Workflow**, then **Custom** under
   **Cross-position preset**. This mask's marks don't match the built-in
   `HBT_RF_v3` preset. Leave **Input unit** at **mm**.

The four marks in this file sit at (0.100, 0.100), (1.900, 0.100), (0.100,
1.900) and (1.900, 1.900) mm, chip-relative, the bottom-left and top-right
ones are the two you'll register on:

3. Type **0.100** / **0.100** into **Mark M1 (mm)** x/y, the bottom-left
   mark's design position.
4. Type **1.900** / **1.900** into **Mark M2 (mm)** x/y, the top-right
   mark's design position.

![Cross-position preset = Custom, Mark M1 and M2 typed](../assets/ebl/second_custom.png)

These two numbers are where the marks sit in the mask design. The next step
is where you tell the tool where they actually landed on the physical chip.

## 3. Enter the mark position found on the chip

5. Under **Existing Pattern on Chip**, set **Existing pattern layer** to
   `L1/D0` (4 polys), the marks layer, read from the *same* uploaded file
   (its layer 1 is identical to `first_exposure.gds`'s). Leave **Move:** at
   its default, **M1**.

Say you put the chip under the SEM, found the M1 mark, and read its stage
position as (9.8010, 9.7995) mm:

6. Type **9.8010** into **Target x** and **9.7995** into **Target y**.

![Existing pattern layer = marks, Target x/y typed to the found position](../assets/ebl/second_existing.png)

The tool computes the shift between where M1 sits in the design (0.100,
0.100) and where you just said it actually is (9.8010, 9.7995), that shift
carries every mark, and the whole existing pattern, into the same physical
frame the new layer needs to land in.

## 4. Fix the grid so both marks are inside it

**Grid Count in job1** (Nx, Ny) auto-fits to the *pattern's* bounding box,
not to where the marks are, so it can come up short of reaching them:

![Second Alignment Pattern: 4x2 auto-fit grid, M2 outside it, red warning](../assets/ebl/second_sap_warn.png)

The red banner, **selected alignment mark outside of exposure grids**, means exactly that: M2 sits outside the 4×2 grid the tool sized off the
bridges alone. Fix it by hand:

7. Increase **Ny** from 2 to **4**.

![Ny bumped to 4: banner gone, both marks inside the grid, registration mark position panel](../assets/ebl/second_sap_fixed.png)

**Cel Origin (mm) in job1** stays at its default (9.700, 9.700), leave it.
The **Registration mark position in job1** panel below now shows both marks
in job1 stage coordinates: Mark 1 (M1) at (9.8000, 9.8000), Mark 2 (M2) at
(11.6000, 11.6000), mark design position plus Cel Origin, read-only.

## 5. Read the Overlayed result

8. Scroll to **Overlayed**. **Mark 1** and **Mark 2** default to `M1` /
   `M2`, leave them. **Shift x (mm)** / **Shift y (mm)** are computed for
   you.

![Overlayed: Mark 1/2 pickers, auto Shift x/y for job3, final plot](../assets/ebl/second_overlay.png)

**Shift x/y** (0.0010, −0.0005 mm here) is the number that goes on the job
sheet for job3, it's what carries the *whole* second-exposure grid array
onto the position the marks were actually found at. It's small in this
walkthrough because the Target position in step 3 was close to
design + Cel Origin; on a real chip with more drift it will be larger.

## 6. Set the dose and read the time

9. Under **Time Calculator**, leave **Dose time (μs / dot)** and **Stage
   movement time (s / grid)** at their defaults (2.000, 15.00) and click
   **Calculate time**.

![Breakdown and Estimated Time after Calculate](../assets/ebl/second_result.png)

**Active grids: 4 / 4** here, every tile in the fixed 4×4 grid holds either
pattern or a mark. The caption under **Estimated Time** is worth reading
literally: finding the alignment marks under the SEM typically adds another
1, 2 hours on top of this number, and that time isn't in the estimate. See
the [job number sheet](job-sheet.md) for where every number from this page,
[first exposure](first-exposure.md) and [dose test](dose-test.md) lands in
the JEOL system, in order.
