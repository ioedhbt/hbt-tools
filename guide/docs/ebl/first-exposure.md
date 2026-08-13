# First exposure

Write the first pattern onto a bare chip: no marks, no registration to anything already there, the write field is just tiled with chip-size grids and centered on the chip. This page walks the **First Exposure** mode with `first_exposure.gds` (with patterns in the center, plus four alignment marks reserved for the [second exposure](second-exposure.md) that follows this one).

Read [EBL calculator basics](index.md) first if this is your first time on this page.

## 1. Set the chip corners you actually found

This input centers the pattern on your device. **Chip Position in the E-beam Holder** must match where the device really sits in the holder. Say you found the chip's bottom-left corner at stage position (104.000, 114.400) mm under the beam. Type that into the **Bottom Left (BL)** x/y fields.

Then locate the top-right corner of your device and read its stage position — (116.000, 126.400) mm in this example. Type that into the **Top Right (TR)** x/y fields.

![BL and TR corner fields typed; TL and BR computed and disabled](../assets/ebl/first_corners.png)

**Top Left (TL)** and **Bottom Right (BR)** are grayed out and computed for
you, with **Shape = Rectangular** and **Editable diagonal = BL / TR**, the other diagonal is derived automatically. 

## 2. Load the mask and pick a layer

Upload your `.gds` file, then set your **Layer to expose**.

![GDS Mask section: first_exposure.gds loaded, layer L1/D2 selected](../assets/ebl/first_layer.png)

## 3. Grid geometry setup

Pick **First Exposure** under **Workflow**. The app automatically computes **Cel Origin (mm) in job1**, **Grid Count in job1** (Nx, Ny) and **Shift (mm) in job3** from the mask's bounding box and the chip corners set in step 1.

![Mask layer size caption and the three auto-computed cards](../assets/ebl/first_inputs.png)

The app uses 0.600 mm grids at the default Cel Origin (9.700, 9.700) to cover it. Change the layer, the chip corners, or the chip size in **Left Computer Setup**, and the three cards recompute automatically.

## 4. Check the plot

![Chip (blue), 4×2 grid (orange) and mask over two tiles](../assets/ebl/first_plot.png)

The chip rectangle now reflects the corners from step 1, the grid is centered on it. You can modify as required.

## 5. Set the dose and read the time

Under **Time Calculator**, type in your **Dose time (μs / dot)** and **Stage movement time (s / grid)** and click **Calculate time**.

<!-- ![Time Calculator inputs before Calculate](../assets/ebl/first_time_in.png) -->

![Breakdown and Estimated Time after Calculate](../assets/ebl/first_result.png)

**Active grids** show only the grids with exposure patterns; the rest are skipped, so exposure and stage-movement time only count the tiles that actually have pattern in them. **Estimated Time** reflects how long the exposure will need to finish.

See the [job number sheet](job-sheet.md) for where every number from this page and [dose test](dose-test.md) lands in the JEOL
system, in order.

## Setup Instruction

Tip: Open **Setup Instruction** in **Left Computer Setup** section. It summarizes your steps and the numbers required in the left-side computer.

![Setup instruction summarizes your steps and the numbers required in the left-side computer](../assets/ebl/setup_instruction.png)

