# Job number sheet

Every field you need from [dose test](dose-test.md), [first exposure](first-exposure.md)
and [second exposure](second-exposure.md), organised the way the JEOL ELS-7000
software asks for them: **Job 1**, then **Job 2**, then **Job 3**. Pick the
section for the exposure you're running, each is self-contained.

## Dose test

Run a dose test before exposing a real device. It draws the same pattern at several exposure times so you can pick the best one: enough to expose the pattern within tolerance, while keeping the yield acceptable.

**Job 1**

* `Chip origin x, y (mm)`: where the grid starts, NOT where the device is in the holder.
* `Cel origin x, y (mm)`: where the `.cel` file is positioned.

**Job 2**

* `Chip size (μm)`: The length of the sides of a grid, typically 600 μm. Test patterns usually fit in one grid; the grids are multiplied in job3.
* `Dotmap`: How many "pixels" will be drawn.

Resolution = Chip Size / Dotmap. Example: Chip size = 600 μm, Dotmap = 60000. Resolution = 10 nm.

**Job 3**

* `Increment dx, dy (mm)`: The distance between each grid, typically `0.6,0.6` mm
* `Grid Count Nx, Ny`: The number of grids in the x and y direction. Using `5,5` will create a 5 by 5 grid.
* `Initial Shift x, y (mm)`: The position shift, based on the position of your chips.
* `Initial dose`: The smallest dose (dwell time) you want to test. For example, 5 µs.
* `Dose increment`: The dose increment between grids. If there are 5 grids, initial dose is 5 µs, and dose increment is 0.2 µs, the EBL will expose at 5 µs, 5.2 µs, 5.4 µs, 5.6 µs, and 5.8 µs.

`modx, mody` are `0,0`, `focus shift` is `0`.

## First exposure

First exposure is a regular exposure without alignment. Make sure you have the desired dose time. Once you input the corners (or position) you found in SEM mode, the app automatically centers the mask pattern on your device.

**Job 1**

* `Chip origin x, y (mm)`: Where the grid starts, NOT where the device is in the holder.
* `Cel origin x, y (mm)`: Where the `.cel` file is positioned.
* `Grid Count Nx, Ny`: How many grids are required for all the patterns in your mask to fit.


**Job 2**

* `Chip size (μm)`: The length of the sides of a grid, typically 600 μm.
* `Dotmap`: How many "pixels" will be drawn.

Resolution = Chip Size / Dotmap. Example: Chip size = 600 μm, Dotmap = 60000. Resolution = 10 nm.

**Job 3**

* `Shift x, y` (mm): How much positional shift is required, to merge the positions from the setup in `job1` to the actual device position as detected by the SEM.

## Second exposure

Source: [second-exposure.md](second-exposure.md), `second_exposure.gds`,
layer `L2/D1` (repeat for `L2/D2` with the same corners, marks and Cel
Origin).

**Job 1**

* `Chip origin x, y (mm)`: Where the grid starts, NOT where the device is in the holder.
* `Cel origin x, y (mm)`: Where the `.cel` file is positioned.
* `Grid Count Nx, Ny`: How many grids are required for all the patterns in your mask to fit.
* `Reg-2 Mark, Mark 1 (M1) and Mark 2 (M2) x, y (mm)`: The mark positions in your original .gds file, added by `Cel origin x, y (mm)` value.


**Job 2**

* `Chip size (μm)`: The length of the sides of a grid, typically 600 μm.
* `Dotmap`: How many "pixels" will be drawn.

Resolution = Chip Size / Dotmap. Example: Chip size = 600 μm, Dotmap = 60000. Resolution = 10 nm.

**Job 3**

* `Shift x, y` (mm): How much positional shift is required, to merge the positions from the setup in `job1` to the actual device position as detected by the SEM.

