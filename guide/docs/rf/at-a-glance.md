# RF at a glance

Bulk-upload devices, de-embed, and read off fT/fmax in one pass.

## Upload

1. Click **Upload** to drop DUT `.s2p` or `.csv` files, one per bias point,
   as many as you like in one go.
2. Click **Clear uploads** to drop every uploaded file and start over.

![Upload dropzone and Clear uploads button](../assets/rf/upload_area.png)

Each file is cached on its content hash plus the current de-embedding and
chart-window settings, so re-running the page (a slider drag, a checkbox
toggle) does not re-parse files that haven't changed.

## The four tabs

![The Overlay, Individual, Summary and Batch De-embed tabs](../assets/rf/tabs_bar.png)

1. **Overlay**, every uploaded file's Bode and Plateau curves on one shared
   axis, for comparing devices at a glance.
2. **Individual**, one file at a time: Bode, Plateau and Smith charts, the
   fT/fmax metric cards, data table, and export.
3. **Summary**, one row per file with fT/fmax, sortable, plus Excel/ZIP
   export.
4. **Batch De-embed**, Open/Short de-embedding applied to every uploaded
   file against one calibration pair. Covered on its own page.

The Overlay tab's two charts stack vertically, Bode (|h21|², Mason U,
MAG/MSG vs frequency) above, Plateau (`f × gain`, the GBP form of the same
data) below. Every uploaded file gets its own set of traces on both, so
sweeps of many bias points overlay directly:

![Overlay tab: Bode Plot Overlay chart](../assets/rf/overlay_bode.png)

![Overlay tab: Plateau Plot Overlay chart](../assets/rf/overlay_plateau.png)

## fT/fmax extraction

Each metric card shows a value and the method that produced it.

![Metric cards: De-embedding, fT, fmax U, fmax MAG, K min, Ib](../assets/rf/metric_cards.png)

The app tries, in order:

1. **Crossing** (`0dB Cross`), a genuine 0 dB crossing: the gain trace
   stayed above 0 dB for at least 10 consecutive points before dropping
   through it. This is the value shown whenever it exists.
2. **Extrapolation** (`Extrap & Plat.`), no crossing was found inside the
   swept band, but the median gain is still positive, so the last few points
   are fit log-linearly and projected forward to their 0 dB crossing.
3. **Plateau**, reported alongside the extrapolated value as a sanity
   check: `f × |gain|` evaluated pointwise, which should sit close to the
   extrapolated fT/fmax for a well-behaved device.

`No Gain` / `No Data` means the trace never clears 0 dB in the swept band, normal for an unbiased or cold device (see the card above: `2.123 GHz`,
method `0dB Cross`).

## Sorting and jumping to a device

Rank devices by fT or fmax in the Summary tab, then open the one you want
directly in Individual, no re-selecting from a dropdown.

1. Click the **fT Cross** (or any) column header to sort by it. Click again
   to reverse the order.

    ![Summary table sorted by fT Cross, descending](../assets/rf/summary_sort.png)

2. Click the checkbox on a row to select it, then click the blue **Open "…"
   in the Individual tab →** button that appears.

    ![Selecting a row and the resulting jump-to-Individual button](../assets/rf/summary_select.png)

## Export

The Summary tab exports the whole table:

![Excel, ZIP (CSV) and copy buttons under the summary table](../assets/rf/summary_export.png)

1. Click **Excel** for a multi-sheet workbook, one `Summary` sheet plus one
   sheet per device.
2. Click **ZIP (CSV)** for the same data as plain CSV files.
3. Click **copy** to copy the summary table to your clipboard for pasting
   into Origin.

The Individual tab's Bode chart carries the same two buttons below it
(Plateau and Smith don't export):

![xlsx download and copy buttons under the Bode chart](../assets/rf/bode_export.png)

1. Click **xlsx** to download the plotted traces as a workbook.
2. Click **copy** to copy the same data to your clipboard for pasting
   straight into Origin, no download needed.

## Handing a device to Extraction or Simulation

From the Individual tab, send the active device on without re-uploading it.

![Send this device to an SSM page: SSM Extraction and Simulation & Fitting buttons](../assets/rf/handoff_buttons.png)

1. Click **→ SSM Extraction** to open the HBT SSM Extraction page with this
   device's S-parameters already loaded.
2. Click **→ Simulation & Fitting** to open the RF Simulator with the same
   device loaded for fitting.

If a 3-step or batch de-embed already ran on this file, a **S-parameters to
send** choice appears first, de-embedded (pads/leads removed, fit the
intrinsic device only) or raw (fit the parasitics too).
