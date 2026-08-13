# Measurement data multi-process

Batch-converts raw instrument exports into the workbooks the DC pages read,
and can hand the result straight to DC Analysis without a re-upload. Sidebar
**Data Processing → Measurement Data Multi-Process**.

The page opens on **B1500A Smart Batch Tool**, a sidebar radio at the top
switches to four other modes (see [Other batch modes](#other-batch-modes)
below).

## Upload

Drop your B1500A CSV exports on the uploader. Multiple files at once,
   350 MB per file, CSV only.

   ![Upload area](../assets/dc/mp_upload.png)

   The tool reads each filename's first two letters to pick a measurement
   type: `BC` → BC diode, `BE` → BE diode, `Fa` → Ic-Vc family, `Gu` →
   Gummel, `Tl` → TLM. Anything else is grouped under **Other** and
   skipped, so name your exports accordingly, or rename them before
   uploading.

## Result

Every recognized file is converted immediately, no button to press. Two
   ZIP downloads appear: one workbook per input file, and one workbook per
   measurement type (all matching files as separate sheets in one `.xlsx`).

   ![Batch result — five files uploaded, one skipped](../assets/dc/mp_result.png)

   Grouped-by-type filenames: `IcVc_Family.xlsx`, `BE_Diode.xlsx`,
   `BC_Diode.xlsx`, `Gummel.xlsx`, `TLM.xlsx`, `Other.xlsx`.

## Hand off to DC Analysis

Click **Analyze Data in DC Analysis** to send the Family/Gummel/BE/BC
   workbooks straight to the B1500A Viewer, no download, no re-upload.
   TLM and Other files are left out; the caption under the button says how
   many were skipped.

   ![Hand-off button](../assets/dc/mp_handoff.png)

   The B1500A Viewer opens with a "Received N file(s) from Multi-Process"
   banner and the received workbooks already listed in its file picker, see
   [B1500A Excel Viewer](b1500a.md).

## Other batch modes

The sidebar radio at the top of the page switches to four more tools that
share this page:

- **B1500A Column Selection & Batch**: upload one sample CSV, pick (or
  build) a column template, save it, then apply that template to a batch of
  other CSVs. Use this when your columns don't match one of the five
  built-in presets.
- **TLM Resistance Avg**: upload a `TLM batch output.xlsx` (the grouped
  download above, when the batch included `Tl…`-prefixed files), name the
  resistance column (default `Rsa`), and get the per-sheet average back as
  a table and a download.
- **E5270B citi File Tool**: accepts `.citi`, `.txt`, and plain `.csv`
  CITI-format exports, and sorts results into BE/BC/Gummel/Family same as
  the CSV tool.
- **HP4155A Data Processing Tool**: batch version of the SMU role
  assignment in [HP4155A Quick Plot](hp4155a.md): assign V/I column pairs
  to Collector/Base/Emitter/PD once, then convert a whole folder of raw
  HP4155A dumps to Family/Gummel/Diode workbooks in one pass.
