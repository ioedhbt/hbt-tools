# TLM analysis

Transfer Length Method fit, contact and sheet resistance from resistance
measured at four fixed contact spacings. Same page as the B1500A viewer,
page selector at the top set to **TLM Analysis**.

**No TLM sweep ships with this repo's demo data.** Neither the four curated
files in `guide/_data/dc/` nor the raw `dc_data/` export folder contain a
`Tl…`-prefixed measurement, every screenshot on this page is the tool's
blank default state, not a real device. Numbers below are formulas from the
code, not a worked example.

## Data this tool wants

Unlike the B1500A viewer, this tab takes no file upload. Type in one pad
width and up to four resistance readings by hand:

1. **Pad width Z (µm)**, one number, defaults to 80.
2. **R @ 4 / 8 / 16 / 32 µm (Ω)**, resistance measured at each of the four
   fixed contact spacings. Leave any of the four blank to exclude it from
   the fit.

   ![TLM Analysis — blank state](../assets/dc/tlm_blank.png)

If you batch-converted a TLM sweep through
[Measurement Data Multi-Process](multi-process.md#other-batch-modes)
first, its **TLM Resistance Avg** mode reads a `TLM batch output.xlsx` and
averages the resistance column per sheet, those per-spacing averages are
what you'd type into the four fields here.

## The fit

With two or more of the four R values filled in, the page fits a line
R = slope·spacing + intercept through them and reports:

- **Contact resistance, Rc**: half the fit's intercept.
- **Sheet resistance, Rsh**: slope × pad width Z.
- **Transfer length, LT**: Rc / slope.
- **Specific contact resistivity, ρc**: Rc · LT · Z (unit-converted to
  Ω·cm²).
- **Goodness, R²**: squared correlation coefficient of the four points
  against the fit.

A plot of measured points plus the fitted line (spacing 0, 40 µm) appears
below the metrics once the fit runs.
