# Open/short de-embedding

Peel pad and lead parasitics with Open and Short standards, using the same
modeled extraction (Gao 2015 §4.2) that powers the Batch De-embed tab.

This page works through it end to end with the demo files: `open.s2p`,
`short.s2p`, and one DUT (`vce3.5_ib280u.s2p`).

## Step 1 — feed the Open and Short standards

1. In the sidebar, turn on **② Device Dummy (Open-Short)**.
2. Drop the Open file under **Dev Open**.
3. Drop the Short file under **Dev Short**.

![Dev Open and Dev Short uploaders under the ② Device Dummy toggle](../assets/rf/dev_open_short.png)

These two files feed the Batch De-embed tab's calibration, upload your DUT
file(s) through the main **Upload DUT .s2p / .csv files** box as usual, then
open the **Batch De-embed** tab.

## What each step removes

- **Open** removes the shunt pad capacitances: Cpbe (base, emitter), Cpce
  (collector, emitter), Cpbc (base, collector), computed from
  `Im(Y_open)/ω` at each port pair.
- **Short** removes the series lead inductances and resistances: Lb, Lc, Le
  (and Rpb/Rpc/Rpe), computed from `Im(Z_short − Z_open)/ω` once the pad
  admittance is subtracted first.

The Batch De-embed tab shows both results as editable override fields,
seeded from the Open/Short calculation:

![Extracted Cpbe/Cpce/Cpbc (fF) and Lb/Lc/Le (pH) override fields](../assets/rf/extracted_values.png)

Edit any field to retune the de-embedding by hand;
click **↺ Reset to defaults** to go back to the computed values.

## Preview plots

Above the override fields, two plots let you sanity-check the calibration
before trusting it:

![Pad Capacitance vs Frequency and Lead Inductance vs Frequency](../assets/rf/cap_ind_preview.png)

Both should be **flat lines** across frequency, that's the signature of a
clean parasitic extraction. A trace that slopes or curves means the Open or
Short standard is not behaving like a pure capacitor/inductor over that
band (parasitic resonance, a bad probe touchdown, or the standard picking
up coupling it shouldn't), and the constant value the override field
defaults to becomes less trustworthy.

## Before / after on the device

Peeling the parasitics off moves fT because the pad capacitance and lead
inductance were loading the intrinsic device. On the demo DUT:

**Before**, raw, no de-embedding (Individual tab, `De-embedding: None`):

![Raw fT card: 2.123 GHz, 0dB Cross](../assets/rf/before_after_raw.png)

**After**, de-embedded (Batch De-embed tab, per-file result):

![De-embedded fT card: 2.455 GHz, 0dB Cross](../assets/rf/before_after_deembedded.png)

fT rises from 2.123 GHz to 2.455 GHz once the pad/lead parasitics are
removed, the intrinsic device is faster than the probe-level measurement
suggested. The Bode and Smith charts for the de-embedded device are on the
same per-file result:

![De-embedded Bode and Smith charts](../assets/rf/deembed_bode_smith.png)

!!! tip
    Round-trip accuracy: peeling these exact parasitics off `open.s2p` /
    `short.s2p`'s matching raw file reproduces the original pre-re-embedded
    device to within ~6e-10 in S-parameters (`guide/_data/rf/VERIFY.md`),     the extraction is exact, not approximate, when the parasitics are a
    clean shunt-C / series-L model.

## Modeled vs Measured source

Each of Open and Short can be de-embedded two ways, chosen independently:

- **Modeled**: builds analytical pad/lead matrices from the override
  values and subtracts them. This is what the numbers above use.
- **Measured**: subtracts the raw Open/Short S-parameters directly
  (Gao §4.2 open-short), with no model fit in between. Falls back to
  Modeled automatically if the corresponding dummy file is missing.
