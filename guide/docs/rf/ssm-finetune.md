# Model extraction: fine tune and check

The extraction gives you a model. This page is where you decide whether to
keep it.

## Where you are on the page

The Extraction page runs top to bottom in five numbered sections. Everything
below is in **4 — Intrinsic Model** and **5 — Review**.

![The page's five numbered sections, from Pad Capacitance through Intrinsic Model](../assets/ssm/page_sections.png)

## Fine-tune by hand

Under **Measured vs Modeled S-Parameters**, open
**✏️ Fine-tune T-topology (Cheng 2022) intrinsic/extrinsic parameters**.

![Measured vs Modeled S-Parameters heading with the fine-tune expander below it](../assets/ssm/finetune_expander.png)

Type over any value and the model curve redraws immediately. Pad parameters
are greyed out here; they auto-sync from the pre-extraction override, so
change them back in [setup](ssm-setup.md) rather than fighting them here.

This is where the transit times get corrected. Cheng's analytic extraction
gives τB = 10.7654 ps and τC = −6.7695 ps for this device; overwrite them with
τB = 19.0480 ps and τC = 0.1500 ps from the
[transit-time fit](ssm-transit-time.md), which is measured across six bias
points rather than split out of one.

Fine-tuned values are saved to the fit cache and come back the next time you
open the same device with the same model.

## Residual and display scale

![S display scale row with a multiplier per S-parameter, and the total residual with its per-parameter breakdown underneath](../assets/ssm/finetune_scale_residual.png)

**S display scale** multiplies each trace before plotting, a way to pull S12
up to a readable size next to S21. It is display only and does **not** change
the residual.

**Total residual** is the fit error over the whole band, with a breakdown per
S-parameter. For this device: 8.66% total, from S11 4.63%, S12 12.30%,
S21 10.12%, S22 7.58%.

Rules of thumb:

- Under ~5% total is a good analytic extraction.
- S12 is almost always the worst of the four; it is the smallest signal and
  the most affected by any leftover pad capacitance.
- One parameter far above the others points at a specific element, not at the
  model. S22 high means the collector side (Rc, Cbc); S11 high means the base
  side.

## Smith and Bode

![Smith chart with measured and modelled S11, S12, S21, S22, next to the fT/fmax gain plot](../assets/ssm/finetune_smith_bode.png)

Measured traces are solid with markers, modelled traces are dashed. Read them
together: the Smith chart shows where in the band the model departs, the gain
plot shows what that costs you in f_T and f_max.

**📊 Smith Chart (Matplotlib)** in Section 5 renders the same data as a
publication figure, without the Plotly toolbar.

## Export

![Export row: xlsx, modeled S2P and copy under the Smith chart, xlsx and copy under the gain plot](../assets/ssm/review_residual.png)

- **xlsx**: the plotted data as a spreadsheet.
- **modeled S2P**: the fitted model as a Touchstone file, ready for a
  simulator.
- **copy**: click this to copy the data to your clipboard for pasting
  straight into Origin.

## τ_total and fmax

![Calculated tau_total and fmax, Topology illustration and Smith Chart expanders](../assets/ssm/review_expanders.png)

**🔢 Calculated τ_total and fmax** reports the two figures of merit computed
from the fitted model rather than measured off the curve:

- **τ_total** = 1/(2πf_T), the total emitter-to-collector delay. Compare it
  against the intercept from the [transit-time fit](ssm-transit-time.md). If
  they disagree badly, one of the two is being driven by a bad element.
- **f_max**, from the fitted model. The extrapolated value on the gain plot is
  the measured counterpart; they should agree within a few percent.

## Hand the model on

![Section 5 Review with Complete Parameter Summary, Fit cache, and the Send Cheng's T to Simulation & Fitting button](../assets/ssm/review_handoff.png)

1. **📋 Complete Parameter Summary** lists every extracted and fine-tuned
   value in one table. Copy it out here.
2. **💾 Fit cache** shows what has been persisted for this device and model.
3. **→ Send Cheng's T to Simulation & Fitting** carries the device, the
   topology and every current parameter value across, so you can keep tuning
   without re-uploading anything.

---

Next: [Simulation & Fitting](sim-fit.md).
