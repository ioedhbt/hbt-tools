# Model extraction: transit time

τ_total against 1/I gives you τ_B + τ_C from the intercept and C_JE from the
slope. It needs several bias points, so load the whole sweep before you start.

Open **📊 Interactive Parameter Extraction**, then
**Cje / τB+τC / τCC / τE from 1/(2πf_T) vs 1/I_C fit**.

The fit plots τ_total against 1/I_C: the intercept is τ_B + τ_C and the
slope carries C_JE.

## Use I_C from measurement

Type the collector current I_C (mA) into the column.

![Transit-time table with fT and tau_total per file, and the IC (mA) column boxed](../assets/ssm/transit_ic_table.png)

## Read the fit

![tau_total versus 1/I_C with the fitted line](../assets/ssm/transit_tau_fit.png)

- **Intercept**: the part of the measured delay that does not shrink with
  current.
- **Slope**: from which C_JE follows.

The intercept is not yet τ_B + τ_C. The collector-charging term
(R_C + R_EE + r_E)·C_BC still has to come off, which is what the inputs
below the plot are for.

## Feed it the real access resistances

![Inputs row with Re-REE, Rc and Cbc boxed, and the resulting slope, intercept, Cje and tauB+tauC](../assets/ssm/transit_inputs_outputs.png)

**Inputs (defaults from this file's extraction)** starts at
Re — REE = `0.0000`, Rc = `0.0000`. Those defaults are wrong unless the
[access-resistance step](ssm-access-r.md) has already run. Type in the three
boxed values:

- **Re — REE (Ω)**, from the Z-parameter fit
- **Rc (Ω)**, from Cold-HBT
- **Cbc total (fF)**, from the T-model extraction

Leave η at 1.000 and T at 300.0 unless you have a Gummel fit that says
otherwise.

The answer moves a long way when you do: everything left in the intercept
was collector charging.

### Splitting τ_B and τ_C

The **Split τ_B / τ_C** row divides the total using an assumed average
collector velocity. With the InP defaults, W_C = 120 nm and
v_c = 4.000e+7 cm/s, the row gives you τ_C and τ_B.

**Use these two in place of Cheng's τB and τC.** The analytic extraction on
the [intrinsic page](ssm-intrinsic.md) can return a negative τC, and a
negative collector transit time is not physical. The transit-time fit uses
the whole bias sweep and the measured f_T, so it is the number to trust.

The per-file table underneath shows where each file lands once the corrected
inputs are in:

![Per-file derived delays table with IC, rE, tauCC, tauE and measured tau_total per file](../assets/ssm/transit_perfile.png)

These extracted values are a reference readout. They do **not** feed back into
the model fit on their own, so type them into the fine-tune panel if you want
them in the shipped model.

## Which current to use

The page follows Cheng's default of I_C, but use I_E. The delay being
fitted is the total emitter-to-collector transit, and at the low-current
end of an InP HBT sweep I_B is a few percent of I_C — enough to bend the
very points that set the intercept.

Whichever you pick, use it for every row. A table with I_C in some rows and
I_E in others fits a line through two different quantities.

## τ_total across files

The multi-file section fits one τ_total reference across every ticked file at
once, which is what the numbers in the table above come from. Tick a subset
to see how much any single bias point is carrying the fit. If the intercept
moves a lot when you drop one point, the sweep is too short.

---

Next: [fine tune and check](ssm-finetune.md).
