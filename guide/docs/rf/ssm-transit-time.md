# Model extraction: transit time

τ_total against 1/I gives you τ_B + τ_C from the intercept and C_JE from the
slope. It needs several bias points, so load the whole sweep before you start.

Open **📊 Interactive Parameter Extraction**, then
**Cje / τB+τC / τCC / τE from 1/(2πf_T) vs 1/I_C fit**.

The relation being fitted is

$$\frac{1}{2\pi f_T} = \tau_B + \tau_C + \frac{\eta kT}{qI_C}C_{JE}
  + \left(R_C + R_{EE} + \frac{\eta kT}{qI_C}\right)C_{BC}$$

so the intercept is τ_B + τ_C and the slope carries C_JE.

## Use I_E = I_C + I_B

The column is labelled **IC (mA)** and Cheng's formulation uses the collector
current. Type the **emitter** current into it instead: I_E = I_C + I_B.

![Transit-time table with fT and tau_total per file, and the IC (mA) column boxed](../assets/ssm/transit_ic_table.png)

1. Overwrite each row's **IC (mA)** box with I_C + I_B for that bias point.

For the demo device:

| File | I_C | I_B | type in | f_T | τ_total |
|---|---|---|---|---|---|
| `ib80u` | 1.600 mA | 80 µA | **1.680** | 1.607 GHz | 99.03 ps |
| `ib120u` | 2.928 mA | 120 µA | **3.048** | 1.869 GHz | 85.17 ps |
| `ib160u` | 4.418 mA | 160 µA | **4.578** | 2.074 GHz | 76.74 ps |
| `ib200u` | 6.075 mA | 200 µA | **6.275** | 2.268 GHz | 70.17 ps |
| `ib240u` | 7.859 mA | 240 µA | **8.099** | 2.330 GHz | 68.31 ps |
| `ib280u` | 8.018 mA | 280 µA | **8.298** | 2.455 GHz | 64.84 ps |

Untick any file you do not want in the fit.

## Read the fit

![tau_total versus 1/I_C with the fitted line, slope 69.05 ps mA and intercept 59.59 ps](../assets/ssm/transit_tau_fit.png)

- **Intercept**: 59.5923 ps, the part of the measured delay that does not
  shrink with current.
- **Slope**: 69.05 ps·mA, from which C_JE follows.

The intercept is not yet τ_B + τ_C. The collector-charging term
(R_C + R_EE + r_E)·C_BC still has to come off, which is what the inputs below
the plot are for.

## Feed it the real access resistances

![Inputs row with Re-REE, Rc and Cbc boxed, and the resulting slope, intercept, Cje and tauB+tauC](../assets/ssm/transit_inputs_outputs.png)

**Inputs (defaults from this file's extraction)** starts at
Re — REE = `0.0000`, Rc = `0.0000`. Those defaults are wrong unless the
[access-resistance step](ssm-access-r.md) has already run. Type in the three
boxed values:

| Box | Value | From |
|---|---|---|
| **Re — REE (Ω)** | 64.3966 | Z-parameter fit |
| **Rc (Ω)** | 19.0178 | Cold-HBT |
| **Cbc total (fF)** | 484.2610 | T-model extraction |

Leave η at 1.000 and T at 300.0 unless you have a Gummel fit that says
otherwise.

The answer moves a long way when you do: **τ_B + τ_C drops from 59.59 ps to
19.1980 ps.** Everything left in the intercept was collector charging.

### Splitting τ_B and τ_C

The **Split τ_B / τ_C** row divides the total using an assumed average
collector velocity. With the InP defaults, W_C = 120 nm and
v_c = 4.000e+7 cm/s:

- **τ_C = 0.1500 ps**
- **τ_B = 19.0480 ps**

**Use these two in place of Cheng's τB and τC.** The analytic extraction on
the [intrinsic page](ssm-intrinsic.md) returns τB = 10.7654 ps and
τC = −6.7695 ps for this device, and a negative collector transit time is not
a physical answer. The transit-time fit uses six bias points and the measured
f_T, so it is the number to trust.

The per-file table underneath shows where each file lands once the corrected
inputs are in:

![Per-file derived delays table with IC, rE, tauCC, tauE and measured tau_total per file](../assets/ssm/transit_perfile.png)

These extracted values are a reference readout. They do **not** feed back into
the model fit on their own, so type them into the fine-tune panel if you want
them in the shipped model.

## Which current to use

Cheng's default is I_C, and the page is built around it. Using I_E instead
shifts every point left by I_B/(I_C·I_E), which mostly steepens the low-current
end of the line.

Use I_E. The delay being fitted is the total emitter-to-collector transit,
and at the low-current end of an InP HBT sweep I_B is a few percent of I_C, which is enough to bend the very
points that set the intercept. Cheng's I_C form is
fine when β is large and you only care about the slope; it is the weaker
choice when you want τ_B + τ_C.

Whichever you pick, use it for every row. A table with I_C in some rows and
I_E in others fits a line through two different quantities.

## τ_total across files

The multi-file section fits one τ_total reference across every ticked file at
once, which is what the numbers in the table above come from. Tick a subset
to see how much any single bias point is carrying the fit. If the intercept
moves a lot when you drop one point, the sweep is too short.

---

Next: [fine tune and check](ssm-finetune.md).
