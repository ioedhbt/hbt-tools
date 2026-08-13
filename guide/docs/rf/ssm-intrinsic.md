# Model extraction: the intrinsic model

With the pads, leads and access resistances settled, the remaining elements
are peeled off one at a time. Each step has its own equation, its own plot,
and a value you can override.

## Pick the topology

Open **Model Selection** and tick **T-topology (Cheng 2022)**.

![Model Selection with T-topology (Cheng 2022) ticked](../assets/ssm/intrinsic_model_select.png)

The π topology is the same extraction against a different equivalent
circuit. Pick one and stay with it; mixing extracted values between the two
is meaningless.

## Work through the steps

Open **📊 Interactive Parameter Extraction**. Every step follows the same
pattern:

![Step 2 Cbex extraction: equation, frequency range slider, plateau plot with the current value dashed in red, the value box, and default/mean/median/low-f/high-f chips](../assets/ssm/intrinsic_cbex.png)

1. The equation the step evaluates, so you can see what it depends on.
2. **Frequency range**, drag it onto the flat part of the curve.
3. The plot, with the value currently in use dashed in red. A good extraction
   is flat over a decade or so; a curve that never settles means an upstream
   value is wrong.
4. The value box, and chips that fill it from a statistic.

Steps that depend on an earlier value say so under the heading
("Depend on: Cex, Cbc, Rbi"), so when a late step looks wrong, go back to
what it depends on rather than fighting the number in front of you.

Some steps add a sweep tool on the right: **Cbex sweep — minimise
std(Cbcx)** searches a Min/Step/Max range for the Cbex that makes Cbcx
flattest. Set the range, click **Calculate**, and it fills the box for you.

![Step 3 Cbcx extraction](../assets/ssm/intrinsic_cbcx.png)

## Cbc: use the median

Cbc defaults to the low-frequency value. **Click the `median` chip to use the
median instead.**

![Cbc value box with the median chip circled and the caption "Click this button to use the median"](../assets/ssm/intrinsic_cbc_median.png)

The low-frequency estimate is taken where the measurement is noisiest. The
median ignores the ringing at both ends of the band and is the more
repeatable choice across a bias sweep. Set it here before you move on; every
later step that depends on Cbc inherits whatever is in the box.

## What you should have

With Re, Rb and Rc from [the access-resistance step](ssm-access-r.md) and Cbc
on its median, the T model gives:

- Cbex, Cbcx
- Rbi, Rbe, Cbe
- Rbc, Cbc
- alpha0
- tauB, tauC

If τC comes out negative, no real collector does that: it is an artefact of
the single analytic extraction, and why the
[transit-time fit](ssm-transit-time.md) is the better source for τB and τC.
Replace both from there before you ship the model.

!!! tip
    The `cold = …` chip fills the box from the cold measurement instead.
    That is the zero-bias junction capacitance, not the biased one, so treat it
    as a sanity check rather than a value to ship.

---

Next: [transit time](ssm-transit-time.md).
