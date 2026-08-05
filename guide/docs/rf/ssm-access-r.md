# Model extraction: access resistance

Get Re, Rb and Rc before the intrinsic fit. Everything downstream leans on
these three numbers, so they are worth a few minutes.

Open **🍊 Access Resistance Extraction** on the Extraction page.

!!! warning "Order matters"
    Run the **Z-parameter method for Re first**, then Cold-HBT for Rb and Rc.
    Gao §5.5.2 removes Re from Z_cor before A, B, C and D are computed, so the
    cold extraction takes Re as an *input*. Open Cold-HBT first and its
    **Re (Ω)** box reads `0.0000`, which quietly gives you Rb and Rc fitted
    against no emitter resistance at all. On this device that is the difference
    between Rb = 29.29 Ω and Rb = 46.75 Ω.

## Step 1: Z-parameter method for Re

Re comes from the intercept of Re(Z₁₂) against 1/I_E across a bias sweep. Load
several bias points before you start. One point cannot fit a line.

1. The table lists every loaded file. Tick the ones to include and type the
   emitter current for each. Use **I_E = I_C + I_B**, not I_C.

    ![Z-parameter bias table with a checkbox, IE (mA) input and Re(Z12) readout per file](../assets/ssm/accessr_zparam_widget.png)

    For the demo device that is:

    | File | I_C | I_B | I_E = I_C + I_B |
    |---|---|---|---|
    | `ib80u` | 1.600 mA | 80 µA | **1.680 mA** |
    | `ib120u` | 2.928 mA | 120 µA | **3.048 mA** |
    | `ib160u` | 4.418 mA | 160 µA | **4.578 mA** |
    | `ib200u` | 6.075 mA | 200 µA | **6.275 mA** |
    | `ib240u` | 7.859 mA | 240 µA | **8.099 mA** |
    | `ib280u` | 8.018 mA | 280 µA | **8.298 mA** |

    The cold point (I_C = 77.68 µA, I_B = 0) is not part of this fit; it feeds
    the Cold-HBT extraction in step 2.

2. Read **Re** off the intercept.

    ![Re(Z12) versus 1/IE with the fitted line and the Re intercept marked](../assets/ssm/accessr_zparam_fit.png)

    **Re = 64.3966 Ω**, with Rbe = 311.84 Ω reported for the lowest-current
    file. Carry that number into step 2.

!!! tip
    The two highest-current points sit almost on top of each other
    (8.099 and 8.298 mA), so together they constrain the fit no better than a
    single point would. Spread the bias sweep out if the intercept looks
    unstable.

## Step 2: Cold-HBT for Rb and Rc

A "cold" measurement is the device with no bias applied. With both junctions
off, Z₁₁ and Z₂₂ reduce to the access network plus the junction capacitances,
so Rb and Rc fall out once Re is known.

1. Set **Cold-HBT source** to **From loaded files** and pick
   `deemb_preext_cold` from the dropdown. Upload it here instead if it is not
   among your loaded DUTs.

    ![Cold-HBT source selector set to From loaded files, with deemb_preext_cold chosen](../assets/ssm/accessr_cold_source.png)

2. Check the **Re (Ω)** box under Cold-HBT. It should already carry the
   64.3966 Ω from step 1. If it reads `0.0000`, the Z-parameter fit has not
   run: go back and do it.

3. Open **📊 Interactive Parameter Extraction**. Step 2 extracts **Cex**,
   which Rb and Rc both depend on, along with Cbc and Rbi.

    ![Step 2 Cex extraction: equation, frequency range slider, plateau plot and value box](../assets/ssm/accessr_cold_cex.png)

4. **Click the `median` chip to use the median**, 57.87 fF.

    ![Cex value box with the median chip circled, reading 57.87 fF](../assets/ssm/accessr_cold_cex_median.png)

    The default (188.7 fF) is the low-frequency value and the mean (101.4 fF)
    is dragged around by the same noise. The per-frequency array is flat once
    you are past the low end, so its median is the number to trust. Everything
    below inherits it.

5. Step 5 now gives **Rb = 46.751 Ω** and **Rc = 19.018 Ω**.

    ![Step 5 Rb and Rc panels with frequency range slider, plateau plot, value box and statistic chips](../assets/ssm/accessr_cold_rbrc.png)

    1. **Frequency range**, narrow it to the band where the curve is flat.
    2. The dashed red line is the value currently in use.
    3. The value box. Type over it to override.
    4. The chips below set the box from a statistic. Click one to use it.

    Both curves settle onto their dashed lines above ~2 GHz, and both medians
    (51.88 Ω and 20.13 Ω) sit close to the defaults. That agreement is the
    sign the extraction is healthy. When mean and median disagree wildly, drag
    the frequency range onto the flat part and take the value from there.

## What you should have

| | Value |
|---|---|
| Rb (=Rpb) | 46.7508 Ω |
| Rc (=Rpc) | 19.0178 Ω |
| Re (=Rpe) | 64.3966 Ω |
| Rbi (cold) | 158637.0901 Ω |
| Cbe (cold) | 119.9716 fF |
| Cbc (cold) | 1458.7848 fF |
| Cex | 57.8714 fF |

## Open-collector

The third method drives the base-emitter junction with the collector open.
Use it when you have that measurement and no usable cold sweep; it is a
substitute for the Cold-HBT step, not an addition to it.

---

Next: [the intrinsic model](ssm-intrinsic.md).
