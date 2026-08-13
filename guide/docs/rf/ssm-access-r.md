# Model extraction: access resistance

Get Re, Rb and Rc before the intrinsic fit. Everything downstream leans on
these three numbers, so they are worth a few minutes.

Open **🍊 Access Resistance Extraction** on the Extraction page.

!!! warning "Order matters"
    Run the **Z-parameter method for Re first**, then Cold-HBT for Rb and Rc.
    Cold-HBT takes Re as an *input*. Open Cold-HBT first and its **Re (Ω)**
    box reads `0.0000`, which quietly gives you Rb and Rc fitted against no
    emitter resistance at all.

## Step 1: Z-parameter method for Re

Re comes from the intercept of Re(Z₁₂) against 1/I_E across a bias sweep. Load
several bias points before you start. One point cannot fit a line.

1. The table lists every loaded file. Tick the ones to include and type the
   emitter current for each. Use **I_E = I_C + I_B**, not I_C.

    ![Z-parameter bias table with a checkbox, IE (mA) input and Re(Z12) readout per file](../assets/ssm/accessr_zparam_widget.png)

    The cold point (I_B = 0) is not part of this fit; it feeds the Cold-HBT
    extraction in step 2.

2. Read **Re** off the intercept.

    ![Re(Z12) versus 1/IE with the fitted line and the Re intercept marked](../assets/ssm/accessr_zparam_fit.png)

    Carry the Re into step 2.

!!! tip
    Bias points that sit almost on top of each other constrain the fit no
    better than a single point would. Spread the bias sweep out if the
    intercept looks unstable.

## Step 2: Cold-HBT for Rb and Rc

A "cold" measurement is the device with no bias applied; with both
junctions off, Rb and Rc fall out once Re is known.

1. Set **Cold-HBT source** to **From loaded files** and pick your cold
   (unbiased) measurement from the dropdown. Upload it here instead if it is
   not among your loaded DUTs.

    ![Cold-HBT source selector set to From loaded files, with the cold measurement chosen](../assets/ssm/accessr_cold_source.png)

2. Check the **Re (Ω)** box under Cold-HBT. It should already carry the
   Re from step 1. If it reads `0.0000`, the Z-parameter fit has not
   run: go back and do it.

3. Open **📊 Interactive Parameter Extraction**. Step 2 extracts Cex, Cbc, and Rbi which Rb and Rc both depend on. If you modify a value, everything below inherits it.
   
4. Step 5 now gives **Rb** and **Rc**.

    ![Step 5 Rb and Rc panels with frequency range slider, plateau plot, value box and statistic chips](../assets/ssm/accessr_cold_rbrc.png)

    1. **Frequency range**, narrow it to the band where the curve is flat.
    2. The dashed red line is the value currently in use.
    3. The value box. Type over it to override.
    4. The chips below set the box from a statistic. Click one to use it.

    Both curves should settle onto their dashed lines on the flat part of
    the band. When mean and median disagree wildly, drag the frequency
    range onto the flat part and take the value from there.

## What you should have

After both steps you should have:

- Rb (= Rpb)
- Rc (= Rpc)
- Re (= Rpe)
- Rbi (cold)
- Cbe (cold)
- Cbc (cold)
- Cex

## Open-collector

The third method drives the base-emitter junction with the collector open.
Use it when you have that measurement and no usable cold sweep; it is a
substitute for the Cold-HBT step, not an addition to it.

---

Next: [the intrinsic model](ssm-intrinsic.md).
