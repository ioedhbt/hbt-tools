"""tuning.md — upper half of the Rbc sweep (360, 430, 500 kΩ).

Each slider step reruns the whole app (~4 s at 1001 frequency points), and
setup eats 25 s, so one call fits five frames. This companion starts at the
Max end and walks left, writing frames 07, 06, 05 so the two runs interleave
into one ascending sequence.

Original docstring follows.

tuning.md — Visual Tuning frames: Rbc swept 10 kΩ to 500 kΩ.

Starts from the extracted model rather than the simulator defaults: run
`guide/_harness/seed_fit_cache.py` first, which writes the corrected Cheng T
parameter set into the per-(DUT, model) fit cache. SSMModelTemplate restores it
on first render, so the preview opens on a fitted device instead of a 374%
residual.

Frames land in <outdir>/frame_NN.png; guide/_harness/build_gif.py assembles
them.
"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf/raw")
DUT = "deemb_preext_vce3.5_ib200u.s2p"

PARAM = "Rbc"
PMIN, PSTEP, PMAX = "10", "70", "500"      # kΩ, 8 stops
FRAMES = 8


def hide_badge(page):
    page.evaluate(
        "() => {let s=document.getElementById('_hb');"
        "if(!s){s=document.createElement('style');s.id='_hb';"
        "document.head.appendChild(s);}"
        "s.textContent='div.st-key-lang_toggle{display:none !important;}';}"
    )


def run(page, shot):
    from session import settle

    page.set_default_timeout(8000)
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page, quiet_ms=300)
    hide_badge(page)

    page.get_by_text("Fit to a measured device", exact=False).first.click()
    settle(page, quiet_ms=300)
    page.locator('[data-testid="stMain"] input[type="file"]').first.set_input_files(
        str(DATA / DUT))
    settle(page, quiet_ms=600)
    hide_badge(page)

    vt = page.get_by_text("Visual Tuning", exact=False).first
    vt.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=250)
    vt.click()
    settle(page, quiet_ms=350)
    hide_badge(page)

    wrap = page.locator('div[data-testid="stMultiSelect"]').filter(
        has_text="Parameters to slide").first
    cb = wrap.get_by_role("combobox").first
    cb.click()
    settle(page, quiet_ms=300)
    cb.type(PARAM, delay=30)
    settle(page, quiet_ms=300)
    page.get_by_role("option", name=PARAM, exact=False).first.click()
    settle(page, quiet_ms=250)
    page.keyboard.press("Escape")
    settle(page, quiet_ms=300)

    for label, value in (("Min", PMIN), ("Step", PSTEP), ("Max", PMAX)):
        boxx = page.get_by_label(label, exact=False).first
        boxx.fill(value)
        boxx.press("Tab")
        settle(page, quiet_ms=250)
    slider = page.get_by_role("slider").first
    # focus(), not click(): a sibling react-aria track div swallows pointer
    # events at the thumb position. Home first so each ArrowRight is one step.
    slider.focus()
    page.keyboard.press("End")
    settle(page, quiet_ms=300)
    hide_badge(page)

    container = vt.locator("xpath=ancestor::div[@data-testid='stLayoutWrapper'][1]")
    vb = container.bounding_box()
    print("VT BOX", vb, flush=True)
    clip = {"x": 380, "y": max(0, vb["y"] - 10), "width": 1180, "height": 700}

    shot("frame_07", clip=clip)
    for i in (6, 5):
        page.keyboard.press("ArrowLeft")
        settle(page, quiet_ms=180)
        hide_badge(page)
        shot(f"frame_{i:02d}", clip=clip)
    print("DONE", flush=True)
