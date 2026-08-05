"""tuning.md — Visual Tuning frames: alpha0 swept 0.95 to 0.99.

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

# The picker lists parameters by display name, not by key: alpha0 shows as α₀.
PARAM = "α₀"
PMIN, PSTEP, PMAX = "0.95", "0.01", "0.99"   # 5 stops
FRAMES = 5


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
    cb.type(PARAM, delay=40)
    settle(page, quiet_ms=300)
    opts = page.get_by_role("option")
    print("OPTS", [opts.nth(i).inner_text() for i in range(opts.count())], flush=True)
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
    page.keyboard.press("Home")
    settle(page, quiet_ms=300)
    hide_badge(page)

    container = vt.locator("xpath=ancestor::div[@data-testid='stLayoutWrapper'][1]")
    vb = container.bounding_box()
    print("VT BOX", vb, flush=True)
    clip = {"x": 380, "y": max(0, vb["y"] - 10), "width": 1180, "height": 700}

    shot("frame_00", clip=clip)
    for i in range(1, FRAMES):
        page.keyboard.press("ArrowRight")
        settle(page, quiet_ms=180)
        hide_badge(page)
        shot(f"frame_{i:02d}", clip=clip)
    print("DONE", flush=True)
