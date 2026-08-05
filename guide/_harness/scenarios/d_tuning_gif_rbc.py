"""tuning.md — Visual Tuning GIF source frames: drag Rbc across 10-500 kOhm.

Frames land in <outdir>/frame_NN.png; guide/_harness/build_gif.py (run
separately, plain PIL, no playwright) assembles them.
"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/deembed_these/deembedded/4x10_left")


def hide_badge(page):
    page.evaluate(
        "() => {"
        "let s = document.getElementById('_hide_badge');"
        "if (!s) { s = document.createElement('style'); s.id = '_hide_badge';"
        "document.head.appendChild(s); }"
        "s.textContent = 'div.st-key-lang_toggle{display:none !important;}';"
        "}"
    )


def run(page, shot):
    from session import settle

    page.set_default_timeout(6000)
    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page)
    hide_badge(page)

    page.get_by_text("Fit to a measured device", exact=False).first.click()
    settle(page)
    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files(str(DATA / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page)

    vt = page.get_by_text("Visual Tuning", exact=False).first
    vt.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    vt.click()
    settle(page, quiet_ms=400)
    hide_badge(page)

    wrap = page.locator('div[data-testid="stMultiSelect"]').filter(has_text="Parameters to slide").first
    cb = wrap.get_by_role("combobox").first
    cb.click()
    settle(page, quiet_ms=400)
    cb.type("Rbc", delay=40)
    settle(page, quiet_ms=400)
    all_opts = page.get_by_role("option")
    for i in range(all_opts.count()):
        print("OPT", i, all_opts.nth(i).inner_text())
    opt = page.get_by_role("option", name="Rbc", exact=False).first
    print("OPT COUNT", page.get_by_role("option", name="Rbc", exact=False).count())
    opt.click()
    settle(page, quiet_ms=300)
    page.keyboard.press("Escape")
    settle(page, quiet_ms=400)

    minbox = page.get_by_label("Min", exact=False).first
    minbox.fill("10")
    minbox.press("Tab")
    settle(page, quiet_ms=300)
    stepbox = page.get_by_label("Step", exact=False).first
    stepbox.fill("40")
    stepbox.press("Tab")
    settle(page, quiet_ms=300)
    maxbox = page.get_by_label("Max", exact=False).first
    maxbox.fill("500")
    maxbox.press("Tab")
    settle(page, quiet_ms=400)
    hide_badge(page)

    slider = page.get_by_role("slider").first
    print("SLIDER BOX", slider.bounding_box())
    # Jump to the minimum end first (Home key) so every subsequent
    # ArrowRight advances by one `step`.  focus() (not click()) — a sibling
    # react-aria track div intercepts pointer events at the thumb's position.
    slider.focus()
    page.keyboard.press("Home")
    settle(page, quiet_ms=400)
    hide_badge(page)

    vt_container = vt.locator("xpath=ancestor::div[@data-testid='stLayoutWrapper'][1]")
    vb = vt_container.bounding_box()
    print("VT BOX", vb)
    clip = {"x": 370, "y": max(0, vb["y"] - 10), "width": 1550, "height": 760}

    shot("frame_00", clip=clip)
    for i in range(1, 12):
        page.keyboard.press("ArrowRight")
        settle(page, quiet_ms=350)
        hide_badge(page)
        shot(f"frame_{i:02d}", clip=clip)
    print("DONE")
