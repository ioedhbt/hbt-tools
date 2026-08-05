"""tuning.md — Visual Tuning GIF, part A: frames 00-05 (Rbc 10 -> ~190 kOhm).
Run with: bash guide/_harness/run.sh d_tuning_gif_rbc_a d_tuning_gif_rbc
Part B (d_tuning_gif_rbc_b.py) continues 06-10 into the same output dir.
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
    settle(page, quiet_ms=500)
    hide_badge(page)

    page.get_by_text("Fit to a measured device", exact=False).first.click()
    settle(page, quiet_ms=500)
    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files(str(DATA / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page, quiet_ms=700)

    vt = page.get_by_text("Visual Tuning", exact=False).first
    vt.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    vt.click()
    settle(page, quiet_ms=300)
    hide_badge(page)

    wrap = page.locator('div[data-testid="stMultiSelect"]').filter(has_text="Parameters to slide").first
    cb = wrap.get_by_role("combobox").first
    cb.click()
    cb.type("Rbc", delay=30)
    settle(page, quiet_ms=300)
    page.get_by_role("option", name="Rbc", exact=False).first.click()
    settle(page, quiet_ms=300)
    page.keyboard.press("Escape")
    settle(page, quiet_ms=300)

    minbox = page.get_by_label("Min", exact=False).first
    minbox.fill("10")
    minbox.press("Tab")
    settle(page, quiet_ms=200)
    stepbox = page.get_by_label("Step", exact=False).first
    stepbox.fill("45")
    stepbox.press("Tab")
    settle(page, quiet_ms=200)
    maxbox = page.get_by_label("Max", exact=False).first
    maxbox.fill("500")
    maxbox.press("Tab")
    settle(page, quiet_ms=300)
    hide_badge(page)

    slider = page.get_by_role("slider").first
    slider.focus()
    page.keyboard.press("Home")

    vt_container = vt.locator("xpath=ancestor::div[@data-testid='stLayoutWrapper'][1]")
    vb = vt_container.bounding_box()
    print("VT BOX", vb)
    clip = {"x": 370, "y": max(0, vb["y"] - 10), "width": 1550, "height": 760}

    shot("frame_00", clip=clip)
    for i in range(1, 6):
        page.keyboard.press("ArrowRight")
        shot(f"frame_{i:02d}", clip=clip)
    print("DONE")
