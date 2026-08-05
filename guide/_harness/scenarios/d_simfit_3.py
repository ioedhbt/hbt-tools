"""sim-fit.md — fit cache: edit a fine-tune value twice so the second rerun
shows the "cache from HH:MM" chip, then open the tau_total expander.

Shots:
  00_cache_chip  — pill row with the cache badge.
  01_tau_total   — Calculated tau_total and fmax expander, opened.
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

    ft = page.get_by_text("Fine-tune T-topology (Cheng 2022) intrinsic/extrinsic parameters",
                          exact=False).first
    ft.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    ft.click()
    settle(page)

    rbi = page.get_by_label("Rbi (Ω)", exact=False).first
    try:
        rbi.fill("12.5")
        rbi.press("Tab")
        settle(page, quiet_ms=600)
        rbi2 = page.get_by_label("Rbi (Ω)", exact=False).first
        rbi2.fill("12.7")
        rbi2.press("Tab")
        settle(page, quiet_ms=600)
    except Exception as e:
        print("EDIT ERR", e)

    hdr = page.get_by_text("🎯 Fit — T-topology (Cheng 2022)", exact=False).first
    hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    hb = hdr.bounding_box()
    print("HDR BOX", hb)
    shot("00_cache_chip", clip={"x": 370, "y": max(0, hb["y"] - 10),
                                "width": 1140, "height": 110})

    tau = page.get_by_text("Calculated", exact=False).first
    try:
        tau.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
        settle(page, quiet_ms=300)
        tau.click()
        settle(page, quiet_ms=500)
        tb = tau.locator("xpath=ancestor::div[@data-testid='stExpander'][1]").bounding_box()
        print("TAU BOX", tb)
        hide_badge(page)
        shot("01_tau_total", clip={"x": 370, "y": max(0, tb["y"] - 8),
                                   "width": 1140, "height": min(tb["height"] + 16, 700)})
    except Exception as e:
        print("TAU ERR", e)
    print("DONE")
