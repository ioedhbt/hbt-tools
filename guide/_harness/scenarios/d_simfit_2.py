"""sim-fit.md — upload a measured device and inspect the fit view.

Uses the de-embedded device deemb_preext_vce3.5_ib200u.s2p (same bias point
as the ssm-setup.md walkthrough, now de-embedded) so the residual reflects an
intrinsic-only fit.

Shots:
  00_uploaded    — the "Fit to a measured device" expander after upload
                   (chip + download + clear buttons).
  01_fit_result  — residual metric row + Smith overlay + fT/fmax card.
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

    exp = page.get_by_text("Fit to a measured device", exact=False).first
    exp.click()
    settle(page)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files(str(DATA / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page)

    exp_box = page.get_by_text("Fit to a measured device", exact=False).first \
        .locator("xpath=ancestor::div[@data-testid='stExpander'][1]")
    eb = exp_box.bounding_box()
    print("EXP BOX", eb)
    shot("00_uploaded", clip={"x": 370, "y": max(0, eb["y"] - 8),
                              "width": 1140, "height": min(eb["height"] + 16, 260)})

    hdr = page.get_by_text("🎯 Fit — T-topology (Cheng 2022)", exact=False).first
    hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=400)
    hide_badge(page)
    hb = hdr.bounding_box()
    print("FIT HDR BOX", hb)
    shot("01_fit_result", clip={"x": 370, "y": max(0, hb["y"] - 10),
                                "width": 1500, "height": 760})
    print("DONE")
