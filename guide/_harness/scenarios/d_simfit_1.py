"""sim-fit.md — model selector + default forward simulation (Cheng's T, no
measurement loaded).

Shots:
  00_model_bar      — the Model segmented control (all six topologies).
  01_fit_expander   — the "Fit to a measured device" expander, collapsed.
  02_forward_charts — Smith + fT/fmax panels for the default (all-zero)
                       forward sim.
"""
ENTRY = "IOED_Tool_Web.py"


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

    page.set_default_timeout(5000)
    hide_badge(page)
    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page)

    shot("00_model_bar", clip={"x": 370, "y": 200, "width": 680, "height": 60})

    exp = page.get_by_text("Fit to a measured device", exact=False).first \
        .locator("xpath=ancestor::div[@data-testid='stExpander'][1]")
    eb = exp.bounding_box()
    print("EXP BOX", eb)
    shot("01_fit_expander", clip={"x": 370, "y": max(0, eb["y"] - 8),
                                  "width": 680, "height": eb["height"] + 16})

    hdr = page.get_by_text("Smith Chart — T-topology (Cheng 2022)", exact=False).first
    hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    hb = hdr.bounding_box()
    print("SMITH HDR BOX", hb)
    shot("02_forward_charts", clip={"x": 370, "y": max(0, hb["y"] - 10),
                                    "width": 1500, "height": 700})
    print("DONE")
