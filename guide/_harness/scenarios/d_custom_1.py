"""custom-model.md — Build model: load Cheng T preset via "Modify", inspect
the schematic.

Shots:
  00_build_landed   — Build model tab, "Modify an existing model" expander open.
  01_loaded_schematic — live schematic after loading the Cheng T preset (all
                         sections revealed).
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

    page.set_default_timeout(6000)
    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page)
    page.get_by_text("Custom model", exact=False).first.click()
    settle(page)
    hide_badge(page)

    page.get_by_text("Build model", exact=True).first.click()
    settle(page)

    mod_hdr = page.get_by_text("Modify an existing model", exact=False).first
    mod_hdr.click()
    settle(page)
    hide_badge(page)
    mod_hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    mb = mod_hdr.bounding_box()
    print("MOD HDR BOX", mb)
    shot("00_build_landed", clip={"x": 370, "y": max(0, mb["y"] - 90),
                                  "width": 1500, "height": 460})

    sel = page.get_by_label("Built-in topology", exact=False).first
    try:
        print("SELECT BOX", sel.bounding_box())
    except Exception as e:
        print("SELECT ERR", e)
    # react-aria combobox: click to open, choose the Cheng T option.
    sel.click()
    settle(page, quiet_ms=300)
    opt = page.get_by_text("Cheng — T (current-source T HBT)", exact=False).first
    opt.click()
    settle(page, quiet_ms=300)

    page.get_by_role("button", name="Load", exact=True).first.click()
    settle(page, quiet_ms=800)
    hide_badge(page)

    hdr = page.get_by_text("Live schematic", exact=False).first
    hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=500)
    hide_badge(page)
    hb = hdr.bounding_box()
    print("SCHEMATIC HDR BOX", hb)
    shot("01_loaded_schematic", clip={"x": 370, "y": max(0, hb["y"] - 10),
                                      "width": 1500, "height": 620})
    print("DONE")
