"""custom-model.md — add r_delay_e / c_delay_e to the emitter branch of the
loaded Cheng T preset.

Shots:
  00_emitter_empty  — "3 . Delay / port extras" section, emitter chip
                       selected, no components yet.
  01_emitter_added  — after adding R + C (renamed r_delay_e / c_delay_e),
                       schematic showing the new branch.
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
    page.get_by_text("Build model", exact=True).first.click()
    settle(page)
    page.get_by_text("Modify an existing model", exact=False).first.click()
    settle(page)

    sel = page.get_by_label("Built-in topology", exact=False).first
    sel.click()
    settle(page, quiet_ms=300)
    page.get_by_text("Cheng — T (current-source T HBT)", exact=False).first.click()
    settle(page, quiet_ms=300)
    page.get_by_role("button", name="Load", exact=True).first.click()
    settle(page, quiet_ms=800)
    hide_badge(page)

    sec3_hdr = page.get_by_text("3 · Delay / port extras", exact=False).first
    sec3_hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=400)
    hide_badge(page)

    # Scope every subsequent lookup to Section 3's bordered container so we
    # never touch the intrinsic-core section's identically-labelled buttons.
    sec3 = sec3_hdr.locator("xpath=ancestor::div[@data-testid='stLayoutWrapper'][1]")
    print("SEC3 COUNT", sec3.count())

    chip = sec3.get_by_text("E delay", exact=False).first
    print("CHIP BOX", chip.bounding_box())
    chip.click()
    settle(page, quiet_ms=400)
    hide_badge(page)

    sb = sec3.bounding_box()
    print("SEC3 BOX", sb)
    shot("00_emitter_empty", clip={"x": 370, "y": max(0, sb["y"] - 10),
                                   "width": 1500, "height": min(sb["height"] + 20, 520)})

    add_series = sec3.get_by_role("button", name="Add series step", exact=False).first
    print("ADD SERIES COUNT", sec3.get_by_role("button", name="Add series step", exact=False).count())
    add_series.click()
    settle(page, quiet_ms=400)

    add_r = sec3.get_by_role("button", name="R (parallel)", exact=False).first
    print("ADD R COUNT", sec3.get_by_role("button", name="R (parallel)", exact=False).count())
    add_r.click()
    settle(page, quiet_ms=400)
    add_c = sec3.get_by_role("button", name="C (parallel)", exact=False).first
    add_c.click()
    settle(page, quiet_ms=400)
    hide_badge(page)

    names = sec3.locator('input[type="text"]')
    n = names.count()
    print("NAME INPUTS", n)
    for i in range(n):
        try:
            print("NAME", i, names.nth(i).input_value())
        except Exception as e:
            print("NAME", i, "ERR", e)
    def sec3_container():
        return page.get_by_text("3 · Delay / port extras", exact=False).first \
            .locator("xpath=ancestor::div[@data-testid='stLayoutWrapper'][1]")

    if n >= 2:
        names.nth(0).fill("r_delay_e")
        names.nth(0).press("Enter")
        settle(page, quiet_ms=800)
        page.wait_for_timeout(600)
        names_after1 = sec3_container().locator('input[type="text"]')
        for i in range(names_after1.count()):
            print("AFTER1", i, names_after1.nth(i).input_value())
        # Target by CURRENT value so a re-render/reorder can't shift the index.
        target = None
        for i in range(names_after1.count()):
            if names_after1.nth(i).input_value() == "C_delay_E":
                target = names_after1.nth(i)
                break
        if target is None:
            target = names_after1.nth(1)
        target.click()
        target.fill("c_delay_e")
        sec3_hdr.click()
        settle(page, quiet_ms=800)
        page.wait_for_timeout(2000)
        settle(page, quiet_ms=800)
        names_after2 = sec3_container().locator('input[type="text"]')
        for i in range(names_after2.count()):
            print("AFTER2", i, names_after2.nth(i).input_value())

        # Force a hard st.rerun() (chip switch) to rule out a stale iframe
        # srcdoc as the cause of the schematic not picking up the rename.
        sec3_container().get_by_text("Port 1 (B)", exact=False).first.click()
        settle(page, quiet_ms=600)
        sec3_container().get_by_text("E delay", exact=False).first.click()
        settle(page, quiet_ms=600)
        page.wait_for_timeout(500)

    hdr = page.get_by_text("Live schematic", exact=False).first
    hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=500)
    hide_badge(page)
    hb = hdr.bounding_box()
    print("SCHEMATIC HDR BOX", hb)
    shot("01_emitter_added", clip={"x": 370, "y": max(0, hb["y"] - 10),
                                   "width": 1500, "height": 750})
    print("DONE")
