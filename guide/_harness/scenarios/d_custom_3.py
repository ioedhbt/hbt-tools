"""custom-model.md — build (Cheng T + r_delay_e/c_delay_e), send to Simulate &
Fit, capture the component-values panel + resulting Smith/Bode.

Shots:
  00_values_panel — "Port / delay extras" group showing r_delay_e / c_delay_e
                     inputs.
  01_sim_result   — Smith chart + fT/fmax for the modified model.
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
    settle(page, quiet_ms=600)
    hide_badge(page)

    sec3_hdr = page.get_by_text("3 · Delay / port extras", exact=False).first
    sec3_hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    sec3 = sec3_hdr.locator("xpath=ancestor::div[@data-testid='stLayoutWrapper'][1]")
    sec3.get_by_text("E delay", exact=False).first.click()
    settle(page, quiet_ms=300)

    sec3.get_by_role("button", name="Add series step", exact=False).first.click()
    settle(page, quiet_ms=300)
    sec3.get_by_role("button", name="R (parallel)", exact=False).first.click()
    settle(page, quiet_ms=300)
    sec3.get_by_role("button", name="C (parallel)", exact=False).first.click()
    settle(page, quiet_ms=300)

    names = sec3.locator('input[type="text"]')
    names.nth(0).fill("r_delay_e")
    names.nth(0).press("Enter")
    settle(page, quiet_ms=500)
    names2 = sec3.locator('input[type="text"]')
    tgt = None
    for i in range(names2.count()):
        if names2.nth(i).input_value() == "C_delay_E":
            tgt = names2.nth(i)
            break
    if tgt is None:
        tgt = names2.nth(1)
    tgt.click()
    tgt.fill("c_delay_e")
    sec3_hdr.click()
    settle(page, quiet_ms=500)

    # Send straight to Simulate / Fit — skip sections 4/5, defaults are fine.
    send_btn = page.get_by_text("Send to Simulate / Fit view", exact=False).first
    send_btn.evaluate("el => el.scrollIntoView({block:'center', behavior:'instant'})")
    settle(page, quiet_ms=300)
    send_btn.click()
    settle(page, quiet_ms=800)
    hide_badge(page)

    values_hdr = page.get_by_text("Port / delay extras", exact=False).first
    values_hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=400)
    hide_badge(page)
    vb = values_hdr.bounding_box()
    print("VALUES BOX", vb)
    shot("00_values_panel", clip={"x": 370, "y": max(0, vb["y"] - 10),
                                  "width": 1500, "height": 160})

    smith_hdr = page.get_by_text("Smith Chart", exact=False).first
    smith_hdr.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=500)
    hide_badge(page)
    shb = smith_hdr.bounding_box()
    print("SMITH BOX", shb)
    shot("01_sim_result", clip={"x": 370, "y": max(0, shb["y"] - 10),
                                "width": 1500, "height": 650})
    print("DONE")
