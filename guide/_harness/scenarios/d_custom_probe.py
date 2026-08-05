"""Probe: Custom model builder — land on Build model tab."""
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
    shot("00_landed", full_page=True)
    print("DONE")
