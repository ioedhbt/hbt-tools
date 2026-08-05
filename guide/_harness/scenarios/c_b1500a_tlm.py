"""B1500A page — TLM Analysis tab, blank/default state (no TLM sweep ships
in the demo data, see guide/_data/dc/CHOICES.md)."""

ENTRY = "IOED_Tool_Web.py"


def run(page, shot):
    from session import settle

    page.get_by_role("link", name="DC Analysis").first.click(timeout=5000)
    settle(page)

    page.evaluate(
        "() => {"
        "let s = document.getElementById('_hide_badge');"
        "if (!s) { s = document.createElement('style'); s.id = '_hide_badge';"
        "document.head.appendChild(s); }"
        "s.textContent = 'div.st-key-lang_toggle{display:none !important;}';"
        "}"
    )

    page.get_by_text("TLM Analysis", exact=True).first.click(timeout=5000)
    settle(page)

    shot("01_tlm_blank")

    toggle_box = page.get_by_text("Select Page", exact=True)
    print("toggle box:", toggle_box.bounding_box(timeout=4000), flush=True)
    z_box = page.get_by_test_id("stNumberInput").filter(has_text="Pad width")
    print("z box:", z_box.bounding_box(timeout=4000), flush=True)
    r32_box = page.get_by_test_id("stNumberInput").filter(has_text="R @ 32")
    print("r32 box:", r32_box.bounding_box(timeout=4000), flush=True)
