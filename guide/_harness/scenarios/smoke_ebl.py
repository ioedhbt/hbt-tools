"""Check the EBL page renders and the gdstk stand-in satisfies the viewer."""

ENTRY = "IOED_Tool_Web.py"


def run(page, shot):
    page.get_by_role("link", name="EBL Calculator").first.click()
    shot("00_ebl", full_page=True)
    body = page.locator('[data-testid="stMain"]').inner_text()
    assert "gdstk` is not installed" not in body, "gdstk shim not picked up"
