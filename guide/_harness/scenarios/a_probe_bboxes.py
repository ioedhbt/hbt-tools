"""Print bounding boxes for at-a-glance UI elements used in annotation."""
ENTRY = "IOED_Tool_Web.py"


def run(page, shot):
    from session import settle

    page.get_by_role("link", name="RF At a Glance").first.click()
    page.wait_for_selector('input[type="file"]')
    settle(page)

    def bb(label, loc):
        try:
            print(label, loc.bounding_box(timeout=3000), flush=True)
        except Exception as e:
            print(label, "FAILED", str(e)[:100], flush=True)

    bb("uploader", page.locator('[data-testid="stFileUploader"]:has-text("Upload DUT")').first)
    bb("clear_btn", page.get_by_role("button", name="Clear uploads").first)
    bb("tab_overlay", page.get_by_role("tab", name="Overlay").first)
    bb("tab_individual", page.get_by_role("tab", name="Individual").first)
    bb("tab_summary", page.get_by_role("tab", name="Summary").first)
    bb("tab_batch", page.get_by_role("tab", name="Batch De-embed").first)
