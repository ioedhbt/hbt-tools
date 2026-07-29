"""
process_flow.py — the "HBT Process Flow Illustration" page.

Layer:       page (Streamlit script, executed top-to-bottom on every interaction)
Renders:     docs/process_flow/inp_hbt_process_flow.html, verbatim, in an iframe.

Why this file lives here
-----------------------
A tool's folder must equal its i18n group key (MAP.md invariant 1), and this
page belongs to the ``ebeam`` group — the one the sidebar labels "Process".
It is **not** part of the EBL calculator.  ``tools/ebeam/AGENTS.md``'s "imports
nothing from the rest of the repo" rule exists so ``launch_ebl_calculator.py``
can ship ``calculator.py`` and its ``gdsii/`` subtree standalone; this page is
portal-only and is not in that set, so it uses ``tools.common`` like every
other page.

The illustration is a hand-written, dependency-free WebGL document owned by
``docs/process_flow/``.  Nothing here parses, patches or regenerates it — the
file is read as text and handed straight to the iframe, so changing the
illustration means editing that one file and nothing else.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from tools.common import i18n
from tools.common.paths import REPO_ROOT

FLOW_HTML = REPO_ROOT / "docs" / "process_flow" / "inp_hbt_process_flow.html"

# The document sizes its own shell with `height:100vh; min-height:560px`, which
# inside an iframe means "however tall the iframe is" — so the slider below
# genuinely resizes the illustration rather than scrolling it, and 560 is the
# floor at which its three-column layout still fits.
_MIN_HEIGHT = 560
_DEFAULT_HEIGHT = 900


@st.cache_data(show_spinner=False)
def _load_flow_html(path_str: str, mtime: float) -> str:
    """Read the illustration, for the download button.

    ``st.iframe`` reads the file itself, so this is only the copy handed to
    ``st.download_button``.  ``mtime`` is unused in the body but part of the
    cache key, so editing the HTML shows up on the next rerun without anyone
    having to clear the cache.
    """
    return Path(path_str).read_text(encoding="utf-8")


st.title(i18n.title("process_flow"))
st.caption(i18n.tool_desc("process_flow"))

if not FLOW_HTML.is_file():
    rel = FLOW_HTML.relative_to(REPO_ROOT)
    st.error(i18n.tr(f"Illustration not found: `{rel}`",
                     f"找不到圖解檔案：`{rel}`"))
    st.stop()

# controls, download = st.columns([3, 1], vertical_alignment="bottom")
# with download:
st.download_button(
    i18n.tr("⬇️ Standalone file", "⬇️ 下載獨立檔"),
    data=_load_flow_html(str(FLOW_HTML), FLOW_HTML.stat().st_mtime),
    file_name=FLOW_HTML.name,
    mime="text/html",
    width="stretch",
    help=i18n.tr(
        "The illustration is one self-contained HTML file with no dependencies. Download it and open it in its own browser tab.",
        "圖解是單一自包含 HTML 檔，無任何相依套件 — 下載後可直接在瀏覽器分頁開啟。"),
)

# Pass the Path, not the text: st.iframe auto-detects its input, and handing
# it an explicit path skips the URL / path / raw-HTML guessing entirely.
st.iframe(FLOW_HTML, height=900)
