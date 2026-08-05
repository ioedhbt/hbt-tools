"""
process_flow.py — the "HBT Process Flow Illustration" page.

Layer:       page (Streamlit script, executed top-to-bottom on every interaction)
Renders:     one of _VARIANTS' HTML files, verbatim, in an iframe, picked by a
             Formal / QAD segmented radio.

Why this file lives here
-----------------------
A tool's folder must equal its i18n group key (MAP.md invariant 1), and this
page belongs to the ``process`` group — the one the sidebar labels "Process".
It is **not** part of the EBL calculator, which lives in the sibling
``tools/process/ebeam/`` folder.  ``tools/process/ebeam/AGENTS.md``'s "imports
nothing from the rest of the repo" rule exists so ``launch_ebl_calculator.py``
can ship ``calculator.py`` and its ``gdsii/`` subtree standalone; this page is
portal-only and is not in that set, so it uses ``tools.common`` like every
other page.

Each illustration is a hand-written, dependency-free WebGL document owned by
this folder.  Nothing here parses or regenerates either one — the chosen file
is read as text and handed straight to the iframe, so changing an illustration
means editing that one file and nothing else.

The one exception is language.  Each illustration carries its own English /
繁體中文 layer and its own toggle, so that the standalone file works on its
own; embedded in the portal there must be exactly one language control, the
sidebar's 🌐.  So this page rewrites a single handshake line in the document,
which tells it the active language and to hide its own button.  Nothing else
about the HTML is touched, and the download button still hands out the
untouched, self-governing file.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from tools.common import i18n
from tools.common.paths import REPO_ROOT
from tools.common.widgets import segmented_radio

_HERE = REPO_ROOT / "tools" / "process" / "process_flow_illustration"

# option value -> (file, English label, Chinese label)
_VARIANTS = {
    "formal": (_HERE / "inp_hbt_process_flow.html",
               "Formal", "正式版"),
    "qad":    (_HERE / "qad_hbt_process_flow.html",
               "QAD", "簡易版"),
}

# The documents size their own shell with `height:100vh; min-height:560px`,
# which inside an iframe means "however tall the iframe is" — so the slider
# below genuinely resizes the illustration rather than scrolling it, and 560
# is the floor at which their three-column layout still fits.
_MIN_HEIGHT = 560
_DEFAULT_HEIGHT = 900

# The illustrations' language handshake, verbatim — both files carry the same
# line. Its default — no language imposed, not embedded — is what makes the
# standalone file self-governing: it reads ?lang=, then localStorage, and
# shows its own toggle.
_HOST_LINE = "const HOST = {lang:null, embed:false};"


@st.cache_data(show_spinner=False)
def _load_flow_html(path_str: str, mtime: float) -> str:
    """Read the illustration verbatim.

    ``mtime`` is unused in the body but part of the cache key, so editing the
    HTML shows up on the next rerun without anyone having to clear the cache.
    """
    return Path(path_str).read_text(encoding="utf-8")


def _hosted(html: str, lang: str) -> str | None:
    """Hand the illustration the portal's language, or ``None`` if it can't.

    ``None`` means the handshake line is gone or duplicated — the caller then
    embeds the file untouched, which still works, just with the document's own
    toggle visible and its own idea of the language.
    """
    if html.count(_HOST_LINE) != 1:
        return None
    return html.replace(_HOST_LINE, f"const HOST = {{lang:'{lang}', embed:true}};")


st.title(i18n.title("process_flow"))
st.caption(i18n.tool_desc("process_flow"))

variant = segmented_radio(
    i18n.tr("Variant", "版本"),
    list(_VARIANTS),
    index=0,
    key="process_flow_variant",
    format_func=lambda k: _VARIANTS[k][2] if i18n.is_zh() else _VARIANTS[k][1],
)
FLOW_HTML, _, _ = _VARIANTS[variant]

if not FLOW_HTML.is_file():
    rel = FLOW_HTML.relative_to(REPO_ROOT)
    st.error(i18n.tr(f"Illustration not found: `{rel}`",
                     f"找不到圖解檔案：`{rel}`"))
    st.stop()

_flow_html = _load_flow_html(str(FLOW_HTML), FLOW_HTML.stat().st_mtime)

# controls, download = st.columns([3, 1], vertical_alignment="bottom")
# with download:
st.download_button(
    i18n.tr("⬇️ Standalone file", "⬇️ 下載獨立檔"),
    data=_flow_html,
    file_name=FLOW_HTML.name,
    mime="text/html",
    width="stretch",
    help=i18n.tr(
        "The illustration is one self-contained HTML file with no dependencies. Download it and open it in its own browser tab.",
        "圖解是單一自包含 HTML 檔，無任何相依套件 — 下載後可直接在瀏覽器分頁開啟。"),
)

_hosted_html = _hosted(_flow_html, "zh" if i18n.is_zh() else "en")
if _hosted_html is None:
    st.warning(i18n.tr(
        "The illustration's language handshake has moved, so it is embedded "
        "untouched and keeps its own language toggle.",
        "圖解的語言接口位置已變動，因此直接內嵌原檔，並保留圖解自己的語言切換鈕。"))

# A local .html path and a raw HTML string take the same route inside
# st.iframe — both end up in srcdoc — so handing over the rewritten text costs
# nothing and behaves exactly like handing over the file.
st.iframe(_hosted_html or _flow_html, height=900)
