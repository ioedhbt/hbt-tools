import os
import streamlit as st

# ⚠️ 整個專案的網頁設定，統一在這裡宣告一次
st.set_page_config(page_title="IOED Lab Portal", layout="wide", page_icon="🔬")

from tools import i18n
from tools.SSM.helpers import segmented_radio


# ── Global button styling ─────────────────────────────────────────────────────
# Streamlit's default (secondary) buttons are white with a thin border, which
# users kept mistaking for labels.  Give every secondary button — st.button,
# st.download_button, form submits, popover triggers, the uploader's "Browse
# files" — a light-gray fill so it reads as a clickable button.  Primary
# buttons (theme blue) and segmented-control chips are deliberately excluded.
# The theme is locked to light mode in .streamlit/config.toml, so fixed hex
# grays are safe.  Keep in sync with the standalone copy in
# tools/ebeam_calculator.py and the iframe copy-button in
# tools/SSM/helpers/chart_export.py.
def _inject_button_css() -> None:
    st.markdown(
        """
        <style>
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-secondaryFormSubmit"] {
            background-color: #E9EDF3;
        }
        button[data-testid="stBaseButton-secondary"]:hover,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
            background-color: #DDE3EB;
        }
        button[data-testid="stBaseButton-secondary"]:active,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:active {
            background-color: #D1D8E2;
        }
        button[data-testid="stBaseButton-secondary"]:disabled,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:disabled {
            background-color: #F1F3F7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_button_css()

# ── Firefox-only styling fix (DISABLED) ───────────────────────────────────────
# On macOS Firefox the default Streamlit chrome renders with near-invisible
# input borders and thin container outlines (text fields blend into the page,
# separators disappear).  Chrome / Safari / Edge and Firefox on Windows/Linux
# look fine.  The Firefox-scoped (`@-moz-document url-prefix()`) fix below is
# currently disabled — re-enable by uncommenting the function and its call.
# def _inject_firefox_fix() -> None:
#     st.markdown(
#         """
#         <style>
#         @-moz-document url-prefix() {
#           /* Text / number / textarea / select inputs — give them a clearly
#              visible border so the field stands out from the page. */
#           div[data-baseweb="input"],
#           div[data-baseweb="base-input"],
#           div[data-baseweb="textarea"],
#           .stTextInput div[data-baseweb="input"],
#           .stNumberInput div[data-baseweb="input"],
#           .stDateInput div[data-baseweb="input"],
#           div[data-baseweb="select"] > div {
#               border: 1.5px solid rgba(49, 51, 63, 0.45) !important;
#               border-radius: 6px !important;
#               background-color: rgba(0, 0, 0, 0.015) !important;
#           }
#           div[data-baseweb="input"]:focus-within,
#           div[data-baseweb="select"] > div:focus-within {
#               border-color: rgba(49, 51, 63, 0.85) !important;
#           }
#           /* Bordered containers + expanders — thicker, clearer outline. */
#           div[data-testid="stExpander"],
#           div[data-testid="stVerticalBlockBorderWrapper"] {
#               border: 1.5px solid rgba(49, 51, 63, 0.30) !important;
#               border-radius: 8px !important;
#           }
#           /* Horizontal rules / dividers — make the separation visible. */
#           hr, div[data-testid="stDivider"] hr {
#               border-top: 1px solid rgba(49, 51, 63, 0.30) !important;
#           }
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )
#
#
# _inject_firefox_fix()

# Local-launch bypass — LAUNCH_Tool.py sets HBT_LOCAL_LAUNCH=1 before
# spawning streamlit so the password screen is skipped on developer
# machines.  Streamlit Cloud / public deployments don't set this, so
# the password gate is preserved there.
_LOCAL_LAUNCH = os.environ.get("HBT_LOCAL_LAUNCH", "").strip().lower() in {
    "1", "true", "yes", "on"
}

def check_password():
    def password_entered():
        try:
            correct_pwd = st.secrets["APP_PASSWORD"]
        except Exception:
            correct_pwd = "IOED" # 本機測試預設密碼
        if st.session_state["pwd_input"] == correct_pwd:
            st.session_state["authenticated"] = True
            del st.session_state["pwd_input"]
        else:
            st.session_state["authenticated"] = False

    if st.session_state.get("authenticated", False):
        return True

    st.title("🔬 IOED Lab Unified Portal")
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Please enter IOED Lab Password. / 請輸入 IOED 實驗室專屬密碼。")
        st.text_input("Access Password / 存取密碼", type="password", on_change=password_entered, key="pwd_input")
        if "authenticated" in st.session_state and not st.session_state["authenticated"]:
            st.error("❌ Password Incorrect / 密碼錯誤")
    return False

# 1. 攔截未登入的使用者 (skipped on local launches)
if not _LOCAL_LAUNCH and not check_password():
    st.stop()

# 2. 語言切換 (drives tools/i18n.get_lang()).  Pinned top-right of the main
#    content area — st.navigation always renders the page list at the top of the
#    sidebar and pushes any sidebar widgets below it, where the language toggle
#    was easy to miss.  Rendered before st.navigation so the nav group headers +
#    tool titles still localize on the same rerun.
_top_spacer, _top_lang = st.columns([5, 2])
with _top_lang:
    segmented_radio(
        i18n.t("language_label"), i18n.LANGS, key="ui_lang",
        format_func=lambda l: f"🌐 {l}", label_visibility="collapsed",
    )

# Sidebar brand (appears under the auto-generated page navigation).
st.sidebar.title(f"🔬 {i18n.t('portal_title')}")
st.sidebar.caption(i18n.t("portal_caption"))
st.sidebar.divider()

# 3. 定義功能頁面 — names + icons come from the i18n registry so the sidebar
#    label and each tool's in-page st.title() can never drift apart again.
def _page(tool_key, *, default=False):
    m = i18n.TOOLS[tool_key]
    return st.Page(m["path"], title=i18n.tool_name(tool_key),
                   icon=m["icon"], default=default)

home_page = st.Page("tools/home.py", title=i18n.t("home_nav"),
                    icon="🏠", default=True)
# The custom-model builder is no longer its own page — it is embedded as a
# "🧩 Custom model" section inside both RF Forward Simulator and HBT SSM
# Extraction (see tools/SSM/custom_model/render_custom_section).

# 4. 建立側邊欄群組導航選單 — Home first, then one group per measurement domain.
nav: dict = {i18n.t("start_group"): [home_page]}
for group_key in i18n.GROUP_ORDER:
    pages = [_page(k) for k, m in i18n.TOOLS.items() if m["group"] == group_key]
    if pages:
        nav[i18n.group_label(group_key)] = pages

pg = st.navigation(nav)

# 5. 執行導航
pg.run()