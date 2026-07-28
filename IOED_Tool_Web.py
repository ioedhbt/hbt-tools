import os
import streamlit as st

# ⚠️ 整個專案的網頁設定，統一在這裡宣告一次
st.set_page_config(page_title="IOED Lab Portal", layout="wide", page_icon="🔬")

from tools import i18n
from tools.SSM.helpers import segmented_radio
# All app-wide CSS lives in tools/ui_theme.py — edit there, not here.
from tools.ui_theme import inject_css, render_ram_badge


inject_css()

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

def _app_password():
    """The configured access password, or None when there isn't one.

    This used to fall back to a hardcoded "IOED" on any exception.  The
    comment called it a local-testing default, but `except Exception` also
    covers "no secrets file", which is what a public Streamlit Cloud
    deployment looks like before APP_PASSWORD is set in the dashboard — so
    the gate silently accepted a password committed in plaintext in this
    repo.  Developer machines don't need the fallback: LAUNCH_Tool.py sets
    HBT_LOCAL_LAUNCH=1 and the gate is skipped entirely (see below).
    """
    try:
        return st.secrets["APP_PASSWORD"]
    except Exception:                                             # noqa: BLE001
        return None


def check_password():
    def password_entered():
        correct_pwd = _app_password()
        if correct_pwd is None:
            st.session_state["authenticated"] = False
            return
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
        if _app_password() is None:
            # Fail closed rather than fall back to a shared default.
            st.error(
                "APP_PASSWORD is not configured for this deployment. / "
                "此部署尚未設定 APP_PASSWORD。")
            st.caption(
                "Set it in the Streamlit Cloud app settings (Secrets), or "
                "run locally with LAUNCH_Tool.py, which skips this gate.")
            return False
        st.info("Please enter IOED Lab Password. / 請輸入 IOED 實驗室專屬密碼。")
        st.text_input("Access Password / 存取密碼", type="password", on_change=password_entered, key="pwd_input")
        if "authenticated" in st.session_state and not st.session_state["authenticated"]:
            st.error("❌ Password Incorrect / 密碼錯誤")
    return False

# 1. 攔截未登入的使用者 (skipped on local launches)
if not _LOCAL_LAUNCH and not check_password():
    st.stop()

# 2. 語言切換 (drives tools/i18n.get_lang()).  Must be rendered BEFORE
#    st.navigation so nav group headers + tool titles localize on the same
#    rerun.  Visual placement (fixed top-right) is handled entirely by CSS in
#    tools/ui_theme.py targeting div.st-key-lang_toggle — the widget itself
#    sits here in the normal flow but is lifted out of the document flow by
#    position:fixed, so it no longer pushes content down.  The RAM badge
#    renders first inside the same container so it sits left of the toggle
#    in the CSS row layout.
with st.container(key="lang_toggle"):
    render_ram_badge()
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

# ── Widget-state keep-alive across page switches ──────────────────────────────
# Streamlit garbage-collects session_state entries belonging to widgets that
# skip a single script run — so every page switch used to wipe the other RF
# pages' fine-tune / forward-sim inputs back to 0 (while the plain sync-hash
# keys survived, blocking any reseed: the "all values are 0 after handoff"
# bug).  Re-assigning each key marks it programmatic for this run, which
# exempts it from cleanup.  Button-like keys refuse assignment — skip them.
_KEEPALIVE_PREFIXES = ("sim_", "rfsim_", "smith_scale_")
for _k in list(st.session_state.keys()):
    if _k.startswith(_KEEPALIVE_PREFIXES):
        try:
            st.session_state[_k] = st.session_state[_k]
        except Exception:                                       # noqa: BLE001
            pass

pg = st.navigation(nav)

# 5. 執行導航
pg.run()