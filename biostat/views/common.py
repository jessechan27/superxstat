"""介面共用工具：session state、控制項、側邊欄資料來源、表格顯示。"""

from __future__ import annotations

import hashlib
import html
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from .. import __version__, data, i18n, runtime
from ..i18n import _


# ------------------------------------------------------------------ state 與控制項
def init(key: str, default) -> None:
    """控制項的初始值只寫入 session_state 一次；之後控制項不再傳 value/default，避免 Streamlit 警告。"""
    if key not in st.session_state:
        st.session_state[key] = default


def ensure_option(key: str, options: list, default) -> None:
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default if default in options else (options[0] if options else None)


def selectbox(label, options, key, default=None, **kw):
    ensure_option(key, list(options), default)
    return st.selectbox(label, options, key=key, **kw)


def multiselect(label, options, key, default=None, **kw):
    opts = list(options)
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in opts]
    else:
        st.session_state[key] = [v for v in (default or []) if v in opts]
    return st.multiselect(label, opts, key=key, **kw)


def checkbox(label, key, default=False, **kw):
    init(key, default)
    return st.checkbox(label, key=key, **kw)


def number(label, key, default=None, **kw):
    init(key, default)
    return st.number_input(label, key=key, **kw)


def text(label, key, default="", **kw):
    init(key, default)
    return st.text_input(label, key=key, **kw)


def color(label, key, default="#000000", **kw):
    init(key, default)
    return st.color_picker(label, key=key, **kw)


def slider(label, key, default, **kw):
    init(key, default)
    return st.slider(label, key=key, **kw)


# 按鈕、上傳、下載、資料編輯器的值不能透過 session_state 指定，跨頁保留時要略過
_NO_SET_PREFIX = ("upload", "style_upload", "dl", "ed:", "btn:", "reset_colors:", "FormSubmitter", "$$")
_NO_SET_SUFFIX = (":az", ":orig", ":rev")


def keep_state() -> None:
    """跨頁保留控制項設定。

    Streamlit 會清掉本次執行中沒有出現的控制項狀態 (切換頁面時設定就會消失)；
    在每次執行開頭把值重新指定一次，就能讓它們留在 session_state。
    """
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and not k.startswith(_NO_SET_PREFIX) and not k.endswith(_NO_SET_SUFFIX):
            try:
                st.session_state[k] = st.session_state[k]
            except Exception:
                pass


# ------------------------------------------------------------------ 資料來源
@dataclass
class Dataset:
    df: pd.DataFrame
    id: str
    name: str
    store: dict
    errors: list


@st.cache_data(show_spinner=False)
def _demo() -> dict:
    return data.make_demo_data()


@st.cache_data(show_spinner=False)
def _demo_xlsx() -> bytes:
    return data.demo_excel_bytes()


@st.cache_data(show_spinner=False, max_entries=20)
def _sheets(content: bytes, name: str) -> list[str]:
    return data.list_sheets(content, name)


@st.cache_data(show_spinner="讀取檔案中…", max_entries=20)
def _load(content: bytes, name: str, sheet, header_row, skiprows: int) -> pd.DataFrame:
    return data.read_table(content, name, sheet, header_row, skiprows)


def sidebar_data() -> Dataset:
    with st.sidebar:
        st.markdown("#### 📁 " + _("資料來源"))
        up = st.file_uploader(_("上傳 Excel / CSV"), type=["xlsx", "xlsm", "xls", "csv", "tsv", "txt"], key="upload",
                              help=_("支援 .xlsx / .xls / .csv / .tsv；CSV 會自動判斷 UTF-8 或 Big5 編碼。"))
        if up is not None:
            content = up.getvalue()
            digest = hashlib.md5(content).hexdigest()[:10]
            sheets = _sheets(content, up.name)
            sheet = selectbox(_("工作表"), sheets, key=f"sheet:{digest}") if len(sheets) > 1 else (sheets[0] if sheets else None)
            with st.expander(_("讀取設定")):
                no_header = checkbox(_("第一列不是欄位名稱"), key=f"nohdr:{digest}")
                header = number(_("欄位名稱在第幾列"), key=f"hdr:{digest}", default=1, min_value=1, max_value=100, step=1,
                                disabled=no_header)
            try:
                raw = _load(content, up.name, sheet, None if no_header else int(header) - 1, 0)
            except Exception as err:
                st.error(_("讀取失敗：{err}", err=err))
                raw = pd.DataFrame()
            ds_id = f"{up.name}:{digest}:{sheet}:{header}:{no_header}"
            name = up.name + (f" › {sheet}" if sheet else "")
        else:
            demo = _demo()
            names = list(data.DEMO_DESCRIPTIONS)
            sheet = selectbox(_("範例資料集 (未上傳時)"), names, key="demo_sheet", default="Groups",
                              format_func=lambda k: f"{k}：{_(data.DEMO_DESCRIPTIONS[k]).split('：')[0].split(':')[0]}")
            st.caption(_(data.DEMO_DESCRIPTIONS[sheet]))
            raw = demo[sheet]
            ds_id = f"demo:{sheet}"
            name = _("範例 › {s}", s=sheet)
            st.download_button("⬇️ " + _("下載範例 Excel (15 個工作表)"), _demo_xlsx(), "superxtat_demo_data.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", on_click="ignore")

        stores = st.session_state.setdefault("ds_store", {})
        store = stores.setdefault(ds_id, {"transforms": [], "edited": None})
        base = store["edited"] if store["edited"] is not None else raw
        df, errors = data.apply_transforms(base, store["transforms"])
        n_t = len(store["transforms"]) + (1 if store["edited"] is not None else 0)
        st.caption(_("**{name}** · {rows:,} 列 × {cols} 欄", name=name, rows=df.shape[0], cols=df.shape[1])
                   + (_(" · 已套用 {n} 個整理步驟", n=n_t) if n_t else ""))
        for e in errors:
            st.warning(e)
    return Dataset(df, ds_id, name, store, errors)


def sidebar_footer() -> None:
    with st.sidebar:
        st.divider()
        st.caption(f"SUPER XTAT v{__version__} · {runtime.runtime_label()}")
        if runtime.IS_PYODIDE:
            st.caption("🔒 " + _("瀏覽器版：所有資料只在你的電腦上處理，不會上傳到任何伺服器。"))


# ------------------------------------------------------------------ 表格顯示
display_table = data.format_table


def show_tables(tables: list, prefix: str, expanded_first: bool = True) -> None:
    for i, (name, tab) in enumerate(tables):
        if tab is None or len(tab) == 0:
            continue
        with st.expander(f"📄 {_(name)}", expanded=expanded_first and i == 0):
            st.dataframe(display_table(i18n.table(tab)), hide_index=True, width="stretch")
            st.download_button("⬇️ CSV", tab.to_csv(index=False).encode("utf-8-sig"), f"{prefix}_{name}.csv", "text/csv",
                               key=f"dl:{prefix}:{i}:{name}", on_click="ignore")


def tables_excel(tables: list) -> bytes:
    return data.to_excel_bytes({name: tab for name, tab in tables if tab is not None and len(tab)})


def swatches(colors: list) -> None:
    from matplotlib.colors import to_hex

    cells = "".join(f'<span title="{to_hex(c)}" style="display:inline-block;width:22px;height:16px;margin-right:2px;'
                    f'border-radius:3px;background:{to_hex(c)}"></span>' for c in colors)
    st.markdown(f'<div style="line-height:0">{cells}</div>', unsafe_allow_html=True)


CSS = """
<style>
:root {--sx-line:#26282C; --sx-dim:#8A8F98; --sx-ink:#F2F3F5; --sx-accent:#3E6AE1;}
.block-container {padding-top: 4.5rem; max-width: 1560px;}
header[data-testid="stHeader"] {background: rgba(0,0,0,.82); backdrop-filter: blur(12px); border-bottom: 1px solid var(--sx-line);}
section[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
h1, h2, h3, h4, h5 {letter-spacing: .06em;}
div[data-testid="stExpander"] details summary p {font-weight: 500; letter-spacing: .03em;}

/* 按鈕：SpaceX 線框，滑過反白；主要按鈕實心白，滑過轉藍 */
button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-secondaryFormSubmit"] {
  background: transparent; border: 1px solid rgba(242,243,245,.45); color: var(--sx-ink);
  transition: background .18s, color .18s, border-color .18s;}
button[data-testid="stBaseButton-secondary"]:hover, button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
  background: var(--sx-ink); color: #000; border-color: var(--sx-ink);}
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--sx-ink); color: #000; border: 1px solid var(--sx-ink); transition: background .18s, color .18s, border-color .18s;}
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  background: var(--sx-accent); border-color: var(--sx-accent); color: #fff;}
button[data-testid^="stBaseButton-primary"] p, button[data-testid^="stBaseButton-secondary"] p {
  text-transform: uppercase; letter-spacing: .12em; font-size: .78rem; font-weight: 500;}
button[data-testid^="stBaseButton"]:disabled {opacity: .35;}

/* 分頁標籤與指標 */
button[data-baseweb="tab"] p {letter-spacing: .08em; font-size: .84rem;}
div[data-baseweb="tab-highlight"] {background-color: var(--sx-ink);}
[data-testid="stMetricLabel"] p {text-transform: uppercase; letter-spacing: .16em; font-size: .68rem; color: var(--sx-dim);}
[data-testid="stMetricValue"] {font-family: "Barlow", sans-serif; font-weight: 300; letter-spacing: .02em;}
[data-testid="stImage"] img {border-radius: 3px;}

/* 頁首 */
.sx-hero {position: relative; padding: 6px 0 20px; margin-bottom: 20px; border-bottom: 1px solid var(--sx-line);}
.sx-hero:after {content: ""; position: absolute; left: 0; bottom: -1px; width: 120px; height: 1px;
  background: linear-gradient(90deg, var(--sx-accent), transparent);}
.sx-kicker {font: 500 .7rem/1 "Barlow", sans-serif; letter-spacing: .42em; text-transform: uppercase; color: var(--sx-dim);}
.sx-title {font: 500 2.2rem/1.15 "Barlow", "Microsoft JhengHei", "PingFang TC", sans-serif; letter-spacing: .08em;
  text-transform: uppercase; color: var(--sx-ink); margin-top: 10px;}
.sx-sub {color: var(--sx-dim); font-size: .92rem; margin-top: 6px; letter-spacing: .02em;}
.sx-brand {font: 600 1.05rem/1 "Barlow", sans-serif; letter-spacing: .5em; color: var(--sx-ink); padding: 2px 0 14px;}
.sx-brand span {color: var(--sx-accent);}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def language_switch() -> None:
    """側邊欄最上方的語言切換。值存在 session_state["lang"]，app.py 每次執行開頭據此設定語言。"""
    with st.sidebar:
        st.segmented_control("Language", list(i18n.LANGS), key="lang", format_func=i18n.LANGS.get,
                             label_visibility="collapsed", width="stretch")


def page_header(kicker: str, title: str, sub: str = "") -> None:
    st.markdown(f'<div class="sx-hero"><div class="sx-kicker">Super Xtat · {html.escape(kicker)}</div>'
                f'<div class="sx-title">{html.escape(title)}</div>'
                + (f'<div class="sx-sub">{html.escape(sub)}</div>' if sub else "") + "</div>", unsafe_allow_html=True)


def np_safe(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v
