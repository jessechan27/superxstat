"""🧑‍💻 自訂程式碼頁：用 matplotlib / seaborn 寫 Python，畫出內建圖表做不到的圖。"""

from __future__ import annotations

import streamlit as st

from .. import code_runner, export, runtime
from ..i18n import _
from . import common as ui
from .page_plot import DPI_CHOICES, style_from_state


def _load_template() -> None:
    st.session_state["code:src"] = code_runner.template_code(st.session_state["code:tpl"])


def _clear_results() -> None:
    old = st.session_state.pop("code_res", None)
    for fig in (old or {}).get("figs", []):
        export.close(fig)


def render(ds) -> None:
    if not runtime.code_execution_enabled():
        st.warning(_("公開網站基於安全考量停用自訂程式碼 (程式會在伺服器上執行)。請在自己電腦上執行本程式 "
                     "(`streamlit run app.py`，以 localhost 開啟) 即可使用；伺服器管理者也可設定環境變數 BIOSTAT_ENABLE_CODE=1。"))
        return
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        ui.selectbox(_("範本"), list(code_runner.TEMPLATES), "code:tpl", on_change=_load_template, format_func=_)
        ui.init("code:src", code_runner.template_code(st.session_state["code:tpl"]))
        st.text_area(_("Python 程式碼"), key="code:src", height=520)
        st.caption(_("可用變數：`df` (目前資料，已套用整理步驟)、`pd`、`np`、`plt`、`sns`、`stats` (scipy.stats)、"
                     "`sm` / `smf` (statsmodels)、`bstats` (本程式統計函式)。程式中建立的每張圖都會顯示在右側。"))
        c1, c2 = st.columns([1.6, 1], vertical_alignment="center")
        with c1:
            use_style = ui.checkbox(_("套用「繪圖」頁的格式 (字型、字級、主題)"), "code:style", default=False)
        run = c2.button("▶ " + _("執行"), type="primary", key="btn:run_code", width="stretch")
    if run:
        _clear_results()
        with st.spinner(_("執行中…")):
            st.session_state["code_res"] = code_runner.run_user_code(st.session_state["code:src"], ds.df,
                                                                     style_from_state() if use_style else None)
    res = st.session_state.get("code_res")
    with right:
        if res is None:
            st.info(_("選一個範本或自己寫程式，按「執行」產生圖表。"))
            return
        if res["error"]:
            st.error(_("程式執行錯誤"))
            st.code(res["error"], language="text")
        if res["stdout"]:
            with st.expander(_("輸出 (print)"), expanded=True):
                st.code(res["stdout"], language="text")
        if not res["figs"]:
            if not res["error"]:
                st.info(_("程式沒有建立任何圖。"))
            return
        c1, c2 = st.columns(2)
        with c1:
            fmt = ui.selectbox(_("下載格式"), list(export.FORMATS), "code:fmt", default="PNG", format_func=_)
        with c2:
            dpi = ui.selectbox(_("解析度 (DPI)"), DPI_CHOICES, "code:dpi", default=300)
        for i, fig in enumerate(res["figs"], 1):
            try:
                prev, _ext, _mime = export.fig_bytes(fig, "PNG", dpi=120, rc=res["rc"])
                content, ext, mime = export.fig_bytes(fig, fmt, dpi=dpi, rc=res["rc"])
            except Exception as err:
                st.error(_("圖 {i} 輸出失敗：{err}", i=i, err=err))
                continue
            st.image(prev, width="content")
            st.download_button(_("下載圖 {i} ({ext}, {kb:,.0f} KB)", i=i, ext=ext.upper(), kb=len(content) / 1024), content, f"custom_figure_{i}.{ext}",
                               mime, key=f"dl:code:{i}", on_click="ignore")
