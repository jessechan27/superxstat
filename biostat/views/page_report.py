"""📑 分析報告頁：選擇變項 → 自動挑選檢定 → 產生含 seaborn 圖表的完整統計報告 → 下載 HTML / Word / Excel。"""

from __future__ import annotations

import importlib.util
import re

import pandas as pd
import streamlit as st

from .. import report as rp
from ..data import categorical_columns, is_categorical_like, numeric_columns
from . import common as ui

HAS_DOCX = importlib.util.find_spec("docx") is not None
HAS_PPTX = importlib.util.find_spec("pptx") is not None


def _id_like(name) -> bool:
    n = str(name).strip().lower()
    return bool(re.search(r"(^|[_\s])(id|no|編號|序號)$", n)) or n.endswith("id")


def _defaults(df: pd.DataFrame) -> tuple[list, str | None, list]:
    """猜測合理的預設：數值結果變項、2–8 類的分組變項、其他少類別的類別變項。"""
    cats = [c for c in df.columns if is_categorical_like(df[c]) and not _id_like(c) and 2 <= df[c].nunique() <= 8]
    text_cats = [c for c in cats if not pd.api.types.is_numeric_dtype(df[c])]
    group = (text_cats or cats or [None])[0]
    nums = [c for c in numeric_columns(df) if c != group and not _id_like(c) and not _codes_group(df, c, group)]
    outcomes = [c for c in nums if _continuous(df[c])][:4] or nums[:4]
    others = [c for c in text_cats if c != group and not _codes_group(df, c, group)][:2]
    return outcomes, group, others


def _continuous(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return len(v) > 0 and (not (v == v.round()).all() or v.nunique() > 10)


def _codes_group(df: pd.DataFrame, col: str, group: str | None) -> bool:
    """col 是否只是分組欄的另一種編碼 (例如 annotation 與 group = 1..5 一一對應)。"""
    if not group:
        return False
    d = df[[col, group]].dropna()
    return len(d) > 0 and d.groupby(group)[col].nunique().max() == 1 and d.groupby(col)[group].nunique().max() == 1


def _stem(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|›\s]+", "_", name).strip("_") or "report"


def _downloads(saved: dict) -> None:
    rep = saved["rep"]
    stem = _stem(rep.title)
    cols = st.columns([1, 1, 1.3, 1, 1], vertical_alignment="bottom")
    cols[0].download_button("HTML 報告", saved["html"].encode("utf-8"), f"{stem}.html", "text/html", key="dl:rep:html",
                            on_click="ignore", width="stretch", help="可直接用瀏覽器開啟，或列印成 PDF")
    if HAS_DOCX:
        if "docx" not in saved:
            saved["docx"] = rp.to_docx(rep)
        cols[1].download_button("Word 報告", saved["docx"], f"{stem}.docx",
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key="dl:rep:docx", on_click="ignore", width="stretch")
    if HAS_PPTX:
        with cols[2]:
            theme = ui.selectbox("簡報風格", list(rp.PPT_THEMES), "rp:ppt_theme", default=rp.DEFAULT_PPT_THEME,
                                 help="Super Xtat 科技風：黑底寬字距；Apple 簡約風：白底大標題、每章有分隔頁")
        key = f"pptx:{theme}"  # 每種風格各自快取
        if key not in saved:
            saved[key] = rp.to_pptx(rep, theme)
        cols[3].download_button("PPT 簡報", saved[key], f"{stem}.pptx",
                                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                key="dl:rep:pptx", on_click="ignore", width="stretch", help="16:9 投影片，每張表、每張圖各一頁")
    if "xlsx" not in saved:
        saved["xlsx"] = rp.to_excel(rep)
    cols[4].download_button("全部表格 Excel", saved["xlsx"], f"{stem}_tables.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl:rep:xlsx",
                            on_click="ignore", width="stretch")


def _png_width(png: bytes) -> int:
    return int.from_bytes(png[16:20], "big")  # PNG IHDR：寬度位於第 16–19 位元組


def _figures(sec_no: int, figs: list) -> None:
    """多面板的寬圖獨佔一列，其餘兩張一列。"""
    row: list = []

    def flush():
        if row:
            for col, (j, f) in zip(st.columns(2), row):
                with col:
                    st.image(f.png, caption=f"圖 {sec_no}.{j}　{f.caption}", width="content")
            row.clear()

    for j, f in enumerate(figs, 1):
        if _png_width(f.png) > 1100:
            flush()
            st.image(f.png, caption=f"圖 {sec_no}.{j}　{f.caption}", width="content")
        else:
            row.append((j, f))
            if len(row) == 2:
                flush()
    flush()


def _show(rep: rp.Report) -> None:
    st.markdown(f"## {rep.title}")
    st.caption(f"資料：{rep.dataset} · 產生時間：{rep.created}")
    with st.container(border=True):
        st.markdown("**重點摘要**")
        st.markdown("\n".join(f"- {h}" for h in rep.highlights))
    for i, sec in enumerate(rep.sections, 1):
        st.markdown(f"### {i:02d}　{sec.title}")
        for p in sec.paragraphs:
            st.markdown(p)
        for w in sec.warnings:
            st.warning(w)
        for j, (name, tab) in enumerate(sec.tables, 1):
            st.markdown(f"**表 {i}.{j}　{name}**")
            st.dataframe(ui.display_table(tab), hide_index=True, width="stretch")
        _figures(i, sec.figures)
    st.markdown(f"### {len(rep.sections) + 1:02d}　統計方法")
    for m in rep.methods:
        st.markdown(m)
    st.caption(rep.software)


def render(ds) -> None:
    df = ds.df
    if df.empty:
        st.info("目前沒有資料。請在左側上傳 Excel / CSV，或選擇範例資料。")
        return
    outcomes0, group0, cats0 = _defaults(df)
    k = f"rp:{ds.id}:"  # 每份資料各自記住選擇
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.6, 1, 1.3])
        with c1:
            outcomes = ui.multiselect("結果變項 (數值)", numeric_columns(df), k + "y", default=outcomes0,
                                      help="要比較或描述的連續變項，例如腫瘤體積、血中濃度")
        with c2:
            group = ui.selectbox("分組變項 (選填)", [None] + categorical_columns(df), k + "g", default=group0,
                                 format_func=lambda c: "（不分組）" if c is None else str(c), help="例如處理組別、性別")
        with c3:
            cats = ui.multiselect("類別變項 (選填)", [c for c in categorical_columns(df) if c != group], k + "c",
                                  default=cats0, help="與分組變項做卡方 / Fisher 關聯分析")
        c4, c5, c6, c7 = st.columns([1.6, 1, 0.65, 0.65], vertical_alignment="bottom")
        with c4:
            title = ui.text("報告標題", "rp:title", default="統計分析報告")
        with c5:
            strategy = ui.selectbox("檢定策略", list(rp.STRATEGIES), "rp:strategy",
                                    help="自動：符合常態且變異數同質用母數檢定，變異數不同質用 Welch，偏離常態用無母數檢定")
        with c6:
            corr = ui.checkbox("相關分析", "rp:corr", default=True)
        with c7:
            figs = ui.checkbox("seaborn 圖表", "rp:figs", default=True)
        go = st.button("產生報告", type="primary", key="btn:report", disabled=not outcomes)
    if go:
        opts = rp.ReportOptions(outcomes=list(outcomes), group=group, categorical=list(cats), strategy=rp.STRATEGIES[strategy],
                                correlation=corr, figures=figs, title=title.strip() or "統計分析報告")
        try:
            with st.spinner("分析中：執行前提檢定、選擇檢定方法、繪製 seaborn 圖表…"):
                rep = rp.build_report(df, opts, ds.name)
                st.session_state["report"] = {"rep": rep, "html": rp.to_html(rep), "ds": ds.id}
        except Exception as err:
            st.error(f"無法產生報告：{err}")
            return
    saved = st.session_state.get("report")
    if not saved:
        st.caption("選好變項後按「產生報告」。系統會依常態性 (Shapiro-Wilk) 與變異數同質性 (Brown-Forsythe) 自動挑選合適的檢定，"
                   "並寫出結果解讀、統計方法說明與 seaborn 圖表。")
        return
    if saved["ds"] != ds.id:
        st.warning("下方報告是用另一份資料產生的；請按「產生報告」重新分析目前的資料。")
    _downloads(saved)
    _show(saved["rep"])
