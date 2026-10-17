"""🗂️ 資料頁：預覽、欄位摘要、整理步驟 (寬轉長、篩選、新增欄位…)、手動編輯與下載。"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from .. import data, i18n
from ..data import QUICK_FUNCS, categorical_columns, numeric_columns
from ..i18n import _
from . import common as ui

KINDS = {  # 顯示名稱 (中文原文，介面上翻譯) → 轉換類型
    "寬轉長 (melt)": "melt",
    "篩選類別值": "filter",
    "篩選數值範圍": "range",
    "條件篩選 (運算式)": "query",
    "移除缺失值": "dropna",
    "數學轉換 (log、Z-score…)": "func",
    "公式新增欄位": "expr",
    "轉換欄位型別": "astype",
    "重新命名欄位": "rename",
    "重新編碼類別值": "recode",
    "轉置 (列 ↔ 欄)": "transpose",
}
EDITOR_MAX_ROWS = 20000


# ------------------------------------------------------------------ callbacks
def _remove(store: dict, i: int) -> None:
    if 0 <= i < len(store["transforms"]):
        store["transforms"].pop(i)


def _undo(store: dict) -> None:
    if store["transforms"]:
        store["transforms"].pop()


def _restore(store: dict) -> None:
    store["transforms"] = []
    store["edited"] = None
    store["ed_ver"] = store.get("ed_ver", 0) + 1


# ------------------------------------------------------------------ 整理步驟
def _func_name(fn: str, col: str) -> str:
    name = re.sub(r"\bx\b", lambda m: col, fn)
    return name if name != fn else f"{col}_{fn}"


def _builder(df: pd.DataFrame, store: dict, ds_id: str) -> None:
    kind = KINDS[ui.selectbox(_("整理動作"), list(KINDS), "dt:kind", format_func=_)]
    cols = list(df.columns)
    nums = numeric_columns(df)
    cats = categorical_columns(df)
    k = f"dt:{kind}:"
    t, err = None, None

    if kind == "melt":
        ids = ui.multiselect(_("保留的識別欄 (每列的 ID，可空白)"), cols, k + "ids", default=[c for c in cols if c not in nums][:1])
        vals = ui.multiselect(_("要堆疊成一欄的數值欄"), [c for c in cols if c not in ids], k + "vals",
                              default=[c for c in nums if c not in ids])
        c1, c2 = st.columns(2)
        with c1:
            var = ui.text(_("新的類別欄名稱"), k + "var", default="Group")
        with c2:
            val = ui.text(_("新的數值欄名稱"), k + "val", default="Value")
        err = None if vals else _("請選擇至少一個要堆疊的欄位")
        t = dict(type="melt", id_vars=ids, value_vars=vals, var_name=var.strip() or "Group", value_name=val.strip() or "Value")
    elif kind == "filter":
        col = ui.selectbox(_("欄位"), cats, k + "col")
        lv = data.levels(df[col]) if col else []
        keep = ui.multiselect(_("保留這些值"), lv, k + "vals:" + str(col), default=lv)
        err = None if keep else _("請至少保留一個值")
        t = dict(type="filter", column=col, values=[str(v) for v in keep])
    elif kind == "range":
        if not nums:
            st.info(_("沒有數值欄位。"))
            return
        col = ui.selectbox(_("數值欄位"), nums, k + "col")
        s = pd.to_numeric(df[col], errors="coerce")
        c1, c2 = st.columns(2)
        with c1:
            lo = ui.number(_("最小值 (含)"), k + "lo:" + col, default=float(s.min()), format="%g")
        with c2:
            hi = ui.number(_("最大值 (含)"), k + "hi:" + col, default=float(s.max()), format="%g")
        err = None if lo <= hi else _("最小值不可大於最大值")
        t = dict(type="range", column=col, lo=float(lo), hi=float(hi))
    elif kind == "query":
        expr = ui.text(_("條件運算式"), k + "expr", placeholder=_('例如：Age >= 18 and Sex == "F"'))
        st.caption(_("可用 and / or / not、==、!=、>、<；欄名含空白或中文時用反引號包住，例如 `體重 (g)` > 20"))
        err = None if expr.strip() else _("請輸入條件")
        t = dict(type="query", expr=expr.strip())
    elif kind == "dropna":
        sel = ui.multiselect(_("檢查這些欄位 (空白 = 任一欄有缺失就移除)"), cols, k + "cols")
        t = dict(type="dropna", columns=sel)
    elif kind == "func":
        if not nums:
            st.info(_("沒有數值欄位。"))
            return
        c1, c2 = st.columns(2)
        with c1:
            col = ui.selectbox(_("數值欄位"), nums, k + "col")
        with c2:
            fn = ui.selectbox(_("轉換"), list(QUICK_FUNCS), k + "fn", format_func=_)
        new = ui.text(_("新欄位名稱"), k + "new", placeholder=_func_name(fn, col))
        t = dict(type="func", column=col, func=fn, new=new.strip() or _func_name(fn, col))
    elif kind == "expr":
        new = ui.text(_("新欄位名稱"), k + "new", placeholder=_("例如 BMI"))
        expr = ui.text(_("公式"), k + "expr", placeholder=_("例如 Weight / (Height / 100) ** 2"))
        st.caption(_("可使用 + − × ÷ (`+ - * /`)、次方 `**`，以及其他欄位名稱。"))
        err = None if new.strip() and expr.strip() else _("請輸入欄位名稱與公式")
        t = dict(type="expr", new=new.strip(), expr=expr.strip())
    elif kind == "astype":
        c1, c2 = st.columns(2)
        with c1:
            col = ui.selectbox(_("欄位"), cols, k + "col")
        with c2:
            to = ui.selectbox(_("轉為"), ["數值", "類別"], k + "to", format_func=_)
        t = dict(type="astype", column=col, to="numeric" if to == "數值" else "category")
    elif kind == "rename":
        c1, c2 = st.columns(2)
        with c1:
            old = ui.selectbox(_("原欄位"), cols, k + "old")
        with c2:
            new = ui.text(_("新名稱"), k + "new")
        err = _("請輸入新名稱") if not new.strip() else (_("已有同名欄位") if new.strip() in cols else None)
        t = dict(type="rename", old=old, new=new.strip())
    elif kind == "recode":
        col = ui.selectbox(_("欄位"), cats, k + "col")
        lv = [str(v) for v in data.levels(df[col])][:50]
        if df[col].nunique() > 50:
            st.caption(_("類別超過 50 個，只列出前 50 個。"))
        st.caption(_("在「新值」欄直接修改；多個原值改成同一個新值即可合併類別。"))
        old_lbl, new_lbl = _("原值"), _("新值")
        edited = st.data_editor(pd.DataFrame({old_lbl: lv, new_lbl: lv}), key=f"ed:recode:{ds_id}:{col}:{i18n.get_lang()}",
                                disabled=[old_lbl], hide_index=True, width="stretch")
        mapping = {a: str(b) for a, b in zip(edited[old_lbl], edited[new_lbl]) if pd.notna(b) and str(b) != a}
        err = None if mapping else _("尚未修改任何值")
        t = dict(type="recode", column=col, mapping=mapping)
    elif kind == "transpose":
        c1, c2 = st.columns(2)
        with c1:
            idx = ui.selectbox(_("作為新欄名的欄位"), [None] + cols, k + "idx", default=cols[0] if cols else None,
                               format_func=lambda c: _("（用列號）") if c is None else str(c))
        with c2:
            new_index = ui.text(_("原欄名變成的欄位名稱"), k + "newidx", default="Sample")
        t = dict(type="transpose", index=idx, new_index=new_index.strip() or "Sample")

    if t is not None and not err:
        try:
            prev = data.apply_transform(df, t)
            st.caption(_("→ 套用後：{rows:,} 列 × {cols} 欄 · {desc}", rows=prev.shape[0], cols=prev.shape[1],
                         desc=data.describe_transform(t)))
        except Exception as e:
            err = _("無法套用：{e}", e=e)
    if err:
        st.caption(f"⚠️ {err}")
    if st.button(_("＋ 加入整理步驟"), type="primary", disabled=bool(err) or t is None, key="btn:add_step"):
        store["transforms"].append(t)
        st.rerun()


def _steps(store: dict) -> None:
    steps = store["transforms"]
    st.markdown("##### " + _("目前的整理步驟"))
    if store["edited"] is not None:
        st.caption(_("● 基礎資料已用「手動編輯」的版本取代"))
    if not steps:
        st.caption(_("尚無整理步驟。每個步驟依序套用，所有頁面都會使用整理後的資料。"))
    for i, t in enumerate(steps):
        c1, c2 = st.columns([8, 1], vertical_alignment="center")
        c1.markdown(f"`{i + 1:02d}`　{data.describe_transform(t)}")
        c2.button("✕", key=f"btn:rm:{i}", on_click=_remove, args=(store, i), help=_("移除此步驟"))
    c1, c2 = st.columns(2)
    c1.button(_("復原上一步"), key="btn:undo", on_click=_undo, args=(store,), disabled=not steps, width="stretch")
    c2.button(_("還原成原始資料"), key="btn:restore", on_click=_restore, args=(store,),
              disabled=not steps and store["edited"] is None, width="stretch")


# ------------------------------------------------------------------ 主畫面
def render(ds) -> None:
    df, store = ds.df, ds.store
    nums = numeric_columns(df)
    c = st.columns(5)
    c[0].metric(_("資料列"), f"{len(df):,}")
    c[1].metric(_("欄位"), df.shape[1])
    c[2].metric(_("數值欄"), len(nums))
    c[3].metric(_("文字 / 類別欄"), df.shape[1] - len(nums))
    c[4].metric(_("缺失值"), f"{int(df.isna().sum().sum()):,}")
    if df.empty and store["edited"] is None and not store["transforms"]:
        st.info(_("目前沒有資料。請在左側上傳 Excel / CSV，或選擇範例資料。"))
        return

    t_view, t_sum, t_tf, t_edit, t_dl = st.tabs([_("預覽"), _("欄位摘要"), _("整理步驟"), _("手動編輯"), _("下載")])
    with t_view:
        st.dataframe(df, height=520, width="stretch")
    with t_sum:
        st.dataframe(i18n.table(data.summary_table(df)), hide_index=True, width="stretch")
        st.caption(_("「型別」決定欄位能用在哪裡：數值欄可做檢定與 Y 軸，類別欄可當分組。型別不對時，可用「整理步驟 → 轉換欄位型別」修正。"))
    with t_tf:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            with st.container(border=True):
                _builder(df, store, ds.id)
        with right:
            _steps(store)
            st.markdown("##### " + _("結果預覽 (前 8 列)"))
            st.dataframe(df.head(8), width="stretch", hide_index=True)
    with t_edit:
        if len(df) > EDITOR_MAX_ROWS:
            st.info(_("資料超過 {n:,} 列，請改用「整理步驟」處理。", n=EDITOR_MAX_ROWS))
        else:
            ver = store.setdefault("ed_ver", 0)
            st.caption(_("直接在表格中修改、新增或刪除列 (勾選列後按 Delete)。按「套用修改」後，目前的整理步驟會合併成新的基礎資料。"))
            edited = st.data_editor(df, key=f"ed:{ds.id}:{ver}", num_rows="dynamic", width="stretch", height=480)
            c1, c2, _c3 = st.columns([1, 1, 3])
            if c1.button(_("套用修改"), type="primary", key="btn:apply_edit", width="stretch"):
                store["edited"] = edited.reset_index(drop=True)
                store["transforms"] = []
                store["ed_ver"] = ver + 1
                st.rerun()
            c2.button(_("還原成原始資料"), key="btn:restore_edit", on_click=_restore, args=(store,),
                      disabled=store["edited"] is None and not store["transforms"], width="stretch")
    with t_dl:
        st.caption(_("下載目前 (已套用整理步驟) 的資料。"))
        stem = re.sub(r"[\\/:*?\"<>|›\s]+", "_", ds.name).strip("_") or "data"
        c1, c2, _c3 = st.columns([1, 1, 3])
        c1.download_button("CSV", df.to_csv(index=False).encode("utf-8-sig"), f"{stem}.csv", "text/csv",
                           key="dl:data:csv", on_click="ignore", width="stretch")
        c2.download_button("Excel", data.to_excel_bytes({"data": df}), f"{stem}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl:data:xlsx",
                           on_click="ignore", width="stretch")
