"""📊 繪圖頁：選圖表 → 對應欄位 → 調整選項與格式 → 即時預覽 → 下載。"""

from __future__ import annotations

import json
from dataclasses import asdict, fields

import pandas as pd
import streamlit as st
from matplotlib.colors import to_hex

from .. import export, fonts, palettes, plots
from ..annotate import CORRECTIONS, DISPLAYS, TESTS
from ..data import categorical_columns, guess_column, is_categorical_like, numeric_columns
from ..style import CONTEXTS, EDGE_OPTIONS, GRID_OPTIONS, LEGEND_LOCS, PRESETS, THEMES, Style
from ..i18n import _
from . import common as ui

STYLE_FIELDS = [f.name for f in fields(Style)]
KEEP_ON_PRESET = {"title", "xlabel", "ylabel", "legend_title", "custom_colors", "xmin", "xmax", "ymin", "ymax"}
DPI_CHOICES = [72, 96, 150, 200, 300, 600, 1200]
ANGLES = [0, 15, 30, 45, 60, 90, -30, -45, -90]


def _angle_label(a: int) -> str:
    return _("0°（水平）") if a == 0 else (_("90°（垂直）") if abs(a) == 90 else f"{a}°")


# ------------------------------------------------------------------ style state
def _sk(name: str) -> str:
    return f"s:{name}"


def _init_style() -> None:
    base = Style()
    for name in STYLE_FIELDS:
        ui.init(_sk(name), getattr(base, name))


def style_from_state() -> Style:
    _init_style()
    return Style(**{name: st.session_state[_sk(name)] for name in STYLE_FIELDS})


def _set_style(d: dict) -> None:
    for k, v in d.items():
        if k in STYLE_FIELDS:
            st.session_state[_sk(k)] = v


def _on_preset() -> None:
    base = asdict(Style())
    base.update(PRESETS.get(st.session_state["style_preset"], {}))
    _set_style({k: v for k, v in base.items() if k not in KEEP_ON_PRESET})


def _on_style_upload() -> None:
    f = st.session_state.get("style_upload")
    if f is None:
        return
    try:
        _set_style(json.loads(f.getvalue().decode("utf-8")))
        st.session_state["style_msg"] = "✅ " + _("已載入格式設定")
    except Exception as err:
        st.session_state["style_msg"] = "⚠️ " + _("無法讀取：{err}", err=err)


def _reset_style() -> None:
    _set_style(asdict(Style()))


# ------------------------------------------------------------------ 欄位對應
def _best_cat(df: pd.DataFrame, opts: list, used: list):
    """預設分組欄：優先選 2–12 個類別的欄位。"""
    for c in opts:
        if c not in used and 2 <= df[c].nunique() <= 12:
            return c
    return next((c for c in opts if c not in used), opts[0] if opts else None)


def _order_widget(df: pd.DataFrame, col: str, ds_id: str) -> list:
    lv = plots.category_levels(df[col])
    key = f"order:{ds_id}:{col}"
    ui.init(key, lv)

    def _set(v):
        st.session_state[key] = v

    sel_now = [v for v in st.session_state[key] if v in lv]
    with st.popover("↕️ " + _("「{col}」類別篩選與排序 ({n}/{t})", col=col, n=len(sel_now), t=len(lv)), width="stretch"):
        st.caption(_("刪除不需要的類別；點選順序即為顯示順序。"))
        sel = ui.multiselect(_("顯示的類別 (依序)"), lv, key, default=lv, label_visibility="collapsed")
        c1, c2, c3 = st.columns(3)
        c1.button(_("原始順序"), key=key + ":orig", on_click=_set, args=(lv,), width="stretch")
        c2.button(_("反轉"), key=key + ":rev", on_click=_set, args=(list(reversed(sel or lv)),), width="stretch")
        c3.button("A → Z", key=key + ":az", on_click=_set, args=(sorted(sel or lv, key=lambda v: str(v)),), width="stretch")
    if len(lv) > 40:
        st.caption("⚠️ " + _("「{col}」有 {n} 個類別，可能太多。", col=col, n=len(lv)))
    return sel or lv


def field_widgets(spec: plots.ChartSpec, df: pd.DataFrame, ds_id: str) -> tuple[dict, dict]:
    nums = numeric_columns(df)
    cats = categorical_columns(df)
    allc = list(df.columns)
    m: dict = {}
    orders: dict = {}
    used: list = []
    for f in spec.fields:
        key = f"f:{spec.key}:{f.key}"
        if f.kind in ("nums", "anys"):
            opts = nums if f.kind == "nums" else allc
            default = [c for c in opts if c not in used][: f.default_n or len(opts)]
            m[f.key] = ui.multiselect(_(f.label), opts, key, default=default)
            continue
        opts = {"num": nums, "cat": cats, "any": allc, "hue": cats}[f.kind]
        free = [c for c in opts if c not in used]
        guess = guess_column(free, list(f.hints)) if f.hints else None
        if f.required:
            options = opts
            if f.kind == "cat":
                default = guess or _best_cat(df, opts, used)
            else:
                default = guess or (free[0] if free else (opts[0] if opts else None))
        else:
            options = [None] + opts
            default = guess
        if not opts:
            st.warning(_("「{f}」需要數值欄位，但目前資料沒有。", f=_(f.label)) if f.kind == "num" else _("「{f}」需要欄位，但目前資料沒有。", f=_(f.label)))
            m[f.key] = None
            continue
        val = ui.selectbox(_(f.label), options, key, default=default, format_func=lambda c: _("（無）") if c is None else str(c),
                           help=_(f.help) if f.help else None)
        m[f.key] = val
        if val is not None:
            used.append(val)
            if f.order and (f.kind == "cat" or is_categorical_like(df[val])):
                orders[val] = _order_widget(df, val, ds_id)
    return m, orders


def opt_widgets(spec: plots.ChartSpec) -> dict:
    o: dict = {}
    if not spec.opts:
        st.caption(_("此圖表沒有額外選項。"))
        return o
    cols = st.columns(2)
    for i, op in enumerate(spec.opts):
        key = f"o:{spec.key}:{op.key}"
        with cols[i % 2]:
            if op.type == "bool":
                o[op.key] = ui.checkbox(_(op.label), key, default=op.default, help=_(op.help) if op.help else None)
            elif op.type == "int":
                o[op.key] = int(ui.number(_(op.label), key, default=int(op.default),
                                          min_value=int(op.min) if op.min is not None else None,
                                          max_value=int(op.max) if op.max is not None else None, step=1, help=_(op.help) if op.help else None))
            elif op.type == "float":
                o[op.key] = float(ui.number(_(op.label), key, default=float(op.default),
                                            min_value=float(op.min) if op.min is not None else None,
                                            max_value=float(op.max) if op.max is not None else None,
                                            step=float(op.step or 0.1), format="%g", help=_(op.help) if op.help else None))
            elif op.type == "select":
                o[op.key] = ui.selectbox(_(op.label), op.choices, key, default=op.default, help=_(op.help) if op.help else None,
                                         format_func=lambda v: _(v) if isinstance(v, str) else str(v))
            elif op.type == "color":
                o[op.key] = ui.color(_(op.label), key, default=op.default)
            else:
                o[op.key] = ui.text(_(op.label), key, default=op.default or "", help=_(op.help) if op.help else None)
    return o


def sig_widgets(spec: plots.ChartSpec, df: pd.DataFrame, m: dict, orders: dict) -> dict | None:
    on = ui.checkbox(_("在圖上加入顯著性標記 (括號 + 星號)"), "sig:on", default=False)
    if not on:
        st.caption(_("自動計算組間比較的 p 值並畫出括號；結果表會顯示在圖下方。"))
        return None
    hue = m.get("hue") if m.get("hue") and m.get("hue") != m.get("x") else None
    col = hue or m.get("x")
    levels = orders.get(col) or (plots.category_levels(df[col]) if col else [])
    c1, c2 = st.columns(2)
    with c1:
        test = ui.selectbox(_("檢定方法"), list(TESTS), "sig:test", default="Welch t 檢定", format_func=_)
        pairs = ui.selectbox(_("比較組合"), ["所有兩兩比較", "與參考組比較", "相鄰組比較"], "sig:pairs", format_func=_)
        ref = ui.selectbox(_("參考組 (對照組)"), levels, "sig:ref:" + str(col), default=levels[0] if levels else None,
                           disabled=pairs != "與參考組比較") if levels else None
    with c2:
        corr = ui.selectbox(_("多重比較校正"), list(CORRECTIONS), "sig:corr", default="Holm", format_func=_,
                            disabled=TESTS[test] in ("tukey", "gh"), help=_("Tukey HSD 與 Games-Howell 本身已校正"))
        display = ui.selectbox(_("顯示方式"), DISPLAYS, "sig:display", format_func=_)
        hide_ns = ui.checkbox(_("隱藏不顯著 (ns)"), "sig:hide_ns", default=False)
    c3, c4 = st.columns(2)
    ui.init("sig:fs", None)
    ui.init("sig:lw", 1.0)
    fsz = c3.number_input(_("標記字級"), min_value=5.0, max_value=30.0, step=0.5, key="sig:fs", placeholder=_("同刻度字級"))
    lw = c4.number_input(_("括號線寬"), min_value=0.2, max_value=5.0, step=0.1, key="sig:lw")
    if hue:
        st.caption(_("有 hue 時，比較的是每個「{x}」類別內的「{hue}」各組。", x=m["x"], hue=hue))
    if not spec.sig:
        return None
    return dict(enabled=True, test=TESTS[test],
                pairs_mode={"所有兩兩比較": "all", "與參考組比較": "ref", "相鄰組比較": "adjacent"}[pairs],
                ref=ref, correction=CORRECTIONS[corr], display=display, hide_ns=hide_ns,
                font_size=fsz, line_width=lw)


# ------------------------------------------------------------------ 顏色
def _color_column(spec: plots.ChartSpec, m: dict, o: dict, df: pd.DataFrame):
    col = m.get(spec.color_field) if spec.color_field else None
    if col is None and "x" in m and o.get("color_by_x", False) and any(f.key == "x" and f.kind == "cat" for f in spec.fields):
        col = m["x"]
    if col is not None and col in df and is_categorical_like(df[col]):
        return col
    return None


def _on_pick(col: str, level: str, key: str) -> None:
    st.session_state.setdefault("color_overrides", {}).setdefault(col, {})[level] = st.session_state[key]


def _reset_colors(col: str) -> None:
    st.session_state.setdefault("color_overrides", {}).pop(col, None)
    for k in [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith("cp:") and f":{col}:" in k]:
        del st.session_state[k]


def color_widgets(spec: plots.ChartSpec, df: pd.DataFrame, m: dict, o: dict, orders: dict) -> dict:
    pal = ui.selectbox(_("類別色盤"), palettes.PALETTE_NAMES, _sk("palette"), format_func=_)
    if pal == palettes.CUSTOM_LABEL:
        st.text_area(_("自訂色碼 (逗號、空白或換行分隔，例如 #1f77b4, #ff7f0e, red)"), key=_sk("custom_colors"), height=80)
    custom = st.session_state.get(_sk("custom_colors"), "")
    ui.swatches(palettes.get_palette(pal, 10, custom))
    c1, c2 = st.columns(2)
    c1.color_picker(_("單一顏色 (未分組時)"), key=_sk("single_color"))
    with c2:
        ui.selectbox(_("連續色階 (熱圖等)"), palettes.COLORMAPS, _sk("cmap"))
        st.checkbox(_("反轉色階"), key=_sk("reverse_cmap"))

    overrides = st.session_state.setdefault("color_overrides", {})
    col = _color_column(spec, m, o, df)
    if col:
        levels = orders.get(col) or plots.category_levels(df[col])
        st.markdown("**" + _("各類別顏色：{col}", col=col) + "**")
        if len(levels) > 30:
            st.caption(_("類別太多 ({n})，僅列出前 30 個。", n=len(levels)))
            levels = levels[:30]
        base = palettes.get_palette(pal, len(levels), custom)
        cols = st.columns(3)
        for i, lv in enumerate(levels):
            key = f"cp:{pal}:{col}:{lv}"
            ui.init(key, overrides.get(col, {}).get(lv) or to_hex(base[i]))
            cols[i % 3].color_picker(str(lv), key=key, on_change=_on_pick, args=(col, lv, key))
        if overrides.get(col):
            st.button(_("重設此欄位的顏色"), on_click=_reset_colors, args=(col,), key=f"reset_colors:{col}")
    return overrides


# ------------------------------------------------------------------ 格式
def style_widgets() -> None:
    _init_style()
    with st.expander("⭐ " + _("快速套用風格 / 存檔"), expanded=True):
        ui.init("style_preset", "預設")
        st.selectbox(_("風格預設"), list(PRESETS), key="style_preset", on_change=_on_preset, format_func=_,
                     help=_("一鍵套用一組格式 (不會清除標題文字)"))
        cc1, cc2 = st.columns(2)
        cc1.download_button(_("💾 下載目前格式 (JSON)"), style_from_state().to_json().encode("utf-8"), "biostat_style.json",
                            "application/json", on_click="ignore", width="stretch")
        cc2.button("↺ " + _("全部重設"), on_click=_reset_style, width="stretch")
        st.file_uploader(_("載入格式檔 (JSON)"), type=["json"], key="style_upload", on_change=_on_style_upload)
        if st.session_state.get("style_msg"):
            st.caption(st.session_state.pop("style_msg"))

    with st.expander("📐 " + _("尺寸、主題與字型")):
        c1, c2 = st.columns(2)
        c1.number_input(_("寬度 (英吋)"), min_value=1.0, max_value=40.0, step=0.5, key=_sk("width"), format="%g")
        c2.number_input(_("高度 (英吋)"), min_value=1.0, max_value=40.0, step=0.5, key=_sk("height"), format="%g")
        st.caption(_("= {w:.1f} × {h:.1f} 公分　(分面圖為每格尺寸)", w=st.session_state[_sk("width")] * 2.54,
                     h=st.session_state[_sk("height")] * 2.54))
        c1, c2 = st.columns(2)
        with c1:
            ui.selectbox(_("主題"), list(THEMES), _sk("theme"), format_func=lambda k: _(THEMES[k]))
        with c2:
            ui.selectbox(_("線條粗細比例"), list(CONTEXTS), _sk("context"), format_func=lambda k: _(CONTEXTS[k]))
        st.slider(_("整體字級縮放"), 0.5, 3.0, step=0.05, key=_sk("font_scale"))
        fonts_list = fonts.available_fonts()
        ui.selectbox(_("字型"), fonts_list, _sk("font"), default=fonts.default_font())
        st.checkbox(_("中文字自動使用中文字型 (避免出現方框)"), key=_sk("use_cjk"))
        c1, c2 = st.columns(2)
        c1.color_picker(_("背景顏色"), key=_sk("bg_color"))
        c2.checkbox(_("透明背景 (匯出)"), key=_sk("transparent"))

    with st.expander("🔤 " + _("標題與座標軸標籤")):
        st.text_input(_("圖標題"), key=_sk("title"), placeholder=_("(無)"))
        c1, c2 = st.columns(2)
        c1.text_input(_("X 軸標題"), key=_sk("xlabel"), placeholder=_("(自動：欄位名稱)"))
        c2.text_input(_("Y 軸標題"), key=_sk("ylabel"), placeholder=_("(自動：欄位名稱)"))
        c1.checkbox(_("隱藏 X 軸標題"), key=_sk("hide_xlabel"))
        c2.checkbox(_("隱藏 Y 軸標題"), key=_sk("hide_ylabel"))
        c1, c2, c3 = st.columns(3)
        c1.number_input(_("標題字級"), min_value=4.0, max_value=60.0, step=0.5, key=_sk("title_size"), format="%g")
        c2.number_input(_("軸標題字級"), min_value=4.0, max_value=60.0, step=0.5, key=_sk("label_size"), format="%g")
        c3.number_input(_("刻度字級"), min_value=4.0, max_value=60.0, step=0.5, key=_sk("tick_size"), format="%g")
        c1, c2, c3 = st.columns(3)
        c1.checkbox(_("標題粗體"), key=_sk("title_bold"))
        c2.checkbox(_("軸標題粗體"), key=_sk("label_bold"))
        with c3:
            ui.selectbox(_("標題對齊"), ["center", "left", "right"], _sk("title_loc"),
                         format_func=lambda k: _({"center": "置中", "left": "靠左", "right": "靠右"}[k]))
        c1, c2 = st.columns(2)
        with c1:
            ui.selectbox(_("X 軸文字角度"), ANGLES, _sk("xtick_rotation"), default=0, format_func=_angle_label,
                         help=_("類別名稱太長、互相重疊時，可改成 30° / 45° / 90°"))
        with c2:
            ui.selectbox(_("Y 軸文字角度"), ANGLES, _sk("ytick_rotation"), default=0, format_func=_angle_label)
        c1, c2 = st.columns(2)
        c1.number_input(_("X 軸文字換行：每行字寬"), min_value=0, max_value=60, step=1, key=_sk("xtick_wrap"),
                        help=_("超過就自動換行 (中文 1 字 = 2 字寬)；0 = 不換行"))
        c2.checkbox(_("文字仍太擠時自動旋轉 30°"), key=_sk("xtick_auto_rotate"), help=_("只在 X 軸文字角度為 0° 時作用"))
        st.color_picker(_("文字顏色"), key=_sk("text_color"))

    with st.expander("📏 " + _("座標軸範圍、刻度與框線")):
        c1, c2 = st.columns(2)
        c1.number_input(_("X 最小值"), key=_sk("xmin"), placeholder=_("自動"), format="%g")
        c2.number_input(_("X 最大值"), key=_sk("xmax"), placeholder=_("自動"), format="%g")
        c1.number_input(_("Y 最小值"), key=_sk("ymin"), placeholder=_("自動"), format="%g")
        c2.number_input(_("Y 最大值"), key=_sk("ymax"), placeholder=_("自動"), format="%g")
        c1.checkbox(_("X 軸對數刻度"), key=_sk("xlog"))
        c2.checkbox(_("Y 軸對數刻度"), key=_sk("ylog"))
        c1.checkbox(_("反轉 X 軸"), key=_sk("invert_x"))
        c2.checkbox(_("反轉 Y 軸"), key=_sk("invert_y"))
        c1, c2 = st.columns(2)
        with c1:
            ui.selectbox(_("格線"), GRID_OPTIONS, _sk("grid"), format_func=_)
        with c2:
            ui.selectbox(_("刻度方向"), ["out", "in", "inout"], _sk("tick_direction"),
                         format_func=lambda k: _({"out": "朝外", "in": "朝內", "inout": "穿越"}[k]))
        c1.number_input(_("框線/刻度粗細"), min_value=0.0, max_value=6.0, step=0.1, key=_sk("spine_width"), format="%g")
        c2.number_input(_("刻度長度"), min_value=0.0, max_value=20.0, step=0.5, key=_sk("tick_length"), format="%g")
        st.checkbox(_("移除上方與右方框線 (despine)"), key=_sk("despine"))

    with st.expander("🏷️ " + _("圖例")):
        c1, c2 = st.columns(2)
        with c1:
            ui.selectbox(_("位置"), list(LEGEND_LOCS), _sk("legend_loc"), format_func=_)
        c2.number_input(_("欄數"), min_value=1, max_value=12, step=1, key=_sk("legend_ncol"))
        st.text_input(_("圖例標題"), key=_sk("legend_title"), placeholder=_("(自動)"))
        c1, c2, c3 = st.columns(3)
        c1.checkbox(_("隱藏標題"), key=_sk("hide_legend_title"))
        c2.checkbox(_("外框"), key=_sk("legend_frame"))
        c3.number_input(_("字級"), min_value=4.0, max_value=40.0, step=0.5, key=_sk("legend_size"), format="%g")

    with st.expander("✏️ " + _("點、線與透明度")):
        st.slider(_("透明度"), 0.05, 1.0, step=0.05, key=_sk("alpha"))
        c1, c2 = st.columns(2)
        c1.number_input(_("點大小"), min_value=0.5, max_value=40.0, step=0.5, key=_sk("marker_size"), format="%g")
        c2.number_input(_("線寬"), min_value=0.1, max_value=10.0, step=0.1, key=_sk("line_width"), format="%g")
        ui.selectbox(_("點/長條邊框顏色"), list(EDGE_OPTIONS), _sk("edge_color"), format_func=_)


# ------------------------------------------------------------------ 主畫面
def _store_fig(res) -> None:
    old = st.session_state.get("plot_res")
    if old is not None and old is not res:
        try:
            export.close(old.fig)
        except Exception:
            pass
    st.session_state["plot_res"] = res


def render(ds) -> None:
    df = ds.df
    if df.empty:
        st.info(_("目前沒有資料。請在左側上傳 Excel / CSV，或選擇範例資料。"))
        return
    left, right = st.columns([1.0, 1.55], gap="large")
    with left:
        c1, c2 = st.columns([1, 1.6])
        with c1:
            cat = ui.selectbox(_("圖表類別"), plots.CATEGORIES, "plot:cat", default="類別比較", format_func=_)
        with c2:
            keys = plots.charts_in(cat)
            key = ui.selectbox(_("圖表類型"), keys, f"plot:chart:{cat}", format_func=lambda k: _(plots.CHARTS[k].name))
        spec = plots.CHARTS[key]
        st.caption(_(spec.description))

        t_map, t_opt, t_fmt, t_col = st.tabs(["📌 " + _("資料對應"), "⚙️ " + _("圖表選項"), "🎨 " + _("格式"), "🖌️ " + _("顏色")])
        with t_map:
            m, orders = field_widgets(spec, df, ds.id)
            facet, wrap, share_y = None, 3, True
            if spec.facet and not spec.figure_level:
                st.divider()
                facet = ui.selectbox(_("分面：依類別拆成多張子圖 (選填)"), [None] + categorical_columns(df), f"facet:{spec.key}",
                                     format_func=lambda c: _("（不分面）") if c is None else str(c))
                if facet:
                    c1, c2 = st.columns(2)
                    ui.init("facet:wrap", 3)
                    wrap = int(c1.number_input(_("每列幾張"), min_value=1, max_value=8, step=1, key="facet:wrap"))
                    with c2:
                        share_y = ui.checkbox(_("共用 Y 軸"), "facet:sharey", default=True)
                    orders[facet] = _order_widget(df, facet, ds.id)
        with t_opt:
            o = opt_widgets(spec)
            sig = None
            if spec.sig:
                st.divider()
                st.markdown(_("**顯著性標記**"))
                sig = sig_widgets(spec, df, m, orders)
        with t_fmt:
            style_widgets()
        with t_col:
            overrides = color_widgets(spec, df, m, o, orders)

    style = style_from_state()
    with right:
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            auto = ui.checkbox(_("即時更新"), "plot:auto", default=True, help=_("大量資料或複雜圖表時可關閉，改按「更新圖表」"))
        clicked = c2.button("🔄 " + _("更新圖表"), type="primary", disabled=auto, width="stretch")
        area = st.container()
        need = auto or clicked or st.session_state.get("plot_res") is None
        if need:
            try:
                with st.spinner(_("繪圖中…")):
                    res = plots.render(spec.key, df, m, o, style, orders=orders, facet=facet, facet_wrap=wrap, share_y=share_y,
                                       sig=sig, overrides=overrides)
                _store_fig(res)
            except Exception as err:
                with area:
                    st.error(_("無法繪製：{err}", err=err))
                    with st.expander(_("錯誤細節")):
                        st.exception(err)
                return
        res = st.session_state["plot_res"]

        st.markdown(_("##### ⬇️ 下載圖檔"))
        d1, d2, d3 = st.columns([1.3, 1, 1])
        with d1:
            fmt = ui.selectbox(_("格式"), list(export.FORMATS), "exp:fmt", default="PNG", format_func=_)
        with d2:
            ui.selectbox(_("解析度 (DPI)"), DPI_CHOICES, _sk("dpi"), default=300)
        with d3:
            ui.selectbox(_("預覽解析度"), [72, 96, 110, 150, 200], _sk("preview_dpi"), default=110)
        st.checkbox(_("自動裁切空白邊界 (tight)"), key=_sk("tight"))
        st.checkbox(_("向量檔文字保持可編輯 (Illustrator / Inkscape)"), key=_sk("editable_text"),
                    help=_("PDF/EPS 嵌入 TrueType 字型、SVG 保留文字物件；變更後需重新繪製"))
        style = style_from_state()
        try:
            prev, _ext, _mime = export.fig_bytes(res.fig, "PNG", dpi=style.preview_dpi, rc=res.rc, tight=style.tight)
            content, ext, mime = export.fig_bytes(res.fig, fmt, dpi=style.dpi, rc=res.rc, tight=style.tight,
                                                  transparent=style.transparent)
        except Exception as err:
            area.error(_("輸出圖檔失敗：{err}", err=err))
            return
        with area:
            st.image(prev, width="content")
        w_px, h_px = res.fig.get_size_inches() * style.dpi
        fname = st.session_state.get("exp:name") or spec.key
        st.download_button("⬇️ " + _("下載 {ext} ({kb:,.0f} KB)", ext=ext.upper(), kb=len(content) / 1024), content, f"{fname}.{ext}", mime,
                           type="primary", width="stretch", on_click="ignore")
        st.caption(_("畫布約 {w:,.0f} × {h:,.0f} 像素 @ {dpi} DPI (裁切前)", w=w_px, h=h_px, dpi=style.dpi))
        ui.text(_("檔名"), "exp:name", default="", placeholder=spec.key)

        if res.notes:
            for n in res.notes:
                st.info(_(n))
        if res.tables:
            st.markdown(_("##### 📑 圖表相關數據"))
            ui.show_tables(res.tables, spec.key)
            st.download_button("⬇️ " + _("下載全部結果 (Excel)"), ui.tables_excel(res.tables), f"{spec.key}_results.xlsx",
                               on_click="ignore", key="dl_all_tables")
