"""圖表註冊表與繪圖函式。

每種圖表以 ChartSpec 描述：需要哪些欄位 (Field)、有哪些圖表專屬選項 (Opt)、以及繪圖函式。
介面依據 ChartSpec 自動產生控制項，因此新增圖表只需在本檔註冊一個函式。

繪圖函式有兩種：
- 軸層級 (axes-level)：draw(ax, df, m, o, ctx)，框架負責建立畫布、分面 (facet) 與顯著性標記。
- 圖層級 (figure-level)：draw(df, m, o, ctx) -> PlotResult，自行建立整張圖 (例如 pairplot、clustermap)。
"""

from __future__ import annotations

import math
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import matplotlib

try:
    matplotlib.use("Agg")
except Exception:  # 已有其他後端 (例如在 Jupyter) 時維持原狀
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, Ellipse, Patch
from scipy import stats as sps

from . import annotate, fonts
from .palettes import DIVERGING_CMAPS
from . import stats as bstats
from .data import as_category_labels, is_categorical_like, natural_key
from .style import Style, build_rc, finalize

PLOT_LOCK = threading.RLock()


# ============================================================ 規格
@dataclass
class Field:
    key: str
    label: str
    kind: str = "num"  # num | cat | any | hue | nums | anys
    required: bool = True
    help: str = ""
    hints: tuple = ()
    order: bool = False
    default_n: int = 0


@dataclass
class Opt:
    key: str
    label: str
    type: str  # bool | int | float | select | text | color
    default: Any = None
    choices: list | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    help: str = ""


@dataclass
class ChartSpec:
    key: str
    name: str
    category: str
    description: str
    fields: list
    opts: list
    draw: Callable
    figure_level: bool = False
    facet: bool = True
    sig: bool = False
    axes_style: bool = True
    despine: bool = True
    uses_cmap: bool = False
    color_field: str | None = "hue"  # 哪個欄位決定顏色 (供「各類別顏色」設定)


@dataclass
class PlotResult:
    fig: Any
    axes: list
    grid: Any = None
    notes: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    title_on_fig: bool | None = None
    rc: dict = field(default_factory=dict)


CHARTS: dict[str, ChartSpec] = {}
CATEGORIES = ["分布", "類別比較", "關係與迴歸", "矩陣與熱圖", "多變量", "生醫專用", "組成與其他"]


def register(key, name, category, description, fields, opts=(), **kw):
    def deco(fn):
        CHARTS[key] = ChartSpec(key, name, category, description, list(fields), list(opts), fn, **kw)
        return fn

    return deco


class Ctx:
    """繪圖時的共用工具：色盤、尺寸、備註與附帶結果表。"""

    def __init__(self, style: Style, overrides: dict | None = None):
        self.s = style
        self.overrides = overrides or {}
        self.notes: list[str] = []
        self.tables: list = []
        self.sig_width = 0.8

    def palette_for(self, col: str | None, lvls: list) -> dict:
        cols = self.s.colors(len(lvls))
        pal = {lv: cols[i] for i, lv in enumerate(lvls)}
        for k, v in self.overrides.get(col, {}).items():
            if k in pal and v:
                pal[k] = v
        return pal

    def subplots(self, **kw):
        figsize = kw.pop("figsize", (self.s.width, self.s.height))
        return plt.subplots(figsize=figsize, **kw)

    @property
    def ms2(self) -> float:
        return float(self.s.marker_size) ** 2

    def edge(self, default=None):
        e = self.s.edge
        return default if e is None else e

    def note(self, text: str) -> None:
        self.notes.append(text)

    def table(self, name: str, df: pd.DataFrame) -> None:
        self.tables.append((name, df))


def category_levels(s: pd.Series) -> list[str]:
    """分組欄的類別 (字串) 清單：數值型依大小排序，文字型依出現順序。"""
    labels = as_category_labels(s)
    if pd.api.types.is_numeric_dtype(s):
        order = s.dropna().drop_duplicates().sort_values()
        return [lab for lab in dict.fromkeys(as_category_labels(order).tolist())]
    return list(dict.fromkeys(labels.dropna().tolist()))


_OFFSETS = [(5, 4), (5, -11), (-5, 4), (-5, -11), (9, 14), (-9, 14), (9, -20), (-9, -20), (14, 0), (-14, 0),
            (12, 26), (-12, 26), (12, -30), (-12, -30), (0, 34), (0, -38)]


def label_points(ax, xs, ys, texts, fontsize: float, color: str = "#222222") -> None:
    """標註點名稱並以貪婪法避開彼此重疊 (簡易版 repel)。離點較遠的標籤會加上細引線。"""
    renderer = ax.figure.canvas.get_renderer()
    placed = []
    for x, y, txt in zip(xs, ys, texts):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        choice = _OFFSETS[0]
        for dx, dy in _OFFSETS:
            t = ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", ha="left" if dx >= 0 else "right",
                            va="bottom", fontsize=fontsize)
            bb = t.get_window_extent(renderer).expanded(1.04, 1.08)
            t.remove()
            if not any(bb.overlaps(p) for p in placed):
                choice = (dx, dy)
                placed.append(bb)
                break
        dx, dy = choice
        far = abs(dx) > 6 or abs(dy) > 12
        ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", ha="left" if dx >= 0 else "right",
                    va="bottom", fontsize=fontsize, color=color,
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5, shrinkA=0, shrinkB=2) if far else None)


def _pf(p: float) -> str:
    """圖上的 p 值文字：p = 0.012 / p < 0.001。"""
    return annotate.fmt_p(p, 3)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ============================================================ 共用選項
ORIENT = Opt("orient", "方向", "select", "垂直", ["垂直", "水平"])
COLOR_BY_X = Opt("color_by_x", "每個類別不同顏色 (未設定 hue 時)", "bool", True)
POINTS = Opt("points", "疊加資料點", "select", "無", ["無", "散點 (strip)", "蜂群 (swarm)"])
POINTS_ON = Opt("points", "疊加資料點", "select", "散點 (strip)", ["無", "散點 (strip)", "蜂群 (swarm)"])
POINT_SIZE = Opt("point_size", "資料點大小", "float", 4.0, min=1.0, max=15.0, step=0.5)
POINT_COLOR = Opt("point_color", "資料點顏色", "select", "黑色", ["黑色", "灰色", "同組色"])
JITTER = Opt("jitter", "抖動寬度 (jitter)", "float", 0.15, min=0.0, max=0.45, step=0.01)
WIDTH = Opt("width", "寬度", "float", 0.7, min=0.1, max=1.0, step=0.05)
ERRORBAR = Opt("errorbar", "誤差線", "select", "SEM (標準誤)", ["SEM (標準誤)", "SD (標準差)", "95% CI", "無"])
CAPSIZE = Opt("capsize", "誤差線端帽寬度", "float", 0.2, min=0.0, max=0.6, step=0.05)
ESTIMATOR = Opt("estimator", "集中趨勢", "select", "平均值", ["平均值", "中位數"])
SUMMARY = Opt("summary", "疊加摘要線", "select", "無", ["無", "平均 ± SD", "平均 ± SEM", "平均 ± 95% CI", "中位數 ± IQR"])
SUMMARY_COLOR = Opt("summary_color", "摘要線顏色", "color", "#000000")

CAT_FIELDS = [
    Field("x", "X 軸 (分組)", "cat", order=True, hints=("group", "組", "treat", "condition", "type")),
    Field("y", "Y 軸 (數值)", "num"),
    Field("hue", "顏色分組 (hue)", "cat", required=False, order=True),
]


def _errorbar(o):
    return {"SEM (標準誤)": "se", "SD (標準差)": "sd", "95% CI": ("ci", 95), "無": None}.get(o.get("errorbar"), "se")


def _estimator(o):
    return "median" if o.get("estimator") == "中位數" else "mean"


def _horiz(o) -> bool:
    return o.get("orient") == "水平"


def cat_kwargs(df, m, o, ctx, *, color_by_default=True) -> dict:
    x, y, hue = m["x"], m["y"], m.get("hue")
    kw: dict = dict(data=df, order=m.get("x_order"))
    if _horiz(o):
        kw.update(x=y, y=x, orient="h")
    else:
        kw.update(x=x, y=y)
    if hue and hue != x:
        kw.update(hue=hue, hue_order=m["hue_order"], palette=ctx.palette_for(hue, m["hue_order"]))
    elif o.get("color_by_x", color_by_default):
        kw.update(hue=x, hue_order=m["x_order"], palette=ctx.palette_for(x, m["x_order"]), legend=False)
    else:
        kw.update(color=ctx.s.single_color)
    return kw


def overlay_points(ax, df, m, o, ctx) -> None:
    kind = o.get("points", "無")
    if kind == "無":
        return
    x, y, hue = m["x"], m["y"], m.get("hue")
    mode = o.get("point_color", "黑色")
    fixed = {"黑色": "#222222", "灰色": "#7a7a7a"}.get(mode)
    kw: dict = dict(data=df, order=m["x_order"], ax=ax, size=o.get("point_size", 4.0), alpha=min(ctx.s.alpha + 0.1, 1.0),
                    legend=False, zorder=3)
    if _horiz(o):
        kw.update(x=y, y=x, orient="h")
    else:
        kw.update(x=x, y=y)
    if hue and hue != x:
        pal = {lv: fixed for lv in m["hue_order"]} if fixed else ctx.palette_for(hue, m["hue_order"])
        kw.update(hue=hue, hue_order=m["hue_order"], palette=pal, dodge=True)
    elif not fixed and o.get("color_by_x", True):
        kw.update(hue=x, hue_order=m["x_order"], palette=ctx.palette_for(x, m["x_order"]))
    else:
        kw["color"] = fixed or ctx.s.single_color
    if not fixed:
        kw.update(edgecolor="#222222", linewidth=0.6)
    if kind.startswith("蜂群"):
        sns.swarmplot(**kw)
    else:
        sns.stripplot(jitter=o.get("jitter", 0.15), **kw)


def draw_summary(ax, df, m, o, ctx) -> None:
    mode = o.get("summary", "無")
    if mode == "無":
        return
    x, y, hue = m["x"], m["y"], m.get("hue")
    use_hue = bool(hue) and hue != x
    pos = annotate.category_positions(m["x_order"], m["hue_order"] if use_hue else None, 0.8, dodge=True)
    half = 0.22 * (0.8 / len(m["hue_order"]) if use_hue else 1.0) * 1.6
    color = o.get("summary_color", "#000000")
    lw = ctx.s.line_width * 1.3
    for (xl, hl), p in pos.items():
        sel = df[x] == xl
        if use_hue:
            sel &= df[hue] == hl
        v = _num(df.loc[sel, y]).dropna().to_numpy()
        if len(v) == 0:
            continue
        if mode.startswith("中位數"):
            c = np.median(v)
            lo, hi = np.percentile(v, [25, 75])
        else:
            c = v.mean()
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            e = {"平均 ± SD": sd, "平均 ± SEM": sd / np.sqrt(len(v)),
                 "平均 ± 95% CI": sps.t.ppf(0.975, max(len(v) - 1, 1)) * sd / np.sqrt(len(v))}[mode]
            lo, hi = c - e, c + e
        if _horiz(o):
            ax.vlines(c, p - half, p + half, color=color, lw=lw * 1.4, zorder=6)
            ax.errorbar(c, p, xerr=[[c - lo], [hi - c]], color=color, lw=lw, capsize=4, zorder=6)
        else:
            ax.hlines(c, p - half, p + half, color=color, lw=lw * 1.4, zorder=6)
            ax.errorbar(p, c, yerr=[[c - lo], [hi - c]], color=color, lw=lw, capsize=4, zorder=6)


# ============================================================ 分布
@register("hist", "📊 直方圖 Histogram", "分布", "數值分布；可疊加 KDE、分組堆疊或並排。",
          [Field("x", "數值變項", "num"), Field("hue", "顏色分組 (hue)", "cat", required=False, order=True)],
          [Opt("bins", "組數 (0 = 自動)", "int", 0, min=0, max=200),
           Opt("stat", "Y 軸統計量", "select", "count", ["count", "frequency", "density", "probability", "percent"]),
           Opt("kde", "疊加 KDE 曲線", "bool", False),
           Opt("element", "外觀", "select", "bars", ["bars", "step", "poly"]),
           Opt("multiple", "多組呈現", "select", "layer", ["layer", "dodge", "stack", "fill"]),
           Opt("fill", "填色", "bool", True),
           Opt("cumulative", "累積", "bool", False),
           Opt("common_norm", "各組共同標準化", "bool", False),
           Opt("rug", "底部加 rug 標記", "bool", False)])
def draw_hist(ax, df, m, o, ctx):
    kw = dict(data=df, x=m["x"], ax=ax, bins=o["bins"] if o["bins"] > 0 else "auto", stat=o["stat"], kde=o["kde"],
              element=o["element"], fill=o["fill"], cumulative=o["cumulative"], alpha=ctx.s.alpha,
              edgecolor=ctx.edge("white"), linewidth=max(ctx.s.line_width * 0.5, 0.3))
    if m.get("hue"):
        kw.update(hue=m["hue"], hue_order=m["hue_order"], palette=ctx.palette_for(m["hue"], m["hue_order"]),
                  multiple=o["multiple"], common_norm=o["common_norm"])
    else:
        kw["color"] = ctx.s.single_color
    sns.histplot(**kw)
    if o["rug"]:
        sns.rugplot(data=df, x=m["x"], hue=m.get("hue"), hue_order=m.get("hue_order"), ax=ax, legend=False,
                    palette=ctx.palette_for(m["hue"], m["hue_order"]) if m.get("hue") else None,
                    color=None if m.get("hue") else ctx.s.single_color)


@register("kde", "🌊 核密度圖 KDE", "分布", "平滑的機率密度曲線。",
          [Field("x", "數值變項", "num"), Field("hue", "顏色分組 (hue)", "cat", required=False, order=True)],
          [Opt("fill", "填色", "bool", True), Opt("bw_adjust", "平滑度 (bw_adjust)", "float", 1.0, min=0.1, max=5.0, step=0.1),
           Opt("multiple", "多組呈現", "select", "layer", ["layer", "stack", "fill"]),
           Opt("common_norm", "各組共同標準化", "bool", False), Opt("cumulative", "累積", "bool", False),
           Opt("cut", "延伸範圍 (cut)", "float", 3.0, min=0.0, max=5.0, step=0.5), Opt("rug", "底部加 rug 標記", "bool", False)])
def draw_kde(ax, df, m, o, ctx):
    kw = dict(data=df, x=m["x"], ax=ax, fill=o["fill"], bw_adjust=o["bw_adjust"], cumulative=o["cumulative"],
              cut=o["cut"], alpha=ctx.s.alpha * 0.6 if o["fill"] else ctx.s.alpha, linewidth=ctx.s.line_width)
    if m.get("hue"):
        kw.update(hue=m["hue"], hue_order=m["hue_order"], palette=ctx.palette_for(m["hue"], m["hue_order"]),
                  multiple=o["multiple"], common_norm=o["common_norm"])
    else:
        kw["color"] = ctx.s.single_color
    sns.kdeplot(**kw)
    if o["rug"]:
        sns.rugplot(data=df, x=m["x"], hue=m.get("hue"), hue_order=m.get("hue_order"), ax=ax, legend=False,
                    palette=ctx.palette_for(m["hue"], m["hue_order"]) if m.get("hue") else None,
                    color=None if m.get("hue") else ctx.s.single_color)


@register("ecdf", "📈 累積分布圖 ECDF", "分布", "經驗累積分布函數，適合比較多組分布。",
          [Field("x", "數值變項", "num"), Field("hue", "顏色分組 (hue)", "cat", required=False, order=True)],
          [Opt("stat", "Y 軸", "select", "proportion", ["proportion", "percent", "count"]),
           Opt("complementary", "互補 (1 − ECDF)", "bool", False)])
def draw_ecdf(ax, df, m, o, ctx):
    kw = dict(data=df, x=m["x"], ax=ax, stat=o["stat"], complementary=o["complementary"], linewidth=ctx.s.line_width)
    if m.get("hue"):
        kw.update(hue=m["hue"], hue_order=m["hue_order"], palette=ctx.palette_for(m["hue"], m["hue_order"]))
    else:
        kw["color"] = ctx.s.single_color
    sns.ecdfplot(**kw)


@register("qq", "📐 Q-Q 常態機率圖", "分布", "檢查資料是否符合常態分布；點越貼近直線越接近常態。",
          [Field("y", "數值變項", "num"), Field("hue", "顏色分組 (hue)", "cat", required=False, order=True)],
          [Opt("standardize", "標準化 (Z-score) 後比較", "bool", True), Opt("show_line", "顯示參考線", "bool", True),
           Opt("show_p", "圖例顯示 Shapiro-Wilk p 值", "bool", True)])
def draw_qq(ax, df, m, o, ctx):
    groups = m.get("hue_order") if m.get("hue") else [None]
    pal = ctx.palette_for(m.get("hue"), groups) if m.get("hue") else {None: ctx.s.single_color}
    for g in groups:
        x = _num(df[m["y"]] if g is None else df.loc[df[m["hue"]] == g, m["y"]]).dropna().to_numpy()
        if len(x) < 3:
            continue
        if o["standardize"]:
            x = (x - x.mean()) / x.std(ddof=1)
        (osm, osr), (slope, inter, _) = sps.probplot(x, dist="norm")
        label = "樣本" if g is None else str(g)
        if o["show_p"]:
            label += f" (Shapiro {_pf(sps.shapiro(x).pvalue)})"
        ax.scatter(osm, osr, s=ctx.ms2, color=pal[g], alpha=ctx.s.alpha, edgecolors=ctx.edge("none"), label=label, zorder=3)
        if o["show_line"]:
            ax.plot(osm, slope * osm + inter, color=pal[g] if g is not None else "#e34948", lw=ctx.s.line_width)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Standardized sample quantiles" if o["standardize"] else "Sample quantiles")
    ax.legend()


# ============================================================ 類別比較
@register("box", "📦 盒鬚圖 Box plot", "類別比較", "中位數、四分位距與離群值；可疊加資料點與顯著性標記。",
          CAT_FIELDS,
          [ORIENT, COLOR_BY_X, POINTS, POINT_SIZE, POINT_COLOR, JITTER, WIDTH,
           Opt("notch", "缺口 (notch，中位數 95% CI)", "bool", False), Opt("showmeans", "顯示平均值標記", "bool", False),
           Opt("showfliers", "顯示離群值 (疊加資料點時自動隱藏)", "bool", True),
           Opt("whis", "鬚長 (× IQR)", "float", 1.5, min=0.5, max=5.0, step=0.5),
           Opt("fill", "填色", "bool", True), Opt("gap", "hue 組間間隙", "float", 0.1, min=0.0, max=0.5, step=0.05)],
          sig=True)
def draw_box(ax, df, m, o, ctx):
    kw = cat_kwargs(df, m, o, ctx)
    line = {"黑色": "black", "深灰": "#333333", "白色": "white"}.get(ctx.s.edge_color, "auto")
    sns.boxplot(**kw, ax=ax, width=o["width"], notch=o["notch"], showmeans=o["showmeans"],
                meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=ctx.s.marker_size * 0.8),
                showfliers=o["showfliers"] and o["points"] == "無", whis=o["whis"], linewidth=ctx.s.line_width,
                fill=o["fill"], gap=o["gap"] if m.get("hue") and m["hue"] != m["x"] else 0, saturation=1, linecolor=line,
                fliersize=ctx.s.marker_size * 0.7)
    overlay_points(ax, df, m, o, ctx)


@register("violin", "🎻 小提琴圖 Violin plot", "類別比較", "分布形狀 + 盒鬚資訊；可分半比較兩組。",
          CAT_FIELDS,
          [ORIENT, COLOR_BY_X, POINTS, POINT_SIZE, POINT_COLOR, JITTER,
           Opt("inner", "內部標示", "select", "box", ["box", "quart", "point", "stick", "無"]),
           Opt("split", "分半顯示 (需 hue)", "bool", False), Opt("cut", "延伸 (cut；0 = 不超過資料範圍)", "float", 0.0, min=0.0, max=3.0, step=0.5),
           Opt("density_norm", "寬度依據", "select", "area", ["area", "count", "width"]),
           Opt("bw_adjust", "平滑度", "float", 1.0, min=0.2, max=3.0, step=0.1), WIDTH, Opt("fill", "填色", "bool", True)],
          sig=True)
def draw_violin(ax, df, m, o, ctx):
    kw = cat_kwargs(df, m, o, ctx)
    inner = None if o["inner"] == "無" else o["inner"]
    sns.violinplot(**kw, ax=ax, inner=inner, split=o["split"] and bool(m.get("hue")), cut=o["cut"], density_norm=o["density_norm"],
                   bw_adjust=o["bw_adjust"], width=o["width"] if o["width"] != 0.7 else 0.8, linewidth=ctx.s.line_width,
                   fill=o["fill"], saturation=1)
    overlay_points(ax, df, m, o, ctx)


@register("bar", "📊 長條圖 (平均 ± 誤差)", "類別比較", "各組平均 (或中位數) 加誤差線，生醫論文最常見；可疊加個別資料點。",
          CAT_FIELDS,
          [ORIENT, COLOR_BY_X, ESTIMATOR, ERRORBAR, CAPSIZE, POINTS_ON, POINT_SIZE, POINT_COLOR, JITTER, WIDTH,
           Opt("fill", "填色", "bool", True), Opt("err_color", "誤差線顏色", "color", "#222222"),
           Opt("bar_alpha", "長條透明度", "float", 0.9, min=0.1, max=1.0, step=0.05)],
          sig=True)
def draw_bar(ax, df, m, o, ctx):
    kw = cat_kwargs(df, m, o, ctx)
    sns.barplot(**kw, ax=ax, estimator=_estimator(o), errorbar=_errorbar(o), capsize=o["capsize"],
                err_kws={"linewidth": ctx.s.line_width, "color": o["err_color"]}, width=o["width"], fill=o["fill"],
                saturation=1, alpha=o["bar_alpha"], edgecolor=ctx.edge(None), linewidth=ctx.s.line_width * 0.8)
    overlay_points(ax, df, m, o, ctx)


@register("point", "🔵 點圖 Point plot (平均 ± 誤差)", "類別比較", "以點與誤差線呈現各組平均，可連線呈現趨勢 (如交互作用圖)。",
          CAT_FIELDS,
          [ORIENT, Opt("color_by_x", "每個類別不同顏色 (未設定 hue 時)", "bool", False), ESTIMATOR, ERRORBAR, CAPSIZE,
           Opt("join", "連線", "bool", True), Opt("marker", "標記", "select", "o", ["o", "s", "D", "^", "v", "X", "P", "*"]),
           Opt("dodge", "hue 錯開距離", "float", 0.3, min=0.0, max=0.8, step=0.05)],
          sig=True)
def draw_point(ax, df, m, o, ctx):
    kw = cat_kwargs(df, m, o, ctx, color_by_default=False)
    has_hue = bool(m.get("hue")) and m["hue"] != m["x"]
    sns.pointplot(**kw, ax=ax, estimator=_estimator(o), errorbar=_errorbar(o), capsize=o["capsize"], markers=o["marker"],
                  linestyles="-" if o["join"] else "none", dodge=o["dodge"] if has_hue else False,
                  err_kws={"linewidth": ctx.s.line_width}, markersize=ctx.s.marker_size, linewidth=ctx.s.line_width)
    if has_hue and len(m["hue_order"]) > 1:
        # seaborn pointplot 以 linspace(-d/2, d/2, n) 錯開；換算成與長條圖相同的寬度定義
        n = len(m["hue_order"])
        ctx.sig_width = o["dodge"] * n / (n - 1)


@register("strip", "⚫ 散點條圖 Strip / Dot plot", "類別比較", "顯示每一筆資料點，可加平均 ± 誤差線 (dot plot)。",
          CAT_FIELDS,
          [ORIENT, COLOR_BY_X, JITTER, POINT_SIZE, SUMMARY, SUMMARY_COLOR], sig=True)
def draw_strip(ax, df, m, o, ctx):
    kw = cat_kwargs(df, m, o, ctx)
    sns.stripplot(**kw, ax=ax, jitter=o["jitter"], size=o["point_size"], dodge=bool(m.get("hue")) and m["hue"] != m["x"],
                  alpha=ctx.s.alpha, edgecolor=ctx.edge("#333333"), linewidth=0.5 if ctx.edge("#333333") != "none" else 0)
    draw_summary(ax, df, m, o, ctx)



@register("swarm", "🐝 蜂群圖 Swarm plot", "類別比較", "資料點不重疊排列，清楚呈現分布與樣本數。",
          CAT_FIELDS,
          [ORIENT, COLOR_BY_X, POINT_SIZE, SUMMARY, SUMMARY_COLOR], sig=True)
def draw_swarm(ax, df, m, o, ctx):
    kw = cat_kwargs(df, m, o, ctx)
    sns.swarmplot(**kw, ax=ax, size=o["point_size"], dodge=bool(m.get("hue")) and m["hue"] != m["x"], alpha=ctx.s.alpha,
                  edgecolor=ctx.edge("#333333"), linewidth=0.5 if ctx.edge("#333333") != "none" else 0)
    draw_summary(ax, df, m, o, ctx)


@register("boxen", "🧱 增強盒鬚圖 Boxen (letter-value)", "類別比較", "大樣本時呈現更多分位數資訊。",
          CAT_FIELDS, [ORIENT, COLOR_BY_X, WIDTH, Opt("k_depth", "分位深度", "select", "tukey", ["tukey", "proportion", "trustworthy", "full"]),
                       Opt("showfliers", "顯示離群值", "bool", True)], sig=True)
def draw_boxen(ax, df, m, o, ctx):
    kw = cat_kwargs(df, m, o, ctx)
    sns.boxenplot(**kw, ax=ax, width=o["width"], k_depth=o["k_depth"], showfliers=o["showfliers"], saturation=1,
                  linewidth=ctx.s.line_width * 0.6)


@register("raincloud", "🌧️ 雨雲圖 Raincloud plot", "類別比較", "半小提琴 + 盒鬚 + 資料點，同時呈現分布、摘要與原始資料。",
          [CAT_FIELDS[0], CAT_FIELDS[1]],
          [ORIENT, Opt("bw", "平滑度", "float", 1.0, min=0.2, max=3.0, step=0.1), Opt("cloud_width", "雲寬度", "float", 0.4, min=0.1, max=0.8, step=0.05),
           Opt("box_width", "盒寬", "float", 0.12, min=0.03, max=0.4, step=0.01), POINT_SIZE, JITTER,
           Opt("cloud_alpha", "雲透明度", "float", 0.6, min=0.1, max=1.0, step=0.05)],
          sig=True, color_field="x")
def draw_raincloud(ax, df, m, o, ctx):
    x, y = m["x"], m["y"]
    order = m["x_order"]
    pal = ctx.palette_for(x, order)
    horiz = _horiz(o)
    rng = np.random.default_rng(0)
    for i, lv in enumerate(order):
        v = _num(df.loc[df[x] == lv, y]).dropna().to_numpy()
        if len(v) == 0:
            continue
        c = pal[lv]
        # 垂直時：雲在右 (+)、點在左 (−)。水平時把偏移量鏡射 (2i − pos)，搭配反轉的 Y 軸讓雲在上方。
        mirror = (lambda pos: 2 * i - pos) if horiz else (lambda pos: pos)
        if len(v) >= 2 and np.ptp(v) > 0:
            kde = sps.gaussian_kde(v)
            kde.set_bandwidth(kde.factor * o["bw"])
            grid = np.linspace(v.min(), v.max(), 200)
            dens = kde(grid) / kde(grid).max() * o["cloud_width"]
            base = i + o["box_width"] / 2 + 0.03
            if horiz:
                ax.fill_between(grid, mirror(base), mirror(base + dens), color=c, alpha=o["cloud_alpha"], lw=0)
            else:
                ax.fill_betweenx(grid, base, base + dens, color=c, alpha=o["cloud_alpha"], lw=0)
        ax.boxplot(v, positions=[i], widths=o["box_width"], vert=not horiz, patch_artist=True, showfliers=False,
                   manage_ticks=False, medianprops=dict(color="black", lw=ctx.s.line_width),
                   boxprops=dict(facecolor=c, edgecolor="black", lw=ctx.s.line_width * 0.8),
                   whiskerprops=dict(lw=ctx.s.line_width * 0.8), capprops=dict(lw=ctx.s.line_width * 0.8))
        jit = rng.uniform(-o["jitter"], o["jitter"], len(v)) * 0.6
        p = mirror(i - o["box_width"] / 2 - 0.12 + jit)
        px, py = (v, p) if horiz else (p, v)
        ax.scatter(px, py, s=o["point_size"] ** 2, color=c, alpha=ctx.s.alpha, edgecolors=ctx.edge("#333333"), lw=0.4, zorder=3)
    ticks = list(range(len(order)))
    if horiz:
        ax.set_yticks(ticks, [str(t) for t in order])
        ax.set_ylim(len(order) - 0.4, -0.7)
        ax.set_xlabel(y)
        ax.set_ylabel(x)
    else:
        ax.set_xticks(ticks, [str(t) for t in order])
        ax.set_xlim(-0.7, len(order) - 0.4 + o["cloud_width"] * 0.6)
        ax.set_xlabel(x)
        ax.set_ylabel(y)


@register("paired", "🔗 配對前後連線圖 Before–After", "類別比較", "同一受試者兩個時間點/條件的配對資料 (每列一位受試者、兩個數值欄)，以連線呈現個體變化並附配對檢定。",
          [Field("a", "條件 1 (例如 Before)", "num", hints=("before", "pre", "baseline", "前")),
           Field("b", "條件 2 (例如 After)", "num", hints=("after", "post", "後")),
           Field("hue", "連線顏色分組", "cat", required=False, order=True)],
          [Opt("test", "配對檢定", "select", "配對 t 檢定", ["配對 t 檢定", "Wilcoxon 符號等級", "不顯示"]),
           Opt("box", "疊加盒鬚圖", "bool", False), Opt("mean_line", "疊加平均線", "bool", True),
           Opt("line_alpha", "個體連線透明度", "float", 0.5, min=0.05, max=1.0, step=0.05), POINT_SIZE,
           Opt("color_change", "依增減上色 (增加/減少)", "bool", False)],
          color_field="hue")
def draw_paired(ax, df, m, o, ctx):
    a, b, hue = m["a"], m["b"], m.get("hue")
    d = df.dropna(subset=[a, b])
    pal = ctx.palette_for(hue, m["hue_order"]) if hue else {}
    up_c, down_c = "#e34948", "#2a78d6"
    for _, r in d.iterrows():
        if o["color_change"]:
            c = up_c if r[b] > r[a] else down_c
        else:
            c = pal.get(r[hue], "#888888") if hue else "#888888"
        ax.plot([0, 1], [r[a], r[b]], color=c, alpha=o["line_alpha"], lw=ctx.s.line_width * 0.8, marker="o",
                ms=o["point_size"], mec="white", mew=0.4, zorder=2)
    if o["box"]:
        ax.boxplot([d[a], d[b]], positions=[-0.22, 1.22], widths=0.16, patch_artist=True, showfliers=False, manage_ticks=False,
                   boxprops=dict(facecolor="#eeeeee", edgecolor="#333333"), medianprops=dict(color="#333333"))
    if o["mean_line"]:
        ax.plot([0, 1], [d[a].mean(), d[b].mean()], color="#222222", lw=ctx.s.line_width * 2.2, marker="D",
                ms=ctx.s.marker_size * 1.2, zorder=4, label="Mean")
    if o["test"] != "不顯示" and len(d) >= 3:
        if o["test"] == "配對 t 檢定":
            p = sps.ttest_rel(d[a], d[b]).pvalue
        else:
            p = sps.wilcoxon(d[a], d[b]).pvalue
        top = max(d[a].max(), d[b].max())
        span = top - min(d[a].min(), d[b].min())
        y = top + span * 0.06
        ax.plot([0, 0, 1, 1], [y, y + span * 0.025, y + span * 0.025, y], color="black", lw=1)
        ax.text(0.5, y + span * 0.035, f"{annotate.stars(p)}  ({_pf(p)})", ha="center", va="bottom", fontsize=ctx.s.fs(ctx.s.tick_size))
        ctx.note(f"{o['test']}：{_pf(p)}，n = {len(d)} 組配對")
    ax.set_xticks([0, 1], [a, b])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Value")
    if hue:
        from matplotlib.lines import Line2D

        ax.legend(handles=[Line2D([], [], color=pal[k], marker="o", label=str(k)) for k in m["hue_order"]], title=hue)


@register("count", "🔢 計數圖 Count plot", "類別比較", "各類別的次數或百分比。",
          [Field("x", "類別變項", "cat", order=True), Field("hue", "顏色分組 (hue)", "cat", required=False, order=True)],
          [ORIENT, COLOR_BY_X, Opt("stat", "統計量", "select", "count", ["count", "percent", "proportion"]), WIDTH,
           Opt("labels", "長條上顯示數值", "bool", True)], color_field="hue")
def draw_count(ax, df, m, o, ctx):
    x, hue = m["x"], m.get("hue")
    kw: dict = dict(data=df, ax=ax, stat=o["stat"], width=o["width"], saturation=1, edgecolor=ctx.edge(None))
    if _horiz(o):
        kw.update(y=x, order=m["x_order"])
    else:
        kw.update(x=x, order=m["x_order"])
    if hue and hue != x:
        kw.update(hue=hue, hue_order=m["hue_order"], palette=ctx.palette_for(hue, m["hue_order"]))
    elif o["color_by_x"]:
        kw.update(hue=x, hue_order=m["x_order"], palette=ctx.palette_for(x, m["x_order"]), legend=False)
    else:
        kw["color"] = ctx.s.single_color
    sns.countplot(**kw)
    if o["labels"]:
        fmt = "%.0f" if o["stat"] == "count" else ("%.1f%%" if o["stat"] == "percent" else "%.2f")
        for cont in ax.containers:
            ax.bar_label(cont, fmt=fmt, padding=2, fontsize=ctx.s.fs(ctx.s.tick_size) * 0.9)


# ============================================================ 關係與迴歸
@register("scatter", "✳️ 散佈圖 Scatter plot", "關係與迴歸", "兩數值變項關係；可依類別或連續變數上色、依數值調整大小、標註點名稱。",
          [Field("x", "X 軸", "num"), Field("y", "Y 軸", "num"),
           Field("hue", "顏色 (類別或連續)", "hue", required=False, order=True),
           Field("size", "點大小 (數值)", "num", required=False), Field("style", "標記形狀 (類別)", "cat", required=False),
           Field("label", "點標籤欄位", "any", required=False)],
          [Opt("fit", "疊加趨勢線", "select", "無", ["無", "線性", "LOWESS"]), Opt("show_r", "顯示相關係數", "bool", False),
           Opt("label_n", "最多標註點數", "int", 20, min=0, max=500),
           Opt("size_min", "最小點大小", "float", 20.0, min=1.0, max=200.0), Opt("size_max", "最大點大小", "float", 200.0, min=5.0, max=1000.0)])
def draw_scatter(ax, df, m, o, ctx):
    x, y, hue = m["x"], m["y"], m.get("hue")
    kw: dict = dict(data=df, x=x, y=y, ax=ax, alpha=ctx.s.alpha, edgecolor=ctx.edge("white"), linewidth=0.5)
    if m.get("size"):
        kw.update(size=m["size"], sizes=(o["size_min"], o["size_max"]))
    else:
        kw["s"] = ctx.ms2 * 1.5
    if m.get("style"):
        kw["style"] = m["style"]
    if hue:
        if m.get("hue_order"):
            kw.update(hue=hue, hue_order=m["hue_order"], palette=ctx.palette_for(hue, m["hue_order"]))
        else:
            kw.update(hue=hue, palette=ctx.s.cmap_name)
    else:
        kw["color"] = ctx.s.single_color
    sns.scatterplot(**kw)
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if o["fit"] != "無" and len(d) > 2:
        sns.regplot(data=d, x=x, y=y, ax=ax, scatter=False, lowess=o["fit"] == "LOWESS", ci=95 if o["fit"] == "線性" else None,
                    color="#444444", line_kws=dict(lw=ctx.s.line_width, ls="--"))
    if o["show_r"] and len(d) > 2:
        r, p = sps.pearsonr(d[x], d[y])
        rs, ps = sps.spearmanr(d[x], d[y])
        ax.text(0.97, 0.03, f"Pearson r = {r:.3f}, {_pf(p)}\nSpearman ρ = {rs:.3f}, {_pf(ps)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=ctx.s.fs(ctx.s.tick_size) * 0.9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.75))
    if m.get("label") and o["label_n"] > 0:
        sub = df.dropna(subset=[x, y])
        if len(sub) > o["label_n"]:
            dist = ((sub[x] - sub[x].mean()) / (sub[x].std() or 1)) ** 2 + ((sub[y] - sub[y].mean()) / (sub[y].std() or 1)) ** 2
            sub = sub.loc[dist.sort_values(ascending=False).index[: o["label_n"]]]
        label_points(ax, sub[x].to_numpy(float), sub[y].to_numpy(float), sub[m["label"]].astype(str).tolist(),
                     ctx.s.fs(ctx.s.tick_size) * 0.8)


@register("line", "📉 折線圖 Line plot", "關係與迴歸", "時間序列 / 劑量趨勢，自動計算平均 ± 誤差；可畫個別受試者軌跡。",
          [Field("x", "X 軸 (時間/劑量)", "any", hints=("day", "time", "week", "hour", "dose", "conc", "天", "時間")),
           Field("y", "Y 軸 (數值)", "num"), Field("hue", "顏色分組 (hue)", "cat", required=False, order=True),
           Field("style", "線條樣式分組", "cat", required=False),
           Field("units", "個別受試者 ID (畫每條軌跡)", "any", required=False)],
          [ESTIMATOR, ERRORBAR, Opt("err_style", "誤差呈現", "select", "band", ["band", "bars"]),
           Opt("markers", "顯示標記", "bool", True), CAPSIZE,
           Opt("spaghetti_alpha", "個別軌跡透明度", "float", 0.35, min=0.05, max=1.0, step=0.05),
           Opt("overlay_mean", "個別軌跡上疊加平均線", "bool", True)])
def draw_line(ax, df, m, o, ctx):
    x, y, hue = m["x"], m["y"], m.get("hue")
    kw: dict = dict(data=df, x=x, y=y, ax=ax, linewidth=ctx.s.line_width)
    if hue:
        kw.update(hue=hue, hue_order=m["hue_order"], palette=ctx.palette_for(hue, m["hue_order"]))
    else:
        kw["color"] = ctx.s.single_color
    if m.get("style"):
        kw["style"] = m["style"]
    marker_kw = {"markers": True} if (o["markers"] and m.get("style")) else ({"marker": "o", "markersize": ctx.s.marker_size} if o["markers"] else {})
    if m.get("units"):
        sns.lineplot(**kw, units=m["units"], estimator=None, alpha=o["spaghetti_alpha"], legend=not o["overlay_mean"],
                     **({"marker": "o", "markersize": ctx.s.marker_size * 0.6} if o["markers"] else {}))
        if o["overlay_mean"]:
            kw["linewidth"] = ctx.s.line_width * 2
            sns.lineplot(**kw, estimator=_estimator(o), errorbar=_errorbar(o), err_style=o["err_style"],
                         err_kws={"capsize": o["capsize"] * 10} if o["err_style"] == "bars" else None, **marker_kw)
    else:
        sns.lineplot(**kw, estimator=_estimator(o), errorbar=_errorbar(o), err_style=o["err_style"],
                     err_kws={"capsize": o["capsize"] * 10} if o["err_style"] == "bars" else None, **marker_kw)


@register("reg", "📏 迴歸圖 Regression", "關係與迴歸", "散佈點 + 迴歸線與信賴區間；可分組、多項式、LOWESS、邏輯迴歸，並顯示方程式與 R²。",
          [Field("x", "X 軸 (自變項)", "num"), Field("y", "Y 軸 (依變項)", "num"),
           Field("hue", "分組 (每組一條線)", "cat", required=False, order=True)],
          [Opt("order", "多項式次方", "int", 1, min=1, max=5), Opt("ci", "信賴區間 (%)，0 = 不顯示", "int", 95, min=0, max=99),
           Opt("lowess", "LOWESS 平滑", "bool", False), Opt("logistic", "邏輯迴歸 (Y 為 0/1)", "bool", False),
           Opt("robust", "穩健迴歸 (降低離群值影響)", "bool", False), Opt("scatter", "顯示散佈點", "bool", True),
           Opt("equation", "顯示方程式 / R² / p", "bool", True), Opt("truncate", "迴歸線限於資料範圍", "bool", True)])
def draw_reg(ax, df, m, o, ctx):
    x, y, hue = m["x"], m["y"], m.get("hue")
    groups = m["hue_order"] if hue else [None]
    pal = ctx.palette_for(hue, groups) if hue else {None: ctx.s.single_color}
    for i, g in enumerate(groups):
        sub = (df if g is None else df[df[hue] == g])[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 3:
            continue
        sns.regplot(data=sub, x=x, y=y, ax=ax, order=o["order"], ci=o["ci"] or None, lowess=o["lowess"], logistic=o["logistic"],
                    robust=o["robust"], scatter=o["scatter"], truncate=o["truncate"], color=pal[g], label=None if g is None else str(g),
                    scatter_kws=dict(s=ctx.ms2 * 1.4, alpha=ctx.s.alpha, edgecolor=ctx.edge("white"), linewidths=0.5),
                    line_kws=dict(lw=ctx.s.line_width * 1.3))
        if o["equation"] and not (o["lowess"] or o["logistic"]):
            if o["order"] == 1:
                lr = sps.linregress(sub[x], sub[y])
                txt = f"y = {lr.slope:.3g}x {'+' if lr.intercept >= 0 else '−'} {abs(lr.intercept):.3g},  R² = {lr.rvalue ** 2:.3f},  {_pf(lr.pvalue)}"
            else:
                coef = np.polyfit(sub[x], sub[y], o["order"])
                pred = np.polyval(coef, sub[x])
                r2 = 1 - np.sum((sub[y] - pred) ** 2) / np.sum((sub[y] - sub[y].mean()) ** 2)
                txt = f"{o['order']} 次多項式  R² = {r2:.3f}"
            if g is not None:
                txt = f"{g}: {txt}"
            ax.text(0.03, 0.97 - i * 0.075, txt, transform=ax.transAxes, va="top", color=pal[g] if hue else ctx.s.text_color,
                    fontsize=ctx.s.fs(ctx.s.tick_size) * 0.9)
    if hue:
        ax.legend(title=hue)


@register("resid", "〰️ 殘差圖 Residual plot", "關係與迴歸", "檢查線性迴歸的殘差是否隨機分布。",
          [Field("x", "X 軸", "num"), Field("y", "Y 軸", "num")],
          [Opt("lowess", "加 LOWESS 平滑線", "bool", True), Opt("order", "多項式次方", "int", 1, min=1, max=5)])
def draw_resid(ax, df, m, o, ctx):
    d = df[[m["x"], m["y"]]].apply(pd.to_numeric, errors="coerce").dropna()
    sns.residplot(data=d, x=m["x"], y=m["y"], ax=ax, lowess=o["lowess"], order=o["order"], color=ctx.s.single_color,
                  scatter_kws=dict(s=ctx.ms2 * 1.4, alpha=ctx.s.alpha), line_kws=dict(color="#e34948", lw=ctx.s.line_width))
    ax.set_ylabel(f"Residuals of {m['y']}")


@register("hexbin", "⬢ 六角密度圖 Hexbin", "關係與迴歸", "大量資料點的二維密度。",
          [Field("x", "X 軸", "num"), Field("y", "Y 軸", "num")],
          [Opt("gridsize", "格數", "int", 30, min=5, max=150), Opt("log", "對數色階", "bool", False), Opt("cbar", "顯示色條", "bool", True)],
          uses_cmap=True, color_field=None)
def draw_hexbin(ax, df, m, o, ctx):
    d = df[[m["x"], m["y"]]].apply(pd.to_numeric, errors="coerce").dropna()
    hb = ax.hexbin(d[m["x"]], d[m["y"]], gridsize=o["gridsize"], cmap=ctx.s.cmap_name, bins="log" if o["log"] else None, mincnt=1,
                   linewidths=0.2)
    if o["cbar"]:
        ax.figure.colorbar(hb, ax=ax, label="Count" + (" (log)" if o["log"] else ""))
    ax.set_xlabel(m["x"])
    ax.set_ylabel(m["y"])


@register("kde2d", "🗺️ 二維密度等高線", "關係與迴歸", "兩變項聯合密度的等高線，可分組。",
          [Field("x", "X 軸", "num"), Field("y", "Y 軸", "num"), Field("hue", "顏色分組", "cat", required=False, order=True)],
          [Opt("fill", "填色", "bool", True), Opt("levels", "等高線層數", "int", 8, min=2, max=30),
           Opt("thresh", "最低密度門檻", "float", 0.05, min=0.0, max=0.5, step=0.01), Opt("points", "疊加散佈點", "bool", False)],
          uses_cmap=True)
def draw_kde2d(ax, df, m, o, ctx):
    kw = dict(data=df, x=m["x"], y=m["y"], ax=ax, fill=o["fill"], levels=o["levels"], thresh=o["thresh"], alpha=ctx.s.alpha)
    if m.get("hue"):
        kw.update(hue=m["hue"], hue_order=m["hue_order"], palette=ctx.palette_for(m["hue"], m["hue_order"]))
    else:
        kw["cmap"] = ctx.s.cmap_name
    sns.kdeplot(**kw)
    if o["points"]:
        sns.scatterplot(data=df, x=m["x"], y=m["y"], ax=ax, s=ctx.ms2 * 0.6, color="#333333", alpha=0.5, legend=False)


# ============================================================ 矩陣與熱圖
def _matrix(df, m, zscore: str = "無") -> pd.DataFrame:
    cols = m.get("values") or []
    if not cols:
        raise ValueError("請選擇至少兩個數值欄位")
    mat = df[cols].apply(pd.to_numeric, errors="coerce")
    if m.get("index"):
        mat.index = df[m["index"]].astype(str)
    mat = mat.dropna(how="all")
    if zscore == "列 (row)":
        mat = mat.sub(mat.mean(1), axis=0).div(mat.std(1, ddof=1).replace(0, np.nan), axis=0)
    elif zscore == "欄 (column)":
        mat = (mat - mat.mean()) / mat.std(ddof=1).replace(0, np.nan)
    return mat


def _heat_center(o, ctx, mat):
    """色階中心：發散色階且資料跨越 0 時自動置中於 0。"""
    if o.get("center") == "0" or (o.get("center") == "自動" and o.get("zscore", "無") != "無"):
        return 0.0
    if o.get("center") == "自動":
        base = ctx.s.cmap_name.removesuffix("_r")
        if base in {c.removesuffix("_r") for c in DIVERGING_CMAPS} and np.nanmin(mat.values) < 0 < np.nanmax(mat.values):
            return 0.0
    return None


@register("corr", "🟥 相關係數熱圖 Correlation heatmap", "矩陣與熱圖", "多個數值變項兩兩相關係數，可標示顯著性星號。",
          [Field("values", "數值變項 (≥ 2)", "nums", default_n=8)],
          [Opt("method", "相關係數", "select", "pearson", ["pearson", "spearman", "kendall"]),
           Opt("mask", "隱藏", "select", "上三角", ["無", "上三角", "下三角"]), Opt("annot", "顯示數值", "bool", True),
           Opt("decimals", "小數位數", "int", 2, min=0, max=4), Opt("stars", "標示顯著性星號", "bool", True),
           Opt("square", "正方形格子", "bool", True), Opt("linewidths", "格線寬度", "float", 0.5, min=0.0, max=3.0, step=0.25),
           Opt("cbar", "顯示色條", "bool", True)],
          facet=False, uses_cmap=True, despine=False, color_field=None)
def draw_corr(ax, df, m, o, ctx):
    cols = m["values"]
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = d.corr(method=o["method"])
    pmat = pd.DataFrame(np.ones_like(corr), index=cols, columns=cols)
    fn = {"pearson": sps.pearsonr, "spearman": sps.spearmanr, "kendall": sps.kendalltau}[o["method"]]
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i < j:
                pair = d[[a, b]].dropna()
                if len(pair) > 2:
                    pmat.loc[a, b] = pmat.loc[b, a] = float(fn(pair[a], pair[b])[1])
    mask = None
    if o["mask"] == "上三角":
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    elif o["mask"] == "下三角":
        mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
    annot = False
    if o["annot"] or o["stars"]:
        annot = corr.copy().astype(object)
        for a in cols:
            for b in cols:
                txt = f"{corr.loc[a, b]:.{o['decimals']}f}" if o["annot"] else ""
                if o["stars"] and a != b:
                    st = annotate.stars(pmat.loc[a, b])
                    txt += ("\n" if o["annot"] else "") + (st if st != "ns" else "")
                annot.loc[a, b] = txt
        annot = annot.values
    cmap = ctx.s.cmap_name if ctx.s.cmap != "viridis" else "RdBu_r"
    sns.heatmap(corr, mask=mask, annot=annot, fmt="", cmap=cmap, vmin=-1, vmax=1, center=0, square=o["square"],
                linewidths=o["linewidths"], linecolor="white", cbar=o["cbar"], ax=ax,
                cbar_kws={"label": f"{o['method'].title()} r", "shrink": 0.8},
                annot_kws={"fontsize": ctx.s.fs(ctx.s.tick_size) * 0.85})
    ctx.table("相關係數", corr.reset_index(names="變項"))
    ctx.table("p 值", pmat.reset_index(names="變項"))
    if o["stars"]:
        ctx.note("星號：* p < 0.05、** p < 0.01、*** p < 0.001、**** p < 0.0001 (未校正多重比較)")


@register("heatmap", "🔥 熱圖 Heatmap (矩陣)", "矩陣與熱圖", "寬格式數值矩陣 (例如基因 × 樣本)，可做列/欄 Z-score。",
          [Field("index", "列標籤欄 (如基因名)", "any", required=False, hints=("gene", "name", "id", "基因")),
           Field("values", "數值欄位 (矩陣的欄)", "nums", default_n=20)],
          [Opt("zscore", "Z-score 標準化", "select", "無", ["無", "列 (row)", "欄 (column)"]),
           Opt("annot", "顯示數值", "bool", False), Opt("decimals", "小數位數", "int", 1, min=0, max=4),
           Opt("center", "色階中心", "select", "自動", ["自動", "0", "不設定"]),
           Opt("linewidths", "格線寬度", "float", 0.0, min=0.0, max=3.0, step=0.25), Opt("square", "正方形格子", "bool", False),
           Opt("cbar_label", "色條標題", "text", "")],
          facet=False, uses_cmap=True, despine=False, color_field=None)
def draw_heatmap(ax, df, m, o, ctx):
    mat = _matrix(df, m, o["zscore"])
    center = _heat_center(o, ctx, mat)
    label = o["cbar_label"] or ("Z-score" if o["zscore"] != "無" else "")
    sns.heatmap(mat, ax=ax, cmap=ctx.s.cmap_name, center=center, annot=o["annot"], fmt=f".{o['decimals']}f",
                linewidths=o["linewidths"], linecolor="white", square=o["square"], cbar_kws={"label": label},
                annot_kws={"fontsize": ctx.s.fs(ctx.s.tick_size) * 0.8}, yticklabels="auto", xticklabels=True)
    ax.set_ylabel(m.get("index") or "")


@register("pivot_heatmap", "🧮 樞紐熱圖 (長格式彙整)", "矩陣與熱圖", "把長格式資料依兩個類別彙整 (平均/總和/次數) 後畫成熱圖。",
          [Field("row", "列 (類別)", "cat", order=True), Field("col", "欄 (類別)", "cat", order=True), Field("value", "數值", "num", required=False)],
          [Opt("agg", "彙整方式", "select", "mean", ["mean", "median", "sum", "count", "std"]), Opt("annot", "顯示數值", "bool", True),
           Opt("decimals", "小數位數", "int", 1, min=0, max=4), Opt("linewidths", "格線寬度", "float", 0.5, min=0.0, max=3.0, step=0.25)],
          facet=False, uses_cmap=True, despine=False, color_field=None)
def draw_pivot_heatmap(ax, df, m, o, ctx):
    if m.get("value"):
        tab = df.pivot_table(index=m["row"], columns=m["col"], values=m["value"], aggfunc=o["agg"])
    else:
        tab = pd.crosstab(df[m["row"]], df[m["col"]])
    tab = tab.reindex(index=m["row_order"], columns=m["col_order"])
    sns.heatmap(tab, ax=ax, cmap=ctx.s.cmap_name, annot=o["annot"], fmt=f".{o['decimals']}f" if m.get("value") and o["agg"] != "count" else ".0f",
                linewidths=o["linewidths"], linecolor="white", cbar_kws={"label": f"{o['agg']}({m['value']})" if m.get("value") else "count"},
                annot_kws={"fontsize": ctx.s.fs(ctx.s.tick_size) * 0.85})
    ctx.table("樞紐表", tab.reset_index())


@register("clustermap", "🌳 階層分群熱圖 Clustermap", "矩陣與熱圖", "列與欄同時做階層分群並畫樹狀圖，基因表現分析常用。",
          [Field("index", "列標籤欄 (如基因名)", "any", required=False, hints=("gene", "name", "id", "基因")),
           Field("values", "數值欄位", "nums", default_n=20),
           Field("row_group", "列分組色條 (類別)", "cat", required=False, hints=("pathway", "group", "class", "type"))],
          [Opt("zscore", "Z-score 標準化", "select", "列 (row)", ["無", "列 (row)", "欄 (column)"]),
           Opt("method", "連結方法", "select", "average", ["average", "complete", "single", "ward", "weighted", "centroid", "median"]),
           Opt("metric", "距離", "select", "euclidean", ["euclidean", "correlation", "cityblock", "cosine", "chebyshev"]),
           Opt("row_cluster", "列分群", "bool", True), Opt("col_cluster", "欄分群", "bool", True),
           Opt("dendro_ratio", "樹狀圖比例", "float", 0.15, min=0.05, max=0.4, step=0.01),
           Opt("show_rows", "顯示列標籤", "bool", True), Opt("linewidths", "格線寬度", "float", 0.0, min=0.0, max=2.0, step=0.25)],
          figure_level=True, facet=False, uses_cmap=True, axes_style=False, color_field="row_group")
def draw_clustermap(df, m, o, ctx):
    cols = m.get("values") or []
    if len(cols) < 2:
        raise ValueError("請選擇至少兩個數值欄位")
    raw = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(raw) < 2:
        raise ValueError("去除缺失值後少於 2 列，無法分群")
    labels = df.loc[raw.index, m["index"]].astype(str) if m.get("index") else raw.index.astype(str)
    mat = raw.set_axis(list(labels), axis=0)
    z = {"無": None, "列 (row)": 0, "欄 (column)": 1}[o["zscore"]]
    row_colors = None
    handles = []
    if m.get("row_group"):
        grp = df.loc[raw.index, m["row_group"]]
        grp.index = mat.index
        lv = [x for x in m.get("row_group_order", []) if x in set(grp)] or list(dict.fromkeys(grp.dropna().tolist()))
        pal = ctx.palette_for(m["row_group"], lv)
        row_colors = grp.map(pal).fillna("#ffffff").rename(m["row_group"])
        handles = [Patch(facecolor=pal[k], label=k) for k in lv]
    method, metric = o["method"], o["metric"]
    if method in ("ward", "centroid", "median"):
        metric = "euclidean"
    g = sns.clustermap(mat, z_score=z, method=method, metric=metric, cmap=ctx.s.cmap_name, center=0 if z is not None else None,
                       row_cluster=o["row_cluster"], col_cluster=o["col_cluster"], figsize=(ctx.s.width, ctx.s.height),
                       dendrogram_ratio=o["dendro_ratio"], row_colors=row_colors, linewidths=o["linewidths"],
                       yticklabels=o["show_rows"], cbar_kws={"label": "Z-score" if z is not None else ""},
                       cbar_pos=(0.02, 0.83, 0.03, 0.15))
    g.ax_heatmap.tick_params(labelsize=ctx.s.fs(ctx.s.tick_size) * 0.85)
    g.ax_heatmap.set_ylabel("")
    if ctx.s.xtick_rotation:
        for lab in g.ax_heatmap.get_xticklabels():
            lab.set_rotation(ctx.s.xtick_rotation)
            lab.set_ha("right")
    if handles:
        g.ax_col_dendrogram.legend(handles=handles, title=m["row_group"], loc="center left", bbox_to_anchor=(1.02, 0.5),
                                   frameon=False, fontsize=ctx.s.fs(ctx.s.legend_size), title_fontsize=ctx.s.fs(ctx.s.legend_size))
    if o["row_cluster"]:
        ctx.table("分群後列順序", pd.DataFrame({"順序": range(1, len(mat) + 1), "列": mat.index[g.dendrogram_row.reordered_ind]}))
    return PlotResult(g.fig, [g.ax_heatmap], title_on_fig=True)


# ============================================================ 多變量
@register("pairplot", "🔲 配對圖 Pair plot", "多變量", "多個數值變項兩兩散佈 + 對角線分布。",
          [Field("values", "數值變項 (2–8 個)", "nums", default_n=4), Field("hue", "顏色分組", "cat", required=False, order=True)],
          [Opt("kind", "非對角圖", "select", "scatter", ["scatter", "kde", "hist", "reg"]),
           Opt("diag_kind", "對角圖", "select", "auto", ["auto", "hist", "kde"]), Opt("corner", "只畫下三角", "bool", False),
           Opt("panel", "每格尺寸 (英吋)", "float", 2.2, min=1.0, max=5.0, step=0.1)],
          figure_level=True, facet=False)
def draw_pairplot(df, m, o, ctx):
    cols = m["values"][:8]
    if len(cols) < 2:
        raise ValueError("請選擇至少兩個數值變項")
    kw = dict(vars=cols, kind=o["kind"], diag_kind=o["diag_kind"], corner=o["corner"], height=o["panel"])
    if m.get("hue"):
        kw.update(hue=m["hue"], hue_order=m["hue_order"], palette=ctx.palette_for(m["hue"], m["hue_order"]))
    if o["kind"] == "scatter":
        kw["plot_kws"] = dict(s=ctx.ms2 * 1.2, alpha=ctx.s.alpha, edgecolor=ctx.edge("white"), linewidth=0.4)
        if not m.get("hue"):
            kw["plot_kws"]["color"] = ctx.s.single_color
    if not m.get("hue"):
        kw["diag_kws"] = dict(color=ctx.s.single_color)
    g = sns.pairplot(df, **kw)
    return PlotResult(g.figure, list(g.axes.flat), grid=g, title_on_fig=True)


@register("jointplot", "🎯 聯合分布圖 Joint plot", "多變量", "散佈圖 + 上方/右方邊際分布。",
          [Field("x", "X 軸", "num"), Field("y", "Y 軸", "num"), Field("hue", "顏色分組 (scatter/kde/hist)", "cat", required=False, order=True)],
          [Opt("kind", "主圖類型", "select", "scatter", ["scatter", "kde", "hist", "hex", "reg", "resid"]),
           Opt("ratio", "主圖 : 邊際圖 比例", "int", 5, min=2, max=10), Opt("marginal_ticks", "邊際圖顯示刻度", "bool", False)],
          figure_level=True, facet=False)
def draw_jointplot(df, m, o, ctx):
    kw = dict(data=df, x=m["x"], y=m["y"], kind=o["kind"], height=ctx.s.height, ratio=o["ratio"], marginal_ticks=o["marginal_ticks"])
    if m.get("hue") and o["kind"] in ("scatter", "kde", "hist"):
        kw.update(hue=m["hue"], hue_order=m["hue_order"], palette=ctx.palette_for(m["hue"], m["hue_order"]))
    else:
        kw["color"] = ctx.s.single_color
    if o["kind"] == "scatter":
        kw.update(s=ctx.ms2 * 1.4, alpha=ctx.s.alpha, edgecolor=ctx.edge("white"))
    g = sns.jointplot(**kw)
    return PlotResult(g.figure, [g.ax_joint, g.ax_marg_x, g.ax_marg_y], title_on_fig=True)


@register("pca", "🧭 主成分分析 PCA", "多變量", "降維散佈圖，含解釋變異比例、組別 95% 信賴橢圓與負荷量箭頭 (biplot)。",
          [Field("values", "特徵變項 (數值)", "nums", default_n=10), Field("hue", "顏色分組", "cat", required=False, order=True),
           Field("label", "點標籤欄位", "any", required=False)],
          [Opt("scale", "標準化 (建議)", "bool", True), Opt("pcs", "主成分", "select", "PC1 vs PC2", ["PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3"]),
           Opt("ellipse", "組別 95% 信賴橢圓", "bool", True), Opt("loadings", "顯示負荷量箭頭 (前 N 個)", "int", 0, min=0, max=30),
           Opt("transpose", "轉置 (欄位為樣本，如基因表現矩陣)", "bool", False)],
          facet=False)
def draw_pca(ax, df, m, o, ctx):
    cols = m["values"]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    groups = df[m["hue"]] if m.get("hue") else None
    labels = df[m["label"]].astype(str) if m.get("label") else None
    if o["transpose"]:
        if m.get("label"):
            X.index = df[m["label"]].astype(str)
        X = X.dropna().T
        groups = None
        labels = pd.Series(X.index, index=X.index)
    else:
        keep = X.dropna().index
        X = X.loc[keep]
        groups = groups.loc[keep] if groups is not None else None
        labels = labels.loc[keep] if labels is not None else None
    if X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError("PCA 至少需要 3 個樣本、2 個特徵 (缺失值的列已排除)")
    A = X.to_numpy(float)
    A = A - A.mean(0)
    if o["scale"]:
        sd = A.std(0, ddof=1)
        A = A / np.where(sd == 0, 1, sd)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    scores = U * S
    var = S ** 2 / (A.shape[0] - 1)
    ratio = var / var.sum()
    i, j = {"PC1 vs PC2": (0, 1), "PC1 vs PC3": (0, 2), "PC2 vs PC3": (1, 2)}[o["pcs"]]
    j = min(j, scores.shape[1] - 1)
    if groups is not None:
        lv = m["hue_order"]
        pal = ctx.palette_for(m["hue"], lv)
        for g in lv:
            mask = (groups == g).to_numpy()
            if not mask.any():
                continue
            pts = scores[mask][:, [i, j]]
            ax.scatter(pts[:, 0], pts[:, 1], s=ctx.ms2 * 1.6, color=pal[g], alpha=ctx.s.alpha, edgecolors=ctx.edge("white"),
                       linewidths=0.5, label=str(g), zorder=3)
            if o["ellipse"] and len(pts) >= 3:
                cov = np.cov(pts.T)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                k = np.sqrt(sps.chi2.ppf(0.95, 2))
                ax.add_patch(Ellipse(pts.mean(0), 2 * k * np.sqrt(max(vals[0], 0)), 2 * k * np.sqrt(max(vals[1], 0)), angle=ang,
                                     facecolor=pal[g], alpha=0.12, edgecolor=pal[g], lw=ctx.s.line_width, zorder=1))
        ax.legend(title=m["hue"])
    else:
        ax.scatter(scores[:, i], scores[:, j], s=ctx.ms2 * 1.6, color=ctx.s.single_color, alpha=ctx.s.alpha,
                   edgecolors=ctx.edge("white"), linewidths=0.5, zorder=3)
    if labels is not None:
        label_points(ax, scores[:, i], scores[:, j], [str(t) for t in labels], ctx.s.fs(ctx.s.tick_size) * 0.75)
    feat_names = list(X.columns)
    load = Vt.T * S / np.sqrt(A.shape[0] - 1)
    if o["loadings"] > 0:
        L = load[:, [i, j]]
        top = np.argsort(-np.hypot(L[:, 0], L[:, 1]))[: o["loadings"]]
        f = 0.85 * np.abs(scores[:, [i, j]]).max() / max(np.abs(L[top]).max(), 1e-12)
        for t in top:
            ax.annotate("", xy=(L[t, 0] * f, L[t, 1] * f), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=ctx.s.line_width * 0.8))
            ax.text(L[t, 0] * f * 1.08, L[t, 1] * f * 1.08, str(feat_names[t]), color="#c0392b", ha="center", va="center",
                    fontsize=ctx.s.fs(ctx.s.tick_size) * 0.8)
    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.set_xlabel(f"PC{i + 1} ({ratio[i]:.1%})")
    ax.set_ylabel(f"PC{j + 1} ({ratio[j]:.1%})")
    npc = min(len(S), 10)
    ctx.table("解釋變異", pd.DataFrame({"主成分": [f"PC{k + 1}" for k in range(npc)], "特徵值": var[:npc],
                                       "解釋比例": ratio[:npc], "累積比例": np.cumsum(ratio)[:npc]}))
    ctx.table("負荷量 (loadings)", pd.DataFrame(load[:, :npc], index=feat_names, columns=[f"PC{k + 1}" for k in range(npc)]).reset_index(names="特徵"))
    sc = pd.DataFrame(scores[:, :npc], columns=[f"PC{k + 1}" for k in range(npc)], index=X.index)
    ctx.table("主成分分數 (scores)", sc.reset_index(names="樣本"))
    ctx.note(f"PC1 解釋 {ratio[0]:.1%}、PC2 解釋 {ratio[1]:.1%} 的變異 (累積 {ratio[:2].sum():.1%})。")


# ============================================================ 生醫專用
@register("volcano", "🌋 火山圖 Volcano plot", "生醫專用", "差異表現分析：log₂ 倍數變化 vs −log₁₀ p 值，標示上調/下調基因。",
          [Field("fc", "log₂ Fold Change", "num", hints=("log2", "fold", "fc", "lfc")),
           Field("p", "p 值 (或校正後 p)", "num", hints=("padj", "fdr", "adj", "qval", "p.?val", "^p$")),
           Field("label", "基因名稱欄", "any", required=False, hints=("gene", "symbol", "name", "id"))],
          [Opt("fc_thresh", "|log₂FC| 門檻", "float", 1.0, min=0.0, max=10.0, step=0.1),
           Opt("p_thresh", "p 值門檻", "float", 0.05, min=1e-10, max=1.0, step=0.01),
           Opt("fc_is_log2", "倍數已是 log₂", "bool", True), Opt("p_is_log", "p 值已是 −log₁₀", "bool", False),
           Opt("top_n", "標註前 N 個顯著基因", "int", 10, min=0, max=100),
           Opt("highlight", "指定標註基因 (逗號分隔)", "text", ""),
           Opt("c_up", "上調顏色", "color", "#e34948"), Opt("c_down", "下調顏色", "color", "#2a78d6"), Opt("c_ns", "不顯著顏色", "color", "#bdbdbd"),
           Opt("lines", "顯示門檻虛線", "bool", True), Opt("counts", "圖例顯示數量", "bool", True)],
          color_field=None)
def draw_volcano(ax, df, m, o, ctx):
    fc = _num(df[m["fc"]])
    if not o["fc_is_log2"]:
        fc = np.log2(fc.where(fc > 0))
    p = _num(df[m["p"]])
    nlp = p if o["p_is_log"] else -np.log10(p.clip(lower=1e-300))
    thr = -np.log10(o["p_thresh"])
    up = (fc >= o["fc_thresh"]) & (nlp >= thr)
    down = (fc <= -o["fc_thresh"]) & (nlp >= thr)
    ns = ~(up | down) & fc.notna() & nlp.notna()
    raster = len(df) > 3000
    for mask, c, name in ((ns, o["c_ns"], "Not sig."), (down, o["c_down"], "Down"), (up, o["c_up"], "Up")):
        lab = f"{name} ({int(mask.sum())})" if o["counts"] else name
        ax.scatter(fc[mask], nlp[mask], s=ctx.ms2 * 0.9, color=c, alpha=ctx.s.alpha if name != "Not sig." else 0.5,
                   edgecolors=ctx.edge("none"), linewidths=0.3, label=lab, rasterized=raster, zorder=2 if name == "Not sig." else 3)
    if o["lines"]:
        kw = dict(color="#777777", ls="--", lw=0.8, zorder=1)
        ax.axhline(thr, **kw)
        if o["fc_thresh"] > 0:
            ax.axvline(o["fc_thresh"], **kw)
            ax.axvline(-o["fc_thresh"], **kw)
    if m.get("label"):
        names = df[m["label"]].astype(str)
        sig = up | down
        score = (nlp.rank(pct=True) + fc.abs().rank(pct=True))[sig]
        idx = list(score.sort_values(ascending=False).index[: o["top_n"]])
        wanted = {g.strip().lower() for g in o["highlight"].split(",") if g.strip()}
        if wanted:
            idx += [i for i in df.index[names.str.lower().isin(wanted)] if i not in idx]
        label_points(ax, fc[idx].to_numpy(float), nlp[idx].to_numpy(float), names[idx].tolist(), ctx.s.fs(ctx.s.tick_size) * 0.8)
    ax.set_xlabel(r"$\log_2$ Fold Change")
    ax.set_ylabel(r"$-\log_{10}$ " + ("adjusted p" if any(k in m["p"].lower() for k in ("adj", "fdr", "q")) else "p-value"))
    ax.legend()
    ctx.note(f"上調 {int(up.sum())}、下調 {int(down.sum())} 個 (|log₂FC| ≥ {o['fc_thresh']}, p < {o['p_thresh']})。")
    out = df.assign(_log2FC=fc, _neglog10p=nlp, Regulation=np.where(up, "Up", np.where(down, "Down", "NS")))
    ctx.table("顯著基因", out[up | down].sort_values("_neglog10p", ascending=False))


@register("km", "⏳ Kaplan-Meier 存活曲線", "生醫專用", "存活曲線、設限標記、信賴區間、log-rank 檢定與風險人數表。",
          [Field("time", "時間", "num", hints=("time", "month", "day", "week", "year", "os", "pfs", "時間")),
           Field("event", "事件 (1 = 發生, 0 = 設限)", "any", hints=("event", "status", "death", "dead", "censor", "事件", "死亡")),
           Field("group", "分組", "cat", required=False, order=True, hints=("group", "risk", "treat", "arm", "組"))],
          [Opt("ci", "顯示 95% 信賴區間", "bool", True), Opt("censor", "顯示設限標記 (+)", "bool", True),
           Opt("risk_table", "顯示風險人數表", "bool", True), Opt("logrank", "顯示 log-rank p 值", "bool", True),
           Opt("median", "標示中位存活時間", "bool", False), Opt("percent", "Y 軸以百分比表示", "bool", False),
           Opt("ylabel", "Y 軸標題", "text", "Survival probability"), Opt("ci_alpha", "信賴區間透明度", "float", 0.15, min=0.05, max=0.5, step=0.05)],
          figure_level=True, facet=False, color_field="group")
def draw_km(df, m, o, ctx):
    s = ctx.s
    t_col, e_col, g_col = m["time"], m["event"], m.get("group")
    d = df[[c for c in (t_col, e_col, g_col) if c]].copy()
    d[t_col] = _num(d[t_col])
    d = d.dropna()
    d["_e"] = bstats._event01(d[e_col])
    groups = m["group_order"] if g_col else [None]
    pal = ctx.palette_for(g_col, groups) if g_col else {None: s.single_color}
    if o["risk_table"]:
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(s.width, s.height * (1 + 0.08 * max(len(groups), 1))), sharex=True,
                                      gridspec_kw=dict(height_ratios=[4, 0.45 + 0.28 * len(groups)], hspace=0.08))
    else:
        fig, ax = plt.subplots(figsize=(s.width, s.height))
        axr = None
    scale = 100 if o["percent"] else 1
    tmax = float(d[t_col].max()) if len(d) else 1.0
    for g in groups:
        sub = d if g is None else d[d[g_col] == g]
        if sub.empty:
            continue
        tab = bstats.km_estimate(sub[t_col], sub["_e"])
        c = pal[g]
        label = f"{g} (n = {len(sub)})" if g is not None else f"All (n = {len(sub)})"
        ax.step(tab["time"], tab["survival"] * scale, where="post", color=c, lw=s.line_width * 1.2, label=label)
        if o["ci"]:
            ax.fill_between(tab["time"], tab["ci_low"] * scale, tab["ci_high"] * scale, step="post", color=c, alpha=o["ci_alpha"], lw=0)
        if o["censor"]:
            ct = sub.loc[sub["_e"] == 0, t_col].to_numpy()
            if len(ct):
                idx = np.searchsorted(tab["time"].to_numpy(), ct, "right") - 1
                ax.plot(ct, tab["survival"].to_numpy()[np.clip(idx, 0, None)] * scale, ls="none", marker="+", color=c,
                        ms=s.marker_size * 1.3, mew=s.line_width)
        if o["median"]:
            med = bstats.km_median(tab)[0]
            if np.isfinite(med):
                ax.plot([0, med, med], [0.5 * scale, 0.5 * scale, 0], color=c, ls=":", lw=s.line_width * 0.8)
    if o["logrank"] and g_col and len(groups) >= 2:
        chi2, dof, p = bstats.logrank_test(d[t_col], d["_e"], d[g_col])
        ax.text(0.03, 0.05, f"Log-rank {_pf(p)}", transform=ax.transAxes, fontsize=s.fs(s.legend_size))
        ctx.note(f"Log-rank 檢定：χ²({dof}) = {chi2:.3f}，{_pf(p)}")
    ax.set_ylim(0, 1.05 * scale)
    ax.set_xlim(s.xmin if s.xmin is not None else 0, s.xmax if s.xmax is not None else tmax * 1.03)
    ax.set_ylabel(o["ylabel"] + (" (%)" if o["percent"] and "%" not in o["ylabel"] else ""))
    ax.set_xlabel(t_col)
    ax.legend(loc="upper right")
    if axr is not None:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
        ticks = [t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
        for r, g in enumerate(groups):
            sub = d if g is None else d[d[g_col] == g]
            for t in ticks:
                axr.text(t, r, str(int((sub[t_col] >= t).sum())), ha="center", va="center", color=pal[g],
                         fontsize=s.fs(s.tick_size) * 0.9)
        axr.set_ylim(len(groups) - 0.5, -0.5)
        axr.set_yticks(range(len(groups)), [str(g) if g is not None else "All" for g in groups])
        for lab, g in zip(axr.get_yticklabels(), groups):
            lab.set_color(pal[g])
        axr.set_xticks(ticks)
        axr.set_xlabel("" if s.hide_xlabel else (s.xlabel or t_col), fontsize=s.fs(s.label_size))
        axr.set_title("Number at risk", loc="left", fontsize=s.fs(s.tick_size), fontweight="bold")
        axr.tick_params(axis="y", length=0, pad=18, labelsize=s.fs(s.tick_size) * 0.9)
        axr.tick_params(axis="x", labelsize=s.fs(s.tick_size))
        for side in ("top", "right", "left"):
            axr.spines[side].set_visible(False)
        axr.grid(False)
    summary = bstats.survival_analysis(df, t_col, e_col, g_col)
    for name, tab in summary.tables:
        ctx.table(name, tab)
    return PlotResult(fig, [ax], title_on_fig=False)


@register("roc", "🎚️ ROC 曲線", "生醫專用", "診斷指標的敏感度/特異度曲線，顯示 AUC (DeLong 95% CI) 與最佳切點。",
          [Field("truth", "真實狀態 (二元)", "cat", hints=("disease", "label", "outcome", "status", "class", "case", "疾病")),
           Field("scores", "預測指標 (可多選)", "nums", default_n=3)],
          [Opt("positive", "陽性類別 (空白 = 自動)", "text", ""), Opt("auto_flip", "AUC < 0.5 時自動反轉方向", "bool", True),
           Opt("show_ci", "圖例顯示 AUC 95% CI", "bool", True), Opt("youden", "標示最佳切點 (Youden)", "bool", True),
           Opt("diag", "顯示對角參考線", "bool", True), Opt("square", "正方形座標", "bool", True)],
          facet=False, color_field=None)
def draw_roc(ax, df, m, o, ctx):
    truth = m["truth"]
    lv = m["truth_order"]
    pos = o["positive"].strip() or ("1" if "1" in lv else (lv[-1] if lv else ""))
    if pos not in lv:
        raise ValueError(f"陽性類別「{pos}」不存在，可用值：{', '.join(lv)}")
    cols = m["scores"]
    colors = ctx.s.colors(len(cols))
    res = bstats.roc_analysis(df, truth, pos, cols)
    for name, tab in res.tables:
        ctx.table(name, tab)
    d = df[[truth] + cols].copy()
    for k, c in enumerate(cols):
        sub = d[[truth, c]].dropna()
        y = (sub[truth].astype(str) == pos).to_numpy().astype(int)
        s = _num(sub[c]).to_numpy()
        fpr, tpr, thr, auc = bstats.roc_curve(y, s)
        flipped = False
        if auc < 0.5 and o["auto_flip"]:
            fpr, tpr, thr, auc = bstats.roc_curve(y, -s)
            flipped = True
        label = f"{c}{' (反向)' if flipped else ''}: AUC = {auc:.3f}"
        if o["show_ci"]:
            _, cov = bstats._delong_components(y, (-s if flipped else s)[None, :])
            se = np.sqrt(max(cov[0, 0], 0))
            label += f" ({max(auc - 1.96 * se, 0):.3f}–{min(auc + 1.96 * se, 1):.3f})"
        ax.plot(fpr, tpr, color=colors[k], lw=ctx.s.line_width * 1.3, label=label)
        if o["youden"]:
            bi = int(np.argmax((tpr - fpr)[1:]) + 1)
            ax.plot(fpr[bi], tpr[bi], "o", color=colors[k], ms=ctx.s.marker_size, mec="white", zorder=4)
    if o["diag"]:
        ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    if o["square"]:
        ax.set_aspect("equal")
    ax.set_xlabel("1 − Specificity (False positive rate)")
    ax.set_ylabel("Sensitivity (True positive rate)")
    ax.legend(loc="lower right")
    ctx.note(f"陽性類別：{truth} = {pos}")


@register("bland", "⚖️ Bland-Altman 一致性圖", "生醫專用", "比較兩種測量方法的一致性：平均偏差與 95% 一致性界限 (LoA)。",
          [Field("a", "方法 A", "num"), Field("b", "方法 B", "num")],
          [Opt("percent", "差異以百分比表示", "bool", False), Opt("loa_ci", "顯示界限的 95% CI", "bool", False),
           Opt("trend", "檢查比例偏差 (迴歸線)", "bool", False), Opt("mean_color", "平均偏差線顏色", "color", "#e34948"),
           Opt("loa_color", "一致性界限顏色", "color", "#555555")],
          facet=True, color_field=None)
def draw_bland(ax, df, m, o, ctx):
    d = df[[m["a"], m["b"]]].apply(pd.to_numeric, errors="coerce").dropna()
    a, b = d[m["a"]].to_numpy(), d[m["b"]].to_numpy()
    mean = (a + b) / 2
    diff = (a - b) / mean * 100 if o["percent"] else a - b
    n = len(diff)
    md, sd = diff.mean(), diff.std(ddof=1)
    lo, hi = md - 1.96 * sd, md + 1.96 * sd
    ax.scatter(mean, diff, s=ctx.ms2 * 1.4, color=ctx.s.single_color, alpha=ctx.s.alpha, edgecolors=ctx.edge("white"), linewidths=0.5, zorder=3)
    ax.axhline(md, color=o["mean_color"], lw=ctx.s.line_width)
    ax.axhline(lo, color=o["loa_color"], lw=ctx.s.line_width, ls="--")
    ax.axhline(hi, color=o["loa_color"], lw=ctx.s.line_width, ls="--")
    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    if o["loa_ci"]:
        t = sps.t.ppf(0.975, n - 1)
        se_md, se_loa = sd / np.sqrt(n), sd * np.sqrt(3 / n)
        for c, se, col in ((md, se_md, o["mean_color"]), (lo, se_loa, o["loa_color"]), (hi, se_loa, o["loa_color"])):
            ax.axhspan(c - t * se, c + t * se, color=col, alpha=0.12, lw=0)
    if o["trend"]:
        lr = sps.linregress(mean, diff)
        xs = np.linspace(mean.min(), mean.max(), 50)
        ax.plot(xs, lr.intercept + lr.slope * xs, color="#2a78d6", lw=ctx.s.line_width * 0.9, ls="-.")
        ctx.note(f"比例偏差：斜率 = {lr.slope:.4f}，{_pf(lr.pvalue)}" + ("（有顯著比例偏差）" if lr.pvalue < 0.05 else ""))
    tr = ax.get_yaxis_transform()
    fs = ctx.s.fs(ctx.s.tick_size) * 0.85
    unit = "%" if o["percent"] else ""
    ax.text(1.01, md, f"Mean\n{md:.3g}{unit}", transform=tr, va="center", fontsize=fs, color=o["mean_color"])
    ax.text(1.01, hi, f"+1.96 SD\n{hi:.3g}{unit}", transform=tr, va="center", fontsize=fs, color=o["loa_color"])
    ax.text(1.01, lo, f"−1.96 SD\n{lo:.3g}{unit}", transform=tr, va="center", fontsize=fs, color=o["loa_color"])
    ax.set_xlabel(f"Mean of {m['a']} and {m['b']}")
    ax.set_ylabel(f"{m['a']} − {m['b']}" + (" (%)" if o["percent"] else ""))
    t1 = sps.ttest_1samp(diff, 0)
    ctx.note(f"平均偏差 (bias) = {md:.4g}{unit} (SD {sd:.4g})，95% 一致性界限 {lo:.4g} ~ {hi:.4g}{unit}；偏差是否為 0：{_pf(t1.pvalue)}，n = {n}")


@register("forest", "🌲 森林圖 Forest plot", "生醫專用", "次族群或統合分析的效應量 (HR / OR / RR / 平均差) 與信賴區間。",
          [Field("label", "項目名稱", "any", hints=("subgroup", "study", "name", "label", "variable", "項目")),
           Field("est", "效應估計值", "num", hints=("hr", "or", "rr", "est", "effect", "mean")),
           Field("lo", "CI 下限", "num", hints=("lower", "low", "lcl", "ci_l", "下限")),
           Field("hi", "CI 上限", "num", hints=("upper", "high", "ucl", "ci_u", "上限")),
           Field("weight", "權重 / 樣本數 (標記大小)", "num", required=False, hints=("^n$", "weight", "size"))],
          [Opt("ref", "參考線位置 (無效應)", "float", 1.0, step=0.1), Opt("log", "X 軸取對數 (比值類建議)", "bool", True),
           Opt("text", "右側顯示數值", "bool", True), Opt("header", "數值欄標題", "text", "HR (95% CI)"),
           Opt("decimals", "小數位數", "int", 2, min=0, max=4), Opt("marker", "標記", "select", "s", ["s", "o", "D", "^"]),
           Opt("color", "標記顏色", "color", "#2a78d6"), Opt("bold_first", "第一列粗體 (如 Overall)", "bool", True)],
          facet=False, color_field=None)
def draw_forest(ax, df, m, o, ctx):
    d = df[[m["label"], m["est"], m["lo"], m["hi"]] + ([m["weight"]] if m.get("weight") else [])].copy()
    for c in (m["est"], m["lo"], m["hi"]):
        d[c] = _num(d[c])
    d = d.dropna(subset=[m["est"]]).reset_index(drop=True)
    n = len(d)
    ypos = np.arange(n)[::-1]
    est, lo, hi = d[m["est"]].to_numpy(), d[m["lo"]].to_numpy(), d[m["hi"]].to_numpy()
    if m.get("weight"):
        w = _num(d[m["weight"]]).fillna(0).to_numpy()
        sizes = 4 + 10 * np.sqrt(w / max(w.max(), 1e-12))
    else:
        sizes = np.full(n, ctx.s.marker_size * 1.2)
    for k in range(n):
        ax.plot([lo[k], hi[k]], [ypos[k]] * 2, color="#333333", lw=ctx.s.line_width, solid_capstyle="butt")
        ax.plot(est[k], ypos[k], marker=o["marker"], ms=sizes[k], color=o["color"], mec="#333333", mew=0.6, zorder=3)
    ax.axvline(o["ref"], color="#888888", ls="--", lw=0.9, zorder=0)
    if o["log"]:
        ax.set_xscale("log")
    ax.set_yticks(ypos, d[m["label"]].astype(str))
    if o["bold_first"] and n:
        ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(m["est"] + " (95% CI)")
    if o["text"]:
        tr = ax.get_yaxis_transform()
        dec = o["decimals"]
        for k in range(n):
            ax.text(1.03, ypos[k], f"{est[k]:.{dec}f} ({lo[k]:.{dec}f}–{hi[k]:.{dec}f})", transform=tr, va="center",
                    fontsize=ctx.s.fs(ctx.s.tick_size), fontweight="bold" if (o["bold_first"] and k == 0) else "normal")
        ax.text(1.03, n - 0.3, o["header"], transform=tr, va="bottom", fontsize=ctx.s.fs(ctx.s.tick_size), fontweight="bold")


@register("dose", "💊 劑量反應曲線 (4PL)", "生醫專用", "四參數邏輯斯 (4PL) 曲線擬合，計算 EC50 / IC50。",
          [Field("x", "濃度 / 劑量", "num", hints=("conc", "dose", "um", "nm", "濃度", "劑量")),
           Field("y", "反應值", "num", hints=("resp", "inhib", "viab", "activity", "%", "反應")),
           Field("hue", "化合物 / 分組", "cat", required=False, order=True)],
          [Opt("points", "資料點呈現", "select", "平均 ± SEM", ["平均 ± SEM", "平均 ± SD", "原始資料點", "不顯示"]),
           Opt("ec50_line", "標示 EC50 虛線", "bool", True), Opt("legend_ec50", "圖例顯示 EC50", "bool", True),
           Opt("log", "X 軸對數刻度", "bool", True), CAPSIZE],
          facet=True)
def draw_dose(ax, df, m, o, ctx):
    x, y, hue = m["x"], m["y"], m.get("hue")
    groups = m["hue_order"] if hue else [None]
    pal = ctx.palette_for(hue, groups) if hue else {None: ctx.s.single_color}
    d = df.copy()
    d[x], d[y] = _num(d[x]), _num(d[y])
    d = d.dropna(subset=[x, y])
    d = d[d[x] > 0]
    rows = []
    for g in groups:
        sub = d if g is None else d[d[hue] == g]
        if sub.empty:
            continue
        c = pal[g]
        if o["points"] == "原始資料點":
            ax.scatter(sub[x], sub[y], color=c, s=ctx.ms2 * 1.2, alpha=ctx.s.alpha, edgecolors=ctx.edge("white"), zorder=3)
        elif o["points"] != "不顯示":
            agg = sub.groupby(x)[y].agg(["mean", "std", "count"]).reset_index()
            err = agg["std"] if o["points"] == "平均 ± SD" else agg["std"] / np.sqrt(agg["count"])
            ax.errorbar(agg[x], agg["mean"], yerr=err.fillna(0), fmt="o", color=c, ms=ctx.s.marker_size, capsize=o["capsize"] * 15,
                        lw=ctx.s.line_width, mec="white", zorder=3)
        try:
            f = bstats.fit_4pl(sub[x], sub[y])
        except Exception as err:
            ctx.note(f"{g if g is not None else '資料'}：4PL 擬合失敗 ({err})")
            continue
        xs = np.logspace(np.log10(sub[x].min()), np.log10(sub[x].max()), 300)
        label = str(g) if g is not None else "4PL fit"
        if o["legend_ec50"]:
            label += f" (EC50 = {f['ec50']:.3g})"
        ax.plot(xs, bstats.four_pl(np.log10(xs), *f["params"]), color=c, lw=ctx.s.line_width * 1.3, label=label)
        if o["ec50_line"]:
            ymid = bstats.four_pl(f["logec50"], *f["params"])
            ax.plot([f["ec50"], f["ec50"]], [ax.get_ylim()[0], ymid], color=c, ls=":", lw=ctx.s.line_width * 0.8)
        rows.append({"組別": label.split(" (")[0], "EC50": f["ec50"], "EC50 95% CI": f"{f['ec50_low']:.3g} – {f['ec50_high']:.3g}",
                     "Hill": f["hill"], "Bottom": f["bottom"], "Top": f["top"], "R²": f["r2"]})
    if o["log"]:
        ax.set_xscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    if rows:
        ctx.table("4PL 參數", pd.DataFrame(rows))
    if (df[x].pipe(_num) <= 0).any():
        ctx.note("濃度 ≤ 0 的資料點無法放在對數軸上，已排除。")


@register("manhattan", "🏙️ Manhattan 圖 (GWAS)", "生醫專用", "全基因組關聯分析：依染色體排列的 −log₁₀ p。",
          [Field("chr", "染色體", "any", hints=("^chr", "chrom", "染色體")), Field("pos", "位置 (bp)", "num", hints=("^bp$", "pos", "position")),
           Field("p", "p 值", "num", hints=("^p$", "pval", "p_value", "p.value")),
           Field("label", "SNP 名稱", "any", required=False, hints=("snp", "rs", "marker", "id"))],
          [Opt("gw", "全基因組顯著門檻", "float", 5e-8, step=1e-8), Opt("suggestive", "提示門檻 (0 = 不畫)", "float", 1e-5, step=1e-6),
           Opt("c1", "顏色 1", "color", "#2a78d6"), Opt("c2", "顏色 2", "color", "#9db9e0"),
           Opt("label_top", "標註各染色體最顯著 SNP (超過門檻)", "bool", True), Opt("point", "點大小", "float", 3.0, min=0.5, max=10.0, step=0.5)],
          facet=False, color_field=None)
def draw_manhattan(ax, df, m, o, ctx):
    d = df[[m["chr"], m["pos"], m["p"]] + ([m["label"]] if m.get("label") else [])].copy()
    d[m["pos"]], d[m["p"]] = _num(d[m["pos"]]), _num(d[m["p"]])
    d = d.dropna(subset=[m["chr"], m["pos"], m["p"]])
    d["_chr"] = as_category_labels(d[m["chr"]]).str.replace("^chr", "", regex=True, case=False)
    chroms = sorted(d["_chr"].unique(), key=natural_key)
    d["_nlp"] = -np.log10(d[m["p"]].clip(lower=1e-300))
    offset, ticks = 0.0, []
    d["_x"] = np.nan
    for k, c in enumerate(chroms):
        sel = d["_chr"] == c
        mx = d.loc[sel, m["pos"]].max()
        d.loc[sel, "_x"] = d.loc[sel, m["pos"]] + offset
        ticks.append(offset + mx / 2)
        offset += mx * 1.02
    colors = np.where(d["_chr"].map({c: k % 2 for k, c in enumerate(chroms)}) == 0, o["c1"], o["c2"])
    ax.scatter(d["_x"], d["_nlp"], c=colors, s=o["point"] ** 2, lw=0, rasterized=len(d) > 5000)
    ax.axhline(-np.log10(o["gw"]), color="#e34948", ls="--", lw=0.9)
    if o["suggestive"] > 0:
        ax.axhline(-np.log10(o["suggestive"]), color="#888888", ls=":", lw=0.9)
    if o["label_top"] and m.get("label"):
        hits = d[d[m["p"]] < o["gw"]]
        if len(hits):
            top = hits.loc[hits.groupby("_chr")["_nlp"].idxmax()]
            label_points(ax, top["_x"].to_numpy(float), top["_nlp"].to_numpy(float), top[m["label"]].astype(str).tolist(),
                         ctx.s.fs(ctx.s.tick_size) * 0.8)
    ax.set_xticks(ticks, chroms)
    ax.set_xlim(-offset * 0.01, offset)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_ylim(0, max(d["_nlp"].max() * 1.08, -np.log10(o["gw"]) * 1.1))
    ctx.note(f"共 {len(d)} 個 SNP，{int((d[m['p']] < o['gw']).sum())} 個達全基因組顯著 (p < {o['gw']:g})。")


# ============================================================ 組成與其他
@register("stacked", "🧱 堆疊長條圖 (組成/百分比)", "組成與其他", "各組的組成比例，例如菌相、細胞類型、反應分級。",
          [Field("x", "X 軸 (樣本/組別)", "cat", order=True), Field("hue", "堆疊類別", "cat", order=True),
           Field("value", "數值 (空白 = 計數)", "num", required=False)],
          [Opt("percent", "轉換為 100% 百分比", "bool", True), Opt("agg", "彙整方式", "select", "sum", ["sum", "mean", "median"]),
           Opt("horizontal", "水平長條", "bool", False), Opt("labels", "顯示數值 (超過門檻)", "bool", False),
           Opt("label_min", "數值顯示門檻", "float", 5.0, min=0.0, step=1.0), WIDTH])
def draw_stacked(ax, df, m, o, ctx):
    x, hue = m["x"], m["hue"]
    if m.get("value"):
        tab = df.assign(_v=_num(df[m["value"]])).pivot_table(index=x, columns=hue, values="_v", aggfunc=o["agg"])
    else:
        tab = pd.crosstab(df[x], df[hue])
    tab = tab.reindex(index=m["x_order"], columns=m["hue_order"]).fillna(0)
    if o["percent"]:
        tab = tab.div(tab.sum(1).replace(0, np.nan), axis=0) * 100
    pal = ctx.palette_for(hue, m["hue_order"])
    pos = np.arange(len(tab))
    base = np.zeros(len(tab))
    for h in tab.columns:
        vals = tab[h].to_numpy()
        kw = dict(color=pal[h], edgecolor=ctx.edge("white"), linewidth=0.8, label=str(h))
        bars = ax.barh(pos, vals, left=base, height=o["width"], **kw) if o["horizontal"] else ax.bar(pos, vals, bottom=base, width=o["width"], **kw)
        if o["labels"]:
            ax.bar_label(bars, labels=[f"{v:.1f}" if v >= o["label_min"] else "" for v in vals], label_type="center",
                         fontsize=ctx.s.fs(ctx.s.tick_size) * 0.75, color="white")
        base += vals
    if o["horizontal"]:
        ax.set_yticks(pos, tab.index)
        ax.invert_yaxis()
        ax.set_xlabel("Percentage (%)" if o["percent"] else (m.get("value") or "Count"))
        ax.set_ylabel(x)
    else:
        ax.set_xticks(pos, tab.index)
        ax.set_ylabel("Percentage (%)" if o["percent"] else (m.get("value") or "Count"))
        ax.set_xlabel(x)
    if o["percent"]:
        (ax.set_xlim if o["horizontal"] else ax.set_ylim)(0, 100)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], title=hue, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ctx.table("組成表", tab.reset_index())


@register("pie", "🥧 圓餅圖 / 甜甜圈圖", "組成與其他", "類別比例 (類別少時使用；多類別建議用長條圖)。",
          [Field("x", "類別", "cat", order=True), Field("value", "數值 (空白 = 計數)", "num", required=False)],
          [Opt("donut", "甜甜圈 (中空)", "bool", False), Opt("pct", "顯示百分比", "bool", True),
           Opt("decimals", "百分比小數位", "int", 1, min=0, max=3), Opt("labels", "標籤位置", "select", "圓外", ["圓外", "圖例"]),
           Opt("startangle", "起始角度", "int", 90, min=0, max=360), Opt("explode", "突出第一塊", "bool", False)],
          facet=True, axes_style=False, color_field="x")
def draw_pie(ax, df, m, o, ctx):
    x = m["x"]
    vals = df.groupby(x)[m["value"]].sum() if m.get("value") else df[x].value_counts()
    vals = vals.reindex(m["x_order"]).fillna(0)
    vals = vals[vals > 0]
    pal = ctx.palette_for(x, list(vals.index))
    wedge = dict(edgecolor=ctx.edge("white"), linewidth=1.5)
    if o["donut"]:
        wedge["width"] = 0.42
    explode = [0.06 if (o["explode"] and k == 0) else 0 for k in range(len(vals))]
    res = ax.pie(vals.values, labels=list(vals.index) if o["labels"] == "圓外" else None, colors=[pal[k] for k in vals.index],
                 autopct=(f"%.{o['decimals']}f%%" if o["pct"] else None), startangle=o["startangle"], counterclock=False,
                 wedgeprops=wedge, explode=explode, pctdistance=0.79 if o["donut"] else 0.62,
                 textprops=dict(fontsize=ctx.s.fs(ctx.s.tick_size)))
    if o["labels"] == "圖例":
        ax.legend(res[0], list(vals.index), title=x, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_aspect("equal")


@register("summary_bar", "📋 摘要數據誤差棒圖 (已算好平均/SD)", "組成與其他", "Excel 中已有平均值與誤差欄 (如 Mean、SD) 時直接作圖。",
          [Field("x", "X 軸 (組別)", "cat", order=True), Field("mean", "平均值欄", "num", hints=("mean", "avg", "平均")),
           Field("err", "誤差欄 (SD/SEM)", "num", hints=("sd", "sem", "se", "err", "std", "誤差")),
           Field("hue", "顏色分組", "cat", required=False, order=True)],
          [Opt("kind", "樣式", "select", "長條", ["長條", "點"]), Opt("one_side", "誤差線只畫上方", "bool", False), CAPSIZE, WIDTH,
           COLOR_BY_X], sig=False)
def draw_summary_bar(ax, df, m, o, ctx):
    x, hue = m["x"], m.get("hue")
    use_hue = bool(hue) and hue != x
    pos = annotate.category_positions(m["x_order"], m["hue_order"] if use_hue else None, o["width"])
    nh = len(m["hue_order"]) if use_hue else 1
    w = o["width"] / nh
    pal = ctx.palette_for(hue, m["hue_order"]) if use_hue else (ctx.palette_for(x, m["x_order"]) if o["color_by_x"] else None)
    seen = set()
    for (xl, hl), p in pos.items():
        sel = df[x] == xl
        if use_hue:
            sel &= df[hue] == hl
        if not sel.any():
            continue
        r = df.loc[sel].iloc[0]
        mu, e = float(r[m["mean"]]), float(r[m["err"]])
        c = pal[hl] if use_hue else (pal[xl] if pal else ctx.s.single_color)
        yerr = [[0], [e]] if o["one_side"] else [[e], [e]]
        lab = str(hl) if use_hue and hl not in seen else None
        seen.add(hl)
        if o["kind"] == "長條":
            ax.bar(p, mu, width=w * 0.95, color=c, edgecolor=ctx.edge(None), label=lab,
                   yerr=yerr, capsize=o["capsize"] * 20, error_kw=dict(lw=ctx.s.line_width, ecolor="#222222"))
        else:
            ax.errorbar(p, mu, yerr=yerr, fmt="o", color=c, ms=ctx.s.marker_size * 1.2, capsize=o["capsize"] * 20,
                        lw=ctx.s.line_width, label=lab)
    ax.set_xticks(range(len(m["x_order"])), m["x_order"])
    ax.set_xlabel(x)
    ax.set_ylabel(m["mean"])
    if use_hue:
        ax.legend(title=hue)


@register("venn", "⭕ 文氏圖 Venn diagram (2–3 組)", "組成與其他", "每個欄位是一組清單 (如基因名)，計算交集數量。",
          [Field("sets", "集合欄位 (2–3 欄)", "anys", default_n=3)],
          [Opt("alpha", "填色透明度", "float", 0.4, min=0.1, max=0.9, step=0.05), Opt("pct", "同時顯示百分比", "bool", False),
           Opt("outline", "外框線", "bool", True)],
          facet=False, axes_style=False, color_field=None)
def draw_venn(ax, df, m, o, ctx):
    cols = m["sets"][:3]
    if len(cols) < 2:
        raise ValueError("請選擇 2 或 3 個欄位")
    sets = [set(df[c].dropna().astype(str).str.strip()) - {""} for c in cols]
    colors = ctx.s.colors(len(cols))
    total = len(set().union(*sets)) or 1
    fs = ctx.s.fs(ctx.s.tick_size) * 1.1

    def lab(n):
        return f"{n}\n({n / total:.0%})" if o["pct"] else str(n)

    regions = []
    if len(cols) == 2:
        A, B = sets
        centers = [(-0.55, 0), (0.55, 0)]
        regions = [((-1.0, 0), A - B, "only " + cols[0]), ((0, 0), A & B, "∩"), ((1.0, 0), B - A, "only " + cols[1])]
        name_pos = [(-0.9, 1.15), (0.9, 1.15)]
    else:
        A, B, C = sets
        centers = [(-0.55, 0.35), (0.55, 0.35), (0, -0.55)]
        regions = [((-1.0, 0.7), A - B - C, cols[0]), ((1.0, 0.7), B - A - C, cols[1]), ((0, -1.15), C - A - B, cols[2]),
                   ((0, 0.85), (A & B) - C, f"{cols[0]}∩{cols[1]}"), ((-0.62, -0.3), (A & C) - B, f"{cols[0]}∩{cols[2]}"),
                   ((0.62, -0.3), (B & C) - A, f"{cols[1]}∩{cols[2]}"), ((0, 0.05), A & B & C, "全部交集")]
        name_pos = [(-1.3, 1.62), (1.3, 1.62), (0, -1.85)]
    for (cx, cy), c in zip(centers, colors):
        ax.add_patch(Circle((cx, cy), 1.0, facecolor=c, alpha=o["alpha"], edgecolor=c if o["outline"] else "none", lw=ctx.s.line_width))
    for (px, py), members, _ in regions:
        ax.text(px, py, lab(len(members)), ha="center", va="center", fontsize=fs)
    for (px, py), name, k in zip(name_pos, cols, range(len(cols))):
        ax.text(px, py, f"{name}\n(n = {len(sets[k])})", ha="center", va="center", fontsize=fs, fontweight="bold", color=colors[k])
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.15 if len(cols) == 3 else -1.4, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    m_len = max(len(r[1]) for r in regions)
    ctx.table("交集成員", pd.DataFrame({name: pd.Series(sorted(mem) + [""] * (m_len - len(mem))) for _, mem, name in regions}))


@register("radar", "🕸️ 雷達圖 Radar chart", "組成與其他", "多個指標的組別輪廓比較 (各指標可標準化到 0–1)。",
          [Field("values", "指標 (數值，≥ 3)", "nums", default_n=6), Field("hue", "組別", "cat", required=False, order=True)],
          [Opt("normalize", "各指標標準化到 0–1", "bool", True), Opt("fill_alpha", "填色透明度", "float", 0.15, min=0.0, max=0.8, step=0.05),
           Opt("agg", "組內彙整", "select", "mean", ["mean", "median"])],
          figure_level=True, facet=False, axes_style=False)
def draw_radar(df, m, o, ctx):
    cols = m["values"]
    if len(cols) < 3:
        raise ValueError("雷達圖至少需要 3 個指標")
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    if m.get("hue"):
        prof = d.groupby(df[m["hue"]]).agg(o["agg"]).reindex(m["hue_order"])
    else:
        prof = d.agg(o["agg"]).to_frame("All").T
    if o["normalize"]:
        rng = (prof.max() - prof.min()).replace(0, 1)
        prof = (prof - prof.min()) / rng if len(prof) > 1 else prof / prof.abs().max().replace(0, 1)
    ang = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(ctx.s.width, ctx.s.height), subplot_kw=dict(polar=True))
    pal = ctx.palette_for(m.get("hue"), list(prof.index)) if m.get("hue") else {"All": ctx.s.single_color}
    for g, row in prof.iterrows():
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(ang, vals, color=pal[g], lw=ctx.s.line_width * 1.3, label=str(g), marker="o", ms=ctx.s.marker_size * 0.7)
        ax.fill(ang, vals, color=pal[g], alpha=o["fill_alpha"])
    ax.set_xticks(ang[:-1], cols)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.tick_params(labelsize=ctx.s.fs(ctx.s.tick_size))
    if m.get("hue"):
        ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.05), title=m["hue"])
    ctx.table("組別輪廓", prof.reset_index())
    return PlotResult(fig, [ax], title_on_fig=False)


# ============================================================ 框架
def prepare(spec: ChartSpec, df: pd.DataFrame, m: dict, orders: dict, facet: str | None) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    m2 = dict(m)
    for f in spec.fields:
        col = m.get(f.key)
        if f.required and (not col or (isinstance(col, list) and len(col) == 0)):
            raise ValueError(f"請選擇「{f.label}」")
        if not col:
            continue
        if f.kind in ("nums", "anys"):
            missing = [c for c in col if c not in work.columns]
            if missing:
                raise ValueError(f"找不到欄位：{', '.join(missing)}")
            continue
        if col not in work.columns:
            raise ValueError(f"找不到欄位「{col}」")
        if f.kind == "cat" or (f.kind == "hue" and is_categorical_like(df[col])):
            work[col] = as_category_labels(work[col])
            lv = [v for v in (orders.get(col) or category_levels(df[col]))]
            work = work[work[col].isin(lv)]
            present = set(work[col].dropna())
            m2[f.key + "_order"] = [v for v in lv if v in present]
        elif f.kind == "num":
            work[col] = _num(work[col])
    if facet:
        work[facet] = as_category_labels(work[facet])
        lv = orders.get(facet) or category_levels(df[facet])
        work = work[work[facet].isin(lv)]
        m2["facet_order"] = [v for v in lv if v in set(work[facet])]
    return work.reset_index(drop=True), m2


def _sig_on(ax, df, m, o, ctx, sig: dict, table_rows: list, facet_label=None):
    if _horiz(o):
        ctx.note("顯著性標記目前僅支援垂直方向的圖。")
        return
    x, y, hue = m["x"], m["y"], m.get("hue")
    tab = annotate.add_significance(
        ax, df, x, y, hue if hue and hue != x else None, m["x_order"], m.get("hue_order") if hue and hue != x else None,
        test=sig["test"], pairs_mode=sig["pairs_mode"], ref=sig.get("ref"), correction=sig.get("correction"),
        display=sig["display"], hide_ns=sig["hide_ns"], line_width=sig.get("line_width", 1.0),
        font_size=sig.get("font_size", 10.0), width=ctx.sig_width, dodge=True)
    if len(tab):
        if facet_label is not None:
            tab.insert(0, "分面", facet_label)
        table_rows.append(tab)


def render(key: str, df: pd.DataFrame, m: dict, o: dict, style: Style, *, orders: dict | None = None,
           facet: str | None = None, facet_wrap: int = 3, share_y: bool = True, sig: dict | None = None,
           overrides: dict | None = None) -> PlotResult:
    spec = CHARTS[key]
    o = {**{op.key: op.default for op in spec.opts}, **(o or {})}
    orders = orders or {}
    facet = facet if (facet and spec.facet and not spec.figure_level) else None
    with PLOT_LOCK:
        cjk = fonts.ensure_cjk_font() if style.use_cjk else None
        rc = build_rc(style, cjk)
        with plt.rc_context(rc), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            work, m2 = prepare(spec, df, m, orders, facet)
            if work.empty:
                raise ValueError("篩選後沒有資料可繪製")
            ctx = Ctx(style, overrides)
            sig_tables: list = []
            use_sig = bool(sig and sig.get("enabled") and spec.sig)
            if spec.figure_level:
                res = spec.draw(work, m2, o, ctx)
            elif facet:
                lv = m2["facet_order"]
                ncols = max(1, min(facet_wrap, len(lv)))
                nrows = math.ceil(len(lv) / ncols)
                fig, axs = plt.subplots(nrows, ncols, figsize=(style.width * ncols, style.height * nrows), sharey=share_y,
                                        squeeze=False)
                axes = []
                for k, level in enumerate(lv):
                    ax = axs.flat[k]
                    ax._biostat_ncols = ncols
                    sub = work[work[facet] == level]
                    spec.draw(ax, sub, m2, o, ctx)
                    ax.set_title(f"{facet} = {level}", fontsize=style.fs(style.label_size))
                    if use_sig:
                        _sig_on(ax, sub, m2, o, ctx, sig, sig_tables, facet_label=level)
                    if share_y and k % ncols:
                        ax.set_ylabel("")
                    axes.append(ax)
                for k in range(len(lv), nrows * ncols):
                    axs.flat[k].remove()
                res = PlotResult(fig, axes, title_on_fig=True)
            else:
                fig, ax = ctx.subplots()
                spec.draw(ax, work, m2, o, ctx)
                if use_sig:
                    _sig_on(ax, work, m2, o, ctx, sig, sig_tables)
                res = PlotResult(fig, [ax])
            finalize(res.fig, res.axes, style, grid=res.grid, facet=bool(facet), axes_style=spec.axes_style,
                     despine=spec.despine, title_on_fig=res.title_on_fig)
            if not spec.axes_style:
                for ax in res.axes:
                    ax.tick_params(labelsize=style.fs(style.tick_size))
            res.notes = ctx.notes + res.notes
            res.tables = ctx.tables + res.tables
            if sig_tables:
                st = pd.concat(sig_tables, ignore_index=True)
                res.tables.insert(0, ("顯著性檢定結果", st))
            res.rc = rc
    return res


def charts_in(category: str) -> list[str]:
    return [k for k, c in CHARTS.items() if c.category == category]
