"""圖表格式設定：Style 資料結構、rcParams 組裝、繪圖後的統一格式化 (標題、座標軸、圖例…)。"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass, field, fields

import seaborn as sns
from cycler import cycler

from . import fonts, palettes

THEMES = {"ticks": "刻度 (ticks，期刊常用)", "white": "白底", "whitegrid": "白底格線", "darkgrid": "灰底格線", "dark": "灰底"}
CONTEXTS = {"paper": "論文 (paper)", "notebook": "一般 (notebook)", "talk": "簡報 (talk)", "poster": "海報 (poster)"}
GRID_OPTIONS = ["主題預設", "無", "Y 軸", "X 軸", "兩者"]
EDGE_OPTIONS = {"自動": None, "白色": "white", "黑色": "black", "深灰": "#333333", "無": "none"}
LEGEND_LOCS = {
    "自動": ("best", None),
    "右上": ("upper right", None),
    "左上": ("upper left", None),
    "右下": ("lower right", None),
    "左下": ("lower left", None),
    "上方置中": ("upper center", None),
    "圖外右側": ("center left", (1.02, 0.5)),
    "圖外上方": ("lower center", (0.5, 1.02)),
    "圖外下方": ("upper center", (0.5, -0.18)),
    "隱藏": (None, None),
}


@dataclass
class Style:
    # 版面
    width: float = 6.0
    height: float = 4.5
    theme: str = "ticks"
    context: str = "notebook"
    font_scale: float = 1.0
    font: str = field(default_factory=fonts.default_font)
    use_cjk: bool = True
    bg_color: str = "#ffffff"
    transparent: bool = False
    # 色彩
    palette: str = "預設 (色盲友善)"
    custom_colors: str = ""
    single_color: str = "#2a78d6"
    cmap: str = "viridis"
    reverse_cmap: bool = False
    # 文字
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    hide_xlabel: bool = False
    hide_ylabel: bool = False
    title_size: float = 14.0
    label_size: float = 12.0
    tick_size: float = 10.0
    title_bold: bool = True
    label_bold: bool = False
    title_loc: str = "center"
    text_color: str = "#000000"
    xtick_rotation: int = 0
    xtick_wrap: int = 12  # X 軸類別文字每行最多幾個字寬 (中文 1 字 = 2)；0 = 不換行
    xtick_auto_rotate: bool = True  # 角度為 0 且文字仍太擠時自動旋轉 30°
    ytick_rotation: int = 0
    # 座標軸
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    xlog: bool = False
    ylog: bool = False
    invert_x: bool = False
    invert_y: bool = False
    grid: str = "主題預設"
    despine: bool = True
    spine_width: float = 1.0
    tick_direction: str = "out"
    tick_length: float = 4.0
    # 圖例
    legend_loc: str = "圖外右側"  # 預設放在圖外，避免蓋住資料
    legend_title: str = ""
    hide_legend_title: bool = False
    legend_frame: bool = False
    legend_size: float = 10.0
    legend_ncol: int = 1
    # 標記與線條
    alpha: float = 0.85
    marker_size: float = 6.0
    line_width: float = 1.5
    edge_color: str = "自動"
    # 匯出
    dpi: int = 300
    preview_dpi: int = 110
    tight: bool = True
    editable_text: bool = True

    # ---- helpers ----
    def fs(self, size: float) -> float:
        return float(size) * float(self.font_scale)

    @property
    def edge(self):
        return EDGE_OPTIONS.get(self.edge_color)

    @property
    def cmap_name(self) -> str:
        name = self.cmap
        if self.reverse_cmap:
            name = name[:-2] if name.endswith("_r") else name + "_r"
        return name

    def colors(self, n: int) -> list:
        return palettes.get_palette(self.palette, n, self.custom_colors)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Style":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


PRESETS: dict[str, dict] = {
    "預設": {},
    "期刊投稿 (Nature / Science)": dict(
        width=3.5, height=2.8, theme="ticks", context="paper", font_scale=1.0, title_size=9, label_size=8,
        tick_size=7, legend_size=7, title_bold=True, spine_width=0.8, line_width=1.0, marker_size=4,
        tick_length=3, dpi=600, palette="NPG (Nature 風格)", despine=True,
    ),
    "GraphPad Prism 風格": dict(
        width=4.5, height=4.0, theme="ticks", context="notebook", title_size=14, label_size=14, tick_size=12,
        title_bold=True, label_bold=True, spine_width=1.5, tick_length=6, line_width=1.5, marker_size=7,
        despine=True, palette="GraphPad 風格", edge_color="黑色", dpi=600,
    ),
    "簡報 (大字體)": dict(
        width=8, height=5.5, theme="whitegrid", context="talk", title_size=20, label_size=16, tick_size=14,
        legend_size=14, line_width=2.5, marker_size=9, spine_width=1.5, dpi=200,
    ),
    "海報": dict(
        width=10, height=7, theme="ticks", context="poster", title_size=26, label_size=22, tick_size=18,
        legend_size=18, line_width=3, marker_size=11, spine_width=2, dpi=300,
    ),
    "極簡黑白印刷": dict(
        theme="white", palette="灰階 (黑白印刷)", single_color="#555555", edge_color="黑色", grid="無",
        despine=True, cmap="Greys",
    ),
}


def build_rc(s: Style, cjk_font: str | None) -> dict:
    """組合 seaborn 主題與使用者設定成 rcParams 字典 (搭配 matplotlib.rc_context 使用，不污染全域狀態)。"""
    rc: dict = {}
    rc.update(sns.axes_style(s.theme))
    rc.update(sns.plotting_context(s.context))
    face = s.bg_color or "#ffffff"
    rc.update(
        {
            "font.family": fonts.font_family(s.font, cjk_font if s.use_cjk else None),
            "font.sans-serif": fonts.font_family(s.font, cjk_font if s.use_cjk else None) + ["DejaVu Sans"],
            "axes.unicode_minus": True,
            "axes.titlesize": s.fs(s.title_size),
            "axes.titleweight": "bold" if s.title_bold else "normal",
            "figure.titlesize": s.fs(s.title_size) * 1.1,
            "figure.titleweight": "bold" if s.title_bold else "normal",
            "axes.labelsize": s.fs(s.label_size),
            "axes.labelweight": "bold" if s.label_bold else "normal",
            "xtick.labelsize": s.fs(s.tick_size),
            "ytick.labelsize": s.fs(s.tick_size),
            "legend.fontsize": s.fs(s.legend_size),
            "legend.title_fontsize": s.fs(s.legend_size),
            "legend.frameon": s.legend_frame,
            "font.size": s.fs(s.tick_size),
            "axes.linewidth": s.spine_width,
            "xtick.major.width": s.spine_width,
            "ytick.major.width": s.spine_width,
            "xtick.minor.width": s.spine_width * 0.7,
            "ytick.minor.width": s.spine_width * 0.7,
            "xtick.major.size": s.tick_length,
            "ytick.major.size": s.tick_length,
            "xtick.direction": s.tick_direction,
            "ytick.direction": s.tick_direction,
            "xtick.bottom": True,
            "ytick.left": True,
            "lines.linewidth": s.line_width,
            "lines.markersize": s.marker_size,
            "patch.linewidth": max(s.line_width * 0.6, 0.5),
            "text.color": s.text_color,
            "axes.labelcolor": s.text_color,
            "axes.titlecolor": s.text_color,
            "xtick.color": s.text_color,
            "ytick.color": s.text_color,
            "figure.facecolor": face,
            "savefig.facecolor": face,
            "axes.prop_cycle": cycler(color=s.colors(10)),
            "image.cmap": s.cmap_name,
            "figure.dpi": 100,
            "pdf.fonttype": 42 if s.editable_text else 3,
            "ps.fonttype": 42 if s.editable_text else 3,
            "svg.fonttype": "none" if s.editable_text else "path",
        }
    )
    if s.theme in ("white", "ticks", "whitegrid"):
        rc["axes.facecolor"] = face
    return rc


def _disp_len(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def wrap_label(text: str, width: int) -> str:
    """長類別名稱自動換行：優先在空白處斷行，過長的單字或中文依顯示寬度硬斷。"""
    if width <= 0 or "$" in text or _disp_len(text) <= width:
        return text
    lines, cur = [], ""
    for word in text.split():
        while _disp_len(word) > width:
            n, cut = 0, 0
            for i, ch in enumerate(word):
                n += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
                if n > width:
                    break
                cut = i + 1
            cut = max(cut, 1)
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:cut])
            word = word[cut:]
        cand = f"{cur} {word}" if cur else word
        if _disp_len(cand) <= width:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _is_number(t: str) -> bool:
    try:
        float(t.replace("−", "-").replace(",", ""))
        return True
    except ValueError:
        return False


def tidy_xticklabels(ax, s: Style) -> None:
    """X 軸刻度文字：類別名稱過長時換行；手動角度優先，否則仍太擠時自動旋轉 30°。"""
    ticks = ax.get_xticks()
    try:
        labels = [str(t) for t in ax.xaxis.get_major_formatter().format_ticks(ticks)]
    except Exception:
        labels = []
    categorical = any(t and "$" not in t and not _is_number(t) for t in labels)
    if categorical and s.xtick_wrap > 0:
        wrapped = [wrap_label(t, s.xtick_wrap) for t in labels]
        if wrapped != labels:
            ax.set_xticks(ticks, labels=wrapped)
            labels = wrapped
    rot = s.xtick_rotation
    if not rot and categorical and s.xtick_auto_rotate:
        lo, hi = sorted(ax.get_xlim())
        shown = [t for x, t in zip(ticks, labels) if lo <= x <= hi and t]
        if shown:
            em = s.fs(s.tick_size)  # 字級 (pt)；半形字約 0.55 em
            widest = max(_disp_len(line) for t in shown for line in t.split("\n")) * 0.55 * em
            slot = ax.figure.get_size_inches()[0] * 72 * ax.get_position().width / len(shown)
            if widest > slot * 0.92:
                rot = 30
    if rot:
        ax.tick_params(axis="x", labelrotation=rot)
        for lab in ax.get_xticklabels():
            lab.set_ha("right" if 0 < rot < 90 else ("left" if -90 < rot < 0 else "center"))
            lab.set_rotation_mode("anchor")


def _apply_legend(s: Style, axes, grid=None, facet: bool = False) -> None:
    loc, anchor = LEGEND_LOCS.get(s.legend_loc, ("best", None))
    targets = []
    if grid is not None and getattr(grid, "_legend", None) is not None:
        targets.append(grid)
    with_leg = [ax for ax in axes if ax.get_legend() is not None]
    if facet and len(with_leg) > 1:
        # 分面圖只保留一個圖例 (第一列最右邊那格)
        keep = with_leg[min(len(with_leg) - 1, getattr(with_leg[0], "_biostat_ncols", 1) - 1)]
        for ax in with_leg:
            if ax is not keep:
                ax.get_legend().remove()
        with_leg = [keep]
    targets.extend(with_leg)

    for obj in targets:
        leg = obj._legend if obj is grid else obj.get_legend()
        if leg is None:
            continue
        if loc is None:
            leg.remove()
            if obj is grid:
                grid._legend = None
            continue
        old_title = leg.get_title().get_text()
        title = s.legend_title or old_title
        if s.hide_legend_title:
            title = None
        if s.legend_loc == "自動":
            # 保留圖表原本的位置 (例如堆疊長條圖放在圖外)，只更新外觀
            leg.set_frame_on(s.legend_frame)
            for t in leg.get_texts():
                t.set_fontsize(s.fs(s.legend_size))
            leg.set_title(title or "", prop={"size": s.fs(s.legend_size)})
            if s.legend_ncol > 1 and hasattr(leg, "set_ncols"):
                leg.set_ncols(s.legend_ncol)
            if s.legend_frame:
                leg.get_frame().set_edgecolor("#888888")
            continue
        n_entries = len(leg.get_texts())
        ncol = s.legend_ncol
        if s.legend_loc in ("圖外上方", "圖外下方") and ncol == 1:
            ncol = max(n_entries, 1)
        kwargs = dict(
            title=title,
            frameon=s.legend_frame,
            fontsize=s.fs(s.legend_size),
            title_fontsize=s.fs(s.legend_size),
            ncol=ncol,
        )
        if anchor is not None:
            kwargs["bbox_to_anchor"] = anchor
            kwargs["borderaxespad"] = 0
        try:
            sns.move_legend(obj, loc, **kwargs)
        except Exception:
            pass
        new = obj._legend if obj is grid else obj.get_legend()
        if new is not None and s.legend_frame:
            new.get_frame().set_edgecolor("#888888")


def finalize(fig, axes, s: Style, *, grid=None, facet: bool = False, axes_style: bool = True, despine: bool = True,
             label_axes: bool = True, title_on_fig: bool | None = None) -> None:
    """繪圖完成後統一套用使用者格式設定 (需在 rc_context 內呼叫)。"""
    axes = [ax for ax in axes if ax is not None]
    if axes_style:
        for ax in axes:
            if label_axes:
                if s.xlabel and ax.get_xlabel():
                    ax.set_xlabel(s.xlabel)
                if s.ylabel and ax.get_ylabel():
                    ax.set_ylabel(s.ylabel)
            if s.hide_xlabel:
                ax.set_xlabel("")
            if s.hide_ylabel:
                ax.set_ylabel("")
            ax.xaxis.label.set_size(s.fs(s.label_size))
            ax.yaxis.label.set_size(s.fs(s.label_size))
            ax.xaxis.label.set_weight("bold" if s.label_bold else "normal")
            ax.yaxis.label.set_weight("bold" if s.label_bold else "normal")
            if s.xlog:
                ax.set_xscale("log")
            if s.ylog:
                ax.set_yscale("log")
            if s.xmin is not None or s.xmax is not None:
                ax.set_xlim(left=s.xmin, right=s.xmax)
            if s.ymin is not None or s.ymax is not None:
                ax.set_ylim(bottom=s.ymin, top=s.ymax)
            if s.invert_x:
                ax.invert_xaxis()
            if s.invert_y:
                ax.invert_yaxis()
            if s.grid != "主題預設":
                ax.grid(False)
                kw = dict(color="#d0d0d0", linewidth=0.7, linestyle="-", zorder=0)
                if s.grid in ("Y 軸", "兩者"):
                    ax.yaxis.grid(True, **kw)
                if s.grid in ("X 軸", "兩者"):
                    ax.xaxis.grid(True, **kw)
                ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=s.fs(s.tick_size), direction=s.tick_direction, length=s.tick_length,
                           width=s.spine_width, colors=s.text_color)
            tidy_xticklabels(ax, s)
            if s.ytick_rotation:
                for lab in ax.get_yticklabels():
                    lab.set_rotation(s.ytick_rotation)
            for sp in ax.spines.values():
                sp.set_linewidth(s.spine_width)
            if despine and s.despine:
                sns.despine(ax=ax)

    if s.title:
        on_fig = title_on_fig if title_on_fig is not None else (len(axes) != 1 or grid is not None)
        weight = "bold" if s.title_bold else "normal"
        if on_fig:
            # y > 1：放在所有子圖上方，避免壓到 seaborn 圖層級圖表 (pairplot、jointplot…) 的頂端
            fig.suptitle(s.title, fontsize=s.fs(s.title_size) * 1.1, fontweight=weight, color=s.text_color, y=1.02)
        else:
            axes[0].set_title(s.title, loc=s.title_loc, fontsize=s.fs(s.title_size), fontweight=weight, color=s.text_color)

    _apply_legend(s, axes, grid=grid, facet=facet)

    if s.transparent:
        fig.patch.set_alpha(0.0)
        for ax in axes:
            ax.patch.set_alpha(0.0)
