"""自訂 Python 程式碼：讓使用者用 matplotlib / seaborn 畫出任何圖。"""

from __future__ import annotations

import contextlib
import io
import traceback

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sps

from . import fonts
from . import stats as bstats
from .plots import PLOT_LOCK
from .style import Style, build_rc

_PRELUDE = """\
# 可用變數：df (目前資料)、pd、np、plt、sns、stats (scipy.stats)、sm、smf (statsmodels)、bstats (本程式統計函式)
# 程式中建立的所有 matplotlib 圖都會顯示在下方，並可下載。
_nums = df.select_dtypes("number").columns.tolist()
num_cols = [c for c in _nums if df[c].nunique() > 10] or _nums          # 連續數值欄 (排除 0/1、週數這類少數值欄)
cat_cols = [c for c in df.columns if c not in num_cols and 2 <= df[c].nunique() <= 12] \\
    or [c for c in df.columns if c not in num_cols]                      # 適合當分組的欄 (2–12 類)
"""

TEMPLATES: dict[str, str] = {
    "盒鬚圖 + 散點 + 顯著性 (入門)": _PRELUDE + """
x, y = (cat_cols[0] if cat_cols else None), num_cols[0]
fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(data=df, x=x, y=y, hue=x, legend=False, width=0.6, showfliers=False, ax=ax)
sns.stripplot(data=df, x=x, y=y, color="black", size=4, jitter=0.15, ax=ax)

# 手動加一個顯著性括號：比較前兩組
groups = df[x].dropna().unique()[:2]
a, b = (df.loc[df[x] == g, y].dropna() for g in groups)
p = stats.ttest_ind(a, b, equal_var=False).pvalue
top = df[y].max() * 1.05
ax.plot([0, 0, 1, 1], [top, top * 1.02, top * 1.02, top], color="black", lw=1)
ax.text(0.5, top * 1.03, f"p = {p:.3g}", ha="center", va="bottom")
ax.set_title(f"{y} by {x}")
sns.despine()
""",
    "多子圖組合 (論文 Figure A/B/C/D)": _PRELUDE + """
fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
x = cat_cols[0] if cat_cols else None
sns.violinplot(data=df, x=x, y=num_cols[0], hue=x, legend=False, inner="quart", ax=axes[0, 0])
sns.barplot(data=df, x=x, y=num_cols[1 % len(num_cols)], hue=x, legend=False, errorbar="se", capsize=0.2, ax=axes[0, 1])
sns.regplot(data=df, x=num_cols[0], y=num_cols[-1], ax=axes[1, 0], scatter_kws=dict(s=20, alpha=0.7))
sns.heatmap(df[num_cols].corr(), cmap="RdBu_r", center=0, annot=True, fmt=".2f", square=True, cbar_kws={"shrink": 0.7}, ax=axes[1, 1])
for ax, letter in zip(axes.flat, "ABCD"):
    ax.text(-0.15, 1.08, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
for ax in axes.flat[:3]:
    sns.despine(ax=ax)
""",
    "分面網格 FacetGrid": _PRELUDE + """
col = cat_cols[1] if len(cat_cols) > 1 else cat_cols[0]
g = sns.FacetGrid(df, col=col, col_wrap=3, height=3, sharey=True)
g.map_dataframe(sns.histplot, x=num_cols[0], kde=True, color="#2a78d6")
g.set_titles("{col_name}")
g.figure.suptitle(f"{num_cols[0]} 的分布 (依 {col})", y=1.03)
""",
    "配對前後連線圖 (Before-After)": _PRELUDE + """
# 需要兩個數值欄 (例如 Before / After)，每列是同一位受試者
a, b = num_cols[0], num_cols[1]
fig, ax = plt.subplots(figsize=(4, 5))
for _, r in df[[a, b]].dropna().iterrows():
    ax.plot([0, 1], [r[a], r[b]], color="#999999", lw=0.8, marker="o", ms=4)
ax.plot([0, 1], [df[a].mean(), df[b].mean()], color="#e34948", lw=3, marker="o", ms=9, label="Mean")
p = stats.ttest_rel(df[a], df[b], nan_policy="omit").pvalue
ax.set_xticks([0, 1], [a, b])
ax.set_xlim(-0.4, 1.4)
ax.set_title(f"Paired t-test p = {p:.3g}")
ax.legend()
sns.despine()
""",
    "3D 散佈圖": _PRELUDE + """
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
c = pd.factorize(df[cat_cols[0]])[0] if cat_cols else None
sc = ax.scatter(df[num_cols[0]], df[num_cols[1]], df[num_cols[2 % len(num_cols)]], c=c, cmap="tab10", s=30, depthshade=True)
ax.set_xlabel(num_cols[0]); ax.set_ylabel(num_cols[1]); ax.set_zlabel(num_cols[2 % len(num_cols)])
ax.view_init(elev=20, azim=-60)
""",
    "斷軸圖 (Broken axis)": _PRELUDE + """
# 數值差距很大時把 Y 軸截斷
x = cat_cols[0] if cat_cols else None
means = df.groupby(x)[num_cols[0]].mean()
fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(5, 5), gridspec_kw=dict(height_ratios=[1, 2], hspace=0.06))
for ax in (top, bottom):
    ax.bar(means.index.astype(str), means.values, color="#2a78d6")
lo_max = means.sort_values().iloc[-2] * 1.2 if len(means) > 1 else means.max()
bottom.set_ylim(0, lo_max)
top.set_ylim(means.max() * 0.9, means.max() * 1.05)
top.spines["bottom"].set_visible(False); bottom.spines["top"].set_visible(False)
top.tick_params(bottom=False)
d = 0.015
kw = dict(transform=top.transAxes, color="k", clip_on=False, lw=1)
top.plot((-d, d), (-d * 2, d * 2), **kw); top.plot((1 - d, 1 + d), (-d * 2, d * 2), **kw)
kw.update(transform=bottom.transAxes)
bottom.plot((-d, d), (1 - d, 1 + d), **kw); bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
""",
    "誤差帶時間序列 + 註解": _PRELUDE + """
x = next((c for c in df.columns if str(c).lower() in ("day", "time", "week", "hour")), num_cols[0])
y = [c for c in num_cols if c != x][0]
hue = cat_cols[0] if cat_cols else None
fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(data=df, x=x, y=y, hue=hue, errorbar="se", marker="o", err_style="band", ax=ax)
ax.axvspan(df[x].min(), df[x].min() + (df[x].max() - df[x].min()) * 0.25, color="#eda100", alpha=0.12, lw=0)
ax.annotate("Treatment start", xy=(df[x].min(), ax.get_ylim()[1] * 0.98), fontsize=9, va="top")
sns.despine()
""",
    "極座標玫瑰圖": _PRELUDE + """
cat = cat_cols[0] if cat_cols else df.columns[0]
counts = df[cat].value_counts()
theta = np.linspace(0, 2 * np.pi, len(counts), endpoint=False)
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
bars = ax.bar(theta, counts.values, width=2 * np.pi / len(counts) * 0.9, color=sns.color_palette("husl", len(counts)), edgecolor="white")
ax.set_xticks(theta, counts.index.astype(str))
ax.set_title(f"{cat} 次數", pad=20)
""",
    "空白範本": _PRELUDE + """
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df[num_cols[0]].dropna().values, marker="o")
ax.set_title("我的圖")
""",
}


# 英文模式下，範本裡的中文註解與圖上文字換成英文 (程式邏輯相同)
_TEMPLATE_EN = {
    "# 可用變數：df (目前資料)、pd、np、plt、sns、stats (scipy.stats)、sm、smf (statsmodels)、bstats (本程式統計函式)":
        "# Available: df (current data), pd, np, plt, sns, stats (scipy.stats), sm, smf (statsmodels), bstats (built-in statistics)",
    "# 程式中建立的所有 matplotlib 圖都會顯示在下方，並可下載。": "# Every matplotlib figure you create is shown and can be downloaded.",
    "# 連續數值欄 (排除 0/1、週數這類少數值欄)": "# continuous numeric columns (excludes 0/1 or week-number style columns)",
    "# 適合當分組的欄 (2–12 類)": "# columns suitable for grouping (2–12 categories)",
    "# 手動加一個顯著性括號：比較前兩組": "# Add a significance bracket by hand: compare the first two groups",
    'f"{num_cols[0]} 的分布 (依 {col})"': 'f"Distribution of {num_cols[0]} (by {col})"',
    "# 需要兩個數值欄 (例如 Before / After)，每列是同一位受試者": "# Needs two numeric columns (e.g. Before / After), one subject per row",
    "# 數值差距很大時把 Y 軸截斷": "# Break the Y axis when values differ by orders of magnitude",
    'f"{cat} 次數"': 'f"{cat} counts"',
    'ax.set_title("我的圖")': 'ax.set_title("My figure")',
}


def template_code(name: str) -> str:
    """範本程式碼 (依目前語言)。"""
    from .i18n import is_en

    code = TEMPLATES[name]
    if is_en():
        for zh, en in _TEMPLATE_EN.items():
            code = code.replace(zh, en)
    return code

def run_user_code(code: str, df: pd.DataFrame, style: Style | None = None) -> dict:
    """執行使用者程式碼，回傳 {figs, stdout, error, rc}。style=None 時只套用中文字型。"""
    out = io.StringIO()
    figs: list = []
    error = None
    with PLOT_LOCK:
        cjk = fonts.ensure_cjk_font()
        rc = build_rc(style, cjk) if style is not None else {"font.family": fonts.font_family(fonts.default_font(), cjk),
                                                             "axes.unicode_minus": True}
        before = set(plt.get_fignums())
        ns: dict = {"df": df.copy(), "pd": pd, "np": np, "plt": plt, "sns": sns, "stats": sps, "matplotlib": matplotlib,
                    "bstats": bstats, "__name__": "__biostat_user__"}
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf

            ns.update(sm=sm, smf=smf)
        except Exception:
            pass
        try:
            with plt.rc_context(rc), contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(compile(code, "<自訂程式碼>", "exec"), ns)
        except Exception:
            tb = traceback.format_exc()
            lines = tb.splitlines()
            # 只保留使用者程式碼相關的追蹤訊息
            keep = [ln for i, ln in enumerate(lines) if "<自訂程式碼>" in ln or i == len(lines) - 1 or
                    (i > 0 and "<自訂程式碼>" in lines[i - 1])]
            error = "\n".join(keep) or tb
        figs = [plt.figure(n) for n in plt.get_fignums() if n not in before]
    return {"figs": figs, "stdout": out.getvalue(), "error": error, "rc": rc}
