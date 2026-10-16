"""在類別圖上自動加入顯著性標記 (括號 + 星號 / p 值)。"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

TESTS = {
    "Welch t 檢定": "welch",
    "Student t 檢定": "student",
    "Mann-Whitney U": "mwu",
    "Tukey HSD (ANOVA 事後)": "tukey",
    "Games-Howell (Welch ANOVA 事後)": "gh",
    "Dunn (Kruskal-Wallis 事後)": "dunn",
}
CORRECTIONS = {"不校正": None, "Bonferroni": "bonferroni", "Holm": "holm", "FDR (Benjamini-Hochberg)": "fdr_bh"}
DISPLAYS = ["星號 (*)", "p 值", "p 值 + 星號"]


def stars(p: float) -> str:
    if p is None or np.isnan(p):
        return "n/a"
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fmt_p(p: float, digits: int = 3) -> str:
    if p is None or np.isnan(p):
        return "p = n/a"
    if p < 10 ** -digits:
        return f"p < {10 ** -digits:.{digits}f}"
    return f"p = {p:.{digits}f}"


def category_positions(order: list, hue_order: list | None, width: float = 0.8, dodge: bool = True) -> dict:
    """回傳 {(x_level, hue_level): x 座標}，與 seaborn 類別圖的 dodge 位置一致。"""
    pos = {}
    for i, xl in enumerate(order):
        if hue_order and dodge:
            k = len(hue_order)
            for j, hl in enumerate(hue_order):
                pos[(xl, hl)] = i - width / 2 + width / k * (j + 0.5)
        else:
            pos[(xl, None)] = float(i)
    return pos


def _pairwise(samples: dict, pairs: list, test: str, correction: str | None) -> pd.DataFrame:
    from . import stats as bstats

    rows = []
    if test in ("tukey", "dunn", "gh"):
        keys = list(samples)
        valid = [k for k in keys if len(samples[k]) >= 2]
        if len(valid) < 2:
            return pd.DataFrame()
        if test == "gh":
            gh = bstats.games_howell({k: samples[k] for k in valid})
            lookup = {(r.group1, r.group2): r for r in gh.itertuples()}
            for a, b in pairs:
                r = lookup.get((a, b)) or lookup.get((b, a))
                if r is not None:
                    rows.append({"group1": a, "group2": b, "statistic": r.t, "p": r.p_adj, "p_adj": r.p_adj})
            return pd.DataFrame(rows)
        if test == "tukey":
            res = stats.tukey_hsd(*[samples[k] for k in valid])
            pmat = res.pvalue
            idx = {k: i for i, k in enumerate(valid)}
            for a, b in pairs:
                if a in idx and b in idx:
                    d = np.mean(samples[a]) - np.mean(samples[b])
                    rows.append({"group1": a, "group2": b, "statistic": d, "p": float(pmat[idx[a], idx[b]])})
            out = pd.DataFrame(rows)
            if len(out):
                out["p_adj"] = out["p"]
            return out
        dunn = bstats.dunn_test({k: samples[k] for k in valid}, correction or "holm")
        lookup = {(r.group1, r.group2): r for r in dunn.itertuples()}
        for a, b in pairs:
            r = lookup.get((a, b)) or lookup.get((b, a))
            if r is not None:
                rows.append({"group1": a, "group2": b, "statistic": r.z, "p": r.p, "p_adj": r.p_adj})
        return pd.DataFrame(rows)

    for a, b in pairs:
        xa, xb = samples.get(a, np.array([])), samples.get(b, np.array([]))
        if len(xa) < 2 or len(xb) < 2:
            continue
        if test == "mwu":
            r = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        else:
            r = stats.ttest_ind(xa, xb, equal_var=(test == "student"))
        rows.append({"group1": a, "group2": b, "statistic": float(r.statistic), "p": float(r.pvalue)})
    out = pd.DataFrame(rows)
    if len(out):
        if correction and len(out) > 1:
            from statsmodels.stats.multitest import multipletests

            out["p_adj"] = multipletests(out["p"], method=correction)[1]
        else:
            out["p_adj"] = out["p"]
    return out


def add_significance(ax, df: pd.DataFrame, x: str, y: str, hue: str | None, order: list, hue_order: list | None,
                     *, test: str = "welch", pairs_mode: str = "all", ref: str | None = None,
                     correction: str | None = "holm", display: str = "星號 (*)", hide_ns: bool = False,
                     line_width: float = 1.0, font_size: float = 10.0, color: str = "black",
                     dodge: bool = True, width: float = 0.8) -> pd.DataFrame:
    """計算成對比較並在 ax 上畫括號。回傳結果表。"""
    d = df[[c for c in {x, y, hue} if c]].dropna()
    use_hue = bool(hue) and hue != x and bool(hue_order)
    pos = category_positions(order, hue_order if use_hue else None, width=width, dodge=dodge)
    results = []

    def label_of(p):
        if display == "p 值":
            return fmt_p(p)
        if display == "p 值 + 星號":
            return f"{fmt_p(p)} {stars(p)}" if p < 0.05 else fmt_p(p)
        return stars(p)

    groups_cmp = []  # (x_level, levels list, column)
    if use_hue:
        for xl in order:
            groups_cmp.append((xl, hue_order, hue))
    else:
        groups_cmp.append((None, order, x))

    for xl, lv, col in groups_cmp:
        sub = d if xl is None else d[d[x] == xl]
        samples = {k: sub.loc[sub[col] == k, y].to_numpy(float) for k in lv}
        if pairs_mode == "ref" and ref in lv:
            pairs = [(ref, k) for k in lv if k != ref]
        elif pairs_mode == "adjacent":
            pairs = list(zip(lv[:-1], lv[1:]))
        else:
            pairs = list(itertools.combinations(lv, 2))
        res = _pairwise(samples, pairs, test, correction)
        if res.empty:
            continue
        res.insert(0, "x", xl if xl is not None else "")
        res["n1"] = [len(samples[a]) for a in res["group1"]]
        res["n2"] = [len(samples[b]) for b in res["group2"]]
        res["label"] = [label_of(p) for p in res["p_adj"]]
        res["pos_a"] = [pos[(xl, a)] if xl is not None else pos[(a, None)] for a in res["group1"]]
        res["pos_b"] = [pos[(xl, b)] if xl is not None else pos[(b, None)] for b in res["group2"]]
        results.append(res)

    if not results:
        return pd.DataFrame()
    table = pd.concat(results, ignore_index=True)

    # 每個位置目前的最高點 (資料最大值)
    tops = {}
    for key, xpos in pos.items():
        xl, hl = key
        if use_hue:
            vals = d.loc[(d[x] == xl) & (d[hue] == hl), y]
        else:
            vals = d.loc[d[x] == xl, y]
        tops[xpos] = float(vals.max()) if len(vals) else np.nan
    finite = [v for v in tops.values() if np.isfinite(v)]
    if not finite:
        return table
    ylo, yhi = ax.get_ylim()
    data_top = max(finite)
    span = max(data_top - min(ylo, min(finite)), 1e-12)
    for k in tops:
        if not np.isfinite(tops[k]):
            tops[k] = data_top
    gap, h, text_h = span * 0.04, span * 0.025, span * 0.08

    draw = table if not hide_ns else table[table["p_adj"] < 0.05]
    draw = draw.assign(span_w=(draw["pos_b"] - draw["pos_a"]).abs()).sort_values(["span_w", "pos_a"])
    ymax_used = data_top
    for r in draw.itertuples():
        lo, hi = sorted((r.pos_a, r.pos_b))
        covered = [p for p in tops if lo - 1e-9 <= p <= hi + 1e-9]
        base = max(tops[p] for p in covered) + gap
        ax.plot([lo, lo, hi, hi], [base, base + h, base + h, base], color=color, lw=line_width, clip_on=False,
                solid_capstyle="butt")
        # 星號字形本身偏高，稍微往下貼近括號
        star_only = set(r.label) == {"*"}
        ax.text((lo + hi) / 2, base + h - (span * 0.012 if star_only else -span * 0.004), r.label, ha="center",
                va="bottom", fontsize=font_size, color=color)
        new_top = base + h + text_h
        for p in covered:
            tops[p] = new_top
        ymax_used = max(ymax_used, new_top)
    if ymax_used > yhi:
        ax.set_ylim(ylo, ymax_used + span * 0.02)
    return table.drop(columns=["pos_a", "pos_b"])
