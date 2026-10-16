"""生物統計分析函式。每個分析回傳 StatResult (多張結果表 + 中文解讀)。"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class StatResult:
    title: str
    tables: list = field(default_factory=list)  # [(名稱, DataFrame)]
    messages: list = field(default_factory=list)  # 解讀文字 (markdown)
    warnings: list = field(default_factory=list)

    def add(self, name: str, df: pd.DataFrame) -> "StatResult":
        self.tables.append((name, df))
        return self


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def sig_text(p: float, alpha: float = ALPHA) -> str:
    if p is None or not np.isfinite(p):
        return "無法計算"
    return f"p = {fmt_p(p)}，{'達' if p < alpha else '未達'}統計顯著 (α = {alpha})"


def _clean(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)


def _q(name: str) -> str:
    """patsy 公式中安全引用欄位名稱 (可含空白、中文)。"""
    return 'Q("' + str(name).replace('"', '\\"') + '")'


def group_samples(df: pd.DataFrame, value: str, group: str, order: list | None = None) -> dict:
    d = df[[value, group]].copy()
    d[value] = pd.to_numeric(d[value], errors="coerce")
    d = d.dropna()
    lv = order or list(dict.fromkeys(d[group].tolist()))
    return {str(k): d.loc[d[group] == k, value].to_numpy(float) for k in lv if (d[group] == k).any()}


# ============================================================ 描述統計
def describe_array(x: np.ndarray, ci: float = 0.95) -> dict:
    n = len(x)
    if n == 0:
        return {"n": 0}
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
    sem = sd / np.sqrt(n) if n > 1 else np.nan
    tcrit = stats.t.ppf((1 + ci) / 2, n - 1) if n > 1 else np.nan
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {
        "n": n, "平均值 Mean": mean, "標準差 SD": sd, "標準誤 SEM": sem,
        f"{int(ci * 100)}% CI 下限": mean - tcrit * sem if n > 1 else np.nan,
        f"{int(ci * 100)}% CI 上限": mean + tcrit * sem if n > 1 else np.nan,
        "中位數 Median": med, "Q1": q1, "Q3": q3, "IQR": q3 - q1,
        "最小值": float(np.min(x)), "最大值": float(np.max(x)),
        "變異係數 CV%": sd / mean * 100 if mean not in (0, np.nan) and n > 1 else np.nan,
        "偏態 Skewness": float(stats.skew(x, bias=False)) if n > 2 else np.nan,
        "峰度 Kurtosis": float(stats.kurtosis(x, bias=False)) if n > 3 else np.nan,
    }


def descriptive(df: pd.DataFrame, values: list[str], group: str | None = None, ci: float = 0.95) -> StatResult:
    res = StatResult("描述性統計")
    rows = []
    for v in values:
        s = pd.to_numeric(df[v], errors="coerce")
        if group:
            for lv, sub in df.assign(_v=s).groupby(group, sort=False):
                x = sub["_v"].dropna().to_numpy(float)
                rows.append({"變項": v, "組別": lv, "缺失數": int(sub["_v"].isna().sum()), **describe_array(x, ci)})
        else:
            rows.append({"變項": v, "缺失數": int(s.isna().sum()), **describe_array(s.dropna().to_numpy(float), ci)})
    res.add("描述性統計", pd.DataFrame(rows))
    return res


# ============================================================ 前提檢定
def normality(df: pd.DataFrame, values: list[str], group: str | None = None) -> StatResult:
    from statsmodels.stats.diagnostic import lilliefors

    res = StatResult("常態性檢定")
    rows = []
    subsets = []
    for v in values:
        if group:
            for lv, x in group_samples(df, v, group).items():
                subsets.append((v, lv, x))
        else:
            subsets.append((v, "全部", _clean(df[v])))
    for v, lv, x in subsets:
        row = {"變項": v, "組別": lv, "n": len(x)}
        if len(x) >= 3:
            w, p = stats.shapiro(x)
            row.update({"Shapiro-Wilk W": w, "Shapiro p": p})
        if len(x) >= 8:
            k2, p2 = stats.normaltest(x)
            row.update({"D'Agostino K²": k2, "D'Agostino p": p2})
        if len(x) >= 5:
            try:
                ks, p3 = lilliefors(x, dist="norm")
                row.update({"Lilliefors D": ks, "Lilliefors p": p3})
            except Exception:
                pass
        p_sw = row.get("Shapiro p", np.nan)
        row["結論 (Shapiro)"] = "—" if not np.isfinite(p_sw) else ("符合常態" if p_sw >= ALPHA else "偏離常態")
        rows.append(row)
    out = pd.DataFrame(rows)
    res.add("常態性檢定", out)
    if "Shapiro p" in out and (out["Shapiro p"] < ALPHA).any():
        res.messages.append("部分組別偏離常態，建議考慮無母數檢定 (Mann-Whitney U、Kruskal-Wallis) 或資料轉換 (如 log)。")
    else:
        res.messages.append("各組資料未顯著偏離常態，可使用 t 檢定 / ANOVA 等母數檢定。")
    res.messages.append("提醒：樣本數大時常態檢定很敏感，請同時參考 Q-Q 圖判斷。")
    return res


def variance_homogeneity(df: pd.DataFrame, value: str, group: str) -> StatResult:
    res = StatResult("變異數同質性檢定")
    samples = [x for x in group_samples(df, value, group).values() if len(x) >= 2]
    if len(samples) < 2:
        res.warnings.append("至少需要兩組、每組 ≥ 2 筆資料。")
        return res
    rows = []
    for name, fn in (("Levene (mean)", lambda: stats.levene(*samples, center="mean")),
                     ("Brown-Forsythe (median)", lambda: stats.levene(*samples, center="median")),
                     ("Bartlett", lambda: stats.bartlett(*samples))):
        r = fn()
        rows.append({"檢定": name, "統計量": r.statistic, "p 值": r.pvalue,
                     "結論": "變異數同質" if r.pvalue >= ALPHA else "變異數不同質"})
    res.add("變異數同質性", pd.DataFrame(rows))
    p_bf = rows[1]["p 值"]
    res.messages.append(
        "Brown-Forsythe 檢定" + sig_text(p_bf) + "。" +
        ("各組變異數差異顯著，建議使用 Welch t 檢定 / Welch ANOVA / Games-Howell。" if p_bf < ALPHA else "可假設變異數同質。"))
    return res


# ============================================================ 兩組比較
def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def effect_label(d: float) -> str:
    d = abs(d)
    if not np.isfinite(d):
        return ""
    return "極小" if d < 0.2 else "小" if d < 0.5 else "中" if d < 0.8 else "大"


def two_group(df: pd.DataFrame, value: str, group: str, a: str, b: str, alternative: str = "two-sided",
              ci: float = 0.95) -> StatResult:
    res = StatResult(f"兩組比較：{a} vs {b}")
    samples = group_samples(df, value, group)
    xa, xb = samples.get(str(a), np.array([])), samples.get(str(b), np.array([]))
    if len(xa) < 2 or len(xb) < 2:
        res.warnings.append("每組至少需要 2 筆資料。")
        return res
    res.add("各組描述", pd.DataFrame([{"組別": a, **describe_array(xa, ci)}, {"組別": b, **describe_array(xb, ci)}]))

    na, nb = len(xa), len(xb)
    va, vb = np.var(xa, ddof=1), np.var(xb, ddof=1)
    diff = np.mean(xa) - np.mean(xb)
    rows = []
    st = stats.ttest_ind(xa, xb, equal_var=True, alternative=alternative)
    dfp = na + nb - 2
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / dfp)
    se_p = sp * np.sqrt(1 / na + 1 / nb)
    tc = stats.t.ppf((1 + ci) / 2, dfp)
    rows.append({"檢定": "Student t 檢定 (等變異)", "統計量": st.statistic, "自由度": dfp, "p 值": st.pvalue,
                 "平均差": diff, "CI 下限": diff - tc * se_p, "CI 上限": diff + tc * se_p})
    wt = stats.ttest_ind(xa, xb, equal_var=False, alternative=alternative)
    se_w = np.sqrt(va / na + vb / nb)
    dfw = se_w ** 4 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    tcw = stats.t.ppf((1 + ci) / 2, dfw)
    rows.append({"檢定": "Welch t 檢定 (不等變異)", "統計量": wt.statistic, "自由度": dfw, "p 值": wt.pvalue,
                 "平均差": diff, "CI 下限": diff - tcw * se_w, "CI 上限": diff + tcw * se_w})
    mw = stats.mannwhitneyu(xa, xb, alternative=alternative)
    rows.append({"檢定": "Mann-Whitney U (無母數)", "統計量": mw.statistic, "自由度": np.nan, "p 值": mw.pvalue,
                 "平均差": np.median(xa) - np.median(xb), "CI 下限": np.nan, "CI 上限": np.nan})
    tests = pd.DataFrame(rows).rename(columns={"平均差": "差異 (平均/中位數)"})
    tests["結論"] = np.where(tests["p 值"] < ALPHA, "顯著", "不顯著")
    res.add("檢定結果", tests)

    d = cohen_d(xa, xb)
    g = d * (1 - 3 / (4 * (na + nb) - 9))
    rbc = 1 - 2 * mw.statistic / (na * nb)
    res.add("效果量", pd.DataFrame([
        {"效果量": "Cohen's d", "數值": d, "大小": effect_label(d)},
        {"效果量": "Hedges' g", "數值": g, "大小": effect_label(g)},
        {"效果量": "Rank-biserial r", "數值": -rbc, "大小": ""},
    ]))

    p_sw = [stats.shapiro(x).pvalue if len(x) >= 3 else np.nan for x in (xa, xb)]
    p_lev = stats.levene(xa, xb, center="median").pvalue
    normal = all(np.isfinite(p) and p >= ALPHA for p in p_sw)
    if normal:
        rec = "Welch t 檢定" if p_lev < ALPHA else "Student t 檢定 (或 Welch t 檢定，較穩健)"
    else:
        rec = "Mann-Whitney U 檢定"
    res.messages.append(f"常態性 (Shapiro p)：{a} = {fmt_p(p_sw[0])}，{b} = {fmt_p(p_sw[1])}；變異數同質性 (Brown-Forsythe) p = {fmt_p(p_lev)}。")
    res.messages.append(f"**建議檢定：{rec}**")
    chosen = rows[1] if (normal and p_lev < ALPHA) else rows[0] if normal else None
    if chosen:
        res.messages.append(f"{chosen['檢定']}：t({chosen['自由度']:.1f}) = {chosen['統計量']:.3f}，{sig_text(chosen['p 值'])}；"
                            f"平均差 = {diff:.4g} ({int(ci * 100)}% CI {chosen['CI 下限']:.4g} ~ {chosen['CI 上限']:.4g})，Cohen's d = {d:.2f} ({effect_label(d)}效果)。")
    else:
        res.messages.append(f"Mann-Whitney U = {mw.statistic:.1f}，{sig_text(mw.pvalue)}。")
    return res


def paired_test(df: pd.DataFrame, col1: str, col2: str, alternative: str = "two-sided", ci: float = 0.95) -> StatResult:
    res = StatResult(f"配對比較：{col1} vs {col2}")
    d = df[[col1, col2]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 3:
        res.warnings.append("至少需要 3 組完整配對資料。")
        return res
    a, b = d[col1].to_numpy(float), d[col2].to_numpy(float)
    diff = a - b
    n = len(diff)
    res.add("描述", pd.DataFrame([{"變項": col1, **describe_array(a, ci)}, {"變項": col2, **describe_array(b, ci)},
                                  {"變項": f"差值 ({col1} − {col2})", **describe_array(diff, ci)}]))
    t = stats.ttest_rel(a, b, alternative=alternative)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    tc = stats.t.ppf((1 + ci) / 2, n - 1)
    rows = [{"檢定": "配對 t 檢定", "統計量": t.statistic, "自由度": n - 1, "p 值": t.pvalue,
             "平均差": diff.mean(), "CI 下限": diff.mean() - tc * se, "CI 上限": diff.mean() + tc * se}]
    try:
        w = stats.wilcoxon(a, b, alternative=alternative)
        rows.append({"檢定": "Wilcoxon 符號等級檢定", "統計量": w.statistic, "自由度": np.nan, "p 值": w.pvalue,
                     "平均差": np.median(diff), "CI 下限": np.nan, "CI 上限": np.nan})
    except ValueError as err:
        res.warnings.append(f"Wilcoxon 無法計算：{err}")
    out = pd.DataFrame(rows).rename(columns={"平均差": "差異 (平均/中位數)"})
    out["結論"] = np.where(out["p 值"] < ALPHA, "顯著", "不顯著")
    res.add("檢定結果", out)
    dz = diff.mean() / np.std(diff, ddof=1)
    p_sw = stats.shapiro(diff).pvalue
    res.add("效果量", pd.DataFrame([{"效果量": "Cohen's dz", "數值": dz, "大小": effect_label(dz)}]))
    res.messages.append(f"差值常態性 Shapiro p = {fmt_p(p_sw)} → 建議使用 **{'配對 t 檢定' if p_sw >= ALPHA else 'Wilcoxon 符號等級檢定'}**。")
    res.messages.append(f"配對 t 檢定：t({n - 1}) = {t.statistic:.3f}，{sig_text(t.pvalue)}；平均差 = {diff.mean():.4g}，Cohen's dz = {dz:.2f}。")
    return res


# ============================================================ 多組比較
def tukey_table(samples: dict, ci: float = 0.95) -> pd.DataFrame:
    keys = [k for k, v in samples.items() if len(v) >= 2]
    r = stats.tukey_hsd(*[samples[k] for k in keys])
    conf = r.confidence_interval(confidence_level=ci)
    rows = []
    for i, j in itertools.combinations(range(len(keys)), 2):
        rows.append({"group1": keys[i], "group2": keys[j], "平均差": r.statistic[i, j],
                     "CI 下限": conf.low[i, j], "CI 上限": conf.high[i, j], "p_adj": r.pvalue[i, j]})
    return pd.DataFrame(rows)


def games_howell(samples: dict) -> pd.DataFrame:
    keys = [k for k, v in samples.items() if len(v) >= 2]
    k = len(keys)
    rows = []
    for a, b in itertools.combinations(keys, 2):
        xa, xb = samples[a], samples[b]
        na, nb = len(xa), len(xb)
        va, vb = np.var(xa, ddof=1) / na, np.var(xb, ddof=1) / nb
        diff = np.mean(xa) - np.mean(xb)
        se = np.sqrt(va + vb)
        dfw = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
        q = abs(diff) / se * np.sqrt(2)
        p = float(stats.studentized_range.sf(q, k, dfw))
        rows.append({"group1": a, "group2": b, "平均差": diff, "t": diff / se, "自由度": dfw, "p_adj": min(p, 1.0)})
    return pd.DataFrame(rows)


def dunn_test(samples: dict, correction: str | None = "holm") -> pd.DataFrame:
    keys = [k for k, v in samples.items() if len(v) >= 1]
    allx = np.concatenate([samples[k] for k in keys])
    ranks = stats.rankdata(allx)
    N = len(allx)
    _, counts = np.unique(allx, return_counts=True)
    tie = np.sum(counts ** 3 - counts) / (12 * (N - 1)) if N > 1 else 0.0
    mean_rank, nk, idx = {}, {}, 0
    for k in keys:
        n = len(samples[k])
        mean_rank[k] = ranks[idx: idx + n].mean()
        nk[k] = n
        idx += n
    rows = []
    for a, b in itertools.combinations(keys, 2):
        se = np.sqrt((N * (N + 1) / 12 - tie) * (1 / nk[a] + 1 / nk[b]))
        z = (mean_rank[a] - mean_rank[b]) / se
        rows.append({"group1": a, "group2": b, "平均等級差": mean_rank[a] - mean_rank[b], "z": z, "p": 2 * stats.norm.sf(abs(z))})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_adj"] = p_adjust_array(out["p"].to_numpy(), correction) if correction else out["p"]
    return out


def p_adjust_array(p: np.ndarray, method: str | None) -> np.ndarray:
    if not method or len(p) < 2:
        return np.asarray(p, float)
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method=method)[1]
    return out


def pairwise_tests(samples: dict, kind: str = "welch", correction: str = "holm") -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(samples, 2):
        xa, xb = samples[a], samples[b]
        if len(xa) < 2 or len(xb) < 2:
            continue
        r = stats.mannwhitneyu(xa, xb) if kind == "mwu" else stats.ttest_ind(xa, xb, equal_var=(kind == "student"))
        rows.append({"group1": a, "group2": b, "統計量": r.statistic, "p": r.pvalue})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_adj"] = p_adjust_array(out["p"].to_numpy(), correction)
    return out


def welch_anova(samples: list[np.ndarray]) -> tuple[float, float, float, float]:
    n = np.array([len(x) for x in samples], float)
    m = np.array([np.mean(x) for x in samples])
    v = np.array([np.var(x, ddof=1) for x in samples])
    w = n / v
    k = len(samples)
    mw = np.sum(w * m) / np.sum(w)
    a = np.sum(w * (m - mw) ** 2) / (k - 1)
    tmp = np.sum((1 - w / np.sum(w)) ** 2 / (n - 1))
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    f = a / b
    df2 = (k ** 2 - 1) / (3 * tmp)
    return f, k - 1, df2, float(stats.f.sf(f, k - 1, df2))


def one_way(df: pd.DataFrame, value: str, group: str, posthoc: str = "Tukey HSD", order: list | None = None) -> StatResult:
    res = StatResult(f"單因子比較：{value} ~ {group}")
    samples = {k: v for k, v in group_samples(df, value, group, order).items() if len(v) >= 2}
    if len(samples) < 2:
        res.warnings.append("至少需要兩組、每組 ≥ 2 筆資料。")
        return res
    xs = list(samples.values())
    res.add("各組描述", pd.DataFrame([{"組別": k, **describe_array(v)} for k, v in samples.items()]))

    allx = np.concatenate(xs)
    grand = allx.mean()
    ssb = sum(len(x) * (x.mean() - grand) ** 2 for x in xs)
    ssw = sum(((x - x.mean()) ** 2).sum() for x in xs)
    k, N = len(xs), len(allx)
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    F = msb / msw
    p = float(stats.f.sf(F, dfb, dfw))
    eta2 = ssb / (ssb + ssw)
    omega2 = (ssb - dfb * msw) / (ssb + ssw + msw)
    res.add("ANOVA 表", pd.DataFrame([
        {"變異來源": "組間 (Between)", "SS": ssb, "df": dfb, "MS": msb, "F": F, "p 值": p, "η²": eta2, "ω²": omega2},
        {"變異來源": "組內 (Within)", "SS": ssw, "df": dfw, "MS": msw, "F": np.nan, "p 值": np.nan, "η²": np.nan, "ω²": np.nan},
        {"變異來源": "總和", "SS": ssb + ssw, "df": N - 1, "MS": np.nan, "F": np.nan, "p 值": np.nan, "η²": np.nan, "ω²": np.nan},
    ]))
    wf, wd1, wd2, wp = welch_anova(xs)
    kw = stats.kruskal(*xs)
    eps2 = kw.statistic * (N + 1) / (N ** 2 - 1)
    lev = stats.levene(*xs, center="median")
    res.add("整體檢定", pd.DataFrame([
        {"檢定": "One-way ANOVA", "統計量": F, "df1": dfb, "df2": dfw, "p 值": p, "效果量": f"η² = {eta2:.3f}"},
        {"檢定": "Welch ANOVA (不等變異)", "統計量": wf, "df1": wd1, "df2": wd2, "p 值": wp, "效果量": ""},
        {"檢定": "Kruskal-Wallis (無母數)", "統計量": kw.statistic, "df1": k - 1, "df2": np.nan, "p 值": kw.pvalue, "效果量": f"ε² = {eps2:.3f}"},
        {"檢定": "Brown-Forsythe 變異數同質", "統計量": lev.statistic, "df1": k - 1, "df2": N - k, "p 值": lev.pvalue, "效果量": ""},
    ]))

    ph = {
        "Tukey HSD": lambda: tukey_table(samples),
        "Games-Howell": lambda: games_howell(samples),
        "Bonferroni t 檢定": lambda: pairwise_tests(samples, "student", "bonferroni"),
        "Holm t 檢定": lambda: pairwise_tests(samples, "welch", "holm"),
        "Dunn (Holm)": lambda: dunn_test(samples, "holm"),
        "Dunn (Bonferroni)": lambda: dunn_test(samples, "bonferroni"),
    }
    if posthoc in ph:
        table = ph[posthoc]()
        if len(table):
            table["顯著性"] = [("****" if q < 1e-4 else "***" if q < 1e-3 else "**" if q < 0.01 else "*" if q < 0.05 else "ns") for q in table["p_adj"]]
        res.add(f"事後檢定：{posthoc}", table)

    p_sw = [stats.shapiro(x).pvalue for x in xs if len(x) >= 3]
    normal = all(q >= ALPHA for q in p_sw)
    if not normal:
        rec = "Kruskal-Wallis + Dunn 事後檢定"
    elif lev.pvalue < ALPHA:
        rec = "Welch ANOVA + Games-Howell 事後檢定"
    else:
        rec = "One-way ANOVA + Tukey HSD 事後檢定"
    res.messages.append(f"One-way ANOVA：F({dfb}, {dfw}) = {F:.3f}，{sig_text(p)}，η² = {eta2:.3f}。")
    res.messages.append(f"Kruskal-Wallis：H({k - 1}) = {kw.statistic:.3f}，{sig_text(kw.pvalue)}。")
    res.messages.append(f"常態性：{'各組皆符合' if normal else '有組別偏離常態'}；變異數同質性 p = {fmt_p(lev.pvalue)}。**建議：{rec}**")
    return res


def two_way(df: pd.DataFrame, value: str, f1: str, f2: str, interaction: bool = True, typ: int = 2) -> StatResult:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    res = StatResult(f"雙因子 ANOVA：{value} ~ {f1} × {f2}")
    d = df[[value, f1, f2]].copy()
    d[value] = pd.to_numeric(d[value], errors="coerce")
    d = d.dropna()
    if typ == 3:
        a, b = f"C({_q(f1)}, Sum)", f"C({_q(f2)}, Sum)"
    else:
        a, b = f"C({_q(f1)})", f"C({_q(f2)})"
    formula = f"{_q(value)} ~ {a} {'*' if interaction else '+'} {b}"
    model = smf.ols(formula, data=d).fit()
    tab = anova_lm(model, typ=typ)
    rename = {}
    for idx in tab.index:
        name = str(idx)
        name = name.replace(f"C({_q(f1)}, Sum)", f1).replace(f"C({_q(f2)}, Sum)", f2)
        name = name.replace(f"C({_q(f1)})", f1).replace(f"C({_q(f2)})", f2).replace(":", " × ")
        rename[idx] = "殘差 (Residual)" if name == "Residual" else name
    tab = tab.rename(index=rename)
    ss_res = tab.loc["殘差 (Residual)", "sum_sq"]
    tab["偏 η² (partial)"] = [s / (s + ss_res) if i != "殘差 (Residual)" and i != "Intercept" else np.nan
                              for i, s in zip(tab.index, tab["sum_sq"])]
    tab = tab.rename(columns={"sum_sq": "SS", "df": "df", "F": "F", "PR(>F)": "p 值"}).reset_index(names="變異來源")
    res.add("ANOVA 表", tab)
    cell = d.groupby([f1, f2], sort=False)[value].agg(["count", "mean", "std", "sem"]).reset_index()
    res.add("各組平均", cell.rename(columns={"count": "n", "mean": "平均", "std": "SD", "sem": "SEM"}))
    for _, r in tab.iterrows():
        if r["變異來源"] not in ("殘差 (Residual)", "Intercept") and np.isfinite(r["p 值"]):
            res.messages.append(f"**{r['變異來源']}**：F = {r['F']:.3f}，{sig_text(r['p 值'])}，偏 η² = {r['偏 η² (partial)']:.3f}")
    resid_p = stats.shapiro(model.resid).pvalue if len(model.resid) >= 3 else np.nan
    res.messages.append(f"殘差常態性 Shapiro p = {fmt_p(resid_p)}；R² = {model.rsquared:.3f}。")
    if interaction:
        res.messages.append("若交互作用顯著，主效應需謹慎解讀，建議進行簡單主效應分析 (可用篩選功能分層比較)。")
    return res


def rm_anova(df: pd.DataFrame, value: str, subject: str, within: str) -> StatResult:
    from statsmodels.stats.anova import AnovaRM

    res = StatResult(f"重複量測 ANOVA：{value} ~ {within} (受試者：{subject})")
    d = df[[value, subject, within]].copy()
    d[value] = pd.to_numeric(d[value], errors="coerce")
    d = d.dropna()
    d[within] = d[within].astype(str)
    d[subject] = d[subject].astype(str)
    wide = d.pivot_table(index=subject, columns=within, values=value, aggfunc="mean").dropna()
    order = list(dict.fromkeys(d[within]))
    wide = wide[[c for c in order if c in wide.columns]]
    if wide.shape[0] < 3 or wide.shape[1] < 2:
        res.warnings.append("需要至少 3 位受試者在每個時間點/條件都有資料。")
        return res
    long = wide.reset_index().melt(id_vars=subject, var_name=within, value_name=value)
    r = AnovaRM(long, depvar=value, subject=subject, within=[within]).fit()
    tab = r.anova_table.reset_index(names="效應")
    tab = tab.rename(columns={"F Value": "F", "Num DF": "df1", "Den DF": "df2", "Pr > F": "p 值"})
    res.add("重複量測 ANOVA", tab)
    fr = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    res.add("Friedman 檢定 (無母數)", pd.DataFrame([{"χ²": fr.statistic, "df": wide.shape[1] - 1, "p 值": fr.pvalue, "受試者數": wide.shape[0]}]))
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        t = stats.ttest_rel(wide[a], wide[b])
        try:
            w = stats.wilcoxon(wide[a], wide[b]).pvalue
        except ValueError:
            w = np.nan
        rows.append({"group1": a, "group2": b, "平均差": (wide[a] - wide[b]).mean(), "配對 t p": t.pvalue, "Wilcoxon p": w})
    ph = pd.DataFrame(rows)
    if len(ph):
        ph["配對 t p (Holm)"] = p_adjust_array(ph["配對 t p"].to_numpy(), "holm")
        ph["Wilcoxon p (Holm)"] = p_adjust_array(ph["Wilcoxon p"].to_numpy(), "holm")
    res.add("事後成對比較", ph)
    p = tab["p 值"].iloc[0]
    res.messages.append(f"重複量測 ANOVA：F({tab['df1'].iloc[0]:.0f}, {tab['df2'].iloc[0]:.0f}) = {tab['F'].iloc[0]:.3f}，{sig_text(p)}。")
    res.messages.append(f"Friedman：χ² = {fr.statistic:.3f}，{sig_text(fr.pvalue)}。（僅納入所有條件皆有資料的 {wide.shape[0]} 位受試者）")
    res.messages.append("註：未進行球形檢定校正 (Greenhouse-Geisser)，若條件數 ≥ 3 且違反球形假設，請參考 Friedman 結果。")
    return res


# ============================================================ 相關與迴歸
def correlation(df: pd.DataFrame, cols: list[str], method: str = "pearson", ci: float = 0.95) -> StatResult:
    res = StatResult(f"相關分析 ({method.title()})")
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    rows = []
    r_mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    p_mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for a, b in itertools.combinations(cols, 2):
        pair = d[[a, b]].dropna()
        n = len(pair)
        if n < 3:
            continue
        r, p = fn(pair[a], pair[b])[:2]
        r, p = float(r), float(p)
        # Fisher z 信賴區間；Spearman / Kendall 使用 Fieller et al. (1957) 的標準誤
        se = {"pearson": np.sqrt(1 / max(n - 3, 1)), "spearman": np.sqrt(1.06 / max(n - 3, 1)),
              "kendall": np.sqrt(0.437 / max(n - 4, 1))}[method]
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        zc = stats.norm.ppf((1 + ci) / 2)
        rows.append({"變項 1": a, "變項 2": b, "n": n, "r": r, "CI 下限": np.tanh(z - zc * se), "CI 上限": np.tanh(z + zc * se),
                     "R²": r * r, "p 值": p})
        r_mat.loc[a, b] = r_mat.loc[b, a] = r
        p_mat.loc[a, b] = p_mat.loc[b, a] = p
    out = pd.DataFrame(rows)
    if len(out):
        out["p (FDR)"] = p_adjust_array(out["p 值"].to_numpy(), "fdr_bh")
        out["強度"] = [("很強" if abs(r) >= 0.8 else "強" if abs(r) >= 0.6 else "中等" if abs(r) >= 0.4 else "弱" if abs(r) >= 0.2 else "極弱") for r in out["r"]]
    res.add("成對相關", out)
    res.add("相關係數矩陣", r_mat.reset_index(names="變項"))
    res.add("p 值矩陣", p_mat.reset_index(names="變項"))
    for r in out.head(10).itertuples():
        res.messages.append(f"{r._1} 與 {r._2}：r = {r.r:.3f} ({r.強度})，{sig_text(r._8)}，n = {r.n}")
    return res


def linear_regression(df: pd.DataFrame, y: str, xs: list[str], cat_xs: list[str] | None = None) -> StatResult:
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson

    cat_xs = cat_xs or []
    res = StatResult(f"線性迴歸：{y} ~ {' + '.join(xs + cat_xs)}")
    cols = [y] + xs + cat_xs
    d = df[cols].copy()
    for c in [y] + xs:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    terms = [_q(x) for x in xs] + [f"C({_q(c)})" for c in cat_xs]
    model = smf.ols(f"{_q(y)} ~ {' + '.join(terms) or '1'}", data=d).fit()
    ci = model.conf_int()
    coef = pd.DataFrame({
        "變項": [_pretty_term(t) for t in model.params.index],
        "係數 B": model.params.values, "標準誤": model.bse.values, "t": model.tvalues.values,
        "p 值": model.pvalues.values, "95% CI 下限": ci[0].values, "95% CI 上限": ci[1].values,
    })
    sd_y = d[y].std(ddof=1)
    coef["標準化 β"] = [np.nan if t == "Intercept" or t not in [_q(x) for x in xs] else
                       model.params[t] * d[xs[[_q(x) for x in xs].index(t)]].std(ddof=1) / sd_y for t in model.params.index]
    res.add("迴歸係數", coef)
    fit = {"n": int(model.nobs), "R²": model.rsquared, "調整後 R²": model.rsquared_adj, "F": model.fvalue,
           "F 檢定 p": model.f_pvalue, "AIC": model.aic, "BIC": model.bic, "RMSE": np.sqrt(model.mse_resid)}
    res.add("模型配適", pd.DataFrame([fit]))
    diag = [{"診斷": "殘差常態性 (Shapiro)", "統計量": np.nan, "p 值": np.nan}]
    if len(model.resid) >= 3:
        sw = stats.shapiro(model.resid)
        diag[0].update({"統計量": sw.statistic, "p 值": sw.pvalue})
    diag.append({"診斷": "Durbin-Watson (自我相關，~2 為佳)", "統計量": durbin_watson(model.resid), "p 值": np.nan})
    if len(terms):
        bp = het_breuschpagan(model.resid, model.model.exog)
        diag.append({"診斷": "Breusch-Pagan (變異數同質)", "統計量": bp[0], "p 值": bp[1]})
    if len(xs) + len(cat_xs) > 1:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X = model.model.exog
        for i, name in enumerate(model.model.exog_names):
            if name != "Intercept":
                diag.append({"診斷": f"VIF：{_pretty_term(name)}", "統計量": variance_inflation_factor(X, i), "p 值": np.nan})
    res.add("模型診斷", pd.DataFrame(diag))
    res.messages.append(f"模型 R² = {model.rsquared:.3f} (調整後 {model.rsquared_adj:.3f})，F 檢定 {sig_text(model.f_pvalue)}。")
    for r in coef.itertuples():
        if r.變項 != "Intercept":
            res.messages.append(f"{r.變項}：B = {r._2:.4g} (95% CI {r._6:.4g} ~ {r._7:.4g})，{sig_text(r._5)}")
    res.messages.append("VIF > 5 (或 10) 表示共線性問題。")
    return res


def _pretty_term(t: str) -> str:
    import re

    t = re.sub(r'C\(Q\("(.+?)"\)\)\[T\.(.+?)\]', r"\1 [\2]", t)
    t = re.sub(r'Q\("(.+?)"\)', r"\1", t)
    return t


def logistic_regression(df: pd.DataFrame, y: str, positive, xs: list[str], cat_xs: list[str] | None = None) -> StatResult:
    import statsmodels.formula.api as smf

    cat_xs = cat_xs or []
    res = StatResult(f"邏輯迴歸：{y} = {positive} ~ {' + '.join(xs + cat_xs)}")
    d = df[[y] + xs + cat_xs].copy()
    for c in xs:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    d["_y"] = (d[y].astype(str) == str(positive)).astype(int)
    if d["_y"].nunique() < 2:
        res.warnings.append("結果變項只有一個類別，無法分析。")
        return res
    terms = [_q(x) for x in xs] + [f"C({_q(c)})" for c in cat_xs]
    model = smf.logit(f"_y ~ {' + '.join(terms) or '1'}", data=d).fit(disp=0, maxiter=200)
    ci = model.conf_int()
    coef = pd.DataFrame({
        "變項": [_pretty_term(t) for t in model.params.index],
        "係數 B": model.params.values, "標準誤": model.bse.values, "z": model.tvalues.values, "p 值": model.pvalues.values,
        "勝算比 OR": np.exp(model.params.values), "OR 95% CI 下限": np.exp(ci[0].values), "OR 95% CI 上限": np.exp(ci[1].values),
    })
    res.add("迴歸係數 (勝算比)", coef)
    prob = model.predict(d)
    fpr, tpr, _, auc = roc_curve(d["_y"].to_numpy(), prob.to_numpy())
    pred = (prob >= 0.5).astype(int)
    acc = float((pred == d["_y"]).mean())
    res.add("模型配適", pd.DataFrame([{
        "n": int(model.nobs), "事件數": int(d["_y"].sum()), "McFadden 偽 R²": model.prsquared, "LR χ²": model.llr,
        "LR 檢定 p": model.llr_pvalue, "AIC": model.aic, "BIC": model.bic, "AUC (C 統計量)": auc, "準確率 (cutoff 0.5)": acc,
    }]))
    res.messages.append(f"模型 LR 檢定 {sig_text(model.llr_pvalue)}，AUC = {auc:.3f}。")
    for r in coef.itertuples():
        if r.變項 != "Intercept":
            res.messages.append(f"{r.變項}：OR = {r._6:.3f} (95% CI {r._7:.3f} ~ {r._8:.3f})，{sig_text(r._5)}")
    return res


# ============================================================ 類別資料
def contingency(df: pd.DataFrame, row: str, col: str, count: str | None = None, yates: bool = True) -> StatResult:
    res = StatResult(f"列聯表分析：{row} × {col}")
    d = df.dropna(subset=[row, col])
    if count:
        ct = d.pivot_table(index=row, columns=col, values=count, aggfunc="sum", fill_value=0)
    else:
        ct = pd.crosstab(d[row], d[col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        res.warnings.append("兩個變項都至少需要 2 個類別。")
        return res
    obs = ct.copy()
    obs["總和"] = obs.sum(axis=1)
    obs.loc["總和"] = obs.sum(axis=0)
    res.add("觀察次數", obs.reset_index())
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    res.add("列百分比 (%)", pct.reset_index())
    chi2, p, dof, exp = stats.chi2_contingency(ct.to_numpy(), correction=yates)
    res.add("期望次數", pd.DataFrame(exp, index=ct.index, columns=ct.columns).reset_index())
    n = ct.to_numpy().sum()
    # Cramér's V 以未經 Yates 校正的 χ² 計算 (校正只用於 2×2 的 p 值)
    chi2_raw = stats.chi2_contingency(ct.to_numpy(), correction=False)[0]
    v = np.sqrt(chi2_raw / (n * (min(ct.shape) - 1)))
    rows = [{"檢定": "Pearson 卡方檢定" + (" (Yates 校正)" if yates and ct.shape == (2, 2) else ""), "統計量": chi2, "df": dof, "p 值": p}]
    lr = stats.chi2_contingency(ct.to_numpy(), correction=False, lambda_="log-likelihood")
    rows.append({"檢定": "概似比 G 檢定", "統計量": lr[0], "df": lr[2], "p 值": lr[1]})
    if ct.shape == (2, 2):
        tab = ct.to_numpy()
        orf, pf = stats.fisher_exact(tab)
        rows.append({"檢定": "Fisher 精確檢定", "統計量": orf, "df": np.nan, "p 值": pf})
        a, b, c, dd = tab.ravel().astype(float) + (0.5 if (tab == 0).any() else 0)
        or_ = a * dd / (b * c)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / dd)
        rr = (a / (a + b)) / (c / (c + dd))
        se_rr = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + dd))
        res.add("2×2 效果量", pd.DataFrame([
            {"指標": f"勝算比 OR ({ct.index[0]} vs {ct.index[1]}，以 {ct.columns[0]} 為事件)", "估計值": or_,
             "95% CI 下限": np.exp(np.log(or_) - 1.96 * se), "95% CI 上限": np.exp(np.log(or_) + 1.96 * se)},
            {"指標": "相對風險 RR", "估計值": rr, "95% CI 下限": np.exp(np.log(rr) - 1.96 * se_rr), "95% CI 上限": np.exp(np.log(rr) + 1.96 * se_rr)},
        ]))
    rows.append({"檢定": "Cramér's V (效果量)", "統計量": v, "df": np.nan, "p 值": np.nan})
    res.add("檢定結果", pd.DataFrame(rows))
    low = (exp < 5).mean()
    res.messages.append(f"卡方檢定：χ²({dof}) = {chi2:.3f}，{sig_text(p)}，Cramér's V = {v:.3f}。")
    if low > 0.2:
        res.warnings.append(f"有 {low * 100:.0f}% 的期望次數 < 5，卡方檢定可能不準確" + ("，請參考 Fisher 精確檢定。" if ct.shape == (2, 2) else "，建議合併類別。"))
    return res


# ============================================================ 存活分析
def _event01(e: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(e):
        return (pd.to_numeric(e, errors="coerce").fillna(0) > 0).astype(int).to_numpy()
    s = e.astype(str).str.strip().str.lower()
    return s.isin({"1", "true", "yes", "y", "dead", "death", "event", "died", "死亡", "事件", "是", "復發"}).astype(int).to_numpy()


def km_estimate(time, event, conf: float = 0.95) -> pd.DataFrame:
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    ok = np.isfinite(t)
    t, e = t[ok], e[ok]
    n = len(t)
    if n == 0:
        return pd.DataFrame(columns=["time", "n_risk", "events", "censored", "survival", "se", "ci_low", "ci_high"])
    uniq = np.unique(t)
    ts = np.sort(t)
    te = np.sort(t[e == 1])
    n_risk = n - np.searchsorted(ts, uniq, "left")
    total = np.searchsorted(ts, uniq, "right") - np.searchsorted(ts, uniq, "left")
    d = np.searchsorted(te, uniq, "right") - np.searchsorted(te, uniq, "left")
    c = total - d
    surv = np.cumprod(1 - d / n_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(n_risk > d, d / (n_risk * (n_risk - d)), np.inf * (d > 0))
    var_sum = np.cumsum(np.nan_to_num(g, nan=0.0))
    z = stats.norm.ppf((1 + conf) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = surv * np.sqrt(var_sum)
        theta_se = np.sqrt(var_sum) / np.abs(np.log(surv))
        lo = surv ** np.exp(z * theta_se)
        hi = surv ** np.exp(-z * theta_se)
    lo = np.where((surv > 0) & (surv < 1), lo, surv)
    hi = np.where((surv > 0) & (surv < 1), hi, np.where(surv >= 1, 1.0, 0.0))
    out = pd.DataFrame({"time": uniq, "n_risk": n_risk, "events": d, "censored": c, "survival": surv, "se": se,
                        "ci_low": np.nan_to_num(lo, nan=0.0), "ci_high": np.nan_to_num(hi, nan=1.0)})
    if uniq[0] > 0:
        out = pd.concat([pd.DataFrame([{"time": 0.0, "n_risk": n, "events": 0, "censored": 0, "survival": 1.0, "se": 0.0,
                                        "ci_low": 1.0, "ci_high": 1.0}]), out], ignore_index=True)
    return out


def km_median(tab: pd.DataFrame) -> tuple[float, float, float]:
    def first_below(col):
        hit = tab.loc[tab[col] <= 0.5, "time"]
        return float(hit.iloc[0]) if len(hit) else np.nan

    return first_below("survival"), first_below("ci_low"), first_below("ci_high")


def logrank_test(time, event, groups) -> tuple[float, int, float]:
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    g = np.asarray(groups)
    ok = np.isfinite(t)
    t, e, g = t[ok], e[ok], g[ok]
    lv = list(dict.fromkeys(g.tolist()))
    k = len(lv)
    if k < 2:
        return np.nan, 0, np.nan
    times = np.unique(t[e == 1])
    nr = np.zeros((len(times), k))
    dd = np.zeros((len(times), k))
    for j, level in enumerate(lv):
        tg = np.sort(t[g == level])
        teg = np.sort(t[(g == level) & (e == 1)])
        nr[:, j] = len(tg) - np.searchsorted(tg, times, "left")
        dd[:, j] = np.searchsorted(teg, times, "right") - np.searchsorted(teg, times, "left")
    n = nr.sum(1)
    d = dd.sum(1)
    E = (d[:, None] * nr / n[:, None]).sum(0)
    O = dd.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(n > 1, d * (n - d) / (n ** 2 * (n - 1)), 0.0)
    V = np.diag((f * n) @ nr) - (nr * f[:, None]).T @ nr
    diff = (O - E)[:-1]
    chi2 = float(diff @ np.linalg.pinv(V[:-1, :-1]) @ diff)
    return chi2, k - 1, float(stats.chi2.sf(chi2, k - 1))


def survival_analysis(df: pd.DataFrame, time: str, event: str, group: str | None = None,
                      landmarks: list[float] | None = None, correction: str = "holm") -> StatResult:
    res = StatResult(f"存活分析 (Kaplan-Meier)：{time}")
    d = df[[c for c in (time, event, group) if c]].copy()
    d[time] = pd.to_numeric(d[time], errors="coerce")
    d = d.dropna()
    d["_e"] = _event01(d[event])
    groups = list(dict.fromkeys(d[group].tolist())) if group else ["全部"]
    rows, lm_rows = [], []
    for lv in groups:
        sub = d if not group else d[d[group] == lv]
        tab = km_estimate(sub[time], sub["_e"])
        med, lo, hi = km_median(tab)
        rows.append({"組別": lv, "n": len(sub), "事件數": int(sub["_e"].sum()), "設限數": int((1 - sub["_e"]).sum()),
                     "中位存活時間": med, "95% CI 下限": lo, "95% CI 上限": hi})
        for L in landmarks or []:
            before = tab[tab["time"] <= L]
            r = before.iloc[-1] if len(before) else tab.iloc[0]
            lm_rows.append({"組別": lv, "時間點": L, "存活率": r["survival"], "95% CI 下限": r["ci_low"], "95% CI 上限": r["ci_high"],
                            "風險人數": int((sub[time] >= L).sum())})
    summ = pd.DataFrame(rows)
    res.add("各組摘要", summ)
    if lm_rows:
        res.add("特定時間點存活率", pd.DataFrame(lm_rows))
    if group and len(groups) >= 2:
        chi2, dof, p = logrank_test(d[time], d["_e"], d[group])
        res.add("Log-rank 檢定", pd.DataFrame([{"χ²": chi2, "df": dof, "p 值": p}]))
        res.messages.append(f"Log-rank 檢定：χ²({dof}) = {chi2:.3f}，{sig_text(p)}。")
        if len(groups) > 2:
            pr = []
            for a, b in itertools.combinations(groups, 2):
                sub = d[d[group].isin([a, b])]
                c2, _, pp = logrank_test(sub[time], sub["_e"], sub[group])
                pr.append({"group1": a, "group2": b, "χ²": c2, "p": pp})
            prd = pd.DataFrame(pr)
            prd["p_adj"] = p_adjust_array(prd["p"].to_numpy(), correction)
            res.add(f"成對 log-rank ({correction})", prd)
    for r in summ.itertuples():
        med = "未達 (NR)" if not np.isfinite(r.中位存活時間) else f"{r.中位存活時間:.2f}"
        res.messages.append(f"{r.組別}：n = {r.n}，事件 {r.事件數}，中位存活時間 = {med}")
    return res


def cox_regression(df: pd.DataFrame, time: str, event: str, covariates: list[str], cat_covs: list[str] | None = None,
                   ties: str = "efron") -> StatResult:
    import statsmodels.formula.api as smf

    cat_covs = cat_covs or []
    res = StatResult("Cox 比例風險迴歸")
    d = df[[time, event] + covariates + cat_covs].copy()
    d[time] = pd.to_numeric(d[time], errors="coerce")
    for c in covariates:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    d["_e"] = _event01(d[event])
    terms = [_q(c) for c in covariates] + [f"C({_q(c)})" for c in cat_covs]
    if not terms:
        res.warnings.append("請至少選擇一個共變數。")
        return res
    model = smf.phreg(f"{_q(time)} ~ {' + '.join(terms)}", data=d, status=d["_e"].to_numpy(), ties=ties).fit()
    names = model.model.exog_names
    ci = np.asarray(model.conf_int())
    params = np.asarray(model.params)
    out = pd.DataFrame({
        "變項": [_pretty_term(t) for t in names], "係數": params, "標準誤": np.asarray(model.bse),
        "風險比 HR": np.exp(params), "HR 95% CI 下限": np.exp(ci[:, 0]), "HR 95% CI 上限": np.exp(ci[:, 1]),
        "p 值": np.asarray(model.pvalues),
    })
    res.add("Cox 迴歸結果", out)
    res.add("模型資訊", pd.DataFrame([{"n": len(d), "事件數": int(d["_e"].sum()), "Log-likelihood": model.llf, "ties": ties}]))
    for r in out.itertuples():
        res.messages.append(f"{r.變項}：HR = {r._4:.3f} (95% CI {r._5:.3f} ~ {r._6:.3f})，{sig_text(r._7)}")
    res.messages.append("HR > 1 表示風險增加；類別變項以第一個類別為參考組。")
    return res


# ============================================================ ROC
def roc_curve(y, score) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, float)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = 1 + idx - tps
    P, N = y.sum(), (~y).sum()
    tpr = np.r_[0.0, tps / max(P, 1)]
    fpr = np.r_[0.0, fps / max(N, 1)]
    thr = np.r_[np.inf, s_sorted[idx]]
    return fpr, tpr, thr, float(_trapz(tpr, fpr))


def _delong_components(y: np.ndarray, scores: np.ndarray):
    """scores: (k, n)。回傳 aucs, 共變異數矩陣 (Sun & Xu 快速 DeLong)。"""
    pos = scores[:, y == 1]
    neg = scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    v01 = np.empty((k, m))
    v10 = np.empty((k, n))
    aucs = np.empty(k)
    for i in range(k):
        tx = stats.rankdata(pos[i])
        ty = stats.rankdata(neg[i])
        tz = stats.rankdata(np.r_[pos[i], neg[i]])
        aucs[i] = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01[i] = (tz[:m] - tx) / n
        v10[i] = 1 - (tz[m:] - ty) / m
    s01 = np.atleast_2d(np.cov(v01))
    s10 = np.atleast_2d(np.cov(v10))
    return aucs, s01 / m + s10 / n


def roc_analysis(df: pd.DataFrame, truth: str, positive, scores: list[str], ci: float = 0.95) -> StatResult:
    res = StatResult(f"ROC 分析：{truth} = {positive}")
    d = df[[truth] + scores].copy()
    for c in scores:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    y = (d[truth].astype(str) == str(positive)).astype(int).to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        res.warnings.append("需要同時有陽性與陰性個案。")
        return res
    z = stats.norm.ppf((1 + ci) / 2)
    rows = []
    for c in scores:
        s = d[c].to_numpy(float)
        fpr, tpr, thr, auc = roc_curve(y, s)
        flipped = auc < 0.5
        if flipped:
            s = -s
            fpr, tpr, thr, auc = roc_curve(y, s)
        aucs, cov = _delong_components(y, s[None, :])
        se = float(np.sqrt(max(cov[0, 0], 0)))
        j = tpr - fpr
        bi = int(np.argmax(j[1:]) + 1)
        cut = thr[bi]
        pred = s >= cut
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        tn = int((~pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        rows.append({
            "指標": c + (" (反向)" if flipped else ""), "AUC": auc, "標準誤": se,
            "AUC CI 下限": max(auc - z * se, 0), "AUC CI 上限": min(auc + z * se, 1),
            "p 值 (vs 0.5)": 2 * stats.norm.sf(abs(auc - 0.5) / se) if se > 0 else np.nan,
            "最佳切點 (Youden)": -cut if flipped else cut, "敏感度": tpr[bi], "特異度": 1 - fpr[bi],
            "PPV": tp / (tp + fp) if tp + fp else np.nan, "NPV": tn / (tn + fn) if tn + fn else np.nan,
            "準確率": (tp + tn) / len(y), "Youden J": j[bi],
        })
    out = pd.DataFrame(rows)
    res.add("ROC 結果", out)
    if len(scores) >= 2:
        S = np.vstack([d[c].to_numpy(float) * (-1 if r["指標"].endswith("(反向)") else 1) for c, r in zip(scores, rows)])
        aucs, cov = _delong_components(y, S)
        pr = []
        for i, j2 in itertools.combinations(range(len(scores)), 2):
            var = cov[i, i] + cov[j2, j2] - 2 * cov[i, j2]
            zz = (aucs[i] - aucs[j2]) / np.sqrt(var) if var > 0 else np.nan
            pr.append({"指標 1": scores[i], "指標 2": scores[j2], "AUC 差": aucs[i] - aucs[j2], "z": zz,
                       "p 值 (DeLong)": 2 * stats.norm.sf(abs(zz)) if np.isfinite(zz) else np.nan})
        res.add("AUC 比較 (DeLong 檢定)", pd.DataFrame(pr))
    for r in rows:
        res.messages.append(f"{r['指標']}：AUC = {r['AUC']:.3f} ({int(ci * 100)}% CI {r['AUC CI 下限']:.3f} ~ {r['AUC CI 上限']:.3f})；"
                            f"最佳切點 {r['最佳切點 (Youden)']:.4g}，敏感度 {r['敏感度']:.1%}、特異度 {r['特異度']:.1%}")
    res.messages.append("AUC 判讀：0.9 以上極佳、0.8–0.9 良好、0.7–0.8 尚可、0.5–0.7 不佳。")
    return res


# ============================================================ 劑量反應
def four_pl(logx, bottom, top, logec50, hill):
    return bottom + (top - bottom) / (1 + 10 ** ((logec50 - logx) * hill))


def fit_4pl(x, y) -> dict:
    from scipy.optimize import curve_fit

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    lx, yy = np.log10(x[ok]), y[ok]
    if len(lx) < 4 or len(np.unique(lx)) < 3:
        raise ValueError("4PL 擬合至少需要 3 個不同的正濃度、4 個資料點")
    slope = np.sign(np.corrcoef(lx, yy)[0, 1]) or 1.0
    p0 = [yy.min(), yy.max(), np.median(lx), slope]
    if slope < 0:
        p0 = [yy.min(), yy.max(), np.median(lx), -1.0]
    popt, pcov = curve_fit(four_pl, lx, yy, p0=p0, maxfev=20000)
    pred = four_pl(lx, *popt)
    ss_res = np.sum((yy - pred) ** 2)
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    dof = max(len(lx) - 4, 1)
    tc = stats.t.ppf(0.975, dof)
    return {"bottom": popt[0], "top": popt[1], "logec50": popt[2], "hill": popt[3], "ec50": 10 ** popt[2],
            "ec50_low": 10 ** (popt[2] - tc * perr[2]), "ec50_high": 10 ** (popt[2] + tc * perr[2]),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan, "n": int(len(lx)), "params": popt}


def dose_response(df: pd.DataFrame, x: str, y: str, group: str | None = None) -> StatResult:
    res = StatResult(f"劑量反應 4PL 擬合：{y} ~ {x}")
    rows = []
    groups = list(dict.fromkeys(df[group].dropna().tolist())) if group else [None]
    for lv in groups:
        sub = df if lv is None else df[df[group] == lv]
        try:
            f = fit_4pl(pd.to_numeric(sub[x], errors="coerce"), pd.to_numeric(sub[y], errors="coerce"))
            rows.append({"組別": lv if lv is not None else "全部", "EC50/IC50": f["ec50"], "EC50 95% CI 下限": f["ec50_low"],
                         "EC50 95% CI 上限": f["ec50_high"], "logEC50": f["logec50"], "Hill 斜率": f["hill"],
                         "Bottom": f["bottom"], "Top": f["top"], "R²": f["r2"], "n": f["n"]})
        except Exception as err:
            res.warnings.append(f"{lv}：擬合失敗 ({err})")
    out = pd.DataFrame(rows)
    res.add("4PL 參數", out)
    for r in rows:
        res.messages.append(f"{r['組別']}：EC50 = {r['EC50/IC50']:.4g} (95% CI {r['EC50 95% CI 下限']:.4g} ~ {r['EC50 95% CI 上限']:.4g})，Hill = {r['Hill 斜率']:.2f}，R² = {r['R²']:.3f}")
    if (pd.to_numeric(df[x], errors="coerce") <= 0).any():
        res.warnings.append("濃度 ≤ 0 的資料點 (如空白對照) 無法取對數，已排除於擬合之外。")
    return res


# ============================================================ 其他
def p_adjust_table(df: pd.DataFrame, pcol: str, methods: list[str]) -> StatResult:
    res = StatResult("多重比較校正")
    out = df.copy()
    p = pd.to_numeric(out[pcol], errors="coerce").to_numpy()
    names = {"bonferroni": "Bonferroni", "holm": "Holm", "fdr_bh": "FDR (BH)", "fdr_by": "FDR (BY)", "sidak": "Šidák", "hommel": "Hommel"}
    for m in methods:
        out[f"{pcol}_{names.get(m, m)}"] = p_adjust_array(p, m)
    res.add("校正結果", out)
    for m in methods:
        col = f"{pcol}_{names.get(m, m)}"
        res.messages.append(f"{names.get(m, m)}：{int((out[col] < ALPHA).sum())} / {int(np.isfinite(p).sum())} 項 p < {ALPHA}")
    return res


def power_analysis(test: str, solve: str, effect: float | None, n: float | None, alpha: float, power: float | None,
                   ratio: float = 1.0, k_groups: int = 3, alternative: str = "two-sided") -> StatResult:
    from statsmodels.stats import power as smp

    res = StatResult("檢定力 / 樣本數估算")
    kw = dict(effect_size=effect, alpha=alpha, power=power)
    if test == "兩獨立樣本 t 檢定":
        engine = smp.TTestIndPower()
        kw.update(nobs1=n, ratio=ratio, alternative=alternative)
    elif test == "配對 / 單一樣本 t 檢定":
        engine = smp.TTestPower()
        kw.update(nobs=n, alternative=alternative)
    else:
        engine = smp.FTestAnovaPower()
        kw.update(nobs=n, k_groups=k_groups)
    key = {"樣本數": "nobs1" if test == "兩獨立樣本 t 檢定" else "nobs", "檢定力": "power", "效果量": "effect_size"}[solve]
    kw[key] = None
    value = float(np.squeeze(engine.solve_power(**kw)))
    kw[key] = value
    row = {"檢定": test, "效果量": kw["effect_size"], "α": alpha, "檢定力": kw["power"]}
    if test == "兩獨立樣本 t 檢定":
        row.update({"第一組 n": kw["nobs1"], "第二組 n": kw["nobs1"] * ratio, "總 n": kw["nobs1"] * (1 + ratio)})
    elif test == "配對 / 單一樣本 t 檢定":
        row.update({"n (配對數)": kw["nobs"]})
    else:
        row.update({"總 n": kw["nobs"], "組數": k_groups, "每組 n": kw["nobs"] / k_groups})
    res.add("結果", pd.DataFrame([row]))
    if solve == "樣本數":
        need = int(np.ceil(value))
        res.messages.append(f"**所需樣本數：{need}**" + (" (每組)" if test == "兩獨立樣本 t 檢定" else " (配對數)" if "配對" in test else " (總數)"))
    elif solve == "檢定力":
        res.messages.append(f"**檢定力 = {value:.3f}**")
    else:
        res.messages.append(f"**可偵測的最小效果量 = {value:.3f}**")
    res.messages.append("效果量參考：Cohen's d 0.2 小、0.5 中、0.8 大；ANOVA 的 Cohen's f 0.1 小、0.25 中、0.4 大。")
    return res
