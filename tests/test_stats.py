"""統計正確性：每個分析都與獨立的參考實作 (scipy / statsmodels / lifelines / sklearn / 暴力法 / 教科書數值) 比對。"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from biostat import data
from biostat import stats as bs

RNG = np.random.default_rng(2024)
DEMO = data.make_demo_data()


def _groups(k=3, n=(12, 15, 18), shift=(0.0, 0.8, 1.6), sd=(1.0, 1.5, 3.0)):
    rows = []
    for i in range(k):
        for v in RNG.normal(10 + shift[i], sd[i], n[i]):
            rows.append({"g": f"G{i + 1}", "y": v})
    return pd.DataFrame(rows)


def _tab(res, name):
    return dict(res.tables)[name]


# ------------------------------------------------------------ 描述與兩組
def test_describe_ci_matches_t_interval():
    x = RNG.normal(5, 2, 23)
    d = bs.describe_array(x)
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    assert d["95% CI 下限"] == pytest.approx(lo) and d["95% CI 上限"] == pytest.approx(hi)
    assert d["標準差 SD"] == pytest.approx(np.std(x, ddof=1))


def test_cohen_d_textbook():
    assert bs.cohen_d(np.array([1, 2, 3, 4, 5.0]), np.array([3, 4, 5, 6, 7.0])) == pytest.approx(-2 / np.sqrt(2.5))


def test_two_group_against_scipy_and_statsmodels():
    from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

    df = _groups(k=2, n=(14, 21), shift=(0, 1.2), sd=(1, 2.5))
    a, b = (df.loc[df.g == g, "y"].to_numpy() for g in ("G1", "G2"))
    t = _tab(bs.two_group(df, "y", "g", "G1", "G2"), "檢定結果").set_index("檢定")
    cm = CompareMeans(DescrStatsW(a), DescrStatsW(b))
    for name, usevar, eq in (("Student t 檢定 (等變異)", "pooled", True), ("Welch t 檢定 (不等變異)", "unequal", False)):
        ref = stats.ttest_ind(a, b, equal_var=eq)
        assert t.loc[name, "統計量"] == pytest.approx(ref.statistic)
        assert t.loc[name, "p 值"] == pytest.approx(ref.pvalue)
        lo, hi = cm.tconfint_diff(usevar=usevar)
        assert (t.loc[name, "CI 下限"], t.loc[name, "CI 上限"]) == pytest.approx((lo, hi))
        assert t.loc[name, "自由度"] == pytest.approx(cm.ttest_ind(usevar=usevar)[2])
    assert t.loc["Mann-Whitney U (無母數)", "p 值"] == pytest.approx(stats.mannwhitneyu(a, b).pvalue)


def test_paired_against_scipy():
    df = DEMO["Paired"]
    t = _tab(bs.paired_test(df, "Before", "After"), "檢定結果").set_index("檢定")
    assert t.loc["配對 t 檢定", "p 值"] == pytest.approx(stats.ttest_rel(df.Before, df.After).pvalue)
    assert t.loc["Wilcoxon 符號等級檢定", "p 值"] == pytest.approx(stats.wilcoxon(df.Before, df.After).pvalue)


# ------------------------------------------------------------ 多組
def test_one_way_anova_welch_kruskal():
    from statsmodels.stats.oneway import anova_oneway

    df = _groups()
    xs = [df.loc[df.g == g, "y"].to_numpy() for g in ("G1", "G2", "G3")]
    res = bs.one_way(df, "y", "g")
    overall = _tab(res, "整體檢定").set_index("檢定")
    ref = stats.f_oneway(*xs)
    assert overall.loc["One-way ANOVA", "統計量"] == pytest.approx(ref.statistic)
    assert overall.loc["One-way ANOVA", "p 值"] == pytest.approx(ref.pvalue)
    w = anova_oneway(xs, use_var="unequal")
    assert overall.loc["Welch ANOVA (不等變異)", "統計量"] == pytest.approx(w.statistic)
    assert overall.loc["Welch ANOVA (不等變異)", "df2"] == pytest.approx(w.df_denom)
    assert overall.loc["Welch ANOVA (不等變異)", "p 值"] == pytest.approx(w.pvalue)
    assert overall.loc["Kruskal-Wallis (無母數)", "p 值"] == pytest.approx(stats.kruskal(*xs).pvalue)
    tab = _tab(res, "ANOVA 表")
    ss_t = ((np.concatenate(xs) - np.concatenate(xs).mean()) ** 2).sum()
    assert tab["η²"].iloc[0] == pytest.approx(tab["SS"].iloc[0] / ss_t)


def test_games_howell_two_groups_equals_welch():
    """k = 2 時 Games-Howell 的 studentized range 分布退化為 Welch t。"""
    a, b = RNG.normal(0, 1, 11), RNG.normal(1, 3, 17)
    gh = bs.games_howell({"a": a, "b": b})
    assert gh["p_adj"].iloc[0] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue, rel=1e-6)


def test_tukey_matches_scipy():
    df = _groups()
    xs = [df.loc[df.g == g, "y"].to_numpy() for g in ("G1", "G2", "G3")]
    tab = bs.tukey_table({"G1": xs[0], "G2": xs[1], "G3": xs[2]})
    ref = stats.tukey_hsd(*xs).pvalue
    assert tab["p_adj"].tolist() == pytest.approx([ref[0, 1], ref[0, 2], ref[1, 2]])


def test_dunn_two_groups_equals_mwu_normal_approx():
    a, b = RNG.normal(0, 1, 13), RNG.normal(0.7, 1, 16)
    dunn = bs.dunn_test({"a": a, "b": b}, None)
    ref = stats.mannwhitneyu(a, b, method="asymptotic", use_continuity=False).pvalue
    assert dunn["p"].iloc[0] == pytest.approx(ref)


def test_p_adjust_bh_textbook():
    out = bs.p_adjust_array(np.array([0.01, 0.04, 0.03, 0.005]), "fdr_bh")
    assert out.tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])


# ------------------------------------------------------------ 相關、迴歸
@pytest.mark.parametrize("method,fn", [("pearson", stats.pearsonr), ("spearman", stats.spearmanr), ("kendall", stats.kendalltau)])
def test_correlation_r_and_p(method, fn):
    df = DEMO["Groups"]
    row = _tab(bs.correlation(df, ["IL6", "TNFa"], method), "成對相關").iloc[0]
    r, p = fn(df.IL6, df.TNFa)[:2]
    assert row["r"] == pytest.approx(r) and row["p 值"] == pytest.approx(p)
    assert row["CI 下限"] < row["r"] < row["CI 上限"]


def test_pearson_ci_fisher_z():
    df = DEMO["Groups"]
    row = _tab(bs.correlation(df, ["IL6", "TNFa"], "pearson"), "成對相關").iloc[0]
    ci = stats.pearsonr(df.IL6, df.TNFa).confidence_interval(0.95)
    assert (row["CI 下限"], row["CI 上限"]) == pytest.approx((ci.low, ci.high))


def test_linear_regression_matches_lstsq():
    df = DEMO["Groups"]
    coef = _tab(bs.linear_regression(df, "TumorVolume", ["IL6", "TNFa"]), "迴歸係數")
    X = np.column_stack([np.ones(len(df)), df.IL6, df.TNFa])
    beta = np.linalg.lstsq(X, df.TumorVolume.to_numpy(), rcond=None)[0]
    assert coef["係數 B"].tolist() == pytest.approx(beta.tolist())


def test_logistic_regression_matches_sklearn():
    from sklearn.linear_model import LogisticRegression

    df = DEMO["Clinical"]
    pos = df["Outcome"].dropna().iloc[0]
    coef = _tab(bs.logistic_regression(df, "Outcome", pos, ["Age"]), "迴歸係數 (勝算比)")
    y = (df["Outcome"].astype(str) == str(pos)).astype(int)
    sk = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10000).fit(df[["Age"]], y)
    assert coef["係數 B"].tolist() == pytest.approx([sk.intercept_[0], sk.coef_[0, 0]], rel=1e-4, abs=1e-6)


# ------------------------------------------------------------ 類別資料
def test_contingency_chi2_fisher_and_cramers_v():
    df = pd.DataFrame({"a": ["x"] * 30 + ["y"] * 30, "b": ["p"] * 20 + ["q"] * 10 + ["p"] * 9 + ["q"] * 21})
    res = bs.contingency(df, "a", "b")
    t = _tab(res, "檢定結果").set_index("檢定")
    ct = pd.crosstab(df.a, df.b).to_numpy()
    chi2, p, *_ = stats.chi2_contingency(ct, correction=True)
    assert t.iloc[0]["統計量"] == pytest.approx(chi2) and t.iloc[0]["p 值"] == pytest.approx(p)
    assert t.loc["Fisher 精確檢定", "p 值"] == pytest.approx(stats.fisher_exact(ct)[1])
    phi = np.sqrt(stats.chi2_contingency(ct, correction=False)[0] / ct.sum())
    assert t.loc["Cramér's V (效果量)", "統計量"] == pytest.approx(phi)
    eff = _tab(res, "2×2 效果量")
    assert eff["估計值"].iloc[0] == pytest.approx(20 * 21 / (10 * 9))


# ------------------------------------------------------------ 存活分析
def test_kaplan_meier_and_logrank_match_lifelines():
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    df = DEMO["Survival"]
    e = bs._event01(df["Event"])
    tab = bs.km_estimate(df["Time_months"], e)
    km = KaplanMeierFitter().fit(df["Time_months"], e)
    ref = km.survival_function_["KM_estimate"]
    merged = tab.set_index("time")["survival"].reindex(ref.index)
    assert merged.dropna().to_numpy() == pytest.approx(ref.loc[merged.dropna().index].to_numpy())
    assert bs.km_median(tab)[0] == pytest.approx(km.median_survival_time_)
    chi2, dof, p = bs.logrank_test(df["Time_months"], e, df["RiskGroup"])
    lr = multivariate_logrank_test(df["Time_months"], df["RiskGroup"], e)
    assert chi2 == pytest.approx(lr.test_statistic) and p == pytest.approx(lr.p_value)


def test_km_greenwood_se_matches_statsmodels():
    from statsmodels.duration.survfunc import SurvfuncRight

    df = DEMO["Survival"]
    e = bs._event01(df["Event"])
    tab = bs.km_estimate(df["Time_months"], e)
    sf = SurvfuncRight(df["Time_months"].to_numpy(), e)
    ours = tab.set_index("time").loc[sf.surv_times]
    assert ours["survival"].to_numpy() == pytest.approx(sf.surv_prob)
    assert ours["se"].to_numpy() == pytest.approx(sf.surv_prob_se, rel=1e-6)


def test_cox_matches_lifelines():
    from lifelines import CoxPHFitter

    df = DEMO["Survival"]
    num = [c for c in data.numeric_columns(df) if c not in ("Time_months", "Event")][:2]
    out = _tab(bs.cox_regression(df, "Time_months", "Event", num), "Cox 迴歸結果")
    d = df[["Time_months"] + num].copy()
    d["E"] = bs._event01(df["Event"])
    cph = CoxPHFitter().fit(d.dropna(), "Time_months", "E")
    assert out["係數"].to_numpy() == pytest.approx(cph.params_[num].to_numpy(), rel=1e-4)


# ------------------------------------------------------------ ROC
def test_auc_matches_sklearn_and_delong_bruteforce():
    from sklearn.metrics import roc_auc_score

    df = DEMO["Diagnostic"]
    pos = sorted(df["Disease"].dropna().unique().tolist())[-1]
    y = (df["Disease"].astype(str) == str(pos)).astype(int).to_numpy()
    s = df["MarkerA"].to_numpy(float)
    out = _tab(bs.roc_analysis(df, "Disease", pos, ["MarkerA"]), "ROC 結果").iloc[0]
    auc = roc_auc_score(y, s)
    assert out["AUC"] == pytest.approx(max(auc, 1 - auc))
    # DeLong 變異數：以 placement values 的 O(mn) 定義直接計算
    sign = -1 if auc < 0.5 else 1
    x, z = sign * s[y == 1], sign * s[y == 0]
    psi = (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])
    v10, v01 = psi.mean(1), psi.mean(0)
    se = np.sqrt(v10.var(ddof=1) / len(x) + v01.var(ddof=1) / len(z))
    assert out["標準誤"] == pytest.approx(se)


# ------------------------------------------------------------ 劑量反應、檢定力
def test_4pl_recovers_known_ec50():
    x = np.repeat(np.logspace(-3, 2, 10), 3)
    y = bs.four_pl(np.log10(x), 5, 95, np.log10(0.8), 1.3) + RNG.normal(0, 0.3, len(x))
    f = bs.fit_4pl(x, y)
    assert f["ec50"] == pytest.approx(0.8, rel=0.05) and f["hill"] == pytest.approx(1.3, rel=0.1)


def test_power_textbook_sample_size():
    res = bs.power_analysis("兩獨立樣本 t 檢定", "樣本數", 0.5, None, 0.05, 0.8)
    assert int(np.ceil(_tab(res, "結果")["第一組 n"].iloc[0])) == 64  # Cohen (1988)：d = 0.5、α = .05、power = .8 → 每組 64
