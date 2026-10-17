"""統計頁英文化：每一種分析在英文模式下，標題、解讀、警告、表名、欄名與文字儲存格都不得殘留中文。"""

import re

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from biostat import data, i18n
from biostat import stats as bs
from biostat.i18n import _

CJK = re.compile(r"[　-〿一-鿿＀-￯]")
D = data.make_demo_data()


def _lab(df, col):
    return df.assign(**{col: data.as_category_labels(df[col])})


ANALYSES = {
    "descriptive": lambda: bs.descriptive(D["Groups"], ["IL6", "TNFa"], "Group"),
    "normality": lambda: bs.normality(D["Groups"], ["IL6", "TumorVolume"], "Group"),
    "variance": lambda: bs.variance_homogeneity(D["Groups"], "IL6", "Group"),
    "two_group": lambda: bs.two_group(D["Groups"], "IL6", "Group", "Control", "DrugA"),
    "two_group_mwu": lambda: bs.two_group(D["Groups"], "TumorVolume", "Group", "Control", "DrugC"),
    "paired": lambda: bs.paired_test(D["Paired"], "Before", "After"),
    "one_way_tukey": lambda: bs.one_way(D["Groups"], "IL6", "Group", "Tukey HSD"),
    "one_way_gh": lambda: bs.one_way(D["Groups"], "IL6", "Group", "Games-Howell"),
    "one_way_dunn": lambda: bs.one_way(D["Groups"], "IL6", "Group", "Dunn (Holm)"),
    "one_way_holm_t": lambda: bs.one_way(D["Groups"], "IL6", "Group", "Holm t 檢定"),
    "two_way": lambda: bs.two_way(D["Groups"], "IL6", "Group", "Sex"),
    "rm_anova": lambda: bs.rm_anova(D["TimeCourse"], "Response", "Subject", "Day"),
    "correlation": lambda: bs.correlation(D["Groups"], ["IL6", "TNFa", "TumorVolume"], "spearman"),
    "linear": lambda: bs.linear_regression(D["Groups"], "TumorVolume", ["IL6", "TNFa"], ["Sex"]),
    "logistic": lambda: bs.logistic_regression(_lab(D["Clinical"], "Outcome"), "Outcome",
                                               data.as_category_labels(D["Clinical"]["Outcome"]).dropna().iloc[0], ["Age"], ["Sex"]),
    "contingency": lambda: bs.contingency(_lab(_lab(D["Clinical"], "Treatment"), "Outcome"), "Treatment", "Outcome"),
    "contingency_2x2": lambda: bs.contingency(_lab(_lab(D["Clinical"], "Sex"), "Smoker"), "Sex", "Smoker"),
    "survival": lambda: bs.survival_analysis(_lab(D["Survival"], "RiskGroup"), "Time_months", "Event", "RiskGroup", [12, 24]),
    "cox": lambda: bs.cox_regression(D["Survival"], "Time_months", "Event",
                                     [c for c in data.numeric_columns(D["Survival"]) if c not in ("Time_months", "Event")][:2]),
    "roc": lambda: bs.roc_analysis(_lab(D["Diagnostic"], "Disease"), "Disease",
                                   sorted(data.as_category_labels(D["Diagnostic"]["Disease"]).dropna().unique())[-1],
                                   ["MarkerA", "MarkerB"]),
    "dose": lambda: bs.dose_response(D["DoseResponse"], "Concentration_uM", "Inhibition_pct", "Compound"),
    "p_adjust": lambda: bs.p_adjust_table(D["Volcano"].head(50), "pvalue" if "pvalue" in D["Volcano"] else "padj",
                                          ["bonferroni", "holm", "fdr_bh"]),
    "power_n": lambda: bs.power_analysis("兩獨立樣本 t 檢定", "樣本數", 0.5, None, 0.05, 0.8),
    "power_power": lambda: bs.power_analysis("配對 / 單一樣本 t 檢定", "檢定力", 0.5, 20, 0.05, None),
    "power_es": lambda: bs.power_analysis("單因子 ANOVA", "效果量", None, 60, 0.05, 0.8, k_groups=3),
}


def _texts(res) -> list[str]:
    out = [res.title] + res.messages + res.warnings
    for name, tab in res.tables:
        tab = i18n.table(tab)
        out += [_(name)] + [str(c) for c in tab.columns]
        out += [v for c in tab.columns if tab[c].dtype == object for v in tab[c] if isinstance(v, str)]
    return out


@pytest.mark.parametrize("name", list(ANALYSES))
def test_stats_english(name):
    i18n.set_lang("en")
    try:
        res = ANALYSES[name]()
        leaks = sorted({t for t in _texts(res) if CJK.search(t)})
    finally:
        i18n.set_lang("zh")
    assert leaks == []
