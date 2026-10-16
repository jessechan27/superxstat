"""🧮 統計分析頁。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import export, plots
from .. import stats as bs
from ..data import categorical_columns, guess_column, numeric_columns
from . import common as ui
from .page_plot import style_from_state

GUIDE = """
| 研究問題 | 資料型態 | 符合常態 / 變異數同質 | 不符合 (無母數) |
|---|---|---|---|
| 兩組獨立樣本平均是否不同 | 數值 + 2 組 | Student t / Welch t 檢定 | Mann-Whitney U |
| 同一受試者前後差異 | 兩個數值欄 (配對) | 配對 t 檢定 | Wilcoxon 符號等級 |
| 三組以上平均是否不同 | 數值 + ≥3 組 | One-way ANOVA + Tukey (變異不同質用 Welch ANOVA + Games-Howell) | Kruskal-Wallis + Dunn |
| 兩個因子的效果與交互作用 | 數值 + 兩個分組 | 雙因子 ANOVA | 資料轉換 (log) 或分層比較 |
| 重複量測 (多時間點) | 長格式 + 受試者 ID | 重複量測 ANOVA | Friedman |
| 兩數值變項的關聯 | 兩數值 | Pearson 相關 | Spearman / Kendall |
| 以多個變項預測數值結果 | 數值結果 | 線性迴歸 | — |
| 預測二元結果 (有病/無病) | 0/1 結果 | 邏輯迴歸 (勝算比) | — |
| 兩個類別變項是否相關 | 類別 × 類別 | 卡方檢定 | 期望值 < 5：Fisher 精確檢定 |
| 存活時間比較 | 時間 + 事件 | Kaplan-Meier + log-rank | — |
| 存活的風險因子 | 時間 + 事件 + 共變數 | Cox 比例風險迴歸 (HR) | — |
| 診斷指標的鑑別力 | 0/1 + 連續指標 | ROC / AUC | — |
| 藥物效價 | 濃度 + 反應 | 4PL 曲線 (EC50 / IC50) | — |

**流程建議**：先用「常態性檢定」與 Q-Q 圖檢查分布 → 「變異數同質性」→ 選擇對應檢定 → 多組比較記得做事後檢定與多重比較校正。
"""

ANALYSES = [
    "描述性統計",
    "常態性檢定",
    "變異數同質性檢定",
    "兩組比較 (t 檢定 / Mann-Whitney)",
    "配對比較 (配對 t / Wilcoxon)",
    "多組比較 (ANOVA / Kruskal-Wallis)",
    "雙因子 ANOVA",
    "重複量測 ANOVA / Friedman",
    "相關分析",
    "線性迴歸",
    "邏輯迴歸",
    "卡方 / Fisher 精確檢定",
    "存活分析 (Kaplan-Meier / log-rank)",
    "Cox 比例風險迴歸",
    "ROC / AUC 診斷分析",
    "劑量反應 (4PL / EC50)",
    "多重比較校正 (p 值)",
    "檢定力 / 樣本數估算",
    "📖 如何選擇統計方法",
]


def _num_sel(label, df, key, exclude=(), hints=()):
    nums = numeric_columns(df)
    free = [c for c in nums if c not in exclude]
    default = (guess_column(free, list(hints)) if hints else None) or (free[0] if free else None)
    return ui.selectbox(label, nums, key, default=default)


def _cat_sel(label, df, key, optional=False, exclude=(), hints=()):
    cats = categorical_columns(df)
    free = [c for c in cats if c not in exclude]
    guess = guess_column(free, list(hints)) if hints else None
    if optional:
        return ui.selectbox(label, [None] + cats, key, default=guess, format_func=lambda c: "（無）" if c is None else str(c))
    best = guess or next((c for c in free if 2 <= df[c].nunique() <= 12), free[0] if free else None)
    return ui.selectbox(label, cats, key, default=best)


def _levels(df, col):
    return plots.category_levels(df[col]) if col else []


def _quick(chart: str, df, m: dict, o: dict | None = None, sig: dict | None = None, orders: dict | None = None):
    s = style_from_state()
    s.title = ""
    s.xlabel = s.ylabel = ""
    try:
        res = plots.render(chart, df, m, o or {}, s, sig=sig, orders=orders or {})
        png, _, _ = export.fig_bytes(res.fig, "PNG", dpi=100, rc=res.rc)
        export.close(res.fig)
        st.image(png, width="stretch")
        st.caption("此圖可到「📊 繪圖」頁進一步調整格式並下載高解析度版本。")
    except Exception as err:
        st.caption(f"(無法產生快速圖：{err})")


def _show(res: bs.StatResult | None, key: str) -> None:
    if res is None:
        return
    for w in res.warnings:
        st.warning(w)
    if res.messages:
        st.markdown("\n".join(f"- {msg}" for msg in res.messages))
    ui.show_tables(res.tables, key, expanded_first=True)
    if res.tables:
        st.download_button("⬇️ 下載全部結果 (Excel)", ui.tables_excel(res.tables), f"{key}.xlsx", on_click="ignore",
                           key=f"dlx:{key}")


def render(ds) -> None:
    df = ds.df
    if df.empty:
        st.info("目前沒有資料。")
        return
    name = ui.selectbox("選擇分析方法", ANALYSES, "st:analysis", default="多組比較 (ANOVA / Kruskal-Wallis)")
    if name.startswith("📖"):
        st.markdown(GUIDE)
        return
    res = None
    quick = None
    box = st.container(border=True)
    with box:
        c1, c2, c3 = st.columns(3)
        if name == "描述性統計":
            with c1:
                vals = ui.multiselect("數值變項", numeric_columns(df), "st:desc:vals", default=numeric_columns(df)[:5])
            with c2:
                grp = _cat_sel("分組 (選填)", df, "st:desc:grp", optional=True)
            with c3:
                ci = ui.selectbox("信賴水準", [0.90, 0.95, 0.99], "st:desc:ci", default=0.95, format_func=lambda v: f"{v:.0%}")
            if vals:
                res = bs.descriptive(df, vals, grp, ci)
                if grp:
                    quick = ("box", {"x": grp, "y": vals[0]}, {"points": "散點 (strip)"}, None)
                else:
                    quick = ("hist", {"x": vals[0]}, {"kde": True}, None)
        elif name == "常態性檢定":
            with c1:
                vals = ui.multiselect("數值變項", numeric_columns(df), "st:norm:vals", default=numeric_columns(df)[:3])
            with c2:
                grp = _cat_sel("分組 (選填)", df, "st:norm:grp", optional=True)
            if vals:
                res = bs.normality(df, vals, grp)
                quick = ("qq", {"y": vals[0], "hue": grp}, {}, None)
        elif name == "變異數同質性檢定":
            with c1:
                v = _num_sel("數值變項", df, "st:var:v")
            with c2:
                g = _cat_sel("分組", df, "st:var:g")
            res = bs.variance_homogeneity(df, v, g)
            quick = ("box", {"x": g, "y": v}, {"points": "散點 (strip)"}, None)
        elif name.startswith("兩組比較"):
            with c1:
                v = _num_sel("數值變項", df, "st:two:v")
                g = _cat_sel("分組", df, "st:two:g", exclude=(v,))
            lv = _levels(df, g)
            with c2:
                a = ui.selectbox("組別 A", lv, f"st:two:a:{g}", default=lv[0] if lv else None)
                b = ui.selectbox("組別 B", lv, f"st:two:b:{g}", default=lv[1] if len(lv) > 1 else None)
            with c3:
                alt = ui.selectbox("對立假設", ["two-sided", "greater", "less"], "st:two:alt",
                                   format_func={"two-sided": "雙尾 (A ≠ B)", "greater": "單尾 (A > B)", "less": "單尾 (A < B)"}.get)
            if a == b:
                st.warning("請選擇兩個不同的組別。")
            elif a is not None:
                d = df.assign(**{g: df[g].map(lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x))})
                res = bs.two_group(d, v, g, a, b, alt)
                quick = ("box", {"x": g, "y": v}, {"points": "散點 (strip)"},
                         dict(enabled=True, test="welch", pairs_mode="all", correction=None, display="p 值 + 星號", hide_ns=False))
                orders = {g: [a, b]}
        elif name.startswith("配對比較"):
            with c1:
                a = _num_sel("條件 1 (例如 Before)", df, "st:pair:a", hints=("before", "pre", "baseline"))
            with c2:
                b = _num_sel("條件 2 (例如 After)", df, "st:pair:b", exclude=(a,), hints=("after", "post"))
            with c3:
                alt = ui.selectbox("對立假設", ["two-sided", "greater", "less"], "st:pair:alt")
            st.caption("資料需為寬格式：每列一位受試者，兩個欄位分別是兩次測量。若資料是長格式，可先到「資料整理」做樞紐轉換或使用重複量測 ANOVA。")
            res = bs.paired_test(df, a, b, alt)
            quick = ("paired", {"a": a, "b": b}, {}, None)
        elif name.startswith("多組比較"):
            with c1:
                v = _num_sel("數值變項", df, "st:one:v")
            with c2:
                g = _cat_sel("分組", df, "st:one:g", exclude=(v,))
            with c3:
                ph = ui.selectbox("事後檢定", ["Tukey HSD", "Games-Howell", "Dunn (Holm)", "Dunn (Bonferroni)", "Holm t 檢定",
                                            "Bonferroni t 檢定"], "st:one:ph")
            order = plots.category_levels(df[g])
            d = df.assign(**{g: plots.as_category_labels(df[g])})
            res = bs.one_way(d, v, g, ph, order)
            test = {"Tukey HSD": "tukey", "Games-Howell": "gh", "Dunn (Holm)": "dunn", "Dunn (Bonferroni)": "dunn"}.get(ph, "welch")
            corr = {"Dunn (Bonferroni)": "bonferroni", "Bonferroni t 檢定": "bonferroni"}.get(ph, "holm")
            quick = ("box", {"x": g, "y": v}, {"points": "散點 (strip)"},
                     dict(enabled=True, test=test, pairs_mode="all", correction=corr, display="星號 (*)", hide_ns=True))
        elif name == "雙因子 ANOVA":
            with c1:
                v = _num_sel("數值變項", df, "st:two_way:v")
            with c2:
                f1 = _cat_sel("因子 A", df, "st:two_way:f1", exclude=(v,))
                f2 = _cat_sel("因子 B", df, "st:two_way:f2", exclude=(v, f1))
            with c3:
                inter = ui.checkbox("包含交互作用 (A × B)", "st:two_way:int", default=True)
                typ = ui.selectbox("平方和類型", [2, 3, 1], "st:two_way:typ", format_func=lambda t: f"Type {'I' * t}")
            if f1 == f2:
                st.warning("兩個因子不能相同。")
            else:
                res = bs.two_way(df, v, f1, f2, inter, typ)
                quick = ("point", {"x": f1, "y": v, "hue": f2}, {}, None)
        elif name.startswith("重複量測"):
            with c1:
                v = _num_sel("數值變項", df, "st:rm:v", hints=("response", "value", "score"))
            with c2:
                subj = _cat_sel("受試者 ID", df, "st:rm:s", hints=("subject", "id", "patient", "mouse", "受試"))
            with c3:
                w = _cat_sel("重複因子 (時間 / 條件)", df, "st:rm:w", exclude=(subj,), hints=("day", "time", "week", "visit", "condition"))
            res = bs.rm_anova(df, v, subj, w)
            quick = ("line", {"x": w, "y": v, "units": subj}, {}, None)
        elif name == "相關分析":
            with c1:
                cols = ui.multiselect("數值變項 (≥ 2)", numeric_columns(df), "st:corr:cols", default=numeric_columns(df)[:5])
            with c2:
                method = ui.selectbox("方法", ["pearson", "spearman", "kendall"], "st:corr:m",
                                      format_func={"pearson": "Pearson (線性)", "spearman": "Spearman (等級)", "kendall": "Kendall τ"}.get)
            if len(cols) >= 2:
                res = bs.correlation(df, cols, method)
                quick = ("corr", {"values": cols}, {"method": method}, None) if len(cols) > 2 else \
                    ("scatter", {"x": cols[0], "y": cols[1]}, {"fit": "線性", "show_r": True}, None)
        elif name == "線性迴歸":
            with c1:
                y = _num_sel("依變項 Y", df, "st:lr:y")
            with c2:
                xs = ui.multiselect("數值自變項", [c for c in numeric_columns(df) if c != y], "st:lr:xs",
                                    default=[c for c in numeric_columns(df) if c != y][:1])
            with c3:
                cxs = ui.multiselect("類別自變項", [c for c in categorical_columns(df) if c not in xs and c != y], "st:lr:cx")
            if xs or cxs:
                res = bs.linear_regression(df, y, xs, cxs)
                if len(xs) == 1:
                    quick = ("reg", {"x": xs[0], "y": y}, {}, None)
        elif name == "邏輯迴歸":
            with c1:
                y = _cat_sel("結果變項 (二元)", df, "st:logit:y", hints=("respond", "outcome", "disease", "event", "status"))
                lv = _levels(df, y)
                pos = ui.selectbox("「事件」類別", lv, f"st:logit:pos:{y}", default="1" if "1" in lv else (lv[-1] if lv else None))
            with c2:
                xs = ui.multiselect("數值預測變項", [c for c in numeric_columns(df) if c != y], "st:logit:xs")
            with c3:
                cxs = ui.multiselect("類別預測變項", [c for c in categorical_columns(df) if c != y and c not in xs], "st:logit:cx")
            if xs or cxs:
                d = df.assign(**{y: plots.as_category_labels(df[y])})
                res = bs.logistic_regression(d, y, pos, xs, cxs)
            else:
                st.info("請選擇至少一個預測變項。")
        elif name.startswith("卡方"):
            with c1:
                r = _cat_sel("列變項", df, "st:chi:r", hints=("treat", "group", "exposure"))
            with c2:
                c = _cat_sel("欄變項", df, "st:chi:c", exclude=(r,), hints=("outcome", "response", "status"))
            with c3:
                cnt = ui.selectbox("次數欄 (已彙整資料才需要)", [None] + numeric_columns(df), "st:chi:n",
                                   format_func=lambda x: "（每列一筆觀察）" if x is None else x)
                yates = ui.checkbox("2×2 表使用 Yates 連續校正", "st:chi:yates", default=True)
            if r == c:
                st.warning("請選擇兩個不同的變項。")
            else:
                d = df.assign(**{r: plots.as_category_labels(df[r]), c: plots.as_category_labels(df[c])})
                res = bs.contingency(d, r, c, cnt, yates)
                if cnt is None:
                    quick = ("stacked", {"x": r, "hue": c}, {"percent": True, "labels": True}, None)
        elif name.startswith("存活分析"):
            with c1:
                t = _num_sel("時間", df, "st:km:t", hints=("time", "month", "day", "os", "pfs"))
            with c2:
                e = ui.selectbox("事件 (1 = 發生)", list(df.columns), "st:km:e",
                                 default=guess_column(list(df.columns), ["event", "status", "death", "dead"]))
            with c3:
                g = _cat_sel("分組 (選填)", df, "st:km:g", optional=True, hints=("group", "risk", "treat", "arm"))
                lm = ui.text("特定時間點存活率 (逗號分隔)", "st:km:lm", default="12, 24, 36")
            marks = [float(x) for x in lm.replace("，", ",").split(",") if x.strip().replace(".", "", 1).isdigit()]
            d = df.assign(**{g: plots.as_category_labels(df[g])}) if g else df
            res = bs.survival_analysis(d, t, e, g, marks)
            quick = ("km", {"time": t, "event": e, "group": g}, {}, None)
        elif name.startswith("Cox"):
            with c1:
                t = _num_sel("時間", df, "st:cox:t", hints=("time", "month", "day", "os", "pfs"))
                e = ui.selectbox("事件 (1 = 發生)", list(df.columns), "st:cox:e",
                                 default=guess_column(list(df.columns), ["event", "status", "death", "dead"]))
            with c2:
                covs = ui.multiselect("數值共變數", [c for c in numeric_columns(df) if c not in (t, e)], "st:cox:covs")
            with c3:
                ccovs = ui.multiselect("類別共變數", [c for c in categorical_columns(df) if c not in (t, e) and c not in covs], "st:cox:ccovs")
                ties = ui.selectbox("同分處理", ["efron", "breslow"], "st:cox:ties")
            if covs or ccovs:
                d = df.copy()
                for cc in ccovs:
                    d[cc] = plots.as_category_labels(d[cc])
                res = bs.cox_regression(d, t, e, covs, ccovs, ties)
            else:
                st.info("請選擇至少一個共變數。")
        elif name.startswith("ROC"):
            with c1:
                truth = _cat_sel("真實狀態", df, "st:roc:y", hints=("disease", "label", "outcome", "status", "case"))
                lv = _levels(df, truth)
                pos = ui.selectbox("陽性類別", lv, f"st:roc:pos:{truth}", default="1" if "1" in lv else (lv[-1] if lv else None))
            with c2:
                scores = ui.multiselect("預測指標", [c for c in numeric_columns(df) if c != truth], "st:roc:s",
                                        default=[c for c in numeric_columns(df) if c != truth][:3])
            if scores:
                d = df.assign(**{truth: plots.as_category_labels(df[truth])})
                res = bs.roc_analysis(d, truth, pos, scores)
                quick = ("roc", {"truth": truth, "scores": scores}, {"positive": str(pos)}, None)
        elif name.startswith("劑量反應"):
            with c1:
                x = _num_sel("濃度 / 劑量", df, "st:dose:x", hints=("conc", "dose", "um", "nm"))
            with c2:
                y = _num_sel("反應值", df, "st:dose:y", exclude=(x,), hints=("resp", "inhib", "viab", "%"))
            with c3:
                g = _cat_sel("化合物 / 分組 (選填)", df, "st:dose:g", optional=True, hints=("compound", "drug", "group"))
            d = df.assign(**{g: plots.as_category_labels(df[g])}) if g else df
            res = bs.dose_response(d, x, y, g)
            quick = ("dose", {"x": x, "y": y, "hue": g}, {}, None)
        elif name.startswith("多重比較校正"):
            with c1:
                pc = _num_sel("p 值欄位", df, "st:padj:p", hints=("^p$", "pval", "p_value", "p.value", "pvalue"))
            with c2:
                methods = ui.multiselect("校正方法", ["bonferroni", "holm", "fdr_bh", "fdr_by", "sidak", "hommel"], "st:padj:m",
                                         default=["bonferroni", "holm", "fdr_bh"])
            if methods:
                res = bs.p_adjust_table(df, pc, methods)
        elif name.startswith("檢定力"):
            with c1:
                test = ui.selectbox("檢定類型", ["兩獨立樣本 t 檢定", "配對 / 單一樣本 t 檢定", "單因子 ANOVA"], "st:pw:test")
                solve = ui.selectbox("要計算", ["樣本數", "檢定力", "效果量"], "st:pw:solve")
            with c2:
                eff = ui.number("效果量 (Cohen's d 或 f)", "st:pw:eff", default=0.5, min_value=0.01, step=0.05, format="%g",
                                disabled=solve == "效果量")
                n = ui.number("樣本數 (t 檢定為每組；ANOVA 為總數)", "st:pw:n", default=20.0, min_value=2.0, step=1.0,
                              disabled=solve == "樣本數")
            with c3:
                alpha = ui.number("顯著水準 α", "st:pw:alpha", default=0.05, min_value=0.001, max_value=0.2, step=0.005, format="%g")
                power = ui.number("檢定力 (1 − β)", "st:pw:power", default=0.8, min_value=0.05, max_value=0.999, step=0.05,
                                  format="%g", disabled=solve == "檢定力")
                k = int(ui.number("組數 (ANOVA)", "st:pw:k", default=3, min_value=2, max_value=20, step=1)) if test == "單因子 ANOVA" else 3
            try:
                res = bs.power_analysis(test, solve, None if solve == "效果量" else eff, None if solve == "樣本數" else n, alpha,
                                        None if solve == "檢定力" else power, k_groups=k)
            except Exception as err:
                st.error(f"無法計算：{err}")

    if res is None and quick is None:
        return
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown(f"#### {res.title if res else name}")
        try:
            _show(res, "stats_" + name.split(" ")[0])
        except Exception as err:
            st.error(f"分析失敗：{err}")
    with right:
        if quick:
            chart, m, o, sig = quick
            _quick(chart, df, m, o, sig, orders=locals().get("orders"))
