"""每一種圖表都用範例資料實際繪製一次，並輸出成各種格式。"""

import matplotlib

matplotlib.use("Agg")

import pytest

from biostat import data, export, plots
from biostat.style import PRESETS, Style

DEMO = data.make_demo_data()

# 圖表 → (範例工作表, 欄位對應, 額外選項)
CASES = {
    "hist": ("Groups", {"x": "TumorVolume", "hue": "Group"}, {"kde": True}),
    "kde": ("Groups", {"x": "IL6", "hue": "Group"}, {}),
    "ecdf": ("Groups", {"x": "IL6", "hue": "Sex"}, {}),
    "qq": ("Groups", {"y": "TumorVolume", "hue": "Group"}, {}),
    "box": ("Groups", {"x": "Group", "y": "TumorVolume"}, {"points": "散點 (strip)"}),
    "violin": ("Groups", {"x": "Group", "y": "IL6", "hue": "Sex"}, {"split": True}),
    "bar": ("Groups", {"x": "Group", "y": "TumorVolume"}, {}),
    "point": ("Groups", {"x": "Group", "y": "IL6", "hue": "Sex"}, {}),
    "strip": ("Groups", {"x": "Group", "y": "TNFa"}, {"summary": "平均 ± SEM"}),
    "swarm": ("Groups", {"x": "Group", "y": "Ki67_pct", "hue": "Sex"}, {"summary": "中位數 ± IQR"}),
    "boxen": ("Groups", {"x": "Group", "y": "TumorVolume"}, {}),
    "raincloud": ("Groups", {"x": "Group", "y": "TumorVolume"}, {}),
    "paired": ("Paired", {"a": "Before", "b": "After"}, {"box": True}),
    "count": ("Clinical",{"x": "Outcome", "hue": "Treatment"}, {"stat": "percent"}),
    "scatter": ("Groups", {"x": "IL6", "y": "TNFa", "hue": "Group", "size": "Weight_g", "label": "SubjectID"},
                {"fit": "線性", "show_r": True}),
    "line": ("TimeCourse", {"x": "Day", "y": "Response", "hue": "Group"}, {}),
    "reg": ("Groups", {"x": "IL6", "y": "TumorVolume", "hue": "Sex"}, {}),
    "resid": ("Groups", {"x": "IL6", "y": "TumorVolume"}, {}),
    "hexbin": ("Volcano", {"x": "log2FoldChange", "y": "baseMean"}, {}),
    "kde2d": ("Groups", {"x": "IL6", "y": "TNFa", "hue": "Sex"}, {}),
    "corr": ("Groups", {"values": ["TumorVolume", "IL6", "TNFa", "Ki67_pct", "Weight_g"]}, {}),
    "heatmap": ("ExprMatrix", {"index": "Gene", "values": [f"Ctrl_{i}" for i in range(1, 6)] + [f"Trt_{i}" for i in range(1, 6)]},
                {"zscore": "列 (row)"}),
    "pivot_heatmap": ("Groups", {"row": "Group", "col": "Sex", "value": "TumorVolume"}, {}),
    "clustermap": ("ExprMatrix", {"index": "Gene", "values": [f"Ctrl_{i}" for i in range(1, 6)] + [f"Trt_{i}" for i in range(1, 6)],
                                  "row_group": "Pathway"}, {}),
    "pairplot": ("Groups", {"values": ["TumorVolume", "IL6", "TNFa"], "hue": "Group"}, {"panel": 1.5}),
    "jointplot": ("Groups", {"x": "IL6", "y": "TNFa", "hue": "Group"}, {}),
    "pca": ("Groups", {"values": ["TumorVolume", "IL6", "TNFa", "Ki67_pct", "Weight_g"], "hue": "Group"}, {"loadings": 4}),
    "volcano": ("Volcano", {"fc": "log2FoldChange", "p": "padj", "label": "Gene"}, {"highlight": "GENE0001"}),
    "km": ("Survival", {"time": "Time_months", "event": "Event", "group": "RiskGroup"}, {"median": True}),
    "roc": ("Diagnostic", {"truth": "Disease", "scores": ["MarkerA", "MarkerB", "MarkerC"]}, {}),
    "bland": ("MethodCompare", {"a": "MethodA", "b": "MethodB"}, {"loa_ci": True, "trend": True}),
    "forest": ("Forest", {"label": "Subgroup", "est": "HR", "lo": "Lower", "hi": "Upper", "weight": "N"}, {}),
    "dose": ("DoseResponse", {"x": "Concentration_uM", "y": "Inhibition_pct", "hue": "Compound"}, {}),
    "manhattan": ("GWAS", {"chr": "CHR", "pos": "BP", "p": "P", "label": "SNP"}, {}),
    "stacked": ("Microbiome", {"x": "Sample", "hue": "Taxon", "value": "Abundance"}, {"labels": True}),
    "pie": ("Clinical", {"x": "Outcome"}, {"donut": True}),
    "summary_bar": ("Groups", {"x": "Group", "mean": "TumorVolume", "err": "IL6"}, {}),
    "venn": ("GeneSets", {"sets": ["RNAseq_DEG", "Proteomics_DEP", "ChIP_targets"]}, {}),
    "radar": ("Groups", {"values": ["TumorVolume", "IL6", "TNFa", "Ki67_pct", "Weight_g"], "hue": "Group"}, {}),
}


def test_every_chart_has_a_case():
    assert set(CASES) == set(plots.CHARTS), set(plots.CHARTS) ^ set(CASES)


@pytest.mark.parametrize("key", sorted(CASES))
def test_render_chart(key):
    sheet, m, o = CASES[key]
    res = plots.render(key, DEMO[sheet], m, o, Style(title=f"Test {key}"))
    png, ext, _ = export.fig_bytes(res.fig, "PNG", dpi=60, rc=res.rc)
    assert ext == "png" and png[:4] == b"\x89PNG"
    export.close(res.fig)


@pytest.mark.parametrize("key", ["box", "bar", "violin", "strip", "swarm", "point", "boxen", "raincloud"])
def test_significance(key):
    sheet, m, o = CASES[key]
    sig = dict(enabled=True, test="welch", pairs_mode="all", correction="holm", display="星號 (*)", hide_ns=False)
    res = plots.render(key, DEMO[sheet], m, o, Style(), sig=sig)
    names = [n for n, _ in res.tables]
    assert "顯著性檢定結果" in names
    export.close(res.fig)


def test_facet_and_formats():
    res = plots.render("box", DEMO["Groups"], {"x": "Group", "y": "IL6"}, {}, Style(), facet="Sex",
                       sig=dict(enabled=True, test="mwu", pairs_mode="ref", ref="Control", correction="bonferroni",
                                display="p 值", hide_ns=True))
    assert len(res.axes) == 2
    for fmt in export.FORMATS:
        content, ext, _ = export.fig_bytes(res.fig, fmt, dpi=50, rc=res.rc)
        assert len(content) > 100, fmt
    export.close(res.fig)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_presets(preset):
    s = Style.from_dict({**Style().__dict__, **PRESETS[preset]})
    res = plots.render("bar", DEMO["Groups"], {"x": "Group", "y": "IL6", "hue": "Sex"}, {}, s)
    export.close(res.fig)


def test_style_options_roundtrip():
    s = Style(xlog=True, ymin=0, ymax=100, grid="兩者", legend_loc="圖外右側", xtick_rotation=45, transparent=True,
              palette="自訂 (輸入色碼)", custom_colors="#ff0000, 00ff00, blue")
    s2 = Style.from_dict(__import__("json").loads(s.to_json()))
    assert s2 == s
    res = plots.render("scatter", DEMO["Groups"], {"x": "IL6", "y": "TNFa", "hue": "Sex"}, {}, s2)
    export.close(res.fig)


def test_category_order_and_color_override():
    res = plots.render("box", DEMO["Groups"], {"x": "Group", "y": "IL6"}, {}, Style(),
                       orders={"Group": ["DrugC", "Control"]}, overrides={"Group": {"Control": "#000000"}})
    labels = [t.get_text() for t in res.axes[0].get_xticklabels()]
    assert labels == ["DrugC", "Control"]
    export.close(res.fig)


def test_chinese_labels():
    df = DEMO["Groups"].rename(columns={"Group": "組別", "IL6": "介白素6"})
    res = plots.render("bar", df, {"x": "組別", "y": "介白素6"}, {}, Style(title="中文標題測試"))
    export.fig_bytes(res.fig, "PDF (向量)", rc=res.rc)
    export.close(res.fig)
