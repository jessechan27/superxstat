"""繪圖頁英文化：每種圖表在英文模式下，圖上文字、備註與附帶數據表都不得殘留中文；所有選項都要有英文。"""

import re

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.text import Text

from biostat import annotate, code_runner, export, i18n, palettes, plots, style
from biostat.locales.en import EN
from biostat.style import Style

from test_plots import CASES, DEMO

CJK = re.compile(r"[一-鿿]")


def test_every_plot_option_has_english():
    keys = list(plots.CATEGORIES)
    for spec in plots.CHARTS.values():
        keys += [spec.name, spec.description]
        keys += [s for f in spec.fields for s in (f.label, f.help)]
        keys += [s for o in spec.opts for s in [o.label, o.help] + [c for c in (o.choices or []) if isinstance(c, str)]]
    keys += list(style.THEMES.values()) + list(style.CONTEXTS.values()) + list(style.GRID_OPTIONS) + list(style.EDGE_OPTIONS)
    keys += list(style.LEGEND_LOCS) + list(style.PRESETS) + list(palettes.PALETTE_NAMES) + list(palettes.COLORMAPS)
    keys += list(annotate.TESTS) + list(annotate.CORRECTIONS) + list(annotate.DISPLAYS) + list(export.FORMATS)
    keys += list(code_runner.TEMPLATES)
    assert sorted({k for k in keys if k and CJK.search(k) and k not in EN}) == []


def test_english_templates_have_no_chinese():
    i18n.set_lang("en")
    try:
        leaks = {name: [ln for ln in code_runner.template_code(name).splitlines() if CJK.search(ln)] for name in code_runner.TEMPLATES}
    finally:
        i18n.set_lang("zh")
    assert {k: v for k, v in leaks.items() if v} == {}


@pytest.mark.parametrize("key", list(CASES))
def test_chart_english(key):
    sheet, m, o = CASES[key]
    sig = dict(enabled=True, test="welch", pairs_mode="all", correction="holm", display="星號 (*)", hide_ns=False) \
        if plots.CHARTS[key].sig else None
    i18n.set_lang("en")
    try:
        res = plots.render(key, DEMO[sheet], m, o, Style(), sig=sig)
        res.fig.canvas.draw()
        texts = [t.get_text() for t in res.fig.findobj(Text)] + list(res.notes)
        for name, tab in res.tables:
            tab = i18n.table(tab)
            texts += [i18n._(name)] + [str(c) for c in tab.columns]
            texts += [v for c in tab.columns if tab[c].dtype == object for v in tab[c] if isinstance(v, str)]
    finally:
        i18n.set_lang("zh")
        plots.plt.close("all")
    assert sorted({t for t in texts if CJK.search(t)}) == []
