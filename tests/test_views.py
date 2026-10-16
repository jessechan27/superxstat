"""以 Streamlit AppTest 實際執行每一頁，確認沒有例外。"""

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["page_data", "page_plot", "page_stats", "page_report", "page_code"]


def _script(page: str) -> str:
    return f"""
import matplotlib
matplotlib.use("Agg")
from biostat.views import common as ui, {page}
ui.keep_state()
ui.inject_css()
ds = ui.sidebar_data()
ui.sidebar_footer()
{page}.render(ds)
"""


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_string(_script(page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_report_page_generates_report():
    at = AppTest.from_string(_script("page_report"), default_timeout=180).run()
    at.button(key="btn:report").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("重點摘要" in m.value for m in at.markdown)


def test_data_page_add_and_undo_step():
    at = AppTest.from_string(_script("page_data"), default_timeout=120).run()
    at.selectbox(key="dt:kind").set_value("移除缺失值").run()
    at.button(key="btn:add_step").click().run()
    assert not at.exception
    store = at.session_state["ds_store"]["demo:Groups"]
    assert [t["type"] for t in store["transforms"]] == ["dropna"]
    at.button(key="btn:undo").click().run()
    assert at.session_state["ds_store"]["demo:Groups"]["transforms"] == []


def test_code_page_runs_template():
    at = AppTest.from_string(_script("page_code"), default_timeout=120).run()
    at.button(key="btn:run_code").click().run()
    assert not at.exception
    res = at.session_state["code_res"]
    assert res["error"] is None and len(res["figs"]) == 1
