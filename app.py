"""Super Xtat 入口：streamlit run app.py"""

from pathlib import Path

import streamlit as st

from biostat.views import common as ui
from biostat.views import page_code, page_data, page_plot, page_report, page_stats

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="Super Xtat", page_icon=str(ASSETS / "icon.svg"), layout="wide",
                   initial_sidebar_state="expanded")
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")

ui.keep_state()
ui.inject_css()
ds = ui.sidebar_data()
ui.sidebar_footer()

# (render, 導覽名稱, 網址, 圖示, 頁首小標, 頁首標題, 說明)
PAGES = [
    (page_data.render, "資料", "data", ":material/table_chart:", "01 / Data", "資料整理",
     "預覽、欄位摘要、寬轉長、篩選、新增欄位與手動編輯；所有頁面都使用整理後的資料"),
    (page_plot.render, "繪圖", "plot", ":material/insights:", "02 / Plot", "seaborn 繪圖",
     "39 種出版級圖表，即時預覽，下載 PNG / TIFF / SVG / PDF / EPS"),
    (page_stats.render, "統計", "stats", ":material/functions:", "03 / Statistics", "統計分析",
     "t 檢定、ANOVA、相關與迴歸、卡方、存活分析、ROC、劑量反應、檢定力"),
    (page_report.render, "報告", "report", ":material/description:", "04 / Report", "分析報告",
     "自動挑選檢定，產生含 seaborn 圖表、結果解讀與統計方法的完整報告"),
    (page_code.render, "程式碼", "code", ":material/code:", "05 / Code", "自訂程式碼",
     "用 Python 自由組合 matplotlib / seaborn，畫出任何圖"),
]


def _page(render, kicker, title, sub):
    def page():
        ui.page_header(kicker, title, sub)
        render(ds)

    return page


nav = st.navigation([st.Page(_page(fn, kicker, title, sub), title=label, icon=icon, url_path=path, default=path == "data")
                     for fn, label, path, icon, kicker, title, sub in PAGES], position="top")
nav.run()
