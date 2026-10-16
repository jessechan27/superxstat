# Super Xtat

上傳 Excel / CSV，完成生物統計分析、自動產生統計報告，並用 seaborn 繪製出版級圖表。

## 功能

| 頁面 | 內容 |
|---|---|
| 資料 | 預覽、欄位摘要、寬轉長、篩選、新增欄位、重新編碼、手動編輯、下載整理後資料 |
| 繪圖 | 39 種 seaborn / matplotlib 圖表、顯著性標記、完整格式設定，下載 PNG / TIFF / SVG / PDF / EPS |
| 統計 | 描述統計、常態性、t 檢定、ANOVA (Tukey / Games-Howell / Dunn)、雙因子與重複量測 ANOVA、相關、線性 / 邏輯迴歸、卡方 / Fisher、Kaplan-Meier、Cox、ROC、4PL 劑量反應、檢定力 |
| 報告 | 依常態性與變異數同質性自動挑選檢定，產生含 seaborn 圖、結果解讀與統計方法的完整報告；可下載 HTML、Word、PPT (4 種風格)、Excel |
| 程式碼 | 用 Python 自由組合 matplotlib / seaborn (僅限本機執行時) |

## 本機執行

```bash
pip install -r requirements.txt
streamlit run app.py --server.address localhost
```

Windows 可直接雙擊 `run.bat`。

## 發佈到 Streamlit Community Cloud

1. 把本專案推送到 GitHub。
2. 到 <https://share.streamlit.io> 用 GitHub 帳號登入 → **Create app** → 選擇這個 repo、分支 `main`、主程式 `app.py`。
3. **Advanced settings** 選 Python 3.12，按 **Deploy**。

`packages.txt` 會安裝 Noto CJK 字型，讓圖表中的中文正常顯示。

### 安全性

「程式碼」頁會在伺服器上執行使用者輸入的 Python，因此**公開網址預設停用**，只有從 `localhost` 連線時才開放。
若確定要在伺服器上開放，可設定環境變數 `BIOSTAT_ENABLE_CODE=1`；強制關閉則設定 `BIOSTAT_DISABLE_CODE=1`。

## 測試

```bash
pip install -r requirements-dev.txt
pytest
```

統計函式皆與 scipy、statsmodels、lifelines、scikit-learn 等獨立實作逐一比對 (`tests/test_stats.py`)。
