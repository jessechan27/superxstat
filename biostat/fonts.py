"""字型管理：找出系統可用字型，並確保中文 (CJK) 字型可用。

matplotlib 3.6 起支援字型回退 (font fallback)：rcParams["font.family"] 設成清單時，
主字型缺字會自動改用清單中的下一個字型，所以「英文用 Arial、中文用微軟正黑體」可以同時成立。
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache

from matplotlib import font_manager as fm

from .runtime import IS_PYODIDE

CJK_CANDIDATES = [
    "Microsoft JhengHei",  # Windows 繁中
    "Microsoft YaHei",  # Windows 簡中
    "PingFang TC",  # macOS
    "PingFang SC",
    "Heiti TC",
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans TC",
    "WenQuanYi Micro Hei",  # Linux (Streamlit Cloud 透過 packages.txt 安裝)
    "WenQuanYi Zen Hei",
    "SimHei",
    "AR PL UMing TW",
    "Arial Unicode MS",
]

# 瀏覽器版沒有中文字型，需要時從 jsDelivr 下載 Google Fonts 的 Noto Sans TC (約 12 MB)。
NOTO_SANS_TC_URL = "https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/notosanstc/NotoSansTC%5Bwght%5D.ttf"

COMMON_FONTS = [
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "DejaVu Sans",
    "Calibri",
    "Segoe UI",
    "Verdana",
    "Times New Roman",
    "Liberation Serif",
    "DejaVu Serif",
    "Georgia",
    "Cambria",
    "Courier New",
    "DejaVu Sans Mono",
]

_downloaded_font: str | None = None


def _names() -> set[str]:
    return {f.name for f in fm.fontManager.ttflist}


@lru_cache(maxsize=1)
def available_fonts() -> list[str]:
    """UI 用的字型清單：常用字型排前面，其餘依字母排序。"""
    names = _names()
    head = [f for f in COMMON_FONTS if f in names]
    head += [f for f in CJK_CANDIDATES if f in names and f not in head]
    rest = sorted(n for n in names if n not in head and not n.startswith("."))
    return head + rest


def default_font() -> str:
    names = _names()
    for f in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if f in names:
            return f
    return "DejaVu Sans"


def find_cjk_font() -> str | None:
    if _downloaded_font:
        return _downloaded_font
    names = _names()
    for name in CJK_CANDIDATES:
        if name in names:
            return name
    return None


def _fetch_bytes_pyodide(url: str) -> bytes:
    # stlite 在 Web Worker 中執行 Python，Worker 允許同步 XHR 並取得二進位內容。
    from js import Uint8Array, XMLHttpRequest  # type: ignore[import-not-found]

    req = XMLHttpRequest.new()
    req.open("GET", url, False)
    req.responseType = "arraybuffer"
    req.send(None)
    if req.status != 200:
        raise OSError(f"HTTP {req.status}")
    return Uint8Array.new(req.response).to_bytes()


def ensure_cjk_font(download: bool = True) -> str | None:
    """回傳可用的中文字型名稱；瀏覽器版必要時下載 Noto Sans TC。失敗則回傳 None。"""
    global _downloaded_font
    name = find_cjk_font()
    if name or not (IS_PYODIDE and download):
        return name
    try:
        data = _fetch_bytes_pyodide(NOTO_SANS_TC_URL)
        path = os.path.join(tempfile.gettempdir(), "NotoSansTC.ttf")
        with open(path, "wb") as fh:
            fh.write(data)
        fm.fontManager.addfont(path)
        _downloaded_font = fm.FontProperties(fname=path).get_name()
        available_fonts.cache_clear()
        return _downloaded_font
    except Exception:
        return None


def font_family(primary: str, cjk: str | None) -> list[str]:
    """組成 rcParams['font.family'] 清單：主字型 → 中文字型 → DejaVu Sans (保底)。"""
    names = _names()
    family: list[str] = []
    for f in (primary, cjk, "DejaVu Sans"):
        if f and f in names and f not in family:
            family.append(f)
    return family or ["DejaVu Sans"]


def has_cjk(text: str) -> bool:
    return any("⺀" <= ch <= "鿿" or "豈" <= ch <= "﫿" or "＀" <= ch <= "￯" for ch in text)
