"""中英文切換。

中文是原文，英文放在 locales/en.py 的對照表；查不到的字串保持中文，所以可以逐步翻譯。
語言存在 ContextVar：Streamlit 每個連線的腳本在自己的執行緒中執行，互不干擾。

    from .i18n import _
    _("資料來源")                      # → "Data source" (英文模式)
    _("共 {n} 筆資料", n=60)           # 含變數時用 {name} 樣板，不要用 f-string
"""

from __future__ import annotations

from contextvars import ContextVar

import pandas as pd

LANGS = {"zh": "中文", "en": "English"}
_lang: ContextVar[str] = ContextVar("lang", default="zh")


def set_lang(lang: str) -> None:
    _lang.set(lang if lang in LANGS else "zh")


def get_lang() -> str:
    return _lang.get()


def is_en() -> bool:
    return _lang.get() == "en"


def tr(lang: str, text: str, **kw) -> str:
    """以指定語言翻譯 (匯出報告時依報告產生當時的語言，而不是目前介面語言)。"""
    if lang == "en":
        from .locales.en import EN

        text = EN.get(text, text)
    return text.format(**kw) if kw else text


def _(text: str, **kw) -> str:
    return tr(_lang.get(), text, **kw)


def table(df: pd.DataFrame) -> pd.DataFrame:
    """表格欄名與文字儲存格 (如「顯著」、「符合常態」) 轉成目前語言；數值與使用者資料不變。"""
    if _lang.get() != "en":
        return df
    from .locales.en import EN

    out = df.rename(columns=lambda c: EN.get(c, c) if isinstance(c, str) else c)
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: EN.get(v, v) if isinstance(v, str) else v)
    return out
