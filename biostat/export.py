"""圖檔匯出：PNG / TIFF / JPG / SVG / PDF / EPS。"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt

from .plots import PLOT_LOCK

FORMATS = {
    "PNG": ("png", "image/png"),
    "TIFF (期刊投稿)": ("tiff", "image/tiff"),
    "JPG": ("jpg", "image/jpeg"),
    "SVG (向量，可編輯)": ("svg", "image/svg+xml"),
    "PDF (向量)": ("pdf", "application/pdf"),
    "EPS (向量)": ("eps", "application/postscript"),
}


def fig_bytes(fig, fmt: str = "PNG", dpi: int = 300, rc: dict | None = None, tight: bool = True,
              transparent: bool = False) -> tuple[bytes, str, str]:
    """回傳 (檔案內容, 副檔名, MIME)。rc 需與繪圖時相同，字型嵌入 (pdf/svg fonttype) 才會一致。"""
    ext, mime = FORMATS.get(fmt, FORMATS["PNG"])
    kw: dict = dict(format=ext, dpi=dpi, bbox_inches="tight" if tight else None, pad_inches=0.08)
    if ext == "tiff":
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    if ext == "jpg":
        kw["pil_kwargs"] = {"quality": 95}
    elif ext != "eps":
        kw["transparent"] = transparent
    buf = io.BytesIO()
    with PLOT_LOCK, plt.rc_context(rc or {}):
        fig.savefig(buf, **kw)
    return buf.getvalue(), ext, mime


def close(fig) -> None:
    with PLOT_LOCK:
        plt.close(fig)
