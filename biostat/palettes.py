"""類別色盤與連續色階。"""

from __future__ import annotations

import re

import seaborn as sns

# 自訂色盤 (hex)。期刊風格色盤數值取自 R 套件 ggsci 的公開定義。
CUSTOM_PALETTES: dict[str, list[str]] = {
    "預設 (色盲友善)": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "Okabe-Ito (色盲友善)": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"],
    "NPG (Nature 風格)": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"],
    "AAAS (Science 風格)": ["#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#BB0021", "#5F559B", "#A20056", "#808180", "#1B1919"],
    "NEJM 風格": ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD", "#FFDC91", "#EE4C97"],
    "Lancet 風格": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6", "#1B1919"],
    "JAMA 風格": ["#374E55", "#DF8F44", "#00A1D5", "#B24745", "#79AF97", "#6A6599", "#80796B"],
    "JCO 風格": ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67", "#8F7700", "#3B3B3B", "#A73030", "#4A6990"],
    "GraphPad 風格": ["#000000", "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628", "#F781BF"],
    "灰階 (黑白印刷)": ["#252525", "#636363", "#969696", "#bdbdbd", "#d9d9d9", "#f0f0f0"],
}

SEABORN_PALETTES = [
    "deep", "muted", "pastel", "bright", "dark", "colorblind",
    "Set1", "Set2", "Set3", "Paired", "Dark2", "Accent", "tab10", "tab20",
    "husl", "hls", "rocket", "mako", "flare", "crest", "viridis", "Spectral", "coolwarm",
]

CUSTOM_LABEL = "自訂 (輸入色碼)"

PALETTE_NAMES = list(CUSTOM_PALETTES) + SEABORN_PALETTES + [CUSTOM_LABEL]

COLORMAPS = [
    "viridis", "magma", "plasma", "inferno", "cividis", "rocket", "mako", "flare", "crest",
    "RdBu_r", "coolwarm", "vlag", "icefire", "bwr", "seismic", "PiYG", "PRGn", "BrBG", "PuOr", "RdYlBu_r", "RdYlGn", "Spectral_r",
    "Blues", "Reds", "Greens", "Purples", "Oranges", "Greys", "YlGnBu", "YlOrRd", "BuPu", "GnBu",
    "turbo", "jet",
]

DIVERGING_CMAPS = {"RdBu_r", "coolwarm", "vlag", "icefire", "bwr", "seismic", "PiYG", "PRGn", "BrBG", "PuOr", "RdYlBu_r", "RdYlGn", "Spectral_r"}

_HEX = re.compile(r"#?[0-9a-fA-F]{6}\b|#?[0-9a-fA-F]{3}\b")


def parse_colors(text: str) -> list[str]:
    """解析使用者輸入的色碼或顏色名稱 (以逗號、空白或換行分隔)。"""
    out: list[str] = []
    for tok in re.split(r"[,\s;]+", text or ""):
        tok = tok.strip()
        if not tok:
            continue
        if _HEX.fullmatch(tok):
            out.append(tok if tok.startswith("#") else "#" + tok)
        else:
            try:
                from matplotlib.colors import to_hex

                out.append(to_hex(tok))
            except ValueError:
                continue
    return out


def get_palette(name: str, n: int, custom: str = "") -> list:
    """取得 n 個顏色。顏色不足時以 husl 補足額外色相 (不重複循環)。"""
    n = max(int(n), 1)
    if name == CUSTOM_LABEL:
        base = parse_colors(custom) or CUSTOM_PALETTES["預設 (色盲友善)"]
    elif name in CUSTOM_PALETTES:
        base = CUSTOM_PALETTES[name]
    else:
        try:
            return [tuple(c) for c in sns.color_palette(name, n)]
        except (ValueError, TypeError):
            base = CUSTOM_PALETTES["預設 (色盲友善)"]
    if n <= len(base):
        return list(base[:n])
    extra = sns.color_palette("husl", n - len(base))
    return list(base) + [tuple(c) for c in extra]
