"""執行環境偵測：本機 / 伺服器 (CPython) 或瀏覽器 (Pyodide / stlite)。"""

from __future__ import annotations

import os
import sys

IS_PYODIDE = sys.platform == "emscripten" or "pyodide" in sys.modules
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _request_host() -> str | None:
    """目前連線的主機名稱 (不含埠號)；不在 Streamlit 連線中時回傳 None。"""
    try:
        import streamlit as st

        host = st.context.headers.get("Host")
    except Exception:
        return None
    return host.rsplit(":", 1)[0].lower() if host else None


def code_execution_enabled() -> bool:
    """自訂程式碼功能是否啟用。

    - 瀏覽器版 (Pyodide)：程式碼在訪客自己的瀏覽器執行，永遠可用。
    - 伺服器版：程式碼在伺服器上執行，公開網站預設停用，只有從本機 (localhost) 連線時才開放。
      環境變數 BIOSTAT_ENABLE_CODE=1 可強制開啟、BIOSTAT_DISABLE_CODE=1 可強制關閉。
    """
    if IS_PYODIDE:
        return True
    if _flag("BIOSTAT_DISABLE_CODE"):
        return False
    if _flag("BIOSTAT_ENABLE_CODE"):
        return True
    host = _request_host()
    return host is None or host in LOCAL_HOSTS


def runtime_label() -> str:
    return "瀏覽器版 (Pyodide)" if IS_PYODIDE else f"Python {sys.version.split()[0]}"
