"""自訂程式碼的安全預設：只有本機連線才開放。"""

import pytest

from biostat import runtime


@pytest.mark.parametrize("host,expected", [
    ("localhost:8501", True), ("127.0.0.1:8502", True), ("[::1]:8501", True),
    ("super-xtat.streamlit.app", False), ("192.168.1.20:8501", False),
])
def test_code_enabled_only_for_local_hosts(monkeypatch, host, expected):
    monkeypatch.delenv("BIOSTAT_ENABLE_CODE", raising=False)
    monkeypatch.delenv("BIOSTAT_DISABLE_CODE", raising=False)
    monkeypatch.setattr(runtime, "_request_host", lambda: host.rsplit(":", 1)[0].lower())
    assert runtime.code_execution_enabled() is expected


def test_env_overrides(monkeypatch):
    monkeypatch.setattr(runtime, "_request_host", lambda: "example.com")
    monkeypatch.setenv("BIOSTAT_ENABLE_CODE", "1")
    assert runtime.code_execution_enabled()
    monkeypatch.setattr(runtime, "_request_host", lambda: "localhost")
    monkeypatch.setenv("BIOSTAT_DISABLE_CODE", "1")
    assert not runtime.code_execution_enabled()
