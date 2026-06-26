import os

from quantaura import net


def test_apply_proxy_sets_env(monkeypatch):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(k, raising=False)
    out = net.apply_proxy("socks5://127.0.0.1:10808")
    assert out == "socks5://127.0.0.1:10808"
    assert os.environ["HTTPS_PROXY"] == "socks5://127.0.0.1:10808"
    assert os.environ["http_proxy"] == "socks5://127.0.0.1:10808"


def test_apply_proxy_empty_is_noop(monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    assert net.apply_proxy("") == ""
    assert net.apply_proxy("   ") == ""
    assert "HTTPS_PROXY" not in os.environ


def test_proxy_url_from_settings(monkeypatch):
    from quantaura.config import Settings
    monkeypatch.setenv("PROXY_URL", "http://127.0.0.1:10809")
    assert Settings.load().proxy_url == "http://127.0.0.1:10809"
