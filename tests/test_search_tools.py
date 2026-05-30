import httpx
from mariana.tools.search import raw_scrape, raw_search


class DummyResponse:
    def __init__(self, status_code=500, text="", data=None):
        self.status_code = status_code
        self.text = text
        self._data = data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP error")

    def json(self):
        return self._data


class DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, *args, **kwargs):
        return DummyResponse(status_code=500, data={})


def test_raw_search_returns_empty_when_searxng_returns_500(monkeypatch):
    monkeypatch.setattr("mariana.tools.search.httpx.Client", DummyClient)
    assert raw_search("test", 9999) == []


def test_raw_scrape_example_returns_text(monkeypatch):
    long_text = "Hello world. " * 20  # >200 chars so readability tier accepts it
    html = f"<html><body><article><p>{long_text}</p></article></body></html>"

    def fake_get(url, headers=None, timeout=None, follow_redirects=None):
        return DummyResponse(status_code=200, text=html)

    # trafilatura.fetch_url is tried first; return None to fall through to readability tier
    monkeypatch.setattr("mariana.tools.search.trafilatura.fetch_url", lambda url: None)
    monkeypatch.setattr("mariana.tools.search.httpx.get", fake_get)
    # Bypass scrape cache
    monkeypatch.setattr("mariana.tools.search._cache_get", lambda url: None)
    monkeypatch.setattr("mariana.tools.search._cache_set", lambda url, content: None)
    result = raw_scrape("https://example.com", max_chars=4000)
    assert "Hello world" in result


def test_raw_scrape_youtube_returns_empty_without_http_call(monkeypatch):
    def fake_get(*args, **kwargs):
        raise AssertionError("HTTP request should not be made for blocked domains")

    monkeypatch.setattr("mariana.tools.search.httpx.get", fake_get)
    assert raw_scrape("https://www.youtube.com/watch?v=dQw4w9WgXcQ", max_chars=4000) == ""
