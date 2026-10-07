"""Audit 2026-09-23: /chat failed for every visitor, and the limiter it relied on
counted Caddy's address instead of the visitor's."""

import importlib

import pytest

import app as app_module


@pytest.fixture(autouse=True)
def _reset_limiter(monkeypatch):
    monkeypatch.setattr(app_module, "_CHAT_LIMITER", None)
    monkeypatch.delenv("CHAT_TOKEN", raising=False)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "pipe", lambda q: {"answer": "ok", "sources": []})
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


# --- input validation: these used to reach Chroma/OpenRouter and 500 ---
@pytest.mark.parametrize("body", [{"query": 123}, {"query": ["a"]}, {"query": "   "}, ["query"], "query"])
def test_bad_query_is_400_not_500(client, body):
    assert client.post("/chat", json=body).status_code == 400


def test_overlong_query_is_400(client):
    assert client.post("/chat", json={"query": "x" * 2001}).status_code == 400
    assert client.post("/chat", json={"query": "x" * 2000}).status_code == 200


def test_oversized_body_is_refused(client):
    r = client.post("/chat", data="x" * (70 * 1024), content_type="application/json")
    assert r.status_code == 413


def test_upstream_failure_is_friendly_503(client, monkeypatch):
    def boom(q):
        raise RuntimeError("openrouter 429")

    monkeypatch.setattr(app_module, "pipe", boom)
    r = client.post("/chat", json={"query": "hello"})
    assert r.status_code == 503
    assert "openrouter" not in r.get_data(as_text=True)


# --- the limiter must key on the visitor, not on the proxy ---
def test_limit_is_per_forwarded_client(client, monkeypatch):
    monkeypatch.setenv("CHAT_MAX_REQUESTS", "2")
    codes = [
        client.post(
            "/chat", json={"query": "hi"}, headers={"X-Forwarded-For": f"203.0.113.{i}"}
        ).status_code
        for i in range(6)
    ]
    assert codes == [200] * 6  # six visitors, one request each


def test_one_forwarded_client_is_still_limited(client, monkeypatch):
    monkeypatch.setenv("CHAT_MAX_REQUESTS", "2")
    h = {"X-Forwarded-For": "203.0.113.9"}
    codes = [client.post("/chat", json={"query": "hi"}, headers=h).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


# --- model list: free slugs only, from the environment ---
def _reload_query(monkeypatch, **env):
    for k in ("CHAT_MODEL", "CHAT_FALLBACK_MODELS", "ALLOW_PAID_MODELS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import query

    return importlib.reload(query)


def test_paid_slugs_are_dropped(monkeypatch):
    q = _reload_query(monkeypatch, CHAT_FALLBACK_MODELS="a/b:free, paid/model ,c/d:free")
    assert q._FALLBACK_MODELS == ["a/b:free", "c/d:free"]


def test_paid_primary_is_not_used(monkeypatch):
    q = _reload_query(monkeypatch, CHAT_MODEL="paid/model", CHAT_FALLBACK_MODELS="a/b:free")
    assert not q._usable(q.CHAT_MODEL)


def test_paid_allowed_only_on_purpose(monkeypatch):
    q = _reload_query(monkeypatch, CHAT_FALLBACK_MODELS="paid/model", ALLOW_PAID_MODELS="1")
    assert q._FALLBACK_MODELS == ["paid/model"]


def test_default_fallbacks_are_free(monkeypatch):
    q = _reload_query(monkeypatch)
    assert q._FALLBACK_MODELS and all(m.endswith(":free") for m in q._FALLBACK_MODELS)


def test_call_llm_tries_models_in_order_and_skips_paid(monkeypatch):
    q = _reload_query(
        monkeypatch, CHAT_MODEL="paid/primary", CHAT_FALLBACK_MODELS="a/b:free,c/d:free"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    tried = []

    class FakeCompletions:
        def create(self, model, **kw):
            tried.append(model)
            if model == "a/b:free":
                raise RuntimeError("429")

            class R:
                choices = [type("C", (), {"message": type("M", (), {"content": " hi "})()})()]

            return R()

    class FakeClient:
        def __init__(self, **kw):
            assert kw["timeout"] <= 60
            self.chat = type("X", (), {"completions": FakeCompletions()})()

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    assert q.call_llm("p") == "hi"
    assert tried == ["a/b:free", "c/d:free"]


def test_no_usable_model_raises_cleanly(monkeypatch):
    q = _reload_query(monkeypatch, CHAT_MODEL="paid/x", CHAT_FALLBACK_MODELS="paid/y")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    with pytest.raises(RuntimeError, match="No usable chat model"):
        q.call_llm("p")
