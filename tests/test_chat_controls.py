"""/chat spends money on every call and is reachable without credentials.

Two controls, both written so that setting nothing preserves today's public-demo
behaviour:

  * CHAT_TOKEN gating — inactive unless the variable is set.
  * Rate limiting — always on, with limits generous enough that a demo visitor
    does not meet them.
"""

import pytest

import app as app_module
from ratelimit import RateLimiter


@pytest.fixture(autouse=True)
def _reset_limiter(monkeypatch):
    # The limiter is process-global and built lazily; without this the first
    # test's counts leak into the rest.
    monkeypatch.setattr(app_module, "_CHAT_LIMITER", None)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "pipe", lambda q: {"answer": "ok", "sources": []})
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _ask(client, **kw):
    return client.post("/chat", json={"query": "hello"}, **kw)


# --- token gating is opt-in ---
def test_chat_is_open_when_no_token_configured(client, monkeypatch):
    monkeypatch.delenv("CHAT_TOKEN", raising=False)
    assert _ask(client).status_code == 200


def test_chat_requires_token_when_configured(client, monkeypatch):
    monkeypatch.setenv("CHAT_TOKEN", "s3cret")
    assert _ask(client).status_code == 401


def test_chat_accepts_the_right_token(client, monkeypatch):
    monkeypatch.setenv("CHAT_TOKEN", "s3cret")
    assert _ask(client, headers={"X-Chat-Token": "s3cret"}).status_code == 200


def test_chat_token_non_ascii_does_not_500(client, monkeypatch):
    """The portfolio-wide compare_digest defect, guarded here too."""
    monkeypatch.setenv("CHAT_TOKEN", "s3cret")
    r = _ask(client, headers={"X-Chat-Token": "ü"})
    assert r.status_code != 500
    assert r.status_code == 401


# --- rate limiting is always on ---
def test_chat_is_rate_limited(client, monkeypatch):
    monkeypatch.delenv("CHAT_TOKEN", raising=False)
    monkeypatch.setenv("CHAT_MAX_REQUESTS", "3")
    monkeypatch.setenv("CHAT_WINDOW_SECONDS", "3600")
    codes = [_ask(client).status_code for _ in range(5)]
    assert codes.count(200) == 3
    assert codes[-1] == 429


def test_blank_env_value_does_not_crash(monkeypatch):
    monkeypatch.setenv("CHAT_MAX_REQUESTS", "")
    assert app_module._env_int("CHAT_MAX_REQUESTS", 30) == 30


# --- limiter unit behaviour ---
def test_limiter_allows_up_to_the_limit_then_refuses():
    lim = RateLimiter(2, 100)
    assert lim.allow("k", now=0.0) is True
    assert lim.allow("k", now=1.0) is True
    assert lim.allow("k", now=2.0) is False


def test_limiter_window_decays():
    lim = RateLimiter(1, 10)
    assert lim.allow("k", now=0.0) is True
    assert lim.allow("k", now=5.0) is False
    assert lim.allow("k", now=11.0) is True


def test_limiter_sweeps_keys_it_is_not_asked_about():
    lim = RateLimiter(5, 10)
    for i in range(50):
        lim.allow(f"src-{i}", now=100.0)
    assert lim.size() == 50
    lim.allow("someone-else", now=10_000.0)
    assert lim.size() == 1  # only the caller that just arrived


def test_separate_addresses_do_not_share_a_bucket():
    lim = RateLimiter(1, 100)
    assert lim.allow("a", now=0.0) is True
    assert lim.allow("b", now=0.0) is True
    assert lim.allow("a", now=1.0) is False
