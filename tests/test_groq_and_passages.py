"""2026-10-18: OpenRouter's free models stop at 50 requests a day, so the demo answered
"busy" for most of the day. Groq's free tier is now tried first, and when every model
is busy the visitor gets the matching passages instead of an error."""

import sys
import types

import pytest

import query

CHUNKS = [
    {
        "text": "Refunds are given within 7 days with a receipt.",
        "source": "policies.txt",
        "distance": 0.1,
    },
    {"text": "Custom cakes need 72 hours notice.", "source": "orders.txt", "distance": 0.2},
]


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    monkeypatch.setattr(query, "GROQ_MODELS", ["groq-a", "groq-b"])
    monkeypatch.setattr(query, "CHAT_MODEL", "or-primary:free")
    monkeypatch.setattr(query, "_FALLBACK_MODELS", ["or-second:free"])


def fake_openai(monkeypatch, answers):
    """Install a stand-in OpenAI client; `answers` maps model -> text, or an Exception to raise."""
    calls = []

    class Completions:
        def __init__(self, base_url):
            self.base_url = base_url

        def create(self, model, **kw):
            calls.append((self.base_url, model))
            out = answers.get(model, RuntimeError("busy"))
            if isinstance(out, Exception):
                raise out
            msg = types.SimpleNamespace(content=out)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class OpenAI:
        def __init__(self, api_key, base_url, **kw):
            self.chat = types.SimpleNamespace(completions=Completions(base_url))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    return calls


def test_groq_is_tried_before_openrouter(keys):
    urls = [p[0] for p in query._providers()]
    assert urls == ["https://api.groq.com/openai/v1", "https://openrouter.ai/api/v1"]


def test_openrouter_still_never_gets_a_paid_model(keys, monkeypatch):
    monkeypatch.setattr(query, "CHAT_MODEL", "some/paid-model")
    assert query._providers()[1][2] == ["or-second:free"]


def test_without_a_groq_key_only_openrouter_is_used(keys, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    assert [p[0] for p in query._providers()] == ["https://openrouter.ai/api/v1"]


def test_falls_through_every_groq_model_then_openrouter(keys, monkeypatch):
    calls = fake_openai(monkeypatch, {"or-primary:free": "from openrouter"})
    assert query.call_llm("q") == "from openrouter"
    assert [m for _, m in calls] == ["groq-a", "groq-b", "or-primary:free"]


def test_empty_reply_counts_as_a_miss(keys, monkeypatch):
    fake_openai(monkeypatch, {"groq-a": "", "groq-b": "real answer"})
    assert query.call_llm("q") == "real answer"


def test_all_busy_gives_the_passages_not_an_error(keys, monkeypatch):
    fake_openai(monkeypatch, {})
    out = query.answer("refund?", CHUNKS)
    assert out["fallback"] is True
    assert out["answer"].startswith(query.BUSY_PREFIX)
    for c in CHUNKS:
        assert c["text"] in out["answer"] and c["source"] in out["answer"]
    assert out["sources"] == ["orders.txt", "policies.txt"]


def test_a_working_model_is_not_marked_fallback(keys, monkeypatch):
    fake_openai(monkeypatch, {"groq-a": "Seven days."})
    out = query.answer("refund?", CHUNKS)
    assert out == {"answer": "Seven days.", "sources": ["orders.txt", "policies.txt"]}


def test_all_busy_and_only_far_off_passages(keys, monkeypatch):
    fake_openai(monkeypatch, {})
    far = [{"text": "Bake at 400.", "source": "x.txt", "distance": 1.8}]
    assert "nothing in the documents matched" in query.answer("capital of France?", far)["answer"]


def test_all_busy_and_nothing_matched(keys, monkeypatch):
    fake_openai(monkeypatch, {})
    assert "nothing in the documents matched" in query.answer("x", [])["answer"]
