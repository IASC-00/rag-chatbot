import pytest

import app as app_module


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "pipe", lambda q: {"answer": "ok", "sources": []})
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_chat_requires_a_query(client):
    assert client.post("/chat", json={}).status_code == 400


def test_chat_returns_the_answer(client):
    body = client.post("/chat", json={"query": "hi"}).get_json()
    assert body["response"]["answer"] == "ok"


def test_chat_failure_does_not_leak_internals(client, monkeypatch):
    def boom(_q):
        raise RuntimeError("/home/ias/secret/path.py exploded")

    monkeypatch.setattr(app_module, "pipe", boom)
    response = client.post("/chat", json={"query": "hi"})
    assert response.status_code == 500
    assert "secret" not in response.get_data(as_text=True)


def test_ingest_disabled_when_no_token_configured(client, monkeypatch):
    monkeypatch.delenv("INGEST_TOKEN", raising=False)
    assert client.post("/ingest").status_code == 503


def test_ingest_rejects_a_missing_token(client, monkeypatch):
    monkeypatch.setenv("INGEST_TOKEN", "secret-token")
    assert client.post("/ingest").status_code == 401


def test_ingest_rejects_a_wrong_token(client, monkeypatch):
    monkeypatch.setenv("INGEST_TOKEN", "secret-token")
    assert client.post("/ingest", headers={"X-Ingest-Token": "wrong"}).status_code == 401


def test_ingest_accepts_the_correct_token(client, monkeypatch):
    monkeypatch.setenv("INGEST_TOKEN", "secret-token")
    monkeypatch.setattr(app_module, "run_ingest", lambda: {"documents": 1, "chunks": 2})
    response = client.post("/ingest", headers={"X-Ingest-Token": "secret-token"})
    assert response.status_code == 200
    assert response.get_json()["chunks"] == 2


def test_health_is_open(client):
    assert client.get("/health").status_code == 200
