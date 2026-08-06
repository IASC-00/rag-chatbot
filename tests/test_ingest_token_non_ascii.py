"""A non-ASCII ingest token must be rejected, not crash the request.

`hmac.compare_digest` raises TypeError on non-ASCII str, and this value is a
request header — fully attacker-controlled. One accented character returned an
unhandled 500 instead of 401. Found by a portfolio-wide sweep after the same
defect appeared in two other repositories.
"""


import pytest

from app import _const_eq
from app import app as flask_app


def test_const_eq_rejects_non_ascii_without_raising():
    assert _const_eq("ü", "abc") is False
    assert _const_eq("🙂", "abc") is False


def test_const_eq_matches_and_rejects_ascii():
    assert _const_eq("tok", "tok") is True
    assert _const_eq("tok", "tok2") is False


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("INGEST_TOKEN", "real-token")
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_ingest_non_ascii_token_returns_401_not_500(client):
    r = client.post("/ingest", headers={"X-Ingest-Token": "ü"})
    assert r.status_code != 500
    assert r.status_code == 401


def test_ingest_wrong_ascii_token_still_401(client):
    assert client.post("/ingest", headers={"X-Ingest-Token": "nope"}).status_code == 401
