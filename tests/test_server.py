"""The web API in demo mode: no API key, no model."""

import os

import pytest

pytest.importorskip("fastapi")
os.environ["GUICHET_DEMO"] = "1"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GUICHET_LOG_DIR", str(tmp_path))
    from fastapi.testclient import TestClient

    import server
    from guichet.logbook import Logbook

    server.logbook = Logbook(tmp_path)
    server.limits = server.Limits()
    return TestClient(server.app)


def test_page_and_health(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["demo"] is True


def test_answered_not_found_and_feedback(client):
    answered = client.post("/api/ask", json={"question": "How long is a récépissé valid?"}).json()
    assert answered["status"] == "answered" and answered["language"] == "en"
    assert answered["sources"][0]["verified_on"]
    missing = client.post("/api/ask", json={"question": "Which bank is best for students?"}).json()
    assert missing["status"] == "not_found" and missing["sources"] == []
    assert client.post("/api/feedback", json={"answer_id": answered["answer_id"], "rating": "down",
                                              "comment": "call me 06 12 34 56 78"}).json()["ok"]


def test_per_visitor_limit(client, monkeypatch):
    import server

    monkeypatch.setattr(server, "PER_USER_HOUR", 2)
    codes = [client.post("/api/ask", json={"question": "Combien de temps est valable un récépissé ?"}).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rejects_empty_and_oversized_questions(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 601}).status_code == 422
