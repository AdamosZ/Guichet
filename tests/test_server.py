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
    privacy = client.get("/privacy")
    assert privacy.status_code == 200 and "90 days" in privacy.text and "90 jours" in privacy.text
    assert 'href="/privacy"' in client.get("/").text


def test_answered_not_found_and_feedback(client):
    answered = client.post("/api/ask", json={"question": "How long is a récépissé valid?"}).json()
    assert answered["status"] == "answered" and answered["language"] == "en"
    assert answered["sources"][0]["verified_on"]
    missing = client.post("/api/ask", json={"question": "Which bank is best for students?"}).json()
    assert missing["status"] == "not_found" and missing["sources"] == []
    assert client.post("/api/feedback", json={"answer_id": answered["answer_id"], "rating": "down",
                                              "comment": "call me 06 12 34 56 78"}).json()["ok"]


def test_per_visitor_limit(client):
    import server

    server.limits = server.Limits(per_hour=2)
    codes = [client.post("/api/ask", json={"question": "Combien de temps est valable un récépissé ?"}).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]


def test_faked_forwarded_header_does_not_reset_the_limit(client):
    import server

    server.limits = server.Limits(per_hour=2)
    question = {"question": "Combien de temps est valable un récépissé ?"}
    codes = []
    for fake in ("1.1.1.1", "8.8.8.8", "9.9.9.9"):
        # The proxy appends the real address (here 93.184.216.34) after whatever the visitor sent.
        headers = {"x-forwarded-for": f"{fake}, 93.184.216.34, 10.0.0.7"}
        codes.append(client.post("/api/ask", json=question, headers=headers).status_code)
    assert codes == [200, 200, 429]


def test_visitor_is_right_most_public_address():
    from types import SimpleNamespace

    import server

    def request(xff):
        return SimpleNamespace(headers={"x-forwarded-for": xff}, client=SimpleNamespace(host="10.0.0.1"))

    assert server._visitor(request("6.6.6.6, 93.184.216.34, 10.1.2.3")) == "93.184.216.34"
    assert server._visitor(request("not-an-ip, 10.1.2.3")) == "10.0.0.1"
    assert server._visitor(request("")) == "10.0.0.1"


def test_feedback_is_validated_and_rate_limited(client):
    import server

    answer_id = client.post("/api/ask", json={"question": "How long is a récépissé valid?"}).json()["answer_id"]
    assert client.post("/api/feedback", json={"answer_id": "../../etc", "rating": "up"}).status_code == 422
    server.feedback_limits = server.Limits(per_hour=2)
    codes = [client.post("/api/feedback", json={"answer_id": answer_id, "rating": "up"}).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]


def test_security_headers_and_no_public_api_schema(client):
    response = client.get("/")
    assert "frame-ancestors" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/openapi.json").status_code == 404


def test_rejects_empty_and_oversized_questions(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 601}).status_code == 422
