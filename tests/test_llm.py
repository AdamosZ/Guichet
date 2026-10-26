"""The model adapter, with a fake Hugging Face router: no network, no key."""

import json

import pytest

from guichet import llm
from guichet.generate import ANSWER_TOOL


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self, *responses):
        self.responses, self.bodies = list(responses), []

    def post(self, url, json=None, headers=None):
        assert url == llm.HF_ROUTER and headers["Authorization"] == "Bearer hf_test"
        self.bodies.append(json)
        return self.responses.pop(0)


def ok(message, finish="stop"):
    return FakeResponse(200, {"choices": [{"message": message, "finish_reason": finish}],
                              "usage": {"prompt_tokens": 50_000, "completion_tokens": 600}})


@pytest.fixture(autouse=True)
def token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def test_forced_function_call_is_parsed():
    form = {"found": True, "answer": "4 ou 6 mois.", "citations": [{"chunk_id": "F1", "quote": "q"}]}
    http = FakeHTTP(ok({"tool_calls": [{"function": {"name": "submit_answer", "arguments": json.dumps(form)}}]}))
    reply = llm.complete("openai/gpt-oss-120b:deepinfra", "sys", "user", tool=ANSWER_TOOL, http=http)
    assert reply.data == form and (reply.tokens_in, reply.tokens_out) == (50_000, 600)
    sent = http.bodies[0]
    assert sent["model"] == "openai/gpt-oss-120b:deepinfra"
    assert sent["tool_choice"]["function"]["name"] == "submit_answer"
    assert sent["tools"][0]["function"]["parameters"] == ANSWER_TOOL["input_schema"]


def test_provider_that_ignores_tools_falls_back_to_json_in_text():
    http = FakeHTTP(ok({"content": 'Voici: {"found": false, "answer": "", "citations": []}'}))
    reply = llm.complete("some/model", "sys", "user", tool=ANSWER_TOOL, http=http)
    assert reply.data["found"] is False and "parse_error" not in reply.data


def test_rate_limit_is_retried_then_succeeds():
    http = FakeHTTP(FakeResponse(429, {"error": "slow down"}), ok({"content": "Bonjour"}))
    assert llm.complete("some/model", "sys", "hello", http=http).text == "Bonjour"
    assert len(http.bodies) == 2


def test_client_errors_are_not_retried():
    http = FakeHTTP(FakeResponse(400, {"error": "bad model"}))
    with pytest.raises(RuntimeError, match="400"):
        llm.complete("some/model", "sys", "hello", http=http)


def test_model_ids_become_safe_file_names():
    assert llm.safe_name("openai/gpt-oss-120b:deepinfra") == "openai--gpt-oss-120b@deepinfra"
    assert llm.safe_name("claude-haiku-5-5") == "claude-haiku-5-5"


def test_missing_permission_is_an_access_error_not_retried():
    http = FakeHTTP(FakeResponse(403, {"error": "insufficient permissions"}))
    with pytest.raises(llm.AccessError, match="403"):
        llm.complete("some/model", "sys", "hello", http=http)
    assert len(http.bodies) == 1
