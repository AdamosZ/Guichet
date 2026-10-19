"""The live pipeline, with a fake model: no API key, no embedding download."""

import json
import re
from types import SimpleNamespace

import guichet.service as service
from guichet.logbook import scrub


class FakeDense:
    def __init__(self, *args, **kwargs):
        pass

    def search(self, query, top_k):
        return []


class FakeClient:
    """Translation call → fixed French text; answer call → cites the first page shown."""

    def __init__(self, pages, quote_ok=True):
        self.pages, self.quote_ok, self.calls = pages, quote_ok, []
        self.messages = self

    def create(self, model, max_tokens, system, messages, **extra):
        user = messages[0]["content"]
        self.calls.append((system, user))
        if system.startswith("Translate"):
            text = "Quelle est la durée de validité d'un récépissé de titre de séjour ?"
        else:
            doc_id = re.search(r"\[(F\d+)\]", user).group(1)
            words = self.pages[doc_id].text.split()[:8]
            quote = " ".join(words) if self.quote_ok else "phrase inventée qui n'existe pas"
            text = json.dumps({"found": True, "answer": "ok",
                               "citations": [{"chunk_id": doc_id, "quote": quote}]})
        if extra.get("tools"):
            block = SimpleNamespace(type="tool_use", input=json.loads(text))
        else:
            block = SimpleNamespace(type="text", text=text)
        return SimpleNamespace(content=[block],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def make(monkeypatch, quote_ok=True, prompt="v1"):
    monkeypatch.setattr(service, "DenseIndex", FakeDense)
    monkeypatch.setattr(service, "PROMPT", prompt)
    g = service.Guichet(client=None)
    g.client = FakeClient(g.pages, quote_ok)
    return g


def test_english_question_is_translated_for_bm25_and_answered(monkeypatch):
    g = make(monkeypatch)
    answer = g.answer("How long is a receipt for a residence permit application valid?")
    assert g.client.calls[0][0].startswith("Translate")
    assert answer.language == "en" and answer.found and answer.sources
    assert "F15763" in answer.pages_seen  # BM25 on the French translation finds the récépissé page


def test_french_question_is_not_translated(monkeypatch):
    g = make(monkeypatch)
    g.answer("Quelle est la durée de validité d'un récépissé ?")
    assert not any(system.startswith("Translate") for system, _ in g.client.calls)


def test_quotes_not_on_the_page_are_dropped_and_answer_withheld(monkeypatch):
    g = make(monkeypatch, quote_ok=False)
    answer = g.answer("Quelle est la durée de validité d'un récépissé ?")
    assert answer.dropped_quotes == 1 and not answer.sources and answer.found is False


def test_scrub_removes_personal_numbers():
    text = "Mon numéro 1 85 05 75 123 456 78, tel 06 12 34 56 78, mail a.b@gmail.com"
    cleaned = scrub(text)
    assert "gmail" not in cleaned and "06 12" not in cleaned and "123 456" not in cleaned


def test_structured_prompt_answers_through_the_form(monkeypatch):
    g = make(monkeypatch, prompt="v2s")
    answer = g.answer("Quelle est la durée de validité d'un récépissé ?")
    assert answer.found and answer.sources and not answer.error
