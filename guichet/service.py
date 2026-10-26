"""The live answering service: the configuration the evaluation showed works best.

    question ─┬─ dense search (original question, multilingual embeddings)
              └─ BM25 (French translation of the question, if it isn't French)
                  → RRF fusion → top 5 distinct pages, full text
                  → Claude Haiku answers with verbatim quotes, or says "not found"
                  → every quote is re-checked against the page; failures are dropped

Measured on 104 human-reviewed question pairs (see README): no fabricated
citations, 20/20 refusals on unanswerable questions; with the v2 answer style,
89% (FR) / 82% (EN) of answerable questions answered with a correct page cited.

The Anthropic client is injected, so the service can be tested without a key.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .chunk import load_chunks
from .corpus import load_corpus
from .generate import (
    ANSWER_TOOL,
    PROMPTS,
    STRUCTURED,
    _prompt,
    parse_reply,
    quote_in,
    request_kwargs,
)
from .index import BM25Index, DenseIndex, reciprocal_rank_fusion
from .langid import detect
from .llm import complete, is_claude

ROOT = Path(__file__).resolve().parents[1]
# Any Claude model id, or a Hugging Face model id with an optional ":provider"
# (e.g. "openai/gpt-oss-120b:deepinfra"). Only switch after it has been
# evaluated on the golden set (python -m guichet.generate run --model …).
MODEL = os.environ.get("GUICHET_MODEL", "claude-haiku-5-5")
# Which answer-writing prompt the live app uses. Only switch after the new
# prompt has been re-evaluated on the golden set (python -m guichet.generate --prompt …).
PROMPT = os.environ.get("GUICHET_PROMPT", "v2s")  # v2s chosen 19 Oct 2026, see README
PAGES = 5
SERVICE_PUBLIC_SEARCH = "https://www.service-public.gouv.fr/particuliers/recherche?keyword="

TRANSLATE_SYSTEM = ("Translate the user's question into natural French as a French student would write it. "
                    "Keep French administrative terms (titre de séjour, CAF, récépissé…) as they are. "
                    "Reply with the translation only.")


@dataclass
class Source:
    doc_id: str
    title: str
    url: str
    quote: str
    verified_on: str | None = None


@dataclass
class Answer:
    question: str
    language: str
    found: bool
    text: str
    sources: list[Source] = field(default_factory=list)
    dropped_quotes: int = 0
    pages_seen: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""
    model: str = ""


class Guichet:
    def __init__(self, client=None, model: str = MODEL, root: Path = ROOT):
        self.model = model
        self.client = client
        self.chunks = load_chunks(root / "data/corpus/chunks.jsonl")
        self.pages = {d.doc_id: d for d in load_corpus(root / "data/corpus/documents.jsonl")}
        cache = Path(os.environ.get("GUICHET_CACHE", root / "data/cache"))
        self.dense = DenseIndex(self.chunks, cache_dir=cache)
        self.sparse = BM25Index(self.chunks)
        self._translations: dict[str, str] = {}

    # ------------------------------------------------------------------ helpers
    def _client(self):
        if self.client is None and is_claude(self.model):
            import anthropic

            self.client = anthropic.Anthropic()
        return self.client

    def _call(self, system: str, user: str, max_tokens: int, tool: dict | None = None):
        return complete(self.model, system, user, tool=tool, max_tokens=max_tokens, client=self._client())

    def translate(self, question: str) -> str:
        if question not in self._translations:
            reply = self._call(TRANSLATE_SYSTEM, question, 300)
            self._translations[question] = reply.text or question
        return self._translations[question]

    # ------------------------------------------------------------------ pipeline
    def retrieve(self, question: str, language: str) -> list[str]:
        sparse_query = question if language == "fr" else self.translate(question)
        fused = reciprocal_rank_fusion([self.dense.search(question, 50),
                                        self.sparse.search(sparse_query, 50)])
        order: list[str] = []
        for index, _score in fused:
            doc_id = self.chunks[index].doc_id
            if doc_id not in order:
                order.append(doc_id)
            if len(order) == PAGES:
                break
        return order

    def answer(self, question: str) -> Answer:
        question = question.strip()
        language = detect(question)
        result = Answer(question=question, language=language, found=False, text="", model=self.model)
        try:
            doc_ids = self.retrieve(question, language)
            result.pages_seen = doc_ids
            row = {"question": question, "language": language,
                   "passages": [{"chunk_id": d, "doc_id": d, "title": self.pages[d].title,
                                 "text": self.pages[d].text} for d in doc_ids]}
            tool = ANSWER_TOOL if PROMPT in STRUCTURED else None
            message = self._call(PROMPTS[PROMPT], _prompt(row), request_kwargs(PROMPT)["max_tokens"], tool)
            result.tokens_in = message.tokens_in
            result.tokens_out = message.tokens_out
            reply = message.data if tool else parse_reply(message.text)
        except Exception as exc:  # network, quota, malformed index…
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        if reply.get("parse_error"):
            result.error = "malformed model reply"
            return result

        seen: set[str] = set()
        for citation in reply.get("citations") or []:
            doc_id, quote = citation.get("chunk_id", ""), citation.get("quote", "")
            page = self.pages.get(doc_id)
            # Only show quotes that really are on the page the model was given.
            if doc_id in doc_ids and page and quote_in(quote, page.text):
                if (doc_id, quote) not in seen:
                    seen.add((doc_id, quote))
                    result.sources.append(Source(doc_id, page.title, page.url, quote, page.verified_on))
            else:
                result.dropped_quotes += 1

        result.found = bool(reply.get("found")) and bool(result.sources)
        result.text = reply.get("answer", "") or ""
        return result

    def not_found_link(self, question: str) -> str:
        from urllib.parse import quote

        query = question if detect(question) == "fr" else self._translations.get(question, question)
        return SERVICE_PUBLIC_SEARCH + quote(query)


def answer_to_dict(answer: Answer) -> dict:
    data = answer.__dict__.copy()
    data["sources"] = [s.__dict__ for s in answer.sources]
    return data


if __name__ == "__main__":
    import sys

    from .generate import _load_env

    _load_env()
    result = Guichet().answer(" ".join(sys.argv[1:]) or "How long is a récépissé valid?")
    print(json.dumps(answer_to_dict(result), indent=2, ensure_ascii=False))
