"""The answer key in the repo must always be valid — a broken label corrupts every number."""

from pathlib import Path

from guichet.corpus import load_corpus
from guichet.golden import load, validate

ROOT = Path(__file__).resolve().parents[1]


def test_golden_set_is_valid():
    items = load(ROOT / "evals/golden.jsonl")
    known = {d.doc_id for d in load_corpus(ROOT / "data/corpus/documents.jsonl")}
    assert items, "golden set is empty"
    assert validate(items, known) == []


def test_unanswerable_items_kept():
    items = load(ROOT / "evals/golden.jsonl")
    assert any(not i.answerable for i in items), "unanswerable items must stay in the set"
