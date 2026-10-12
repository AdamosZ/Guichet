"""Walk through unreviewed golden items one at a time and mark them reviewed.

    python -m scripts.review_golden

For each item it prints both questions, both answers, the gold documents'
URLs, the supporting sentences from those pages, the independent pre-check's
verdict, and the real post the question came from. Open the URL, check the answer against the page, then:

    a  accept as-is (sets reviewed: true)
    n  add a note, then accept
    f  flag: add a note, leave unreviewed (needs fixing in the file)
    s  skip for now
    q  save and quit

Progress is saved after every decision, so quitting halfway loses nothing.
Corrections to the question or answer text are made directly in
evals/golden.jsonl — this tool is for the yes/no pass, not for editing.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from guichet.corpus import load_corpus
from guichet.golden import GOLDEN_PATH, load, save

DOCS_PATH = Path("data/corpus/documents.jsonl")


def _wrap(label: str, text: str) -> str:
    body = textwrap.fill(text, width=88, subsequent_indent=" " * 12)
    return f"  {label:<10}{body}"


def main() -> None:
    items = load(GOLDEN_PATH)
    documents = {d.doc_id: d for d in load_corpus(DOCS_PATH)}
    pending = [i for i in items if not i.reviewed]
    print(f"{len(pending)} of {len(items)} items still need review\n")

    for count, item in enumerate(pending, start=1):
        print("=" * 92)
        kind = "answerable" if item.answerable else "UNANSWERABLE (should have no gold docs)"
        print(f"[{count}/{len(pending)}] {item.item_id}  ·  {item.category}  ·  {kind}")
        print(_wrap("Q (fr)", item.question_fr))
        print(_wrap("Q (en)", item.question_en))
        print(_wrap("A (fr)", item.answer_fr or "—"))
        print(_wrap("A (en)", item.answer_en or "—"))
        for doc_id in item.gold_doc_ids:
            doc = documents.get(doc_id)
            if doc:
                print(f"  source    {doc_id}  {doc.title}\n            {doc.url}")
            else:
                print(f"  source    {doc_id}  (NOT IN CORPUS)")
        for ev in item.evidence:
            print(_wrap("page says", f"[{ev.get('doc_id')}] « {ev.get('quote', '')} »"))
        if item.check:
            print(_wrap("pre-check", f"{item.check.get('verdict')}: {item.check.get('reason', '')}"))
        if item.source_url:
            print(f"  asked at  {item.source_url}")
        if item.notes:
            print(_wrap("notes", item.notes))

        while True:
            choice = input("\n  [a]ccept  [n]ote+accept  [f]lag  [s]kip  [q]uit > ").strip().lower()
            if choice in {"a", "n", "f", "s", "q"}:
                break

        if choice == "q":
            break
        if choice in {"n", "f"}:
            note = input("  note: ").strip()
            item.notes = f"{item.notes} | {note}".strip(" |") if item.notes else note
        if choice in {"a", "n"}:
            item.reviewed = True
        save(items, GOLDEN_PATH)

    save(items, GOLDEN_PATH)
    done = sum(i.reviewed for i in items)
    print(f"\nsaved. {done}/{len(items)} reviewed.")


if __name__ == "__main__":
    main()
