"""Walk through unreviewed golden items one at a time and mark them reviewed.

    python -m scripts.review_golden           # full view; flagged items first
    python -m scripts.review_golden --quick   # compact view of triage-clear items

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


def _triage(item) -> str:
    return (item.check or {}).get("triage", {}).get("level", "")


def _quick(item, count: int, total: int) -> None:
    print("-" * 92)
    print(f"[{count}/{total}] {item.item_id}  ·  {item.category}  ·  pre-checked, triage clear")
    print(_wrap("Q (fr)", item.question_fr))
    print(_wrap("Q (en)", item.question_en))
    print(_wrap("A (en)", item.answer_en or "— (unanswerable)"))
    if item.evidence:
        print(_wrap("page says", f"« {item.evidence[0].get('quote', '')} »"))


def main() -> None:
    import sys

    quick = "--quick" in sys.argv
    items = load(GOLDEN_PATH)
    documents = {d.doc_id: d for d in load_corpus(DOCS_PATH)}
    pending = [i for i in items if not i.reviewed]
    # Items the triage pass changed or wants a human to look at come first.
    order = {"CHANGED": 0, "LOOK": 0, "": 1, "GREEN": 2}
    pending.sort(key=lambda i: order.get(_triage(i), 1))
    if quick:
        pending = [i for i in pending if _triage(i) == "GREEN"]
    print(f"{len(pending)} items in this pass ({sum(not i.reviewed for i in items)} unreviewed in total)\n")

    for count, item in enumerate(pending, start=1):
        if quick:
            _quick(item, count, len(pending))
            choice = ""
            while choice not in {"a", "f", "s", "q"}:
                choice = input("  [a]ccept  [f]lag  [s]kip  [q]uit > ").strip().lower()
            if choice == "q":
                break
            if choice == "f":
                note = input("  note: ").strip()
                item.notes = f"{item.notes} | {note}".strip(" |") if item.notes else note
            if choice == "a":
                item.reviewed = True
            save(items, GOLDEN_PATH)
            continue

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
        if _triage(item) in {"CHANGED", "LOOK"}:
            print(_wrap(">>> LOOK", f"[{_triage(item)}] {item.check['triage']['issue']}"))
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
