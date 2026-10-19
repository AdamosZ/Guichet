"""Have the official pages changed since the corpus was crawled?

    python -m scripts.freshness              # check every page in the corpus
    python -m scripts.freshness --limit 10   # quick check

Government rules change (the July 2026 housing-aid rule for non-EU students is
an example this project ran into). For every page in data/corpus/fiche_ids.txt
this re-downloads the live page, compares its text with the stored copy, and
lists:

- pages whose content changed, with their old and new "Vérifié le" dates;
- golden items that cite a changed page, whose reference answers need
  re-checking before they can be trusted again;
- pages that no longer load.

Writes reports/freshness.json and reports/freshness.md (the Markdown is what
the weekly GitHub Action posts). Nothing in the corpus or answer key is
changed automatically: a changed page means a human re-reviews.

After the review, accept the new versions so the app answers from them:

    python -m scripts.freshness --accept F1169,F2417 --by adam --note "answers still correct"

The pages are replaced in the corpus, chunks are rebuilt, and the review is
logged in evals/page_reviews.jsonl. If a reference answer was wrong, fix it in
evals/golden.jsonl first.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import time
from pathlib import Path

from guichet.corpus import FICHE_URL, _client, extract, load_corpus

ROOT = Path(__file__).resolve().parents[1]


def _fingerprint(text: str) -> str:
    words = re.findall(r"\w+", text.lower())
    return hashlib.sha256(" ".join(words).encode()).hexdigest()[:16]


def _changed_lines(old: str, new: str, limit: int = 6) -> list[str]:
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
    lines = [d for d in diff if d[:1] in "+-" and not d.startswith(("+++", "---")) and d[1:].strip()]
    return lines[:limit]


def check(limit: int | None = None, pause: float = 0.4) -> dict:
    stored = {d.doc_id: d for d in load_corpus(ROOT / "data/corpus/documents.jsonl")}
    ids = [i.strip() for i in (ROOT / "data/corpus/fiche_ids.txt").read_text().split() if i.strip()]
    if limit:
        ids = ids[:limit]
    golden = [json.loads(line) for line in (ROOT / "evals/golden.jsonl").read_text(encoding="utf-8").splitlines()]

    changed, missing = [], []
    with _client() as client:
        for ref in ids:
            try:
                response = client.get(FICHE_URL.format(ref=ref))
                response.raise_for_status()
                live = extract(response.text, ref, str(response.url))
            except Exception as exc:  # network error, 404, layout change
                missing.append({"doc_id": ref, "error": type(exc).__name__})
                continue
            if live is None:
                missing.append({"doc_id": ref, "error": "no usable content"})
                continue
            old = stored.get(ref)
            if old is None or _fingerprint(old.text) != _fingerprint(live.text):
                changed.append({
                    "doc_id": ref, "title": live.title, "url": live.url,
                    "verified_before": old.verified_on if old else None,
                    "verified_now": live.verified_on,
                    "sample_diff": _changed_lines(old.text if old else "", live.text),
                })
            time.sleep(pause)

    changed_ids = {c["doc_id"] for c in changed}
    affected = [{"item_id": g["item_id"], "pages": sorted(set(g["gold_doc_ids"]) & changed_ids),
                 "question_en": g["question_en"]}
                for g in golden if set(g["gold_doc_ids"]) & changed_ids]
    return {"checked": len(ids), "changed": changed, "missing": missing,
            "affected_items": affected, "checked_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}


def to_markdown(report: dict) -> str:
    redated = [c for c in report["changed"] if c["verified_before"] != c["verified_now"]]
    out = [f"## Freshness check, {report['checked_at']}",
           f"{report['checked']} pages checked: **{len(report['changed'])} changed** "
           f"({len(redated)} with a new « Vérifié le » date, the strongest sign of a rule update), "
           f"{len(report['missing'])} unreachable, **{len(report['affected_items'])} golden items to re-review**.",
           "",
           "_A changed page with the same « Vérifié le » date is often only a reworded link or menu; "
           "the sample lines below show which._", ""]
    if report["affected_items"]:
        out.append("### Golden items citing a changed page")
        out += [f"- `{a['item_id']}` ({', '.join(a['pages'])}): {a['question_en']}" for a in report["affected_items"]]
        out.append("")
    if report["changed"]:
        out.append("### Changed pages")
        for c in report["changed"]:
            flag = "🔴 re-dated" if c["verified_before"] != c["verified_now"] else "🟡 same date"
            out.append(f"- {flag} [{c['doc_id']}]({c['url']}) {c['title']}: "
                       f"vérifié {c['verified_before']} → {c['verified_now']}")
            out += [f"    `{line[:140]}`" for line in c["sample_diff"]]
        out.append("")
    if report["missing"]:
        out.append("### Unreachable")
        out += [f"- {m['doc_id']}: {m['error']}" for m in report["missing"]]
    return "\n".join(out)


def accept(refs: list[str], reviewer: str, note: str, pause: float = 0.4) -> list[dict]:
    """A human checked the changed pages: swap the live versions into the corpus.

    Re-downloads each page, replaces it in data/corpus/documents.jsonl, rebuilds
    chunks.jsonl (the embedding cache rebuilds itself on next start) and appends
    one line per page to evals/page_reviews.jsonl so the review is on record.
    """
    from dataclasses import asdict

    from guichet.chunk import build_chunks

    corpus_path = ROOT / "data/corpus/documents.jsonl"
    documents = load_corpus(corpus_path)
    position = {d.doc_id: i for i, d in enumerate(documents)}
    unknown = [r for r in refs if r not in position]
    if unknown:
        raise SystemExit(f"not in the corpus: {', '.join(unknown)}")

    records = []
    with _client() as client:
        for ref in refs:
            response = client.get(FICHE_URL.format(ref=ref))
            response.raise_for_status()
            live = extract(response.text, ref, str(response.url))
            if live is None:
                raise SystemExit(f"{ref}: no usable content, nothing changed")
            old = documents[position[ref]]
            documents[position[ref]] = live
            records.append({"doc_id": ref, "reviewed_on": time.strftime("%Y-%m-%d"), "reviewed_by": reviewer,
                            "verified_before": old.verified_on, "verified_now": live.verified_on,
                            "fingerprint_before": _fingerprint(old.text),
                            "fingerprint_now": _fingerprint(live.text), "note": note})
            time.sleep(pause)

    with corpus_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")
    build_chunks(corpus_path, ROOT / "data/corpus/chunks.jsonl")
    with (ROOT / "evals/page_reviews.jsonl").open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--accept", help="comma-separated Fxxxx pages a human re-checked: update them in the corpus")
    parser.add_argument("--by", default="", help="who re-checked them (required with --accept)")
    parser.add_argument("--note", default="", help="what was checked")
    args = parser.parse_args()
    if args.accept:
        if not args.by:
            raise SystemExit("--accept needs --by <name>")
        for record in accept([r.strip() for r in args.accept.split(",") if r.strip()], args.by, args.note):
            print(f"updated {record['doc_id']}: vérifié {record['verified_before']} → {record['verified_now']}")
        return
    report = check(args.limit)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/freshness.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = to_markdown(report)
    (ROOT / "reports/freshness.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
