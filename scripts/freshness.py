"""Keep the official pages behind Guichet up to date.

    python -m scripts.freshness              # report what changed, change nothing
    python -m scripts.freshness --auto       # update the app's pages, flag test answers
    python -m scripts.freshness --accept F1169,F2417 --by adam --note "…"   # manual accept

Government rules change (the July 2026 housing-aid rule for non-EU students is
an example this project ran into). Every page in data/corpus/fiche_ids.txt is
re-downloaded and compared with the stored copy.

--auto is what the weekly GitHub Action runs. Two different rules apply:

- The app always gets the newest page. The live official page is the truth,
  so every changed page replaces the stored copy, no human needed.
- The answer key only stays "reviewed" if the evidence still holds. A test
  item citing a changed page stays reviewed only when each of its evidence
  quotes for that page is still on the live page word for word and the page's
  « Vérifié le » date didn't move. Otherwise it goes back to unreviewed (so it
  stops counting in the reported numbers) and is listed for a human to
  re-check with `make review`.

Then the free retrieval eval is re-run and compared with the previous results.
Nothing here calls a paid API. Every decision is logged in
evals/page_reviews.jsonl; the summary goes to reports/freshness.md.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from guichet.corpus import FICHE_URL, Document, _client, extract, load_corpus

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus/documents.jsonl"
REVIEWS = ROOT / "evals/page_reviews.jsonl"
GOLDEN = ROOT / "evals/golden.jsonl"
RETRIEVAL = ROOT / "reports/retrieval.json"
# A drop larger than this on the headline mode is called out in the report.
DROP_ALERT = 0.02
HEADLINE = ("hybrid-trans", ("hit@1", "recall@5"))


def _fingerprint(text: str) -> str:
    words = re.findall(r"\w+", text.lower())
    return hashlib.sha256(" ".join(words).encode()).hexdigest()[:16]


def _changed_lines(old: str, new: str, limit: int = 6) -> list[str]:
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
    lines = [d for d in diff if d[:1] in "+-" and not d.startswith(("+++", "---")) and d[1:].strip()]
    return lines[:limit]


def _today() -> str:
    return time.strftime("%Y-%m-%d")


# --------------------------------------------------------------------- crawl
def crawl(limit: int | None = None, pause: float = 0.4) -> tuple[dict, dict[str, Document]]:
    """Re-download the pages. Returns the report and the live versions of changed pages."""
    stored = {d.doc_id: d for d in load_corpus(CORPUS)}
    ids = [i.strip() for i in (ROOT / "data/corpus/fiche_ids.txt").read_text().split() if i.strip()]
    if limit:
        ids = ids[:limit]
    golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line]

    changed, missing, live_docs = [], [], {}
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
                live_docs[ref] = live
                changed.append({
                    "doc_id": ref, "title": live.title, "url": live.url,
                    "verified_before": old.verified_on if old else None,
                    "verified_now": live.verified_on,
                    "sample_diff": _changed_lines(old.text if old else "", live.text),
                })
            time.sleep(pause)

    changed_ids = set(live_docs)
    affected = [{"item_id": g["item_id"], "pages": sorted(set(g["gold_doc_ids"]) & changed_ids),
                 "question_en": g["question_en"]}
                for g in golden if set(g["gold_doc_ids"]) & changed_ids]
    report = {"checked": len(ids), "changed": changed, "missing": missing,
              "affected_items": affected, "checked_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    return report, live_docs


def check(limit: int | None = None, pause: float = 0.4) -> dict:
    return crawl(limit, pause)[0]


# --------------------------------------------------------------------- apply
def apply_pages(live_docs: dict[str, Document], reviewer: str, note: str,
                notes: dict[str, str] | None = None) -> list[dict]:
    """Replace pages in the corpus, rebuild chunks, log each replacement."""
    from guichet.chunk import build_chunks

    documents = load_corpus(CORPUS)
    position = {d.doc_id: i for i, d in enumerate(documents)}
    records = []
    for ref, live in live_docs.items():
        old = documents[position[ref]]
        documents[position[ref]] = live
        records.append({"doc_id": ref, "reviewed_on": _today(), "reviewed_by": reviewer,
                        "verified_before": old.verified_on, "verified_now": live.verified_on,
                        "fingerprint_before": _fingerprint(old.text),
                        "fingerprint_now": _fingerprint(live.text),
                        "note": (notes or {}).get(ref, note)})
    with CORPUS.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")
    build_chunks(CORPUS, ROOT / "data/corpus/chunks.jsonl")
    with REVIEWS.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def accept(refs: list[str], reviewer: str, note: str, pause: float = 0.4) -> list[dict]:
    """A human checked the changed pages: swap the live versions into the corpus."""
    known = {d.doc_id for d in load_corpus(CORPUS)}
    unknown = [r for r in refs if r not in known]
    if unknown:
        raise SystemExit(f"not in the corpus: {', '.join(unknown)}")
    live_docs = {}
    with _client() as client:
        for ref in refs:
            response = client.get(FICHE_URL.format(ref=ref))
            response.raise_for_status()
            live = extract(response.text, ref, str(response.url))
            if live is None:
                raise SystemExit(f"{ref}: no usable content, nothing changed")
            live_docs[ref] = live
            time.sleep(pause)
    return apply_pages(live_docs, reviewer, note)


# --------------------------------------------------------------- answer key
def triage_items(golden: list[dict], live_docs: dict[str, Document],
                 redated: set[str]) -> tuple[list[dict], list[dict]]:
    """Split reviewed items citing changed pages into still-OK and needs-a-human.

    An item is cleared only if, for every changed page it cites, it has evidence
    quotes from that page, all of them are still on the live page, and the page
    was not re-dated. Anything else is sent back to human review.
    """
    from guichet.generate import quote_in

    cleared, flagged = [], []
    for item in golden:
        pages = sorted(set(item["gold_doc_ids"]) & set(live_docs))
        if not pages or not item.get("reviewed"):
            continue
        reasons = []
        for ref in pages:
            quotes = [e["quote"] for e in item.get("evidence", []) if e.get("doc_id") == ref]
            if ref in redated:
                reasons.append(f"{ref} re-dated (official update)")
            elif not quotes:
                reasons.append(f"{ref} changed and the item has no evidence quote to check")
            else:
                lost = [q for q in quotes if not quote_in(q, live_docs[ref].text)]
                if lost:
                    reasons.append(f"{ref}: {len(lost)} of {len(quotes)} evidence quotes no longer on the page")
        entry = {"item_id": item["item_id"], "pages": pages, "question_en": item["question_en"]}
        (flagged if reasons else cleared).append({**entry, "reasons": reasons})
    return cleared, flagged


def unreview(flagged: list[dict]) -> None:
    """Send flagged items back to human review, with the reason in their notes."""
    by_id = {f["item_id"]: f for f in flagged}
    lines = GOLDEN.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        if item["item_id"] in by_id:
            item["reviewed"] = False
            stamp = f"[freshness {_today()}: {'; '.join(by_id[item['item_id']]['reasons'])}]"
            item["notes"] = (item.get("notes", "") + " " + stamp).strip()
        out.append(json.dumps(item, ensure_ascii=False))
    GOLDEN.write_text("\n".join(out) + "\n", encoding="utf-8")


# ------------------------------------------------------------ retrieval eval
def headline(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    mode, metrics = HEADLINE
    langs = data.get(mode, {}).get("by_language", {})
    return {f"{lang} {m}": round(langs[lang][m], 3) for lang in sorted(langs) for m in metrics}


def rerun_eval() -> tuple[dict, dict]:
    before = headline(RETRIEVAL)
    subprocess.run([sys.executable, "-m", "guichet.run_eval"], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    return before, headline(RETRIEVAL)


# ------------------------------------------------------------------- report
def to_markdown(report: dict) -> str:
    redated = [c for c in report["changed"] if c["verified_before"] != c["verified_now"]]
    auto = report.get("auto")
    out = [f"## Freshness check, {report['checked_at']}",
           f"{report['checked']} pages checked: **{len(report['changed'])} changed** "
           f"({len(redated)} with a new « Vérifié le » date, the strongest sign of a rule update), "
           f"{len(report['missing'])} unreachable, **{len(report['affected_items'])} golden items to re-review**.",
           ""]
    if auto:
        out += ["### What happened automatically",
                f"- The app now uses the new version of all {len(report['changed'])} changed pages.",
                f"- {len(auto['cleared'])} test answers stayed reviewed: every evidence quote is still on the "
                "page and the page wasn't re-dated."]
        if auto["cleared"]:
            out.append("  " + ", ".join(f"`{c['item_id']}`" for c in auto["cleared"]))
        out.append("")
        if auto["flagged"]:
            out.append(f"### 👉 {len(auto['flagged'])} test answers need you (`make review`)")
            out += [f"- `{f['item_id']}`: {f['question_en']}  \n  _{'; '.join(f['reasons'])}_"
                    for f in auto["flagged"]]
            out += ["", "They no longer count in the reported numbers until re-reviewed.", ""]
        if auto.get("eval_after"):
            out.append("### Retrieval check (hybrid-trans)")
            out.append("| metric | before | after |")
            out.append("|---|---|---|")
            for key, after in auto["eval_after"].items():
                before = auto["eval_before"].get(key)
                drop = before is not None and before - after > DROP_ALERT
                shown = f"{before:.3f}" if before is not None else "–"
                out.append(f"| {key} | {shown} | {after:.3f}{' ⚠️' if drop else ''} |")
            if auto["flagged"]:
                out.append("\n_The answer key shrank this week, so part of any change is the smaller set._")
            out.append("")
    else:
        out += ["_A changed page with the same « Vérifié le » date is often only a reworded link or menu; "
                "the sample lines below show which._", ""]
        if report["affected_items"]:
            out.append("### Golden items citing a changed page")
            out += [f"- `{a['item_id']}` ({', '.join(a['pages'])}): {a['question_en']}"
                    for a in report["affected_items"]]
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
        out.append("### Unreachable (kept the stored copy)")
        out += [f"- {m['doc_id']}: {m['error']}" for m in report["missing"]]
    return "\n".join(out)


def run_auto(limit: int | None = None, evaluate: bool = True) -> dict:
    report, live_docs = crawl(limit)
    report["auto"] = {"cleared": [], "flagged": [], "eval_before": {}, "eval_after": {}}
    if not live_docs:
        return report
    redated = {c["doc_id"] for c in report["changed"] if c["verified_before"] != c["verified_now"]}
    golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line]
    cleared, flagged = triage_items(golden, live_docs, redated)
    page_notes = {}
    for ref in live_docs:
        sent_back = [f["item_id"] for f in flagged if ref in f["pages"]]
        kept = [c["item_id"] for c in cleared if ref in c["pages"]]
        if sent_back:
            page_notes[ref] = "auto-updated; test items sent back for review: " + ", ".join(sent_back)
        elif kept:
            page_notes[ref] = "auto-updated; evidence still on the page for " + ", ".join(kept)
        else:
            page_notes[ref] = "auto-updated; no reviewed test item cites this page"
    apply_pages(live_docs, "auto", "", page_notes)
    if flagged:
        unreview(flagged)
    report["auto"].update(cleared=cleared, flagged=flagged)
    # Only the items sent back actually need a human this week.
    report["affected_items"] = [{k: f[k] for k in ("item_id", "pages", "question_en")} for f in flagged]
    if evaluate:
        before, after = rerun_eval()
        report["auto"].update(eval_before=before, eval_after=after)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--auto", action="store_true", help="update changed pages and triage the answer key")
    parser.add_argument("--no-eval", action="store_true", help="with --auto: skip the retrieval re-run")
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
    report = run_auto(args.limit, not args.no_eval) if args.auto else check(args.limit)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/freshness.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = to_markdown(report)
    (ROOT / "reports/freshness.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
