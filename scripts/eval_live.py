"""Evaluate the pipeline exactly as deployed: every reviewed test question goes
through guichet.service.Guichet.answer, the same code the live site runs.

    python -m scripts.eval_live               # all reviewed items, FR and EN (~$1.25)
    python -m scripts.eval_live --limit 6     # smoke test
    python -m scripts.eval_live --score-only  # re-score a finished run, no API calls

What differs from the offline eval (guichet.generate) and why this exists:
the live site translates questions with Claude (the offline eval used cached
opus-mt translations), searches the *current* pages (the weekly job keeps
updating them), and only shows quotes that survive re-checking. The numbers
on the site should describe what the site does, so they come from here.

Answers are saved in the generations format, so the LLM judge can grade them:
    inspect eval guichet/judge.py -T variant=live -T prompt=v2s --model mockllm/model
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from guichet.generate import _load_env, generations_path
from guichet.golden import load
from guichet.langid import detect
from scripts.weekly_report import DEFAULT_MODEL, PRICES

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/live_eval.json"
OFFLINE = ROOT / "reports/generation__claude-haiku-5-5__pages5__v2.json"


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _cost(row: dict) -> float:
    price_in, price_out = PRICES.get(row.get("model") or DEFAULT_MODEL, PRICES[DEFAULT_MODEL])
    return row["tokens_in"] / 1e6 * price_in + row["tokens_out"] / 1e6 * price_out


def score(rows: list[dict], gold: dict[str, dict]) -> dict:
    """Per-language metrics, computed by code from the saved answers."""
    out = {}
    for language in ("fr", "en"):
        mine = [r for r in rows if r["language"] == language]
        ok = [r for r in mine if not r.get("error")]
        answerable = [r for r in ok if gold[r["item_id"]]["answerable"]]
        unanswerable = [r for r in ok if not gold[r["item_id"]]["answerable"]]

        def cites_gold(r):
            return bool(set(r["source_ids"]) & set(gold[r["item_id"]]["gold_doc_ids"]))

        def saw_gold(r):
            return bool(set(r["pages_seen"]) & set(gold[r["item_id"]]["gold_doc_ids"]))

        shown = sum(len(r["source_ids"]) for r in ok)
        dropped = sum(r["dropped_quotes"] for r in ok)
        answered = [r for r in ok if r["found"]]
        out[language] = {
            "n": len(mine),
            "errors": len(mine) - len(ok),
            # The headline: answerable questions answered with a correct official page cited.
            "answered_with_gold_citation": _mean([float(r["found"] and cites_gold(r)) for r in answerable]),
            "answered_citing_wrong_page": _mean([float(r["found"] and not cites_gold(r)) for r in answerable]),
            "said_not_found": _mean([float(not r["found"]) for r in answerable]),
            "correct_abstention_rate": _mean([float(not r["found"]) for r in unanswerable]),
            # Retrieval in the live pipeline: was a correct page among the 5 it read?
            "gold_page_in_top5": _mean([float(saw_gold(r)) for r in answerable]),
            # Quotes the model proposed that failed the word-for-word check (never shown).
            "quotes_dropped_rate": round(dropped / (shown + dropped), 3) if shown + dropped else None,
            "answer_language_ok": _mean([float(detect(r["answer"]) == language) for r in answered if r["answer"]]),
            "seconds_mean": _mean([r["seconds"] for r in ok]),
            "cost_per_question_usd": _mean([_cost(r) for r in ok]),
        }
    return out


def run(limit: int | None, workers: int) -> Path:
    from guichet.service import MODEL, PROMPT, Guichet

    _load_env()
    out = generations_path(MODEL, "live", PROMPT)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row.get("error"):
                done.add((row["item_id"], row["language"]))

    todo = [(item, language, getattr(item, f"question_{language}"))
            for item in load() if item.reviewed for language in ("fr", "en")]
    todo = [t for t in todo if (t[0].item_id, t[1]) not in done]
    if limit:
        todo = todo[:limit]
    print("loading the live pipeline (search index, pages)…")
    guichet = Guichet()
    print(f"{len(todo)} questions to ask ({len(done)} already done), model {MODEL}, prompt {PROMPT}")

    lock = threading.Lock()

    def ask(task):
        item, language, question = task
        from guichet.service import Answer

        started = time.time()
        try:
            answer = guichet.answer(question)
        except Exception as exc:  # recorded as a failure (retried on the next run), never a crash
            answer = Answer(question=question, language=language, found=False, text="",
                            error=f"unhandled {type(exc).__name__}: {exc}"[:300])
        return item, language, answer, time.time() - started

    with out.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(ask, t) for t in todo]
        for n, future in enumerate(as_completed(futures), start=1):
            item, language, answer, seconds = future.result()
            row = {
                "item_id": item.item_id, "language": language, "model": MODEL, "prompt": PROMPT,
                "question": answer.question, "detected_language": answer.language,
                "found": answer.found, "answer": answer.text,
                "source_ids": [s.doc_id for s in answer.sources], "quotes": [s.quote for s in answer.sources],
                "pages_seen": answer.pages_seen, "dropped_quotes": answer.dropped_quotes,
                "tokens_in": answer.tokens_in, "tokens_out": answer.tokens_out,
                "seconds": round(seconds, 1), "error": answer.error,
            }
            with lock:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            status = "ERROR " + answer.error[:60] if answer.error else ("answered" if answer.found else "not found")
            print(f"  {n}/{len(todo)}  {item.item_id} [{language}]  {status}  {seconds:.1f}s")
    return out


def report(path: Path) -> dict:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["item_id"], row["language"])
        if key not in rows or not row.get("error"):  # a later successful retry replaces an error
            rows[key] = row
    gold = {item.item_id: {"answerable": item.answerable, "gold_doc_ids": item.gold_doc_ids}
            for item in load() if item.reviewed}
    rows = [r for k, r in rows.items() if k[0] in gold]
    summary = score(rows, gold)
    offline = json.loads(OFFLINE.read_text(encoding="utf-8"))["summary"] if OFFLINE.exists() else {}
    REPORT.write_text(json.dumps({"run": path.name, "checked_at": time.strftime("%Y-%m-%d %H:%M"),
                                  "summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")

    labels = [
        ("n", "answers", "{}"),
        ("errors", "errors", "{}"),
        ("answered_with_gold_citation", "answered, citing a correct page", "{:.1%}"),
        ("answered_citing_wrong_page", "answered, citing only other pages", "{:.1%}"),
        ("said_not_found", "said 'not found' (answerable)", "{:.1%}"),
        ("correct_abstention_rate", "refused unanswerable questions", "{:.0%}"),
        ("gold_page_in_top5", "correct page among the 5 read", "{:.1%}"),
        ("quotes_dropped_rate", "proposed quotes that failed the check", "{:.1%}"),
        ("answer_language_ok", "answered in the question's language", "{:.0%}"),
        ("seconds_mean", "seconds per answer", "{:.1f}"),
        ("cost_per_question_usd", "cost per question (USD)", "{:.4f}"),
    ]
    print(f"\nLive pipeline ({path.name})")
    print(f"{'':42}{'FR':>9}{'EN':>9}   offline eval FR / EN")
    for key, label, fmt in labels:
        cells = ["–" if summary[lang].get(key) is None else fmt.format(summary[lang][key]) for lang in ("fr", "en")]
        before = ""
        if key in ("answered_with_gold_citation", "correct_abstention_rate") and offline:
            before = f"   {offline['fr'][key]:.1%} / {offline['en'][key]:.1%}"
        print(f"  {label:<40}{cells[0]:>9}{cells[1]:>9}{before}")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only N questions (smoke test)")
    parser.add_argument("--workers", type=int, default=4, help="questions in parallel")
    parser.add_argument("--score-only", action="store_true", help="re-score the saved run, no API calls")
    args = parser.parse_args()
    from guichet.service import MODEL, PROMPT

    path = generations_path(MODEL, "live", PROMPT)
    if not args.score_only:
        path = run(args.limit, args.workers)
    if not path.exists():
        raise SystemExit(f"no run found at {path}")
    report(path)


if __name__ == "__main__":
    main()
