"""Weekly health report for the live app, in one Markdown page.

    python -m scripts.weekly_report                          # local user_logs/
    python -m scripts.weekly_report --hf adamosz/guichet-logs
    python -m scripts.weekly_report --hf adamosz/guichet-logs --purge

Covers the last 7 days: questions asked, how many were answered / not found /
failed, 👍/👎, estimated cost and the latest freshness check. With --details
(on your own machine only) it also lists the questions Guichet couldn't answer
and the 👎 comments. Writes reports/weekly.md (and reports/weekly.json with
--details).

The default output holds counts only, because the GitHub issue it becomes is
public: what students typed never leaves the private logs dataset.

--purge also deletes log files older than the retention period (90 days) from
the private dataset; the app already deletes them from its own disk.

The weekly GitHub Action (ops/weekly.yml) runs this and opens an issue, so
the report arrives by e-mail like any GitHub notification.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from guichet.logbook import RETENTION_DAYS, is_expired
from scripts.review_user_questions import load_logs

ROOT = Path(__file__).resolve().parents[1]
# USD per million tokens (input, output). Rows without a model are from before
# models were logged, when the app always used Claude Haiku 5.5.
# Claude: platform.claude.com/docs/en/about-claude/pricing (19 Oct 2026).
# Hugging Face providers: router.huggingface.co/v1/models (26 Oct 2026).
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "openai/gpt-oss-120b:deepinfra": (0.037, 0.17),
    "openai/gpt-oss-120b:novita": (0.05, 0.25),
    "openai/gpt-oss-20b:deepinfra": (0.03, 0.14),
    "deepseek-ai/DeepSeek-V4-Flash:deepinfra": (0.09, 0.18),
}
DEFAULT_MODEL = "claude-haiku-5-5"


def _when(row: dict) -> datetime:
    return datetime.strptime(row["time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def summarise(answers: list[dict], feedback: dict[str, list[dict]], now: datetime | None = None,
              days: int = 7) -> dict:
    now = now or datetime.now(UTC)
    start = now - timedelta(days=days)
    week = [a for a in answers if _when(a) >= start]
    ratings = [f for a in week for f in feedback.get(a["answer_id"], [])]
    errors = [a for a in week if a.get("error")]
    found = [a for a in week if a.get("found") and not a.get("error")]
    missing = [a for a in week if not a.get("found") and not a.get("error")]
    cost, unpriced = 0.0, set()
    for a in week:
        model = a.get("model") or DEFAULT_MODEL
        if model not in PRICES:
            unpriced.add(model)
            continue
        price_in, price_out = PRICES[model]
        cost += a.get("tokens_in", 0) / 1e6 * price_in + a.get("tokens_out", 0) / 1e6 * price_out
    pages = Counter(p for a in found for p in a.get("sources", []))
    down = [(a, f) for a in week for f in feedback.get(a["answer_id"], []) if f["rating"] == "down"]
    return {
        "from": start.strftime("%Y-%m-%d"), "to": now.strftime("%Y-%m-%d"),
        "questions": len(week),
        "languages": dict(Counter(a.get("language", "?") for a in week)),
        "answered": len(found), "not_found": len(missing), "errors": len(errors),
        "thumbs_up": sum(f["rating"] == "up" for f in ratings),
        "thumbs_down": sum(f["rating"] == "down" for f in ratings),
        "cost_usd": round(cost, 2),
        "models": dict(Counter(a.get("model") or DEFAULT_MODEL for a in week)),
        "unpriced_models": sorted(unpriced),
        "top_pages": pages.most_common(5),
        "not_found_questions": [a["question"] for a in missing][:15],
        "down_comments": [{"question": a["question"], "comment": f.get("comment", "")} for a, f in down][:15],
        "error_samples": sorted({a["error"][:80] for a in errors})[:5],
    }


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "–"


def to_markdown(s: dict, freshness: dict | None = None, details: bool = False) -> str:
    """The report. Without `details` it holds counts only, never what anyone typed:
    the GitHub issue it becomes is public. Use details=True on your own machine."""
    n = s["questions"]
    rated = s["thumbs_up"] + s["thumbs_down"]
    out = [f"## Guichet weekly report, {s['from']} → {s['to']}", ""]
    if n == 0:
        out += ["No questions this week.", ""]
    else:
        langs = ", ".join(f"{k.upper()} {v}" for k, v in sorted(s["languages"].items()))
        out += [
            "| | |", "|---|---|",
            f"| Questions | **{n}** ({langs}) |",
            f"| Answered with a checked source | {s['answered']} ({_pct(s['answered'], n)}) |",
            f"| Not found | {s['not_found']} ({_pct(s['not_found'], n)}) |",
            f"| Errors | {s['errors']} ({_pct(s['errors'], n)}) |",
            f"| Feedback | 👍 {s['thumbs_up']} · 👎 {s['thumbs_down']} ({_pct(s['thumbs_down'], rated)} negative) |",
            f"| Estimated cost | ${s['cost_usd']:.2f}"
            + (f" (no price for {', '.join(s['unpriced_models'])})" if s.get("unpriced_models") else "") + " |",
            f"| Model | {', '.join(f'{m} ({k})' for m, k in s.get('models', {}).items())} |",
            "",
        ]
        if s["errors"] and s["errors"] / n > 0.05:
            kinds = sorted({e.split(":")[0] for e in s["error_samples"]})
            out += [f"⚠️ **{_pct(s['errors'], n)} of questions failed** ({', '.join(kinds)}).", ""]
        if details:
            if s["not_found_questions"]:
                out += ["### Couldn't answer (candidates for new pages)"]
                out += [f"- {q}" for q in s["not_found_questions"]] + [""]
            if s["down_comments"]:
                out += ["### 👎 feedback"]
                out += [f"- {d['question']}" + (f" — _{d['comment']}_" if d["comment"] else "")
                        for d in s["down_comments"]] + [""]
        elif s["not_found_questions"] or s["down_comments"]:
            out += ["_Unanswered questions and 👎 comments stay private: see them with "
                    "`python -m scripts.weekly_report --hf <logs dataset> --details`._", ""]
        if s["top_pages"]:
            out += ["### Most cited pages"]
            out += [f"- [{doc}](https://www.service-public.gouv.fr/particuliers/vosdroits/{doc}) × {k}"
                    for doc, k in s["top_pages"]] + [""]
    if freshness:
        out += ["### Official pages",
                f"Last check {freshness['checked_at']}: {len(freshness['changed'])} of {freshness['checked']} "
                f"pages changed, {len(freshness['affected_items'])} test answers to re-check, "
                f"{len(freshness['missing'])} unreachable.", ""]
    out.append("_Run `python -m scripts.review_user_questions` to turn real questions into test candidates._")
    return "\n".join(out)


def purge_dataset(repo: str, token: str | None) -> list[str]:
    from huggingface_hub import CommitOperationDelete, HfApi

    api = HfApi(token=token)
    old = [f for f in api.list_repo_files(repo, repo_type="dataset") if is_expired(f.rsplit("/", 1)[-1])]
    if old:
        api.create_commit(repo, repo_type="dataset", commit_message=f"Delete logs older than {RETENTION_DAYS} days",
                          operations=[CommitOperationDelete(path_in_repo=f) for f in old])
    return old


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", help="private HF dataset holding the app logs")
    parser.add_argument("--logs", default=str(ROOT / "user_logs"), help="local log folder")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--purge", action="store_true", help="delete expired logs from the dataset (needs --hf)")
    parser.add_argument("--details", action="store_true",
                        help="include what users typed (private use only: never for the public issue)")
    args = parser.parse_args()

    folder = Path(args.logs)
    token = None
    if args.hf:
        from huggingface_hub import snapshot_download

        from guichet.generate import _load_env

        _load_env()
        token = os.environ.get("HF_TOKEN")
        if args.purge:
            removed = purge_dataset(args.hf, token)
            print(f"purged {len(removed)} expired log files from {args.hf}")
        folder = Path(snapshot_download(args.hf, repo_type="dataset", token=token, allow_patterns=["logs/*"]))

    answers, feedback = load_logs(folder) if folder.exists() else ([], {})
    stats = summarise(answers, feedback, days=args.days)
    fresh_path = ROOT / "reports/freshness.json"
    freshness = json.loads(fresh_path.read_text(encoding="utf-8")) if fresh_path.exists() else None

    (ROOT / "reports").mkdir(exist_ok=True)
    if args.details:
        (ROOT / "reports/weekly.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = to_markdown(stats, freshness, details=args.details)
    (ROOT / "reports/weekly.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
