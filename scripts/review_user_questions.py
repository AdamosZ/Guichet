"""Turn real users' questions into candidates for the answer key.

    python -m scripts.review_user_questions                 # local user_logs/
    python -m scripts.review_user_questions --hf adamosz/guichet-logs

Shows each logged question with the app's answer and any feedback (👎 first),
and lets you mark it:
    k  keep as a candidate golden item (appended to evals/candidates_from_users.jsonl)
    s  skip      q  quit

Kept questions then go through the same pipeline as the forum questions:
answer written from the service-public page, independent check, human review.
They never enter the answer key directly, so a wrong app answer can't become
a "correct" label.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals/candidates_from_users.jsonl"


def load_logs(folder: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    answers, feedback = [], {}
    for path in sorted(folder.rglob("answers-*.jsonl")):
        answers += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for path in sorted(folder.rglob("feedback-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                feedback.setdefault(row["answer_id"], []).append(row)
    return answers, feedback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", help="private HF dataset holding the app logs")
    args = parser.parse_args()
    folder = ROOT / "user_logs"
    if args.hf:
        from huggingface_hub import snapshot_download

        from guichet.generate import _load_env

        _load_env()
        folder = Path(snapshot_download(args.hf, repo_type="dataset", token=os.environ.get("HF_TOKEN")))

    answers, feedback = load_logs(folder)
    done = set()
    if OUT.exists():
        done = {json.loads(line)["answer_id"] for line in OUT.read_text().splitlines() if line}
    todo = [a for a in answers if a["answer_id"] not in done and not a.get("error")]
    # Negative feedback first, then questions the app couldn't answer, then the rest.
    def priority(a):
        ratings = [f["rating"] for f in feedback.get(a["answer_id"], [])]
        return (0 if "down" in ratings else 1 if not a["found"] else 2)
    todo.sort(key=priority)
    print(f"{len(answers)} logged answers, {len(todo)} not yet reviewed\n")

    for a in todo:
        fb = feedback.get(a["answer_id"], [])
        print("=" * 90)
        print(f"[{a['language']}] {a['question']}")
        print(f"  found: {a['found']}   sources: {', '.join(a['sources']) or '-'}")
        print(f"  answer: {a['answer'][:400]}")
        for f in fb:
            print(f"  feedback: {f['rating']} {f.get('comment', '')}")
        choice = ""
        while choice not in {"k", "s", "q"}:
            choice = input("  [k]eep as candidate  [s]kip  [q]uit > ").strip().lower()
        if choice == "q":
            break
        if choice == "k":
            with OUT.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"answer_id": a["answer_id"], "question": a["question"],
                                         "language": a["language"], "app_found": a["found"],
                                         "feedback": [f["rating"] for f in fb]}, ensure_ascii=False) + "\n")
    print(f"\ncandidates so far: {sum(1 for _ in OUT.open()) if OUT.exists() else 0} in {OUT.name}")


if __name__ == "__main__":
    main()
