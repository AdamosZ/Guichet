"""Compare answer-writing models side by side on the same questions and pages.

    python -m scripts.compare_models claude-haiku-5-5:v2 openai/gpt-oss-120b:deepinfra:v2s

Each argument is MODEL:PROMPT (the prompt is the last ':' part). Only the
(question, language) pairs that every run answered are compared, so a spot
check and a full run can sit side by side. Everything here is computed by
code from saved runs: no API calls, no cost. Correctness is judged
separately (guichet/judge.py).
"""

from __future__ import annotations

import argparse
import json

from guichet.generate import CONTEXT_VARIANTS, contexts_path, generations_path, score_row, summarise
from guichet.golden import load
from scripts.weekly_report import PRICES

ROWS = [
    ("answered_with_gold_citation", "Answered, citing a correct page", "{:.0%}"),
    ("correct_abstention_rate", "Refused unanswerable questions", "{:.0%}"),
    ("citation_valid_rate", "Citations point to pages it was given", "{:.0%}"),
    ("quote_faithful_rate", "Quotes are word for word", "{:.0%}"),
    ("abstained_evidence_in_context", "Gave up although the answer was in front of it", "{:.0%}"),
    ("answer_language_ok", "Answered in the question's language", "{:.0%}"),
    ("parse_errors", "Malformed replies", "{}"),
]


def _run(spec: str, variant: str) -> tuple[str, str, dict]:
    model, prompt = spec.rsplit(":", 1)
    path = generations_path(model, variant, prompt)
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["item_id"], row["language"])] = row
    return model, prompt, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="MODEL:PROMPT, e.g. claude-haiku-5-5:v2")
    parser.add_argument("--context", default="pages5", choices=sorted(CONTEXT_VARIANTS))
    args = parser.parse_args()

    ctx = {(r["item_id"], r["language"]): r
           for r in map(json.loads, contexts_path(args.context).read_text(encoding="utf-8").splitlines())}
    evidence = {item.item_id: item.evidence for item in load()}
    runs = [_run(spec, args.context) for spec in args.runs]
    common = set(ctx)
    for _model, _prompt, rows in runs:
        common &= set(rows)
    if not common:
        raise SystemExit("These runs have no question in common.")

    table = {}
    for model, prompt, rows in runs:
        scores = [score_row(rows[key], ctx[key], evidence.get(key[0])) for key in sorted(common)]
        summary = summarise(scores)
        seconds = [rows[k]["seconds"] for k in common if rows[k].get("seconds") is not None]
        table[f"{model} ({prompt})"] = (summary, seconds, model)

    names = list(table)
    width = max(len(n) for n in names)
    print(f"{len(common)} answers in common ({sum(1 for k in common if k[1] == 'fr')} FR, "
          f"{sum(1 for k in common if k[1] == 'en')} EN)\n")
    for language in ("fr", "en"):
        print(f"[{language.upper()}]")
        print(" " * 50 + "  ".join(n[:width].ljust(width) for n in names))
        for key, label, fmt in ROWS:
            cells = []
            for name in names:
                value = table[name][0][language].get(key)
                cells.append(("–" if value is None else fmt.format(value)).ljust(width))
            print(f"  {label:<48}" + "  ".join(cells))
        print()
    print("Speed and cost per question")
    for name in names:
        _summary, seconds, model = table[name]
        speed = f"{sum(seconds) / len(seconds):.1f}s average" if seconds else "not recorded"
        price = PRICES.get(model)
        cost = f"≈ ${50_000 / 1e6 * price[0] + 800 / 1e6 * price[1]:.4f} (50k tokens in, 800 out)" if price else "no price"
        print(f"  {name:<{width}}  {speed:<16}  {cost}")


if __name__ == "__main__":
    main()
