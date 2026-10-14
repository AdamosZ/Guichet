"""How far can the judge be trusted? Compare its grades with the hand grades.

    python -m scripts.judge_agreement                # latest judge log in ./logs

Prints exact agreement, Cohen's kappa (agreement corrected for chance), the
confusion matrix, and the judge's overall verdicts per language. Writes
reports/judge.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

GRADES = ("C", "P", "I")


def judge_grades(log_path: str | None = None) -> dict[str, str]:
    from inspect_ai.log import list_eval_logs, read_eval_log

    if log_path is None:
        logs = list_eval_logs("logs")
        if not logs:
            raise SystemExit("no Inspect logs found in ./logs — run the judge first")
        log_path = logs[0].name  # most recent first
    log = read_eval_log(log_path)
    out = {}
    for sample in log.samples or []:
        score = next(iter((sample.scores or {}).values()), None)
        if score is not None and score.value in GRADES:
            out[str(sample.id)] = score.value
    return out


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[g] * right[g] for g in GRADES) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main() -> None:
    judge = judge_grades(sys.argv[1] if len(sys.argv) > 1 else None)
    human = {}
    path = Path("evals/human_grades.jsonl")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            human[row["id"]] = row["grade"]

    pairs = [(human[i], judge[i]) for i in human if i in judge]
    report: dict = {"n_judged": len(judge)}
    for language in ("fr", "en"):
        grades = Counter(g for i, g in judge.items() if i.endswith(f"::{language}"))
        total = sum(grades.values())
        report[f"judge_{language}"] = {g: round(grades[g] / total, 3) for g in GRADES} | {"n": total}

    if pairs:
        agree = sum(a == b for a, b in pairs) / len(pairs)
        report["agreement"] = {"n": len(pairs), "exact": round(agree, 3),
                               "cohen_kappa": round(cohen_kappa(pairs), 3),
                               "confusion_human_x_judge": {f"{h}->{j}": c for (h, j), c
                                                           in sorted(Counter(pairs).items())}}
    Path("reports/judge.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
