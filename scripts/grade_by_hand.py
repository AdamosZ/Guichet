"""Grade a fixed sample of generated answers by hand, blind to the judge.

    python -m scripts.grade_by_hand              # pages5 answers, 24 of them

Shows the question, the reviewed reference answer and the system's answer.
Press c (correct), p (partially correct: nothing false but a key fact or
condition missing), i (incorrect), s (skip), q (quit). Saves after every grade
to evals/human_grades.jsonl. The sample is fixed (seeded), half French and half
English, and the judge's verdicts are never shown here, so your grades can be
used to measure how far the judge can be trusted.
"""

from __future__ import annotations

import json
import random
import sys
import textwrap
from pathlib import Path

from guichet.judge import load_samples

OUT = Path("evals/human_grades.jsonl")
N_PER_LANGUAGE = 12


def pick(variant: str) -> list:
    samples = load_samples(variant)
    rng = random.Random(2026)
    chosen = []
    for language in ("fr", "en"):
        pool = [s for s in samples if s.metadata["language"] == language]
        rng.shuffle(pool)
        taken_items = {s.metadata["item_id"] for s in chosen}
        # Prefer items not already picked in the other language, for variety.
        pool.sort(key=lambda s: s.metadata["item_id"] in taken_items)
        chosen.extend(pool[:N_PER_LANGUAGE])
    return chosen


def wrap(label: str, text: str) -> str:
    return f"  {label:<11}" + textwrap.fill(text, width=86, subsequent_indent=" " * 13)


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "pages5"
    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    sample = [s for s in pick(variant) if s.id not in done]
    print(f"{len(done)} graded, {len(sample)} to go\n")
    for n, s in enumerate(sample, start=1):
        print("=" * 98)
        print(f"[{n}/{len(sample)}] {s.id}")
        print(wrap("question", s.input))
        print(wrap("reference", s.target))
        print(wrap("system", s.metadata["submitted"]))
        choice = ""
        while choice not in {"c", "p", "i", "s", "q"}:
            choice = input("\n  [c]orrect  [p]artial  [i]ncorrect  [s]kip  [q]uit > ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        row = {"id": s.id, "variant": variant, "grade": choice.upper(), "by": "adam"}
        with OUT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    total = len(OUT.read_text().splitlines()) if OUT.exists() else 0
    print(f"\nsaved. {total} hand grades in {OUT}")


if __name__ == "__main__":
    main()
