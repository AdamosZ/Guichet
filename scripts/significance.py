"""Paired bootstrap 95% confidence intervals for the headline comparisons.

    python -m scripts.significance

Each question exists in both languages, so comparisons are paired by item:
resampling items (not queries) keeps the French and English versions together.
"""

from __future__ import annotations

import random
from pathlib import Path

from guichet.run_eval import run

CHUNKS = Path("data/corpus/chunks.jsonl")
GOLDEN = Path("evals/golden.jsonl")
MODES = ("bm25", "bm25-trans", "dense", "hybrid", "hybrid-lang", "hybrid-trans")


def per_item(results, language: str, metric: str) -> dict[str, float]:
    return {r.item_id: r.scores()[metric] for r in results
            if r.language == language and r.answerable and r.gold}


def bootstrap(a: dict[str, float], b: dict[str, float], n: int = 5000,
              seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    ids = sorted(a)
    mean = sum(a[i] - b[i] for i in ids) / len(ids)
    draws = sorted(
        sum(a[i] - b[i] for i in sample) / len(sample)
        for sample in ([rng.choice(ids) for _ in ids] for _ in range(n))
    )
    return mean, draws[int(0.025 * n)], draws[int(0.975 * n)]


def main() -> None:
    results = {mode: run(mode, False, 10, CHUNKS, GOLDEN) for mode in MODES}
    print("English minus French, same questions (negative = English worse)")
    for mode in MODES:
        for metric in ("hit@1", "recall@5"):
            mean, lo, hi = bootstrap(per_item(results[mode], "en", metric),
                                     per_item(results[mode], "fr", metric))
            print(f"  {mode:<12} {metric:<9} {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print("\nEnglish queries, mode vs mode")
    for a, b in (("hybrid-trans", "hybrid"), ("hybrid-lang", "hybrid"), ("hybrid", "dense")):
        for metric in ("hit@1", "recall@5", "mrr"):
            mean, lo, hi = bootstrap(per_item(results[a], "en", metric),
                                     per_item(results[b], "en", metric))
            print(f"  {a} − {b:<8} {metric:<9} {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
