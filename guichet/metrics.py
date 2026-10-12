"""Retrieval metrics, reported separately for each query language.

Everything here operates at document level. Retrieval returns chunks, but the
answer key labels documents, because asking a human to decide which of a
document's thirty chunks is "the" right one produces noisy labels and a metric
that moves when you change the chunk size rather than when retrieval improves.

Three metrics, because they answer different questions:

- recall@k: did the right document make it into the window the model sees?
  This is the ceiling on everything downstream — a document not retrieved can
  never be cited.
- MRR: how high did it land? Separates "barely scraped in at rank 9" from
  "first result", which recall@k cannot see.
- nDCG@k: handles the items with several correct documents, where rewarding
  all of them matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _doc_sequence(hits) -> list[str]:
    """Collapse a chunk ranking into a document ranking, first occurrence wins."""
    seen: set[str] = set()
    ordered: list[str] = []
    for hit in hits:
        doc_id = hit.chunk.doc_id
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def recall_at_k(ranked_docs: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(set(ranked_docs[:k]) & gold) / len(gold)


def hit_at_k(ranked_docs: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return 1.0 if set(ranked_docs[:k]) & gold else 0.0


def reciprocal_rank(ranked_docs: list[str], gold: set[str]) -> float:
    for position, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in gold:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked_docs: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranked_docs[:k], start=1)
        if doc_id in gold
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return gain / ideal if ideal else float("nan")


@dataclass
class QueryResult:
    item_id: str
    language: str
    query: str
    gold: set[str]
    ranked_docs: list[str]
    answerable: bool

    def scores(self, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in ks:
            out[f"recall@{k}"] = recall_at_k(self.ranked_docs, self.gold, k)
            out[f"hit@{k}"] = hit_at_k(self.ranked_docs, self.gold, k)
        out["mrr"] = reciprocal_rank(self.ranked_docs, self.gold)
        out["ndcg@10"] = ndcg_at_k(self.ranked_docs, self.gold, 10)
        return out


def aggregate(results: list[QueryResult],
              ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    """Mean of each metric over the answerable queries."""
    scored = [r for r in results if r.answerable and r.gold]
    if not scored:
        return {}
    keys = scored[0].scores(ks).keys()
    all_scores = [r.scores(ks) for r in scored]
    return {key: sum(s[key] for s in all_scores) / len(all_scores) for key in keys}


def by_language(results: list[QueryResult]) -> dict[str, dict[str, float]]:
    languages = sorted({r.language for r in results})
    return {lang: aggregate([r for r in results if r.language == lang])
            for lang in languages}


def gap_report(per_language: dict[str, dict[str, float]],
               baseline: str = "fr", compare: str = "en") -> dict[str, float]:
    """The headline number: how much worse is retrieval in the other language."""
    if baseline not in per_language or compare not in per_language:
        return {}
    gaps: dict[str, float] = {}
    for metric, base_value in per_language[baseline].items():
        other = per_language[compare].get(metric)
        if other is None or base_value == 0:
            continue
        gaps[metric] = (other - base_value) / base_value
    return gaps


def format_table(per_language: dict[str, dict[str, float]],
                 metrics: tuple[str, ...] = ("hit@1", "hit@3", "recall@5", "mrr", "ndcg@10")) -> str:
    # Corpus language first, so the gap reads "how much worse is English".
    languages = sorted(per_language, key=lambda lang: (lang != "fr", lang))
    header = f"{'metric':<12}" + "".join(f"{lang:>10}" for lang in languages)
    if len(languages) == 2:
        header += f"{'gap':>10}"
    lines = [header, "-" * len(header)]

    for metric in metrics:
        row = f"{metric:<12}"
        values = [per_language[lang].get(metric, float('nan')) for lang in languages]
        row += "".join(f"{v:>10.3f}" for v in values)
        if len(values) == 2 and values[0]:
            row += f"{(values[1] - values[0]) / values[0]:>9.1%}"
        lines.append(row)
    return "\n".join(lines)
