"""Run the retrieval evaluation and report the cross-lingual gap.

Usage:
    python -m guichet.run_eval                       # hybrid, all modes compared
    python -m guichet.run_eval --mode dense
    python -m guichet.run_eval --include-unreviewed  # for development only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunk import load_chunks
from .golden import load as load_golden
from .index import Retriever
from .metrics import QueryResult, _doc_sequence, aggregate, by_language, format_table


def run(mode: str, include_unreviewed: bool, top_k: int,
        chunks_path: Path, golden_path: Path,
        foreign_sparse_weight: float = 0.0) -> list[QueryResult]:
    items = load_golden(golden_path)
    if not include_unreviewed:
        items = [i for i in items if i.reviewed]
    if not items:
        raise SystemExit(
            "No reviewed golden items. Review the seed set, or pass "
            "--include-unreviewed to see provisional numbers."
        )

    retriever = Retriever(load_chunks(chunks_path), mode=mode,
                          foreign_sparse_weight=foreign_sparse_weight)

    results: list[QueryResult] = []
    for item in items:
        for language, query in (("fr", item.question_fr), ("en", item.question_en)):
            hits = retriever.search(query, top_k=top_k)
            results.append(
                QueryResult(
                    item_id=item.item_id,
                    language=language,
                    query=query,
                    gold=set(item.gold_doc_ids),
                    ranked_docs=_doc_sequence(hits),
                    answerable=item.answerable,
                )
            )
    return results


def failures(results: list[QueryResult], k: int = 5) -> list[QueryResult]:
    """Queries whose correct document never made it into the top k."""
    return [r for r in results
            if r.answerable and r.gold and not set(r.ranked_docs[:k]) & r.gold]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["dense", "bm25", "hybrid", "hybrid-lang", "all"])
    parser.add_argument("--foreign-sparse-weight", type=float, default=0.0,
                        help="BM25 weight for non-French queries in hybrid-lang mode")
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunks", type=Path, default=Path("data/corpus/chunks.jsonl"))
    parser.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/retrieval.json"))
    args = parser.parse_args()

    modes = ["bm25", "dense", "hybrid", "hybrid-lang"] if args.mode == "all" else [args.mode]
    report: dict[str, dict] = {"_meta": {
        "include_unreviewed": args.include_unreviewed,
        "provisional": args.include_unreviewed,
        "foreign_sparse_weight": args.foreign_sparse_weight,
    }}

    for mode in modes:
        results = run(mode, args.include_unreviewed, args.top_k,
                      args.chunks, args.golden, args.foreign_sparse_weight)
        per_language = by_language(results)
        overall = aggregate(results)

        print(f"\n{'=' * 52}\n  {mode.upper()}   ({len(results)} queries)\n{'=' * 52}")
        print(format_table(per_language))

        missed = failures(results)
        if missed:
            print(f"\n  missed entirely (gold doc not in top 5): {len(missed)}")
            for result in missed[:8]:
                print(f"    [{result.language}] {result.item_id}: {result.query[:62]}")
                print(f"         wanted {sorted(result.gold)}, got {result.ranked_docs[:3]}")

        report[mode] = {
            "overall": overall,
            "by_language": per_language,
            "n_queries": len(results),
            "n_missed_at_5": len(missed),
            "misses": [
                {"item_id": r.item_id, "language": r.language, "query": r.query,
                 "gold": sorted(r.gold), "got": r.ranked_docs[:5]}
                for r in missed
            ],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
