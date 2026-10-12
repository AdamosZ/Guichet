import math

from guichet.metrics import (
    QueryResult,
    aggregate,
    by_language,
    format_table,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_and_hit():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"b", "z"}, 2) == 0.5
    assert hit_at_k(ranked, {"c"}, 2) == 0.0
    assert hit_at_k(ranked, {"c"}, 3) == 1.0


def test_empty_gold_is_nan_not_zero():
    # Unanswerable items must not silently score 0 and drag the mean down.
    assert math.isnan(recall_at_k(["a"], set(), 5))
    assert math.isnan(ndcg_at_k(["a"], set(), 5))


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "gold"], {"gold"}) == 1 / 3
    assert reciprocal_rank(["x"], {"gold"}) == 0.0


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k(["g1", "g2", "x"], {"g1", "g2"}, 10) == 1.0
    assert 0 < ndcg_at_k(["x", "g1"], {"g1"}, 10) < 1


def _result(lang, ranked, gold, answerable=True):
    return QueryResult("i", lang, "q", set(gold), ranked, answerable)


def test_aggregate_skips_unanswerable():
    results = [_result("en", ["a"], {"a"}), _result("en", ["a"], set(), answerable=False)]
    assert aggregate(results)["hit@1"] == 1.0


def test_by_language_and_gap_direction():
    results = [_result("fr", ["a"], {"a"}), _result("en", ["b", "a"], {"a"})]
    per_lang = by_language(results)
    assert per_lang["fr"]["hit@1"] == 1.0 and per_lang["en"]["hit@1"] == 0.0
    table = format_table(per_lang)
    # French is the baseline column, so the gap is negative when English is worse.
    assert table.splitlines()[0].split()[1:3] == ["fr", "en"]
    assert "-100.0%" in table
