import json
from datetime import UTC

from scripts.freshness import _fingerprint, to_markdown
from scripts.review_user_questions import load_logs


def test_fingerprint_ignores_whitespace_and_punctuation_only():
    assert _fingerprint("Le récépissé est valable 4 mois.") == _fingerprint("Le  récépissé\nest valable 4 mois")
    assert _fingerprint("valable 4 mois") != _fingerprint("valable 6 mois")


def test_freshness_report_flags_redated_pages():
    report = {"checked_at": "now", "checked": 2, "missing": [],
              "changed": [{"doc_id": "F1", "url": "u", "title": "t", "verified_before": "1 mai",
                           "verified_now": "1 juin", "sample_diff": ["-a", "+b"]}],
              "affected_items": [{"item_id": "x-1", "pages": ["F1"], "question_en": "q?"}]}
    md = to_markdown(report)
    assert "re-dated" in md and "x-1" in md and "1 golden items to re-review" in md


def test_review_loader_joins_feedback(tmp_path):
    (tmp_path / "answers-2026-10-16.jsonl").write_text(json.dumps({"answer_id": "a1", "question": "q"}) + "\n")
    (tmp_path / "feedback-2026-10-16.jsonl").write_text(json.dumps({"answer_id": "a1", "rating": "down"}) + "\n")
    answers, feedback = load_logs(tmp_path)
    assert answers[0]["answer_id"] == "a1" and feedback["a1"][0]["rating"] == "down"


def test_log_retention(tmp_path):
    from datetime import date

    from guichet.logbook import Logbook, is_expired

    today = date(2026, 10, 19)
    assert is_expired("answers-2026-07-20.jsonl", today, 90)
    assert not is_expired("answers-2026-07-21.jsonl", today, 90)
    assert not is_expired("notes.jsonl", today, 90)
    (tmp_path / "answers-2020-01-01.jsonl").write_text("{}\n")
    (tmp_path / "answers-2999-01-01.jsonl").write_text("{}\n")
    Logbook(tmp_path)  # purges on start
    assert [p.name for p in tmp_path.iterdir()] == ["answers-2999-01-01.jsonl"]


def test_weekly_report_counts_last_seven_days():
    from datetime import datetime

    from scripts.weekly_report import summarise, to_markdown

    now = datetime(2026, 10, 19, 12, tzinfo=UTC)
    row = {"language": "en", "sources": [], "tokens_in": 50_000, "tokens_out": 400, "error": ""}
    answers = [
        {**row, "answer_id": "a", "time": "2026-10-18T10:00:00Z", "question": "q1", "found": True, "sources": ["F1"]},
        {**row, "answer_id": "b", "time": "2026-10-17T10:00:00Z", "question": "Which bank?", "found": False},
        {**row, "answer_id": "c", "time": "2026-10-16T10:00:00Z", "question": "q3", "found": False, "error": "Timeout"},
        {**row, "answer_id": "old", "time": "2026-10-01T10:00:00Z", "question": "old", "found": True},
    ]
    feedback = {"a": [{"rating": "down", "comment": "missing the deadline"}], "b": [{"rating": "up"}]}
    s = summarise(answers, feedback, now=now)
    assert (s["questions"], s["answered"], s["not_found"], s["errors"]) == (3, 1, 1, 1)
    assert (s["thumbs_up"], s["thumbs_down"]) == (1, 1)
    assert s["cost_usd"] == round(3 * (0.05 * 0.10 + 0.0004 * 0.50), 2)
    md = to_markdown(s)
    assert "Which bank?" in md and "missing the deadline" in md and "failed" in md
    assert "No questions" in to_markdown(summarise([], {}, now=now))
