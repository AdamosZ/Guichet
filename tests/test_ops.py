import json

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
