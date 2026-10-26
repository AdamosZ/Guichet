"""Usage log for the live app: what was asked, what came back, and feedback.

Privacy rules, applied before anything is written:
- no IP addresses, names or accounts are stored;
- e-mail addresses, phone numbers and long digit runs (social security,
  foreigner file and bank numbers) are replaced by placeholders.

Records go to a local JSONL folder, one file per kind and day. If HF_LOG_REPO
and HF_TOKEN are set (on the hosted app), the folder is synced every 10 minutes
to a *private* Hugging Face dataset so the logs survive restarts.

Retention: files older than RETENTION_DAYS (90) are deleted locally at start-up
and once a day; scripts/weekly_report.py --purge deletes them from the dataset.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

LOG_DIR = Path(os.environ.get("GUICHET_LOG_DIR", "user_logs"))
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "90"))
_FILE_DATE = re.compile(r"-(\d{4}-\d{2}-\d{2})\.jsonl$")


def is_expired(name: str, today: date | None = None, days: int = RETENTION_DAYS) -> bool:
    """True for a log file (…-YYYY-MM-DD.jsonl) older than the retention period."""
    match = _FILE_DATE.search(name)
    if not match:
        return False
    return date.fromisoformat(match.group(1)) < (today or date.today()) - timedelta(days=days)


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+|00)\d[\d .-]{7,}\d|\b0\d(?:[ .-]?\d{2}){4}\b")
_DIGITS = re.compile(r"\b\d[\d ]{8,}\d\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,}\b")


def scrub(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _IBAN.sub("[iban]", text)
    text = _PHONE.sub("[phone]", text)
    return _DIGITS.sub("[number]", text)


class Logbook:
    def __init__(self, folder: Path = LOG_DIR):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._scheduler = None
        repo, token = os.environ.get("HF_LOG_REPO"), os.environ.get("HF_TOKEN")
        if repo and token:
            from huggingface_hub import CommitScheduler

            self._scheduler = CommitScheduler(repo_id=repo, repo_type="dataset", folder_path=folder,
                                              path_in_repo="logs", every=10, private=True, token=token)
        self._purged_on = ""
        self.purge()

    def purge(self) -> list[str]:
        """Delete local log files past the retention period."""
        removed = []
        for path in self.folder.glob("*.jsonl"):
            if is_expired(path.name):
                path.unlink(missing_ok=True)
                removed.append(path.name)
        self._purged_on = time.strftime("%Y-%m-%d")
        return removed

    def _write(self, kind: str, record: dict) -> None:
        if self._purged_on != time.strftime("%Y-%m-%d"):
            self.purge()
        path = self.folder / f"{kind}-{time.strftime('%Y-%m-%d')}.jsonl"
        lock = self._scheduler.lock if self._scheduler else self._lock
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def question(self, answer: dict) -> str:
        answer_id = uuid.uuid4().hex[:12]
        self._write("answers", {
            "answer_id": answer_id, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": scrub(answer["question"]), "language": answer["language"],
            "found": answer["found"], "answer": scrub(answer["text"]),
            "sources": [s["doc_id"] for s in answer["sources"]], "pages_seen": answer["pages_seen"],
            "dropped_quotes": answer["dropped_quotes"], "error": answer["error"],
            "tokens_in": answer["tokens_in"], "tokens_out": answer["tokens_out"],
            "model": answer.get("model", ""),
        })
        return answer_id

    def feedback(self, answer_id: str, rating: str, comment: str = "") -> None:
        self._write("feedback", {
            "answer_id": answer_id, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rating": rating, "comment": scrub(comment)[:1000],
        })
