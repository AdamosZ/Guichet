"""HTTP API + web page for the live app.

    python app.py                     # http://127.0.0.1:7860
    GUICHET_DEMO=1 python app.py      # no API key: canned answers, for UI work

Endpoints:
    GET  /                 the page (static/index.html)
    GET  /privacy          privacy notice (static/privacy.html)
    POST /api/ask          {"question": "..."} -> answer with verified sources
    POST /api/feedback     {"answer_id": "...", "rating": "up"|"down", "comment": "..."}
    GET  /api/health

Environment:
    ANTHROPIC_API_KEY      required unless GUICHET_DEMO=1 (.env locally, a secret when hosted)
    DAILY_CAP              max questions per day for the whole app (default 300)
    PER_USER_HOUR          max questions per visitor per hour (default 15)
    HF_LOG_REPO, HF_TOKEN  optional: sync anonymised logs to a private HF dataset
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from guichet.generate import _load_env
from guichet.langid import detect
from guichet.logbook import Logbook
from guichet.service import Answer, Source, answer_to_dict

ROOT = Path(__file__).resolve().parent
_load_env()
DAILY_CAP = int(os.environ.get("DAILY_CAP", "300"))
PER_USER_HOUR = int(os.environ.get("PER_USER_HOUR", "15"))
FEEDBACK_PER_HOUR = int(os.environ.get("FEEDBACK_PER_HOUR", "30"))
DEMO = os.environ.get("GUICHET_DEMO") == "1"


class Demo:
    """Stands in for the model when GUICHET_DEMO=1, with the three states the page must handle."""

    def answer(self, question: str) -> Answer:
        language = detect(question)
        lowered = question.lower()
        if "error" in lowered:
            return Answer(question=question, language=language, found=False, text="", error="demo error")
        if any(w in lowered for w in ("lille", "bank", "banque", "best", "meilleur")):
            return Answer(question=question, language=language, found=False,
                          text="The official pages don't cover this.")
        text = {"en": ("It depends on why you received it.\n"
                       "- First application: valid for 4 or 6 months.\n"
                       "- Renewal of a residence card: valid for 3 months.\n"
                       "In exceptional cases a récépissé (the receipt for your application) can be renewed "
                       "for at most 3 more months."),
                "fr": ("Le récépissé de première demande est valable 4 ou 6 mois. Il peut exceptionnellement "
                       "être renouvelé pour 3 mois maximum. Pour un renouvellement, il est valable 3 mois.")}
        return Answer(question=question, language=language, found=True, text=text[language], sources=[
            Source("F15763", "Qu'est-ce qu'un récépissé de demande de titre de séjour ?",
                   "https://www.service-public.gouv.fr/particuliers/vosdroits/F15763",
                   "Le récépissé peut être exceptionnellement renouvelé pour 3 mois maximum.",
                   "17 novembre 2023"),
            Source("F2231", "Étudiant étranger en France : visa de long séjour ou titre de séjour",
                   "https://www.service-public.gouv.fr/particuliers/vosdroits/F2231",
                   "Vous recevez un récépissé valable 3 mois.", "01 août 2026"),
        ], pages_seen=["F15763", "F2231", "F16162", "F17319", "F12189"])

    def not_found_link(self, question: str) -> str:
        return "https://www.service-public.gouv.fr/particuliers/recherche?keyword=" + question[:80]


def _engine():
    if DEMO:
        return Demo()
    from guichet.service import Guichet

    return Guichet()


engine = _engine()
logbook = Logbook()


class Limits:
    """Per-visitor hourly limit (keyed by a salted hash, kept in memory only) and a daily cap."""

    def __init__(self, per_hour: int | None = None, daily_cap: int | None = None):
        self.lock = threading.Lock()
        self.salt = secrets.token_hex(8)
        self.seen: dict[str, deque] = defaultdict(deque)
        self.day, self.count = time.strftime("%Y-%m-%d"), 0
        self.per_hour = per_hour or PER_USER_HOUR
        self.daily_cap = daily_cap or DAILY_CAP

    def take(self, visitor: str) -> str | None:
        key = hashlib.sha256((self.salt + visitor).encode()).hexdigest()[:16]
        now = time.time()
        with self.lock:
            history = self.seen[key]
            while history and now - history[0] > 3600:
                history.popleft()
            if not history:
                del self.seen[key]  # keep memory bounded: forget idle visitors
                history = self.seen[key]
            if len(history) >= self.per_hour:
                return "slow"
            today = time.strftime("%Y-%m-%d")
            if today != self.day:
                self.day, self.count = today, 0
            if self.count >= self.daily_cap:
                return "cap"
            self.count += 1
            history.append(now)
        return None


limits = Limits()
feedback_limits = Limits(per_hour=FEEDBACK_PER_HOUR, daily_cap=10 * DAILY_CAP)
app = FastAPI(title="Guichet", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

# The page's script and styles are inline; fonts come from Google Fonts. Framing
# is allowed only by Hugging Face, which shows the Space inside its own page.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
        "base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'self' https://huggingface.co https://*.hf.space"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=600)


class FeedbackIn(BaseModel):
    answer_id: str = Field(pattern="^[0-9a-f]{12}$")
    rating: str = Field(pattern="^(up|down)$")
    comment: str = Field(default="", max_length=1000)


def _visitor(request: Request) -> str:
    """The visitor's IP address, as seen by the hosting proxy.

    Anyone can send their own X-Forwarded-For header, but proxies only ever
    append to it, so the left-most entries are the ones a visitor can fake. The
    right-most *public* address was added by the hosting infrastructure and is
    the real client; private addresses after it are internal proxy hops.
    """
    hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
    for hop in reversed(hops):
        try:
            address = ipaddress.ip_address(hop)
        except ValueError:
            continue
        if address.is_global:
            return str(address)
    return request.client.host if request.client else "unknown"


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


@app.get("/privacy")
def privacy():
    return FileResponse(ROOT / "static/privacy.html")


@app.get("/api/health")
def health():
    return {"ok": True, "demo": DEMO}


@app.post("/api/ask")
def ask(body: AskIn, request: Request):
    question = body.question.strip()
    language = detect(question)
    blocked = limits.take(_visitor(request))
    if blocked:
        return JSONResponse({"status": blocked, "language": language}, status_code=429)
    answer = engine.answer(question)
    data = answer_to_dict(answer)
    answer_id = logbook.question(data)
    status = "error" if answer.error else ("answered" if answer.found else "not_found")
    return {
        "status": status, "answer_id": answer_id, "language": answer.language,
        "question": question, "answer": answer.text if answer.found else "",
        "sources": data["sources"] if answer.found else [],
        "pages_searched": len(answer.pages_seen),
        "search_link": engine.not_found_link(question),
    }


@app.post("/api/feedback")
def feedback(body: FeedbackIn, request: Request):
    if feedback_limits.take(_visitor(request)):
        return JSONResponse({"ok": False}, status_code=429)
    logbook.feedback(body.answer_id, body.rating, body.comment)
    return {"ok": True}
