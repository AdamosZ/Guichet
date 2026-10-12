"""Fetch and clean the source corpus from service-public.gouv.fr.

The corpus is deliberately narrow: the administrative pages an international
student in France actually needs. Breadth would make the golden set
unlabellable in the time available, and a small corpus that is fully
hand-labelled is worth more than a large one that is not.

Every document keeps its URL and the "Vérifié le" date the page carries, so a
stale answer can always be traced back to a stale source.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from .normalize import display_clean

BASE = "https://www.service-public.gouv.fr"
FICHE_URL = BASE + "/particuliers/vosdroits/{ref}"

# Section pages to crawl for fiche links. Chosen to cover the situations a
# foreign student hits in their first year.
SEED_SECTIONS = [
    "N105",     # Entrée et séjour des étrangers
    "N19804",   # Visa
    "N110",     # Titres de séjour
    "N19806",   # Étudiants étrangers
    "N20360",   # Logement / aides
    "N19811",   # Travail des étrangers
    "N31931",
    "N32475",
    "N19805",
    "N19807",
    "N19808",
    "N19809",
    "N19810",
    "N19812",
]

# Pages worth having even if the crawl does not surface them.
EXTRA_FICHES = [
    "F16162",  # Visa de long séjour
    "F16163",
    "F2231",   # Titre de séjour étudiant
    "F17359",
    "F15763",  # Récépissé
    "F2718",
    "F12012",  # APL
    "F1280",
    "F13472",  # Sécurité sociale étudiant
    "F34006",
    "F2834",   # Travail étudiant étranger
    "F1361",
    "F2654",   # Compte bancaire
    "F31248",
    "F1988",   # Impôts / résidence fiscale
    "F2708",
]

DROP_SELECTORS = (
    "script", "style", "nav", "footer", "header", "aside", "form", "noscript",
    ".fr-modal", "[role=dialog]", ".fr-share", ".fr-breadcrumb", ".fr-sidemenu",
    ".fr-skiplinks", ".fr-notice", ".fr-follow", ".fr-btns-group",
    "[aria-hidden=true]", ".sp-favoris",
)

VERIFIED_RE = re.compile(r"Vérifié le\s+(\d{1,2}\s+\w+\s+\d{4})")


@dataclass
class Document:
    """One administrative page, cleaned and ready to chunk."""

    doc_id: str
    url: str
    title: str
    verified_on: str | None
    text: str
    n_chars: int
    fetched_at: str


def _client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": "guichet-research/0.1 (student eval project)"},
    )


def discover_fiches(client: httpx.Client, sections: list[str]) -> list[str]:
    """Collect fiche references (Fxxxxx) linked from the seed section pages."""
    found: set[str] = set()
    for section in sections:
        url = f"{BASE}/particuliers/vosdroits/{section}"
        try:
            response = client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"  ! {section}: {type(exc).__name__}")
            continue
        refs = re.findall(r"/particuliers/vosdroits/(F\d+)", response.text)
        found.update(refs)
        print(f"  {section}: {len(set(refs))} fiches")
        time.sleep(0.4)
    return sorted(found)


def extract(html: str, doc_id: str, url: str) -> Document | None:
    """Turn one fiche's HTML into a cleaned Document."""
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one("article") or soup.select_one("main")
    if body is None:
        return None

    for selector in DROP_SELECTORS:
        for node in body.select(selector):
            node.decompose()

    heading = soup.find("h1")
    title = display_clean(heading.get_text(" ", strip=True)) if heading else doc_id

    raw = body.get_text("\n", strip=True)
    text = display_clean(raw)

    match = VERIFIED_RE.search(text)
    verified = match.group(1) if match else None

    # Drop the title and the verification line from the body so they are not
    # duplicated in every chunk.
    text = VERIFIED_RE.sub("", text)
    if text.startswith(title):
        text = text[len(title):].lstrip("\n ")

    if len(text) < 400:
        return None

    return Document(
        doc_id=doc_id,
        url=url,
        title=title,
        verified_on=verified,
        text=text.strip(),
        n_chars=len(text),
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def fetch_corpus(out_path: Path, limit: int = 60, pause: float = 0.4) -> list[Document]:
    """Crawl, clean and save the corpus as JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []

    with _client() as client:
        print("discovering fiches from section pages")
        refs = discover_fiches(client, SEED_SECTIONS)
        for extra in EXTRA_FICHES:
            if extra not in refs:
                refs.append(extra)
        print(f"\n{len(refs)} candidate fiches, fetching up to {limit}\n")

        seen: set[str] = set()
        for ref in refs:
            if len(documents) >= limit:
                break
            if ref in seen:
                continue
            seen.add(ref)

            url = FICHE_URL.format(ref=ref)
            try:
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  ! {ref}: {type(exc).__name__}")
                continue

            document = extract(response.text, ref, str(response.url))
            if document is None:
                print(f"  - {ref}: no usable content")
                continue

            documents.append(document)
            print(f"  {len(documents):>3}. {ref}  {document.n_chars:>6} chars  {document.title[:60]}")
            time.sleep(pause)

    with out_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")

    print(f"\nwrote {len(documents)} documents to {out_path}")
    return documents


def load_corpus(path: Path) -> list[Document]:
    """Read the saved corpus back."""
    with path.open(encoding="utf-8") as handle:
        return [Document(**json.loads(line)) for line in handle if line.strip()]


def fetch_ids(ids_path: Path, out_path: Path, pause: float = 0.4) -> list[Document]:
    """Re-crawl exactly the fiches listed in ids_path (one Fxxxx per line)."""
    refs = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    documents: list[Document] = []
    with _client() as client:
        for ref in refs:
            try:
                response = client.get(FICHE_URL.format(ref=ref))
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  ! {ref}: {type(exc).__name__}")
                continue
            document = extract(response.text, ref, str(response.url))
            if document:
                documents.append(document)
                print(f"  {len(documents):>3}. {ref}  {document.title[:60]}")
            time.sleep(pause)
    with out_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")
    print(f"\nwrote {len(documents)} documents to {out_path}")
    return documents


if __name__ == "__main__":
    import sys

    ids = Path("data/corpus/fiche_ids.txt")
    if "--discover" in sys.argv or not ids.exists():
        fetch_corpus(Path("data/corpus/documents.jsonl"))
    else:
        # Default: rebuild the exact published corpus, so results stay comparable.
        fetch_ids(ids, Path("data/corpus/documents.jsonl"))
