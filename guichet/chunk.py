"""Split documents into retrievable chunks.

Chunking is where a lot of retrieval quality is quietly won or lost. Two
choices here that are worth defending:

1. Chunks break on blank lines, never mid-sentence. Administrative text is
   full of conditions ("si vous êtes...", "sauf si..."); a chunk cut through
   one of those produces an answer that states the rule without its exception,
   which is worse than no answer.

2. Each chunk carries its document title in the text that gets embedded. Many
   fiches use near-identical wording, and the title is often the only thing
   distinguishing "carte de séjour étudiant" from "carte de séjour salarié".
   The title is kept out of `body` so it is not shown twice to the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .corpus import Document

TARGET_CHARS = 1100
MAX_CHARS = 1800
OVERLAP_CHARS = 180
MIN_CHARS = 120


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    url: str
    title: str
    body: str
    position: int

    @property
    def embed_text(self) -> str:
        """What actually gets embedded: title for context, then the body."""
        return f"{self.title}\n\n{self.body}"


def _paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def _split_long(paragraph: str) -> list[str]:
    """Break a paragraph that exceeds MAX_CHARS on sentence boundaries."""
    if len(paragraph) <= MAX_CHARS:
        return [paragraph]

    pieces: list[str] = []
    current = ""
    for line in paragraph.split("\n"):
        if len(current) + len(line) + 1 > TARGET_CHARS and current:
            pieces.append(current.strip())
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_document(document: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer = ""
    position = 0

    def flush() -> None:
        nonlocal buffer, position
        body = buffer.strip()
        if len(body) >= MIN_CHARS:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{position:03d}",
                    doc_id=document.doc_id,
                    url=document.url,
                    title=document.title,
                    body=body,
                    position=position,
                )
            )
            position += 1
        buffer = ""

    for paragraph in _paragraphs(document.text):
        for piece in _split_long(paragraph):
            if len(buffer) + len(piece) + 2 > TARGET_CHARS and buffer:
                tail = buffer[-OVERLAP_CHARS:]
                flush()
                # Carry a little context forward so a rule and its condition
                # are not separated by a hard boundary.
                buffer = tail.strip()
            buffer = f"{buffer}\n\n{piece}" if buffer else piece

    flush()
    return chunks


def build_chunks(corpus_path: Path, out_path: Path) -> list[Chunk]:
    from .corpus import load_corpus

    documents = load_corpus(corpus_path)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    sizes = [len(c.body) for c in chunks]
    print(f"{len(documents)} documents -> {len(chunks)} chunks")
    print(f"chunk chars: min {min(sizes)}  median {sorted(sizes)[len(sizes) // 2]}  max {max(sizes)}")
    print(f"wrote {out_path}")
    return chunks


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]


if __name__ == "__main__":
    build_chunks(Path("data/corpus/documents.jsonl"), Path("data/corpus/chunks.jsonl"))
