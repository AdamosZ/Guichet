"""Generation half: answer from retrieved pages, with citations a program can check.

Three steps, split so only one of them needs an API key:

    python -m guichet.generate contexts            # retrieval only, no key -> data/runs/contexts.jsonl
    python -m guichet.generate run                 # calls Claude (needs ANTHROPIC_API_KEY in .env)
    python -m guichet.generate score               # checks citations, abstention; no key

The model must answer only from the retrieved passages, in the language of the
question, and return JSON with a verbatim quote for every citation. Scoring is
then mechanical, with no LLM judge:

- citation validity: the cited chunk was actually retrieved for this question
- quote fidelity: the quoted text really appears in that chunk
- gold citation: at least one citation points at a gold document
- abstention: unanswerable questions should get "not found"; answerable ones
  whose gold page was retrieved should not
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

RUNS = Path("data/runs")
CONTEXTS = RUNS / "contexts.jsonl"
DEFAULT_MODEL = "claude-haiku-5-5"
RETRIEVAL_MODE = "hybrid-trans"
TOP_K = 5

SYSTEM = """You answer questions from international students about French administration.
Use ONLY the numbered passages provided, which come from service-public.gouv.fr.

Rules:
- Answer in the same language as the question (English or French).
- Every factual sentence must be supported by a passage. Do not add anything from your own knowledge.
- For each passage you rely on, give a short quote copied EXACTLY, character for character, from that passage (in French, as written).
- If the passages do not contain the answer, say so: set "found" to false and give no citations. Do not guess.

Reply with JSON only, no other text:
{"found": true or false, "answer": "<your answer, or a one-sentence explanation that the sources don't cover it>",
 "citations": [{"chunk_id": "<id exactly as given>", "quote": "<exact quote>"}]}"""


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


# --------------------------------------------------------------------------- contexts

def build_contexts(out: Path = CONTEXTS, mode: str = RETRIEVAL_MODE, top_k: int = TOP_K) -> None:
    from .chunk import load_chunks
    from .golden import load
    from .index import Retriever

    retriever = Retriever(load_chunks(Path("data/corpus/chunks.jsonl")), mode=mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out.open("w", encoding="utf-8") as handle:
        for item in load():
            if not item.reviewed:
                continue
            for language, question in (("fr", item.question_fr), ("en", item.question_en)):
                hits = retriever.search(question, top_k=top_k)
                handle.write(json.dumps({
                    "item_id": item.item_id, "language": language, "question": question,
                    "answerable": item.answerable, "gold_doc_ids": item.gold_doc_ids,
                    "retrieval_mode": mode,
                    "passages": [{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id,
                                  "title": h.chunk.title, "text": h.chunk.body} for h in hits],
                }, ensure_ascii=False) + "\n")
                rows += 1
    print(f"wrote {rows} retrieval contexts ({mode}, top {top_k}) to {out}")


# --------------------------------------------------------------------------- generation

def _load_env() -> None:
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                # .env wins over anything already exported in the shell.
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


LANGUAGE_NAME = {"fr": "French", "en": "English"}


def _prompt(row: dict) -> str:
    blocks = [f"[{p['chunk_id']}] {p['title']}\n{p['text']}" for p in row["passages"]]
    language = LANGUAGE_NAME[row["language"]]
    return ("Passages:\n\n" + "\n\n---\n\n".join(blocks)
            + f"\n\nQuestion: {row['question']}"
            + f"\n\nWrite the \"answer\" field in {language}. Quotes stay in French, exactly as written.")


def parse_reply(text: str) -> dict:
    """Pull the JSON object out of the reply; a malformed reply is scored as a failure."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {"found": None, "answer": text.strip(), "citations": [], "parse_error": True}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"found": None, "answer": text.strip(), "citations": [], "parse_error": True}
    data.setdefault("citations", [])
    return data


def run(model: str = DEFAULT_MODEL, contexts: Path = CONTEXTS, limit: int | None = None) -> Path:
    import anthropic

    _load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set — put it in .env (never commit it).")

    client = anthropic.Anthropic()
    out = RUNS / f"generations__{model}.jsonl"
    done: set[tuple[str, str]] = set()
    if out.exists():  # resume: skip rows already generated
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["item_id"], row["language"]))

    rows = [json.loads(line) for line in contexts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    tokens_in = tokens_out = 0
    with out.open("a", encoding="utf-8") as handle:
        for n, row in enumerate(rows, start=1):
            if (row["item_id"], row["language"]) in done:
                continue
            for attempt in range(4):
                try:
                    message = client.messages.create(
                        model=model, max_tokens=4096, system=SYSTEM,
                        messages=[{"role": "user", "content": _prompt(row)}],
                    )
                    break
                except (anthropic.RateLimitError, anthropic.APIConnectionError,
                        anthropic.InternalServerError):
                    time.sleep(2 ** attempt * 5)
            else:
                raise SystemExit("API kept failing; rerun later — progress is saved.")
            text = "".join(block.text for block in message.content if block.type == "text")
            tokens_in += message.usage.input_tokens
            tokens_out += message.usage.output_tokens
            handle.write(json.dumps({
                "item_id": row["item_id"], "language": row["language"], "model": model,
                "stop_reason": message.stop_reason, "raw": text, **parse_reply(text),
            }, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"  {n}/{len(rows)}  {row['item_id']} [{row['language']}]")
    print(f"\nwrote {out}\ntokens this run: {tokens_in:,} in / {tokens_out:,} out")
    return out


# --------------------------------------------------------------------------- scoring

def score_row(gen: dict, ctx: dict) -> dict:
    passages = {p["chunk_id"]: p for p in ctx["passages"]}
    citations = gen.get("citations") or []
    valid = [c for c in citations if c.get("chunk_id") in passages]
    faithful = [c for c in valid
                if c.get("quote") and _norm(c["quote"]) in _norm(passages[c["chunk_id"]]["text"])]
    gold = set(ctx["gold_doc_ids"])
    gold_retrieved = any(p["doc_id"] in gold for p in ctx["passages"])
    abstained = gen.get("found") is False
    from .langid import detect

    answer = gen.get("answer") or ""
    return {
        "item_id": ctx["item_id"], "language": ctx["language"], "answerable": ctx["answerable"],
        "gold_retrieved": gold_retrieved, "abstained": abstained,
        "parse_error": bool(gen.get("parse_error")),
        "truncated": gen.get("stop_reason") == "max_tokens",
        "answer_language_ok": detect(answer) == ctx["language"] if answer else None,
        "n_citations": len(citations),
        "citation_valid_rate": len(valid) / len(citations) if citations else None,
        "quote_faithful_rate": len(faithful) / len(citations) if citations else None,
        "cites_gold": any(passages[c["chunk_id"]]["doc_id"] in gold for c in valid),
    }


def summarise(scores: list[dict]) -> dict:
    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else None

    out: dict = {}
    for language in ("fr", "en"):
        rows = [s for s in scores if s["language"] == language]
        answerable = [s for s in rows if s["answerable"]]
        reachable = [s for s in answerable if s["gold_retrieved"]]
        unanswerable = [s for s in rows if not s["answerable"]]
        out[language] = {
            "n": len(rows),
            "parse_errors": sum(s["parse_error"] for s in rows),
            "truncated": sum(s["truncated"] for s in rows),
            "answer_language_ok": mean(None if s["answer_language_ok"] is None
                                       else float(s["answer_language_ok"]) for s in rows),
            "citation_valid_rate": mean(s["citation_valid_rate"] for s in rows),
            "quote_faithful_rate": mean(s["quote_faithful_rate"] for s in rows),
            # Of answerable questions whose gold page was retrieved: did the answer cite it?
            "cites_gold_when_retrieved": mean(float(s["cites_gold"]) for s in reachable),
            # Wrongly said "not found" although the gold page was in front of it.
            "false_abstention_rate": mean(float(s["abstained"]) for s in reachable),
            # Correctly said "not found" on questions the corpus can't answer.
            "correct_abstention_rate": mean(float(s["abstained"]) for s in unanswerable),
        }
    return out


def score(model: str = DEFAULT_MODEL, contexts: Path = CONTEXTS) -> dict:
    ctx = {(r["item_id"], r["language"]): r for r in
           map(json.loads, contexts.read_text(encoding="utf-8").splitlines())}
    path = RUNS / f"generations__{model}.jsonl"
    gens = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = [score_row(g, ctx[(g["item_id"], g["language"])]) for g in gens]
    summary = summarise(scores)
    report = Path("reports") / f"generation__{model}.json"
    report.write_text(json.dumps({"model": model, "summary": summary, "rows": scores},
                                 indent=2, ensure_ascii=False), encoding="utf-8")
    for language, values in summary.items():
        print(f"\n[{language}]")
        for key, value in values.items():
            print(f"  {key:<28} {value}")
    print(f"\nwrote {report}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["contexts", "run", "score"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    args = parser.parse_args()
    if args.step == "contexts":
        build_contexts()
    elif args.step == "run":
        run(args.model, limit=args.limit)
    else:
        score(args.model)


if __name__ == "__main__":
    main()
