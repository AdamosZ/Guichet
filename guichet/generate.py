"""Generation half: answer from retrieved pages, with citations a program can check.

Three steps, split so only one of them needs an API key:

    python -m guichet.generate contexts --context top5   # retrieval only, no key
    python -m guichet.generate run --context top5        # calls Claude (ANTHROPIC_API_KEY in .env)
    python -m guichet.generate score --context top5      # checks citations, abstention; no key

--context picks how much the model sees: top5 / top10 passages, or the full
text of the top 3 / 5 pages (pages3 / pages5).

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
DEFAULT_MODEL = "claude-haiku-5-5"
RETRIEVAL_MODE = "hybrid-trans"

# How much of the retrieved material the model sees.
#   chunks: the top-k retrieved passages
#   pages:  the full text of the top-k distinct pages, in retrieval order
CONTEXT_VARIANTS = {
    "top5": ("chunks", 5),
    "top10": ("chunks", 10),
    "pages3": ("pages", 3),
    "pages5": ("pages", 5),
}
DEFAULT_CONTEXT = "top5"


def contexts_path(variant: str) -> Path:
    return RUNS / f"contexts__{variant}.jsonl"


def _suffix(prompt: str) -> str:
    # v1 files keep their original names so earlier results stay reproducible.
    return "" if prompt == "v1" else f"__{prompt}"


def generations_path(model: str, variant: str, prompt: str = "v1") -> Path:
    from .llm import safe_name

    model = safe_name(model)
    return RUNS / f"generations__{model}__{variant}{_suffix(prompt)}.jsonl"


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

# v2: same rules, written for the reader. Pilot feedback on v1: answers talked
# about "the passages", hedged before answering, and buried conditions in prose.
SYSTEM_V2 = """You help international students with French administrative procedures.
Use ONLY the numbered official pages provided, which come from service-public.gouv.fr.

How to write the answer (the reader is often stressed and reading in a second language):
- First sentence: the direct answer. "Yes", "No", "It depends on…", or the number, deadline or amount asked for.
- If the answer depends on conditions, list each one on its own line starting with "- ", in plain words,
  keeping the exact amounts, durations and thresholds from the page. Say whether ALL are required or ONE is enough.
- If the page says what to do next (where to apply, which office, which form), end with one line about it.
- Short sentences. Explain any French administrative term the first time you use it.
- Never mention "passages", "sources", "the text provided" or yourself. If needed, say "the official page".

Rules:
- Every fact must come from the pages. Add nothing from your own knowledge.
- For each page you rely on, give a short quote copied EXACTLY, character for character, from that page (in French, as written).
- If the pages do not answer the question, set "found" to false and give no citations. Do not guess.

Reply with JSON only, no other text:
{"found": true or false, "answer": "<the answer, formatted as above>",
 "citations": [{"chunk_id": "<id exactly as given>", "quote": "<exact quote>"}]}"""

# v2s: the v2 instructions, but the reply is forced through a tool ("fill in this
# form") instead of free-text JSON, so a malformed reply can't happen. v2 broke
# the JSON format on 5 of 208 answers, which a student would see as an error.
PROMPTS = {"v1": SYSTEM, "v2": SYSTEM_V2, "v2s": SYSTEM_V2}
STRUCTURED = {"v2s"}

ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the answer to the student's question, with exact quotes from the pages.",
    "input_schema": {
        "type": "object",
        "properties": {
            "found": {"type": "boolean",
                      "description": "false if the pages do not answer the question"},
            "answer": {"type": "string",
                       "description": "The answer, formatted as instructed (direct answer first, '- ' bullets)."},
            "citations": {
                "type": "array",
                "items": {"type": "object",
                          "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}},
                          "required": ["chunk_id", "quote"]},
            },
        },
        "required": ["found", "answer", "citations"],
    },
}


def request_kwargs(prompt: str) -> dict:
    """Extra arguments for messages.create: force the answer form for structured prompts."""
    if prompt in STRUCTURED:
        return {"tools": [ANSWER_TOOL], "tool_choice": {"type": "tool", "name": ANSWER_TOOL["name"]},
                "max_tokens": 8192}
    return {"max_tokens": 4096}


def read_message(message) -> tuple[str, dict]:
    """(raw text for the record, parsed reply) from either a form (tool) or a text reply."""
    for block in message.content:
        if getattr(block, "type", "") == "tool_use":
            data = dict(block.input)
            data.setdefault("citations", [])
            return json.dumps(data, ensure_ascii=False), data
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    return text, parse_reply(text)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", _norm(text))


def quote_in(quote: str, text: str) -> bool:
    """Is the quote a contiguous word sequence of the text?

    Compares words, not characters, so extraction artefacts in the stored page
    (a space before a comma, a line break inside "1er") don't count as misquotes,
    while any changed, added, dropped or reordered word still does.
    """
    q, t = _words(quote), _words(text)
    if not q:
        return False
    first = q[0]
    return any(t[i:i + len(q)] == q for i, word in enumerate(t) if word == first)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


# --------------------------------------------------------------------------- contexts

def build_contexts(variant: str = DEFAULT_CONTEXT, mode: str = RETRIEVAL_MODE) -> None:
    from .chunk import load_chunks
    from .corpus import load_corpus
    from .golden import load
    from .index import Retriever

    kind, k = CONTEXT_VARIANTS[variant]
    retriever = Retriever(load_chunks(Path("data/corpus/chunks.jsonl")), mode=mode)
    pages = {d.doc_id: d for d in load_corpus(Path("data/corpus/documents.jsonl"))} if kind == "pages" else {}
    out = contexts_path(variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out.open("w", encoding="utf-8") as handle:
        for item in load():
            if not item.reviewed:
                continue
            for language, question in (("fr", item.question_fr), ("en", item.question_en)):
                if kind == "chunks":
                    hits = retriever.search(question, top_k=k)
                    passages = [{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id,
                                 "title": h.chunk.title, "text": h.chunk.body} for h in hits]
                else:
                    order: list[str] = []
                    for hit in retriever.search(question, top_k=50):
                        if hit.chunk.doc_id not in order:
                            order.append(hit.chunk.doc_id)
                    passages = [{"chunk_id": doc_id, "doc_id": doc_id, "title": pages[doc_id].title,
                                 "text": pages[doc_id].text} for doc_id in order[:k]]
                handle.write(json.dumps({
                    "item_id": item.item_id, "language": language, "question": question,
                    "answerable": item.answerable, "gold_doc_ids": item.gold_doc_ids,
                    "retrieval_mode": mode, "context": variant, "passages": passages,
                }, ensure_ascii=False) + "\n")
                rows += 1
    print(f"wrote {rows} contexts ({mode}, {variant}) to {out}")


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


def run(model: str = DEFAULT_MODEL, variant: str = DEFAULT_CONTEXT, limit: int | None = None,
        prompt: str = "v1", items: set[str] | None = None) -> Path:
    from .llm import AccessError, complete, is_claude

    _load_env()
    needed = "ANTHROPIC_API_KEY" if is_claude(model) else "HF_TOKEN"
    if not os.environ.get(needed):
        raise SystemExit(f"{needed} not set: put it in .env (never commit it).")

    contexts = contexts_path(variant)
    out = generations_path(model, variant, prompt)
    done: set[tuple[str, str]] = set()
    if out.exists():  # resume: skip rows already generated
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("stop_reason") != "error":  # failed calls are retried on the next run
                done.add((row["item_id"], row["language"]))

    rows = [json.loads(line) for line in contexts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if items:
        rows = [r for r in rows if r["item_id"] in items]
    if limit:
        rows = rows[:limit]
    tokens_in = tokens_out = failures = attempts = 0
    tool = ANSWER_TOOL if prompt in STRUCTURED else None
    max_tokens = request_kwargs(prompt)["max_tokens"]
    with out.open("a", encoding="utf-8") as handle:
        for n, row in enumerate(rows, start=1):
            if (row["item_id"], row["language"]) in done:
                continue
            started = time.time()
            attempts += 1
            try:
                result = complete(model, PROMPTS[prompt], _prompt(row), tool=tool, max_tokens=max_tokens)
            except AccessError as exc:
                raise SystemExit(f"\nStopped: the model can't be called with these credentials.\n{exc}") from None
            except Exception as exc:  # keep going: a failed call is scored as a failure, not a crash
                failures += 1
                print(f"  {n}/{len(rows)}  {row['item_id']} [{row['language']}]  FAILED {type(exc).__name__}: "
                      f"{str(exc)[:120]}")
                if failures >= 5 and failures == attempts:
                    raise SystemExit("\nStopped: the first 5 calls all failed. Fix the error above, "
                                     "then rerun.") from None
                reply, raw, stop = {"found": None, "answer": "", "citations": [], "parse_error": True}, str(exc), "error"
            else:
                reply = result.data if tool else parse_reply(result.text)
                raw, stop = result.raw, result.stop_reason
                tokens_in += result.tokens_in
                tokens_out += result.tokens_out
                print(f"  {n}/{len(rows)}  {row['item_id']} [{row['language']}]  {time.time() - started:.1f}s")
            handle.write(json.dumps({
                "item_id": row["item_id"], "language": row["language"], "model": model,
                "stop_reason": stop, "seconds": round(time.time() - started, 1), "raw": raw, **reply,
            }, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"\nwrote {out}\ntokens this run: {tokens_in:,} in / {tokens_out:,} out")
    return out


# --------------------------------------------------------------------------- scoring

def score_row(gen: dict, ctx: dict, evidence: list[dict] | None = None) -> dict:
    passages = {p["chunk_id"]: p for p in ctx["passages"]}
    # Was the passage that actually contains the answer in front of the model?
    # Document-level retrieval can succeed while the answer-bearing chunk of a
    # long page stays outside the top k.
    context_text = _norm(" ".join(p["text"] for p in ctx["passages"]))
    evidence_in_context = any(_norm(e.get("quote", ""))[:60] in context_text
                              for e in (evidence or []) if e.get("quote"))
    citations = gen.get("citations") or []
    valid = [c for c in citations if c.get("chunk_id") in passages]
    faithful = [c for c in valid
                if c.get("quote") and quote_in(c["quote"], passages[c["chunk_id"]]["text"])]
    gold = set(ctx["gold_doc_ids"])
    gold_retrieved = any(p["doc_id"] in gold for p in ctx["passages"])
    abstained = gen.get("found") is False
    from .langid import detect

    answer = gen.get("answer") or ""
    return {
        "item_id": ctx["item_id"], "language": ctx["language"], "answerable": ctx["answerable"],
        "gold_retrieved": gold_retrieved, "evidence_in_context": evidence_in_context,
        "abstained": abstained,
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
        evidenced = [s for s in answerable if s["evidence_in_context"]]
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
            # End to end: answerable questions answered with a citation to a gold page.
            "answered_with_gold_citation": mean(float(s["cites_gold"] and not s["abstained"])
                                                for s in answerable),
            # Said "not found" although a gold page was retrieved (document level)...
            "abstained_gold_page_retrieved": mean(float(s["abstained"]) for s in reachable),
            # ...and the stricter version: the answer-bearing passage itself was in context.
            "abstained_evidence_in_context": mean(float(s["abstained"]) for s in evidenced),
            # Correctly said "not found" on questions the corpus can't answer.
            "correct_abstention_rate": mean(float(s["abstained"]) for s in unanswerable),
        }
    return out


def score(model: str = DEFAULT_MODEL, variant: str = DEFAULT_CONTEXT, prompt: str = "v1") -> dict:
    contexts = contexts_path(variant)
    ctx = {(r["item_id"], r["language"]): r for r in
           map(json.loads, contexts.read_text(encoding="utf-8").splitlines())}
    path = generations_path(model, variant, prompt)
    gens = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    from .golden import load

    evidence = {item.item_id: item.evidence for item in load()}
    scores = [score_row(g, ctx[(g["item_id"], g["language"])], evidence.get(g["item_id"]))
              for g in gens]
    summary = summarise(scores)
    from .llm import safe_name

    report = Path("reports") / f"generation__{safe_name(model)}__{variant}{_suffix(prompt)}.json"
    report.write_text(json.dumps({"model": model, "context": variant, "prompt": prompt, "summary": summary,
                                  "rows": scores},
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
    parser.add_argument("--context", default=DEFAULT_CONTEXT, choices=sorted(CONTEXT_VARIANTS))
    parser.add_argument("--prompt", default="v1", choices=sorted(PROMPTS))
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    parser.add_argument("--items", default="", help="comma-separated item_ids: run only these (spot check)")
    args = parser.parse_args()
    if args.step == "contexts":
        build_contexts(args.context)
    elif args.step == "run":
        items = {i.strip() for i in args.items.split(",") if i.strip()} or None
        run(args.model, args.context, limit=args.limit, prompt=args.prompt, items=items)
    else:
        score(args.model, args.context, args.prompt)


if __name__ == "__main__":
    main()
