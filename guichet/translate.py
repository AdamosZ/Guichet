"""Query translation: give BM25 a French query it can actually match.

BM25 cannot match "residence permit" to "titre de séjour". Translating the
English query to French first, and running BM25 on the translation, tests
whether the cross-lingual gap is a vocabulary problem that translation fixes.

Translations are computed once and stored in data/translations/<model>.jsonl,
keyed by the exact query text, so the eval (and CI) never needs the model:

    pip install torch transformers sentencepiece sacremoses
    python -m guichet.translate            # fills the cache for every EN question

Dense retrieval keeps the original query; only the sparse side is translated.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MT = "Helsinki-NLP/opus-mt-en-fr"
CACHE_DIR = Path("data/translations")


def cache_path(model: str = DEFAULT_MT) -> Path:
    return CACHE_DIR / (model.replace("/", "__") + ".jsonl")


def load_cache(model: str = DEFAULT_MT) -> dict[str, str]:
    path = cache_path(model)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["en"]: row["fr"] for row in rows}


def translate_all(queries: list[str], model: str = DEFAULT_MT, batch: int = 16) -> dict[str, str]:
    from transformers import MarianMTModel, MarianTokenizer

    cache = load_cache(model)
    todo = [q for q in dict.fromkeys(queries) if q not in cache]
    if todo:
        tokenizer = MarianTokenizer.from_pretrained(model)
        mt = MarianMTModel.from_pretrained(model)
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            outputs = mt.generate(**encoded, max_new_tokens=256, num_beams=4)
            for source, target in zip(chunk, tokenizer.batch_decode(outputs, skip_special_tokens=True),
                                      strict=True):
                cache[source] = target
            print(f"  translated {min(start + batch, len(todo))}/{len(todo)}")
        path = cache_path(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for source in sorted(cache):
                handle.write(json.dumps({"en": source, "fr": cache[source]}, ensure_ascii=False) + "\n")
    return cache


if __name__ == "__main__":
    from .golden import load

    questions = [item.question_en for item in load()]
    result = translate_all(questions)
    print(f"{len(result)} translations cached in {cache_path()}")
