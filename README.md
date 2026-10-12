# Theia

An evaluation harness for cross-lingual retrieval over French administrative
documents.

International students in France ask their questions in English. The
authoritative answers — titres de séjour, APL, working-hour limits, renewal
deadlines — exist only in French, on service-public.gouv.fr. This project
measures how much retrieval quality is lost across that language boundary, and
tests what closes the gap.

The deliverable is the measurement, not the chatbot.

## Status

Day 1 of the build. Corpus, chunking, three retrieval modes and the retrieval
metrics are working and produce numbers. The golden set is seeded but **not yet
human-reviewed**, so every figure below is provisional.

## Provisional finding

18 question pairs (same question asked in French and English), hit@1:

| mode   | French | English | gap  |
|--------|--------|---------|------|
| BM25   | 0.867  | 0.333   | −62% |
| Dense  | 0.800  | 0.600   | −25% |
| Hybrid | 0.867  | 0.467   | −46% |

BM25 failing on English is expected: keyword matching cannot cross languages.

The result worth following up is that **hybrid retrieval is worse than dense
alone for English queries** (0.467 vs 0.600), despite hybrid being the standard
recommendation. Reciprocal Rank Fusion treats both rankings as equally
trustworthy. For an out-of-language query the sparse ranking is close to noise —
it matches on stray tokens like "France" or "residence" — and fusion promotes
that noise over correct dense hits.

Next test: language-aware fusion. Detect the query language and down-weight or
drop the sparse ranking when it does not match the corpus language.

**Caveat.** n=18 with unreviewed labels. This is evidence the method works, not
a result.

## Layout

```
guichet/
  normalize.py   French normalisation; display vs index text kept separate
  corpus.py      Crawl and clean service-public.gouv.fr fiches
  chunk.py       Paragraph-boundary chunking, title carried into embed text
  index.py       Dense (fastembed) + BM25 + RRF fusion, mode-swappable
  metrics.py     recall@k, hit@k, MRR, nDCG, split by query language
  golden.py      Answer-key schema and validation
  run_eval.py    CLI entry point
scripts/
  seed_golden.py Seed questions drafted by reading sources directly
data/corpus/     documents.jsonl, chunks.jsonl
evals/golden.jsonl
reports/         retrieval.json
```

## Running it

```bash
pip install -r requirements.txt
python -m guichet.corpus                 # crawl (~5 min, polite delays)
python -m guichet.chunk
python -m scripts.seed_golden
python -m guichet.golden                 # validate the answer key
python -m guichet.run_eval --include-unreviewed
```

Drop `--include-unreviewed` once labels are reviewed; the harness refuses to
report on unreviewed items by default.

## How the answer key was made

This decides how much the numbers are worth, so it is stated plainly.

Candidate questions and draft answers were written by reading the cited source
documents directly — never by querying the system under test, which would make
the answer key circular. Every item ships with `reviewed: false` and must be
checked by a human against its source before it counts toward a reported
metric.

Three items are deliberately **unanswerable**: the corpus genuinely does not
contain the answer. A system that answers them confidently is hallucinating,
and catching that matters more than any accuracy figure.

The corpus also keeps off-topic documents (retirement, bereavement, property
purchase) as distractors. A corpus containing only relevant documents flatters
retrieval.

## Still to build

- Human review of the seed set, and expansion to ~80 pairs
- Language-aware fusion, measured against the current baseline
- Generation layer with citation verification
- Faithfulness scoring, with judge-vs-human agreement measured on a
  hand-labelled subset
- CI running the suite on every push

## Limitations

- Source pages carry a "Vérifié le" date; some are from 2023. Answers are only
  as current as the crawl.
- Not legal advice, and not a substitute for the préfecture.
- Single embedding model so far (paraphrase-multilingual-MiniLM-L12-v2).
  Larger multilingual models are untested here.

## Licence

Code MIT. Source documents belong to DILA / service-public.gouv.fr and are not
redistributed — `guichet.corpus` fetches them.

## Data licence

Corpus text comes from service-public.gouv.fr (DILA), reused under the
Licence Ouverte / Open Licence Etalab 2.0. Each document keeps its source URL
and "Vérifié le" date.
