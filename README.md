# Guichet

[![ci](https://github.com/AdamosZ/Guichet/actions/workflows/ci.yml/badge.svg)](https://github.com/AdamosZ/Guichet/actions/workflows/ci.yml)

An evaluation harness for cross-lingual retrieval over French administrative
documents.

International students in France ask their questions in English. The
authoritative answers (titres de séjour, APL, working-hour limits, renewal
deadlines) exist only in French, on service-public.gouv.fr. This project
measures how much retrieval quality is lost across that language boundary, and
tests what closes the gap.

The deliverable is the measurement, not the chatbot.

> *Guichet*: the counter at the préfecture or the CAF where you queue to ask
> your question.

## Status

Corpus, chunking, four retrieval modes and the retrieval metrics are working.
The 18-pair seed set has been **human-reviewed against its sources** (12 Oct
2026); review corrected one wrong source label and one unsupported answer.

## First finding

18 reviewed question pairs (the same question asked in French and in English;
15 answerable, 3 deliberately unanswerable), hit@1:

| mode                  | French | English | gap  |
|-----------------------|--------|---------|------|
| BM25                  | 0.867  | 0.267   | −69% |
| Dense                 | 0.800  | 0.600   | −25% |
| Hybrid (RRF)          | 0.867  | 0.467   | −46% |
| **Hybrid, language-aware** | **0.867** | **0.600** | **−31%** |

BM25 failing on English is expected: keyword matching cannot cross languages.

The result worth following up is that **standard hybrid retrieval is worse than
dense alone for English queries** (0.467 vs 0.600), despite hybrid being the
usual recommendation. Reciprocal Rank Fusion treats both rankings as equally
trustworthy. For an out-of-language query the sparse ranking is close to noise
(it matches stray tokens like "France" or "residence") and fusion promotes that
noise over correct dense hits.

**Language-aware fusion** detects the query language and drops the BM25 ranking
when it does not match the corpus language. On this set it keeps hybrid's
French score and recovers dense's English score: the best of both. Partial
down-weighting (BM25 weight 0.25 or 0.5) did not beat dropping it.

**Caveat.** 15 answerable pairs is a small sample: one query flipping moves
hit@1 by 0.067. The direction is consistent across hit@3 and MRR, but the set
is being expanded to ~80 pairs before any claim is made about effect size.

**What the review caught.** One seed item asked how many hours a *foreign*
student may work, but its source page covers *EU* students only; the corpus
has no non-EU equivalent. The unreviewed label had counted an English miss that
was partly the label's fault. This is the reason unreviewed items never count.

## Quickstart

```bash
pip install -e ".[dev]"
make test               # unit tests
make validate           # check the answer key against the corpus
make review             # walk through unreviewed golden items
make eval               # reviewed items only: reportable numbers
make eval-provisional   # includes unreviewed items: development only
```

The corpus is committed under `data/corpus/` so results are reproducible
(source pages change). `make corpus` re-crawls it (~5 min, polite delays).

The harness refuses to report on unreviewed items unless explicitly told to,
and marks such reports `provisional` in their metadata.

## Layout

```
guichet/
  normalize.py   French normalisation; display text vs index text kept separate
  corpus.py      Crawl and clean service-public.gouv.fr fiches
  chunk.py       Paragraph-boundary chunking, title carried into embed text
  langid.py      One-bit query language detector (French or not)
  index.py       Dense (fastembed) + BM25 + weighted RRF, mode-swappable
  metrics.py     recall@k, hit@k, MRR, nDCG, split by query language
  golden.py      Answer-key schema and validation
  run_eval.py    CLI entry point
scripts/
  seed_golden.py   Seed questions drafted by reading sources directly
  review_golden.py Interactive review pass over the answer key
tests/           Unit tests, plus a check that the answer key stays valid
evals/golden.jsonl
data/corpus/     documents.jsonl, chunks.jsonl
reports/         JSON output of each run
docs/            Methodology
```

CI runs lint, unit tests and answer-key validation on every push, then the full
retrieval eval; results appear in the run's summary and as a downloadable
artifact.

## Method

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md). In short:

- Questions and answers were written by reading the source documents, never by
  querying the system under test.
- Unreviewed labels never count toward a reported metric.
- Deliberately unanswerable questions stay in the set: a system that answers
  them confidently is hallucinating.
- Off-topic documents stay in the corpus as distractors.

## Roadmap

- [x] Corpus, chunking, BM25 / dense / hybrid retrieval, per-language metrics
- [x] Language-aware fusion, measured against the baseline
- [x] CI on every push
- [x] Human review of the seed set
- [ ] Expansion to ~80 pairs
- [ ] Generation layer with programmatic citation verification
- [ ] Faithfulness scoring (Promptfoo or Inspect), with judge-vs-human agreement on a hand-labelled subset
- [ ] Report: numbers, failure taxonomy, limitations

## Limitations

- Source pages carry a "Vérifié le" date; some are from 2023. Answers are only
  as current as the crawl.
- Not legal advice, and not a substitute for the préfecture.
- One embedding model so far (paraphrase-multilingual-MiniLM-L12-v2). Larger
  multilingual models are untested here.
- The language detector is a stopword heuristic, correct on all 36 seed
  queries; mixed-language queries ("how to renew my titre de séjour") are the
  case to watch as the set grows.

## Licence

Code: MIT. Corpus text: © DILA / service-public.gouv.fr, reused under the
[Licence Ouverte / Open Licence Etalab 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).
Each document keeps its source URL and "Vérifié le" date.
