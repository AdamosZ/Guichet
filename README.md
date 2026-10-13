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

- **Answer key:** 104 question pairs (94 answerable, 10 deliberately
  unanswerable), **all human-reviewed** against their source pages, after
  independent checks by Claude. 86 are real questions people posted (Stack
  Exchange, the Ameli forum, Alexia, CommentÇaMarche, expat forums), paraphrased
  and stripped of personal details, each linked to its original post.
- **Corpus:** 192 service-public.gouv.fr pages, 3,405 chunks. 138 pages are
  distractors no question points to.

## Results

94 answerable pairs, the same question asked in French and in English.

| mode                   | FR hit@1 | EN hit@1 | FR recall@5 | EN recall@5 |
|------------------------|----------|----------|-------------|-------------|
| BM25                   | 0.553    | 0.234    | 0.842       | 0.324       |
| Dense (multilingual)   | 0.457    | 0.372    | 0.663       | 0.666       |
| Hybrid (RRF)           | 0.596    | 0.426    | 0.829       | 0.602       |
| Hybrid, language-aware | 0.596    | 0.394    | 0.829       | 0.666       |

English minus French on the same questions, with paired-bootstrap 95% CIs
(`python -m scripts.significance`):

| mode                   | recall@5 gap                 |
|------------------------|------------------------------|
| BM25                   | −0.52  [−0.61, −0.42]        |
| Dense                  | +0.00  [−0.07, +0.07]        |
| Hybrid (RRF)           | −0.23  [−0.31, −0.15]        |
| Hybrid, language-aware | −0.16  [−0.23, −0.10]        |

### What this shows

1. **Hybrid search's advantage is French-only.** Dense retrieval is
   language-neutral (no measurable gap) but weaker overall. BM25 is what makes
   hybrid the best French system, and BM25 does not cross languages. So the
   standard recommendation, hybrid, builds in a cross-lingual gap of about 23
   recall points that the dense model alone does not have.
2. **Language-aware fusion narrows that gap but doesn't close it.** Dropping
   BM25 for English queries cuts the gap from −0.23 to −0.16. Compared directly
   with plain hybrid on English queries, its +0.06 recall@5 has a CI that
   crosses zero ([−0.02, +0.14]) at n=94. That's suggestive, not established.
3. **Real questions are mixed-language.** 24 of the 94 English questions use
   French terms ("titre de séjour", "CAF", "récépissé"). BM25 can match those,
   which is why switching it off entirely costs some hit@1.

### How the result changed as the data got real

On the first 18 hand-written pairs, language-aware fusion looked like a clean
win (English hit@1 0.467 → 0.600). With 104 real, reviewed questions and a
harder corpus, that effect shrank inside the noise, and the more robust finding
turned out to be point 1. The seed numbers are kept in git history.

**Next experiments:** translate English queries to French before BM25 (gives
BM25 a query it can match), partial BM25 weighting for mixed-language queries,
and a stronger multilingual embedding model.

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
(source pages change). `make corpus` re-crawls exactly the pages listed in
`data/corpus/fiche_ids.txt`.

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
  harvest_stackexchange.py  Collect real candidate questions (provenance)
  review_golden.py Interactive review pass over the answer key
  significance.py  Paired-bootstrap CIs for the headline comparisons
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
- [x] Expansion to 104 pairs from real posted questions, independently pre-checked
- [x] Human review of all 104 items
- [x] Paired-bootstrap confidence intervals
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
