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

- **Answer key:** 104 question pairs, 10 of them deliberately unanswerable. 86 of
  them are real questions posted by people in France (Stack Exchange, the Ameli
  forum, Alexia, CommentÇaMarche, expat forums), paraphrased and stripped of
  personal details, each linked to its original post.
- **Review:** 17 items are human-reviewed. The other 87 have passed an
  independent pre-check (21 corrected, 0 dropped) and are awaiting human review.
- **Corpus:** 192 service-public.gouv.fr pages, 3,405 chunks. 138 pages are
  distractors that no question points to.

## Results

### Seed set: 18 pairs, human-reviewed

hit@1:

| mode                       | French | English | gap  |
|----------------------------|--------|---------|------|
| BM25                       | 0.867  | 0.267   | −69% |
| Dense                      | 0.800  | 0.600   | −25% |
| Hybrid (RRF)               | 0.867  | 0.467   | −46% |
| Hybrid, language-aware     | 0.867  | 0.600   | −31% |

On the seed set, standard hybrid retrieval was worse than dense alone for
English queries, and language-aware fusion (dropping BM25 for non-French
queries) recovered it.

### Expanded set: 104 pairs, provisional (87 items not yet human-reviewed)

94 answerable pairs, 192-page corpus:

| mode                   | FR hit@1 | EN hit@1 | FR recall@5 | EN recall@5 | EN MRR |
|------------------------|----------|----------|-------------|-------------|--------|
| BM25                   | 0.553    | 0.234    | 0.837       | 0.324       | 0.296  |
| Dense                  | 0.457    | 0.394    | 0.663       | 0.666       | 0.539  |
| Hybrid (RRF)           | 0.585    | 0.426    | 0.829       | 0.613       | 0.542  |
| Hybrid, language-aware | 0.585    | 0.415    | 0.829       | 0.666       | 0.554  |

**What changed with more, harder data.** The seed result only partly holds.
English queries still lose 20–30% against French in every mode. Language-aware
fusion still gives the best English recall@5 and MRR, but standard hybrid now
edges it at English hit@1.

One reason is visible in the data: **24 of the 94 English questions contain
French administrative terms** ("titre de séjour", "CAF", "récépissé"), because
that is how students actually write. BM25 can match those terms, so dropping it
entirely for English queries throws away a real signal. The seed set, written
by hand, had too few of these questions to show it. This is exactly why the set
was expanded with real questions.

Also new: the multilingual embedding model is weaker than BM25 on French queries
(hit@1 0.457 vs 0.553) once the corpus has many near-identical permit pages.

**Next experiments:** partial BM25 weighting for mixed-language queries, query
translation (EN → FR) before BM25, and a stronger multilingual embedding model.

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
- [ ] Human review of the 87 new items
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
