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

| mode                          | FR hit@1 | EN hit@1 | FR recall@5 | EN recall@5 |
|-------------------------------|----------|----------|-------------|-------------|
| BM25                          | 0.553    | 0.234    | 0.842       | 0.324       |
| Dense (multilingual)          | 0.457    | 0.372    | 0.663       | 0.666       |
| Hybrid (RRF)                  | 0.596    | 0.426    | 0.829       | 0.602       |
| Hybrid, language-aware        | 0.596    | 0.394    | 0.829       | 0.666       |
| **Hybrid, translated BM25**   | **0.596**| **0.585**| **0.829**   | **0.738**   |

English minus French on the same questions, paired-bootstrap 95% CIs
(`python -m scripts.significance`):

| mode                     | hit@1 gap               | recall@5 gap            |
|--------------------------|-------------------------|-------------------------|
| BM25                     | −0.32 [−0.44, −0.20]    | −0.52 [−0.62, −0.43]    |
| Dense                    | −0.09 [−0.19, +0.02]    | +0.00 [−0.07, +0.07]    |
| Hybrid (RRF)             | −0.17 [−0.29, −0.05]    | −0.23 [−0.31, −0.15]    |
| Hybrid, language-aware   | −0.20 [−0.31, −0.10]    | −0.16 [−0.24, −0.10]    |
| Hybrid, translated BM25  | **−0.01 [−0.11, +0.09]**| −0.09 [−0.17, −0.02]    |

### What this shows

1. **Hybrid search's advantage is French-only.** Dense retrieval is
   language-neutral but weaker. BM25 is what makes hybrid the best French
   system, and BM25 can't match "residence permit" to "titre de séjour". So
   standard hybrid retrieval builds in a cross-lingual gap of about 23 recall
   points.
2. **Translating the query for BM25 closes most of it.** Translating English
   questions to French (Helsinki-NLP opus-mt-en-fr, a small free model) before
   BM25, while dense keeps the original query, lifts English hit@1 from 0.426
   to 0.585 (+0.16, CI [+0.07, +0.26]) and English recall@5 from 0.602 to 0.738
   (+0.14, CI [+0.06, +0.22]). The hit@1 gap with French becomes
   indistinguishable from zero. French results are untouched.
3. **It isn't a back-translation artefact.** About half the English questions
   were themselves translated from French forum posts, so translating them back
   could just recover the original wording. On the 38 questions originally
   posted in English, the gain holds (recall@5 +0.17, CI [+0.07, +0.29]).
4. **Simply dropping BM25 for English queries is not enough.** Language-aware
   fusion narrows the recall gap a little, but its edge over plain hybrid isn't
   significant at n=94. Real English questions often contain French terms
   ("titre de séjour", "CAF", 24 of 94), which BM25 can use. Translation keeps
   those terms *and* fixes the rest.

### How the result changed as the data got real

On the first 18 hand-written pairs, language-aware fusion looked like a clean
win. With 104 real, reviewed questions and a harder corpus, it shrank inside
the noise, and a more robust finding came out: the gap comes from the sparse
side, and query translation fixes it. The seed numbers are kept in git history.

**Next:** an LLM as the translator (does a better translation help further?),
a stronger multilingual embedding model, then the generation half: answers
with checked citations, and faithfulness scoring.

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
  translate.py   EN→FR query translation for BM25, cached in data/translations/
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
- [x] Query translation (EN→FR) for the sparse side
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
