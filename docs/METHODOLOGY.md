# Methodology

These are commitments, not aspirations. If one has to be broken, the report
says so.

## 1. The answer key is never written by the system under test

Candidate questions and draft answers are written by reading the source
documents directly. The retriever and any generator are never queried while
labelling. An answer key the system wrote for itself measures nothing.

## 2. Unreviewed labels never count

Every golden item ships with `reviewed: false`. A human checks it against its
source and flips the flag (`make review`). `run_eval` excludes unreviewed items
by default; `--include-unreviewed` exists for development and marks the report
`provisional`.

## 3. Questions come in matched pairs

Each item is the same question in French and in English. The headline number is
the gap between them, and pairing means that gap is not confounded by the two
sets asking about different things.

## 4. Unanswerable questions stay in

Some questions have no answer in the corpus. They have no gold documents and
are excluded from retrieval metrics. They exist for the generation half: a
system that answers them confidently is hallucinating, and catching that matters
more than accuracy.

## 5. Distractors stay in the corpus

The corpus includes off-topic fiches (retirement, bereavement, buying property).
A corpus containing only relevant documents flatters retrieval.

## 6. Labels are at document level

Retrieval returns chunks; the answer key labels documents. Asking a human which
of thirty chunks is "the" right one gives noisy labels and a metric that moves
with chunk size rather than retrieval quality.

## 7. Generation is judged with a named framework

The generation half uses Promptfoo or Inspect rather than a bespoke harness, and
the LLM judge is checked against human labels on a hand-labelled subset before
its scores are trusted.

## Metrics

| metric    | question it answers |
|-----------|---------------------|
| recall@k  | Did the right document reach the window the model sees? |
| hit@k     | Did at least one right document? |
| MRR       | How high did the first right document land? |
| nDCG@10   | For items with several right documents, were they all ranked well? |

All are reported separately for French and English queries, with the gap
expressed relative to French.

## Retrieval modes

| mode         | description |
|--------------|-------------|
| `bm25`       | Keyword matching over accent-folded, elision-split, acronym-expanded French tokens |
| `dense`      | Multilingual sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2), cosine |
| `hybrid`     | Reciprocal Rank Fusion of both, k=60, equal weights |
| `hybrid-lang`| As `hybrid`, but the BM25 ranking is weighted by `--foreign-sparse-weight` (default 0) when the query is not French |
