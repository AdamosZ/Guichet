# Methodology

These are commitments, not aspirations. If one has to be broken, the report
says so.

## 1. The answer key is never written by the system under test

Candidate questions and draft answers are written by reading the source
documents directly. The retriever and any generator are never queried while
labelling. An answer key the system wrote for itself measures nothing.

## 1b. Questions are real, answers are not taken from where the questions came from

Beyond the 18 seed items, questions come from real posts by people dealing with
French administration: Stack Exchange (Expatriates, Travel; CC BY-SA), the Ameli
assurés forum, Alexia, CommentÇaMarche, expat forums, and the government's own
Services Publics+ feedback site. Each is paraphrased into one clear question,
stripped of personal details, and keeps a `source_url` to the original post.

Only the question is taken from the post. Forum replies are often wrong, so
answers are written from the service-public page alone, with the supporting
sentences stored verbatim in `evidence`.

Each new item then gets an **independent pre-check** by a reviewer who did not
draft it. The reviewer reads the full page, not just the quoted sentence: a
quote cut short ("vous pouvez bénéficier d'une aide au logement uniquement dans
l'un des 3 cas suivants") can reverse the meaning. The pre-check is advisory;
human review is still required before an item counts.

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
