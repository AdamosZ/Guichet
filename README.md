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
| BM25                          | 0.553    | 0.234    | 0.837       | 0.324       |
| Dense (multilingual)          | 0.457    | 0.372    | 0.663       | 0.666       |
| Hybrid (RRF)                  | 0.596    | 0.426    | 0.829       | 0.602       |
| Hybrid, language-aware        | 0.596    | 0.372    | 0.829       | 0.666       |
| **Hybrid, translated BM25**   | **0.596**| **0.574**| **0.829**   | **0.743**   |

English minus French on the same questions, paired-bootstrap 95% CIs
(`python -m scripts.significance`):

| mode                     | hit@1 gap               | recall@5 gap            |
|--------------------------|-------------------------|-------------------------|
| BM25                     | −0.32 [−0.44, −0.20]    | −0.51 [−0.61, −0.42]    |
| Dense                    | −0.09 [−0.19, +0.02]    | +0.00 [−0.07, +0.07]    |
| Hybrid (RRF)             | −0.17 [−0.29, −0.05]    | −0.23 [−0.31, −0.15]    |
| Hybrid, language-aware   | −0.22 [−0.33, −0.12]    | −0.16 [−0.24, −0.10]    |
| Hybrid, translated BM25  | **−0.02 [−0.13, +0.07]**| −0.09 [−0.16, −0.01]    |

### What this shows

1. **Hybrid search's advantage is French-only.** Dense retrieval is
   language-neutral but weaker. BM25 is what makes hybrid the best French
   system, and BM25 can't match "residence permit" to "titre de séjour". So
   standard hybrid retrieval builds in a cross-lingual gap of about 23 recall
   points.
2. **Translating the query for BM25 closes most of it.** Translating English
   questions to French (Helsinki-NLP opus-mt-en-fr, a small free model) before
   BM25, while dense keeps the original query, lifts English hit@1 from 0.426
   to 0.574 (+0.15, CI [+0.06, +0.25]) and English recall@5 from 0.602 to 0.743
   (+0.14, CI [+0.06, +0.22]). The hit@1 gap with French becomes
   indistinguishable from zero. French results are untouched.
3. **It isn't a back-translation artefact.** About half the English questions
   were themselves translated from French forum posts, so translating them back
   could just recover the original wording. On the 38 questions originally
   posted in English, the gain holds (recall@5 +0.18, CI [+0.08, +0.30]).
4. **A better translator has little room left.** Giving BM25 the *perfect*
   translation (the human-reviewed French version of each question) as an
   upper bound reaches English recall@5 0.808 vs 0.743 with opus-mt
   (+0.065, CI [0.00, +0.13]) and no gain at hit@1. A small free MT model
   captures most of what translation can offer here.
5. **Simply dropping BM25 for English queries is not enough.** Language-aware
   fusion narrows the recall gap a little, but its edge over plain hybrid isn't
   significant at n=94. Real English questions often contain French terms
   ("titre de séjour", "CAF", 24 of 94), which BM25 can use. Translation keeps
   those terms *and* fixes the rest.

### How the result changed as the data got real

On the first 18 hand-written pairs, language-aware fusion looked like a clean
win. With 104 real, reviewed questions and a harder corpus, it shrank inside
the noise, and a more robust finding came out: the gap comes from the sparse
side, and query translation fixes it. The seed numbers are kept in git history.

## Generation

Claude Haiku 5.5 answers from what `hybrid-trans` retrieved, in the question's
language, and must quote the passage behind every citation. Citations are
checked by code, with no LLM judge.

### How much context should the model see?

End to end: the share of answerable questions answered **and** citing a correct
page (94 per language), against the top-5-passage baseline:

| context given to the model | FR    | EN    | Δ vs top 5 (FR / EN, 95% CI)                      | cost / 208 answers |
|----------------------------|-------|-------|---------------------------------------------------|--------------------|
| top 5 passages             | 0.713 | 0.638 | —                                                 | ≈ $0.14            |
| top 10 passages            | 0.734 | 0.766 | +0.02 [−0.03, +0.07] / +0.13 [+0.05, +0.21]       | ≈ $0.15            |
| top 3 full pages           | 0.862 | 0.777 | +0.15 [+0.06, +0.23] / +0.14 [+0.06, +0.22]       | ≈ $0.60            |
| **top 5 full pages**       | **0.926** | **0.809** | **+0.21 [+0.13, +0.30] / +0.17 [+0.10, +0.26]** | ≈ $1.25        |

Constant across all four runs:

- **Citations:** 100% point at a passage the model was actually given.
- **Quotes:** 99–100% are exact word sequences from the cited page. The rare
  exceptions are a reordered or swapped word, never invented content.
- **Unanswerable questions:** all 20 (10 × 2 languages) correctly answered "not
  found" in every run. More context did not make the model start guessing.

**What this shows**

1. **The bottleneck was the paragraph, not the model.** With 5 passages, the
   model often had the right page but not the paragraph holding the answer, and
   correctly refused. Giving it whole pages lifts end-to-end accuracy by 17–21
   points, at no cost in faithfulness or refusals.
2. **Accuracy costs tokens.** The best setting uses about 9× the input tokens
   of the baseline. At this scale that is still about a cent per 2 answers.
3. **Once generation stops being the bottleneck, the language gap reappears.**
   With full pages, English trails French by 12 points (−0.12, CI [−0.19,
   −0.04]), up from a non-significant 7 points. The remaining gap is the
   retrieval gap from the first half: French queries simply bring back the
   right pages more often. Fixing retrieval for English is still the lever.

### Is the answer actually right? An LLM judge, checked against a human

Citation checks prove the sources are real, not that the answer is correct.
Claude Sonnet 5.5, run as an [Inspect](https://inspect.aisi.org.uk/) task
(`guichet/judge.py`), grades each answer of the best setting (5 full pages)
against the human-reviewed reference: **C** correct, **P** partial (nothing
false, but a key fact or condition missing), **I** incorrect.

| judge verdict (answered questions) | FR (n=87) | EN (n=79) |
|------------------------------------|-----------|-----------|
| Correct                            | 70%       | 66%       |
| Partial                            | 30%       | 28%       |
| Incorrect                          | 0%        | 6% (5)    |

**Can the judge be trusted?** Before seeing any verdict, I graded a fixed
sample of 24 answers by hand (12 FR, 12 EN; `scripts/grade_by_hand.py`):

- **Exact agreement:** 71% (17/24). Cohen's κ = 0.33, which is only *fair*.
- **Direction of the disagreements:** 5 of 7 were the judge saying "correct"
  where I said "partial". The judge is more lenient than a human about
  missing conditions, so its 66–70% "correct" is an **upper bound**. On the
  hand-graded sample, I rated 63% correct vs the judge's 75%.
- **Not yet validated:** neither of us found an incorrect answer in the 24, so
  the judge's 5 "incorrect" verdicts aren't calibrated. Read individually,
  4 of the 5 are *hedged non-answers*: the model reports "found" but then says
  the sources don't settle the question. One (arg-103) is debatable: the
  answer leaves 25-year-olds out of a rule that changes at 25 and at 26.

**Takeaway.** On this task an LLM judge is usable for "is anything wrong?" but
unreliable on the correct-vs-partial boundary. That boundary is subjective even
for a human, and it's where a judge needs a sharper rubric and a held-out
calibration set before its numbers can be reported on their own.

## Try it: the live app

The best measured configuration runs as a web app: ask in English or French,
and get an answer that quotes service-public.gouv.fr word for word, with links,
or an honest "not found" plus a link to search the official site.

```
question ─┬─ dense search (original question)
          └─ BM25 (French translation, if the question isn't French)
              → fusion → top 5 full pages → Claude Haiku → every quote re-checked
```

- **Live translation:** the live app translates questions with Haiku (fractions
  of a cent). The evaluation used opus-mt, and the upper-bound test above shows
  a better translator doesn't hurt.
- **Quotes re-checked:** each quote is verified against its page again before
  display, and an answer whose quotes all fail is shown as "not found".
- **Privacy:** anonymous logging only. No IPs or accounts are kept, and
  e-mails, phone numbers and long digit runs are scrubbed. Logs are deleted
  after 90 days, both on the server and in the backup dataset. The notice is
  at `/privacy`.
- **Feedback:** "Did this answer your question?" on every answer, with an optional comment when the answer is no.
- **Spending:** per-session and daily question caps (cost is about $0.006 per
  question), on top of a spending limit set in the Anthropic console.
- **Operations:** pushing to `main` runs the tests and redeploys
  (`ops/deploy.yml`). A weekly GitHub issue reports usage, unanswered questions,
  👎 feedback, cost and page changes (`ops/weekly.yml`, see "The rules change" below).

The interface is a single hand-built page (`static/index.html`) on a small
FastAPI server (`server.py`). Each question gets a numbered queue ticket, and
each official quote carries a stamp with the page's own « Vérifié le » date.
It works in English and French, and in light and dark mode.

```bash
pip install -e ".[app]"
python app.py                         # http://127.0.0.1:7860 (needs ANTHROPIC_API_KEY in .env)
GUICHET_DEMO=1 python app.py          # page only, canned answers, no key
python -m scripts.deploy_space --space <hf-user>/guichet   # Docker Space (Hugging Face PRO)
```

### Answer style, changed because of a real user

The first pilot question ("as a foreign student working part-time, can I get
the prime d'activité?") got a technically careful answer that talked about
"the passages", hedged, and buried the eligibility conditions in prose. Prompt
**v2** asks for the direct answer first, the conditions as a list with exact
amounts, and the next step. The change was re-run on all 104 pairs before
shipping:

| (pages5, Haiku 5.5)                           | v1          | v2          |
|-----------------------------------------------|-------------|-------------|
| Judged fully correct (FR / EN)                | 70% / 66%   | 85% / 75%   |
| Paired gain in judge score, 95% CI (FR / EN)  |             | +0.08 [+0.03, +0.13] / +0.06 [+0.01, +0.12] |
| Answered, correct page cited (FR / EN)        | 93% / 81%   | 89% / 82%   |
| Invented citations / unanswerable refused     | 0 / 20 of 20 | 0 / 20 of 20 |
| Malformed replies                             | 2           | 5           |

v2 was clearly more correct but broke the reply format more often, which a
student would see as an error. **v2s** keeps the v2 instructions and forces the
reply through a tool call ("fill in this form"), so a malformed reply can't
happen. Because only the delivery changed, v2s was spot-checked rather than
fully re-run: on 24 answers (the 5 that broke plus 19 others, both languages)
there were 0 malformed replies, every citation was valid, and the found/not-found
decisions matched v2. The live app uses v2s. The judge's calibration
(κ = 0.33 against hand grades) was measured on v1 answers.

### Keeping it honest after launch

- **Real questions feed the eval.** `python -m scripts.review_user_questions`
  walks through logged questions, 👎 first. Kept ones become *candidates*, and
  go through the same source-based drafting and human review as every other
  item, so a wrong app answer can never become a "correct" label.
- **The rules change; Guichet keeps up on its own, for free.** Every Monday
  `ops/weekly.yml` runs `python -m scripts.freshness --auto`, which re-downloads
  all pages and applies two different rules:
  - *The app always gets the newest page.* The live official page is the
    truth, so every changed page replaces the stored copy automatically. Only
    the changed paragraphs are re-embedded, and the app is redeployed.
  - *The answer key needs evidence.* A test item citing a changed page stays
    reviewed only if each of its evidence quotes is still on the live page word
    for word and the page's « Vérifié le » date didn't move. Otherwise it goes
    back to unreviewed, so it stops counting in the reported numbers until a
    human re-checks it (`make review`).

  Then the retrieval eval is re-run and compared with the week before. One
  GitHub issue reports all of it, including usage, 👎 feedback and unanswered
  questions (`scripts/weekly_report.py`). Every page update is logged in
  `evals/page_reviews.jsonl`. No paid API is involved.
  The first manual run (16 Oct 2026) found 5 changed pages; all 6 affected
  reference answers were still correct, and every evidence quote was still on
  the live pages. The new versions were accepted, and the international
  driving permit page turned out to have dropped an online tracking service
  that the app would otherwise still have mentioned. Effect on the results:
  BM25-only French recall@5 0.842 → 0.837; nothing else moved. The first
  automatic test run (19 Oct) caught the bank page being officially re-dated
  that same day, and sent its test item back for review.

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
- [x] Generation with checked citations; context-size experiment; LLM judge calibrated against hand grades
- [x] Live web app with feedback loop and weekly freshness monitor
- [ ] Generation layer with programmatic citation verification
- [ ] Faithfulness scoring (Promptfoo or Inspect), with judge-vs-human agreement on a hand-labelled subset
- [ ] Report: numbers, failure taxonomy, limitations

## Limitations

- Source pages carry a "Vérifié le" date; some are from 2023. Answers are only
  as current as the crawl.
- Not legal advice, and not a substitute for the préfecture.
- One embedding model so far (paraphrase-multilingual-MiniLM-L12-v2). Larger
  multilingual models are untested here.
- The language detector is a stopword heuristic. It is correct on all 208
  questions, including English ones full of French terms ("How long is a
  récépissé valid?"), after a fix that counts each accented word once rather
  than each accented character twice. Generation runs used the earlier
  version, which misread 3 English questions as French.

## Licence

Code: MIT. Corpus text: © DILA / service-public.gouv.fr, reused under the
[Licence Ouverte / Open Licence Etalab 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).
Each document keeps its source URL and "Vérifié le" date.
