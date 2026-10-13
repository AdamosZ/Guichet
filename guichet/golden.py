"""The golden set: the answer key the whole project rests on.

Every question exists as a matched pair — the same question in French and in
English — because the headline measurement is the gap between them. Pairing
them means the comparison is not confounded by the two language sets asking
about different things.

On how the labels were made, stated plainly because it decides how much the
numbers are worth: candidate questions and draft answers were written by
reading the source documents directly, never by querying the system under
test. A human then reviewed every label and corrected it. Labels that were
not human-reviewed are marked `reviewed: false` and are excluded from the
headline metrics, because an answer key the system wrote for itself measures
nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

GOLDEN_PATH = Path("evals/golden.jsonl")


@dataclass
class GoldenItem:
    """One question, asked in both languages, with its verified answer."""

    item_id: str
    question_fr: str
    question_en: str
    # The documents that genuinely contain the answer. More than one is normal
    # and is why recall@k is measured rather than plain accuracy.
    gold_doc_ids: list[str]
    # The answer a correct system should give, in prose. Used for faithfulness
    # judging, not for string matching.
    answer_fr: str
    answer_en: str
    category: str
    # Questions the corpus genuinely cannot answer. A system that answers these
    # anyway is hallucinating, and that is the single most important thing to
    # measure, so they are deliberately included.
    answerable: bool = True
    reviewed: bool = False
    # Who signed the item off: "adam" (human) or "claude" (verified against the
    # source page by Claude at Adam's request). Reported separately.
    reviewed_by: str = ""
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    # Where the question came from: a real post by a real person, paraphrased
    # and stripped of personal details. Empty for the original seed items.
    source_url: str = ""
    source_site: str = ""
    # Verbatim sentences from the gold documents that support the answer, so a
    # reviewer can check the label without reading the whole page.
    evidence: list[dict] = field(default_factory=list)
    # Result of the independent pre-check (a second pass by someone who did not
    # write the item). Advisory only: it never replaces human review.
    check: dict = field(default_factory=dict)


CATEGORIES = (
    "sejour",      # titres de séjour, récépissé, renouvellement
    "visa",        # visas, entrée
    "etudes",      # statut étudiant, inscription
    "travail",     # droit au travail, stage, alternance
    "logement",    # APL, CAF, bail
    "sante",       # sécurité sociale, CPAM
    "argent",      # banque, impôts, frais
    "procedure",   # délais, pièces, où déposer
    "unanswerable",
)


def load(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [GoldenItem(**json.loads(line)) for line in handle if line.strip()]


def save(items: list[GoldenItem], path: Path = GOLDEN_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def validate(items: list[GoldenItem], known_doc_ids: set[str]) -> list[str]:
    """Catch the label mistakes that silently corrupt every downstream number."""
    problems: list[str] = []
    seen_ids: set[str] = set()

    for item in items:
        where = f"[{item.item_id}]"
        if item.item_id in seen_ids:
            problems.append(f"{where} duplicate item_id")
        seen_ids.add(item.item_id)

        if item.category not in CATEGORIES:
            problems.append(f"{where} unknown category {item.category!r}")

        if not item.question_fr.strip() or not item.question_en.strip():
            problems.append(f"{where} missing one language of the question pair")

        if item.answerable:
            if not item.gold_doc_ids:
                problems.append(f"{where} answerable but no gold_doc_ids")
            unknown = set(item.gold_doc_ids) - known_doc_ids
            if unknown:
                problems.append(f"{where} gold_doc_ids not in corpus: {sorted(unknown)}")
            if not item.answer_en.strip() or not item.answer_fr.strip():
                problems.append(f"{where} answerable but missing an answer")
        else:
            if item.gold_doc_ids:
                problems.append(f"{where} marked unanswerable but has gold_doc_ids")

    return problems


def summary(items: list[GoldenItem]) -> str:
    reviewed = sum(1 for i in items if i.reviewed)
    unanswerable = sum(1 for i in items if not i.answerable)
    by_category: dict[str, int] = {}
    for item in items:
        by_category[item.category] = by_category.get(item.category, 0) + 1

    lines = [
        f"{len(items)} items  ({reviewed} human-reviewed, {len(items) - reviewed} pending)",
        f"{unanswerable} deliberately unanswerable",
        "by category: " + ", ".join(f"{k} {v}" for k, v in sorted(by_category.items())),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    from .corpus import load_corpus

    items = load()
    known = {d.doc_id for d in load_corpus(Path("data/corpus/documents.jsonl"))}
    print(summary(items))
    problems = validate(items, known)
    if problems:
        print(f"\n{len(problems)} problems:")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    print("\nno problems found")
