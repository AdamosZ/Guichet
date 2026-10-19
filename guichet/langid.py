"""Tiny query-language detector: French or not.

Language-aware fusion only needs one bit — does the query share the corpus
language? — so a stopword-and-diacritics heuristic is enough and avoids a
model dependency. Accented *words* count as French evidence, once each: real
English questions borrow French terms ("How long is a récépissé valid?"), so
counting every accented character would let one borrowed word outvote the
English around it.
"""

from __future__ import annotations

import re

_FR = frozenset("""
le la les un une des du de au aux et ou est sont mon ma mes ton ta tes son sa ses
je tu il elle nous vous ils elles on que qui quoi quel quelle quels quelles
dans pour par sur avec sans pas ne plus combien comment quand pourquoi
dois doit peut peux puis faire être avoir mon votre vos notre nos ce cette ces
""".split())

_EN = frozenset("""
the a an of to in for on with without and or is are am be can could should would
do does did how what when where why which who my your his her our their i you he
she we they it this that these those many much long after before if
""".split())

_ACCENTED = re.compile(r"[àâäçéèêëîïôöùûüÿœæ]", re.IGNORECASE)
_WORD = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæ']+", re.IGNORECASE)


def detect(text: str) -> str:
    """Return "fr" or "en". Ties go to "fr", the corpus language."""
    words = [w.lower().split("'")[-1] for w in _WORD.findall(text)]
    fr = sum(w in _FR for w in words) + sum(bool(_ACCENTED.search(w)) for w in words if w not in _EN)
    en = sum(w in _EN for w in words)
    return "en" if en > fr else "fr"
