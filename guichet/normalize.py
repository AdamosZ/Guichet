"""French text normalisation for retrieval.

Two different jobs, deliberately kept apart:

- `display_clean` tidies text that a human or a model will read. It keeps
  accents, case and punctuation, because "a" and "à" are different words and
  a model that reads de-accented French answers worse.
- `index_text` produces the token stream used for keyword (BM25) matching.
  It folds accents, case and elisions, because a user typing "titre de sejour"
  on an English keyboard should still match "titre de séjour".

Applying the index-side folding to text handed to the model is a real and
easy mistake; it costs answer quality for no retrieval gain.
"""

from __future__ import annotations

import re
import unicodedata

# Apostrophes, quotes and spaces that French pages use interchangeably.
_APOSTROPHES = dict.fromkeys(map(ord, "’‘ʼ´`"), "'")
_SPACES = dict.fromkeys(map(ord, "     "), " ")
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")

_LIGATURES = {"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE", "ﬁ": "fi", "ﬂ": "fl"}

# Tooltip markers service-public.gouv.fr leaves in the rendered text.
_ARTEFACTS = (
    re.compile(r"\s*:\s*titleContent\b"),
    re.compile(r"\bTout déplier\b"),
    re.compile(r"\bTout replier\b"),
)

# Elisions: l'étudiant -> l étudiant, qu'il -> qu il. Without this, BM25 sees
# "l'étudiant" as one token and a query for "étudiant" misses it.
_ELISION = re.compile(
    r"\b(l|d|j|n|s|c|t|m|qu|jusqu|lorsqu|puisqu|quoiqu)'", flags=re.IGNORECASE
)

# Administrative acronyms. Users write one form, the sources use the other.
# This is the French counterpart of entity resolution: cheap, unglamorous, and
# responsible for more recall than any model choice.
ACRONYMS: dict[str, str] = {
    "caf": "caisse allocations familiales",
    "apl": "aide personnalisee au logement",
    "als": "allocation de logement sociale",
    "ofii": "office francais immigration integration",
    "vls-ts": "visa long sejour valant titre de sejour",
    "vlsts": "visa long sejour valant titre de sejour",
    "ts": "titre de sejour",
    "crous": "centre regional oeuvres universitaires scolaires",
    "cpam": "caisse primaire assurance maladie",
    "urssaf": "union recouvrement securite sociale allocations familiales",
    "ceseda": "code entree sejour etrangers droit asile",
    "rib": "releve identite bancaire",
    "caf.fr": "caisse allocations familiales",
    "anef": "administration numerique etrangers en france",
    "dcem": "document circulation etranger mineur",
    "tsa": "titre de sejour",
    "pacs": "pacte civil de solidarite",
    "cv": "curriculum vitae",
    "siret": "systeme identification repertoire etablissements",
    "ameli": "assurance maladie en ligne",
    "msa": "mutualite sociale agricole",
    "caq": "certificat acceptation quebec",
}


def strip_accents(text: str) -> str:
    """Fold accented Latin characters onto their base letters."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def display_clean(text: str) -> str:
    """Tidy text for reading. Keeps accents, case and meaning."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_APOSTROPHES).translate(_SPACES).translate(_DASHES)
    for pattern in _ARTEFACTS:
        text = pattern.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalise_numbers(text: str) -> str:
    """Make French and English number formats comparable.

    French writes 1 234,56 where English writes 1,234.56. A query for
    "1200 euros" should match a page that says "1 200 €".
    """
    text = re.sub(r"(?<=\d)[    ](?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    text = text.replace("€", " euros ")
    return text


def expand_acronyms(text: str) -> str:
    """Append the expansion of any known acronym, keeping the original.

    Both forms are kept so a query matches whichever one it used.
    """
    out: list[str] = []
    for token in text.split():
        out.append(token)
        key = token.strip(".,;:()[]'\"").lower()
        expansion = ACRONYMS.get(key)
        if expansion:
            out.append(expansion)
    return " ".join(out)


def index_text(text: str) -> str:
    """Produce the folded token stream used for keyword matching."""
    text = display_clean(text)
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    text = normalise_numbers(text)
    text = _ELISION.sub(r"\1 ", text)
    text = strip_accents(text).lower()
    text = expand_acronyms(text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Short, high-frequency French and English words carry no retrieval signal.
STOPWORDS = frozenset(
    """
    a au aux avec ce ces dans de des du elle en et eux il je la le les leur lui
    ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui
    sa se ses son sur ta te tes toi ton tu un une vos votre vous c d j l m n s t
    y est sont etre avoir fait plus peut doit si comme tout tous toute
    the of and to in is are for on with as at by be can do does how what when
    where which who you your i it this that from or if my an a
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Tokenise already-normalised text for BM25, dropping stopwords."""
    return [t for t in index_text(text).split() if t not in STOPWORDS and len(t) > 1]
