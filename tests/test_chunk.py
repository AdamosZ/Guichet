from guichet.chunk import MAX_CHARS, MIN_CHARS, chunk_document
from guichet.corpus import Document


def _doc(text):
    return Document(doc_id="F1", url="https://example", title="Titre", verified_on=None,
                    text=text, n_chars=len(text), fetched_at="2026-10-12")


def test_chunks_keep_title_out_of_body_but_in_embed_text():
    paragraph = "Vous devez demander le renouvellement deux mois avant expiration. " * 4
    chunks = chunk_document(_doc(paragraph))
    assert chunks
    assert not chunks[0].body.startswith("Titre")
    assert chunks[0].embed_text.startswith("Titre\n\n")


def test_chunk_sizes_bounded_and_ids_stable():
    text = "\n\n".join(f"Paragraphe {i}. " + "texte administratif " * 30 for i in range(20))
    chunks = chunk_document(_doc(text))
    assert len(chunks) > 1
    assert all(MIN_CHARS <= len(c.body) <= MAX_CHARS + 400 for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"F1::{i:03d}" for i in range(len(chunks))]
