from guichet.chunk import Chunk
from guichet.index import BM25Index, reciprocal_rank_fusion
from guichet.langid import detect


def test_rrf_equal_weights_matches_plain_rrf():
    a = [(1, 9.0), (2, 8.0)]
    b = [(2, 5.0), (3, 4.0)]
    fused = dict(reciprocal_rank_fusion([a, b]))
    assert fused[2] > fused[1] > fused[3]


def test_rrf_zero_weight_drops_ranking():
    a = [(1, 9.0), (2, 8.0)]
    b = [(3, 5.0), (2, 4.0)]
    fused = reciprocal_rank_fusion([a, b], weights=[1.0, 0.0])
    assert [i for i, _ in fused] == [1, 2]


def test_bm25_folds_accents():
    chunks = [
        Chunk("d1::000", "d1", "u", "Titre de séjour", "Renouveler son titre de séjour étudiant.", 0),
        Chunk("d2::000", "d2", "u", "Retraite", "Demander sa pension de retraite.", 0),
        Chunk("d3::000", "d3", "u", "Impôts", "Déclarer ses revenus en ligne.", 0),
        Chunk("d4::000", "d4", "u", "Logement", "Demander une aide au logement.", 0),
    ]
    hits = BM25Index(chunks).search("titre de sejour", top_k=2)
    assert hits[0][0] == 0


def test_detect_language():
    assert detect("Combien d'heures puis-je travailler ?") == "fr"
    assert detect("titre de séjour") == "fr"
    assert detect("How many hours can I work?") == "en"
    assert detect("What should I do if my residence permit is stolen?") == "en"


def test_every_english_question_has_a_cached_translation():
    # The *-trans modes read translations from disk; CI must never need the MT model.
    from pathlib import Path

    from guichet.golden import load
    from guichet.translate import load_cache

    root = Path(__file__).resolve().parents[1]
    import os

    cwd = os.getcwd()
    os.chdir(root)
    try:
        cache = load_cache()
        missing = [i.item_id for i in load(root / "evals/golden.jsonl") if i.question_en not in cache]
    finally:
        os.chdir(cwd)
    assert not missing, f"run `python -m guichet.translate`; missing: {missing[:5]}"


def test_dense_cache_reembeds_only_changed_chunks(tmp_path, monkeypatch):
    import numpy as np

    from guichet.chunk import Chunk
    from guichet.index import DenseIndex

    calls = []

    def fake_encode(self, texts):
        calls.append(len(texts))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    monkeypatch.setattr(DenseIndex, "encode", fake_encode)
    chunks = [Chunk(f"F{i}-0", f"F{i}", "u", "t", f"body {i}", 0) for i in range(4)]
    DenseIndex(chunks, cache_dir=tmp_path)
    assert calls == [4]
    DenseIndex(chunks, cache_dir=tmp_path)            # exact cache hit
    assert calls == [4]
    chunks[2] = Chunk("F2-0", "F2", "u", "t", "body 2, rewritten by the ministry", 0)
    index = DenseIndex(chunks, cache_dir=tmp_path)    # same ids, new text
    assert calls == [4, 1] and index.reused == 3
    assert index.vectors[2][0] == len(chunks[2].embed_text)
