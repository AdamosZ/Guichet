from guichet.generate import parse_reply, score_row

CTX = {
    "item_id": "x", "language": "en", "answerable": True, "gold_doc_ids": ["F1"],
    "passages": [
        {"chunk_id": "F1::000", "doc_id": "F1", "title": "t", "text": "Le récépissé est valable 4 mois."},
        {"chunk_id": "F9::000", "doc_id": "F9", "title": "t", "text": "Autre chose."},
    ],
}


def test_parse_reply_extracts_json_and_flags_garbage():
    assert parse_reply('Sure: {"found": true, "answer": "a", "citations": []}')["found"] is True
    assert parse_reply("no json here")["parse_error"] is True


def test_quote_must_appear_in_the_cited_chunk():
    good = {"found": True, "citations": [{"chunk_id": "F1::000", "quote": "valable 4 mois"}]}
    invented = {"found": True, "citations": [{"chunk_id": "F1::000", "quote": "valable 6 mois"}]}
    assert score_row(good, CTX)["quote_faithful_rate"] == 1.0
    assert score_row(invented, CTX)["quote_faithful_rate"] == 0.0


def test_citing_an_unretrieved_chunk_is_invalid():
    gen = {"found": True, "citations": [{"chunk_id": "F7::000", "quote": "x"}]}
    row = score_row(gen, CTX)
    assert row["citation_valid_rate"] == 0.0 and row["cites_gold"] is False


def test_abstention_recorded():
    assert score_row({"found": False, "citations": []}, CTX)["abstained"] is True


def test_wrong_answer_language_is_caught():
    gen = {"found": True, "answer": "Le récépissé est valable 4 mois.", "citations": []}
    assert score_row(gen, CTX)["answer_language_ok"] is False  # CTX question is English


def test_quote_check_ignores_spacing_but_not_wording():
    from guichet.generate import quote_in

    page = "Il permet de circuler dans l'espace Schengen , à condition de ne pas dépasser 90 jours."
    assert quote_in("dans l'espace Schengen, à condition", page)
    assert not quote_in("dans l'espace Schengen, sous condition", page)
    assert not quote_in("à condition dans l'espace Schengen", page)


def test_structured_reply_is_read_from_the_tool_call():
    from types import SimpleNamespace

    from guichet.generate import read_message, request_kwargs

    form = SimpleNamespace(type="tool_use", input={"found": True, "answer": "Oui.", "citations": []})
    raw, reply = read_message(SimpleNamespace(content=[form]))
    assert reply["found"] is True and "parse_error" not in reply
    assert request_kwargs("v2s")["tool_choice"]["name"] == "submit_answer"
    assert "tools" not in request_kwargs("v1")
