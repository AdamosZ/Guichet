from scripts.judge_agreement import cohen_kappa


def test_kappa_perfect_and_chance():
    assert cohen_kappa([("C", "C"), ("I", "I"), ("P", "P")]) == 1.0
    # Judge always says C: high raw agreement on a C-heavy set, but kappa ~ 0.
    pairs = [("C", "C")] * 9 + [("I", "C")]
    assert abs(cohen_kappa(pairs)) < 1e-9


def test_hand_grading_sample_is_fixed_and_balanced():
    from scripts.grade_by_hand import pick

    first, second = pick("pages5"), pick("pages5")
    assert [s.id for s in first] == [s.id for s in second]
    assert sum(s.metadata["language"] == "fr" for s in first) == 12
