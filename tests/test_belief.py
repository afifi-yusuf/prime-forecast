from prime_forecast.belief import BeliefState, SearchResult, SearchStore


def test_merge_update_clamps_p():
    b = BeliefState()
    b.merge_update({"p": 1.5})
    assert b.p == 0.95
    b.merge_update({"p": -3})
    assert b.p == 0.05
    b.merge_update({"p": "not a number"})
    assert b.p == 0.05  # unchanged


def test_merge_update_accumulates_evidence():
    b = BeliefState()
    b.merge_update({"evidence_for": ["a"], "evidence_against": ["b"]})
    b.merge_update({"evidence_for": ["a", "c"]})  # dedup "a"
    assert b.evidence_for == ["a", "c"]
    assert b.evidence_against == ["b"]
    assert b.step == 2


def test_compact_if_needed_truncates():
    b = BeliefState(evidence_for=["x" * 300] * 10)
    assert b.compact_if_needed(threshold=1000)
    assert len(b.evidence_for) == 5  # 4 kept + truncation marker
    assert "truncated" in b.evidence_for[-1]


def test_search_store_roundtrip():
    store = SearchStore()
    results = [SearchResult(index=0, title="t", url="u", snippet="s", body="b")]
    idx = store.add_search("query", results)
    assert idx == 0
    assert store.get_results(0, [0])[0].title == "t"
    assert store.get_results(0, [5]) == []
    assert store.get_results(3, [0]) == []


def test_from_dict_ignores_unknown_keys():
    b = BeliefState.from_dict({"p": 0.3, "bogus": 1})
    assert b.p == 0.3
