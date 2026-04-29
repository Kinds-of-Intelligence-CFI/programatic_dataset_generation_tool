import pytest

from generation.generate import Spec
from generation.utils import exclude, grid, weighted


# ---- grid ---------------------------------------------------------------


def test_grid_cartesian_product_count():
    specs = grid({"a": [1, 2], "b": [3, 4, 5]})
    assert len(specs) == 6


def test_grid_capabilities_key_becomes_set():
    specs = grid({"capabilities": [set(), {"cap_a"}, {"cap_a", "cap_b"}]})
    cap_sets = [s.capabilities for s in specs]
    assert set() in cap_sets
    assert {"cap_a"} in cap_sets
    assert {"cap_a", "cap_b"} in cap_sets
    for s in specs:
        assert "capabilities" not in s.params


def test_grid_other_keys_become_params():
    specs = grid({"size": [3], "n_distractors": [2]})
    assert len(specs) == 1
    s = specs[0]
    assert s.params == {"size": 3, "n_distractors": 2}
    assert s.capabilities == set()


def test_grid_empty_dict_returns_one_empty_spec():
    specs = grid({})
    assert specs == [Spec()]


def test_grid_empty_value_list_returns_empty():
    assert grid({"a": []}) == []
    assert grid({"a": [1, 2], "b": []}) == []


def test_grid_value_order_matches_itertools_product():
    specs = grid({"a": [1, 2], "b": [10, 20]})
    pairs = [(s.params["a"], s.params["b"]) for s in specs]
    # itertools.product over dict-insertion order: rightmost varies fastest
    assert pairs == [(1, 10), (1, 20), (2, 10), (2, 20)]


def test_grid_capabilities_accepts_any_iterable():
    specs = grid({"capabilities": [["cap_a"], ("cap_a", "cap_b")]})
    cap_sets = [s.capabilities for s in specs]
    assert {"cap_a"} in cap_sets
    assert {"cap_a", "cap_b"} in cap_sets
    for s in specs:
        assert isinstance(s.capabilities, set)


def test_grid_does_not_share_param_dicts():
    specs = grid({"a": [1, 2]})
    specs[0].params["x"] = "mutated"
    assert "x" not in specs[1].params


def test_grid_does_not_share_capability_sets():
    specs = grid({"capabilities": [{"cap_a"}, {"cap_b"}]})
    specs[0].capabilities.add("extra")
    assert "extra" not in specs[1].capabilities


def test_grid_combines_capabilities_with_params():
    specs = grid({
        "capabilities": [set(), {"cap_a"}],
        "size": [3, 4],
    })
    assert len(specs) == 4
    combos = {(frozenset(s.capabilities), s.params["size"]) for s in specs}
    assert combos == {
        (frozenset(), 3),
        (frozenset(), 4),
        (frozenset({"cap_a"}), 3),
        (frozenset({"cap_a"}), 4),
    }


# ---- exclude ------------------------------------------------------------


def test_exclude_drops_predicate_true():
    specs = [Spec(params={"i": i}) for i in range(5)]
    out = exclude(specs, lambda s: s.params["i"] % 2 == 0)
    assert [s.params["i"] for s in out] == [1, 3]


def test_exclude_preserves_order():
    specs = [Spec(params={"i": i}) for i in [3, 1, 4, 1, 5, 9, 2]]
    out = exclude(specs, lambda s: s.params["i"] == 1)
    assert [s.params["i"] for s in out] == [3, 4, 5, 9, 2]


def test_exclude_predicate_no_matches_returns_full_list():
    specs = [Spec(params={"i": i}) for i in range(3)]
    out = exclude(specs, lambda s: False)
    assert out == specs


def test_exclude_empty_input_returns_empty():
    assert exclude([], lambda s: True) == []


def test_exclude_does_not_mutate_input():
    specs = [Spec(params={"i": i}) for i in range(3)]
    snapshot = list(specs)
    exclude(specs, lambda s: s.params["i"] == 0)
    assert specs == snapshot


# ---- weighted -----------------------------------------------------------


def test_weighted_default_only():
    specs = [Spec(params={"i": i}) for i in range(2)]
    out = weighted(specs, {"default": 3})
    assert len(out) == 6
    assert [s.params["i"] for s in out] == [0, 0, 0, 1, 1, 1]


def test_weighted_predicate_overrides_default():
    specs = [Spec(params={"i": i}) for i in range(3)]
    out = weighted(specs, {
        lambda s: s.params["i"] == 1: 4,
        "default": 1,
    })
    counts = {s.params["i"]: 0 for s in specs}
    for s in out:
        counts[s.params["i"]] += 1
    assert counts == {0: 1, 1: 4, 2: 1}


def test_weighted_unmatched_with_no_default_is_weight_one():
    specs = [Spec(params={"i": i}) for i in range(3)]
    out = weighted(specs, {lambda s: s.params["i"] == 0: 5})
    counts = {s.params["i"]: 0 for s in specs}
    for s in out:
        counts[s.params["i"]] += 1
    assert counts == {0: 5, 1: 1, 2: 1}


def test_weighted_unmatched_uses_explicit_default():
    specs = [Spec(params={"i": i}) for i in range(3)]
    out = weighted(specs, {
        lambda s: s.params["i"] == 0: 5,
        "default": 0,
    })
    assert [s.params["i"] for s in out] == [0, 0, 0, 0, 0]


def test_weighted_multiple_predicates_match_raises():
    specs = [Spec(params={"i": 1})]
    with pytest.raises(ValueError):
        weighted(specs, {
            lambda s: s.params["i"] == 1: 2,
            lambda s: s.params["i"] > 0: 3,
        })


def test_weighted_zero_drops_spec():
    specs = [Spec(params={"i": i}) for i in range(3)]
    out = weighted(specs, {
        lambda s: s.params["i"] == 1: 0,
        "default": 1,
    })
    assert [s.params["i"] for s in out] == [0, 2]


def test_weighted_one_is_noop():
    specs = [Spec(params={"i": i}) for i in range(3)]
    out = weighted(specs, {"default": 1})
    assert [s.params["i"] for s in out] == [0, 1, 2]


def test_weighted_copies_are_contiguous():
    specs = [Spec(params={"i": 0}), Spec(params={"i": 1})]
    out = weighted(specs, {"default": 3})
    assert [s.params["i"] for s in out] == [0, 0, 0, 1, 1, 1]


def test_weighted_negative_weight_raises():
    specs = [Spec(params={"i": 0})]
    with pytest.raises(ValueError):
        weighted(specs, {"default": -1})


def test_weighted_non_int_weight_raises():
    specs = [Spec(params={"i": 0})]
    with pytest.raises(TypeError):
        weighted(specs, {"default": 1.5})


def test_weighted_empty_specs_returns_empty():
    assert weighted([], {"default": 5}) == []
