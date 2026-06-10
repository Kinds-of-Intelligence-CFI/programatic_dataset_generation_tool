import json
from pathlib import Path

import pytest

from generation.glossary import DemandDefinition, Glossary


def _def(desc: str = "d") -> DemandDefinition:
    return DemandDefinition(description=desc, paper_link="http://example.com/p")


def _gloss() -> Glossary:
    return Glossary(
        name="t",
        tree={
            "Social": {
                "Affect": {"class identifier": _def("infer affect")},
                "ToM": _def("track beliefs"),
            },
            "Memory": {
                "Working": _def("hold info briefly"),
            },
            "attention": _def("a top-level demand"),
        },
    )


def test_leaf_short_names_are_the_usable_demands():
    g = _gloss()
    assert g.names == frozenset(
        {"class identifier", "ToM", "Working", "attention"}
    )


def test_grouping_nodes_are_not_usable_demands():
    g = _gloss()
    assert "Social" not in g
    assert "Affect" not in g
    assert "ToM" in g


def test_top_level_demand_has_no_ancestry():
    g = _gloss()
    assert "attention" in g
    assert g.ancestry("attention") == ()
    assert g.category("attention") is None


def test_ancestry_is_root_first_full_path():
    g = _gloss()
    assert g.ancestry("class identifier") == ("Social", "Affect")
    assert g.ancestry("ToM") == ("Social",)


def test_category_is_top_level_group():
    g = _gloss()
    assert g.category("class identifier") == "Social"
    assert g.category("Working") == "Memory"


def test_ancestry_of_unknown_raises():
    g = _gloss()
    with pytest.raises(KeyError):
        g.ancestry("Social")  # a group, not a usable demand


def test_definition_returns_leaf():
    g = _gloss()
    assert g.definition("class identifier").description == "infer affect"


def test_definition_of_unknown_raises():
    g = _gloss()
    with pytest.raises(KeyError):
        g.definition("nope")


def test_unknown_returns_sorted_missing_names():
    g = _gloss()
    assert g.unknown(["ToM", "bogus", "another", "ToM"]) == ["another", "bogus"]


def test_unknown_empty_when_all_present():
    g = _gloss()
    assert g.unknown(["ToM", "Working"]) == []


def test_non_dict_non_definition_value_raises():
    with pytest.raises(TypeError, match="bad"):
        Glossary(name="t", tree={"bad": "just a string"})  # type: ignore[dict-item]


def test_duplicate_leaf_name_raises():
    with pytest.raises(ValueError, match="Duplicate demand name 'Working'"):
        Glossary(
            name="t",
            tree={
                "Memory": {"Working": _def()},
                "Executive": {"Working": _def()},
            },
        )


def test_leaf_name_clashing_with_group_name_raises():
    with pytest.raises(ValueError, match="both a demand and a group"):
        Glossary(
            name="t",
            tree={
                "Social": {"ToM": _def()},
                "ToM": {"sub": _def()},
            },
        )


def test_empty_group_contributes_no_demands():
    g = Glossary(name="t", tree={"Language": {}})
    assert g.names == frozenset()


def test_to_jsonable_round_trips_name_definitions_and_ancestry():
    g = Glossary(
        name="mine",
        tree={
            "Social": {
                "Affect": {
                    "empathy": DemandDefinition(
                        description="feel with",
                        paper_link="http://example.com/paper",
                        notes="n",
                    ),
                },
            },
        },
    )
    out = g.to_jsonable()
    assert out["name"] == "mine"
    assert "separator" not in out
    assert out["demands"]["empathy"] == {
        "description": "feel with",
        "paper_link": "http://example.com/paper",
        "notes": "n",
        "ancestry": ["Social", "Affect"],
    }


# ---- from_file --------------------------------------------------------------


def _write_json(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "gloss.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_from_file_round_trips(tmp_path: Path):
    p = _write_json(
        tmp_path,
        {
            "name": "loaded",
            "tree": {
                "Social": {
                    "Affect": {
                        "class identifier": {
                            "description": "infer affect",
                            "paper_link": "http://example.com/p",
                        }
                    },
                    "ToM": {"description": "beliefs", "paper_link": "http://x"},
                }
            },
        },
    )
    g = Glossary.from_file(p)
    assert g.name == "loaded"
    assert g.names == frozenset({"class identifier", "ToM"})
    assert g.definition("class identifier").description == "infer affect"
    assert g.ancestry("class identifier") == ("Social", "Affect")


def test_from_file_missing_envelope_keys_raises(tmp_path: Path):
    p = _write_json(tmp_path, {"tree": {}})
    with pytest.raises(ValueError, match="'name' and 'tree'"):
        Glossary.from_file(p)


def test_from_file_demand_without_paper_link_raises(tmp_path: Path):
    p = _write_json(
        tmp_path,
        {"name": "x", "tree": {"ToM": {"description": "beliefs"}}},
    )
    with pytest.raises(ValueError, match="missing required 'paper_link'"):
        Glossary.from_file(p)


def test_from_file_reserved_node_name_raises(tmp_path: Path):
    # A reserved word used as a node name (here a group child) is rejected,
    # since from_file() uses those keys to tell a demand from a group.
    p = _write_json(
        tmp_path,
        {"name": "x", "tree": {"notes": {"description": "d", "paper_link": "u"}}},
    )
    with pytest.raises(ValueError, match="reserved"):
        Glossary.from_file(p)


def test_from_file_enforces_leaf_uniqueness(tmp_path: Path):
    p = _write_json(
        tmp_path,
        {
            "name": "x",
            "tree": {
                "A": {"dup": {"description": "1", "paper_link": "u"}},
                "B": {"dup": {"description": "2", "paper_link": "u"}},
            },
        },
    )
    with pytest.raises(ValueError, match="Duplicate demand name 'dup'"):
        Glossary.from_file(p)
