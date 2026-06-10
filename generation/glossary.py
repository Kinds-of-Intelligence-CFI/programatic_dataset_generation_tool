from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Union


@dataclass(frozen=True)
class DemandDefinition:
    """A single usable demand: the leaf of a glossary tree."""

    description: str
    paper_link: str
    notes: str | None = None


GlossaryTree = dict[str, Union["GlossaryTree", DemandDefinition]]

# Keys reserved for a DemandDefinition in the JSON form; a group or demand name
# must not collide with these or from_file() cannot tell a leaf from a group.
_RESERVED_KEYS = frozenset({"description", "paper_link", "notes"})


class Glossary:
    """A tree of allowed demand names.

    Authored as a nested dict. A dict value is a grouping node - organizational
    only, never a usable demand. A DemandDefinition value is a usable demand
    keyed by its own (short) name; that name is what researchers put in
    ``spec.demands``, regardless of how deep it sits in the tree. The groups
    above a demand are kept as its ancestry for grouping and analysis.

    Demand (leaf) names must be unique across the whole tree, and a demand name
    must not also be used as a group name.
    """

    def __init__(self, tree: GlossaryTree, *, name: str) -> None:
        self.name = name
        self._defs: dict[str, DemandDefinition] = {}
        self._ancestry: dict[str, tuple[str, ...]] = {}
        self._groups: set[str] = set()
        self._walk(tree, ())

        clashes = sorted(self._defs.keys() & self._groups)
        if clashes:
            raise ValueError(
                f"In glossary {self.name!r} these names are used as both a demand "
                f"and a group: {', '.join(clashes)}"
            )

    def _walk(self, node: GlossaryTree, ancestry: tuple[str, ...]) -> None:
        for key, value in node.items():
            if isinstance(value, DemandDefinition):
                if key in self._defs:
                    raise ValueError(
                        f"Duplicate demand name {key!r} in glossary {self.name!r}"
                    )
                self._defs[key] = value
                self._ancestry[key] = ancestry
            elif isinstance(value, dict):
                self._groups.add(key)
                self._walk(value, ancestry + (key,))
            else:
                raise TypeError(
                    f"Glossary node {key!r} must be a dict (grouping node) or a "
                    f"DemandDefinition (usable demand), got {type(value).__name__}"
                )

    def __contains__(self, name: object) -> bool:
        return name in self._defs

    @property
    def names(self) -> frozenset[str]:
        """The set of usable demand names (leaves only, not grouping nodes)."""
        return frozenset(self._defs)

    def unknown(self, names: Iterable[str]) -> list[str]:
        """Return the given names that are not usable demands, sorted."""
        return sorted(n for n in set(names) if n not in self._defs)

    def definition(self, name: str) -> DemandDefinition:
        try:
            return self._defs[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a demand in glossary {self.name!r}"
            ) from None

    def ancestry(self, name: str) -> tuple[str, ...]:
        """The groups above a demand, root first; empty if it is top-level."""
        if name not in self._defs:
            raise KeyError(f"{name!r} is not a demand in glossary {self.name!r}")
        return self._ancestry[name]

    def category(self, name: str) -> str | None:
        """The top-level group a demand sits under, or None - the analysis bucket."""
        anc = self.ancestry(name)
        return anc[0] if anc else None

    def to_jsonable(self) -> dict:
        return {
            "name": self.name,
            "demands": {
                n: {
                    "description": d.description,
                    "paper_link": d.paper_link,
                    "notes": d.notes,
                    "ancestry": list(self._ancestry[n]),
                }
                for n, d in sorted(self._defs.items())
            },
        }

    @classmethod
    def from_file(cls, path: str | Path) -> "Glossary":
        """Load a glossary from a JSON file of the form
        ``{"name": str, "tree": {...}}``.

        In the tree, a node object is a *demand* if it contains a "description"
        key (it must then also have "paper_link"); otherwise it is a *group*
        whose keys are child node names. Group and demand names may not be one
        of the reserved keys (description, paper_link, notes).
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "name" not in data or "tree" not in data:
            raise ValueError(
                f"Glossary file {str(path)!r} must be a JSON object with "
                f"'name' and 'tree' keys"
            )
        tree = cls._tree_from_jsonable(data["tree"], data["name"])
        return cls(tree, name=data["name"])

    @staticmethod
    def _tree_from_jsonable(node: Any, glossary_name: str) -> GlossaryTree:
        if not isinstance(node, dict):
            raise ValueError(
                f"Glossary {glossary_name!r}: expected an object for a group, "
                f"got {type(node).__name__}"
            )
        out: GlossaryTree = {}
        for key, value in node.items():
            if key in _RESERVED_KEYS:
                raise ValueError(
                    f"Glossary {glossary_name!r}: node name {key!r} is reserved; "
                    f"group and demand names may not be one of {sorted(_RESERVED_KEYS)}"
                )
            if isinstance(value, dict) and "description" in value:
                if "paper_link" not in value:
                    raise ValueError(
                        f"Glossary {glossary_name!r}: demand {key!r} is missing "
                        f"required 'paper_link'"
                    )
                out[key] = DemandDefinition(
                    description=value["description"],
                    paper_link=value["paper_link"],
                    notes=value.get("notes"),
                )
            else:
                out[key] = Glossary._tree_from_jsonable(value, glossary_name)
        return out


# Default glossary used by the team at LCFI - update as new papers come out 
DEFAULT_GLOSSARY = Glossary(
    name="LCFI-default",
    tree={
        "Learning": {
            "Heuristic": {},
            "Concept": {},
            "Abstract": {},
            "Predict": {},
        },
        "Attention": {
            "Focus": {},
            "Search": {},
            "Gist": {},
        },
        "Language": {
            "Create": {},
            "Parse": {},
            "Adapt": {},
            "Apply": {},
        },
        "Critical": {
            "Reflexive": {},
            "Persona": {},
            "Integrate": {},
        },
        "Social": {
            "ToM": {},
            "Affect": {},
            "Relation": {},
        },
        "Spatial": {
            "Centering": {},
            "Temporal": {},
            "Relative": {},
        },
        "Executive": {
            "Planning": {},
            "Reasoning": {},
            "Decision": {},
        },
        "Memory": {
            "Working": {},
            "Procedure": {},
            "Episodic": {},
            "Semantic": {},
        },
    },
)
