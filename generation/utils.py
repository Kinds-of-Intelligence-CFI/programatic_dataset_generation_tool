from __future__ import annotations

import itertools
from typing import Any, Callable, Iterable, Literal

from generation.generate import Spec


CAPABILITIES_KEY = "capabilities"
DEFAULT_KEY = "default"

Predicate = Callable[[Spec], bool]
WeightMap = dict[Predicate | Literal["default"], int]


def grid(grid_spec: dict[str, Iterable[Any]]) -> list[Spec]:
    """Cartesian product of a dict of value-lists into a list of Specs.

    The key "capabilities" is special: each value at that key becomes the
    full set of capabilities for that combination. All other keys are placed
    verbatim into Spec.params.
    """
    if not grid_spec:
        return [Spec()]

    keys = list(grid_spec.keys())
    value_lists = [list(grid_spec[k]) for k in keys]

    out: list[Spec] = []
    for combo in itertools.product(*value_lists):
        capabilities: set[str] = set()
        params: dict[str, Any] = {}
        for key, val in zip(keys, combo):
            if key == CAPABILITIES_KEY:
                capabilities = set(val)
            else:
                params[key] = val
        out.append(Spec(capabilities=capabilities, params=params))
    return out


def exclude(specs: list[Spec], predicate: Predicate) -> list[Spec]:
    """Return specs for which `predicate(spec)` is False."""
    return [s for s in specs if not predicate(s)]


def weighted(specs: list[Spec], weight_map: WeightMap) -> list[Spec]:
    """Duplicate each spec by an integer weight chosen from `weight_map`.

    Keys in `weight_map` are predicates `Callable[[Spec], bool]` or the literal
    string "default". Each spec must match at most one predicate; matching
    multiple predicates is an error so the bucket structure stays explicit.
    Specs matching no predicate use "default" if present, else weight 1.
    """
    for key, value in weight_map.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"weight_map values must be int, got {type(value).__name__} for key {key!r}"
            )
        if value < 0:
            raise ValueError(f"weight_map values must be non-negative, got {value} for key {key!r}")

    predicates: list[tuple[Predicate, int]] = [
        (k, v) for k, v in weight_map.items() if k != "default"  # type: ignore[misc]
    ]
    default_weight: int = weight_map.get("default", 1)

    out: list[Spec] = []
    for spec in specs:
        matches = [(pred, w) for pred, w in predicates if pred(spec)]
        if len(matches) > 1:
            raise ValueError(
                f"Spec {spec!r} matched {len(matches)} predicates in weight_map; "
                "predicates must be mutually exclusive"
            )
        weight = matches[0][1] if matches else default_weight
        out.extend([spec] * weight)
    return out
