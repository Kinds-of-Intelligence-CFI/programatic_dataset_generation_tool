from __future__ import annotations

import itertools
from typing import Any, Callable, Iterable, Literal

from generation.generate import SampleSpec


DEMANDS_KEY = "demands"
DEFAULT_KEY = "default"

Predicate = Callable[[SampleSpec], bool]
WeightMap = dict[Predicate | Literal["default"], int]


def grid(grid_spec: dict[str, Iterable[Any]]) -> list[SampleSpec]:
    """Cartesian product of a dict of value-lists into a list of SampleSpecs.

    The key "demands" is special: each value at that key becomes the
    full set of demands for that combination. All other keys are placed
    verbatim into SampleSpec.params.
    """
    if not grid_spec:
        return [SampleSpec()]

    keys = list(grid_spec.keys())
    value_lists = [list(grid_spec[k]) for k in keys]

    out: list[SampleSpec] = []
    for combo in itertools.product(*value_lists):
        demands: set[str] = set()
        params: dict[str, Any] = {}
        for key, val in zip(keys, combo):
            if key == DEMANDS_KEY:
                demands = set(val)
            else:
                params[key] = val
        out.append(SampleSpec(demands=demands, params=params))
    return out


def exclude(specs: list[SampleSpec], predicate: Predicate) -> list[SampleSpec]:
    """Return specs for which `predicate(spec)` is False."""
    return [s for s in specs if not predicate(s)]


def weighted(specs: list[SampleSpec], weight_map: WeightMap) -> list[SampleSpec]:
    """Duplicate each spec by an integer weight chosen from `weight_map`.

    Keys in `weight_map` are predicates `Callable[[SampleSpec], bool]` or the literal
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

    out: list[SampleSpec] = []
    for spec in specs:
        matches = [(pred, w) for pred, w in predicates if pred(spec)]
        if len(matches) > 1:
            raise ValueError(
                f"SampleSpec {spec!r} matched {len(matches)} predicates in weight_map; "
                "predicates must be mutually exclusive"
            )
        weight = matches[0][1] if matches else default_weight
        out.extend([spec] * weight)
    return out
