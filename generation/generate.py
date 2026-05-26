from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dataset.stimulus import Stimulus


@dataclass
class SampleSpec:
    demands: set[str] = field(default_factory=set)
    params: dict = field(default_factory=dict)


Condition = list[SampleSpec]


class GenerateFn(Protocol):
    """Signature for a user-supplied stimulus generator.

    Implementations should use `rng` for all randomness. Reaching for module-level
    `random` or `numpy.random` breaks the runner's determinism contract.
    """

    def __call__(self, spec: SampleSpec, rng: random.Random) -> Stimulus: ...
