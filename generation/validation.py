from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from dataset.stimulus import Stimulus
    from generation.generate import SampleSpec


ValidatorFn = Callable[["Stimulus", "SampleSpec"], None]

WILDCARD = "*"


@dataclass(frozen=True)
class ValidatorEntry:
    name: str
    demand: str
    fn: ValidatorFn
    level: int | None = None


_REGISTRY: dict[str, ValidatorEntry] = {}


class ValidationError(Exception):
    def __init__(
        self,
        demand: str,
        validator_name: str,
        sample_id: str | None,
        original: BaseException,
    ) -> None:
        super().__init__(
            f"Validator {validator_name!r} for demand {demand!r} "
            f"rejected sample {sample_id!r}: {original}"
        )
        self.demand = demand
        self.validator_name = validator_name
        self.sample_id = sample_id
        self.original = original


def validates(
    *, name: str, demand: str, level: int | None = None
) -> Callable[[ValidatorFn], ValidatorFn]:
    """Register a validator.

    `name` is the string recorded in `Stimulus.validators_ran` when this validator
    runs; it must be unique across the registry. `demand` is matched against
    `spec.demands`; the literal `"*"` matches every stimulus. If `level` is set
    the validator only runs when `spec.demands[demand] == level` - in particular
    `level=0` makes the validator an "assert this demand is absent" check.
    """

    def decorator(fn: ValidatorFn) -> ValidatorFn:
        if name in _REGISTRY:
            raise ValueError(f"Validator name {name!r} is already registered")
        _REGISTRY[name] = ValidatorEntry(name=name, demand=demand, fn=fn, level=level)
        return fn

    return decorator


def get_validator(name: str) -> ValidatorEntry | None:
    return _REGISTRY.get(name)


def registered_demands() -> set[str]:
    return {e.demand for e in _REGISTRY.values() if e.demand != WILDCARD}


def run_validators(stimulus: "Stimulus", spec: "SampleSpec") -> None:
    matching = sorted(
        (e for e in _REGISTRY.values() if _entry_matches(e, spec)),
        key=lambda e: e.name,
    )
    for entry in matching:
        try:
            entry.fn(stimulus, spec)
        except BaseException as exc:
            raise ValidationError(
                demand=entry.demand,
                validator_name=entry.name,
                sample_id=stimulus.sample_id,
                original=exc,
            ) from exc
    stimulus.validators_ran = [e.name for e in matching]


def _entry_matches(entry: ValidatorEntry, spec: "SampleSpec") -> bool:
    if entry.demand == WILDCARD:
        return True
    if entry.demand not in spec.demands:
        return False
    return entry.level is None or spec.demands[entry.demand] == entry.level


def _clear_registry() -> None:
    _REGISTRY.clear()
