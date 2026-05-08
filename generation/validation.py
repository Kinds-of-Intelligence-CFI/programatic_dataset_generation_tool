from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from dataset.stimulus import Stimulus
    from generation.generate import Spec


ValidatorFn = Callable[["Stimulus", "Spec"], None]

WILDCARD = "*"


@dataclass(frozen=True)
class ValidatorEntry:
    name: str
    capability: str
    fn: ValidatorFn


_REGISTRY: dict[str, ValidatorEntry] = {}


class ValidationError(Exception):
    def __init__(
        self,
        capability: str,
        validator_name: str,
        sample_id: str | None,
        original: BaseException,
    ) -> None:
        super().__init__(
            f"Validator {validator_name!r} for capability {capability!r} "
            f"rejected sample {sample_id!r}: {original}"
        )
        self.capability = capability
        self.validator_name = validator_name
        self.sample_id = sample_id
        self.original = original


def validates(*, name: str, capability: str) -> Callable[[ValidatorFn], ValidatorFn]:
    """Register a validator.

    `name` is the string recorded in `Stimulus.validators_ran` when this validator
    runs; it must be unique across the registry. `capability` is matched against
    `spec.capabilities`; the literal `"*"` matches every stimulus.
    """

    def decorator(fn: ValidatorFn) -> ValidatorFn:
        if name in _REGISTRY:
            raise ValueError(f"Validator name {name!r} is already registered")
        _REGISTRY[name] = ValidatorEntry(name=name, capability=capability, fn=fn)
        return fn

    return decorator


def get_validator(name: str) -> ValidatorEntry | None:
    return _REGISTRY.get(name)


def registered_capabilities() -> set[str]:
    return {e.capability for e in _REGISTRY.values() if e.capability != WILDCARD}


def run_validators(stimulus: "Stimulus", spec: "Spec") -> None:
    matching = sorted(
        (
            e
            for e in _REGISTRY.values()
            if e.capability == WILDCARD or e.capability in spec.capabilities
        ),
        key=lambda e: e.name,
    )
    for entry in matching:
        try:
            entry.fn(stimulus, spec)
        except BaseException as exc:
            raise ValidationError(
                capability=entry.capability,
                validator_name=entry.name,
                sample_id=stimulus.sample_id,
                original=exc,
            ) from exc
    stimulus.validators_ran = [e.name for e in matching]


def _clear_registry() -> None:
    _REGISTRY.clear()
