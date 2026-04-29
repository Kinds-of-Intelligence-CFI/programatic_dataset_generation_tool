from dataclasses import dataclass, field
from typing import Any, Literal

from generation.generate import Spec


@dataclass
class Content:
    type: Literal["text", "image"]
    data: str


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Content]


@dataclass
class Stimulus:
    sample_id: str
    spec: Spec
    messages: list[Message]
    target: Any
    invariants_checked: list[str]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.invariants_checked:
            raise ValueError(
                "Stimulus.invariants_checked must be non-empty. "
                "To explicitly opt out of validation, pass ['none']."
            )
