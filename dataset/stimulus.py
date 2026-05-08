from dataclasses import dataclass, field
from typing import Literal

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
    spec: Spec
    messages: list[Message]
    target: str
    # Leave as None to let the runner auto-assign a sequential id.
    sample_id: str | None = None
    metadata: dict = field(default_factory=dict)
    # Set by the runner after validation. User generators should leave this alone.
    validators_ran: list[str] = field(default_factory=list)
