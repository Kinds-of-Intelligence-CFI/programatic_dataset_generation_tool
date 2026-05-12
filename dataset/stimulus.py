import base64
import mimetypes
from dataclasses import dataclass, field
from typing import Literal

from generation.generate import Spec


@dataclass
class ContentText:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ContentImage:
    image: str
    type: Literal["image"] = "image"
    detail: Literal["auto", "low", "high"] = "auto"

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        suffix: str,
        detail: Literal["auto", "low", "high"] = "auto",
    ) -> "ContentImage":
        mime = mimetypes.types_map.get(f".{suffix.lstrip('.')}", "application/octet-stream")
        b64 = base64.b64encode(data).decode("ascii")
        return cls(image=f"data:{mime};base64,{b64}", detail=detail)


Content = ContentText | ContentImage


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Content]


@dataclass
class Stimulus:
    spec: Spec
    messages: list[Message]
    target: str
    sample_id: str | None = None
    metadata: dict = field(default_factory=dict)
    validators_ran: list[str] = field(default_factory=list)
