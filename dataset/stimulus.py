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


@dataclass
class ContentAudio:
    audio: str
    format: Literal["wav", "mp3"]
    type: Literal["audio"] = "audio"

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        format: Literal["wav", "mp3"],
    ) -> "ContentAudio":
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg"}[format]
        b64 = base64.b64encode(data).decode("ascii")
        return cls(audio=f"data:{mime};base64,{b64}", format=format)


@dataclass
class ContentVideo:
    video: str
    format: Literal["mp4", "mpeg", "mov"]
    type: Literal["video"] = "video"

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        format: Literal["mp4", "mpeg", "mov"],
    ) -> "ContentVideo":
        mime = {"mp4": "video/mp4", "mpeg": "video/mpeg", "mov": "video/quicktime"}[format]
        b64 = base64.b64encode(data).decode("ascii")
        return cls(video=f"data:{mime};base64,{b64}", format=format)


@dataclass
class ContentDocument:
    document: str
    type: Literal["document"] = "document"
    filename: str | None = None
    mime_type: str | None = None

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        suffix: str,
        filename: str | None = None,
    ) -> "ContentDocument":
        mime = mimetypes.types_map.get(
            f".{suffix.lstrip('.')}", "application/octet-stream"
        )
        b64 = base64.b64encode(data).decode("ascii")
        return cls(
            document=f"data:{mime};base64,{b64}",
            filename=filename,
            mime_type=mime,
        )


Content = ContentText | ContentImage | ContentAudio | ContentVideo | ContentDocument


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
