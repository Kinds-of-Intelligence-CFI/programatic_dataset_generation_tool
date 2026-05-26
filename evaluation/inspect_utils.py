import json
from pathlib import Path
from typing import Any

from inspect_ai._util.content import (
    Content as InspectContent,
)
from inspect_ai._util.content import (
    ContentAudio,
    ContentDocument,
    ContentImage,
    ContentText,
    ContentVideo,
)
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)

_ROLE_TO_CLASS: dict[str, type[ChatMessage]] = {
    "system": ChatMessageSystem,
    "user": ChatMessageUser,
    "assistant": ChatMessageAssistant,
    "tool": ChatMessageTool,
}


def load_dataset(path: str | Path) -> MemoryDataset:
    dataset_dir = Path(path)
    manifest_path = dataset_dir / "manifest.json"
    jsonl_path = dataset_dir / "stimuli.jsonl"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {dataset_dir}")
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"stimuli.jsonl not found in {dataset_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = manifest.get("name")

    samples: list[Sample] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(_record_to_sample(json.loads(line), dataset_dir))

    return MemoryDataset(samples=samples, name=name, location=str(dataset_dir))


def _record_to_sample(record: dict[str, Any], dataset_dir: Path) -> Sample:
    messages = record["messages"]

    if _is_single_user_text(messages):
        sample_input: str | list[ChatMessage] = messages[0]["content"]
    else:
        sample_input = [_message_to_chat(m, dataset_dir) for m in messages]

    metadata: dict[str, Any] = {
        "_stimulus": {
            "spec": record["spec"],
            "functional": record.get("functional"),
            "validators_ran": record.get("validators_ran", []),
            "dataset_dir": str(dataset_dir),
        },
        **record.get("metadata", {}),
    }

    return Sample(
        id=record["sample_id"],
        input=sample_input,
        target=record["target"],
        metadata=metadata,
    )


def _is_single_user_text(messages: list[dict[str, Any]]) -> bool:
    return (
        len(messages) == 1
        and messages[0]["role"] == "user"
        and isinstance(messages[0]["content"], str)
    )


def _message_to_chat(message: dict[str, Any], dataset_dir: Path) -> ChatMessage:
    role = message["role"]
    cls = _ROLE_TO_CLASS.get(role)
    if cls is None:
        raise ValueError(f"Unknown message role: {role!r}")
    content = message["content"]
    if isinstance(content, str):
        return cls(content=content)
    parts = [_content_part_to_inspect(p, dataset_dir) for p in content]
    return cls(content=parts)


def _content_part_to_inspect(
    part: dict[str, Any], dataset_dir: Path
) -> InspectContent:
    t = part.get("type")
    if t == "text":
        return ContentText(text=part["text"])
    if t == "image":
        return ContentImage(
            image=_resolve_asset_path(part["image"], dataset_dir),
            detail=part.get("detail", "auto"),
        )
    if t == "audio":
        return ContentAudio(
            audio=_resolve_asset_path(part["audio"], dataset_dir),
            format=part["format"],
        )
    if t == "video":
        return ContentVideo(
            video=_resolve_asset_path(part["video"], dataset_dir),
            format=part["format"],
        )
    if t == "document":
        doc_kwargs: dict[str, Any] = {
            "document": _resolve_asset_path(part["document"], dataset_dir),
        }
        if part.get("filename") is not None:
            doc_kwargs["filename"] = part["filename"]
        if part.get("mime_type") is not None:
            doc_kwargs["mime_type"] = part["mime_type"]
        return ContentDocument(**doc_kwargs)
    raise ValueError(f"Unknown content type in dataset: {t!r}")


def _resolve_asset_path(image_ref: str, dataset_dir: Path) -> str:
    """Turn an ``assets/...`` relative reference into an absolute path.

    Pass URLs, data URIs, and absolute paths through unchanged. If the
    relative path doesn't resolve to a file on disk, leave it as-is so
    diagnostic errors point at the original reference.
    """
    if image_ref.startswith(("http://", "https://", "data:")):
        return image_ref
    candidate = Path(image_ref)
    if candidate.is_absolute():
        return image_ref
    resolved = dataset_dir / candidate
    if resolved.is_file():
        return str(resolved)
    return image_ref
