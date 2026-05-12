import base64
import hashlib
import json
import mimetypes
import re
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

from dataset.serialise import stimulus_to_dict, to_jsonable
from dataset.stimulus import Content, ContentImage, ContentText, Message, Stimulus
from generation.generate import Spec

_PACKAGE_NAME = "programatic-dataset-generation-tool"
_FALLBACK_VERSION = "0.1.0"

_SAFE_SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_DATA_URI_RE = re.compile(r"^data:([^;]+);base64,(.*)$", re.DOTALL)


class AssetCache:
    """In-memory dedup map (sha256 -> relative path) for files copied from source paths.

    Lives for the duration of one ``write_dataset`` call. When the same bytes
    are referenced by multiple stimuli (whether via the same path or via
    different paths that happen to hold identical content), only one file
    lands in ``assets/files/``.
    """

    def __init__(self) -> None:
        self._by_hash: dict[str, str] = {}

    def get_or_put(
        self,
        data: bytes,
        output_dir: Path,
        *,
        suggested_stem: str,
        suffix: str,
    ) -> str:
        h = hashlib.sha256(data).hexdigest()
        if h in self._by_hash:
            return self._by_hash[h]
        rel = f"assets/files/{suggested_stem}__{h[:12]}.{suffix}"
        path = output_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._by_hash[h] = rel
        return rel


def normalize_content(
    content: Content,
    *,
    sample_id: str,
    message_index: int,
    content_index: int,
    output_dir: Path,
    cache: AssetCache,
) -> Content:
    """Rewrite a Content's reference field so it points inside ``output_dir/assets/``.

    Dispatch on the reference string:
      - ``data:...;base64,...``  -> decode, write to ``assets/inline/`` named by
        sample/message/content position (no dedup; each call gets its own file).
      - ``http(s)://...``        -> unchanged (external by intent).
      - ``assets/...`` (relative) -> unchanged (idempotent).
      - any other path string    -> read bytes, hash, dedup via ``cache``,
        write to ``assets/files/{stem}__{hash[:12]}.{ext}`` if new.

    ``ContentText`` is returned unchanged.
    """
    if isinstance(content, ContentText):
        return content
    if isinstance(content, ContentImage):
        image_str = content.image
        if _is_data_uri(image_str):
            rel = _write_inline_from_data_uri(
                image_str,
                output_dir=output_dir,
                sample_id=sample_id,
                message_index=message_index,
                content_index=content_index,
            )
            return replace(content, image=rel)
        if _is_http_url(image_str):
            return content
        if _is_already_relative_asset(image_str):
            return content
        rel = _copy_path_to_files(image_str, output_dir=output_dir, cache=cache)
        return replace(content, image=rel)
    raise TypeError(f"Unknown Content variant: {type(content).__name__}")


def write_jsonl(
    path: Path,
    stimuli: Iterable[Stimulus],
    *,
    output_dir: Path,
    cache: AssetCache,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for s in stimuli:
            normalized = _normalize_stimulus(s, output_dir=output_dir, cache=cache)
            f.write(json.dumps(stimulus_to_dict(normalized)))
            f.write("\n")
            count += 1
    return count


def write_manifest(
    path: Path,
    *,
    name: str,
    specs: list[Spec],
    global_seed: int,
    n_reps: int,
    n_stimuli: int,
) -> None:
    manifest = {
        "name": name,
        "library_version": _library_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "global_seed": global_seed,
        "n_reps": n_reps,
        "n_stimuli": n_stimuli,
        "specs": [to_jsonable(s) for s in specs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_dataset(
    output_dir: Path,
    *,
    name: str,
    stimuli: Iterable[Stimulus],
    specs: list[Spec],
    global_seed: int,
    n_reps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = AssetCache()
    n_stimuli = write_jsonl(
        output_dir / "stimuli.jsonl",
        stimuli,
        output_dir=output_dir,
        cache=cache,
    )
    write_manifest(
        output_dir / "manifest.json",
        name=name,
        specs=specs,
        global_seed=global_seed,
        n_reps=n_reps,
        n_stimuli=n_stimuli,
    )


# ---- internals --------------------------------------------------------------


def _normalize_stimulus(
    stimulus: Stimulus, *, output_dir: Path, cache: AssetCache
) -> Stimulus:
    sample_id = stimulus.sample_id
    if sample_id is None:
        raise ValueError(
            "Stimulus.sample_id must be set before write; the runner assigns sequential ids."
        )
    new_messages: list[Message] = []
    for msg_idx, message in enumerate(stimulus.messages):
        if isinstance(message.content, str):
            new_messages.append(message)
            continue
        new_content: list[Content] = []
        needs_sample_id_check = any(
            isinstance(c, ContentImage) and _is_data_uri(c.image)
            for c in message.content
        )
        if needs_sample_id_check:
            _validate_sample_id(sample_id)
        for content_idx, item in enumerate(message.content):
            new_content.append(
                normalize_content(
                    item,
                    sample_id=sample_id,
                    message_index=msg_idx,
                    content_index=content_idx,
                    output_dir=output_dir,
                    cache=cache,
                )
            )
        new_messages.append(Message(role=message.role, content=new_content))
    return replace(stimulus, messages=new_messages)


def _validate_sample_id(sample_id: str) -> None:
    if not _SAFE_SAMPLE_ID_RE.match(sample_id):
        raise ValueError(
            f"sample_id contains characters unsafe for filenames: {sample_id!r}. "
            f"Allowed pattern: [A-Za-z0-9_.-]+"
        )


def _is_data_uri(s: str) -> bool:
    return _DATA_URI_RE.match(s) is not None


def _is_http_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _is_already_relative_asset(s: str) -> bool:
    return s.startswith("assets/") and not Path(s).is_absolute()


def _write_inline_from_data_uri(
    data_uri: str,
    *,
    output_dir: Path,
    sample_id: str,
    message_index: int,
    content_index: int,
) -> str:
    _validate_sample_id(sample_id)
    match = _DATA_URI_RE.match(data_uri)
    assert match is not None  # already passed _is_data_uri
    mime_type, b64_payload = match.group(1), match.group(2)
    suffix = _suffix_for_mime(mime_type)
    data = base64.b64decode(b64_payload)
    rel = f"assets/inline/{sample_id}_{message_index}_{content_index}.{suffix}"
    path = output_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return rel


def _copy_path_to_files(
    path_str: str, *, output_dir: Path, cache: AssetCache
) -> str:
    source = Path(path_str)
    data = source.read_bytes()  # raises FileNotFoundError with the path
    suffix = source.suffix.lstrip(".") or "bin"
    return cache.get_or_put(
        data, output_dir, suggested_stem=source.stem, suffix=suffix
    )


def _suffix_for_mime(mime_type: str) -> str:
    guessed = mimetypes.guess_extension(mime_type)
    if guessed:
        return guessed.lstrip(".")
    return "bin"


def _library_version() -> str:
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _FALLBACK_VERSION
