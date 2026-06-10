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
from dataset.stimulus import (
    Content,
    ContentAudio,
    ContentDocument,
    ContentImage,
    ContentText,
    ContentVideo,
    Message,
    Stimulus,
)
from generation.generate import SampleSpec
from generation.glossary import Glossary

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

    ``ContentText`` is returned unchanged. All non-text variants share the
    same string-reference dispatch via ``_normalize_string_ref``.
    """
    if isinstance(content, ContentText):
        return content

    def _rewrite(ref: str) -> str:
        return _normalize_string_ref(
            ref,
            sample_id=sample_id,
            message_index=message_index,
            content_index=content_index,
            output_dir=output_dir,
            cache=cache,
        )

    if isinstance(content, ContentImage):
        return replace(content, image=_rewrite(content.image))
    if isinstance(content, ContentAudio):
        return replace(content, audio=_rewrite(content.audio))
    if isinstance(content, ContentVideo):
        return replace(content, video=_rewrite(content.video))
    if isinstance(content, ContentDocument):
        return replace(content, document=_rewrite(content.document))
    raise TypeError(f"Unknown Content variant: {type(content).__name__}")


def _normalize_string_ref(
    ref: str,
    *,
    sample_id: str,
    message_index: int,
    content_index: int,
    output_dir: Path,
    cache: AssetCache,
) -> str:
    if _is_data_uri(ref):
        return _write_inline_from_data_uri(
            ref,
            output_dir=output_dir,
            sample_id=sample_id,
            message_index=message_index,
            content_index=content_index,
        )
    if _is_http_url(ref):
        return ref
    if _is_already_relative_asset(ref):
        return ref
    return _copy_path_to_files(ref, output_dir=output_dir, cache=cache)


def _media_ref(content: Content) -> str | None:
    """The data-bearing string for a media variant, or None for text."""
    if isinstance(content, ContentImage):
        return content.image
    if isinstance(content, ContentAudio):
        return content.audio
    if isinstance(content, ContentVideo):
        return content.video
    if isinstance(content, ContentDocument):
        return content.document
    return None


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
    specs: list[SampleSpec],
    global_seed: int,
    n_reps: int,
    n_stimuli: int,
    functional: SampleSpec | None = None,
    glossary: Glossary | None = None,
) -> None:
    manifest = {
        "name": name,
        "library_version": _library_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "global_seed": global_seed,
        "n_reps": n_reps,
        "n_stimuli": n_stimuli,
        "functional": to_jsonable(functional) if functional is not None else None,
        "glossary": glossary.to_jsonable() if glossary is not None else None,
        "specs": [to_jsonable(s) for s in specs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_dataset(
    output_dir: Path,
    *,
    name: str,
    stimuli: Iterable[Stimulus],
    specs: list[SampleSpec],
    global_seed: int,
    n_reps: int,
    functional: SampleSpec | None = None,
    glossary: Glossary | None = None,
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
        functional=functional,
        glossary=glossary,
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
            (ref := _media_ref(c)) is not None and _is_data_uri(ref)
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


_MIME_SUFFIX_OVERRIDES = {
    # stdlib's mimetypes registers .wav as audio/x-wav (or audio/vnd.wave on
    # some Pythons) and so doesn't recognise the de-facto audio/wav. Inspect
    # uses audio/wav in its own examples; map it back to .wav so user-supplied
    # data URIs and our own ContentAudio.from_bytes both round-trip cleanly.
    "audio/wav": "wav",
}


def _suffix_for_mime(mime_type: str) -> str:
    if mime_type in _MIME_SUFFIX_OVERRIDES:
        return _MIME_SUFFIX_OVERRIDES[mime_type]
    guessed = mimetypes.guess_extension(mime_type)
    if guessed:
        return guessed.lstrip(".")
    return "bin"


def _library_version() -> str:
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _FALLBACK_VERSION
