"""Read-only access to a generated dataset folder for the viewer.

Pure functions, no framework dependencies. Mirrors the manifest/JSONL reading
and asset-path handling already used by ``evaluation/inspect_utils.py`` but
without constructing Inspect types. Records are augmented with two derived,
read-only fields the viewer needs and the dataset does not store directly:

  - ``spec_index``: position of the stimulus's spec in ``manifest.specs``.
  - ``modality``: a single short tag summarising the message content types.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# Priority order when a stimulus mixes content types: the first present wins.
_MODALITY_PRIORITY: list[tuple[str, str]] = [
    ("image", "img"),
    ("video", "video"),
    ("audio", "audio"),
    ("document", "doc"),
    ("text", "txt"),
]


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    manifest: Path
    stimuli: Path
    assets: Path


def resolve_dataset(path: str | Path) -> DatasetPaths:
    """Validate that ``path`` is a dataset folder and return its key paths.

    Raises ``FileNotFoundError`` (with a message pointing at what is missing)
    if the directory, manifest, or stimuli file is absent.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    manifest = root / "manifest.json"
    stimuli = root / "stimuli.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest.json not found in {root}")
    if not stimuli.is_file():
        raise FileNotFoundError(f"stimuli.jsonl not found in {root}")
    return DatasetPaths(
        root=root, manifest=manifest, stimuli=stimuli, assets=root / "assets"
    )


def load_manifest(paths: DatasetPaths) -> dict[str, Any]:
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def iter_stimuli(paths: DatasetPaths) -> Iterator[dict[str, Any]]:
    """Yield each stimulus record, augmented with ``spec_index`` and ``modality``.

    Reads ``stimuli.jsonl`` one line at a time so the whole dataset is never
    held in memory here; callers that need a list materialise it themselves.
    """
    manifest_specs = load_manifest(paths).get("specs", [])
    canonical_specs = [canonical_spec(s) for s in manifest_specs]
    with paths.stimuli.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["spec_index"] = _spec_index_from_canonical(
                record.get("spec"), canonical_specs
            )
            record["modality"] = modality_of(record)
            yield record


def canonical_spec(spec: dict[str, Any] | None) -> str:
    """Stable string form of a spec for equality matching.

    Demands are emitted as ``{name: level}`` dicts and params as a plain dict;
    both round-trip through ``to_jsonable`` on the writer side. ``sort_keys``
    keeps the canonical form insensitive to dict ordering.
    """
    if spec is None:
        return "null"
    demands = spec.get("demands", {})
    params = spec.get("params", {})
    return json.dumps({"demands": demands, "params": params}, sort_keys=True)


def spec_index_of(
    spec: dict[str, Any] | None, manifest_specs: list[dict[str, Any]]
) -> int | None:
    return _spec_index_from_canonical(
        spec, [canonical_spec(s) for s in manifest_specs]
    )


def _spec_index_from_canonical(
    spec: dict[str, Any] | None, canonical_specs: list[str]
) -> int | None:
    key = canonical_spec(spec)
    try:
        return canonical_specs.index(key)
    except ValueError:
        return None


def modality_of(record: dict[str, Any]) -> str:
    """Summarise a stimulus's content types as one short tag (e.g. ``img``)."""
    present: set[str] = set()
    for message in record.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            present.add("text")
            continue
        for item in content or []:
            present.add(item.get("type", "text"))
    for content_type, tag in _MODALITY_PRIORITY:
        if content_type in present:
            return tag
    return "txt"


def safe_asset_path(paths: DatasetPaths, rel: str) -> Path | None:
    """Resolve ``rel`` against the assets dir, refusing escapes.

    Returns the resolved path only if it stays within ``assets/`` and points at
    an existing file; otherwise ``None`` (caller turns this into a 404).
    """
    assets_root = paths.assets.resolve()
    candidate = (assets_root / rel).resolve()
    if assets_root not in candidate.parents and candidate != assets_root:
        return None
    if not candidate.is_file():
        return None
    return candidate
