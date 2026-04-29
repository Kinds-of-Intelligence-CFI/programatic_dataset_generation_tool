import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

from dataset.serialise import stimulus_to_dict, to_jsonable
from dataset.stimulus import Stimulus
from generation.generate import Spec

_PACKAGE_NAME = "programatic-dataset-generation-tool"
_FALLBACK_VERSION = "0.1.0"


def save_asset(
    output_dir: Path,
    data: bytes,
    suffix: str,
    name_hint: str | None = None,
) -> str:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    if name_hint is not None:
        path = assets_dir / f"{name_hint}.{suffix}"
        if path.exists():
            raise FileExistsError(f"Asset already exists: {path}")
    else:
        n = 0
        while True:
            path = assets_dir / f"{n:04d}.{suffix}"
            if not path.exists():
                break
            n += 1

    path.write_bytes(data)
    return f"assets/{path.name}"


def write_jsonl(path: Path, stimuli: Iterable[Stimulus]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for s in stimuli:
            f.write(json.dumps(stimulus_to_dict(s)))
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
    n_stimuli = write_jsonl(output_dir / "stimuli.jsonl", stimuli)
    write_manifest(
        output_dir / "manifest.json",
        name=name,
        specs=specs,
        global_seed=global_seed,
        n_reps=n_reps,
        n_stimuli=n_stimuli,
    )


def _library_version() -> str:
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _FALLBACK_VERSION
