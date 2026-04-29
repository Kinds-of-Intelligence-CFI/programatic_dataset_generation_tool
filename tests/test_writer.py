import json
import random
from pathlib import Path

from dataset.stimulus import Content, Message, Stimulus
from dataset.writer import (
    save_asset,
    write_dataset,
    write_jsonl,
    write_manifest,
)
from generation.generate import Spec

FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def test_save_asset_writes_bytes_and_returns_relative_path(tmp_path: Path):
    rel = save_asset(tmp_path, FAKE_PNG, suffix="png")
    assert rel.startswith("assets/")
    assert rel.endswith(".png")
    on_disk = tmp_path / rel
    assert on_disk.is_file()
    assert on_disk.read_bytes() == FAKE_PNG


def test_save_asset_produces_unique_names(tmp_path: Path):
    paths = {save_asset(tmp_path, FAKE_PNG, suffix="png") for _ in range(5)}
    assert len(paths) == 5


def test_save_asset_uses_name_hint_when_given(tmp_path: Path):
    rel = save_asset(tmp_path, FAKE_PNG, suffix="png", name_hint="sample_0001_grid")
    assert rel == "assets/sample_0001_grid.png"
    assert (tmp_path / rel).read_bytes() == FAKE_PNG


def _stimulus(sample_id: str, target: str = "x") -> Stimulus:
    return Stimulus(
        sample_id=sample_id,
        spec=Spec(capabilities={"cap_a"}, params={"n": 1}),
        messages=[Message(role="user", content="hello")],
        target=target,
    )


def test_write_jsonl_streams_one_line_per_stimulus(tmp_path: Path):
    path = tmp_path / "stimuli.jsonl"
    stimuli = [_stimulus(f"s_{i}") for i in range(4)]
    n = write_jsonl(path, iter(stimuli))
    assert n == 4
    lines = path.read_text().splitlines()
    assert len(lines) == 4
    for raw, expected in zip(lines, stimuli):
        d = json.loads(raw)
        assert d["sample_id"] == expected.sample_id


def test_write_jsonl_consumes_iterable_lazily(tmp_path: Path):
    consumed = []

    def gen():
        for i in range(3):
            consumed.append(i)
            yield _stimulus(f"s_{i}")

    write_jsonl(tmp_path / "out.jsonl", gen())
    assert consumed == [0, 1, 2]


def test_write_manifest_contains_all_required_fields(tmp_path: Path):
    path = tmp_path / "manifest.json"
    specs = [Spec(capabilities={"b", "a"}, params={"k": 1})]
    write_manifest(
        path,
        name="my_dataset",
        specs=specs,
        global_seed=42,
        n_reps=3,
        n_stimuli=3,
    )
    m = json.loads(path.read_text())
    assert m["name"] == "my_dataset"
    assert m["global_seed"] == 42
    assert m["n_reps"] == 3
    assert m["n_stimuli"] == 3
    assert "library_version" in m
    assert "timestamp" in m
    assert m["specs"] == [{"capabilities": ["a", "b"], "params": {"k": 1}}]


def test_write_dataset_end_to_end_with_random_content(tmp_path: Path):
    rng = random.Random(0xC0FFEE)
    output_dir = tmp_path / "demo_dataset"
    output_dir.mkdir()

    specs = [Spec(capabilities={"cap_a"}, params={"i": i}) for i in range(5)]

    def build_stimuli():
        for i, spec in enumerate(specs):
            messages: list[Message] = [
                Message(role="user", content=f"q-{rng.randint(0, 999)}"),
            ]
            if i == 0:
                rel = save_asset(output_dir, FAKE_PNG, suffix="png")
                messages.append(
                    Message(
                        role="assistant",
                        content=[Content(type="image", data=rel)],
                    )
                )
            yield Stimulus(
                sample_id=f"s_{i:04d}",
                spec=spec,
                messages=messages,
                target=rng.choice(["a", "b", "c"]),
                metadata={"rep_index": 0},
            )

    write_dataset(
        output_dir,
        name="demo_dataset",
        stimuli=build_stimuli(),
        specs=specs,
        global_seed=0xC0FFEE,
        n_reps=1,
    )

    jsonl = output_dir / "stimuli.jsonl"
    manifest = output_dir / "manifest.json"
    assets = output_dir / "assets"

    assert jsonl.is_file()
    assert manifest.is_file()
    assert assets.is_dir()

    lines = jsonl.read_text().splitlines()
    assert len(lines) == 5

    first = json.loads(lines[0])
    image_content = first["messages"][1]["content"][0]
    assert image_content["type"] == "image"
    assert (output_dir / image_content["data"]).is_file()

    m = json.loads(manifest.read_text())
    assert m["n_stimuli"] == 5
    assert m["n_reps"] == 1
    assert m["global_seed"] == 0xC0FFEE
    assert len(m["specs"]) == 5
