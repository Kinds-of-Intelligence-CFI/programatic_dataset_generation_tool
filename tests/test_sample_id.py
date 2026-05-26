import json
import random
from pathlib import Path

import pytest

from dataset.stimulus import Message, Stimulus
from generation.generate import SampleSpec
from generation.runner import run


def _read_ids(out: Path) -> list[str]:
    return [
        json.loads(line)["sample_id"]
        for line in (out / "stimuli.jsonl").read_text().splitlines()
    ]


def _gen_no_id(spec: SampleSpec, rng: random.Random) -> Stimulus:
    return Stimulus(
        spec=spec,
        messages=[Message(role="user", content=f"hello {spec.params['i']}")],
        target=str(spec.params["i"]),
    )


def test_auto_ids_when_all_none(tmp_path: Path):
    specs = [SampleSpec(params={"i": i}) for i in range(4)]
    out = tmp_path / "ds"
    run(_gen_no_id, specs, n_reps=2, output_dir=out, seed=0)
    assert _read_ids(out) == [str(i) for i in range(8)]


def test_user_set_ids_preserved(tmp_path: Path):
    def gen(spec: SampleSpec, rng: random.Random) -> Stimulus:
        return Stimulus(
            sample_id=f"named_{spec.params['i']}",
            spec=spec,
            messages=[Message(role="user", content="hi")],
            target="t",
        )

    specs = [SampleSpec(params={"i": i}) for i in range(3)]
    out = tmp_path / "ds"
    run(gen, specs, n_reps=1, output_dir=out, seed=0)
    assert _read_ids(out) == ["named_0", "named_1", "named_2"]


def test_mixed_user_and_auto_ids(tmp_path: Path):
    def gen(spec: SampleSpec, rng: random.Random) -> Stimulus:
        sid = f"named_{spec.params['i']}" if spec.params["user_set"] else None
        return Stimulus(
            sample_id=sid,
            spec=spec,
            messages=[Message(role="user", content="hi")],
            target="t",
        )

    specs = [
        SampleSpec(params={"i": 0, "user_set": True}),
        SampleSpec(params={"i": 1, "user_set": False}),
        SampleSpec(params={"i": 2, "user_set": True}),
        SampleSpec(params={"i": 3, "user_set": False}),
    ]
    out = tmp_path / "ds"
    run(gen, specs, n_reps=1, output_dir=out, seed=0)
    assert _read_ids(out) == ["named_0", "1", "named_2", "3"]


def test_duplicate_user_ids_raise(tmp_path: Path):
    def gen(spec: SampleSpec, rng: random.Random) -> Stimulus:
        return Stimulus(
            sample_id="same",
            spec=spec,
            messages=[Message(role="user", content="hi")],
            target="t",
        )

    specs = [SampleSpec(params={"i": i}) for i in range(2)]
    with pytest.raises(ValueError, match="same"):
        run(gen, specs, n_reps=1, output_dir=tmp_path / "ds", seed=0)


def test_auto_collides_with_user_id_raises(tmp_path: Path):
    def gen(spec: SampleSpec, rng: random.Random) -> Stimulus:
        sid = "3" if spec.params["i"] == 0 else None
        return Stimulus(
            sample_id=sid,
            spec=spec,
            messages=[Message(role="user", content="hi")],
            target="t",
        )

    specs = [SampleSpec(params={"i": i}) for i in range(4)]
    with pytest.raises(ValueError, match="'3'"):
        run(gen, specs, n_reps=1, output_dir=tmp_path / "ds", seed=0)


def test_auto_ids_deterministic_across_runs(tmp_path: Path):
    specs = [SampleSpec(params={"i": i}) for i in range(5)]
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run(_gen_no_id, specs, n_reps=2, output_dir=out_a, seed=42, max_workers=4)
    run(_gen_no_id, specs, n_reps=2, output_dir=out_b, seed=42, max_workers=4)
    ids_a = _read_ids(out_a)
    ids_b = _read_ids(out_b)
    assert ids_a == ids_b == [str(i) for i in range(10)]
