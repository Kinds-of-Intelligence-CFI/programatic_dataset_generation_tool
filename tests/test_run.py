import json
import random
import time
import warnings
from pathlib import Path

import pytest

from dataset.stimulus import Message, Stimulus
from generation.generate import Spec
from generation.runner import run
from generation.validation import (
    ValidationError,
    _clear_registry,
    validates,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    _clear_registry()
    yield
    _clear_registry()


def _trivial_generator(spec: Spec, rng: random.Random) -> Stimulus:
    return Stimulus(
        sample_id=f"s_{spec.params['i']}_{rng.randint(0, 10**9)}",
        spec=spec,
        messages=[Message(role="user", content=f"hello {spec.params['i']}")],
        target=str(spec.params["i"]),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_produces_n_specs_times_n_reps_stimuli(tmp_path: Path):
    specs = [Spec(params={"i": i}) for i in range(4)]
    out = tmp_path / "ds"
    run(_trivial_generator, specs, n_reps=3, output_dir=out, seed=0)

    records = _read_jsonl(out / "stimuli.jsonl")
    assert len(records) == 4 * 3


def test_run_writes_jsonl_in_deterministic_submission_order(tmp_path: Path):
    specs = [Spec(params={"i": i}) for i in range(8)]

    def jittery_generator(spec: Spec, rng: random.Random) -> Stimulus:
        time.sleep(rng.uniform(0.0, 0.02))
        return _trivial_generator(spec, rng)

    out = tmp_path / "ds"
    run(jittery_generator, specs, n_reps=2, output_dir=out, seed=7, max_workers=4)

    records = _read_jsonl(out / "stimuli.jsonl")
    targets = [r["target"] for r in records]
    expected = [str(spec_index) for spec_index in range(8) for _ in range(2)]
    assert targets == expected


def test_run_is_byte_reproducible_across_runs(tmp_path: Path):
    specs = [Spec(params={"i": i}) for i in range(5)]

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run(_trivial_generator, specs, n_reps=4, output_dir=out_a, seed=123, max_workers=4)
    run(_trivial_generator, specs, n_reps=4, output_dir=out_b, seed=123, max_workers=4)

    assert (out_a / "stimuli.jsonl").read_bytes() == (out_b / "stimuli.jsonl").read_bytes()


def test_run_seeds_rng_per_sample(tmp_path: Path):
    specs = [Spec(params={"i": i}) for i in range(3)]
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    run(_trivial_generator, specs, n_reps=2, output_dir=out_a, seed=99)
    run(_trivial_generator, specs, n_reps=2, output_dir=out_b, seed=99)

    ids_a = [r["sample_id"] for r in _read_jsonl(out_a / "stimuli.jsonl")]
    ids_b = [r["sample_id"] for r in _read_jsonl(out_b / "stimuli.jsonl")]

    assert ids_a == ids_b
    assert len(set(ids_a)) == len(ids_a)


def test_run_calls_validators_per_sample(tmp_path: Path):
    calls: list[str] = []

    @validates(name="check", capability="cap_a")
    def check(stimulus, spec):
        calls.append(stimulus.sample_id)

    specs = [
        Spec(capabilities={"cap_a"}, params={"i": i}) for i in range(3)
    ]
    out = tmp_path / "ds"
    run(_trivial_generator, specs, n_reps=2, output_dir=out, seed=0)

    assert len(calls) == 3 * 2
    records = _read_jsonl(out / "stimuli.jsonl")
    assert sorted(calls) == sorted(r["sample_id"] for r in records)


def test_run_aborts_on_validator_failure(tmp_path: Path):
    @validates(name="picky", capability="cap_a")
    def picky(stimulus, spec):
        if spec.params["i"] == 2:
            raise AssertionError("nope on i=2")

    specs = [Spec(capabilities={"cap_a"}, params={"i": i}) for i in range(5)]
    out = tmp_path / "ds"

    with pytest.raises(ValidationError) as excinfo:
        run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0, max_workers=2)

    err = excinfo.value
    assert err.capability == "cap_a"
    assert err.validator_name == "picky"
    assert isinstance(err.__cause__, AssertionError)


def test_run_warns_about_capabilities_without_validators(tmp_path: Path):
    @validates(name="check", capability="cap_known")
    def check(stimulus, spec):
        pass

    specs = [
        Spec(capabilities={"cap_known", "cap_missing_a"}, params={"i": 0}),
        Spec(capabilities={"cap_missing_b"}, params={"i": 1}),
    ]
    out = tmp_path / "ds"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0)

    messages = [str(w.message) for w in caught]
    assert any("cap_missing_a" in m and "cap_missing_b" in m for m in messages)
    assert all("cap_known" not in m for m in messages)


def test_run_writes_manifest_with_correct_fields(tmp_path: Path):
    specs = [Spec(capabilities={"cap_a"}, params={"i": i}) for i in range(3)]

    @validates(name="check", capability="cap_a")
    def check(stimulus, spec):
        pass

    out = tmp_path / "demo"
    run(_trivial_generator, specs, n_reps=2, output_dir=out, seed=42, name="demo")

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["name"] == "demo"
    assert manifest["global_seed"] == 42
    assert manifest["n_reps"] == 2
    assert manifest["n_stimuli"] == 6
    assert len(manifest["specs"]) == 3
    assert "library_version" in manifest
    assert "timestamp" in manifest


def test_run_with_max_workers_1_matches_parallel(tmp_path: Path):
    specs = [Spec(params={"i": i}) for i in range(6)]

    out_seq = tmp_path / "seq"
    out_par = tmp_path / "par"
    run(_trivial_generator, specs, n_reps=2, output_dir=out_seq, seed=5, max_workers=1)
    run(_trivial_generator, specs, n_reps=2, output_dir=out_par, seed=5, max_workers=4)

    assert (out_seq / "stimuli.jsonl").read_bytes() == (out_par / "stimuli.jsonl").read_bytes()


def test_run_records_validators_ran_in_jsonl(tmp_path: Path):
    @validates(name="cap_a_check", capability="cap_a")
    def _a(s, sp):
        pass

    @validates(name="cap_b_check", capability="cap_b")
    def _b(s, sp):
        pass

    @validates(name="universal", capability="*")
    def _u(s, sp):
        pass

    specs = [
        Spec(capabilities={"cap_a"}, params={"i": 0}),
        Spec(capabilities={"cap_a", "cap_b"}, params={"i": 1}),
        Spec(capabilities=set(), params={"i": 2}),
    ]
    out = tmp_path / "ds"
    run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0)

    records = _read_jsonl(out / "stimuli.jsonl")
    by_target = {r["target"]: r["validators_ran"] for r in records}
    assert by_target["0"] == ["cap_a_check", "universal"]
    assert by_target["1"] == ["cap_a_check", "cap_b_check", "universal"]
    assert by_target["2"] == ["universal"]


def test_run_default_name_falls_back_to_output_dir(tmp_path: Path):
    specs = [Spec(params={"i": 0})]
    out = tmp_path / "my_dataset"
    run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["name"] == "my_dataset"
