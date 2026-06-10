import json
import random
import time
import warnings
from pathlib import Path

import pytest

from dataset.stimulus import Message, Stimulus
from generation.generate import SampleSpec
from generation.glossary import DemandDefinition, Glossary
from generation.runner import current_output_dir
from generation.runner import run as _run
from generation.validation import (
    ValidationError,
    _clear_registry,
    validates,
)


def run(*args, **kwargs):
    # These tests use synthetic demand names and are orthogonal to glossary
    # enforcement; default the check off. Glossary behaviour has dedicated tests.
    kwargs.setdefault("glossary", None)
    return _run(*args, **kwargs)


@pytest.fixture(autouse=True)
def _isolated_registry():
    _clear_registry()
    yield
    _clear_registry()


def _trivial_generator(spec: SampleSpec, rng: random.Random) -> Stimulus:
    return Stimulus(
        sample_id=f"s_{spec.params['i']}_{rng.randint(0, 10**9)}",
        spec=spec,
        messages=[Message(role="user", content=f"hello {spec.params['i']}")],
        target=str(spec.params["i"]),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_produces_n_specs_times_n_reps_stimuli(tmp_path: Path):
    specs = [SampleSpec(params={"i": i}) for i in range(4)]
    out = tmp_path / "ds"
    run(_trivial_generator, specs, n_reps=3, output_dir=out, seed=0)

    records = _read_jsonl(out / "stimuli.jsonl")
    assert len(records) == 4 * 3


def test_run_writes_jsonl_in_deterministic_submission_order(tmp_path: Path):
    specs = [SampleSpec(params={"i": i}) for i in range(8)]

    def jittery_generator(spec: SampleSpec, rng: random.Random) -> Stimulus:
        time.sleep(rng.uniform(0.0, 0.02))
        return _trivial_generator(spec, rng)

    out = tmp_path / "ds"
    run(jittery_generator, specs, n_reps=2, output_dir=out, seed=7, max_workers=4)

    records = _read_jsonl(out / "stimuli.jsonl")
    targets = [r["target"] for r in records]
    expected = [str(spec_index) for spec_index in range(8) for _ in range(2)]
    assert targets == expected


def test_run_is_byte_reproducible_across_runs(tmp_path: Path):
    specs = [SampleSpec(params={"i": i}) for i in range(5)]

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run(_trivial_generator, specs, n_reps=4, output_dir=out_a, seed=123, max_workers=4)
    run(_trivial_generator, specs, n_reps=4, output_dir=out_b, seed=123, max_workers=4)

    assert (out_a / "stimuli.jsonl").read_bytes() == (out_b / "stimuli.jsonl").read_bytes()


def test_run_seeds_rng_per_sample(tmp_path: Path):
    specs = [SampleSpec(params={"i": i}) for i in range(3)]
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

    @validates(name="check", demand="cap_a")
    def check(stimulus, spec):
        calls.append(stimulus.sample_id)

    specs = [
        SampleSpec(demands={"cap_a": 1}, params={"i": i}) for i in range(3)
    ]
    out = tmp_path / "ds"
    run(_trivial_generator, specs, n_reps=2, output_dir=out, seed=0)

    assert len(calls) == 3 * 2
    records = _read_jsonl(out / "stimuli.jsonl")
    assert sorted(calls) == sorted(r["sample_id"] for r in records)


def test_run_aborts_on_validator_failure(tmp_path: Path):
    @validates(name="picky", demand="cap_a")
    def picky(stimulus, spec):
        if spec.params["i"] == 2:
            raise AssertionError("nope on i=2")

    specs = [SampleSpec(demands={"cap_a": 1}, params={"i": i}) for i in range(5)]
    out = tmp_path / "ds"

    with pytest.raises(ValidationError) as excinfo:
        run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0, max_workers=2)

    err = excinfo.value
    assert err.demand == "cap_a"
    assert err.validator_name == "picky"
    assert isinstance(err.__cause__, AssertionError)


def test_run_warns_about_demands_without_validators(tmp_path: Path):
    @validates(name="check", demand="cap_known")
    def check(stimulus, spec):
        pass

    specs = [
        SampleSpec(demands={"cap_known": 1, "cap_missing_a": 1}, params={"i": 0}),
        SampleSpec(demands={"cap_missing_b": 1}, params={"i": 1}),
    ]
    out = tmp_path / "ds"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0)

    messages = [str(w.message) for w in caught]
    assert any("cap_missing_a" in m and "cap_missing_b" in m for m in messages)
    assert all("cap_known" not in m for m in messages)


def test_run_writes_manifest_with_correct_fields(tmp_path: Path):
    specs = [SampleSpec(demands={"cap_a": 1}, params={"i": i}) for i in range(3)]

    @validates(name="check", demand="cap_a")
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
    specs = [SampleSpec(params={"i": i}) for i in range(6)]

    out_seq = tmp_path / "seq"
    out_par = tmp_path / "par"
    run(_trivial_generator, specs, n_reps=2, output_dir=out_seq, seed=5, max_workers=1)
    run(_trivial_generator, specs, n_reps=2, output_dir=out_par, seed=5, max_workers=4)

    assert (out_seq / "stimuli.jsonl").read_bytes() == (out_par / "stimuli.jsonl").read_bytes()


def test_run_records_validators_ran_in_jsonl(tmp_path: Path):
    @validates(name="cap_a_check", demand="cap_a")
    def _a(s, sp):
        pass

    @validates(name="cap_b_check", demand="cap_b")
    def _b(s, sp):
        pass

    @validates(name="universal", demand="*")
    def _u(s, sp):
        pass

    specs = [
        SampleSpec(demands={"cap_a": 1}, params={"i": 0}),
        SampleSpec(demands={"cap_a": 1, "cap_b": 1}, params={"i": 1}),
        SampleSpec(demands={}, params={"i": 2}),
    ]
    out = tmp_path / "ds"
    run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0)

    records = _read_jsonl(out / "stimuli.jsonl")
    by_target = {r["target"]: r["validators_ran"] for r in records}
    assert by_target["0"] == ["cap_a_check", "universal"]
    assert by_target["1"] == ["cap_a_check", "cap_b_check", "universal"]
    assert by_target["2"] == ["universal"]


def test_run_default_name_falls_back_to_output_dir(tmp_path: Path):
    specs = [SampleSpec(params={"i": 0})]
    out = tmp_path / "my_dataset"
    run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["name"] == "my_dataset"


def test_generate_fn_can_read_current_output_dir(tmp_path: Path):
    def generator(spec: SampleSpec, rng: random.Random) -> Stimulus:
        seen = current_output_dir()
        stimulus = _trivial_generator(spec, rng)
        stimulus.metadata = {"seen_output_dir": str(seen)}
        return stimulus

    specs = [SampleSpec(params={"i": i}) for i in range(4)]
    out = tmp_path / "ds"
    run(generator, specs, n_reps=2, output_dir=out, seed=0, max_workers=4)

    records = _read_jsonl(out / "stimuli.jsonl")
    assert records
    assert all(r["metadata"]["seen_output_dir"] == str(out) for r in records)


def test_validator_can_read_current_output_dir(tmp_path: Path):
    seen: list[Path] = []

    @validates(name="see_output_dir", demand="*")
    def see_output_dir(stimulus, spec):
        seen.append(current_output_dir())

    specs = [SampleSpec(params={"i": i}) for i in range(3)]
    out = tmp_path / "ds"
    run(_trivial_generator, specs, n_reps=2, output_dir=out, seed=0, max_workers=2)

    assert len(seen) == 3 * 2
    assert all(p == out for p in seen)


def test_current_output_dir_outside_run_raises():
    with pytest.raises(RuntimeError, match="current_output_dir"):
        current_output_dir()


# ---- functional spec --------------------------------------------------------


def test_run_passes_merged_spec_to_generator(tmp_path: Path):
    received: list[SampleSpec] = []

    def recorder(spec: SampleSpec, rng: random.Random) -> Stimulus:
        received.append(spec)
        return _trivial_generator(spec, rng)

    specs = [
        SampleSpec(demands={"cap_exp": 2}, params={"i": 0}),
        SampleSpec(demands={},             params={"i": 1}),
    ]
    functional = SampleSpec(demands={"cap_func": 3}, params={"shared": 42})
    run(
        recorder, specs, n_reps=1,
        output_dir=tmp_path / "ds", seed=0, max_workers=1,
        functional=functional,
    )
    assert received[0].demands == {"cap_func": 3, "cap_exp": 2}
    assert received[0].params == {"shared": 42, "i": 0}
    assert received[1].demands == {"cap_func": 3}
    assert received[1].params == {"shared": 42, "i": 1}


def test_run_splits_spec_and_functional_on_stimulus(tmp_path: Path):
    specs = [SampleSpec(demands={"cap_exp": 2}, params={"i": 0})]
    functional = SampleSpec(demands={"cap_func": 3}, params={"shared": 42})
    out = tmp_path / "ds"
    run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0, functional=functional)

    record = _read_jsonl(out / "stimuli.jsonl")[0]
    assert record["spec"] == {"demands": {"cap_exp": 2}, "params": {"i": 0}}
    assert record["functional"] == {"demands": {"cap_func": 3}, "params": {"shared": 42}}


def test_run_without_functional_leaves_functional_null_in_jsonl(tmp_path: Path):
    specs = [SampleSpec(demands={"cap_exp": 1}, params={"i": 0})]
    out = tmp_path / "ds"
    run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0)

    record = _read_jsonl(out / "stimuli.jsonl")[0]
    assert record["spec"] == {"demands": {"cap_exp": 1}, "params": {"i": 0}}
    assert record["functional"] is None


def test_run_functional_demand_validators_run_on_every_sample(tmp_path: Path):
    calls: list[str] = []

    @validates(name="check_func", demand="cap_func")
    def check(stimulus, spec):
        calls.append(stimulus.sample_id)

    specs = [
        SampleSpec(demands={}, params={"i": 0}),
        SampleSpec(demands={}, params={"i": 1}),
    ]
    functional = SampleSpec(demands={"cap_func": 1})
    run(_trivial_generator, specs, n_reps=2,
        output_dir=tmp_path / "ds", seed=0, functional=functional)

    assert len(calls) == 2 * 2


def test_run_param_key_in_both_functional_and_spec_raises(tmp_path: Path):
    specs = [
        SampleSpec(params={"i": 0, "shared": 1}),
        SampleSpec(params={"i": 1, "shared": 2}),
    ]
    functional = SampleSpec(params={"shared": 99})
    with pytest.raises(ValueError, match="shared"):
        run(_trivial_generator, specs, n_reps=1,
            output_dir=tmp_path / "ds", seed=0, functional=functional)


def test_run_param_conflict_error_lists_all_offending_keys(tmp_path: Path):
    specs = [SampleSpec(params={"i": 0, "a": 1, "b": 2})]
    functional = SampleSpec(params={"a": 9, "b": 8})
    with pytest.raises(ValueError) as excinfo:
        run(_trivial_generator, specs, n_reps=1,
            output_dir=tmp_path / "ds", seed=0, functional=functional)
    msg = str(excinfo.value)
    assert "a" in msg and "b" in msg


def test_run_no_missing_validator_warning_when_demand_only_in_functional(tmp_path: Path):
    @validates(name="check_func", demand="cap_func")
    def check(stimulus, spec):
        pass

    specs = [SampleSpec(params={"i": 0})]
    functional = SampleSpec(demands={"cap_func": 1})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(_trivial_generator, specs, n_reps=1,
            output_dir=tmp_path / "ds", seed=0, functional=functional)
    assert not any("cap_func" in str(w.message) for w in caught)


def test_run_writes_functional_to_manifest(tmp_path: Path):
    specs = [SampleSpec(demands={"cap_exp": 2}, params={"i": 0})]
    functional = SampleSpec(demands={"cap_func": 3}, params={"shared": 1})
    out = tmp_path / "ds"
    run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0, functional=functional)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["functional"] == {"demands": {"cap_func": 3}, "params": {"shared": 1}}
    assert manifest["specs"] == [{"demands": {"cap_exp": 2}, "params": {"i": 0}}]


def test_run_manifest_functional_is_null_when_not_supplied(tmp_path: Path):
    specs = [SampleSpec(demands={"cap_a": 1}, params={"i": 0})]
    out = tmp_path / "ds"
    run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["functional"] is None


# ---- functional + experimental demand collisions ----------------------------


def test_run_demand_key_in_both_functional_and_spec_raises(tmp_path: Path):
    specs = [
        SampleSpec(demands={"cap_a": 1}, params={"i": 0}),
        SampleSpec(demands={"cap_a": 2}, params={"i": 1}),
    ]
    functional = SampleSpec(demands={"cap_a": 1})
    with pytest.raises(ValueError, match="cap_a"):
        run(_trivial_generator, specs, n_reps=1,
            output_dir=tmp_path / "ds", seed=0, functional=functional)


def test_run_demand_collision_raises_even_when_levels_match(tmp_path: Path):
    specs = [SampleSpec(demands={"cap_a": 3}, params={"i": 0})]
    functional = SampleSpec(demands={"cap_a": 3})
    with pytest.raises(ValueError, match="cap_a"):
        run(_trivial_generator, specs, n_reps=1,
            output_dir=tmp_path / "ds", seed=0, functional=functional)


def test_run_demand_conflict_error_lists_all_offending_keys(tmp_path: Path):
    specs = [SampleSpec(demands={"cap_a": 1, "cap_b": 1, "cap_c": 1}, params={"i": 0})]
    functional = SampleSpec(demands={"cap_a": 9, "cap_b": 8})
    with pytest.raises(ValueError) as excinfo:
        run(_trivial_generator, specs, n_reps=1,
            output_dir=tmp_path / "ds", seed=0, functional=functional)
    msg = str(excinfo.value)
    assert "cap_a" in msg and "cap_b" in msg
    assert "cap_c" not in msg


# ---- glossary enforcement ---------------------------------------------------


def _glossary() -> Glossary:
    return Glossary(
        name="test-gloss",
        tree={
            "Language": {
                "creation": DemandDefinition(
                    description="produce language", paper_link="http://x"
                )
            },
            "addition": DemandDefinition(
                description="add numbers", paper_link="http://x"
            ),
        },
    )


def test_run_raises_on_demand_outside_glossary(tmp_path: Path):
    specs = [SampleSpec(demands={"additon": 1}, params={"i": 0})]
    with pytest.raises(ValueError, match="additon") as excinfo:
        _run(_trivial_generator, specs, n_reps=1,
             output_dir=tmp_path / "ds", seed=0, glossary=_glossary())
    # suggests the correctly-spelled neighbour
    assert "addition" in str(excinfo.value)


def test_run_glossary_checks_functional_demands(tmp_path: Path):
    specs = [SampleSpec(demands={"addition": 1}, params={"i": 0})]
    functional = SampleSpec(demands={"unknown_cap": 1})
    with pytest.raises(ValueError, match="unknown_cap"):
        _run(_trivial_generator, specs, n_reps=1,
             output_dir=tmp_path / "ds", seed=0,
             functional=functional, glossary=_glossary())


def test_run_passes_when_all_demands_in_glossary(tmp_path: Path):
    specs = [
        SampleSpec(demands={"addition": 1}, params={"i": 0}),
        SampleSpec(demands={"creation": 1}, params={"i": 1}),
    ]
    out = tmp_path / "ds"
    _run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0,
         glossary=_glossary())
    assert len(_read_jsonl(out / "stimuli.jsonl")) == 2


def test_run_glossary_none_disables_check(tmp_path: Path):
    specs = [SampleSpec(demands={"anything_goes": 1}, params={"i": 0})]
    out = tmp_path / "ds"
    _run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0, glossary=None)
    assert len(_read_jsonl(out / "stimuli.jsonl")) == 1


def test_run_records_glossary_in_manifest(tmp_path: Path):
    specs = [SampleSpec(demands={"addition": 1}, params={"i": 0})]
    out = tmp_path / "ds"
    _run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0,
         glossary=_glossary())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["glossary"]["name"] == "test-gloss"
    assert "addition" in manifest["glossary"]["demands"]
    assert manifest["glossary"]["demands"]["addition"]["ancestry"] == []


def test_run_glossary_none_records_null_in_manifest(tmp_path: Path):
    specs = [SampleSpec(params={"i": 0})]
    out = tmp_path / "ds"
    _run(_trivial_generator, specs, n_reps=1, output_dir=out, seed=0, glossary=None)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["glossary"] is None


def test_run_disjoint_demands_merge_without_error(tmp_path: Path):
    received: list[SampleSpec] = []

    def recorder(spec, rng):
        received.append(spec)
        return _trivial_generator(spec, rng)

    specs = [SampleSpec(demands={"cap_exp": 5}, params={"i": 0})]
    functional = SampleSpec(demands={"cap_func": 2})
    run(recorder, specs, n_reps=1,
        output_dir=tmp_path / "ds", seed=0, functional=functional)

    assert received[0].demands == {"cap_func": 2, "cap_exp": 5}
