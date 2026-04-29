import pytest

from dataset.stimulus import Message, Stimulus
from generation.generate import Spec
from generation.validation import (
    ValidationError,
    _clear_registry,
    get_validator,
    registered_capabilities,
    run_validators,
    validates,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    _clear_registry()
    yield
    _clear_registry()


def _stimulus(
    capabilities: set[str],
    sample_id: str = "s_001",
) -> tuple[Stimulus, Spec]:
    spec = Spec(capabilities=capabilities, params={})
    stimulus = Stimulus(
        sample_id=sample_id,
        spec=spec,
        messages=[Message(role="user", content="hi")],
        target=None,
    )
    return stimulus, spec


def test_validates_decorator_registers_function():
    @validates(name="check_a", capability="cap_a")
    def check(stimulus, spec):
        return None

    entry = get_validator("check_a")
    assert entry is not None
    assert entry.fn is check
    assert entry.capability == "cap_a"


def test_get_validator_returns_none_for_unregistered():
    assert get_validator("never_registered") is None


def test_duplicate_validator_name_rejected():
    @validates(name="dupe", capability="cap_a")
    def first(stimulus, spec):
        pass

    with pytest.raises(ValueError, match="dupe"):

        @validates(name="dupe", capability="cap_b")
        def second(stimulus, spec):
            pass


def test_multiple_validators_per_capability_all_run():
    calls: list[str] = []

    @validates(name="check_one", capability="cap_a")
    def check_one(stimulus, spec):
        calls.append("one")

    @validates(name="check_two", capability="cap_a")
    def check_two(stimulus, spec):
        calls.append("two")

    stimulus, spec = _stimulus({"cap_a"})
    run_validators(stimulus, spec)
    assert calls == ["one", "two"]


def test_registered_capabilities_lists_all_caps_with_validators():
    @validates(name="a", capability="cap_a")
    def _a(s, sp):
        pass

    @validates(name="b", capability="cap_b")
    def _b(s, sp):
        pass

    assert registered_capabilities() == {"cap_a", "cap_b"}


def test_registered_capabilities_excludes_wildcard():
    @validates(name="all", capability="*")
    def _all(s, sp):
        pass

    @validates(name="a", capability="cap_a")
    def _a(s, sp):
        pass

    assert registered_capabilities() == {"cap_a"}


def test_run_validators_runs_for_all_spec_capabilities():
    calls: list[str] = []

    @validates(name="check_a", capability="cap_a")
    def check_a(stimulus, spec):
        calls.append("a")

    @validates(name="check_b", capability="cap_b")
    def check_b(stimulus, spec):
        calls.append("b")

    stimulus, spec = _stimulus({"cap_a", "cap_b"})
    run_validators(stimulus, spec)
    assert sorted(calls) == ["a", "b"]


def test_run_validators_skips_capabilities_without_validators():
    calls: list[str] = []

    @validates(name="check_a", capability="cap_a")
    def check_a(stimulus, spec):
        calls.append("a")

    stimulus, spec = _stimulus({"cap_a", "cap_unregistered"})
    run_validators(stimulus, spec)
    assert calls == ["a"]
    assert stimulus.validators_ran == ["check_a"]


def test_run_validators_populates_validators_ran():
    @validates(name="check_a", capability="cap_a")
    def _a(s, sp):
        pass

    @validates(name="check_b", capability="cap_b")
    def _b(s, sp):
        pass

    stimulus, spec = _stimulus({"cap_a", "cap_b"})
    run_validators(stimulus, spec)
    assert stimulus.validators_ran == ["check_a", "check_b"]


def test_validators_ran_is_sorted():
    @validates(name="zeta", capability="cap_a")
    def _z(s, sp):
        pass

    @validates(name="alpha", capability="cap_a")
    def _a(s, sp):
        pass

    @validates(name="mu", capability="cap_a")
    def _m(s, sp):
        pass

    stimulus, spec = _stimulus({"cap_a"})
    run_validators(stimulus, spec)
    assert stimulus.validators_ran == ["alpha", "mu", "zeta"]


def test_run_validators_empty_when_no_match():
    @validates(name="check_a", capability="cap_a")
    def _a(s, sp):
        pass

    stimulus, spec = _stimulus({"unrelated"})
    run_validators(stimulus, spec)
    assert stimulus.validators_ran == []


def test_wildcard_validator_runs_for_all_specs():
    calls: list[str] = []

    @validates(name="universal", capability="*")
    def universal(stimulus, spec):
        calls.append(stimulus.sample_id)

    s1, sp1 = _stimulus(set(), sample_id="empty_caps")
    s2, sp2 = _stimulus({"cap_a"}, sample_id="with_cap")
    run_validators(s1, sp1)
    run_validators(s2, sp2)

    assert calls == ["empty_caps", "with_cap"]
    assert s1.validators_ran == ["universal"]
    assert s2.validators_ran == ["universal"]


def test_wildcard_validator_runs_once_per_sample():
    calls: list[str] = []

    @validates(name="universal", capability="*")
    def universal(stimulus, spec):
        calls.append(stimulus.sample_id)

    stimulus, spec = _stimulus({"cap_a", "cap_b", "cap_c"})
    run_validators(stimulus, spec)

    assert calls == ["s_001"]
    assert stimulus.validators_ran == ["universal"]


def test_wildcard_and_capability_validators_both_run():
    @validates(name="universal", capability="*")
    def _u(s, sp):
        pass

    @validates(name="specific", capability="cap_a")
    def _s(s, sp):
        pass

    stimulus, spec = _stimulus({"cap_a"})
    run_validators(stimulus, spec)
    assert stimulus.validators_ran == ["specific", "universal"]


def test_run_validators_raises_validation_error_with_context():
    @validates(name="failing", capability="cap_a")
    def failing(stimulus, spec):
        raise AssertionError("expected 4 distractors, found 3")

    stimulus, spec = _stimulus({"cap_a"}, sample_id="s_042")
    with pytest.raises(ValidationError) as excinfo:
        run_validators(stimulus, spec)

    err = excinfo.value
    assert err.capability == "cap_a"
    assert err.validator_name == "failing"
    assert err.sample_id == "s_042"
    assert isinstance(err.__cause__, AssertionError)
    assert "expected 4 distractors" in str(err.__cause__)


def test_validation_error_message_identifies_sample_and_validator():
    @validates(name="failing", capability="cap_a")
    def failing(stimulus, spec):
        raise ValueError("bad")

    stimulus, spec = _stimulus({"cap_a"}, sample_id="s_xyz")
    with pytest.raises(ValidationError) as excinfo:
        run_validators(stimulus, spec)

    msg = str(excinfo.value)
    assert "s_xyz" in msg
    assert "failing" in msg
    assert "cap_a" in msg


def test_validators_ran_not_set_when_validator_fails():
    @validates(name="ok", capability="cap_a")
    def _ok(s, sp):
        pass

    @validates(name="bad", capability="cap_a")
    def _bad(s, sp):
        raise AssertionError("nope")

    stimulus, spec = _stimulus({"cap_a"})
    with pytest.raises(ValidationError):
        run_validators(stimulus, spec)
    assert stimulus.validators_ran == []
