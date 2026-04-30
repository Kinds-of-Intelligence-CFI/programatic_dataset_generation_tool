import json

import pytest

from dataset.serialise import (
    stimulus_from_dict,
    stimulus_to_dict,
    to_jsonable,
)
from dataset.stimulus import Content, Message, Stimulus
from generation.generate import Spec


def _example_stimulus() -> Stimulus:
    return Stimulus(
        sample_id="s_001",
        spec=Spec(
            capabilities={"spatial_perspective", "first_order_belief"},
            params={"grid_size": 4, "distractors": 2},
        ),
        messages=[
            Message(role="system", content="You are a careful reasoner."),
            Message(
                role="user",
                content=[
                    Content(type="text", data="What does the cat see?"),
                    Content(type="image", data="assets/sample_0001_grid.png"),
                ],
            ),
        ],
        target="left",
        validators_ran=["spatial_perspective"],
        metadata={"simulated_user_prompt": "Answer with one word."},
    )


def test_to_jsonable_sorts_sets_deterministically():
    out = to_jsonable({"caps", "another", "third"})
    assert out == ["another", "caps", "third"]


def test_to_jsonable_recurses_through_dataclass_and_set():
    spec = Spec(capabilities={"b_cap", "a_cap"}, params={"n": 3})
    out = to_jsonable(spec)
    assert out == {"capabilities": ["a_cap", "b_cap"], "params": {"n": 3}}


def test_to_jsonable_rejects_unsupported_type():
    with pytest.raises(TypeError):
        to_jsonable(complex(1, 2))


def test_stimulus_round_trip_via_json():
    original = _example_stimulus()
    payload = json.dumps(stimulus_to_dict(original))
    restored = stimulus_from_dict(json.loads(payload))

    assert restored.sample_id == original.sample_id
    assert restored.spec.capabilities == original.spec.capabilities
    assert restored.spec.params == original.spec.params
    assert restored.target == original.target
    assert restored.validators_ran == original.validators_ran
    assert restored.metadata == original.metadata
    assert len(restored.messages) == len(original.messages)
    assert restored.messages[0].role == "system"
    assert restored.messages[0].content == "You are a careful reasoner."
    assert isinstance(restored.messages[1].content, list)
    assert restored.messages[1].content[1].type == "image"
    assert restored.messages[1].content[1].data == "assets/sample_0001_grid.png"


def test_stimulus_capabilities_serialise_as_sorted_list():
    s = _example_stimulus()
    d = stimulus_to_dict(s)
    assert d["spec"]["capabilities"] == ["first_order_belief", "spatial_perspective"]


def test_validators_ran_defaults_to_empty():
    s = Stimulus(
        sample_id="x",
        spec=Spec(),
        messages=[],
        target="",
    )
    assert s.validators_ran == []


def test_empty_validators_ran_round_trips():
    original = Stimulus(
        sample_id="x",
        spec=Spec(),
        messages=[],
        target="",
    )
    restored = stimulus_from_dict(json.loads(json.dumps(stimulus_to_dict(original))))
    assert restored.validators_ran == []
