import json

import pytest

from dataset.serialise import (
    stimulus_from_dict,
    stimulus_to_dict,
    to_jsonable,
)
from dataset.stimulus import ContentImage, ContentText, Message, Stimulus
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
                    ContentText(text="What does the cat see?"),
                    ContentImage(
                        image="assets/files/grid__a1b2c3d4e5f6.png",
                        detail="high",
                    ),
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

    user_content = restored.messages[1].content
    assert isinstance(user_content, list)
    assert isinstance(user_content[0], ContentText)
    assert user_content[0].text == "What does the cat see?"
    assert isinstance(user_content[1], ContentImage)
    assert user_content[1].image == "assets/files/grid__a1b2c3d4e5f6.png"
    assert user_content[1].detail == "high"


def test_content_text_serialises_with_text_key():
    s = Stimulus(
        spec=Spec(),
        messages=[Message(role="user", content=[ContentText(text="hi")])],
        target="",
    )
    d = stimulus_to_dict(s)
    assert d["messages"][0]["content"][0] == {"type": "text", "text": "hi"}


def test_content_image_serialises_with_image_key_and_detail():
    s = Stimulus(
        spec=Spec(),
        messages=[
            Message(
                role="user",
                content=[ContentImage(image="assets/files/foo.png", detail="low")],
            )
        ],
        target="",
    )
    d = stimulus_to_dict(s)
    assert d["messages"][0]["content"][0] == {
        "type": "image",
        "image": "assets/files/foo.png",
        "detail": "low",
    }


def test_content_image_detail_defaults_to_auto():
    c = ContentImage(image="assets/files/foo.png")
    assert c.detail == "auto"


def test_content_image_from_bytes_produces_data_uri():
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"fake"
    c = ContentImage.from_bytes(png_bytes, suffix="png")
    assert c.image.startswith("data:image/png;base64,")
    assert c.detail == "auto"


def test_content_image_from_bytes_round_trip_preserves_data_uri():
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"abc"
    original = Stimulus(
        spec=Spec(),
        messages=[
            Message(
                role="user",
                content=[ContentImage.from_bytes(png_bytes, suffix="png")],
            )
        ],
        target="",
    )
    restored = stimulus_from_dict(json.loads(json.dumps(stimulus_to_dict(original))))
    img = restored.messages[0].content[0]
    assert isinstance(img, ContentImage)
    assert img.image == original.messages[0].content[0].image


def test_stimulus_from_dict_rejects_unknown_content_type():
    payload = {
        "sample_id": "s_0",
        "spec": {"capabilities": [], "params": {}},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "video", "video": "/tmp/x.mp4"}],
            }
        ],
        "target": "",
        "validators_ran": [],
        "metadata": {},
    }
    with pytest.raises(ValueError, match="video"):
        stimulus_from_dict(payload)


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
