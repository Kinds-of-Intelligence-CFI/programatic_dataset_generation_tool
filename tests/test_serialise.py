import json

import pytest

from dataset.serialise import (
    stimulus_from_dict,
    stimulus_to_dict,
    to_jsonable,
)
from dataset.stimulus import (
    ContentAudio,
    ContentDocument,
    ContentImage,
    ContentText,
    ContentVideo,
    Message,
    Stimulus,
)
from generation.generate import SampleSpec


def _example_stimulus() -> Stimulus:
    return Stimulus(
        sample_id="s_001",
        spec=SampleSpec(
            demands={"spatial_perspective", "first_order_belief"},
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
    spec = SampleSpec(demands={"b_cap", "a_cap"}, params={"n": 3})
    out = to_jsonable(spec)
    assert out == {"demands": ["a_cap", "b_cap"], "params": {"n": 3}}


def test_to_jsonable_rejects_unsupported_type():
    with pytest.raises(TypeError):
        to_jsonable(complex(1, 2))


def test_stimulus_round_trip_via_json():
    original = _example_stimulus()
    payload = json.dumps(stimulus_to_dict(original))
    restored = stimulus_from_dict(json.loads(payload))

    assert restored.sample_id == original.sample_id
    assert restored.spec.demands == original.spec.demands
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
        spec=SampleSpec(),
        messages=[Message(role="user", content=[ContentText(text="hi")])],
        target="",
    )
    d = stimulus_to_dict(s)
    assert d["messages"][0]["content"][0] == {"type": "text", "text": "hi"}


def test_content_image_serialises_with_image_key_and_detail():
    s = Stimulus(
        spec=SampleSpec(),
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
        spec=SampleSpec(),
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


# ---- ContentAudio -----------------------------------------------------------


def test_content_audio_serialises_with_audio_key_and_format():
    s = Stimulus(
        spec=SampleSpec(),
        messages=[
            Message(
                role="user",
                content=[ContentAudio(audio="assets/files/clip.wav", format="wav")],
            )
        ],
        target="",
    )
    d = stimulus_to_dict(s)
    assert d["messages"][0]["content"][0] == {
        "type": "audio",
        "audio": "assets/files/clip.wav",
        "format": "wav",
    }


def test_content_audio_round_trip_preserves_format():
    original = Stimulus(
        spec=SampleSpec(),
        messages=[
            Message(
                role="user",
                content=[ContentAudio(audio="assets/files/x.mp3", format="mp3")],
            )
        ],
        target="",
    )
    restored = stimulus_from_dict(json.loads(json.dumps(stimulus_to_dict(original))))
    part = restored.messages[0].content[0]
    assert isinstance(part, ContentAudio)
    assert part.audio == "assets/files/x.mp3"
    assert part.format == "mp3"


def test_content_audio_from_bytes_produces_data_uri():
    c = ContentAudio.from_bytes(b"RIFF....fakewav", format="wav")
    assert c.audio.startswith("data:audio/wav;base64,")
    assert c.format == "wav"


def test_content_audio_from_bytes_mp3_uses_audio_mpeg_mime():
    c = ContentAudio.from_bytes(b"\xff\xfb\x00\x00fake", format="mp3")
    assert c.audio.startswith("data:audio/mpeg;base64,")
    assert c.format == "mp3"


def test_stimulus_from_dict_rejects_audio_without_format():
    payload = {
        "sample_id": "s_0",
        "spec": {"demands": [], "params": {}},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "audio", "audio": "x.wav"}],
            }
        ],
        "target": "",
        "validators_ran": [],
        "metadata": {},
    }
    with pytest.raises(KeyError, match="format"):
        stimulus_from_dict(payload)


# ---- ContentVideo -----------------------------------------------------------


def test_content_video_serialises_with_video_key_and_format():
    s = Stimulus(
        spec=SampleSpec(),
        messages=[
            Message(
                role="user",
                content=[ContentVideo(video="assets/files/clip.mp4", format="mp4")],
            )
        ],
        target="",
    )
    d = stimulus_to_dict(s)
    assert d["messages"][0]["content"][0] == {
        "type": "video",
        "video": "assets/files/clip.mp4",
        "format": "mp4",
    }


def test_content_video_round_trip_preserves_format():
    original = Stimulus(
        spec=SampleSpec(),
        messages=[
            Message(
                role="user",
                content=[ContentVideo(video="assets/files/x.mov", format="mov")],
            )
        ],
        target="",
    )
    restored = stimulus_from_dict(json.loads(json.dumps(stimulus_to_dict(original))))
    part = restored.messages[0].content[0]
    assert isinstance(part, ContentVideo)
    assert part.video == "assets/files/x.mov"
    assert part.format == "mov"


def test_content_video_from_bytes_produces_data_uri():
    c = ContentVideo.from_bytes(b"\x00\x00\x00\x18ftypmp42", format="mp4")
    assert c.video.startswith("data:video/mp4;base64,")
    assert c.format == "mp4"


def test_content_video_from_bytes_mov_uses_quicktime_mime():
    c = ContentVideo.from_bytes(b"fakequicktimebytes", format="mov")
    assert c.video.startswith("data:video/quicktime;base64,")
    assert c.format == "mov"


def test_stimulus_from_dict_rejects_video_without_format():
    payload = {
        "sample_id": "s_0",
        "spec": {"demands": [], "params": {}},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "video", "video": "x.mp4"}],
            }
        ],
        "target": "",
        "validators_ran": [],
        "metadata": {},
    }
    with pytest.raises(KeyError, match="format"):
        stimulus_from_dict(payload)


# ---- ContentDocument --------------------------------------------------------


def test_content_document_serialises_with_all_fields():
    s = Stimulus(
        spec=SampleSpec(),
        messages=[
            Message(
                role="user",
                content=[
                    ContentDocument(
                        document="assets/files/report.pdf",
                        filename="report.pdf",
                        mime_type="application/pdf",
                    )
                ],
            )
        ],
        target="",
    )
    d = stimulus_to_dict(s)
    assert d["messages"][0]["content"][0] == {
        "type": "document",
        "document": "assets/files/report.pdf",
        "filename": "report.pdf",
        "mime_type": "application/pdf",
    }


def test_content_document_round_trip_preserves_optional_fields():
    original = Stimulus(
        spec=SampleSpec(),
        messages=[
            Message(
                role="user",
                content=[
                    ContentDocument(
                        document="assets/files/spec.pdf",
                        filename="spec.pdf",
                        mime_type="application/pdf",
                    )
                ],
            )
        ],
        target="",
    )
    restored = stimulus_from_dict(json.loads(json.dumps(stimulus_to_dict(original))))
    part = restored.messages[0].content[0]
    assert isinstance(part, ContentDocument)
    assert part.document == "assets/files/spec.pdf"
    assert part.filename == "spec.pdf"
    assert part.mime_type == "application/pdf"


def test_content_document_optional_fields_default_to_none():
    c = ContentDocument(document="assets/files/x.pdf")
    assert c.filename is None
    assert c.mime_type is None


def test_content_document_round_trip_with_only_document_field():
    original = Stimulus(
        spec=SampleSpec(),
        messages=[
            Message(
                role="user",
                content=[ContentDocument(document="assets/files/raw.pdf")],
            )
        ],
        target="",
    )
    restored = stimulus_from_dict(json.loads(json.dumps(stimulus_to_dict(original))))
    part = restored.messages[0].content[0]
    assert isinstance(part, ContentDocument)
    assert part.filename is None
    assert part.mime_type is None


def test_content_document_from_bytes_populates_mime_type():
    c = ContentDocument.from_bytes(b"%PDF-1.4 fake", suffix="pdf")
    assert c.document.startswith("data:application/pdf;base64,")
    assert c.mime_type == "application/pdf"
    assert c.filename is None


def test_content_document_from_bytes_keeps_filename_when_provided():
    c = ContentDocument.from_bytes(b"%PDF-1.4", suffix="pdf", filename="report.pdf")
    assert c.filename == "report.pdf"


def test_stimulus_from_dict_rejects_unknown_content_type():
    payload = {
        "sample_id": "s_0",
        "spec": {"demands": [], "params": {}},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "reasoning", "reasoning": "..."}],
            }
        ],
        "target": "",
        "validators_ran": [],
        "metadata": {},
    }
    with pytest.raises(ValueError, match="reasoning"):
        stimulus_from_dict(payload)


def test_stimulus_demands_serialise_as_sorted_list():
    s = _example_stimulus()
    d = stimulus_to_dict(s)
    assert d["spec"]["demands"] == ["first_order_belief", "spatial_perspective"]


def test_validators_ran_defaults_to_empty():
    s = Stimulus(
        sample_id="x",
        spec=SampleSpec(),
        messages=[],
        target="",
    )
    assert s.validators_ran == []


def test_empty_validators_ran_round_trips():
    original = Stimulus(
        sample_id="x",
        spec=SampleSpec(),
        messages=[],
        target="",
    )
    restored = stimulus_from_dict(json.loads(json.dumps(stimulus_to_dict(original))))
    assert restored.validators_ran == []


# ---- functional spec --------------------------------------------------------


def test_stimulus_round_trip_preserves_functional():
    original = Stimulus(
        sample_id="s_0",
        spec=SampleSpec(demands={"cap_exp"}, params={"i": 0}),
        functional=SampleSpec(demands={"cap_func"}, params={"shared": 1}),
        messages=[Message(role="user", content="hi")],
        target="x",
    )
    restored = stimulus_from_dict(json.loads(json.dumps(stimulus_to_dict(original))))
    assert restored.functional is not None
    assert restored.functional.demands == {"cap_func"}
    assert restored.functional.params == {"shared": 1}


def test_stimulus_round_trip_with_no_functional_defaults_to_none():
    original = Stimulus(
        sample_id="s_0",
        spec=SampleSpec(),
        messages=[],
        target="",
    )
    restored = stimulus_from_dict(json.loads(json.dumps(stimulus_to_dict(original))))
    assert restored.functional is None


def test_stimulus_serialises_functional_as_top_level_field():
    s = Stimulus(
        sample_id="s_0",
        spec=SampleSpec(demands={"cap_exp"}),
        functional=SampleSpec(demands={"cap_func"}),
        messages=[],
        target="",
    )
    d = stimulus_to_dict(s)
    assert d["spec"] == {"demands": ["cap_exp"], "params": {}}
    assert d["functional"] == {"demands": ["cap_func"], "params": {}}


def test_stimulus_serialises_functional_as_null_by_default():
    s = Stimulus(spec=SampleSpec(), messages=[], target="")
    d = stimulus_to_dict(s)
    assert d["functional"] is None
