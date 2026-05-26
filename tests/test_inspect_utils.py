import json
from pathlib import Path

import pytest
from inspect_ai._util.content import ContentAudio as InspectContentAudio
from inspect_ai._util.content import ContentDocument as InspectContentDocument
from inspect_ai._util.content import ContentImage as InspectContentImage
from inspect_ai._util.content import ContentText as InspectContentText
from inspect_ai._util.content import ContentVideo as InspectContentVideo
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
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
from dataset.writer import write_dataset
from evaluation.inspect_utils import load_dataset
from generation.generate import SampleSpec


def _write(
    tmp_path: Path,
    stimuli: list[Stimulus],
    name: str = "ds",
) -> Path:
    out = tmp_path / name
    specs = [s.spec for s in stimuli] or [SampleSpec()]
    write_dataset(
        out,
        name=name,
        stimuli=stimuli,
        specs=specs,
        global_seed=0,
        n_reps=1,
    )
    return out


def _stimulus(
    sample_id: str = "s_0",
    *,
    messages: list[Message] | None = None,
    target: str = "answer",
    metadata: dict | None = None,
    demands: set[str] | None = None,
) -> Stimulus:
    return Stimulus(
        sample_id=sample_id,
        spec=SampleSpec(
            demands=demands if demands is not None else {"cap_a"},
            params={"n": 1},
        ),
        messages=messages
        if messages is not None
        else [Message(role="user", content="hello")],
        target=target,
        metadata=metadata if metadata is not None else {},
    )


# ---- basic loading ----------------------------------------------------------


def test_load_dataset_returns_memory_dataset_with_manifest_name(tmp_path: Path):
    out = _write(tmp_path, [_stimulus("s_0"), _stimulus("s_1")], name="demo")

    ds = load_dataset(out)

    assert isinstance(ds, MemoryDataset)
    assert ds.name == "demo"
    assert len(ds) == 2


def test_load_dataset_preserves_jsonl_order_and_ids(tmp_path: Path):
    stimuli = [_stimulus(f"s_{i}", target=f"t_{i}") for i in range(3)]
    out = _write(tmp_path, stimuli)

    ds = load_dataset(out)

    assert [s.id for s in ds] == ["s_0", "s_1", "s_2"]
    assert [s.target for s in ds] == ["t_0", "t_1", "t_2"]


def test_stimulus_metadata_block_contains_spec_and_dataset_dir(tmp_path: Path):
    stim = _stimulus("s_0", demands={"cap_a", "cap_b"})
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    block = sample.metadata["_stimulus"]
    assert block["spec"]["demands"] == ["cap_a", "cap_b"]
    assert block["dataset_dir"] == str(out)
    assert block["validators_ran"] == []
    # messages no longer live in metadata - they're on Sample.input
    assert "messages" not in block


def test_user_metadata_is_flat_alongside_stimulus_block(tmp_path: Path):
    stim = _stimulus("s_0", metadata={"foo": 1, "bar": "x"})
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    assert sample.metadata["foo"] == 1
    assert sample.metadata["bar"] == "x"
    assert "_stimulus" in sample.metadata


# ---- string-input shortcut --------------------------------------------------


def test_single_user_text_message_populates_input_as_string(tmp_path: Path):
    stim = _stimulus(
        "s_0",
        messages=[Message(role="user", content="what is the capital of France?")],
    )
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    assert sample.input == "what is the capital of France?"


# ---- typed list[ChatMessage] input ------------------------------------------


def test_multiple_messages_become_list_of_chat_messages(tmp_path: Path):
    stim = _stimulus(
        "s_0",
        messages=[
            Message(role="system", content="you are helpful"),
            Message(role="user", content="hi"),
        ],
    )
    out = _write(tmp_path, [stim])
    sample = load_dataset(out)[0]

    assert isinstance(sample.input, list)
    assert len(sample.input) == 2
    assert isinstance(sample.input[0], ChatMessageSystem)
    assert sample.input[0].content == "you are helpful"
    assert isinstance(sample.input[1], ChatMessageUser)
    assert sample.input[1].content == "hi"


@pytest.mark.parametrize(
    "role,expected_cls",
    [
        ("system", ChatMessageSystem),
        ("user", ChatMessageUser),
        ("assistant", ChatMessageAssistant),
        ("tool", ChatMessageTool),
    ],
)
def test_all_four_roles_map_correctly(role, expected_cls, tmp_path: Path):
    # Pair with an extra message so the shortcut doesn't kick in.
    stim = _stimulus(
        "s_0",
        messages=[
            Message(role=role, content="x"),
            Message(role="user", content="continue"),
        ],
    )
    out = _write(tmp_path, [stim])
    sample = load_dataset(out)[0]

    assert isinstance(sample.input, list)
    assert isinstance(sample.input[0], expected_cls)


def test_multimodal_user_message_uses_typed_content_parts(tmp_path: Path):
    stim = _stimulus(
        "s_0",
        messages=[
            Message(
                role="user",
                content=[
                    ContentText(text="describe this"),
                    ContentImage(image="assets/files/foo__abc123def456.png"),
                ],
            )
        ],
    )
    out = _write(tmp_path, [stim])
    sample = load_dataset(out)[0]

    assert isinstance(sample.input, list)
    parts = sample.input[0].content
    assert isinstance(parts, list)
    assert isinstance(parts[0], InspectContentText)
    assert parts[0].text == "describe this"
    assert isinstance(parts[1], InspectContentImage)


def test_relative_image_path_resolved_to_absolute_at_load(tmp_path: Path):
    stim = _stimulus(
        "s_0",
        messages=[
            Message(
                role="user",
                content=[
                    ContentText(text="see this"),
                    ContentImage(image="assets/files/foo__abc123def456.png"),
                ],
            )
        ],
    )
    out = _write(tmp_path, [stim])
    # Place a fake asset so resolution has something to point at.
    fake = out / "assets" / "files" / "foo__abc123def456.png"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"\x89PNG\r\n\x1a\n")

    sample = load_dataset(out)[0]
    image = sample.input[0].content[1]
    assert isinstance(image, InspectContentImage)
    assert Path(image.image).is_absolute()
    assert image.image == str(fake)


def test_relative_path_unchanged_when_file_missing(tmp_path: Path):
    # If the asset doesn't exist at load time we leave the path alone rather
    # than silently rewriting to a wrong absolute path.
    stim = _stimulus(
        "s_0",
        messages=[
            Message(
                role="user",
                content=[
                    ContentText(text="x"),
                    ContentImage(image="assets/files/missing__ffffffffffff.png"),
                ],
            )
        ],
    )
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    image = sample.input[0].content[1]
    assert image.image == "assets/files/missing__ffffffffffff.png"


def test_content_image_preserves_detail(tmp_path: Path):
    stim = _stimulus(
        "s_0",
        messages=[
            Message(
                role="user",
                content=[
                    ContentText(text="hi"),
                    ContentImage(image="assets/files/foo__abc.png", detail="high"),
                ],
            )
        ],
    )
    out = _write(tmp_path, [stim])
    sample = load_dataset(out)[0]
    image = sample.input[0].content[1]
    assert isinstance(image, InspectContentImage)
    assert image.detail == "high"


def test_mixed_content_preserves_order(tmp_path: Path):
    stim = _stimulus(
        "s_0",
        messages=[
            Message(
                role="user",
                content=[
                    ContentText(text="first"),
                    ContentImage(image="assets/files/a__111111111111.png"),
                    ContentText(text="third"),
                ],
            )
        ],
    )
    out = _write(tmp_path, [stim])
    parts = load_dataset(out)[0].input[0].content
    assert [type(p) for p in parts] == [
        InspectContentText,
        InspectContentImage,
        InspectContentText,
    ]
    assert parts[0].text == "first"
    assert parts[2].text == "third"


# ---- end-to-end roundtrip ---------------------------------------------------


def test_roundtrip_inline_image_resolves_to_real_file_on_disk(tmp_path: Path):
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"abc"
    stim = _stimulus(
        "s_0",
        messages=[
            Message(role="system", content="be precise"),
            Message(
                role="user",
                content=[
                    ContentText(text="describe"),
                    ContentImage.from_bytes(png_bytes, suffix="png"),
                ],
            ),
            Message(role="assistant", content="ok"),
        ],
    )
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    assert isinstance(sample.input, list)
    assert [m.role for m in sample.input] == ["system", "user", "assistant"]

    user_content = sample.input[1].content
    assert isinstance(user_content[0], InspectContentText)
    assert user_content[0].text == "describe"
    image = user_content[1]
    assert isinstance(image, InspectContentImage)
    assert Path(image.image).is_absolute()
    assert Path(image.image).read_bytes() == png_bytes


# ---- audio / video / document load paths ------------------------------------


def test_load_audio_via_from_bytes_resolves_to_typed_inspect_content(tmp_path: Path):
    wav_bytes = b"RIFF\x00\x00\x00\x00WAVEfmtfake"
    stim = _stimulus(
        "s_audio",
        messages=[
            Message(
                role="user",
                content=[
                    ContentText(text="transcribe"),
                    ContentAudio.from_bytes(wav_bytes, format="wav"),
                ],
            )
        ],
    )
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    parts = sample.input[0].content
    assert isinstance(parts[1], InspectContentAudio)
    assert Path(parts[1].audio).is_absolute()
    assert Path(parts[1].audio).read_bytes() == wav_bytes
    assert parts[1].format == "wav"


def test_load_video_via_from_bytes_resolves_to_typed_inspect_content(tmp_path: Path):
    mp4_bytes = b"\x00\x00\x00\x18ftypmp42fake"
    stim = _stimulus(
        "s_video",
        messages=[
            Message(
                role="user",
                content=[
                    ContentText(text="describe"),
                    ContentVideo.from_bytes(mp4_bytes, format="mp4"),
                ],
            )
        ],
    )
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    parts = sample.input[0].content
    assert isinstance(parts[1], InspectContentVideo)
    assert Path(parts[1].video).is_absolute()
    assert Path(parts[1].video).read_bytes() == mp4_bytes
    assert parts[1].format == "mp4"


def test_load_document_via_from_bytes_resolves_to_typed_inspect_content(tmp_path: Path):
    pdf_bytes = b"%PDF-1.4 fake"
    stim = _stimulus(
        "s_doc",
        messages=[
            Message(
                role="user",
                content=[
                    ContentText(text="summarise"),
                    ContentDocument.from_bytes(
                        pdf_bytes, suffix="pdf", filename="report.pdf"
                    ),
                ],
            )
        ],
    )
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    parts = sample.input[0].content
    assert isinstance(parts[1], InspectContentDocument)
    assert Path(parts[1].document).is_absolute()
    assert Path(parts[1].document).read_bytes() == pdf_bytes
    assert parts[1].filename == "report.pdf"
    assert parts[1].mime_type == "application/pdf"


def test_load_document_without_optional_fields(tmp_path: Path):
    stim = _stimulus(
        "s_doc",
        messages=[
            Message(
                role="user",
                content=[
                    ContentText(text="hi"),
                    ContentDocument(document="assets/files/x__abcdefabcdef.pdf"),
                ],
            )
        ],
    )
    out = _write(tmp_path, [stim])
    # Place a stub so resolution finds something.
    fake = out / "assets" / "files" / "x__abcdefabcdef.pdf"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"%PDF-1.4")

    sample = load_dataset(out)[0]
    part = sample.input[0].content[1]
    assert isinstance(part, InspectContentDocument)
    assert Path(part.document).is_absolute()


# ---- error paths ------------------------------------------------------------


def test_load_dataset_raises_on_missing_manifest(tmp_path: Path):
    out = tmp_path / "broken"
    out.mkdir()
    (out / "stimuli.jsonl").write_text("")

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_dataset(out)


def test_load_dataset_raises_on_missing_jsonl(tmp_path: Path):
    out = tmp_path / "broken"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"name": "x"}))

    with pytest.raises(FileNotFoundError, match="stimuli.jsonl"):
        load_dataset(out)


def test_empty_dataset_loads_with_zero_samples(tmp_path: Path):
    out = _write(tmp_path, [], name="empty")

    ds = load_dataset(out)
    assert isinstance(ds, MemoryDataset)
    assert ds.name == "empty"
    assert len(ds) == 0


# ---- functional spec --------------------------------------------------------


def test_stimulus_metadata_exposes_functional_when_set(tmp_path: Path):
    stim = Stimulus(
        sample_id="s_0",
        spec=SampleSpec(demands={"cap_exp"}, params={"i": 0}),
        functional=SampleSpec(demands={"cap_func"}, params={"shared": 1}),
        messages=[Message(role="user", content="hi")],
        target="x",
    )
    out = _write(tmp_path, [stim])
    sample = load_dataset(out)[0]
    block = sample.metadata["_stimulus"]
    assert block["functional"] == {"demands": ["cap_func"], "params": {"shared": 1}}
    assert block["spec"] == {"demands": ["cap_exp"], "params": {"i": 0}}


def test_stimulus_metadata_exposes_functional_as_none_when_absent(tmp_path: Path):
    out = _write(tmp_path, [_stimulus("s_0")])
    sample = load_dataset(out)[0]
    assert sample.metadata["_stimulus"]["functional"] is None
