import asyncio
import json
from pathlib import Path

import pytest
from inspect_ai._util.content import ContentImage, ContentText
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelName,
)
from inspect_ai.solver import TaskState

from dataset.stimulus import Content, Message, Stimulus
from dataset.writer import write_dataset
from evaluation.inspect_utils import add_messages_from_metadata, load_dataset
from generation.generate import Spec


def _write(
    tmp_path: Path,
    stimuli: list[Stimulus],
    name: str = "ds",
) -> Path:
    out = tmp_path / name
    specs = [s.spec for s in stimuli] or [Spec()]
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
    capabilities: set[str] | None = None,
) -> Stimulus:
    return Stimulus(
        sample_id=sample_id,
        spec=Spec(
            capabilities=capabilities if capabilities is not None else {"cap_a"},
            params={"n": 1},
        ),
        messages=messages
        if messages is not None
        else [Message(role="user", content="hello")],
        target=target,
        metadata=metadata if metadata is not None else {},
    )


def test_load_dataset_returns_memory_dataset_with_manifest_name(tmp_path: Path):
    out = _write(tmp_path, [_stimulus("s_0"), _stimulus("s_1")], name="demo")

    ds = load_dataset(out)

    assert isinstance(ds, MemoryDataset)
    assert ds.name == "demo"
    assert len(ds) == 2


def test_load_dataset_preserves_jsonl_order_and_core_fields(tmp_path: Path):
    stimuli = [
        _stimulus(f"s_{i}", target=f"t_{i}", capabilities={"cap_a", "cap_b"})
        for i in range(3)
    ]
    out = _write(tmp_path, stimuli)

    ds = load_dataset(out)

    assert [s.id for s in ds] == ["s_0", "s_1", "s_2"]
    assert [s.target for s in ds] == ["t_0", "t_1", "t_2"]
    for sample in ds:
        block = sample.metadata["_stimulus"]
        assert block["spec"]["capabilities"] == ["cap_a", "cap_b"]
        assert block["dataset_dir"] == str(out)
        assert block["validators_ran"] == []
        assert block["messages"] == [{"role": "user", "content": "hello"}]


def test_user_metadata_is_flat_alongside_stimulus_block(tmp_path: Path):
    stim = _stimulus("s_0", metadata={"foo": 1, "bar": "x"})
    out = _write(tmp_path, [stim])

    ds = load_dataset(out)
    sample = ds[0]

    assert sample.metadata["foo"] == 1
    assert sample.metadata["bar"] == "x"
    assert "_stimulus" in sample.metadata
    assert sample.metadata["_stimulus"]["messages"] == [
        {"role": "user", "content": "hello"}
    ]


def test_single_user_text_message_populates_input(tmp_path: Path):
    stim = _stimulus(
        "s_0",
        messages=[Message(role="user", content="what is the capital of France?")],
    )
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    assert sample.input == "what is the capital of France?"


def test_input_shortcut_skipped_for_multiple_messages(tmp_path: Path):
    stim = _stimulus(
        "s_0",
        messages=[
            Message(role="system", content="you are helpful"),
            Message(role="user", content="hi"),
        ],
    )
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    assert sample.input == ""
    assert len(sample.metadata["_stimulus"]["messages"]) == 2


def test_input_shortcut_skipped_for_non_user_role(tmp_path: Path):
    stim = _stimulus(
        "s_0",
        messages=[Message(role="system", content="you are helpful")],
    )
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    assert sample.input == ""


def test_input_shortcut_skipped_for_list_typed_content(tmp_path: Path):
    stim = _stimulus(
        "s_0",
        messages=[
            Message(
                role="user",
                content=[
                    Content(type="text", data="describe this"),
                    Content(type="image", data="assets/foo.png"),
                ],
            )
        ],
    )
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    assert sample.input == ""
    content_blocks = sample.metadata["_stimulus"]["messages"][0]["content"]
    assert isinstance(content_blocks, list)
    assert content_blocks[1]["type"] == "image"


def test_input_shortcut_skipped_for_zero_messages(tmp_path: Path):
    stim = _stimulus("s_0", messages=[])
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    assert sample.input == ""


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


def _state(
    *,
    metadata: dict | None = None,
    messages: list | None = None,
    sample_id: str = "s_0",
    input_: str = "ignored",
) -> TaskState:
    return TaskState(
        model=ModelName("anthropic/claude-3-5-sonnet-20241022"),
        sample_id=sample_id,
        epoch=0,
        input=input_,
        messages=list(messages) if messages else [],
        metadata=metadata if metadata is not None else {},
    )


async def _noop_generate(state, **kwargs):
    return state


def _run_solver(state: TaskState) -> TaskState:
    solver = add_messages_from_metadata()
    return asyncio.run(solver(state, _noop_generate))


def _meta(messages_list: list[dict]) -> dict:
    return {"_stimulus": {"messages": messages_list}}


def test_appends_single_user_string_message():
    state = _state(metadata=_meta([{"role": "user", "content": "hello"}]))

    new_state = _run_solver(state)

    assert len(new_state.messages) == 1
    msg = new_state.messages[0]
    assert isinstance(msg, ChatMessageUser)
    assert msg.content == "hello"


@pytest.mark.parametrize(
    "role,expected_cls",
    [
        ("system", ChatMessageSystem),
        ("user", ChatMessageUser),
        ("assistant", ChatMessageAssistant),
        ("tool", ChatMessageTool),
    ],
)
def test_role_to_class_mapping(role, expected_cls):
    state = _state(metadata=_meta([{"role": role, "content": "x"}]))

    new_state = _run_solver(state)

    assert isinstance(new_state.messages[0], expected_cls)
    assert new_state.messages[0].role == role
    assert new_state.messages[0].content == "x"


def test_preserves_message_order():
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "c"},
        {"role": "user", "content": "d"},
    ]
    state = _state(metadata=_meta(msgs))

    new_state = _run_solver(state)

    assert [m.role for m in new_state.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [m.content for m in new_state.messages] == ["a", "b", "c", "d"]


def test_appends_after_existing_messages():
    seed = ChatMessageUser(content="seed")
    state = _state(
        messages=[seed],
        metadata=_meta(
            [
                {"role": "assistant", "content": "follow1"},
                {"role": "user", "content": "follow2"},
            ]
        ),
    )

    new_state = _run_solver(state)

    assert len(new_state.messages) == 3
    assert new_state.messages[0].content == "seed"
    assert isinstance(new_state.messages[0], ChatMessageUser)
    assert new_state.messages[1].content == "follow1"
    assert isinstance(new_state.messages[1], ChatMessageAssistant)
    assert new_state.messages[2].content == "follow2"
    assert isinstance(new_state.messages[2], ChatMessageUser)


def test_does_not_mutate_input_or_target():
    state = _state(
        input_="original input",
        metadata=_meta([{"role": "user", "content": "x"}]),
    )
    target_before = state.target

    new_state = _run_solver(state)

    assert new_state.input == "original input"
    assert new_state.target is target_before


def test_string_content_remains_string():
    state = _state(metadata=_meta([{"role": "user", "content": "plain text"}]))

    new_state = _run_solver(state)

    assert new_state.messages[0].content == "plain text"


def test_text_content_part_becomes_ContentText():
    state = _state(
        metadata=_meta(
            [{"role": "user", "content": [{"type": "text", "data": "hi"}]}]
        )
    )

    new_state = _run_solver(state)

    content = new_state.messages[0].content
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], ContentText)
    assert content[0].text == "hi"


def test_image_content_part_becomes_ContentImage():
    state = _state(
        metadata=_meta(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "data": "assets/0001.png"}],
                }
            ]
        )
    )

    new_state = _run_solver(state)

    content = new_state.messages[0].content
    assert isinstance(content, list)
    assert len(content) == 1
    assert isinstance(content[0], ContentImage)
    assert content[0].image == "assets/0001.png"


def test_mixed_content_parts_preserve_order():
    state = _state(
        metadata=_meta(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "data": "first"},
                        {"type": "image", "data": "assets/foo.png"},
                        {"type": "text", "data": "third"},
                    ],
                }
            ]
        )
    )

    new_state = _run_solver(state)

    parts = new_state.messages[0].content
    assert isinstance(parts, list)
    assert [type(p) for p in parts] == [ContentText, ContentImage, ContentText]
    assert parts[0].text == "first"
    assert parts[1].image == "assets/foo.png"
    assert parts[2].text == "third"


def test_raises_when_stimulus_key_missing():
    state = _state(metadata={})

    with pytest.raises(Exception):
        _run_solver(state)


def test_raises_when_messages_key_missing():
    state = _state(metadata={"_stimulus": {}})

    with pytest.raises(Exception):
        _run_solver(state)


def test_raises_when_messages_empty_list():
    state = _state(metadata={"_stimulus": {"messages": []}})

    with pytest.raises(Exception):
        _run_solver(state)


def test_roundtrip_with_load_dataset(tmp_path: Path):
    stim = _stimulus(
        "s_0",
        messages=[
            Message(role="system", content="be precise"),
            Message(
                role="user",
                content=[
                    Content(type="text", data="describe"),
                    Content(type="image", data="assets/foo.png"),
                ],
            ),
            Message(role="assistant", content="ok"),
        ],
    )
    out = _write(tmp_path, [stim])

    sample = load_dataset(out)[0]
    state = _state(
        metadata=dict(sample.metadata),
        messages=[],
        sample_id=str(sample.id),
        input_=sample.input if isinstance(sample.input, str) else "",
    )

    new_state = _run_solver(state)

    assert [m.role for m in new_state.messages] == ["system", "user", "assistant"]
    assert new_state.messages[0].content == "be precise"

    user_content = new_state.messages[1].content
    assert isinstance(user_content, list)
    assert isinstance(user_content[0], ContentText)
    assert user_content[0].text == "describe"
    assert isinstance(user_content[1], ContentImage)
    assert user_content[1].image == "assets/foo.png"

    assert new_state.messages[2].content == "ok"
