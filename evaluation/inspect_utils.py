import json
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageAssistant, ChatMessageTool, ChatMessageUser, Content, ContentImage, ContentText
from inspect_ai.solver import Generate, TaskState, solver


def load_dataset(path: str | Path) -> MemoryDataset:
    dataset_dir = Path(path)
    manifest_path = dataset_dir / "manifest.json"
    jsonl_path = dataset_dir / "stimuli.jsonl"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {dataset_dir}")
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"stimuli.jsonl not found in {dataset_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = manifest.get("name")

    samples: list[Sample] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(_record_to_sample(json.loads(line), dataset_dir))

    return MemoryDataset(samples=samples, name=name, location=str(dataset_dir))


def _record_to_sample(record: dict[str, Any], dataset_dir: Path) -> Sample:
    messages = record["messages"]
    input_text = ""
    if (
        len(messages) == 1
        and messages[0]["role"] == "user"
        and isinstance(messages[0]["content"], str)
    ):
        input_text = messages[0]["content"]

    metadata: dict[str, Any] = {
        "_stimulus": {
            "messages": messages,
            "spec": record["spec"],
            "validators_ran": record.get("validators_ran", []),
            "dataset_dir": str(dataset_dir),
        },
        **record.get("metadata", {}),
    }

    return Sample(
        id=record["sample_id"],
        input=input_text,
        target=record["target"],
        metadata=metadata,
    )


@solver
def add_messages_from_metadata():
    async def solve(state: TaskState, generate: Generate):
        messages = state.metadata["_stimulus"]["messages"]
        if not messages:
            raise ValueError("_stimulus.messages is empty")
        for message in messages:
            content: str | list[Content] = []
            role = message["role"]
            if isinstance(message["content"], str):
                content = message["content"]
            else:
                content = []
                assert isinstance(message["content"], list)
                for content_item in message["content"]:
                    if content_item["type"] == "text":
                        content.append(ContentText(text=content_item["data"]))
                    elif content_item["type"] == "image":
                        content.append(ContentImage(image=content_item["data"]))
                    else:
                        raise ValueError(f"Unknown content type: {content_item['type']}")
            
            if role == "system":
                message = ChatMessageSystem(content=content)
            elif role == "user":
                message = ChatMessageUser(content=content)
            elif role == "assistant":
                message = ChatMessageAssistant(content=content)
            elif role == "tool":
                message = ChatMessageTool(content=content)
            else:
                raise ValueError(f"Unknown message role: {role}")
            state.messages.append(message)

        return state
    return solve