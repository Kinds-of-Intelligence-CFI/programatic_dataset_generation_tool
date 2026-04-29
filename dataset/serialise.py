from dataclasses import fields, is_dataclass
from typing import Any

from dataset.stimulus import Content, Message, Stimulus
from generation.generate import Spec

_JSON_PRIMITIVES = (str, int, float, bool, type(None))


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, _JSON_PRIMITIVES):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, set | frozenset):
        return sorted(to_jsonable(x) for x in obj)
    if isinstance(obj, list | tuple):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    raise TypeError(f"Cannot serialise object of type {type(obj).__name__}")


def stimulus_to_dict(s: Stimulus) -> dict:
    return to_jsonable(s)


def stimulus_from_dict(d: dict) -> Stimulus:
    spec = Spec(
        capabilities=set(d["spec"]["capabilities"]),
        params=dict(d["spec"]["params"]),
    )
    messages = [_message_from_dict(m) for m in d["messages"]]
    return Stimulus(
        sample_id=d["sample_id"],
        spec=spec,
        messages=messages,
        target=d["target"],
        validators_ran=list(d.get("validators_ran", [])),
        metadata=dict(d.get("metadata", {})),
    )


def _message_from_dict(d: dict) -> Message:
    content = d["content"]
    if isinstance(content, list):
        content = [Content(type=c["type"], data=c["data"]) for c in content]
    return Message(role=d["role"], content=content)
