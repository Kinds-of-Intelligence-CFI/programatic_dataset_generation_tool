from dataclasses import dataclass, field

from dataset.stimulus import Stimulus


@dataclass
class Dataset:
    name: str
    stimuli: list[Stimulus]
    manifest: dict = field(default_factory=dict)
