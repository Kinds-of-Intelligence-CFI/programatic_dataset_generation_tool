from dataclasses import dataclass, field


@dataclass
class Spec:
    capabilities: set[str] = field(default_factory=set)
    params: dict = field(default_factory=dict)


Condition = list[Spec]
