"""The one record type of the public text corpora.

Target semantics: after ``targets.to_difficulty`` a HIGHER target always means HARDER. Loaders
return the source's own scale (CLEAR: BT easiness, higher = easier; OneStop: level 0/1/2 as a
float, higher = harder) and say so in ``DATA_LICENCES.md``.
"""

import json
from dataclasses import asdict, dataclass

SOURCE_CLEAR = "clear"
SOURCE_ONESTOP = "onestop"


@dataclass(frozen=True)
class TextExample:
    source: str
    text: str
    target: float
    level: int | None
    group_id: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if not self.group_id:
            raise ValueError("group_id must not be empty")

    def to_json(self) -> str:
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )

    @classmethod
    def from_json(cls, text: str) -> "TextExample":
        data = json.loads(text)
        level = data["level"]
        return cls(
            source=data["source"],
            text=data["text"],
            target=float(data["target"]),
            level=None if level is None else int(level),
            group_id=data["group_id"],
        )
