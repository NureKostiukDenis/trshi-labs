from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(kw_only=True)
class GeneralChunk:
    """Base chunk dataclass with fields common to all Qdrant payloads."""

    text: str
    title: str
    source: str = ""
    chunk_index: int = 0

    def to_payload(self) -> dict[str, Any]:
        """Serialize chunk fields into a Qdrant-compatible payload dictionary."""
        return asdict(self)


@dataclass(kw_only=True)
class WikiChunk(GeneralChunk):
    """Chunk dataclass for cleaned MediaWiki articles."""

    source: str = "wiki"
    section: str = ""
    tier: str | None = None
    categories: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class QuestChunk(GeneralChunk):
    """Chunk dataclass for cleaned BetterQuesting quests."""

    source: str = "quest"
    chapter: str = ""
    quest_id: str = ""
    is_main: bool = False
    required_items: list[str] = field(default_factory=list)
    required_item_names: list[str] = field(default_factory=list)
    rewards: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
