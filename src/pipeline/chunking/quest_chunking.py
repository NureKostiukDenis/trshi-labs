import json
from pathlib import Path
from typing import Any

from chunking.chunking import Chunker, ChunkingStrategy
from data import QuestChunk


def _format_item_id(item: dict[str, Any]) -> str:
    """Format an item dict as 'mod:item:damage' (or 'mod:item' if damage is 0)."""
    item_id = str(item.get("id", "")).strip()
    damage = item.get("Damage", 0)
    if damage:
        return f"{item_id}:{damage}"
    return item_id


class QuestChunkingStrategy(ChunkingStrategy):
    """Chunking strategy for cleaned BetterQuesting JSON files."""

    def __init__(self, max_chars: int = 3000, overlap_chars: int = 300) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def _split_long_text(self, text: str, budget: int) -> list[str]:
        """Split overly long quest descriptions while preserving word boundaries."""
        if len(text) <= budget:
            return [text]

        chunks: list[str] = []
        start = 0
        step = max(1, budget - self.overlap_chars)
        while start < len(text):
            end = min(len(text), start + budget)
            if end < len(text):
                space_idx = text.rfind(" ", start, end)
                if space_idx > start + budget // 2:
                    end = space_idx
            part = text[start:end].strip()
            if part:
                chunks.append(part)
            if end >= len(text):
                break
            start = max(start + step, end - self.overlap_chars)
        return chunks

    @staticmethod
    def _extract_chapter(metadata: dict[str, Any]) -> str:
        if metadata.get("chapter"):
            return str(metadata["chapter"])
        rel_path = metadata.get("relative_path")
        if rel_path:
            parts = Path(str(rel_path)).parts
            if len(parts) > 1:
                return parts[0]
        return ""

    @staticmethod
    def _extract_prerequisites(data: dict[str, Any]) -> list[str]:
        prereqs = data.get("preRequisites", {})
        if not isinstance(prereqs, dict):
            return []
        result: list[str] = []
        for entry in prereqs.values():
            if isinstance(entry, dict) and "questIDHigh" in entry and "questIDLow" in entry:
                result.append(f"{entry['questIDHigh']}:{entry['questIDLow']}")
        return result

    @staticmethod
    def _format_readable_item(item: dict[str, Any], fallback_id: str) -> str:
        """Format an enriched item with localized name, machine type, and voltage."""
        count = item.get("Count", 1)
        name = str(item.get("name") or item.get("OreDict") or "").strip()
        if not name:
            return f"{fallback_id} x{count}"

        details = [str(item[k]).strip() for k in ("machine_type", "voltage") if item.get(k)]
        if details:
            return f"{name} ({', '.join(details)}) x{count}"
        return f"{name} x{count}"

    @classmethod
    def _extract_required_items(cls, data: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
        """Return (metadata_item_ids, item_names, human_readable_lines) from tasks."""
        tasks = data.get("tasks", [])
        if not isinstance(tasks, list):
            return [], [], []

        item_ids: list[str] = []
        item_names: list[str] = []
        readable: list[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            for item in task.get("requiredItems", []):
                if not isinstance(item, dict):
                    continue
                formatted_id = _format_item_id(item)
                if not formatted_id:
                    continue
                meta_id = str(item.get("nesql_id") or formatted_id)
                item_ids.append(meta_id)
                if item.get("name"):
                    item_names.append(str(item["name"]).strip())
                readable.append(cls._format_readable_item(item, formatted_id))
        return (
            list(dict.fromkeys(item_ids)),
            list(dict.fromkeys(item_names)),
            readable,
        )

    def chunk(self, text: str, title: str, **metadata: Any) -> list[QuestChunk]:
        data = json.loads(text)
        props = data.get("properties", {}).get("betterquesting", {})

        quest_title = str(props.get("name") or title).strip()
        desc = str(props.get("desc", "")).strip()
        chapter = self._extract_chapter(metadata)
        quest_id = f"{data.get('questIDHigh', 0)}:{data.get('questIDLow', 0)}"
        is_main = bool(props.get("isMain", 0))
        prerequisites = self._extract_prerequisites(data)
        required_items, required_item_names, readable_items = self._extract_required_items(data)

        rewards = [str(r) for r in data.get("rewards", []) if r]

        header = f"[{chapter} > {quest_title}]" if chapter else f"[{quest_title}]"

        body_sections: list[str] = []
        if desc:
            body_sections.append(desc)
        if readable_items:
            body_sections.append("Required items: " + ", ".join(readable_items))
        if rewards:
            body_sections.append("Rewards: " + ", ".join(rewards))

        full_body = "\n\n".join(body_sections).strip()
        if not full_body:
            full_body = quest_title

        budget = max(200, self.max_chars - len(header) - 2)
        text_parts = self._split_long_text(full_body, budget)

        return [
            QuestChunk(
                text=f"{header}\n{part}",
                title=quest_title,
                chunk_index=idx,
                chapter=chapter,
                quest_id=quest_id,
                is_main=is_main,
                required_items=required_items,
                required_item_names=required_item_names,
                rewards=rewards,
                prerequisites=prerequisites,
            )
            for idx, part in enumerate(text_parts)
        ]


def main() -> None:
    from config import load_config

    cfg = load_config()
    source_dir = cfg.quests_enriched_dir if cfg.quests_enriched_dir.exists() else cfg.quests_clean_dir
    chunker = Chunker(QuestChunkingStrategy())
    chunks = chunker.chunk_directory(
        source_dir,
        glob_pattern="*.json",
        output_jsonl=cfg.chunks_dir / "quest_chunks.jsonl",
    )
    print(f"Created {len(chunks)} quest chunks -> {cfg.chunks_dir / 'quest_chunks.jsonl'}")


if __name__ == "__main__":
    main()
