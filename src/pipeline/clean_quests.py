import json
import re
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

MC_FORMAT_PATTERN = re.compile(r"§[0-9a-fklmnor]", re.IGNORECASE)
NBT_SUFFIX_PATTERN = re.compile(r":\d+$")
HASH_SUFFIX_PATTERN = re.compile(r"-[A-Za-z0-9_-]{22}==$")

IGNORED_BETTERQUESTING_KEYS = {
    "completion_animation",
    "completion_particle",
    "confetti_icon",
    "icon",
    "particle_count",
    "snd_complete",
    "snd_update",
    "autoClaim",
    "countAsQuest",
    "globalShare",
    "isGlobal",
    "isSilent",
    "lockedProgress",
    "partySingleReward",
    "repeatTime",
    "repeat_relative",
    "simultaneous",
    "visibility",
}

IGNORED_TASK_KEYS = {
    "autoConsume",
    "consume",
    "groupDetect",
    "ignoreNBT",
    "index",
    "partialMatch",
}

IGNORED_FOOD_IDS = {
    "BiomesOPlenty:food",
    "minecraft:apple",
    "minecraft:baked_potato",
    "minecraft:bread",
    "minecraft:cake",
    "minecraft:carrot",
    "minecraft:cooked_beef",
    "minecraft:cooked_chicken",
    "minecraft:cooked_fished",
    "minecraft:cooked_porkchop",
    "minecraft:cookie",
    "minecraft:golden_apple",
    "minecraft:golden_carrot",
    "minecraft:melon",
    "minecraft:mushroom_stew",
    "minecraft:potato",
    "minecraft:pumpkin_pie",
}


def strip_mc_formatting(text: str) -> str:
    """Remove Minecraft color and formatting codes (§x) from text."""
    return MC_FORMAT_PATTERN.sub("", text).strip()


def strip_nbt_suffixes(obj: Any) -> Any:
    """Recursively strip NBT type suffixes (':1', ':8', ':10', etc.) from dict keys."""
    if isinstance(obj, dict):
        return {NBT_SUFFIX_PATTERN.sub("", k): strip_nbt_suffixes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_nbt_suffixes(item) for item in obj]
    return obj


def _is_ignored_reward_item(item_id: str) -> bool:
    """Return True if the reward item is a coin, lootbag, or food."""
    if not item_id:
        return True
    if item_id.startswith("dreamcraft:Coin"):
        return True
    if item_id == "enhancedlootbags:lootbag":
        return True
    if item_id in IGNORED_FOOD_IDS or item_id.startswith(("harvestcraft:", "SpiceOfLife:")):
        return True
    return False


def _extract_reward_label(item: dict[str, Any]) -> str | None:
    """Extract a compact name or ID for a non-ignored reward item."""
    item_id = str(item.get("id", ""))
    if _is_ignored_reward_item(item_id):
        return None
    display_name = item.get("tag", {}).get("display", {}).get("Name")
    if isinstance(display_name, str) and display_name.strip():
        return strip_mc_formatting(display_name)
    return item_id


def _clean_tasks(tasks: dict[str, Any]) -> list[dict[str, Any]]:
    """Remove technical fields from quest tasks."""
    cleaned_tasks: list[dict[str, Any]] = []
    for task in tasks.values():
        if not isinstance(task, dict):
            continue
        cleaned_task = {k: v for k, v in task.items() if k not in IGNORED_TASK_KEYS}
        if isinstance(cleaned_task.get("requiredItems"), dict):
            items: list[dict[str, Any]] = []
            for item in cleaned_task["requiredItems"].values():
                if not isinstance(item, dict):
                    continue
                compact_item = {k: v for k, v in item.items() if not (k == "OreDict" and v == "")}
                items.append(compact_item)
            cleaned_task["requiredItems"] = items
        cleaned_tasks.append(cleaned_task)
    return cleaned_tasks


def _clean_rewards(rewards: dict[str, Any]) -> list[str]:
    """Drop coins, lootbags, and food; keep compact names/IDs for remaining rewards."""
    labels: list[str] = []
    for reward in rewards.values():
        if not isinstance(reward, dict):
            continue
        for container_key in ("rewards", "choices"):
            container = reward.get(container_key)
            if not isinstance(container, dict):
                continue
            for item in container.values():
                if not isinstance(item, dict):
                    continue
                label = _extract_reward_label(item)
                if label:
                    labels.append(label)
    return labels


def clean_quest_data(raw_quest: dict[str, Any]) -> dict[str, Any]:
    """Clean a single BetterQuesting quest JSON object."""
    data = strip_nbt_suffixes(raw_quest)

    props = data.get("properties", {}).get("betterquesting", {})
    if isinstance(props, dict):
        cleaned_props: dict[str, Any] = {}
        for k, v in props.items():
            if k.startswith("notification_") or k in IGNORED_BETTERQUESTING_KEYS:
                continue
            if k in ("name", "desc") and isinstance(v, str):
                cleaned_props[k] = strip_mc_formatting(v)
            else:
                cleaned_props[k] = v
        data["properties"] = {"betterquesting": cleaned_props}

    if isinstance(data.get("tasks"), dict):
        data["tasks"] = _clean_tasks(data["tasks"])

    if isinstance(data.get("rewards"), dict):
        data["rewards"] = _clean_rewards(data["rewards"])

    return data


def _clean_rel_path(rel_path: Path) -> Path:
    """Strip BetterQuesting base64 hash suffixes from folder and file names."""
    cleaned_dirs = [HASH_SUFFIX_PATTERN.sub("", part) for part in rel_path.parts[:-1]]
    cleaned_stem = HASH_SUFFIX_PATTERN.sub("", rel_path.stem)
    return Path(*cleaned_dirs, f"{cleaned_stem}{rel_path.suffix}")


def _resolve_unique_path(dst_path: Path) -> Path:
    """Avoid overwriting if two quests in the same folder share a truncated stem."""
    if not dst_path.exists():
        return dst_path
    idx = 2
    while True:
        candidate = dst_path.with_name(f"{dst_path.stem}_{idx}{dst_path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def clean_quests(source_root: Path, target_root: Path) -> int:
    """Clean all quest JSON files from source_root and write results to target_root."""
    if target_root.exists():
        shutil.rmtree(target_root)

    files = sorted(p for p in source_root.rglob("*.json") if p.is_file())
    count_cleaned = 0

    for src_path in tqdm(files, desc="Cleaning quests", unit="file"):
        rel_path = _clean_rel_path(src_path.relative_to(source_root))
        dst_path = _resolve_unique_path(target_root / rel_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        raw_quest = json.loads(src_path.read_text(encoding="utf-8"))
        cleaned_quest = clean_quest_data(raw_quest)
        dst_path.write_text(
            json.dumps(cleaned_quest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        count_cleaned += 1

    return count_cleaned


def main() -> None:
    count = clean_quests(Path("data/quests"), Path("data/quests_clean"))
    print(f"Cleaned {count} quest files -> data/quests_clean")


if __name__ == "__main__":
    main()
