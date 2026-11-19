import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import DatabaseConfig, get_db_connection, load_config

MACHINE_TYPE_PATTERN = re.compile(r"^Machine Type:\s*(.+)$", re.MULTILINE)
VOLTAGE_PATTERN = re.compile(r"^Voltage IN:\s*(.+)$", re.MULTILINE)
SNBT_KEY_PATTERN = re.compile(r"([{,])\s*([A-Za-z0-9_.-]+)\s*:")
SNBT_NUM_SUFFIX_PATTERN = re.compile(r":\s*(-?\d+(?:\.\d+)?)[bslfdBSLFD]\b")


@dataclass(frozen=True)
class ItemInfo:
    """Normalized item metadata loaded from the nesql 'item' table."""

    nesql_id: str
    localized_name: str
    machine_type: str | None = None
    voltage: str | None = None


@dataclass
class ItemLookup:
    """Multi-tier lookup tables for resolving quest items against nesql."""

    by_nbt: dict[tuple[str, str, int, str], ItemInfo]
    no_nbt: dict[tuple[str, str, int], ItemInfo]
    any_item: dict[tuple[str, str, int], ItemInfo]
    ore_dict: dict[str, str]


def _parse_tooltip_fields(tooltip: str | None) -> tuple[str | None, str | None]:
    """Extract Machine Type and Voltage IN from a nesql item tooltip if present."""
    if not tooltip:
        return None, None
    machine_match = MACHINE_TYPE_PATTERN.search(tooltip)
    voltage_match = VOLTAGE_PATTERN.search(tooltip)
    machine_type = machine_match.group(1).strip() if machine_match else None
    voltage = voltage_match.group(1).strip() if voltage_match else None
    return machine_type, voltage


def _normalize_nbt(nbt_value: Any) -> str:
    """Normalize a Python dict or SNBT/JSON string into a canonical compact string."""
    if nbt_value is None:
        return ""
    if isinstance(nbt_value, dict):
        return json.dumps(nbt_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    raw = str(nbt_value).strip()
    if not raw:
        return ""

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    # Convert common SNBT forms ({Key:1b, ...}) to JSON when possible
    json_like = SNBT_KEY_PATTERN.sub(r'\1"\2":', raw)
    json_like = SNBT_NUM_SUFFIX_PATTERN.sub(r":\1", json_like)
    try:
        parsed = json.loads(json_like)
        if isinstance(parsed, dict):
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    return re.sub(r"\s+", "", raw)


def load_item_lookup(db_config: DatabaseConfig) -> ItemLookup:
    """Load item (with/without NBT) and ore_dictionary mappings from nesql DB."""
    by_nbt: dict[tuple[str, str, int, str], ItemInfo] = {}
    no_nbt: dict[tuple[str, str, int], ItemInfo] = {}
    any_item: dict[tuple[str, str, int], ItemInfo] = {}
    ore_dict: dict[str, str] = {}

    item_query = "SELECT id, mod_id, internal_name, item_damage, localized_name, tooltip, nbt FROM item"
    ore_query = "SELECT id, name FROM ore_dictionary"

    with get_db_connection(db_config) as conn:
        cursor = conn.cursor()
        cursor.execute(item_query)
        for row in cursor.fetchall():
            if isinstance(row, tuple):
                nesql_id, mod_id, internal_name, item_damage, localized_name, tooltip, nbt = row
            else:
                nesql_id = row["id"]
                mod_id = row["mod_id"]
                internal_name = row["internal_name"]
                item_damage = row["item_damage"]
                localized_name = row["localized_name"]
                tooltip = row["tooltip"]
                nbt = row["nbt"]

            base_key = (str(mod_id), str(internal_name), int(item_damage or 0))
            machine_type, voltage = _parse_tooltip_fields(tooltip)
            info = ItemInfo(
                nesql_id=str(nesql_id),
                localized_name=str(localized_name or "").strip(),
                machine_type=machine_type,
                voltage=voltage,
            )

            any_item.setdefault(base_key, info)
            norm_nbt = _normalize_nbt(nbt)
            if norm_nbt:
                by_nbt[(base_key[0], base_key[1], base_key[2], norm_nbt)] = info
            else:
                no_nbt.setdefault(base_key, info)

        try:
            cursor.execute(ore_query)
            for row in cursor.fetchall():
                if isinstance(row, tuple):
                    od_id, od_name = row
                else:
                    od_id = row["id"]
                    od_name = row["name"]
                if od_name:
                    ore_dict[str(od_name)] = str(od_id)
        except Exception:
            pass

    return ItemLookup(
        by_nbt=by_nbt,
        no_nbt=no_nbt,
        any_item=any_item,
        ore_dict=ore_dict,
    )


def build_quest_name_lookup(source_root: Path) -> dict[tuple[int, int], str]:
    """Build (questIDHigh, questIDLow) -> quest_name mapping across cleaned quests."""
    quest_names: dict[tuple[int, int], str] = {}
    for path in source_root.rglob("*.json"):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        q_high = data.get("questIDHigh")
        q_low = data.get("questIDLow")
        name = data.get("properties", {}).get("betterquesting", {}).get("name")
        if q_high is not None and q_low is not None and name:
            quest_names[(int(q_high), int(q_low))] = str(name)
    return quest_names


def _resolve_item_info(
    mod_id: str,
    internal_name: str,
    damage: int,
    tag: Any,
    lookup: ItemLookup,
) -> ItemInfo | None:
    """Resolve item preferring exact NBT match, then non-NBT, then any matching item."""
    if tag:
        norm_tag = _normalize_nbt(tag)
        if norm_tag:
            exact = lookup.by_nbt.get((mod_id, internal_name, damage, norm_tag))
            if exact is not None:
                return exact

    base_key = (mod_id, internal_name, damage)
    return lookup.no_nbt.get(base_key) or lookup.any_item.get(base_key)


def _enrich_item(
    item: dict[str, Any],
    lookup: ItemLookup,
) -> dict[str, Any]:
    """Attach localized_name, nesql_id, machine_type, voltage, or ore_dict fallback."""
    raw_id = str(item.get("id", ""))
    if ":" not in raw_id:
        return item

    mod_id, internal_name = raw_id.split(":", maxsplit=1)
    damage = int(item.get("Damage", 0))
    info = _resolve_item_info(mod_id, internal_name, damage, item.get("tag"), lookup)

    enriched = dict(item)
    if info is not None:
        enriched["nesql_id"] = info.nesql_id
        if info.localized_name:
            enriched["name"] = info.localized_name
        if info.machine_type:
            enriched["machine_type"] = info.machine_type
        if info.voltage:
            enriched["voltage"] = info.voltage
        return enriched

    # Fallback to ore_dictionary if exact item was not found
    ore_name = str(item.get("OreDict", "")).strip()
    if ore_name and ore_name in lookup.ore_dict:
        enriched["nesql_id"] = lookup.ore_dict[ore_name]
        enriched["name"] = ore_name
    elif ore_name:
        enriched["name"] = ore_name

    return enriched


def enrich_quest_data(
    quest: dict[str, Any],
    item_lookup: ItemLookup,
    quest_names: dict[tuple[int, int], str],
) -> dict[str, Any]:
    """Enrich a single cleaned quest dict with item metadata and prerequisite names."""
    enriched = dict(quest)

    tasks = enriched.get("tasks")
    if isinstance(tasks, list):
        new_tasks: list[dict[str, Any]] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            new_task = dict(task)
            req_items = new_task.get("requiredItems")
            if isinstance(req_items, list):
                new_task["requiredItems"] = [
                    _enrich_item(item, item_lookup) if isinstance(item, dict) else item for item in req_items
                ]
            new_tasks.append(new_task)
        enriched["tasks"] = new_tasks

    prereqs = enriched.get("preRequisites")
    if isinstance(prereqs, dict):
        prereq_names: list[str] = []
        for entry in prereqs.values():
            if not isinstance(entry, dict):
                continue
            q_high = entry.get("questIDHigh")
            q_low = entry.get("questIDLow")
            if q_high is not None and q_low is not None:
                name = quest_names.get((int(q_high), int(q_low)))
                if name:
                    prereq_names.append(name)
        if prereq_names:
            enriched["prerequisiteNames"] = prereq_names

    return enriched


def enrich_quests(
    source_root: Path,
    target_root: Path,
    db_config: DatabaseConfig,
) -> int:
    """Enrich all cleaned quest JSONs from source_root and write to target_root."""
    if target_root.exists():
        shutil.rmtree(target_root)

    item_lookup = load_item_lookup(db_config)
    quest_names = build_quest_name_lookup(source_root)
    files = sorted(p for p in source_root.rglob("*.json") if p.is_file())

    count_enriched = 0
    for src_path in tqdm(files, desc="Enriching quests", unit="file"):
        rel_path = src_path.relative_to(source_root)
        dst_path = target_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        quest = json.loads(src_path.read_text(encoding="utf-8"))
        enriched = enrich_quest_data(quest, item_lookup, quest_names)
        dst_path.write_text(
            json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        count_enriched += 1

    return count_enriched


def main() -> None:
    cfg = load_config()
    count = enrich_quests(cfg.quests_clean_dir, cfg.quests_enriched_dir, cfg.db)
    print(f"Enriched {count} quest files -> {cfg.quests_enriched_dir}")


if __name__ == "__main__":
    main()
