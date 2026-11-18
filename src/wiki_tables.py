import re

ROWSPAN_PATTERN = re.compile(r"""\browspan\s*=\s*["']?(\d+)["']?""", re.IGNORECASE)
COLSPAN_PATTERN = re.compile(r"""\bcolspan\s*=\s*["']?(\d+)["']?""", re.IGNORECASE)


def _split_outside_brackets(text: str, delimiter: str) -> list[str]:
    """Split text by delimiter only outside [[...]], {{...}}, and <...>."""
    parts: list[str] = []
    buf: list[str] = []
    bracket_depth = 0
    angle_depth = 0
    i = 0
    delim_len = len(delimiter)

    while i < len(text):
        two = text[i : i + 2]
        if two in ("[[", "{{"):
            bracket_depth += 1
            buf.append(two)
            i += 2
            continue
        if two in ("]]", "}}") and bracket_depth > 0:
            bracket_depth -= 1
            buf.append(two)
            i += 2
            continue

        ch = text[i]
        if ch == "<":
            angle_depth += 1
        elif ch == ">" and angle_depth > 0:
            angle_depth -= 1

        if (
            bracket_depth == 0
            and angle_depth == 0
            and text[i : i + delim_len] == delimiter
        ):
            parts.append("".join(buf))
            buf.clear()
            i += delim_len
            continue

        buf.append(ch)
        i += 1

    parts.append("".join(buf))
    return parts


def _parse_cell(raw_cell: str) -> tuple[int, int, str]:
    """Parse a single MediaWiki cell into (rowspan, colspan, content)."""
    parts = _split_outside_brackets(raw_cell, "|")
    if len(parts) >= 2:
        attrs = parts[0]
        content = "|".join(parts[1:])
    else:
        attrs = ""
        content = parts[0]

    rowspan_match = ROWSPAN_PATTERN.search(attrs)
    colspan_match = COLSPAN_PATTERN.search(attrs)
    rowspan = max(1, int(rowspan_match.group(1))) if rowspan_match else 1
    colspan = max(1, int(colspan_match.group(1))) if colspan_match else 1
    return rowspan, colspan, content.strip()


def _parse_wikitable(lines: list[str]) -> tuple[str | None, list[list[tuple[int, int, str]]]]:
    """Parse MediaWiki table lines into an optional caption and raw rows."""
    caption: str | None = None
    rows: list[list[tuple[int, int, str]]] = []
    current_row: list[tuple[int, int, str]] = []

    def flush_row() -> None:
        nonlocal current_row
        if current_row:
            rows.append(current_row)
            current_row = []

    for line in lines[1:-1]:  # skip opening '{|' and closing '|}'
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("|+"):
            caption = stripped[2:].strip()
            continue

        if stripped.startswith("|-"):
            flush_row()
            continue

        if stripped.startswith("|") or stripped.startswith("!"):
            sep = "||" if stripped.startswith("|") else "!!"
            raw_cells = _split_outside_brackets(stripped[1:], sep)
            for raw_cell in raw_cells:
                current_row.append(_parse_cell(raw_cell))
            continue

        if current_row:
            rs, cs, prev_text = current_row[-1]
            joined = f"{prev_text} {stripped}".strip()
            current_row[-1] = (rs, cs, joined)

    flush_row()
    return caption, rows


def _expand_spans(rows: list[list[tuple[int, int, str]]]) -> list[list[str]]:
    """Expand rowspan and colspan into a uniform 2D grid, skipping empty rows."""
    grid: list[list[str]] = []
    active_spans: dict[int, tuple[int, str]] = {}

    for raw_row in rows:
        row_cells: dict[int, str] = {}

        # Fill columns occupied by rowspans from previous rows
        next_active: dict[int, tuple[int, str]] = {}
        for col_idx, (remaining, text) in active_spans.items():
            row_cells[col_idx] = text
            if remaining > 1:
                next_active[col_idx] = (remaining - 1, text)
        active_spans = next_active

        col_idx = 0
        for rowspan, colspan, text in raw_row:
            while col_idx in row_cells:
                col_idx += 1
            for offset in range(colspan):
                target_col = col_idx + offset
                row_cells[target_col] = text
                if rowspan > 1:
                    active_spans[target_col] = (rowspan - 1, text)
            col_idx += colspan

        if not row_cells:
            continue

        max_col = max(row_cells) + 1
        expanded_row = [row_cells.get(c, "").strip() for c in range(max_col)]
        if any(expanded_row):
            grid.append(expanded_row)

    if not grid:
        return []

    width = max(len(r) for r in grid)
    return [r + [""] * (width - len(r)) for r in grid]


def _format_markdown_table(caption: str | None, grid: list[list[str]]) -> list[str]:
    """Format an expanded 2D grid as a Markdown table."""
    if not grid:
        return [caption] if caption else []

    out: list[str] = []
    if caption:
        out.append(caption)

    def fmt_row(row: list[str]) -> str:
        escaped = [cell.replace("|", "\\|") for cell in row]
        return "| " + " | ".join(escaped) + " |"

    out.append(fmt_row(grid[0]))
    out.append("| " + " | ".join("---" for _ in grid[0]) + " |")
    for row in grid[1:]:
        out.append(fmt_row(row))
    return out


BORDER_ZERO_PATTERN = re.compile(r"""\bborder\s*=\s*["']?0["']?""", re.IGNORECASE)
MD_TABLE_SEP_PATTERN = re.compile(r"^\|\s*---+\s*\|")
WIKI_CELL_ATTR_PREFIX = re.compile(
    r"^\|\s*(?:colspan|rowspan|style|class|width|height|align|valign|bgcolor)\s*=",
    re.IGNORECASE,
)


def _unwrap_layout_table(table_lines: list[str]) -> list[str]:
    """Unwrap an outer layout table (e.g. border="0" or wrapping nested tables)."""
    out: list[str] = []
    for line in table_lines[1:-1]:
        stripped = line.strip()
        if not stripped or stripped.startswith("|-") or stripped.startswith("|+"):
            continue
        if stripped.startswith("|") and not stripped.startswith("|---"):
            # Keep already-converted Markdown table rows intact
            if (
                stripped.endswith("|")
                and stripped.count("|") >= 2
                and not WIKI_CELL_ATTR_PREFIX.match(stripped)
            ):
                out.append(line)
                continue
            parts = _split_outside_brackets(stripped[1:], "|")
            content = "|".join(parts[1:]).strip() if len(parts) >= 2 else parts[0].strip()
            if content:
                out.append(content)
            continue
        out.append(line)
    return out


def _find_innermost_table(lines: list[str]) -> tuple[int, int] | None:
    """Find (start, end_exclusive) of the first innermost {| ... |} table."""
    stack: list[int] = []
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("{|"):
            stack.append(idx)
        elif stripped.startswith("|}") and stack:
            start = stack.pop()
            return start, idx + 1
    return None


def convert_wikitables_to_markdown(text: str) -> str:
    """Convert all MediaWiki tables ({| ... |}) in text to expanded Markdown tables."""
    lines = text.splitlines()

    while (span := _find_innermost_table(lines)) is not None:
        start, end = span
        table_lines = lines[start:end]
        header = table_lines[0]
        has_nested_md_table = any(
            MD_TABLE_SEP_PATTERN.match(l.lstrip()) for l in table_lines[1:-1]
        )
        is_layout = bool(BORDER_ZERO_PATTERN.search(header)) or has_nested_md_table

        if is_layout:
            replacement = _unwrap_layout_table(table_lines)
        else:
            caption, raw_rows = _parse_wikitable(table_lines)
            grid = _expand_spans(raw_rows)
            replacement = _format_markdown_table(caption, grid)

        lines[start:end] = replacement

    return "\n".join(lines)
