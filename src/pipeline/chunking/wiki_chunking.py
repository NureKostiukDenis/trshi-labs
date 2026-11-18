import re
from pathlib import Path
from typing import Any

from chunking.chunking import Chunker, ChunkingStrategy
from data import WikiChunk

HEADING_PATTERN = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
CATEGORY_PATTERN = re.compile(r"\[\[Category:([^\]|]+)(?:\|[^\]]*)?\]\]", re.IGNORECASE)
INFOBOX_TIER_PATTERN = re.compile(r"^\|\s*tier\s*=\s*(.+)$", re.IGNORECASE | re.MULTILINE)
WIKILINK_PATTERN = re.compile(r"\[\[([^|\]]+\|)?([^\]]+)\]\]")


def _strip_wikilink_markup(text: str) -> str:
    """Convert [[Target|Label]] or [[Target]] to plain label text."""
    return WIKILINK_PATTERN.sub(r"\2", text).strip()


def _extract_categories(text: str) -> tuple[list[str], str]:
    """Extract [[Category:...]] names and remove category tags from text."""
    categories = [m.group(1).strip() for m in CATEGORY_PATTERN.finditer(text)]
    cleaned = CATEGORY_PATTERN.sub("", text)
    return categories, cleaned


def _extract_tier(text: str) -> str | None:
    """Extract voltage tier from an Infobox if present."""
    match = INFOBOX_TIER_PATTERN.search(text)
    if not match:
        return None
    tier = _strip_wikilink_markup(match.group(1))
    return tier or None


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split MediaWiki text into (section_path, body) pairs by == headings ==."""
    sections: list[tuple[str, str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_path = "Overview"
    buf: list[str] = []

    for line in text.splitlines():
        match = HEADING_PATTERN.match(line.strip())
        if match:
            body = "\n".join(buf).strip()
            if body:
                sections.append((current_path, body))
            buf.clear()

            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_path = " > ".join(h for _, h in heading_stack)
        else:
            buf.append(line)

    body = "\n".join(buf).strip()
    if body:
        sections.append((current_path, body))

    return sections


def _linearize_markdown_table(lines: list[str]) -> list[str]:
    """Convert a Markdown table into '(Header: Value | ...)' lines."""
    if len(lines) <= 2:
        return lines

    def parse_row(line: str) -> list[str]:
        stripped = line.strip().strip("|")
        return [cell.replace("\\|", "|").strip() for cell in stripped.split("|")]

    headers = parse_row(lines[0])
    linearized: list[str] = []
    for row_line in lines[2:]:
        cells = parse_row(row_line)
        pairs = [
            f"{headers[idx]}: {cells[idx]}"
            for idx in range(min(len(headers), len(cells)))
            if cells[idx]
        ]
        if pairs:
            linearized.append("(" + " | ".join(pairs) + ")")
    return linearized


def _split_markdown_table(lines: list[str], max_chars: int) -> list[str]:
    """Split a Markdown table by rows while repeating its header in every chunk."""
    if len(lines) <= 2:
        return ["\n".join(lines)]

    header = lines[:2]
    header_len = sum(len(l) + 1 for l in header)
    chunks: list[str] = []
    current = list(header)
    current_len = header_len

    for row in lines[2:]:
        row_len = len(row) + 1
        if len(current) > 2 and current_len + row_len > max_chars:
            chunks.append("\n".join(current))
            current = list(header)
            current_len = header_len
        current.append(row)
        current_len += row_len

    if len(current) > 2:
        chunks.append("\n".join(current))

    return chunks


class WikiChunkingStrategy(ChunkingStrategy):
    """Structural + table-aware chunking strategy for cleaned MediaWiki pages."""

    def __init__(
        self,
        max_chars: int = 2000,
        overlap_chars: int = 200,
        linearize_table_min_rows: int = 10,
        min_chars: int = 300,
    ) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.linearize_table_min_rows = linearize_table_min_rows
        self.min_chars = min_chars

    def _merge_short_sections(
        self, sections: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Merge short sections (< min_chars) with adjacent sections within max_chars."""
        if len(sections) <= 1 or self.min_chars <= 0:
            return sections

        def combine_paths(p1: str, p2: str) -> str:
            if p1 == p2:
                return p1
            top1 = p1.split(" > ", maxsplit=1)[0]
            top2 = p2.split(" > ", maxsplit=1)[0]
            if top1 == top2:
                return top1
            return f"{p1} + {p2}"

        def combine_bodies(p1: str, b1: str, p2: str, b2: str) -> str:
            sub2 = p2.rsplit(" > ", maxsplit=1)[-1]
            return f"{b1}\n\n{sub2}:\n{b2}"

        budget = max(200, self.max_chars - 120)
        merged: list[tuple[str, str]] = []
        i = 0

        while i < len(sections):
            path, body = sections[i]
            while (
                len(body) < self.min_chars
                and i + 1 < len(sections)
                and len(body) + len(sections[i + 1][1]) + 20 <= budget
            ):
                next_path, next_body = sections[i + 1]
                body = combine_bodies(path, body, next_path, next_body)
                path = combine_paths(path, next_path)
                i += 1

            if (
                len(body) < self.min_chars
                and merged
                and len(merged[-1][1]) + len(body) + 20 <= budget
            ):
                prev_path, prev_body = merged[-1]
                merged[-1] = (
                    combine_paths(prev_path, path),
                    combine_bodies(prev_path, prev_body, path, body),
                )
            else:
                merged.append((path, body))
            i += 1

        return merged

    def _split_section_body(self, body: str, prefix: str) -> list[str]:
        """Split a section body into chunks within max_chars, keeping tables intact."""
        budget = max(200, self.max_chars - len(prefix) - 2)
        blocks = self._group_blocks(body)
        parts: list[str] = []
        current_blocks: list[str] = []
        current_len = 0

        def flush() -> None:
            nonlocal current_blocks, current_len
            if current_blocks:
                parts.append(f"{prefix}\n" + "\n\n".join(current_blocks))
                current_blocks = []
                current_len = 0

        for is_table, block_lines in blocks:
            if is_table and len(block_lines) - 2 > self.linearize_table_min_rows:
                flush()
                linear_lines = _linearize_markdown_table(block_lines)
                for text_part in self._split_lines_to_budget(linear_lines, budget):
                    parts.append(f"{prefix}\n{text_part}")
                continue

            block_text = "\n".join(block_lines)
            if is_table and len(block_text) > budget:
                flush()
                for table_part in _split_markdown_table(block_lines, budget):
                    parts.append(f"{prefix}\n{table_part}")
                continue

            if len(block_text) > budget:
                flush()
                for text_part in self._split_long_text(block_text, budget):
                    parts.append(f"{prefix}\n{text_part}")
                continue

            added_len = len(block_text) + (2 if current_blocks else 0)
            if current_blocks and current_len + added_len > budget:
                flush()
            current_blocks.append(block_text)
            current_len += added_len

        flush()
        return parts

    @staticmethod
    def _group_blocks(body: str) -> list[tuple[bool, list[str]]]:
        """Group section lines into (is_markdown_table, lines) blocks."""
        blocks: list[tuple[bool, list[str]]] = []
        lines = body.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue

            if line.lstrip().startswith("|"):
                table_lines: list[str] = []
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                is_md_table = len(table_lines) >= 2 and "---" in table_lines[1]
                blocks.append((is_md_table, table_lines))
                continue

            para_lines: list[str] = []
            while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith("|"):
                para_lines.append(lines[i])
                i += 1
            blocks.append((False, para_lines))

        return blocks

    @staticmethod
    def _split_lines_to_budget(lines: list[str], budget: int) -> list[str]:
        """Group lines into chunks without exceeding budget."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in lines:
            added = len(line) + (1 if current else 0)
            if current and current_len + added > budget:
                chunks.append("\n".join(current))
                current = [line]
                current_len = len(line)
            else:
                current.append(line)
                current_len += added
        if current:
            chunks.append("\n".join(current))
        return chunks

    def _split_long_text(self, text: str, budget: int) -> list[str]:
        """Split a long plain-text block with character overlap."""
        chunks: list[str] = []
        start = 0
        step = max(1, budget - self.overlap_chars)
        while start < len(text):
            end = min(len(text), start + budget)
            if end < len(text):
                space_idx = text.rfind(" ", start, end)
                if space_idx > start + budget // 2:
                    end = space_idx
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(start + step, end - self.overlap_chars)
        return [c for c in chunks if c]

    def chunk(self, text: str, title: str, **metadata: Any) -> list[WikiChunk]:
        categories, cleaned_text = _extract_categories(text)
        tier = _extract_tier(cleaned_text)
        sections = self._merge_short_sections(_split_sections(cleaned_text))

        chunks: list[WikiChunk] = []
        chunk_idx = 0
        for section_path, body in sections:
            prefix = f"[{title} > {section_path}]"
            for chunk_text in self._split_section_body(body, prefix):
                chunks.append(
                    WikiChunk(
                        text=chunk_text,
                        title=title,
                        chunk_index=chunk_idx,
                        section=section_path,
                        tier=tier,
                        categories=categories,
                    )
                )
                chunk_idx += 1
        return chunks


def main() -> None:
    chunker = Chunker(WikiChunkingStrategy())
    chunks = chunker.chunk_directory(
        Path("data/wiki_clean"),
        output_jsonl=Path("data/chunks/wiki_chunks.jsonl"),
    )
    print(f"Created {len(chunks)} wiki chunks -> data/chunks/wiki_chunks.jsonl")


if __name__ == "__main__":
    main()
