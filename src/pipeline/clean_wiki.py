import re
import shutil
from pathlib import Path

import mwparserfromhell
from tqdm import tqdm

from pipeline.wiki_tables import convert_wikitables_to_markdown

BLACKLISTED_WIKI_PAGES = {
    "Recycler Blacklist (Data)",
}

COLOR_TEMPLATES = {"color", "colour", "textcolor", "bg", "bgcolor", "nowrap"}

HTML_STYLE_TAGS = (
    "span",
    "font",
    "div",
    "p",
    "center",
    "small",
    "big",
    "b",
    "strong",
    "i",
    "em",
    "u",
    "s",
    "strike",
    "del",
    "ins",
    "sup",
    "sub",
    "blockquote",
    "code",
    "tt",
    "kbd",
    "var",
    "pre",
    "nowiki",
    "syntaxhighlight",
    "source",
    "mark",
    "abbr",
    "cite",
    "q",
)

STYLE_ATTR_PATTERN = re.compile(
    r"""\s+(?:style|class|bgcolor|color|align|valign|width|height|border|cellpadding|cellspacing)\s*=\s*(?:"[^"]*"|'[^']*'|[^\s|>]+)""",
    re.IGNORECASE,
)

HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
REF_TAG_PATTERN = re.compile(
    r"<ref(?:\s[^>]*?)?(?<!/)>(.*?)</ref>",
    re.IGNORECASE | re.DOTALL,
)
REF_EMPTY_PATTERN = re.compile(
    r"<ref(?:\s[^>]*?)?/>|<references(?:\s[^>]*?)?/?>|</references>",
    re.IGNORECASE,
)
GALLERY_PATTERN = re.compile(
    r"<gallery(?:\s[^>]*)?>(.*?)</gallery>",
    re.IGNORECASE | re.DOTALL,
)
BR_PATTERN = re.compile(r"</?br(?:\s[^>]*)?/?>|<div\s*/>", re.IGNORECASE)
STYLE_TAG_OPEN_CLOSE = re.compile(
    rf"</?(?:{'|'.join(HTML_STYLE_TAGS)})(?:\s[^>]*)?\s*/?>",
    re.IGNORECASE,
)
BOLD_ITALIC_PATTERN = re.compile(r"'{2,5}(.+?)'{2,5}")

FILE_OPTION_KEYWORDS = {
    "thumb",
    "thumbnail",
    "frame",
    "framed",
    "frameless",
    "border",
    "left",
    "right",
    "center",
    "centre",
    "none",
    "baseline",
    "sub",
    "super",
    "top",
    "text-top",
    "middle",
    "bottom",
    "text-bottom",
    "upright",
}
FILE_SIZE_PATTERN = re.compile(r"^\d*(?:x\d+)?px$", re.IGNORECASE)
FILE_KV_OPTION_PATTERN = re.compile(
    r"^(?:upright|alt|link|class|page|lang|thumb|thumbnail)\s*=",
    re.IGNORECASE,
)


def _unwrap_color_templates(text: str) -> str:
    """Replace {{Color|...|content}} templates with their inner text."""
    wikicode = mwparserfromhell.parse(text)
    for template in reversed(wikicode.filter_templates(recursive=True)):
        name = str(template.name).strip().lower()
        if name not in COLOR_TEMPLATES:
            continue
        if template.has("text"):
            replacement = str(template.get("text").value)
        elif template.has(2):
            replacement = str(template.get(2).value)
        elif template.has(1):
            replacement = str(template.get(1).value)
        else:
            replacement = ""
        try:
            wikicode.replace(template, replacement)
        except ValueError:
            pass
    return str(wikicode)


def _split_top_level_pipes(text: str) -> list[str]:
    """Split text on '|' only outside nested [[...]] and {{...}}."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    i = 0
    while i < len(text):
        two = text[i : i + 2]
        if two in ("[[", "{{"):
            depth += 1
            buf.append(two)
            i += 2
            continue
        if two in ("]]", "}}") and depth > 0:
            depth -= 1
            buf.append(two)
            i += 2
            continue
        if text[i] == "|" and depth == 0:
            parts.append("".join(buf))
            buf.clear()
        else:
            buf.append(text[i])
        i += 1
    parts.append("".join(buf))
    return parts


def _extract_file_caption(link_text: str) -> str:
    """Extract the caption from [[File:...|...]] parameters, ignoring layout options."""
    caption_parts: list[str] = []
    for raw_part in _split_top_level_pipes(link_text):
        part = raw_part.strip()
        if not part:
            continue
        lowered = part.lower()
        if lowered in FILE_OPTION_KEYWORDS or FILE_SIZE_PATTERN.match(part) or FILE_KV_OPTION_PATTERN.match(part):
            continue
        caption_parts.append(part)
    return " ".join(caption_parts).strip()


def _remove_file_links(text: str) -> str:
    """Replace [[File:...]] and [[Image:...]] embeds with their caption text if present."""
    wikicode = mwparserfromhell.parse(text)
    for link in reversed(wikicode.filter_wikilinks(recursive=True)):
        title = str(link.title).strip().lower()
        if not title.startswith(("file:", "image:")):
            continue
        caption = _extract_file_caption(str(link.text)) if link.text is not None else ""
        try:
            if caption:
                wikicode.replace(link, f"{caption} ")
            else:
                wikicode.remove(link)
        except ValueError:
            pass
    return str(wikicode)


def _replace_ref_tags(text: str) -> str:
    """Replace <ref>footnote</ref> with ' (footnote)' and drop empty ref/references tags."""
    text = REF_EMPTY_PATTERN.sub("", text)
    return REF_TAG_PATTERN.sub(
        lambda m: f" ({m.group(1).strip()})" if m.group(1).strip() else "",
        text,
    )


def _unwrap_infobox_templates(text: str) -> str:
    """Convert {{Infobox ...}} templates into '(Key: Value, ...)' summaries."""
    wikicode = mwparserfromhell.parse(text)
    for template in reversed(wikicode.filter_templates(recursive=True)):
        name = str(template.name).strip().lower()
        if not name.startswith("infobox"):
            continue
        pairs: list[str] = []
        for param in template.params:
            key = str(param.name).strip()
            if key.lower() in ("image", "img"):
                continue
            val = str(param.value).strip()
            if not val:
                continue
            pairs.append(f"{key.capitalize()}: {val}")
        replacement = f"({', '.join(pairs)})\n\n" if pairs else ""
        try:
            wikicode.replace(template, replacement)
        except ValueError:
            pass
    return str(wikicode)


def _unwrap_wikilinks(text: str) -> str:
    """Convert [[Target|Label]] -> Label and [[Target]] -> Target (preserving [[Category:...]])."""
    wikicode = mwparserfromhell.parse(text)
    for link in reversed(wikicode.filter_wikilinks(recursive=True)):
        title = str(link.title).strip()
        if title.lower().startswith("category:"):
            continue
        label = str(link.text).strip() if link.text is not None else title
        try:
            wikicode.replace(link, label)
        except ValueError:
            pass
    return str(wikicode)


def _replace_gallery_tags(text: str) -> str:
    """Replace <gallery>...</gallery> blocks with their extracted image captions."""

    def _convert_block(match: re.Match[str]) -> str:
        body = match.group(1)
        captions: list[str] = []
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or "|" not in line:
                continue
            _, rest = line.split("|", maxsplit=1)
            caption = _extract_file_caption(rest)
            if caption:
                captions.append(caption)

        if not captions:
            return ""

        prefix_char = text[match.start() - 1] if match.start() > 0 else "\n"
        if prefix_char == "=":
            return ", ".join(captions)
        leading = "\n" if prefix_char not in ("\n", "") else ""
        return leading + "\n".join(captions)

    return GALLERY_PATTERN.sub(_convert_block, text)


def clean_wikitext_styling(text: str) -> str:
    """Remove visual styling elements from MediaWiki wikitext while preserving content."""
    text = HTML_COMMENT_PATTERN.sub("", text)
    text = _unwrap_color_templates(text)
    text = _replace_gallery_tags(text)
    text = _remove_file_links(text)
    text = _unwrap_infobox_templates(text)
    text = _unwrap_wikilinks(text)
    text = _replace_ref_tags(text)
    text = BR_PATTERN.sub(" ", text)
    text = STYLE_TAG_OPEN_CLOSE.sub("", text)

    # Remove wikitext bold/italic quotes (''...'' and '''...''')
    text = BOLD_ITALIC_PATTERN.sub(r"\1", text)

    # Expand MediaWiki tables (with rowspan/colspan) into Markdown tables
    text = convert_wikitables_to_markdown(text)

    # Remove any remaining inline CSS/HTML styling attributes outside tables
    text = STYLE_ATTR_PATTERN.sub("", text)

    # Normalize trailing spaces and excessive blank lines
    lines = [re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned + "\n" if cleaned else ""


def clean_wiki_pages(source_root: Path, target_root: Path) -> int:
    """Clean all .txt wiki files from source_root and write results to target_root."""
    if target_root.exists():
        shutil.rmtree(target_root)

    files = sorted(p for p in source_root.rglob("*.txt") if p.is_file() and p.stem not in BLACKLISTED_WIKI_PAGES)
    count_cleaned = 0

    for src_path in tqdm(files, desc="Cleaning wiki styling", unit="file"):
        rel_path = src_path.relative_to(source_root)
        dst_path = target_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        raw_text = src_path.read_text(encoding="utf-8")
        cleaned_text = clean_wikitext_styling(raw_text)
        dst_path.write_text(cleaned_text, encoding="utf-8")
        count_cleaned += 1

    return count_cleaned


def main() -> None:
    count = clean_wiki_pages(Path("data/wiki"), Path("data/wiki_clean"))
    print(f"Cleaned {count} wiki files -> data/wiki_clean")


if __name__ == "__main__":
    main()
