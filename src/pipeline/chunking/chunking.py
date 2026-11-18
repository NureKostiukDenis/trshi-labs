import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tqdm import tqdm

from data import GeneralChunk


class ChunkingStrategy(ABC):
    """Abstract base strategy for splitting a document into chunks."""

    @abstractmethod
    def chunk(self, text: str, title: str, **metadata: Any) -> list[GeneralChunk]:
        """Split document text into a list of GeneralChunk instances."""


class Chunker:
    """Context class that executes a configurable ChunkingStrategy."""

    def __init__(self, strategy: ChunkingStrategy) -> None:
        self.strategy = strategy

    def set_strategy(self, strategy: ChunkingStrategy) -> None:
        """Swap the active chunking strategy at runtime."""
        self.strategy = strategy

    def chunk_text(self, text: str, title: str, **metadata: Any) -> list[GeneralChunk]:
        """Chunk a single document using the active strategy."""
        return self.strategy.chunk(text, title=title, **metadata)

    def chunk_directory(
        self,
        source_root: Path,
        glob_pattern: str = "*.txt",
        output_jsonl: Path | None = None,
    ) -> list[GeneralChunk]:
        """Chunk all matching files in source_root and optionally write JSONL."""
        files = sorted(p for p in source_root.rglob(glob_pattern) if p.is_file())
        all_chunks: list[GeneralChunk] = []

        for src_path in tqdm(files, desc="Chunking files", unit="file"):
            title = src_path.stem
            text = src_path.read_text(encoding="utf-8")
            chunks = self.chunk_text(
                text,
                title=title,
                relative_path=str(src_path.relative_to(source_root)),
            )
            all_chunks.extend(chunks)

        if output_jsonl is not None:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with output_jsonl.open("w", encoding="utf-8") as f:
                for chunk in all_chunks:
                    f.write(json.dumps(chunk.to_payload(), ensure_ascii=False) + "\n")

        return all_chunks
