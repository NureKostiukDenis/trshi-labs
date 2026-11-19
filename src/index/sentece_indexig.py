import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer


def load_model(
    model_name: str,
    device: str | None = None,
) -> SentenceTransformer:
    """Load a SentenceTransformer embedding model."""
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    """Encode a sequence of passages/chunks into embedding vectors."""
    if not texts:
        return []
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
    )
    return embeddings.tolist()


def encode_query(
    model: SentenceTransformer,
    query: str,
    *,
    query_instruction: str,
    normalize_embeddings: bool = True,
) -> list[float]:
    """Encode a single search query into an embedding vector."""
    prompt_text = f"{query_instruction}{query}" if query_instruction else query
    embedding = model.encode(
        prompt_text,
        normalize_embeddings=normalize_embeddings,
    )
    return embedding.tolist()


def load_chunks_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load chunk payload dicts from a JSONL file."""
    return [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_chunks_jsonl(
    model: SentenceTransformer,
    jsonl_path: Path,
    *,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = True,
) -> list[tuple[dict[str, Any], list[float]]]:
    """Load chunks from JSONL and encode each chunk's 'text' field."""
    records = load_chunks_jsonl(jsonl_path)
    texts = [str(rec.get("text", "")) for rec in records]
    vectors = encode_texts(
        model,
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
    )
    return list(zip(records, vectors, strict=True))
