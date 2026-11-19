from pathlib import Path

from index.sentece_indexig import (
    DEFAULT_MODEL_NAME,
    encode_chunks_jsonl,
    encode_query,
    load_model,
)

MODEL_NAME = DEFAULT_MODEL_NAME
DEVICE: str | None = None  # None = auto (CUDA if available, else CPU)
BATCH_SIZE = 64
NORMALIZE_EMBEDDINGS = True

CHUNK_FILES = [
    Path("data/chunks/wiki_chunks.jsonl"),
    Path("data/chunks/quest_chunks.jsonl"),
]


def main() -> None:
    model = load_model(MODEL_NAME, device=DEVICE)
    dim = model.get_embedding_dimension()
    print(f"Loaded model '{MODEL_NAME}' (dim={dim}, device={model.device})")

    for chunk_path in CHUNK_FILES:
        if not chunk_path.exists():
            print(f"Skipping missing file: {chunk_path}")
            continue
        encoded = encode_chunks_jsonl(
            model,
            chunk_path,
            batch_size=BATCH_SIZE,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=True,
        )
        print(f"Encoded {len(encoded)} chunks from {chunk_path}")

    sample_vec = encode_query(model, "How to build Alloy Blast Smelter?")
    print(f"Sample query vector length: {len(sample_vec)}")


if __name__ == "__main__":
    main()
