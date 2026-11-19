from src.config import load_config
from src.index.sentece_indexig import (
    encode_chunks_jsonl,
    load_model,
)
from src.index.vectorization import QdrantVectorizer


def main() -> None:
    cfg = load_config()
    idx_cfg = cfg.index

    model = load_model(idx_cfg.model_name, device=idx_cfg.device)
    dim = model.get_embedding_dimension()

    if dim is None:
        dim = 512

    print(f"Loaded model '{idx_cfg.model_name}' (dim={dim}, device={model.device})")

    vectorizer = QdrantVectorizer(cfg.qdrant)
    vectorizer.ensure_collection(vector_size=dim)
    vectorizer.create_payload_indexes()

    total_upserted = 0
    for filename in idx_cfg.chunk_files:
        chunk_path = cfg.chunks_dir / filename
        if not chunk_path.exists():
            print(f"Skipping missing file: {chunk_path}")
            continue

        encoded = encode_chunks_jsonl(
            model,
            chunk_path,
            batch_size=idx_cfg.batch_size,
            normalize_embeddings=idx_cfg.normalize_embeddings,
            show_progress_bar=True,
        )
        points = vectorizer.build_points(encoded)
        upserted = vectorizer.upsert_points(points)
        total_upserted += upserted
        print(f"Upserted {upserted} points from {chunk_path} into '{vectorizer.collection_name}'")

    print(f"Total upserted points: {total_upserted}")


if __name__ == "__main__":
    main()
