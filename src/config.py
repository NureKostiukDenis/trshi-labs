import os
import sqlite3
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_FILE = Path("config/pipeline.toml")


@dataclass
class DatabaseConfig:
    """Configuration for connecting to the nesql item/recipe database."""

    backend: str = "sqlite"
    sqlite_path: Path = Path("data/nesql.db")
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "nesql"
    pg_user: str = "postgres"
    pg_password: str = ""


@dataclass
class IndexConfig:
    """Configuration for embedding model and chunk indexing."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str | None = None
    batch_size: int = 64
    normalize_embeddings: bool = True
    chunk_files: list[str] = field(
        default_factory=lambda: ["wiki_chunks.jsonl", "quest_chunks.jsonl"]
    )


@dataclass
class QdrantConfig:
    """Configuration for connecting to Qdrant (local storage or remote server)."""

    mode: str = "local"
    path: Path = Path("data/qdrant")
    url: str = "http://localhost:6333"
    api_key: str = ""
    collection_name: str = "gtnh_rag"
    distance: str = "cosine"


@dataclass
class PipelineConfig:
    """Central configuration for data downloading, cleaning, enrichment, and chunking."""

    wiki_url: str = "https://wiki.gtnewhorizons.com/wiki/Main_Page"
    git_url: str = "https://github.com/GTNewHorizons/GT-New-Horizons-Modpack/tree/master"
    git_quests_path: str = "/config/betterquesting/DefaultQuests/Quests"
    wiki_dir: Path = Path("data/wiki")
    wiki_clean_dir: Path = Path("data/wiki_clean")
    quests_dir: Path = Path("data/quests")
    quests_clean_dir: Path = Path("data/quests_clean")
    quests_enriched_dir: Path = Path("data/quests_enriched")
    chunks_dir: Path = Path("data/chunks")
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)


def load_config(config_path: Path = DEFAULT_CONFIG_FILE) -> PipelineConfig:
    """Load pipeline configuration from TOML file (if present) and environment variables."""
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    paths_cfg = raw.get("paths", {})
    sources_cfg = raw.get("sources", {})
    db_cfg = raw.get("database", {})
    index_cfg = raw.get("index", {})
    qdrant_cfg = raw.get("qdrant", {})

    db = DatabaseConfig(
        backend=os.getenv("NESQL_DB_BACKEND", db_cfg.get("backend", "sqlite")),
        sqlite_path=Path(os.getenv("NESQL_SQLITE_PATH", db_cfg.get("sqlite_path", "data/nesql.db"))),
        pg_host=os.getenv("NESQL_PG_HOST", db_cfg.get("pg_host", "localhost")),
        pg_port=int(os.getenv("NESQL_PG_PORT", db_cfg.get("pg_port", 5432))),
        pg_database=os.getenv("NESQL_PG_DATABASE", db_cfg.get("pg_database", "nesql")),
        pg_user=os.getenv("NESQL_PG_USER", db_cfg.get("pg_user", "postgres")),
        pg_password=os.getenv("NESQL_PG_PASSWORD", db_cfg.get("pg_password", "")),
    )

    index = IndexConfig(
        model_name=os.getenv("INDEX_MODEL_NAME", index_cfg.get("model_name", IndexConfig.model_name)),
        device=os.getenv("INDEX_DEVICE", index_cfg.get("device")),
        batch_size=int(os.getenv("INDEX_BATCH_SIZE", index_cfg.get("batch_size", IndexConfig.batch_size))),
        normalize_embeddings=bool(
            index_cfg.get("normalize_embeddings", IndexConfig.normalize_embeddings)
        ),
        chunk_files=list(
            index_cfg.get("chunk_files", ["wiki_chunks.jsonl", "quest_chunks.jsonl"])
        ),
    )

    qdrant = QdrantConfig(
        mode=os.getenv("QDRANT_MODE", qdrant_cfg.get("mode", QdrantConfig.mode)),
        path=Path(os.getenv("QDRANT_PATH", qdrant_cfg.get("path", "data/qdrant"))),
        url=os.getenv("QDRANT_URL", qdrant_cfg.get("url", QdrantConfig.url)),
        api_key=os.getenv("QDRANT_API_KEY", qdrant_cfg.get("api_key", "")),
        collection_name=os.getenv(
            "QDRANT_COLLECTION",
            qdrant_cfg.get("collection_name", QdrantConfig.collection_name),
        ),
        distance=os.getenv(
            "QDRANT_DISTANCE", qdrant_cfg.get("distance", QdrantConfig.distance)
        ),
    )

    return PipelineConfig(
        wiki_url=sources_cfg.get("wiki_url", PipelineConfig.wiki_url),
        git_url=sources_cfg.get("git_url", PipelineConfig.git_url),
        git_quests_path=sources_cfg.get("git_quests_path", PipelineConfig.git_quests_path),
        wiki_dir=Path(paths_cfg.get("wiki_dir", "data/wiki")),
        wiki_clean_dir=Path(paths_cfg.get("wiki_clean_dir", "data/wiki_clean")),
        quests_dir=Path(paths_cfg.get("quests_dir", "data/quests")),
        quests_clean_dir=Path(paths_cfg.get("quests_clean_dir", "data/quests_clean")),
        quests_enriched_dir=Path(paths_cfg.get("quests_enriched_dir", "data/quests_enriched")),
        chunks_dir=Path(paths_cfg.get("chunks_dir", "data/chunks")),
        db=db,
        index=index,
        qdrant=qdrant,
    )


def get_db_connection(db_config: DatabaseConfig) -> Any:
    """Open a DBAPI-compatible connection to SQLite or PostgreSQL."""
    backend = db_config.backend.lower()
    if backend == "sqlite":
        if not db_config.sqlite_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {db_config.sqlite_path}")
        conn = sqlite3.connect(db_config.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    if backend in ("postgres", "postgresql"):
        try:
            import psycopg  # type: ignore[import-not-found]

            return psycopg.connect(
                host=db_config.pg_host,
                port=db_config.pg_port,
                dbname=db_config.pg_database,
                user=db_config.pg_user,
                password=db_config.pg_password,
            )
        except ImportError:
            try:
                import psycopg2  # type: ignore[import-not-found]

                return psycopg2.connect(
                    host=db_config.pg_host,
                    port=db_config.pg_port,
                    dbname=db_config.pg_database,
                    user=db_config.pg_user,
                    password=db_config.pg_password,
                )
            except ImportError as exc:
                raise ImportError("PostgreSQL backend requires 'psycopg' or 'psycopg2' installed.") from exc

    raise ValueError(f"Unsupported database backend: {db_config.backend}")


def get_qdrant_client(qdrant_config: QdrantConfig) -> Any:
    """Create a QdrantClient in local embedded mode or remote server mode."""
    try:
        from qdrant_client import QdrantClient  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Qdrant integration requires 'qdrant-client' installed (.venv/bin/pip install qdrant-client)."
        ) from exc

    mode = qdrant_config.mode.lower()
    if mode == "local":
        qdrant_config.path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(qdrant_config.path))
    if mode == "server":
        return QdrantClient(
            url=qdrant_config.url,
            api_key=qdrant_config.api_key or None,
        )
    raise ValueError(f"Unsupported Qdrant mode: {qdrant_config.mode}")
