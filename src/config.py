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


def load_config(config_path: Path = DEFAULT_CONFIG_FILE) -> PipelineConfig:
    """Load pipeline configuration from TOML file (if present) and environment variables."""
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    paths_cfg = raw.get("paths", {})
    sources_cfg = raw.get("sources", {})
    db_cfg = raw.get("database", {})

    db = DatabaseConfig(
        backend=os.getenv("NESQL_DB_BACKEND", db_cfg.get("backend", "sqlite")),
        sqlite_path=Path(os.getenv("NESQL_SQLITE_PATH", db_cfg.get("sqlite_path", "data/nesql.db"))),
        pg_host=os.getenv("NESQL_PG_HOST", db_cfg.get("pg_host", "localhost")),
        pg_port=int(os.getenv("NESQL_PG_PORT", db_cfg.get("pg_port", 5432))),
        pg_database=os.getenv("NESQL_PG_DATABASE", db_cfg.get("pg_database", "nesql")),
        pg_user=os.getenv("NESQL_PG_USER", db_cfg.get("pg_user", "postgres")),
        pg_password=os.getenv("NESQL_PG_PASSWORD", db_cfg.get("pg_password", "")),
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
