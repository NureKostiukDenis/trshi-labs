import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.index.sentece_indexig import encode_query, load_model
from src.index.vectorization import QdrantVectorizer

DEFAULT_EVAL_PATH = Path("tests/eval-set.jsonl")


@dataclass
class QueryEvalResult:
    """Evaluation outcome for a single query against Qdrant."""

    eval_id: str
    scenario: str
    query: str
    expected_source: str
    expected_title: str
    rank: int | None

    @property
    def hit_at_5(self) -> bool:
        return self.rank is not None and self.rank <= 5

    @property
    def hit_at_10(self) -> bool:
        return self.rank is not None and self.rank <= 10


def load_eval_set(eval_path: Path = DEFAULT_EVAL_PATH) -> list[dict[str, Any]]:
    """Load query -> relevant_chunk records from JSONL."""
    return [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_matching_chunk(
    hit_payload: dict[str, Any],
    expected_chunk: dict[str, Any],
) -> bool:
    """Check whether a retrieved Qdrant payload matches the expected relevant chunk."""
    return QdrantVectorizer.generate_point_id(hit_payload) == QdrantVectorizer.generate_point_id(expected_chunk)


def compute_recall_at_k(results: list[QueryEvalResult], k: int) -> float:
    """Compute Recall@K (fraction of queries with the relevant chunk in top-K)."""
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.rank is not None and r.rank <= k)
    return hits / len(results)


def compute_mrr_at_k(results: list[QueryEvalResult], k: int = 10) -> float:
    """Compute Mean Reciprocal Rank@K."""
    if not results:
        return 0.0
    total_rr = sum(1.0 / r.rank for r in results if r.rank is not None and r.rank <= k)
    return total_rr / len(results)


def evaluate_retrieval(
    eval_path: Path = DEFAULT_EVAL_PATH,
    top_k: int = 10,
) -> list[QueryEvalResult]:
    """Run all eval queries against Qdrant and record the rank of each relevant chunk."""
    cfg = load_config()
    model = load_model(cfg.index.model_name, device=cfg.index.device)
    vectorizer = QdrantVectorizer(cfg.qdrant)

    records = load_eval_set(eval_path)
    results: list[QueryEvalResult] = []

    for rec in records:
        query = str(rec["query"])
        expected = rec["relevant_chunk"]
        expected_id = QdrantVectorizer.generate_point_id(expected)

        vector = encode_query(
            model,
            query,
            query_instruction=cfg.search.query_instruction,
        )
        hits = vectorizer.search(vector, search_config=cfg.search, limit=top_k)

        found_rank: int | None = None
        for idx, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            if hit.id == expected_id or is_matching_chunk(payload, expected):
                found_rank = idx
                break

        results.append(
            QueryEvalResult(
                eval_id=str(rec["id"]),
                scenario=str(rec.get("scenario", "")),
                query=query,
                expected_source=str(expected.get("source", "")),
                expected_title=str(expected.get("title", "")),
                rank=found_rank,
            )
        )

    return results


def print_eval_report(results: list[QueryEvalResult]) -> None:
    """Render per-query ranks and aggregate Recall@5 / Recall@10 / MRR@10."""
    console = Console()
    table = Table(title="Retrieval Evaluation (Recall@5 / Recall@10)")
    table.add_column("ID", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Expected Title", style="white")
    table.add_column("Rank", justify="right")
    table.add_column("R@5", justify="center")
    table.add_column("R@10", justify="center")

    for r in results:
        rank_str = str(r.rank) if r.rank is not None else "-"
        r5 = "[green]OK[/green]" if r.hit_at_5 else "[red]MISS[/red]"
        r10 = "[green]OK[/green]" if r.hit_at_10 else "[red]MISS[/red]"
        table.add_row(r.eval_id, r.expected_source, r.expected_title, rank_str, r5, r10)

    console.print(table)

    r5_score = compute_recall_at_k(results, 5)
    r10_score = compute_recall_at_k(results, 10)
    mrr_score = compute_mrr_at_k(results, 10)
    console.print(
        f"[bold]Total queries:[/bold] {len(results)} | "
        f"[bold green]Recall@5:[/bold green] {r5_score:.2%} | "
        f"[bold green]Recall@10:[/bold green] {r10_score:.2%} | "
        f"[bold cyan]MRR@10:[/bold cyan] {mrr_score:.4f}"
    )


def main() -> None:
    results = evaluate_retrieval(DEFAULT_EVAL_PATH, top_k=20)
    print_eval_report(results)


if __name__ == "__main__":
    main()
