import argparse
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from config import load_config
from index.sentece_indexig import encode_query, load_model
from index.vectorization import QdrantVectorizer

console = Console()


def _build_filter(source: str | None) -> Filter | None:
    if not source or source == "all":
        return None
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


def _render_results(hits: list[Any], query: str, source: str) -> None:
    if not hits:
        console.print("[yellow]No results found.[/yellow]")
        return

    summary = Table(title=f"Search: {query!r} (source={source})", show_lines=False)
    summary.add_column("#", style="cyan", width=3)
    summary.add_column("Score", style="green", width=7)
    summary.add_column("Source", style="magenta", width=7)
    summary.add_column("Title", style="bold white")
    summary.add_column("Location / Meta", style="dim")

    for idx, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        src = str(payload.get("source", "-"))
        title = str(payload.get("title", "-"))
        if src == "wiki":
            meta = str(payload.get("section", ""))
            if payload.get("tier"):
                meta += f" | tier={payload['tier']}"
        else:
            meta = str(payload.get("chapter", ""))
            if payload.get("quest_id"):
                meta += f" | id={payload['quest_id']}"
        summary.add_row(str(idx), f"{hit.score:.4f}", src, title, meta)

    console.print(summary)

    for idx, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        text = str(payload.get("text", "")).strip()
        preview = text if len(text) <= 600 else text[:600].rstrip() + "..."
        title = f"[{idx}] {payload.get('title', '-')} ({hit.score:.4f})"
        console.print(Panel(preview, title=title, border_style="blue"))


def run_search(
    vectorizer: QdrantVectorizer,
    model: Any,
    query: str,
    *,
    limit: int = 5,
    source: str = "all",
) -> None:
    vector = encode_query(model, query)
    hits = vectorizer.client.query_points(
        collection_name=vectorizer.collection_name,
        query=vector,
        query_filter=_build_filter(source),
        limit=limit,
    ).points
    _render_results(hits, query, source)


def interactive_tui(vectorizer: QdrantVectorizer, model: Any, limit: int = 5) -> None:
    source = "all"
    console.print(
        Panel(
            "Commands: [bold]/source all|wiki|quest[/bold], "
            "[bold]/top N[/bold], [bold]/help[/bold], [bold]/exit[/bold]",
            title=f"GTNH RAG Search ({vectorizer.collection_name})",
            border_style="green",
        )
    )

    while True:
        try:
            raw = Prompt.ask(f"[bold cyan]search[/bold cyan] [{source}, top={limit}]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        text = raw.strip()
        if not text:
            continue
        if text in ("/exit", "/quit", "exit", "quit"):
            break
        if text == "/help":
            console.print(
                "  /source all|wiki|quest  - filter results by source\n"
                "  /top N                  - set number of results\n"
                "  /exit                   - exit search"
            )
            continue
        if text.startswith("/source"):
            parts = text.split(maxsplit=1)
            if len(parts) == 2 and parts[1] in ("all", "wiki", "quest"):
                source = parts[1]
                console.print(f"[green]Source filter set to: {source}[/green]")
            else:
                console.print("[red]Usage: /source all|wiki|quest[/red]")
            continue
        if text.startswith("/top"):
            parts = text.split(maxsplit=1)
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
                limit = int(parts[1])
                console.print(f"[green]Top-K set to: {limit}[/green]")
            else:
                console.print("[red]Usage: /top N (N > 0)[/red]")
            continue

        run_search(vectorizer, model, text, limit=limit, source=source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive TUI search over Qdrant.")
    parser.add_argument("query", nargs="*", help="Optional one-shot search query.")
    parser.add_argument("--top", type=int, default=5, help="Number of results.")
    parser.add_argument(
        "--source",
        choices=("all", "wiki", "quest"),
        default="all",
        help="Filter by source.",
    )
    args = parser.parse_args()

    cfg = load_config()
    with console.status("[bold green]Loading embedding model and Qdrant...[/bold green]"):
        model = load_model(cfg.index.model_name, device=cfg.index.device)
        vectorizer = QdrantVectorizer(cfg.qdrant)

    if args.query:
        run_search(
            vectorizer,
            model,
            " ".join(args.query),
            limit=args.top,
            source=args.source,
        )
    else:
        interactive_tui(vectorizer, model, limit=args.top)


if __name__ == "__main__":
    main()
