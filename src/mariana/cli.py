import atexit
import re
import time
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from mariana.graph import get_graph
from mariana.state import initial_state
from mariana.utils.config import load_config, save_setting
from mariana.utils.llm import check_ollama, get_planner_llm, get_summarizer_llm
from mariana.utils.searxng import SearXNGManager

console = Console()


def _status_label(success: bool, text: str) -> str:
    icon = "[green]✔[/]" if success else "[red]✖[/]"
    return f"{icon} {text}"


def _apply_overrides(cfg, model, iterations):
    if model:
        cfg.planner_model = model
        cfg.summarizer_model = model
        get_planner_llm.cache_clear()
        get_summarizer_llm.cache_clear()
    if iterations is not None:
        cfg.max_iterations = iterations


def _preflight(model=None, iterations=None):
    cfg = load_config()
    _apply_overrides(cfg, model, iterations)

    ollama_ok, result = check_ollama()
    if not ollama_ok:
        console.print(Panel(f"[red]Ollama check failed:[/] {result}", title="Preflight error"))
        raise SystemExit(1)
    console.print(_status_label(True, f"Ollama reachable at {cfg.ollama_base_url}"))
    if isinstance(result, list):
        available_models = result
        console.print(_status_label(True, f"Models available: {', '.join(available_models[:5]) or 'none'}"))
        if cfg.planner_model not in available_models:
            console.print(Panel(
                f"[red]Configured model '{cfg.planner_model}' is not available on Ollama.[/]\n"
                f"Available models: {', '.join(available_models)}\n"
                f"Pull one with: ollama pull {cfg.planner_model}",
                title="Model unavailable"
            ))
            raise SystemExit(1)

    manager = SearXNGManager()
    try:
        port = manager.ensure_running()
    except Exception as exc:
        console.print(Panel(f"[red]SearXNG failed to start:[/] {exc}", title="Preflight error"))
        raise SystemExit(1)
    console.print(_status_label(True, f"SearXNG running on port {port}"))
    atexit.register(manager.stop)
    return cfg, manager


def _run_query(query: str, cfg):
    console.print(Panel(f"[blue]Research started:[/] {query}", title="Mariana"))
    console.print(f"[white]Expected iterations:[/] {cfg.max_iterations}")
    console.print(f"[white]Search results per question:[/] {cfg.max_results}")
    console.print(f"[white]Delay between fetches:[/] {cfg.search_delay_seconds:.1f}s")
    start = time.time()

    graph = get_graph()
    output = graph.invoke(initial_state(query))

    elapsed = time.time() - start
    report = output.get("final_report", "")
    if not report:
        console.print(Panel("[red]No report was generated.", title="Error"))
        raise SystemExit(1)

    console.print(Panel(f"[green]Research complete in {elapsed:.1f} seconds.[/]", title="Done"))
    console.print(Markdown(report))


@click.group()
def cli():
    """Mariana — free, local deep research agent."""
    pass


@cli.command()
@click.argument("query")
@click.option("--model", default=None, help="Override Ollama model")
@click.option("--iterations", default=None, type=int, help="Max research iterations")
def research(query, model, iterations):
    "Run deep research on a question and save a Markdown report."
    cfg, _ = _preflight(model=model, iterations=iterations)
    _run_query(query, cfg)


@cli.command()
def interactive():
    "Enter queries in a loop. Ctrl+C to quit."
    cfg, _ = _preflight()
    try:
        while True:
            query = console.input("[bold green]Enter research query:[/] ")
            if not query.strip():
                continue
            _run_query(query, cfg)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Goodbye.[/]")


@cli.command()
def config():
    "Show current configuration."
    cfg = load_config()
    table = Table(title="Mariana Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Key", style="cyan")
    table.add_column("Value", overflow="fold")
    for key, value in sorted(cfg.model_dump().items()):
        table.add_row(str(key), str(value))
    console.print(table)


@cli.command()
@click.argument("model")
def set_model(model):
    "Set the Ollama model to use."
    save_setting("MARIANA_PLANNER_MODEL", model)
    save_setting("MARIANA_SUMMARIZER_MODEL", model)
    console.print(Panel(f"[green]Model updated to[/] {model}", title="Config"))


@cli.command()
def reports():
    "List saved research reports."
    cfg = load_config()
    output_dir = Path(cfg.output_dir)
    if not output_dir.exists():
        console.print(Panel("[yellow]No reports found yet.", title="Reports"))
        return

    files = sorted(output_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        console.print(Panel("[yellow]No reports found yet.", title="Reports"))
        return

    table = Table(title="Saved Reports", show_header=True, header_style="bold magenta")
    table.add_column("Filename")
    table.add_column("Modified", style="green")
    for path in files:
        table.add_row(path.name, str(path.stat().st_mtime))
    console.print(table)
