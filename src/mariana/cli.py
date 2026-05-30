import atexit
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from mariana.graph import get_graph
from mariana.state import SubQuestion, initial_state
from mariana.utils.config import load_config, save_setting
from mariana.utils.llm import check_ollama, get_planner_llm, get_summarizer_llm
from mariana.utils.searxng import SearXNGManager
from mariana.utils.tracing import write_trace

NODE_LABELS = {
    "plan":      "Orchestrating outline",
    "search":    "Searching & scraping",
    "summarize": "Summarizing sources",
    "reflect":   "Reflecting on coverage",
    "report":    "Writing report",
}

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


def _query_thread_id(query: str) -> str:
    """Stable thread ID derived from the query for LangGraph checkpointing."""
    return hashlib.md5(query.strip().lower().encode()).hexdigest()[:16]


def _print_session_summary(state: dict, elapsed: float, report_path: Path | None = None) -> None:
    sub_questions = state.get("sub_questions", [])
    answered = [sq for sq in sub_questions if isinstance(sq, SubQuestion) and sq.answered]
    total_sources = sum(len(sq.results) for sq in sub_questions if isinstance(sq, SubQuestion))
    usable_sources = sum(
        len([r for r in sq.results if len(r.content or "") > 200])
        for sq in sub_questions if isinstance(sq, SubQuestion)
    )
    domains = {
        urlparse(r.url).netloc.replace("www.", "")
        for sq in sub_questions if isinstance(sq, SubQuestion)
        for r in sq.results
    }
    llm_calls = state.get("llm_call_count", 0)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Sections completed", f"{len(answered)} / {len(sub_questions)}")
    table.add_row("Sources scraped", f"{total_sources} ({usable_sources} usable)")
    table.add_row("Unique domains", str(len(domains)))
    table.add_row("LLM calls", str(llm_calls))
    table.add_row("Total time", f"{elapsed:.0f}s")
    if report_path:
        table.add_row("Report", str(report_path))
    console.print(Panel(table, title="Session Summary", border_style="dim"))


def _run_query(query: str, cfg, fresh: bool = False):
    console.print(Panel(f"[blue]Research started:[/] {query}", title="Mariana"))
    console.print(f"[white]Iterations:[/] {cfg.max_iterations}  "
                  f"[white]Results/question:[/] {cfg.max_results}  "
                  f"[white]Delay:[/] {cfg.search_delay_seconds:.1f}s")
    start = time.time()

    thread_id = _query_thread_id(query)
    run_config = {"configurable": {"thread_id": thread_id}}
    if fresh:
        console.print(f"[dim]--fresh: new thread {thread_id}[/]")
        run_config["configurable"]["thread_id"] = thread_id + "_" + str(int(time.time()))

    graph = get_graph()
    final_state: dict = {}

    try:
        for event in graph.stream(initial_state(query), config=run_config, stream_mode="updates"):
            for node_name, node_output in event.items():
                label = NODE_LABELS.get(node_name, node_name)
                node_status = node_output.get("status", "")
                elapsed_so_far = time.time() - start
                console.print(
                    f"  [dim]{elapsed_so_far:5.1f}s[/]  [bold cyan]{label}[/]  {node_status}"
                )
                final_state.update(node_output)
    except TypeError:
        # Checkpointer not available — fall back to invoke without config
        final_state = graph.invoke(initial_state(query))

    elapsed = time.time() - start
    report = final_state.get("final_report", "")
    if not report:
        console.print(Panel("[red]No report was generated.", title="Error"))
        raise SystemExit(1)

    # Extract report path from status message
    status_msg = final_state.get("status", "")
    report_path: Path | None = None
    path_match = re.search(r"Report saved to (.+)", status_msg)
    if path_match:
        report_path = Path(path_match.group(1).strip())

    # Write trace file
    try:
        trace_path = write_trace(final_state, elapsed, report_path)
        console.print(f"[dim]Trace → {trace_path}[/]")
    except Exception:
        pass

    _print_session_summary(final_state, elapsed, report_path)
    console.print(Markdown(report))


@click.group()
def cli():
    """Mariana — free, local deep research agent."""
    pass


@cli.command()
@click.argument("query")
@click.option("--model", default=None, help="Override Ollama model")
@click.option("--iterations", default=None, type=int, help="Max research iterations")
@click.option("--fresh", is_flag=True, default=False, help="Ignore saved checkpoint and start fresh")
def research(query, model, iterations, fresh):
    "Run deep research on a question and save a Markdown report."
    cfg, _ = _preflight(model=model, iterations=iterations)
    _run_query(query, cfg, fresh=fresh)


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


@cli.command()
def status():
    "Show Ollama and SearXNG status without starting a research run."
    cfg = load_config()
    table = Table(title="Mariana Status", show_header=False, box=None, padding=(0, 2))

    ollama_ok, result = check_ollama()
    table.add_row("Ollama", f"[green]OK[/] {cfg.ollama_base_url}" if ollama_ok else f"[red]FAIL[/] {result}")
    if isinstance(result, list):
        table.add_row("Models", ", ".join(result[:8]))
    table.add_row("Planner model", cfg.planner_model)
    table.add_row("Summarizer model", cfg.summarizer_model)

    from mariana.utils.searxng import SearXNGManager as _M
    mgr = _M()
    searxng_ok = mgr._is_responsive(cfg.searxng_port)
    table.add_row("SearXNG", f"[green]running[/] port {cfg.searxng_port}" if searxng_ok else "[yellow]not running[/]")

    output_dir = Path(cfg.output_dir)
    reports_count = len(list(output_dir.glob("*.md"))) if output_dir.exists() else 0
    traces_count = len(list(output_dir.glob("*_trace.json"))) if output_dir.exists() else 0
    table.add_row("Reports saved", str(reports_count))
    table.add_row("Trace files", str(traces_count))
    table.add_row("Output dir", str(output_dir))

    console.print(Panel(table, border_style="blue"))

