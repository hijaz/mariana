import atexit
import os
import re
import resource
import time
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from mariana.state import ResearchGoal, initial_state
from mariana.utils.config import load_config, save_setting
from mariana.utils.llm import check_ollama, get_llm, get_planner_llm, get_summarizer_llm
from mariana.utils.searxng import SearXNGManager
from mariana.utils.tracing import write_trace

# Ask the Linux OOM killer to spare our process — prefer killing other things first.
# Writing a negative value requires CAP_SYS_RESOURCE; we attempt it and log the result.
_oom_adj_path = Path(f"/proc/{os.getpid()}/oom_score_adj")
try:
    _oom_adj_path.write_text("-200")
except OSError:
    # Unprivileged users can only raise (not lower) oom_score_adj.
    # Write 0 explicitly to ensure default, then note it in startup.
    pass


def _rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

NODE_LABELS = {
    "init_document":    "Initialising document",
    "plan_toc":         "Planning table of contents",
    "select_section":   "Selecting next section",
    "generate_queries": "Generating search queries",
    "process_section":  "Researching section",
    "check_completion": "Checking completion",
    "finalize":         "Writing final report",
}

console = Console()


def _status_label(success: bool, text: str) -> str:
    icon = "[green]✔[/]" if success else "[red]✖[/]"
    return f"{icon} {text}"


def _apply_overrides(cfg, model, iterations):
    if model:
        cfg.planner_model = model
        cfg.summarizer_model = model
        get_llm.cache_clear()
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


def _print_session_summary(state: dict, elapsed: float, report_path: Path | None = None) -> None:
    from mariana.utils.store import get_toc
    doc_id = state.get("doc_id", "")
    total_sources = state.get("total_sources_scraped", 0)
    total_words = state.get("total_words_written", 0)
    llm_calls = state.get("llm_call_count", 0)

    sections: list[dict] = []
    if doc_id:
        try:
            toc = get_toc(doc_id)
            sections = [n for n in toc if n.get("status") == "complete"]
        except Exception:
            pass

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Sections completed", str(len(sections)))
    table.add_row("Sources scraped", str(total_sources))
    table.add_row("Words written", str(total_words))
    table.add_row("LLM calls", str(llm_calls))
    table.add_row("Total time", f"{elapsed:.0f}s")
    if state.get("stop_reason"):
        table.add_row("Stopped because", state["stop_reason"])
    if report_path:
        table.add_row("Report", str(report_path))
    console.print(Panel(table, title="Session Summary", border_style="dim"))


def _step(label: str, fn, state: dict, start: float) -> dict:
    """Call a node function, log entry/exit with timing and RSS, merge result into state."""
    t0 = time.time()
    console.print(f"  [dim]{t0 - start:5.1f}s  {_rss_mb():4d}MB[/]  [bold cyan]{label}[/]  …")
    try:
        result = fn(state)
    except Exception as exc:
        console.print(f"  [red]ERROR in {label}:[/] {exc}  (RSS {_rss_mb()} MB)")
        raise
    state = {**state, **result}
    t1 = time.time()
    status = result.get("status", "")
    console.print(f"  [dim]{t1 - start:5.1f}s  {_rss_mb():4d}MB[/]  [bold cyan]{label}[/]  [green]done[/]  {status}")
    return state


def _run_pipeline(state: dict, start: float) -> dict:
    """
    Execute the research pipeline as a plain Python loop.
    Identical logic to the LangGraph topology but with zero framework overhead.
    """
    from mariana.nodes import (
        check_completion_node,
        finalize_node,
        generate_queries_node,
        init_document_node,
        plan_toc_node,
        process_section_node,
        select_section_node,
    )

    state = _step(NODE_LABELS["init_document"],    init_document_node,    state, start)
    state = _step(NODE_LABELS["plan_toc"],         plan_toc_node,         state, start)

    while True:
        state = _step(NODE_LABELS["select_section"],   select_section_node,   state, start)
        if state.get("should_stop"):
            break
        state = _step(NODE_LABELS["generate_queries"], generate_queries_node, state, start)
        state = _step(NODE_LABELS["process_section"],  process_section_node,  state, start)
        state = _step(NODE_LABELS["check_completion"], check_completion_node, state, start)
        if state.get("should_stop"):
            break

    state = _step(NODE_LABELS["finalize"], finalize_node, state, start)
    return state


def _run_query(query: str, cfg, goal: ResearchGoal | None = None):
    console.print(Panel(f"[blue]Research started:[/] {query}", title="Mariana"))
    g = goal or ResearchGoal.from_cli()
    console.print(f"[white]Iterations:[/] {g.max_iterations}  "
                  f"[white]Results/question:[/] {cfg.max_results}  "
                  f"[white]Delay:[/] {cfg.search_delay_seconds:.1f}s")
    if g.required_topics:
        console.print(f"[white]Required topics:[/] {', '.join(g.required_topics)}")
    if g.max_runtime:
        console.print(f"[white]Max runtime:[/] {g.max_runtime}")
    if g.target_words:
        console.print(f"[white]Target words:[/] {g.target_words}")
    if g.max_sources:
        console.print(f"[white]Max sources:[/] {g.max_sources}")
    start = time.time()

    final_state = _run_pipeline(initial_state(query, goal=g), start)

    elapsed = time.time() - start
    report = final_state.get("final_report", "")
    if not report:
        console.print(Panel("[red]No report was generated.", title="Error"))
        raise SystemExit(1)

    report_path: Path | None = None
    if report and Path(report).exists():
        report_path = Path(report)

    # Write trace file
    try:
        trace_path = write_trace(final_state, elapsed, report_path)
        console.print(f"[dim]Trace → {trace_path}[/]")
    except Exception:
        pass

    _print_session_summary(final_state, elapsed, report_path)
    console.print(Markdown(final_state.get("final_report", "")))



@click.group()
def cli():
    """Mariana — free, local deep research agent."""
    pass


@cli.command()
@click.argument("query")
@click.option("--model", default=None, help="Override Ollama model")
@click.option("--iterations", default=None, type=int, help="Max research iterations")
@click.option("--for", "duration", default=None,
              help="Run until time limit, e.g. 30m, 2h, 1h30m")
@click.option("--sources", default=None, type=int,
              help="Stop after scraping this many sources")
@click.option("--words", default=None, type=int,
              help="Stop after writing this many words")
@click.option("--require", multiple=True,
              help="Required topic(s) to cover before stopping (can repeat)")
def research(query, model, iterations, duration, sources, words, require):
    "Run deep research on a question and save a Markdown report."
    cfg, _ = _preflight(model=model, iterations=iterations)
    goal = ResearchGoal.from_cli(
        duration=duration,
        max_sources=sources,
        target_words=words,
        max_iterations=iterations,
        required_topics=require,
    )
    _run_query(query, cfg, goal=goal)


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

