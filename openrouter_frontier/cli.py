"""Rich-formatted ``openrouter-frontier`` command-line interface."""

import json
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from ._util import filter_primary_quantization
from .client import get_model_stats, score_model_providers
from .render import fmt_pct, fmt_seconds, fmt_tps
from .resolver import search_models
from .profile_args import describe_profile
from .scoring import ScoringConfig

console = Console()


def _graded(value: Optional[float], text: str, thresholds, higher_is_better: bool = True) -> Text:
    """Colour ``text`` green/yellow/red by where ``value`` falls among three thresholds."""
    if value is None:
        return Text(text, style="dim")
    great, good, ok = thresholds
    if higher_is_better:
        style = "bold green" if value >= great else "green" if value >= good else "yellow" if value >= ok else "red"
    else:
        style = "bold green" if value < great else "green" if value < good else "yellow" if value < ok else "red"
    return Text(text, style=style)


def _color_hit_rate(rate_pct: float) -> Text:
    return _graded(rate_pct, f"{rate_pct:.1f}%", (80.0, 50.0, 30.0))


def _color_latency(lat_ms: Optional[float], text: str) -> Text:
    return _graded(lat_ms, text, (1000, 2500, 5000), higher_is_better=False)


def _color_tps(tps: Optional[float], text: str) -> Text:
    return _graded(tps, text, (60, 30, 15))


def _color_uptime(upt_pct: Optional[float], text: str) -> Text:
    return _graded(upt_pct, text, (99.0, 95.0, 85.0))


def _fail(msg: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {msg}")
    raise click.Abort()


@click.group()
@click.version_option(version=__version__)
def main():
    """OpenRouter Frontier: 24h provider stats, cache hit rates, pricing, and utility scores."""


@main.command(name="stats")
@click.argument("model", required=True)
@click.option("--provider", "-p", default=None, help="Filter to a specific provider (name or slug).")
@click.option(
    "--sort", "-s", default="cache",
    type=click.Choice(["cache", "score", "token_cost", "latency", "tps", "uptime", "input", "output", "tokens", "share", "name"]),
    help="Sort column.",
)
@click.option("--top", "-n", default=None, type=int, help="Limit output to top N providers.")
@click.option("--json-output", "--json", is_flag=True, help="Output raw JSON.")
def stats_command(model: str, provider: Optional[str], sort: str, top: Optional[int], json_output: bool):
    """View 24h provider performance (cache hit, latency, TPS, uptime, pricing) for a model."""
    try:
        stats = get_model_stats(model)
    except Exception as e:
        _fail(str(e))

    if provider:
        p = stats.get_provider(provider)
        if not p:
            console.print(f"Available providers: {', '.join(x.name for x in stats.providers)}")
            _fail(f"Provider '{provider}' not found for model '{stats.model_id}'.")
        providers = [p]
    else:
        providers = stats.sort_by(sort)
        if top:
            providers = providers[:top]

    if json_output:
        out = stats.to_dict()
        out["providers"] = [p.to_dict() for p in providers]
        click.echo(json.dumps(out, indent=2))
        return

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{stats.model_name}[/bold cyan] ([dim]{stats.model_id}[/dim])\n"
            f"Canonical Permaslug: [bold yellow]{stats.permaslug}[/bold yellow]\n"
            f"Providers: [bold]{stats.provider_count}[/bold] | "
            f"Total Tokens (24h): [bold]{stats.formatted_total_tokens}[/bold] | "
            f"Weighted Cache Hit: [bold green]{stats.formatted_weighted_cache_hit_rate}[/bold green]",
            title="OpenRouter 24h Model & Provider Analytics",
            border_style="cyan",
        )
    )

    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("#", style="dim", justify="right", width=3)
    table.add_column("Provider", style="bold white", no_wrap=True)
    table.add_column("Quant", style="dim")
    for col in ("Cache", "Latency", "TPS", "Uptime", "Input", "Output", "Tokens", "Share"):
        table.add_column(col, justify="right")

    for idx, p in enumerate(providers, 1):
        table.add_row(
            str(idx),
            p.name,
            p.quantization,
            _color_hit_rate(p.cache_hit_rate_pct),
            _color_latency(p.latency_p50_ms, p.formatted_latency),
            _color_tps(p.throughput_p50_tps, p.formatted_tps),
            _color_uptime(p.uptime_1d_pct, p.formatted_uptime),
            p.formatted_input_price,
            p.formatted_output_price,
            p.formatted_tokens,
            p.formatted_token_share,
        )

    if not provider and len(providers) > 1:
        table.add_section()
        table.add_row(
            "—",
            "[bold italic]Summary / Avg[/bold italic]",
            "",
            _color_hit_rate(stats.weighted_cache_hit_rate * 100.0),
            _color_latency(stats.avg_latency_p50_ms, stats.formatted_avg_latency),
            _color_tps(stats.avg_throughput_p50_tps, stats.formatted_avg_tps),
            _color_uptime(stats.avg_uptime_1d_pct, stats.formatted_avg_uptime),
            stats.formatted_weighted_input_price,
            stats.formatted_weighted_output_price,
            stats.formatted_total_tokens,
            "100.0%",
        )

    console.print(table)
    console.print("[dim]Observed effective usage over the trailing 24h (UTC) plus live endpoint metrics.[/dim]\n")


def _task_options(f):
    """Task-profile options shared by ``score``, ``cache`` and ``compare``."""
    opts = [
        click.option("--new-tokens", "-a", "--prompt-tokens", "-c", "new_tokens", default=2000, type=int, show_default=True, help="New prompt tokens per turn (a)."),
        click.option("--task-tokens", default=300_000, type=int, show_default=True, help="Transcript size the task grows to."),
        click.option("--output-tokens", default=10_000, type=int, show_default=True, help="Total completion tokens over the task."),
        click.option("--completion-tokens", "-o", default=None, type=int, help="Completion tokens per turn (overrides --output-tokens)."),
        click.option("--turns", "-N", default=None, type=int, help="Explicit number of turns."),
        click.option("--time-value", "-t", default=5.0, type=float, show_default=True, help="Value of your time in USD/hr; 0 disables."),
        click.option("--prefill-multiplier", default=100.0, type=float, show_default=True, help="Prompt-processing speed as a multiple of decode throughput."),
        click.option("--overhead-seconds", default=0.0, type=float, help="Fixed per-request wait charged on top of prefill and decode."),
        click.option("--price-failures/--no-failures", default=True, help="Price failures from 24h uptime."),
        click.option("--routing", type=click.Choice(["sticky", "order"]), default="sticky", show_default=True, help="OpenRouter routing policy assumed."),
        click.option("--miss-policy", type=click.Choice(["rewrite", "process"]), default="rewrite", show_default=True, help="Cache-miss billing policy."),
        click.option("--cache", "cache_mode", type=click.Choice(["aggregate", "cold", "assumed"]), default="aggregate", show_default=True, help="Hit-rate source."),
        click.option("--assumed-hit-rate", default=0.0, type=float, help="Hit rate for --cache assumed."),
        click.option("--sigma-h", default=0.0, type=float, help="Epistemic std. dev. of the hit rate."),
        click.option("--lambda-proc", default=0.0, type=float, help="Risk aversion to process variance."),
        click.option("--lambda-par", default=0.0, type=float, help="Risk aversion to parameter variance."),
        click.option("--discount/--no-discount", default=True, help="Use net (discounted) prices."),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


def _config(kw: Dict[str, Any]) -> ScoringConfig:
    return ScoringConfig(
        new_tokens_per_turn=kw["new_tokens"], task_tokens=kw["task_tokens"], output_tokens=kw["output_tokens"],
        completion_tokens=kw["completion_tokens"], turns=kw["turns"],
        time_value_usd_per_hour=kw["time_value"], prefill_multiplier=kw["prefill_multiplier"], overhead_seconds=kw["overhead_seconds"], price_failures=kw["price_failures"],
        routing=kw["routing"], miss_policy=kw["miss_policy"], cache_mode=kw["cache_mode"],
        assumed_hit_rate=kw["assumed_hit_rate"], sigma_h=kw["sigma_h"],
        lambda_proc=kw["lambda_proc"], lambda_par=kw["lambda_par"], apply_discount=kw["discount"],
    )


def _usd(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 100 else (f"${x:,.2f}" if abs(x) >= 1 else f"${x:.4f}")


@main.command(name="score")
@click.argument("model", required=True)
@_task_options
@click.option("--all-quants", is_flag=True, help="Include every quantization variant, not just the primary (fp8).")
@click.option("--provider", "-p", default=None, help="Filter to a specific provider.")
@click.option("--top", "-n", default=None, type=int, help="Limit to top N providers.")
@click.option("--json-output", "--json", is_flag=True, help="Output raw JSON.")
def score_command(model: str, all_quants: bool, provider: Optional[str], top: Optional[int], json_output: bool, **kw):
    """Rank providers by the expected cost of a whole task (tokens + time + failures)."""
    cfg = _config(kw)

    try:
        scores = score_model_providers(model, config=cfg)
    except Exception as e:
        _fail(str(e))

    scores = filter_primary_quantization(scores, all_quants)

    if provider:
        q = provider.lower()
        scores = [s for s in scores if q in s.provider_name.lower() or q in s.provider_slug.lower()]
        if not scores:
            _fail(f"Provider '{provider}' not found for model.")
    elif top:
        scores = scores[:top]

    if json_output:
        click.echo(json.dumps([s.to_dict() for s in scores], indent=2))
        return

    show_obj = cfg.lambda_proc > 0 or cfg.lambda_par > 0
    console.print()
    console.print(
        Panel.fit(
            f"Model: [bold cyan]{model}[/bold cyan] | {describe_profile(cfg)}\n"
            f"Quants: [bold]{'All' if all_quants else 'Primary'}[/bold]",
            title="ProviderScore Task Cost",
            border_style="magenta",
        )
    )

    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Provider", style="bold white", no_wrap=True)
    table.add_column("Task $", style="bold green", justify="right", no_wrap=True)
    if show_obj:
        table.add_column("Objective $", justify="right")
    for col in ("Token $", "Fail $", "Time $", "Cache Hit", "TTFT", "TPS", "Turn Time", "Task Time", "Uptime", "$/M"):
        table.add_column(col, justify="right")

    for s in scores:
        row: List = [s.provider_name + ("*" if s.imputed else ""), s.formatted_task_cost]
        if show_obj:
            row.append(s.formatted_objective)
        row += [
            s.formatted_token_cost,
            s.formatted_failure_cost,
            s.formatted_time_cost,
            _color_hit_rate(s.cache_hit_rate * 100.0),
            fmt_seconds(s.ttft_seconds),
            fmt_tps(s.throughput_tps, " tps"),
            s.formatted_turn_time,
            s.formatted_task_time,
            fmt_pct(s.uptime_pct),
            f"${s.task_cost_per_m:.4f}",
        ]
        table.add_row(*row)

    console.print(table)
    console.print(f"[dim]Task $ = Token $ + Fail $ + Time $. Time $ = Task Time at ${cfg.time_value_usd_per_hour:.0f}/hr; no other column contains time. "
                  "Task Time = prefill (new tokens, and the prefix again after a cache miss) + decoding + overhead. TTFT is shown, not charged. "
                  "$/M = Task $ per 1M submitted tokens. * = missing telemetry imputed worst-case.[/dim]\n")


@main.command(name="cache")
@click.argument("model", required=True)
@click.argument("provider", required=False)
@_task_options
@click.option("--json-output", "--json", is_flag=True, help="Output raw JSON.")
def cache_command(model: str, provider: Optional[str], json_output: bool, **kw):
    """Quick lookup of cache hit rate, latency, TPS, and utility score for a provider."""
    try:
        stats = get_model_stats(model)
    except Exception as e:
        _fail(str(e))

    cfg = _config(kw)

    if not provider:
        providers = stats.sort_by("cache")
        if json_output:
            click.echo(json.dumps([p.to_dict() for p in providers], indent=2))
            return
        console.print()
        console.print(f"[bold cyan]24h Providers for {stats.model_name}:[/bold cyan]")
        for idx, p in enumerate(providers, 1):
            console.print(
                f"  {idx:2d}. [bold white]{p.name:18}[/bold white] "
                f"Cache: {_color_hit_rate(p.cache_hit_rate_pct)} "
                f"| Latency: {_color_latency(p.latency_p50_ms, p.formatted_latency)} "
                f"| TPS: {_color_tps(p.throughput_p50_tps, p.formatted_tps)} "
                f"| Up: {_color_uptime(p.uptime_1d_pct, p.formatted_uptime)} "
                f"| Input: {p.formatted_input_price}"
            )
        console.print()
        return

    p = stats.get_provider(provider)
    if not p:
        _fail(f"Provider '{provider}' not found for '{stats.model_id}'.")
    score = next((sb for sb in stats.score_providers(cfg) if sb.endpoint_id == p.endpoint_id), None)

    if json_output:
        out = p.to_dict()
        if score:
            out["score_evaluation"] = score.to_dict()
        click.echo(json.dumps(out, indent=2))
        return

    console.print()
    console.print(f"[bold cyan]Model:[/bold cyan] {stats.model_name} ({stats.model_id})")
    console.print(f"[bold cyan]Provider:[/bold cyan] {p.name} ({p.slug}, {p.quantization})")
    console.print(f"[bold cyan]24h Published Cache Hit:[/bold cyan] {_color_hit_rate(p.cache_hit_rate_pct)}")
    if score:
        console.print(
            f"[bold cyan]Expected Task Cost:[/bold cyan] [bold green]{score.formatted_task_cost}[/bold green] "
            f"({score.turns} turns × {score.new_tokens}+{score.completion_tokens} tok, {score.routing} routing)  "
            f"[bold cyan]Expected Task Time:[/bold cyan] {score.formatted_task_time} ({score.formatted_turn_time}/turn)"
        )
        console.print(
            f"  tokens {score.formatted_token_cost} | time {score.formatted_time_cost} (gen {score.formatted_decode_cost}, prefill new tokens {_usd(score.prefill_new_cost_usd)}, re-prefill on cache miss {_usd(score.prefill_miss_cost_usd)}, overhead {score.formatted_ttft_cost}) | failures {score.formatted_failure_cost} | "
            f"miss premium {score.formatted_miss_premium} | perfect cache {_usd(score.perfect_cache_cost_usd)} | cold {_usd(score.cold_cache_cost_usd)}"
        )
        console.print(f"[bold cyan]Prices /M:[/bold cyan] input ${score.input_price:.4f} | read ${score.read_price:.4f} | write ${score.write_price:.4f} | miss ${score.miss_price:.4f}")
    console.print(f"[bold cyan]Latency (p50):[/bold cyan] {_color_latency(p.latency_p50_ms, p.formatted_latency)}")
    console.print(f"[bold cyan]Throughput (p50):[/bold cyan] {_color_tps(p.throughput_p50_tps, p.formatted_tps)}")
    console.print(f"[bold cyan]Uptime (24h):[/bold cyan] {_color_uptime(p.uptime_1d_pct, p.formatted_uptime)}")
    console.print(f"[bold cyan]Tokens Served (24h):[/bold cyan] {p.formatted_tokens} ({p.formatted_token_share} share)")
    console.print()


@main.command(name="compare")
@click.argument("model", required=True)
@click.argument("providers", nargs=-1, required=True)
@_task_options
def compare_command(model: str, providers: List[str], **kw):
    """Compare named providers side by side (cache, latency, TPS, uptime, cost score)."""
    try:
        stats = get_model_stats(model)
    except Exception as e:
        _fail(str(e))

    cfg = _config(kw)
    scored = {sb.endpoint_id: sb for sb in stats.score_providers(cfg)}
    selected = []
    for name in providers:
        p = stats.get_provider(name)
        if p:
            selected.append(p)
        else:
            console.print(f"[yellow]Warning: provider '{name}' not found.[/yellow]")
    if not selected:
        _fail("No matching providers found to compare.")

    table = Table(
        title=f"Provider Comparison for {stats.model_name} ({cfg.n_turns} turns × {cfg.new_tokens_per_turn}+{cfg.completion_per_turn} tok)",
        header_style="bold magenta",
    )
    table.add_column("Provider", style="bold white", no_wrap=True)
    table.add_column("Task $", justify="right", style="bold green")
    for col in ("CacheHit", "Latency", "TPS", "Uptime", "Tokens (24h)", "Share"):
        table.add_column(col, justify="right")

    for p in selected:
        sb = scored.get(p.endpoint_id)
        table.add_row(
            p.name,
            sb.formatted_task_cost if sb else "--",
            _color_hit_rate(p.cache_hit_rate_pct),
            _color_latency(p.latency_p50_ms, p.formatted_latency),
            _color_tps(p.throughput_p50_tps, p.formatted_tps),
            _color_uptime(p.uptime_1d_pct, p.formatted_uptime),
            p.formatted_tokens,
            p.formatted_token_share,
        )
    console.print(table)


@main.command(name="search")
@click.argument("query", required=True)
@click.option("--limit", "-n", default=10, type=int, show_default=True, help="Maximum results.")
def search_command(query: str, limit: int):
    """Search the OpenRouter model catalog."""
    results = search_models(query, limit=limit)
    if not results:
        console.print(f"[yellow]No models found matching '{query}'.[/yellow]")
        return

    table = Table(title=f"Models matching '{query}'", header_style="bold magenta")
    table.add_column("ID", style="bold cyan")
    table.add_column("Name", style="white")
    table.add_column("Canonical Permaslug", style="dim")
    table.add_column("Context", justify="right")
    for m in results:
        table.add_row(m["id"], m["name"], m["canonical_slug"], m["context_length"])
    console.print(table)


if __name__ == "__main__":
    main()
