"""Rich-formatted ``openrouter-frontier`` command-line interface."""

import json
from typing import List, Optional

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


@main.command(name="score")
@click.argument("model", required=True)
@click.option("--prompt-tokens", "-c", default=2000, type=int, show_default=True, help="Prompt tokens per turn (C).")
@click.option("--completion-tokens", "-o", default=500, type=int, show_default=True, help="Completion tokens per turn (O).")
@click.option("--time-value", "-t", default=0.0, type=float, show_default=True, help="Time value in USD/hr; 0 = pure token cost.")
@click.option("--price-failures/--no-failures", default=True, help="Include failure risk cost from endpoint uptime.")
@click.option("--discount/--no-discount", default=True, help="Apply advertised endpoint discounts.")
@click.option("--all-quants", is_flag=True, help="Include every quantization variant, not just the primary (fp8).")
@click.option("--provider", "-p", default=None, help="Filter to a specific provider.")
@click.option("--top", "-n", default=None, type=int, help="Limit to top N providers.")
@click.option("--json-output", "--json", is_flag=True, help="Output raw JSON.")
def score_command(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    time_value: float,
    price_failures: bool,
    discount: bool,
    all_quants: bool,
    provider: Optional[str],
    top: Optional[int],
    json_output: bool,
):
    """Rank providers by ProviderScore expected cost per turn (token + time + failure risk)."""
    cfg = ScoringConfig(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        time_value_usd_per_hour=time_value,
        price_failures=price_failures,
        apply_discount=discount,
    )

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

    mode = "Pure Token Cost" if time_value == 0 and not price_failures else "Full Utility Model"
    console.print()
    console.print(
        Panel.fit(
            f"Model: [bold cyan]{model}[/bold cyan] | Turn: [bold]{prompt_tokens}[/bold] prompt + [bold]{completion_tokens}[/bold] completion tokens\n"
            f"Mode: [bold yellow]{mode}[/bold yellow] | Time Value: [bold]${time_value:.2f}/hr[/bold] | Failure Risk: [bold]{'Yes' if price_failures else 'No'}[/bold]\n"
            f"Discounts: [bold]{'Applied' if discount else 'List Price'}[/bold] | Quants: [bold]{'All' if all_quants else 'Primary'}[/bold]",
            title="ProviderScore Evaluation & Cost per 1M tok",
            border_style="magenta",
        )
    )

    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Provider", style="bold white", no_wrap=True)
    table.add_column("Scored $/M", style="bold green", justify="right", no_wrap=True)
    table.add_column("Token $/M", justify="right")
    if time_value > 0:
        table.add_column("Time $/M", justify="right")
    if price_failures:
        table.add_column("Fail $/M", justify="right")
    table.add_column("CacheHit", justify="right")
    for col in ("Hit $/M", "Miss $/M", "Latency", "TPS", "Uptime"):
        table.add_column(col, justify="right")

    for s in scores:
        row: List = [s.provider_name, s.formatted_total_cost, s.formatted_token_cost]
        if time_value > 0:
            row.append(s.formatted_time_cost)
        if price_failures:
            row.append(s.formatted_failure_cost)
        row += [
            _color_hit_rate(s.cache_hit_rate * 100.0),
            f"${s.hit_price:.4f}",
            f"${s.miss_price:.4f}",
            fmt_seconds(s.ttft_seconds),
            fmt_tps(s.throughput_tps, " tps"),
            fmt_pct(s.uptime_pct),
        ]
        table.add_row(*row)

    console.print(table)
    console.print("[dim]Lower Scored $/M is better. CacheHit is the published 24h token-weighted cache hit rate.[/dim]\n")


@main.command(name="cache")
@click.argument("model", required=True)
@click.argument("provider", required=False)
@click.option("--prompt-tokens", "-c", default=2000, type=int, help="Prompt tokens for score evaluation.")
@click.option("--completion-tokens", "-o", default=500, type=int, help="Completion tokens for score evaluation.")
@click.option("--json-output", "--json", is_flag=True, help="Output raw JSON.")
def cache_command(model: str, provider: Optional[str], prompt_tokens: int, completion_tokens: int, json_output: bool):
    """Quick lookup of cache hit rate, latency, TPS, and utility score for a provider."""
    try:
        stats = get_model_stats(model)
    except Exception as e:
        _fail(str(e))

    cfg = ScoringConfig(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

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
    score = p.evaluate_score(cfg)

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
            f"[bold cyan]Expected Token Cost (per 1M tok):[/bold cyan] [bold green]{score.formatted_token_cost}[/bold green] "
            f"({prompt_tokens} prompt + {completion_tokens} completion)"
        )
        console.print(f"[bold cyan]Cache Hit Price:[/bold cyan] ${score.hit_price:.4f} /M | [bold cyan]Miss Price:[/bold cyan] ${score.miss_price:.4f} /M")
    console.print(f"[bold cyan]Latency (p50):[/bold cyan] {_color_latency(p.latency_p50_ms, p.formatted_latency)}")
    console.print(f"[bold cyan]Throughput (p50):[/bold cyan] {_color_tps(p.throughput_p50_tps, p.formatted_tps)}")
    console.print(f"[bold cyan]Uptime (24h):[/bold cyan] {_color_uptime(p.uptime_1d_pct, p.formatted_uptime)}")
    console.print(f"[bold cyan]Tokens Served (24h):[/bold cyan] {p.formatted_tokens} ({p.formatted_token_share} share)")
    console.print()


@main.command(name="compare")
@click.argument("model", required=True)
@click.argument("providers", nargs=-1, required=True)
@click.option("--prompt-tokens", "-c", default=2000, type=int, help="Prompt tokens per turn.")
@click.option("--completion-tokens", "-o", default=500, type=int, help="Completion tokens per turn.")
def compare_command(model: str, providers: List[str], prompt_tokens: int, completion_tokens: int):
    """Compare named providers side by side (cache, latency, TPS, uptime, cost score)."""
    try:
        stats = get_model_stats(model)
    except Exception as e:
        _fail(str(e))

    cfg = ScoringConfig(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
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
        title=f"Provider Comparison for {stats.model_name} ({prompt_tokens} in / {completion_tokens} out)",
        header_style="bold magenta",
    )
    table.add_column("Provider", style="bold white", no_wrap=True)
    table.add_column("Scored $/M", justify="right", style="bold green")
    for col in ("CacheHit", "Latency", "TPS", "Uptime", "Tokens (24h)", "Share"):
        table.add_column(col, justify="right")

    for p in selected:
        sb = p.evaluate_score(cfg)
        table.add_row(
            p.name,
            sb.formatted_total_cost if sb else "--",
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
