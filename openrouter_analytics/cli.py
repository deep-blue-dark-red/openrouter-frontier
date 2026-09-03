import json
import click
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .client import OpenRouterAnalytics, get_model_stats
from .resolver import search_models, resolve_model

console = Console()


def _color_hit_rate(rate_pct: float) -> Text:
    text = f"{rate_pct:.1f}%"
    if rate_pct >= 80.0:
        return Text(text, style="bold green")
    elif rate_pct >= 50.0:
        return Text(text, style="green")
    elif rate_pct >= 30.0:
        return Text(text, style="yellow")
    else:
        return Text(text, style="red")


def _color_latency(lat_ms: Optional[float], text: str) -> Text:
    if lat_ms is None:
        return Text(text, style="dim")
    if lat_ms < 1000:
        return Text(text, style="bold green")
    elif lat_ms < 2500:
        return Text(text, style="green")
    elif lat_ms < 5000:
        return Text(text, style="yellow")
    else:
        return Text(text, style="red")


def _color_tps(tps: Optional[float], text: str) -> Text:
    if tps is None:
        return Text(text, style="dim")
    if tps >= 60:
        return Text(text, style="bold green")
    elif tps >= 30:
        return Text(text, style="green")
    elif tps >= 15:
        return Text(text, style="yellow")
    else:
        return Text(text, style="red")


def _color_uptime(upt_pct: Optional[float], text: str) -> Text:
    if upt_pct is None:
        return Text(text, style="dim")
    if upt_pct >= 99.0:
        return Text(text, style="bold green")
    elif upt_pct >= 95.0:
        return Text(text, style="green")
    elif upt_pct >= 85.0:
        return Text(text, style="yellow")
    else:
        return Text(text, style="red")


@click.group()
@click.version_option(version="0.1.0")
def main():
    """OpenRouter Analytics: Inspect 24h provider stats, cache hit rates, latency, TPS, and uptime."""
    pass


@main.command(name="stats")
@click.argument("model", required=True)
@click.option("--provider", "-p", default=None, help="Filter to a specific provider (name or slug).")
@click.option(
    "--sort",
    "-s",
    default="cache",
    type=click.Choice(["cache", "latency", "tps", "uptime", "input", "output", "tokens", "share", "name"]),
    help="Sort column (default: cache)."
)
@click.option("--top", "-n", default=None, type=int, help="Limit output to top N providers.")
@click.option("--json-output", "--json", is_flag=True, help="Output raw JSON.")
def stats_command(model: str, provider: Optional[str], sort: str, top: Optional[int], json_output: bool):
    """View 24h provider performance (cache hit, latency, TPS, uptime, pricing) for a model."""
    try:
        stats = get_model_stats(model)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()

    if provider:
        p = stats.get_provider(provider)
        if not p:
            console.print(f"[bold red]Provider '{provider}' not found for model '{stats.model_id}'.[/bold red]")
            console.print(f"Available providers: {', '.join(p.name for p in stats.providers)}")
            raise click.Abort()
        providers = [p]
    else:
        providers = stats.sort_by(sort)
        if top:
            providers = providers[:top]

    if json_output:
        out = stats.to_dict()
        if provider:
            out["providers"] = [p.to_dict() for p in providers]
        click.echo(json.dumps(out, indent=2))
        return

    # Header Panel
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{stats.model_name}[/bold cyan] ([dim]{stats.model_id}[/dim])\n"
            f"Canonical Permaslug: [bold yellow]{stats.permaslug}[/bold yellow]\n"
            f"Providers: [bold]{stats.provider_count}[/bold] | "
            f"Total Tokens (Today): [bold]{stats.formatted_total_tokens}[/bold] | "
            f"Weighted Cache Hit: [bold green]{stats.formatted_weighted_cache_hit_rate}[/bold green]",
            title="OpenRouter 24h Model & Provider Analytics",
            border_style="cyan"
        )
    )

    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("#", style="dim", justify="right", width=3)
    table.add_column("Provider", style="bold white", no_wrap=True)
    table.add_column("Cache", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("TPS", justify="right")
    table.add_column("Uptime", justify="right")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Share", justify="right")

    for idx, p in enumerate(providers, 1):
        table.add_row(
            str(idx),
            p.name,
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
    console.print("[dim]Data represents observed effective usage today (UTC) and live endpoint metrics across OpenRouter.[/dim]\n")


@main.command(name="cache")
@click.argument("model", required=True)
@click.argument("provider", required=False)
@click.option("--json-output", "--json", is_flag=True, help="Output raw JSON.")
def cache_command(model: str, provider: Optional[str], json_output: bool):
    """Quick lookup of cache hit rate, latency, TPS, and uptime for a provider."""
    try:
        stats = get_model_stats(model)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()

    if provider:
        p = stats.get_provider(provider)
        if not p:
            console.print(f"[bold red]Provider '{provider}' not found for '{stats.model_id}'.[/bold red]")
            raise click.Abort()

        if json_output:
            click.echo(json.dumps(p.to_dict(), indent=2))
            return

        console.print()
        console.print(f"[bold cyan]Model:[/bold cyan] {stats.model_name} ({stats.model_id})")
        console.print(f"[bold cyan]Provider:[/bold cyan] {p.name} ({p.slug})")
        console.print(f"[bold cyan]24h Cache Hit Rate:[/bold cyan] {_color_hit_rate(p.cache_hit_rate_pct)}")
        console.print(f"[bold cyan]Latency (p50):[/bold cyan] {_color_latency(p.latency_p50_ms, p.formatted_latency)}")
        console.print(f"[bold cyan]Throughput (p50):[/bold cyan] {_color_tps(p.throughput_p50_tps, p.formatted_tps)}")
        console.print(f"[bold cyan]Uptime (24h):[/bold cyan] {_color_uptime(p.uptime_1d_pct, p.formatted_uptime)}")
        console.print(f"[bold cyan]Effective Input Price:[/bold cyan] {p.formatted_input_price}")
        console.print(f"[bold cyan]Effective Output Price:[/bold cyan] {p.formatted_output_price}")
        console.print(f"[bold cyan]Tokens Served:[/bold cyan] {p.formatted_tokens} ({p.formatted_token_share} share)")
        console.print()
    else:
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


@main.command(name="compare")
@click.argument("model", required=True)
@click.argument("providers", nargs=-1, required=True)
def compare_command(model: str, providers: List[str]):
    """Compare multiple providers side by side (cache, latency, TPS, uptime, price)."""
    try:
        stats = get_model_stats(model)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()

    selected = []
    for prov_name in providers:
        p = stats.get_provider(prov_name)
        if p:
            selected.append(p)
        else:
            console.print(f"[yellow]Warning: Provider '{prov_name}' not found.[/yellow]")

    if not selected:
        console.print("[red]No matching providers found to compare.[/red]")
        raise click.Abort()

    table = Table(title=f"Provider Comparison for {stats.model_name}", header_style="bold magenta")
    table.add_column("Provider", style="bold white", no_wrap=True)
    table.add_column("Cache", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("TPS", justify="right")
    table.add_column("Uptime", justify="right")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Tokens (24h)", justify="right")
    table.add_column("Share", justify="right")

    for p in selected:
        table.add_row(
            p.name,
            _color_hit_rate(p.cache_hit_rate_pct),
            _color_latency(p.latency_p50_ms, p.formatted_latency),
            _color_tps(p.throughput_p50_tps, p.formatted_tps),
            _color_uptime(p.uptime_1d_pct, p.formatted_uptime),
            p.formatted_input_price,
            p.formatted_output_price,
            p.formatted_tokens,
            p.formatted_token_share,
        )

    console.print(table)


@main.command(name="search")
@click.argument("query", required=True)
def search_command(query: str):
    """Search for models on OpenRouter."""
    results = search_models(query)
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
