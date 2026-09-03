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


@click.group()
@click.version_option(version="0.1.0")
def main():
    """OpenRouter Analytics: Inspect 24h provider stats, cache hit rates, and effective prices."""
    pass


@main.command(name="stats")
@click.argument("model", required=True)
@click.option("--provider", "-p", default=None, help="Filter to a specific provider (name or slug).")
@click.option("--sort", "-s", default="cache", type=click.Choice(["cache", "input", "output", "tokens", "share", "name"]), help="Sort column.")
@click.option("--top", "-n", default=None, type=int, help="Limit output to top N providers.")
@click.option("--json-output", "--json", is_flag=True, help="Output raw JSON.")
def stats_command(model: str, provider: Optional[str], sort: str, top: Optional[int], json_output: bool):
    """View 24h provider usage, cache hit rate, and pricing for a model."""
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
            f"Weighted Cache Hit Rate: [bold green]{stats.formatted_weighted_cache_hit_rate}[/bold green]",
            title="OpenRouter 24h Model & Provider Analytics",
            border_style="cyan"
        )
    )

    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("#", style="dim", justify="right", width=3)
    table.add_column("Provider", style="bold white", no_wrap=True)
    table.add_column("Cache Hit", justify="right")
    table.add_column("Input ($/M)", justify="right")
    table.add_column("Output ($/M)", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Share", justify="right")

    for idx, p in enumerate(providers, 1):
        table.add_row(
            str(idx),
            p.name,
            _color_hit_rate(p.cache_hit_rate_pct),
            p.formatted_input_price,
            p.formatted_output_price,
            p.formatted_tokens,
            p.formatted_token_share,
        )

    if not provider and len(providers) > 1:
        table.add_section()
        table.add_row(
            "—",
            "[bold italic]Weighted Average[/bold italic]",
            _color_hit_rate(stats.weighted_cache_hit_rate * 100.0),
            stats.formatted_weighted_input_price,
            stats.formatted_weighted_output_price,
            stats.formatted_total_tokens,
            "100.0%",
        )

    console.print(table)
    console.print("[dim]Data represents observed effective usage today (UTC) across OpenRouter.[/dim]\n")


@main.command(name="cache")
@click.argument("model", required=True)
@click.argument("provider", required=False)
@click.option("--json-output", "--json", is_flag=True, help="Output raw JSON.")
def cache_command(model: str, provider: Optional[str], json_output: bool):
    """Quick lookup of cache hit rate for a model (and optional provider)."""
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
        console.print(f"[bold cyan]Effective Input Price:[/bold cyan] {p.formatted_input_price}")
        console.print(f"[bold cyan]Effective Output Price:[/bold cyan] {p.formatted_output_price}")
        console.print(f"[bold cyan]Tokens Served:[/bold cyan] {p.formatted_tokens} ({p.formatted_token_share} share)")
        console.print()
    else:
        # Show all ranked by cache hit rate
        providers = stats.sort_by("cache")
        if json_output:
            click.echo(json.dumps([p.to_dict() for p in providers], indent=2))
            return

        console.print()
        console.print(f"[bold cyan]24h Cache Hit Rates for {stats.model_name}:[/bold cyan]")
        for idx, p in enumerate(providers, 1):
            console.print(
                f"  {idx:2d}. [bold white]{p.name:20}[/bold white] "
                f"Hit Rate: {_color_hit_rate(p.cache_hit_rate_pct)} "
                f"| Input: {p.formatted_input_price} "
                f"| Volume: {p.formatted_tokens}"
            )
        console.print()


@main.command(name="compare")
@click.argument("model", required=True)
@click.argument("providers", nargs=-1, required=True)
def compare_command(model: str, providers: List[str]):
    """Compare multiple providers side by side for a given model."""
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
    table.add_column("Cache Hit", justify="right")
    table.add_column("Input ($/M)", justify="right")
    table.add_column("Output ($/M)", justify="right")
    table.add_column("Tokens (24h)", justify="right")
    table.add_column("Share", justify="right")

    for p in selected:
        table.add_row(
            p.name,
            _color_hit_rate(p.cache_hit_rate_pct),
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
