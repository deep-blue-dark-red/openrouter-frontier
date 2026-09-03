import json
import click
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .client import OpenRouterAnalytics, get_model_stats, score_model_providers
from .resolver import search_models, resolve_model
from .scoring import ScoringConfig, ScoreBreakdown

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
    """OpenRouter Analytics: Inspect 24h provider stats, cache hit rates, pricing, and utility scores."""
    pass


@main.command(name="stats")
@click.argument("model", required=True)
@click.option("--provider", "-p", default=None, help="Filter to a specific provider (name or slug).")
@click.option(
    "--sort",
    "-s",
    default="cache",
    type=click.Choice(["cache", "score", "token_cost", "latency", "tps", "uptime", "input", "output", "tokens", "share", "name"]),
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


@main.command(name="score")
@click.argument("model", required=True)
@click.option("--prompt-tokens", "-c", default=2000, type=int, help="Prompt context tokens per turn (C). Default: 2000.")
@click.option("--completion-tokens", "-o", default=500, type=int, help="Completion tokens per turn (O). Default: 500.")
@click.option("--time-value", "-t", default=0.0, type=float, help="Economic time value in USD/hr. Default: 0.0 (pure token cost).")
@click.option("--price-failures/--no-failures", default=True, help="Include failure risk cost based on endpoint uptime.")
@click.option("--prior", default=0.5, type=float, help="Prior cache hit rate for shrinkage. Default: 0.5.")
@click.option("--prior-weight", "-w", default=1e9, type=float, help="Prior weight tokens (W). Default: 1e9.")
@click.option("--discount/--no-discount", default=True, help="Apply endpoint discounts if listed.")
@click.option("--provider", "-p", default=None, help="Filter to a specific provider.")
@click.option("--top", "-n", default=None, type=int, help="Limit to top N providers.")
@click.option("--json-output", "--json", is_flag=True, help="Output raw JSON.")
def score_command(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    time_value: float,
    price_failures: bool,
    prior: float,
    prior_weight: float,
    discount: bool,
    provider: Optional[str],
    top: Optional[int],
    json_output: bool,
):
    """
    Evaluate and rank providers using the ProviderUtility scoring model.
    Calculates expected cost per turn (Token Cost + Time Cost + Failure Risk).
    """
    cfg = ScoringConfig(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        time_value_usd_per_hour=time_value,
        price_failures=price_failures,
        prior=prior,
        prior_weight_tokens=prior_weight,
        apply_discount=discount,
    )

    try:
        scores = score_model_providers(model, config=cfg)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()

    if provider:
        q = provider.lower()
        scores = [s for s in scores if q in s.provider_name.lower() or q in s.provider_slug.lower()]
        if not scores:
            console.print(f"[bold red]Provider '{provider}' not found for model.[/bold red]")
            raise click.Abort()
    elif top:
        scores = scores[:top]

    if json_output:
        click.echo(json.dumps([s.to_dict() for s in scores], indent=2))
        return

    # Header Panel
    mode_desc = "Pure Token Cost" if time_value == 0 and not price_failures else "Full Utility Model"
    console.print()
    console.print(
        Panel.fit(
            f"Model: [bold cyan]{model}[/bold cyan] | Turn: [bold]{prompt_tokens}[/bold] prompt + [bold]{completion_tokens}[/bold] completion tokens\n"
            f"Mode: [bold yellow]{mode_desc}[/bold yellow] | Time Value: [bold]${time_value:.2f}/hr[/bold] | Failure Risk: [bold]{'Yes' if price_failures else 'No'}[/bold]\n"
            f"Shrinkage: prior=[bold]{prior * 100:.0f}%[/bold], weight=[bold]{prior_weight / 1e9:.1f}B[/bold] tokens | Discounts: [bold]{'Applied' if discount else 'List Price'}[/bold]",
            title="ProviderUtility Evaluation & Cost per Turn",
            border_style="magenta"
        )
    )

    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Rank", style="dim", justify="right", width=4)
    table.add_column("Provider", style="bold white", no_wrap=True)
    table.add_column("Total Cost/Turn", justify="right", style="bold green")
    table.add_column("Token Cost", justify="right")
    if time_value > 0:
        table.add_column("Time Cost", justify="right")
    if price_failures:
        table.add_column("Fail Risk", justify="right")
    table.add_column("h (used)", justify="right")
    table.add_column("h (pub)", justify="right", style="dim")
    table.add_column("Hit $/M", justify="right")
    table.add_column("Miss $/M", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("TPS", justify="right")
    table.add_column("Uptime", justify="right")

    for s in scores:
        row = [
            f"#{s.rank}",
            s.provider_name,
            f"${s.total_cost_usd:.6f}",
            f"${s.token_cost_usd:.6f}",
        ]
        if time_value > 0:
            row.append(f"${s.time_cost_usd:.6f}")
        if price_failures:
            row.append(f"${s.failure_cost_usd:.6f}")

        row.extend([
            _color_hit_rate(s.h_used * 100.0),
            f"{s.h_raw * 100.0:.1f}%",
            f"${s.hit_price:.4f}",
            f"${s.miss_price:.4f}",
            f"{s.ttft_seconds:.2f}s" if s.ttft_seconds else "--",
            f"{s.throughput_tps:.0f} tps" if s.throughput_tps else "--",
            f"{s.uptime_pct:.1f}%" if s.uptime_pct else "--",
        ])
        table.add_row(*row)

    console.print(table)
    console.print("[dim]Lower Total Cost represents higher utility. Includes shrinkage and cache hit pricing.[/dim]\n")


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
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()

    cfg = ScoringConfig(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    if provider:
        p = stats.get_provider(provider)
        if not p:
            console.print(f"[bold red]Provider '{provider}' not found for '{stats.model_id}'.[/bold red]")
            raise click.Abort()

        score = p.evaluate_score(cfg)

        if json_output:
            out = p.to_dict()
            if score:
                out["score_evaluation"] = score.to_dict()
            click.echo(json.dumps(out, indent=2))
            return

        console.print()
        console.print(f"[bold cyan]Model:[/bold cyan] {stats.model_name} ({stats.model_id})")
        console.print(f"[bold cyan]Provider:[/bold cyan] {p.name} ({p.slug})")
        console.print(f"[bold cyan]24h Published Cache Hit:[/bold cyan] {_color_hit_rate(p.cache_hit_rate_pct)}")
        if score:
            console.print(f"[bold cyan]Shrunken Hit Rate (h_used):[/bold cyan] {_color_hit_rate(score.h_used * 100.0)}")
            console.print(f"[bold cyan]Expected Token Cost / Turn:[/bold cyan] [bold green]{score.formatted_token_cost}[/bold green] ({prompt_tokens} prompt + {completion_tokens} completion)")
            console.print(f"[bold cyan]Cache Hit Price:[/bold cyan] ${score.hit_price:.4f} /M | [bold cyan]Miss Price:[/bold cyan] ${score.miss_price:.4f} /M")
        console.print(f"[bold cyan]Latency (p50):[/bold cyan] {_color_latency(p.latency_p50_ms, p.formatted_latency)}")
        console.print(f"[bold cyan]Throughput (p50):[/bold cyan] {_color_tps(p.throughput_p50_tps, p.formatted_tps)}")
        console.print(f"[bold cyan]Uptime (24h):[/bold cyan] {_color_uptime(p.uptime_1d_pct, p.formatted_uptime)}")
        console.print(f"[bold cyan]Tokens Served (24h):[/bold cyan] {p.formatted_tokens} ({p.formatted_token_share} share)")
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
@click.option("--prompt-tokens", "-c", default=2000, type=int, help="Prompt tokens for turn evaluation.")
@click.option("--completion-tokens", "-o", default=500, type=int, help="Completion tokens for turn evaluation.")
def compare_command(model: str, providers: List[str], prompt_tokens: int, completion_tokens: int):
    """Compare multiple providers side by side (cache, latency, TPS, uptime, cost score)."""
    try:
        stats = get_model_stats(model)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()

    cfg = ScoringConfig(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
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

    table = Table(title=f"Provider Comparison for {stats.model_name} ({prompt_tokens} in / {completion_tokens} out)", header_style="bold magenta")
    table.add_column("Provider", style="bold white", no_wrap=True)
    table.add_column("Total Cost/Turn", justify="right", style="bold green")
    table.add_column("h (used)", justify="right")
    table.add_column("Cache (pub)", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("TPS", justify="right")
    table.add_column("Uptime", justify="right")
    table.add_column("Tokens (24h)", justify="right")
    table.add_column("Share", justify="right")

    for p in selected:
        sb = p.evaluate_score(cfg)
        cost_str = f"${sb.total_cost_usd:.6f}" if sb else "--"
        h_used_str = _color_hit_rate(sb.h_used * 100.0) if sb else Text("--")
        table.add_row(
            p.name,
            cost_str,
            h_used_str,
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
