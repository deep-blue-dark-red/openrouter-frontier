#!/usr/bin/env python3
"""score_providers.py - rank a model's providers by ProviderUtility expected cost per turn.

Usage:
  ./score_providers.py [MODEL] [-c PROMPT_TOKENS] [-o COMPLETION_TOKENS] [-t USD_PER_HOUR] [-n TOP]

See the README's "Scoring model" section for the full formulas.
"""

import argparse
import json
from typing import List, Optional

import _bootstrap  # noqa: F401  (makes the local package and .venv importable)

from openrouter_analytics._util import filter_primary_quantization
from openrouter_analytics.client import score_model_providers
from openrouter_analytics.render import Column, fmt_pct, fmt_seconds, fmt_tps, print_table
from openrouter_analytics.resolver import resolve_model
from openrouter_analytics.scoring import EndpointPricing, ScoreBreakdown, ScoringConfig, evaluate_endpoint


def evaluate_provider_utility(
    prompt_price_per_m: float,
    completion_price_per_m: float,
    cache_read_price_per_m: Optional[float] = None,
    cache_write_price_per_m: Optional[float] = None,
    request_fee: float = 0.0,
    cache_hit_rate: Optional[float] = None,
    total_tokens_served: int = 0,
    ttft_seconds: Optional[float] = None,
    throughput_tps: Optional[float] = None,
    uptime_pct: Optional[float] = None,
    prompt_tokens: int = 2000,
    completion_tokens: int = 500,
    time_value_usd_per_hour: float = 0.0,
    price_failures: bool = True,
    prior: float = 0.5,
    prior_weight_tokens: float = 1e9,
    provider_name: str = "Unknown",
    provider_slug: str = "unknown",
) -> ScoreBreakdown:
    """Score a single hypothetical provider from raw numbers, without touching the network.

    Prices are USD per million tokens; ``cache_hit_rate`` is 0..1; ``uptime_pct`` is 0..100.
    """
    pricing = EndpointPricing(
        prompt=prompt_price_per_m,
        completion=completion_price_per_m,
        input_cache_read=cache_read_price_per_m,
        input_cache_write=cache_write_price_per_m,
        request_fee=request_fee,
    )
    config = ScoringConfig(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        time_value_usd_per_hour=time_value_usd_per_hour,
        price_failures=price_failures,
        prior=prior,
        prior_weight_tokens=prior_weight_tokens,
    )
    return evaluate_endpoint(
        pricing=pricing,
        cache_hit_rate=cache_hit_rate,
        total_tokens=total_tokens_served,
        ttft_seconds=ttft_seconds,
        throughput_tps=throughput_tps,
        uptime_pct=uptime_pct,
        config=config,
        provider_name=provider_name,
        provider_slug=provider_slug,
    )


def print_scores(results: List[ScoreBreakdown], model_name: str, cfg: ScoringConfig, quant_desc: str) -> None:
    show_time = cfg.time_value_usd_per_hour > 0
    cols = [Column("Provider", 16), Column("Scored $/M", 12, ">"), Column("Token $/M", 11, ">")]
    if show_time:
        cols.append(Column("Time $/M", 11, ">"))
    if cfg.price_failures:
        cols.append(Column("Fail $/M", 10, ">"))
    cols += [
        Column("CacheHit", 8, ">"),
        Column("h(pub)", 7, ">"),
        Column("Hit $/M", 8, ">"),
        Column("Miss $/M", 9, ">"),
        Column("Latency", 8, ">"),
        Column("TPS", 5, ">"),
        Column("Uptime", 7, ">"),
    ]

    rows = []
    for r in results:
        row = [r.provider_name, r.formatted_total_cost, r.formatted_token_cost]
        if show_time:
            row.append(r.formatted_time_cost)
        if cfg.price_failures:
            row.append(r.formatted_failure_cost)
        row += [
            r.formatted_h_used,
            r.formatted_h_raw,
            f"${r.hit_price:.4f}",
            f"${r.miss_price:.4f}",
            fmt_seconds(r.ttft_seconds),
            fmt_tps(r.throughput_tps),
            fmt_pct(r.uptime_pct),
        ]
        rows.append(row)

    mode = "Pure Token Cost" if not show_time and not cfg.price_failures else "Full Utility Model"
    print_table(
        cols,
        rows,
        title=f"ProviderUtility Evaluation: {model_name}",
        subtitle_lines=[
            f"Mode: {mode}  •  Turn: {cfg.prompt_tokens} prompt + {cfg.completion_tokens} completion tokens"
            f"  •  Time Value: ${cfg.time_value_usd_per_hour:.2f}/hr",
            f"Shrinkage: prior={cfg.prior * 100:.0f}%, weight={cfg.prior_weight_tokens / 1e9:.1f}B tokens"
            f"  •  Discounts: {'Applied' if cfg.apply_discount else 'List Price'}"
            f"  •  Failure Risk: {'Yes' if cfg.price_failures else 'No'}  •  {quant_desc}",
        ],
        footer="Lower Scored $/M is better. CacheHit is the shrunk hit rate used in scoring; h(pub) is the published 24h rate.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank OpenRouter providers for a model by ProviderUtility expected cost per turn.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", nargs="?", default="z-ai/glm-5.3-flash", help="Model slug or shorthand")
    parser.add_argument("-c", "--prompt-tokens", type=int, default=2000, help="Prompt tokens per turn (C)")
    parser.add_argument("-o", "--completion-tokens", type=int, default=500, help="Completion tokens per turn (O)")
    parser.add_argument("-t", "--time-value", type=float, default=0.0, help="Time value in USD/hr (0 = pure token cost)")
    parser.add_argument("--no-failures", action="store_true", help="Disable uptime failure-risk pricing")
    parser.add_argument("--prior", type=float, default=0.5, help="Bayesian shrinkage prior hit rate (0..1)")
    parser.add_argument("--prior-weight", type=float, default=1e9, help="Bayesian prior weight in tokens (W)")
    parser.add_argument("--no-discount", action="store_true", help="Use list pricing, ignoring endpoint discounts")
    parser.add_argument("--all-quants", action="store_true", help="Include all quantizations (default: primary fp8 variant only)")
    parser.add_argument("-n", "--top", type=int, default=10, help="Number of providers to display")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    model_id, canonical_slug, display_name = resolve_model(args.model)
    cfg = ScoringConfig(
        prompt_tokens=args.prompt_tokens,
        completion_tokens=args.completion_tokens,
        time_value_usd_per_hour=args.time_value,
        price_failures=not args.no_failures,
        prior=args.prior,
        prior_weight_tokens=args.prior_weight,
        apply_discount=not args.no_discount,
    )

    results = score_model_providers(canonical_slug, config=cfg)
    results = filter_primary_quantization(results, args.all_quants)[: args.top]

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    quant_desc = "Quantization: ALL" if args.all_quants else "Quantization: primary"
    print_scores(results, f"{display_name} ({model_id})", cfg, quant_desc)


if __name__ == "__main__":
    main()
