#!/usr/bin/env python3
"""score_providers.py - rank a model's providers by ProviderScore expected cost per turn.

Usage:
  ./score_providers.py [MODEL] [-c PROMPT_TOKENS] [-o COMPLETION_TOKENS] [-t USD_PER_HOUR] [-n TOP]

See the README's "Scoring model" section for the full formulas.
"""

import argparse
import json
from typing import List, Optional

import _bootstrap  # noqa: F401  (makes the local package and .venv importable)

from openrouter_frontier._util import filter_primary_quantization
from openrouter_frontier.client import score_model_providers
from openrouter_frontier.render import Column, fmt_pct, fmt_seconds, fmt_tps, print_table
from openrouter_frontier.resolver import resolve_model
from openrouter_frontier.profile_args import add_task_args, config_from_args, describe_profile
from openrouter_frontier.scoring import EndpointInputs, EndpointPricing, ScoreBreakdown, ScoringConfig, evaluate_endpoint


def evaluate_provider_utility(
    prompt_price_per_m: float,
    completion_price_per_m: float,
    cache_read_price_per_m: Optional[float] = None,
    cache_write_price_per_m: Optional[float] = None,
    request_fee: float = 0.0,
    cache_hit_rate: Optional[float] = None,
    ttft_seconds: Optional[float] = None,
    ttft_p90_seconds: Optional[float] = None,
    throughput_tps: Optional[float] = None,
    throughput_p90_tps: Optional[float] = None,
    uptime_pct: Optional[float] = None,
    config: Optional[ScoringConfig] = None,
    provider_name: str = "Unknown",
    provider_slug: str = "unknown",
) -> ScoreBreakdown:
    """Score a single hypothetical endpoint from raw numbers, without touching the network.

    Prices are USD per million tokens; ``cache_hit_rate`` is 0..1; ``uptime_pct`` is 0..100.
    """
    pricing = EndpointPricing(
        prompt=prompt_price_per_m,
        completion=completion_price_per_m,
        input_cache_read=cache_read_price_per_m,
        input_cache_write=cache_write_price_per_m,
        request_fee=request_fee,
    )
    inputs = EndpointInputs(
        cache_hit_rate=cache_hit_rate, uptime_pct=uptime_pct,
        ttft_p50=ttft_seconds, ttft_p90=ttft_p90_seconds,
        tps_p50=throughput_tps, tps_p90=throughput_p90_tps,
    )
    return evaluate_endpoint(
        pricing=pricing, inputs=inputs, config=config or ScoringConfig(),
        provider_name=provider_name, provider_slug=provider_slug,
    )


def print_scores(results: List[ScoreBreakdown], model_name: str, cfg: ScoringConfig, quant_desc: str) -> None:
    show_obj = cfg.lambda_proc > 0 or cfg.lambda_par > 0
    cols = [Column("Provider", 16), Column("Task $", 9, ">")]
    if show_obj:
        cols.append(Column("Objective $", 11, ">"))
    cols += [
        Column("Token $", 9, ">"),
        Column("Fail $", 8, ">"),
        Column("Time $", 8, ">"),
        Column("Cache Hit", 9, ">"),
        Column("TTFT", 7, ">"),
        Column("TPS", 5, ">"),
        Column("Turn Time", 9, ">"),
        Column("Task Time", 9, ">"),
        Column("Uptime", 7, ">"),
        Column("$/M", 8, ">"),
    ]

    rows = []
    for r in results:
        row = [r.provider_name + ("*" if r.imputed else ""), r.formatted_task_cost]
        if show_obj:
            row.append(r.formatted_objective)
        row += [
            r.formatted_token_cost,
            r.formatted_failure_cost,
            r.formatted_time_cost,
            r.formatted_cache_hit_rate,
            fmt_seconds(r.ttft_seconds),
            fmt_tps(r.throughput_tps),
            r.formatted_turn_time,
            r.formatted_task_time,
            fmt_pct(r.uptime_pct),
            f"${r.task_cost_per_m:.4f}",
        ]
        rows.append(row)

    footer = (f"Task $ = Token $ + Fail $ + Time $, the expected cost of the whole task; lower is better.  "
              f"Token $ = money billed for tokens (new, output, cached reads, cache-miss re-reads).  "
              f"Fail $ = money for cold retries after failures.  "
              f"Time $ = Task Time valued at ${cfg.time_value_usd_per_hour:.0f}/hr; nothing else contains time.  "
              f"Task Time = expected wall-clock: prefill of new tokens and of the prefix after a cache miss, decoding, overhead.  "
              f"TTFT = published median-to-p90 lognormal mean, shown, not charged.  "
              f"Cache Hit = published 24h rate.  $/M = Task $ per 1M submitted tokens (secondary).")
    if any(r.imputed for r in results):
        footer += "  * = missing telemetry imputed as the worst observed for this model."
    print_table(
        cols,
        rows,
        title=f"ProviderScore Task Cost: {model_name}",
        subtitle_lines=[describe_profile(cfg), quant_desc],
        footer=footer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank OpenRouter providers for a model by the expected cost of a whole task (ProviderScore).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", nargs="?", default="z-ai/glm-5.3-flash", help="Model slug or shorthand")
    add_task_args(parser)
    parser.add_argument("--all-quants", action="store_true", help="Include all quantizations (default: primary fp8 variant only)")
    parser.add_argument("-n", "--top", type=int, default=10, help="Number of providers to display")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    model_id, canonical_slug, display_name = resolve_model(args.model)
    cfg = config_from_args(args)

    results = score_model_providers(canonical_slug, config=cfg)
    results = filter_primary_quantization(results, args.all_quants)[: args.top]

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    quant_desc = "Quantization: ALL" if args.all_quants else "Quantization: primary"
    print_scores(results, f"{display_name} ({model_id})", cfg, quant_desc)


if __name__ == "__main__":
    main()
