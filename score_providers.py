#!/usr/bin/env python3
"""
score_providers.py - Executable Python script implementing the ProviderUtility scoring model.

The token-cost part of the model:
  in        = pricing.prompt            (USD per million tokens, from public endpoints API)
  out       = pricing.completion        (USD per million tokens)
  read      = pricing.input_cache_read  (may be absent)
  write     = pricing.input_cache_write (may be absent)

derived:
  hitPrice  = read ?? in
  missPrice = read is absent ? in : (write ?? in)
  h         = read is absent ? 0 : clamp(cacheHitRate, 0, 1)

Bayesian shrinkage:
  h_used    = (h · T + prior · W) / (T + W)

per turn (C prompt tokens, O completion tokens):
  tokenCost = [ C · ( h_used · hitPrice + (1 − h_used) · missPrice ) + O · out ] / 1_000_000 + requestFee

full utility terms:
  timeCost    = (TimeValueUsdPerHour / 3600) · (ttft + O / throughput)
  failureRisk = (1 − uptime) · ( C · h_used · (missPrice − hitPrice) / 1e6 + timeValuePerSecond · ttft )
  totalCost   = tokenCost + timeCost + failureRisk

Usage:
  ./score_providers.py [MODEL] [--prompt-tokens C] [--completion-tokens O] [--time-value $/hr] [--top N]
"""

import sys
import os
import glob
import json
import argparse
from typing import Optional, List, Dict, Any

# Ensure local package and virtualenv dependencies are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

venv_site_packages = glob.glob(os.path.join(SCRIPT_DIR, ".venv/lib/python*/site-packages"))
if venv_site_packages:
    sys.path.insert(0, venv_site_packages[0])

from openrouter_analytics.scoring import (
    ScoringConfig,
    EndpointPricing,
    ScoreBreakdown,
    evaluate_endpoint,
)
from openrouter_analytics.client import score_model_providers, get_model_stats


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
    """
    Pure Python function evaluating the ProviderUtility scoring model for a single provider.
    
    :param prompt_price_per_m: Input price in USD per million tokens.
    :param completion_price_per_m: Output price in USD per million tokens.
    :param cache_read_price_per_m: Prompt cache read price in USD per million tokens (if supported).
    :param cache_write_price_per_m: Prompt cache write price in USD per million tokens (if billed).
    :param request_fee: Fixed fee per request in USD.
    :param cache_hit_rate: Published 24h cache hit rate (0.0 to 1.0).
    :param total_tokens_served: Total tokens served by endpoint today (for Bayesian shrinkage).
    :param ttft_seconds: Time to first token in seconds (median p50 latency).
    :param throughput_tps: Output tokens per second (median p50 throughput).
    :param uptime_pct: Trailing 24h availability percentage (0.0 to 100.0).
    :param prompt_tokens: Context/prompt tokens per turn (C). Default 2000.
    :param completion_tokens: Output/completion tokens per turn (O). Default 500.
    :param time_value_usd_per_hour: Opportunity cost of time in USD/hr. Default 0.0.
    :param price_failures: Whether to model failure risk. Default True.
    :param prior: Bayesian shrinkage prior belief. Default 0.5.
    :param prior_weight_tokens: Weight of prior belief in tokens (W). Default 1e9.
    :return: ScoreBreakdown dataclass with total_cost_usd, token_cost_usd, time_cost_usd, etc.
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


def score_model(
    model: str = "z-ai/glm-5.3-flash",
    prompt_tokens: int = 2000,
    completion_tokens: int = 500,
    time_value_usd_per_hour: float = 0.0,
    price_failures: bool = True,
    prior: float = 0.5,
    prior_weight_tokens: float = 1e9,
    apply_discount: bool = True,
) -> List[ScoreBreakdown]:
    """
    Fetches OpenRouter analytics for a given model and returns providers ranked by expected cost per turn.
    """
    config = ScoringConfig(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        time_value_usd_per_hour=time_value_usd_per_hour,
        price_failures=price_failures,
        prior=prior,
        prior_weight_tokens=prior_weight_tokens,
        apply_discount=apply_discount,
    )
    return score_model_providers(model, config=config)


def print_table(
    results: List[ScoreResult],
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    time_value: float,
    prior: float,
    prior_weight: float,
    price_failures: bool,
    discounts_applied: bool = True,
):
    """Prints a clean, modern table using connected dashes and space separation (no vertical bars, no ===)."""
    show_time = time_value > 0
    show_failures = price_failures

    cols = [
        ("Provider", 16, "<"),
        ("Scored Cost", 12, ">"),
        ("Token Cost", 11, ">"),
    ]
    if show_time:
        cols.append(("Time Cost", 11, ">"))
    if show_failures:
        cols.append(("Fail Risk", 10, ">"))
    cols.extend([
        ("h(used)", 8, ">"),
        ("h(pub)", 7, ">"),
        ("Hit $/M", 8, ">"),
        ("Miss $/M", 9, ">"),
        ("Latency", 8, ">"),
        ("TPS", 5, ">"),
        ("Uptime", 7, ">"),
    ])

    header_parts = [f"{name:>{w}}" if a == ">" else f"{name:<{w}}" for name, w, a in cols]
    header_line = "  ".join(header_parts)
    divider = "─" * len(header_line)

    mode = "Pure Token Cost" if time_value == 0 and not price_failures else "Full Utility Model"
    banner_info = (
        f"Mode: {mode}  •  Turn: {prompt_tokens} prompt + {completion_tokens} completion tokens  •  Time Value: ${time_value:.2f}/hr\n"
        f"Shrinkage: prior={prior * 100:.0f}%, weight={prior_weight / 1e9:.1f}B tokens  •  Discounts: {'Applied' if discounts_applied else 'List Price'}  •  Failure Risk: {'Yes' if price_failures else 'No'}"
    )

    print()
    print(divider)
    print(f"ProviderUtility Evaluation: {model_name}")
    print(banner_info)
    print(divider)
    print(header_line)
    print(divider)

    for r in results:
        row_vals = [
            r.provider_name,
            f"${r.total_cost_usd:.6f}",
            f"${r.token_cost_usd:.6f}",
        ]
        if show_time:
            row_vals.append(f"${r.time_cost_usd:.6f}")
        if show_failures:
            row_vals.append(f"${r.failure_cost_usd:.6f}")

        lat_str = f"{r.ttft_seconds:.2f}s" if r.ttft_seconds else "--"
        tps_str = f"{r.throughput_tps:.0f}" if r.throughput_tps else "--"
        upt_str = f"{r.uptime_pct:.1f}%" if r.uptime_pct else "--"

        row_vals.extend([
            f"{r.h_used * 100.0:.1f}%",
            f"{r.h_raw * 100.0:.1f}%",
            f"${r.hit_price:.4f}",
            f"${r.miss_price:.4f}",
            lat_str,
            tps_str,
            upt_str,
        ])

        row_str = "  ".join(
            f"{val:>{w}}" if a == ">" else f"{val:<{w}}"
            for val, (_, w, a) in zip(row_vals, cols)
        )
        print(row_str)

    print(divider)
    print("Lower Total Cost represents higher utility. Ranks include cache hit rates, shrinkage, and endpoint metrics.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OpenRouter providers using ProviderUtility scoring model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", nargs="?", default="z-ai/glm-5.3-flash", help="Model slug or shorthand")
    parser.add_argument("-c", "--prompt-tokens", type=int, default=2000, help="Prompt context tokens per turn (C)")
    parser.add_argument("-o", "--completion-tokens", type=int, default=500, help="Completion tokens per turn (O)")
    parser.add_argument("-t", "--time-value", type=float, default=0.0, help="Time value in USD/hr (0 for pure token cost)")
    parser.add_argument("--no-failures", action="store_true", help="Disable uptime failure risk modeling")
    parser.add_argument("--prior", type=float, default=0.5, help="Bayesian shrinkage prior (0.0 - 1.0)")
    parser.add_argument("--prior-weight", type=float, default=1e9, help="Bayesian prior weight tokens (W)")
    parser.add_argument("--no-discount", action="store_true", help="Use raw list pricing without endpoint discounts")
    parser.add_argument("-n", "--top", type=int, default=10, help="Number of top providers to display")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args()

    results = score_model(
        model=args.model,
        prompt_tokens=args.prompt_tokens,
        completion_tokens=args.completion_tokens,
        time_value_usd_per_hour=args.time_value,
        price_failures=not args.no_failures,
        prior=args.prior,
        prior_weight_tokens=args.prior_weight,
        apply_discount=not args.no_discount,
    )

    if args.json:
        print(json.dumps([r.to_dict() for r in results[:args.top]], indent=2))
        return

    print_table(
        results[:args.top],
        model_name=args.model,
        prompt_tokens=args.prompt_tokens,
        completion_tokens=args.completion_tokens,
        time_value=args.time_value,
        prior=args.prior,
        prior_weight=args.prior_weight,
        price_failures=not args.no_failures,
        discounts_applied=not args.no_discount,
    )

if __name__ == "__main__":
    main()
