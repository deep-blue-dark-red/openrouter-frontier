#!/usr/bin/env python3
"""
frontier.py - Pareto Frontier Analysis for OpenRouter Model Providers.

Identifies Pareto-optimal providers balancing:
  - Scored Cost / Token Cost (lower is better)
  - Latency / TTFT (lower is better)
  - Throughput / TPS (higher is better)
  - Cache Hit Rate (higher is better)
  - Uptime (higher is better)

A provider is Pareto-optimal if no other provider is strictly better in all dimensions.

Usage:
  ./frontier.py [MODEL] [--prompt-tokens C] [--completion-tokens O] [--all-quants] [--json]
"""

import sys
import os
import glob
import json
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Ensure local package and virtualenv dependencies are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

venv_site_packages = glob.glob(os.path.join(SCRIPT_DIR, ".venv/lib/python*/site-packages"))
if venv_site_packages:
    sys.path.insert(0, venv_site_packages[0])

from openrouter_analytics.scoring import (
    ScoringConfig,
    ScoreBreakdown,
)
from openrouter_analytics.client import score_model_providers
from openrouter_analytics.resolver import resolve_model


@dataclass
class FrontierCandidate:
    provider_name: str
    provider_slug: str
    scored_cost_usd: float
    token_cost_usd: float
    h_used: float
    h_raw: float
    hit_price: float
    miss_price: float
    ttft_seconds: Optional[float]
    throughput_tps: Optional[float]
    uptime_pct: Optional[float]
    quantization: str
    is_pareto: bool = False
    niche: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "provider_slug": self.provider_slug,
            "scored_cost_usd": self.scored_cost_usd,
            "token_cost_usd": self.token_cost_usd,
            "h_used": self.h_used,
            "h_raw": self.h_raw,
            "hit_price": self.hit_price,
            "miss_price": self.miss_price,
            "ttft_seconds": self.ttft_seconds,
            "throughput_tps": self.throughput_tps,
            "uptime_pct": self.uptime_pct,
            "quantization": self.quantization,
            "is_pareto": self.is_pareto,
            "niche": self.niche,
        }


def compute_pareto_frontier(
    candidates: List[FrontierCandidate],
    objectives: List[str] = None,
) -> List[FrontierCandidate]:
    """
    Computes multi-objective Pareto frontier.
    By default evaluates:
      - scored_cost_usd (minimize)
      - ttft_seconds (minimize)
      - throughput_tps (maximize)
      - uptime_pct (maximize)
      - h_used (maximize)
    """
    if not candidates:
        return []

    # Filter candidates with valid metrics
    for i, a in enumerate(candidates):
        is_dominated = False
        for j, b in enumerate(candidates):
            if i == j:
                continue

            # Check if b dominates a:
            # b is as good or better in all objectives, and strictly better in at least one
            cost_b_better = b.scored_cost_usd <= a.scored_cost_usd
            cost_b_strict = b.scored_cost_usd < a.scored_cost_usd

            # Latency (lower is better)
            lat_a = a.ttft_seconds if a.ttft_seconds is not None else 999.0
            lat_b = b.ttft_seconds if b.ttft_seconds is not None else 999.0
            lat_b_better = lat_b <= lat_a
            lat_b_strict = lat_b < lat_a

            # Throughput (higher is better)
            tps_a = a.throughput_tps if a.throughput_tps is not None else 0.0
            tps_b = b.throughput_tps if b.throughput_tps is not None else 0.0
            tps_b_better = tps_b >= tps_a
            tps_b_strict = tps_b > tps_a

            # Uptime (higher is better)
            upt_a = a.uptime_pct if a.uptime_pct is not None else 0.0
            upt_b = b.uptime_pct if b.uptime_pct is not None else 0.0
            upt_b_better = upt_b >= upt_a
            upt_b_strict = upt_b > upt_a

            # Cache hit rate (higher is better)
            h_b_better = b.h_used >= a.h_used
            h_b_strict = b.h_used > a.h_used

            if (cost_b_better and lat_b_better and tps_b_better and upt_b_better and h_b_better) and (
                cost_b_strict or lat_b_strict or tps_b_strict or upt_b_strict or h_b_strict
            ):
                is_dominated = True
                break

        a.is_pareto = not is_dominated

    # Classify niches for frontier providers
    min_cost = min(c.scored_cost_usd for c in candidates)
    min_lat = min((c.ttft_seconds for c in candidates if c.ttft_seconds), default=999.0)
    max_tps = max((c.throughput_tps for c in candidates if c.throughput_tps), default=0.0)
    max_hit = max(c.h_used for c in candidates)
    max_upt = max((c.uptime_pct for c in candidates if c.uptime_pct), default=0.0)

    for c in candidates:
        if not c.is_pareto:
            continue
        traits = []
        if abs(c.scored_cost_usd - min_cost) < 1e-6:
            traits.append("Lowest Cost")
        if c.ttft_seconds and abs(c.ttft_seconds - min_lat) < 0.05:
            traits.append("Lowest Latency")
        if c.throughput_tps and abs(c.throughput_tps - max_tps) < 3.0:
            traits.append("Highest TPS")
        if abs(c.h_used - max_hit) < 0.01:
            traits.append("Best Cache Hit")
        if c.uptime_pct and abs(c.uptime_pct - max_upt) < 0.2:
            traits.append("Highest Uptime")

        if not traits:
            traits.append("Balanced Trade-off")
        c.niche = " • ".join(traits)

    return candidates


def print_frontier_table(
    candidates: List[FrontierCandidate],
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    filter_desc: str = "",
):
    """Prints a clean, modern Pareto frontier table."""
    cols = [
        ("Provider", 16, "<"),
        ("Scored Cost", 12, ">"),
        ("Token Cost", 11, ">"),
        ("Latency", 8, ">"),
        ("TPS", 5, ">"),
        ("h(used)", 8, ">"),
        ("Uptime", 7, ">"),
        ("Pareto Frontier", 15, "<"),
        ("Niche / Advantage", 32, "<"),
    ]

    header_parts = [f"{name:>{w}}" if a == ">" else f"{name:<{w}}" for name, w, a in cols]
    header_line = "  ".join(header_parts)
    divider = "─" * len(header_line)

    banner_info = (
        f"Multi-Objective Pareto Analysis  •  Turn: {prompt_tokens} prompt + {completion_tokens} completion tokens\n"
        f"Evaluation: Cost vs Latency vs TPS vs Cache vs Uptime  •  {filter_desc}"
    )

    print()
    print(divider)
    print(f"Pareto Frontier Evaluation: {model_name}")
    print(banner_info)
    print(divider)
    print(header_line)
    print(divider)

    for c in candidates:
        lat_str = f"{c.ttft_seconds:.2f}s" if c.ttft_seconds else "--"
        tps_str = f"{c.throughput_tps:.0f}" if c.throughput_tps else "--"
        upt_str = f"{c.uptime_pct:.1f}%" if c.uptime_pct else "--"
        status_str = "★ OPTIMAL" if c.is_pareto else "Dominated"
        niche_str = c.niche if c.is_pareto else "--"

        row_vals = [
            c.provider_name,
            f"${c.scored_cost_usd:.6f}",
            f"${c.token_cost_usd:.6f}",
            lat_str,
            tps_str,
            f"{c.h_used * 100.0:.1f}%",
            upt_str,
            status_str,
            niche_str,
        ]

        row_str = "  ".join(
            f"{val:>{w}}" if a == ">" else f"{val:<{w}}"
            for val, (_, w, a) in zip(row_vals, cols)
        )
        print(row_str)

    print(divider)
    print("★ OPTIMAL indicates non-dominated Pareto-optimal providers on the cost-performance efficiency frontier.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Pareto frontier across OpenRouter providers for a model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", nargs="?", default="z-ai/glm-5.3-flash", help="Model slug or shorthand")
    parser.add_argument("-c", "--prompt-tokens", type=int, default=2000, help="Prompt tokens per turn (C)")
    parser.add_argument("-o", "--completion-tokens", type=int, default=500, help="Completion tokens per turn (O)")
    parser.add_argument("--quant", type=str, default="auto", help="Quantization filter ('fp8', 'auto', or 'all')")
    parser.add_argument("--all-quants", action="store_true", help="Include all quantizations and unquantized endpoints")
    parser.add_argument("--optimal-only", action="store_true", help="Display only Pareto-optimal providers")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args()

    model_id, canonical_slug, display_name = resolve_model(args.model)

    config = ScoringConfig(
        prompt_tokens=args.prompt_tokens,
        completion_tokens=args.completion_tokens,
        time_value_usd_per_hour=0.0,
        price_failures=True,
    )

    scores = score_model_providers(canonical_slug, config=config)

    # Determine quantization filtering
    quant_filter = None
    if not args.all_quants:
        if args.quant == "auto":
            # Detect primary quantization if model has one (e.g. fp8 for glm-5.3-flash)
            quants = [getattr(s, "quantization", None) for s in scores if getattr(s, "quantization", None)]
            if "fp8" in quants:
                quant_filter = "fp8"
        elif args.quant.lower() != "all":
            quant_filter = args.quant.lower()

    candidates = []
    for s in scores:
        q = getattr(s, "quantization", "unknown") or "unknown"
        if quant_filter and q != quant_filter:
            continue
        candidates.append(
            FrontierCandidate(
                provider_name=s.provider_name,
                provider_slug=s.provider_slug,
                scored_cost_usd=s.total_cost_usd,
                token_cost_usd=s.token_cost_usd,
                h_used=s.h_used,
                h_raw=s.h_raw,
                hit_price=s.hit_price,
                miss_price=s.miss_price,
                ttft_seconds=s.ttft_seconds,
                throughput_tps=s.throughput_tps,
                uptime_pct=s.uptime_pct,
                quantization=q,
            )
        )

    # Sort candidates by cost
    candidates.sort(key=lambda c: c.scored_cost_usd)
    candidates = compute_pareto_frontier(candidates)

    if args.optimal_only:
        candidates = [c for c in candidates if c.is_pareto]

    if args.json:
        print(json.dumps([c.to_dict() for c in candidates], indent=2))
        return

    filter_desc = f"Quantization: {quant_filter.upper()}" if quant_filter else "Quantization: ALL"
    print_frontier_table(
        candidates,
        model_name=f"{display_name} ({model_id})",
        prompt_tokens=args.prompt_tokens,
        completion_tokens=args.completion_tokens,
        filter_desc=filter_desc,
    )


if __name__ == "__main__":
    main()
