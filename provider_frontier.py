#!/usr/bin/env python3
"""
provider_frontier.py - Pareto Frontier Analysis for OpenRouter Models & Providers.

Supports dual modes:
  1. Model Frontier: Multi-objective Pareto analysis across all 400+ models in OpenRouter
     (Cost vs. Context Window vs. Cache Read Price vs. Output Price).
  2. Provider Frontier: Multi-objective Pareto analysis across all serving providers for a specific model
     (Cost vs. Latency vs. TPS Throughput vs. Cache Hit Rate vs. Uptime).

Usage:
  ./provider_frontier.py [MODEL]                 # Provider Pareto frontier for a specific model
  ./provider_frontier.py --models                # Model Pareto frontier across all OpenRouter models
  ./provider_frontier.py --optimal-only          # Show only Pareto-optimal candidates
"""

import sys
import os
import glob
import json
import socket
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Optimize connection on macOS (avoid IPv6 timeout)
try:
    import urllib3.util.connection
    urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

venv_site_packages = glob.glob(os.path.join(SCRIPT_DIR, ".venv/lib/python*/site-packages"))
if venv_site_packages:
    sys.path.insert(0, venv_site_packages[0])

from openrouter_analytics.scoring import ScoringConfig, ScoreBreakdown
from openrouter_analytics.client import score_model_providers
from openrouter_analytics.resolver import resolve_model, get_all_models


# ==============================================================================
# Provider Frontier Candidate & Logic
# ==============================================================================

@dataclass
class ProviderCandidate:
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


def compute_provider_pareto(candidates: List[ProviderCandidate]) -> List[ProviderCandidate]:
    """Computes Pareto frontier for providers serving a specific model."""
    for i, a in enumerate(candidates):
        is_dominated = False
        for j, b in enumerate(candidates):
            if i == j:
                continue

            cost_b_better = b.scored_cost_usd <= a.scored_cost_usd
            cost_b_strict = b.scored_cost_usd < a.scored_cost_usd

            lat_a = a.ttft_seconds if a.ttft_seconds is not None else 999.0
            lat_b = b.ttft_seconds if b.ttft_seconds is not None else 999.0
            lat_b_better = lat_b <= lat_a
            lat_b_strict = lat_b < lat_a

            tps_a = a.throughput_tps if a.throughput_tps is not None else 0.0
            tps_b = b.throughput_tps if b.throughput_tps is not None else 0.0
            tps_b_better = tps_b >= tps_a
            tps_b_strict = tps_b > tps_a

            upt_a = a.uptime_pct if a.uptime_pct is not None else 0.0
            upt_b = b.uptime_pct if b.uptime_pct is not None else 0.0
            upt_b_better = upt_b >= upt_a
            upt_b_strict = upt_b > upt_a

            h_b_better = b.h_used >= a.h_used
            h_b_strict = b.h_used > a.h_used

            if (cost_b_better and lat_b_better and tps_b_better and upt_b_better and h_b_better) and (
                cost_b_strict or lat_b_strict or tps_b_strict or upt_b_strict or h_b_strict
            ):
                is_dominated = True
                break

        a.is_pareto = not is_dominated

    min_cost = min(c.scored_cost_usd for c in candidates) if candidates else 0.0
    min_lat = min((c.ttft_seconds for c in candidates if c.ttft_seconds), default=999.0)
    max_tps = max((c.throughput_tps for c in candidates if c.throughput_tps), default=0.0)
    max_hit = max((c.h_used for c in candidates), default=0.0)
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

        c.niche = " • ".join(traits) if traits else "Balanced Trade-off"

    return candidates


# ==============================================================================
# Model Frontier Candidate & Logic
# ==============================================================================

@dataclass
class ModelCandidate:
    model_id: str
    model_name: str
    prompt_price_per_m: float
    completion_price_per_m: float
    cache_read_price_per_m: Optional[float]
    context_length: int
    turn_cost_usd: float
    is_pareto: bool = False
    niche: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "prompt_price_per_m": self.prompt_price_per_m,
            "completion_price_per_m": self.completion_price_per_m,
            "cache_read_price_per_m": self.cache_read_price_per_m,
            "context_length": self.context_length,
            "turn_cost_usd": self.turn_cost_usd,
            "is_pareto": self.is_pareto,
            "niche": self.niche,
        }


def compute_model_pareto(candidates: List[ModelCandidate]) -> List[ModelCandidate]:
    """Computes Pareto frontier across models (Cost vs. Context Window vs. Cache Price)."""
    for i, a in enumerate(candidates):
        is_dominated = False
        for j, b in enumerate(candidates):
            if i == j:
                continue

            cost_b_better = b.turn_cost_usd <= a.turn_cost_usd
            cost_b_strict = b.turn_cost_usd < a.turn_cost_usd

            ctx_b_better = b.context_length >= a.context_length
            ctx_b_strict = b.context_length > a.context_length

            read_a = a.cache_read_price_per_m if a.cache_read_price_per_m is not None else a.prompt_price_per_m
            read_b = b.cache_read_price_per_m if b.cache_read_price_per_m is not None else b.prompt_price_per_m
            read_b_better = read_b <= read_a
            read_b_strict = read_b < read_a

            out_b_better = b.completion_price_per_m <= a.completion_price_per_m
            out_b_strict = b.completion_price_per_m < a.completion_price_per_m

            if (cost_b_better and ctx_b_better and read_b_better and out_b_better) and (
                cost_b_strict or ctx_b_strict or read_b_strict or out_b_strict
            ):
                is_dominated = True
                break

        a.is_pareto = not is_dominated

    min_cost = min(c.turn_cost_usd for c in candidates) if candidates else 0.0
    max_ctx = max(c.context_length for c in candidates) if candidates else 0
    min_read = min((c.cache_read_price_per_m for c in candidates if c.cache_read_price_per_m is not None), default=999.0)

    for c in candidates:
        if not c.is_pareto:
            continue
        traits = []
        if abs(c.turn_cost_usd - min_cost) < 1e-6:
            traits.append("Cheapest Model")
        if c.context_length == max_ctx:
            traits.append(f"Max Context ({max_ctx // 1000}k)")
        elif c.context_length >= 1_000_000:
            traits.append("1M Context")
        if c.cache_read_price_per_m and abs(c.cache_read_price_per_m - min_read) < 1e-4:
            traits.append("Cheapest Cache Read")

        c.niche = " • ".join(traits) if traits else "Cost/Context Trade-off"

    return candidates


# ==============================================================================
# Rendering
# ==============================================================================

def print_provider_table(candidates: List[ProviderCandidate], model_name: str, C: int, O: int, filter_desc: str):
    cols = [
        ("Provider", 16, "<"),
        ("Scored Cost", 12, ">"),
        ("Token Cost", 11, ">"),
        ("Latency", 8, ">"),
        ("TPS", 5, ">"),
        ("CacheHit", 8, ">"),
        ("Uptime", 7, ">"),
        ("Pareto Frontier", 15, "<"),
        ("Niche / Advantage", 32, "<"),
    ]

    header_parts = [f"{name:>{w}}" if a == ">" else f"{name:<{w}}" for name, w, a in cols]
    header_line = "  ".join(header_parts)
    divider = "─" * len(header_line)

    banner_info = (
        f"Provider Pareto Frontier  •  Turn: {C} prompt + {O} completion tokens\n"
        f"Evaluation: Cost vs Latency vs TPS vs Cache vs Uptime  •  {filter_desc}"
    )

    print()
    print(divider)
    print(f"Provider Pareto Frontier: {model_name}")
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

        print("  ".join(f"{val:>{w}}" if a == ">" else f"{val:<{w}}" for val, (_, w, a) in zip(row_vals, cols)))

    print(divider)
    print("★ OPTIMAL indicates non-dominated Pareto-optimal providers on the cost-performance efficiency frontier.\n")


def print_model_table(candidates: List[ModelCandidate], C: int, O: int):
    cols = [
        ("Model ID", 34, "<"),
        ("Turn Cost", 11, ">"),
        ("Prompt $/M", 10, ">"),
        ("Compl $/M", 10, ">"),
        ("Read $/M", 9, ">"),
        ("Context", 10, ">"),
        ("Frontier", 12, "<"),
        ("Niche / Advantage", 28, "<"),
    ]

    header_parts = [f"{name:>{w}}" if a == ">" else f"{name:<{w}}" for name, w, a in cols]
    header_line = "  ".join(header_parts)
    divider = "─" * len(header_line)

    banner_info = (
        f"Catalog-Wide Model Pareto Frontier across {len(candidates)} Active Models\n"
        f"Multi-Objective Optimization: Turn Cost ({C} in + {O} out) vs Context Length vs Cache Pricing"
    )

    print()
    print(divider)
    print("OpenRouter Model Pareto Frontier")
    print(banner_info)
    print(divider)
    print(header_line)
    print(divider)

    for c in candidates:
        read_str = f"${c.cache_read_price_per_m:.4f}" if c.cache_read_price_per_m is not None else "--"
        status_str = "★ OPTIMAL" if c.is_pareto else "Dominated"
        niche_str = c.niche if c.is_pareto else "--"
        ctx_str = f"{c.context_length // 1000}k" if c.context_length >= 1000 else str(c.context_length)

        row_vals = [
            c.model_id[:34],
            f"${c.turn_cost_usd:.6f}",
            f"${c.prompt_price_per_m:.4f}",
            f"${c.completion_price_per_m:.4f}",
            read_str,
            ctx_str,
            status_str,
            niche_str,
        ]

        print("  ".join(f"{val:>{w}}" if a == ">" else f"{val:<{w}}" for val, (_, w, a) in zip(row_vals, cols)))

    print(divider)
    print("★ OPTIMAL indicates non-dominated models on the global cost-vs-context efficiency frontier.\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute Pareto frontier across OpenRouter models or providers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", nargs="?", default=None, help="Model slug (or omit / use 'models' for Model Frontier)")
    parser.add_argument("--models", action="store_true", help="Analyze Pareto frontier across all OpenRouter models")
    parser.add_argument("-c", "--prompt-tokens", type=int, default=2000, help="Prompt context tokens per turn (C)")
    parser.add_argument("-o", "--completion-tokens", type=int, default=500, help="Completion tokens per turn (O)")
    parser.add_argument("--quant", type=str, default="auto", help="Quantization filter for provider mode ('fp8', 'auto', 'all')")
    parser.add_argument("--all-quants", action="store_true", help="Include all quantizations in provider mode")
    parser.add_argument("--optimal-only", action="store_true", help="Display only Pareto-optimal candidates")
    parser.add_argument("--top", type=int, default=25, help="Max candidates to display in model mode")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args()

    # 1. Model Frontier Mode
    if args.models or (args.model and args.model.lower() in ("models", "all", "catalog")):
        raw_models = get_all_models()
        candidates: List[ModelCandidate] = []
        for m in raw_models:
            p = m.get("pricing", {})
            raw_prompt = float(p.get("prompt") or 0.0)
            raw_compl = float(p.get("completion") or 0.0)
            prompt_p = raw_prompt * 1e6 if raw_prompt < 0.01 else raw_prompt
            compl_p = raw_compl * 1e6 if raw_compl < 0.01 else raw_compl
            raw_read = float(p.get("input_cache_read") or 0.0) if p.get("input_cache_read") else None
            read_p = (raw_read * 1e6 if raw_read < 0.01 else raw_read) if raw_read is not None else None
            ctx = int(m.get("context_length") or 0)

            if prompt_p <= 0 or compl_p <= 0 or ctx <= 0:
                continue

            turn_cost = (args.prompt_tokens * prompt_p + args.completion_tokens * compl_p) / 1e6
            candidates.append(
                ModelCandidate(
                    model_id=m["id"],
                    model_name=m.get("name") or m["id"],
                    prompt_price_per_m=prompt_p,
                    completion_price_per_m=compl_p,
                    cache_read_price_per_m=read_p,
                    context_length=ctx,
                    turn_cost_usd=turn_cost,
                )
            )

        candidates = compute_model_pareto(candidates)
        candidates.sort(key=lambda c: c.turn_cost_usd)

        if args.optimal_only:
            candidates = [c for c in candidates if c.is_pareto]
        else:
            # Show Pareto optimal first, then top by cost up to args.top
            pareto_set = [c for c in candidates if c.is_pareto]
            other_set = [c for c in candidates if not c.is_pareto][: max(0, args.top - len(pareto_set))]
            candidates = pareto_set + other_set
            candidates.sort(key=lambda c: (not c.is_pareto, c.turn_cost_usd))

        if args.json:
            print(json.dumps([c.to_dict() for c in candidates], indent=2))
            return

        print_model_table(candidates, args.prompt_tokens, args.completion_tokens)
        return

    # 2. Provider Frontier Mode (for a single model)
    target_model = args.model or "z-ai/glm-5.3-flash"
    model_id, canonical_slug, display_name = resolve_model(target_model)

    config = ScoringConfig(
        prompt_tokens=args.prompt_tokens,
        completion_tokens=args.completion_tokens,
        time_value_usd_per_hour=0.0,
        price_failures=True,
    )

    scores = score_model_providers(canonical_slug, config=config)

    quant_filter = None
    if not args.all_quants:
        if args.quant == "auto":
            quants = [getattr(s, "quantization", None) for s in scores if getattr(s, "quantization", None)]
            if "fp8" in quants:
                quant_filter = "fp8"
        elif args.quant.lower() != "all":
            quant_filter = args.quant.lower()

    candidates: List[ProviderCandidate] = []
    for s in scores:
        q = getattr(s, "quantization", "unknown") or "unknown"
        if quant_filter and q != quant_filter:
            continue
        candidates.append(
            ProviderCandidate(
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

    candidates.sort(key=lambda c: c.scored_cost_usd)
    candidates = compute_provider_pareto(candidates)

    if args.optimal_only:
        candidates = [c for c in candidates if c.is_pareto]

    if args.json:
        print(json.dumps([c.to_dict() for c in candidates], indent=2))
        return

    filter_desc = f"Quantization: {quant_filter.upper()}" if quant_filter else "Quantization: ALL"
    print_provider_table(
        candidates,
        model_name=f"{display_name} ({model_id})",
        C=args.prompt_tokens,
        O=args.completion_tokens,
        filter_desc=filter_desc,
    )


if __name__ == "__main__":
    main()
