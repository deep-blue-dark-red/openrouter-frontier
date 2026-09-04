#!/usr/bin/env python3
"""provider_frontier.py - Pareto frontier of the providers serving one model.

Instead of collapsing latency, throughput, and reliability into a single dollar figure via a
time-value guess, this reports which providers are non-dominated across five objectives:

    minimise  scored cost per turn      (ProviderUtility, pure token cost + failure risk)
    minimise  p50 time to first token
    maximise  p50 throughput (tokens/s)
    maximise  shrunk cache hit rate
    maximise  24h uptime

A secondary ``--models`` mode runs a catalog-wide frontier over turn cost, context length,
cache-read price, and completion price. For a cost-vs-benchmark-quality frontier with efficient point
detection, see model_frontier.py.

Usage:
  ./provider_frontier.py z-ai/glm-5.3-flash
  ./provider_frontier.py z-ai/glm-5.3-flash --optimal-only --all-quants
  ./provider_frontier.py --models --optimal-only
"""

import argparse
import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import _bootstrap  # noqa: F401

from openrouter_frontier._util import filter_primary_quantization, price_per_million
from openrouter_frontier.client import score_model_providers
from openrouter_frontier.pareto import Objective, pareto_mask
from openrouter_frontier.render import Column, fmt_context, fmt_pct, fmt_seconds, fmt_tps, print_table
from openrouter_frontier.resolver import get_all_models, resolve_model
from openrouter_frontier.scoring import ScoringConfig

OPTIMAL = "★ OPTIMAL"
DOMINATED = "Dominated"


def _near(a: Optional[float], b: float, tol: float) -> bool:
    return a is not None and abs(a - b) < tol


# ------------------------------------------------------------------ provider frontier

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


PROVIDER_OBJECTIVES = [
    Objective(lambda c: c.scored_cost_usd, minimize=True),
    Objective(lambda c: c.ttft_seconds, minimize=True),
    Objective(lambda c: c.throughput_tps, minimize=False),
    Objective(lambda c: c.h_used, minimize=False),
    Objective(lambda c: c.uptime_pct, minimize=False),
]


def compute_provider_pareto(candidates: List[ProviderCandidate]) -> List[ProviderCandidate]:
    """Mark Pareto-optimal providers and label each with the objectives it leads on."""
    if not candidates:
        return candidates
    for c, optimal in zip(candidates, pareto_mask(candidates, PROVIDER_OBJECTIVES)):
        c.is_pareto = optimal

    min_cost = min(c.scored_cost_usd for c in candidates)
    min_lat = min((c.ttft_seconds for c in candidates if c.ttft_seconds is not None), default=None)
    max_tps = max((c.throughput_tps for c in candidates if c.throughput_tps is not None), default=None)
    max_hit = max(c.h_used for c in candidates)
    max_upt = max((c.uptime_pct for c in candidates if c.uptime_pct is not None), default=None)

    for c in candidates:
        if not c.is_pareto:
            continue
        traits = []
        if _near(c.scored_cost_usd, min_cost, 1e-6):
            traits.append("Lowest Cost")
        if min_lat is not None and _near(c.ttft_seconds, min_lat, 0.05):
            traits.append("Lowest Latency")
        if max_tps is not None and _near(c.throughput_tps, max_tps, 3.0):
            traits.append("Highest TPS")
        if _near(c.h_used, max_hit, 0.01):
            traits.append("Best Cache Hit")
        if max_upt is not None and _near(c.uptime_pct, max_upt, 0.2):
            traits.append("Highest Uptime")
        c.niche = " • ".join(traits) or "Balanced Trade-off"
    return candidates


def print_provider_table(candidates: List[ProviderCandidate], model_name: str, cfg: ScoringConfig, filter_desc: str) -> None:
    cols = [
        Column("Provider", 16),
        Column("Scored $/M", 12, ">"),
        Column("Token $/M", 11, ">"),
        Column("Latency", 8, ">"),
        Column("TPS", 5, ">"),
        Column("CacheHit", 8, ">"),
        Column("Uptime", 7, ">"),
        Column("Pareto Frontier", 15),
        Column("Niche / Advantage", 32),
    ]
    rows = [
        [
            c.provider_name,
            f"${c.scored_cost_usd:.6f}",
            f"${c.token_cost_usd:.6f}",
            fmt_seconds(c.ttft_seconds),
            fmt_tps(c.throughput_tps),
            f"{c.h_used * 100.0:.1f}%",
            fmt_pct(c.uptime_pct),
            OPTIMAL if c.is_pareto else DOMINATED,
            c.niche if c.is_pareto else "--",
        ]
        for c in candidates
    ]
    print_table(
        cols, rows,
        title=f"Provider Pareto Frontier: {model_name}",
        subtitle_lines=[
            f"Turn: {cfg.prompt_tokens} prompt + {cfg.completion_tokens} completion tokens",
            f"Objectives: Cost ↓  Latency ↓  TPS ↑  CacheHit ↑  Uptime ↑  •  {filter_desc}",
        ],
        footer=f"{OPTIMAL} marks non-dominated providers: no other provider is at least as good on every objective and better on one.",
    )


# ------------------------------------------------------------------ catalog model frontier

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

    @property
    def effective_read_price(self) -> float:
        """Cache-read price, or the prompt price when the model has no cache."""
        return self.cache_read_price_per_m if self.cache_read_price_per_m is not None else self.prompt_price_per_m


MODEL_OBJECTIVES = [
    Objective(lambda c: c.turn_cost_usd, minimize=True),
    Objective(lambda c: c.context_length, minimize=False),
    Objective(lambda c: c.effective_read_price, minimize=True),
    Objective(lambda c: c.completion_price_per_m, minimize=True),
]


def compute_model_pareto(candidates: List[ModelCandidate]) -> List[ModelCandidate]:
    if not candidates:
        return candidates
    for c, optimal in zip(candidates, pareto_mask(candidates, MODEL_OBJECTIVES)):
        c.is_pareto = optimal

    min_cost = min(c.turn_cost_usd for c in candidates)
    max_ctx = max(c.context_length for c in candidates)
    min_read = min((c.cache_read_price_per_m for c in candidates if c.cache_read_price_per_m is not None), default=None)

    for c in candidates:
        if not c.is_pareto:
            continue
        traits = []
        if _near(c.turn_cost_usd, min_cost, 1e-6):
            traits.append("Cheapest Model")
        if c.context_length == max_ctx:
            traits.append(f"Max Context ({fmt_context(max_ctx)})")
        elif c.context_length >= 1_000_000:
            traits.append("1M Context")
        if min_read is not None and _near(c.cache_read_price_per_m, min_read, 1e-4):
            traits.append("Cheapest Cache Read")
        c.niche = " • ".join(traits) or "Cost/Context Trade-off"
    return candidates


def print_model_table(candidates: List[ModelCandidate], cfg: ScoringConfig, total: int) -> None:
    cols = [
        Column("Model ID", 34),
        Column("Turn Cost", 11, ">"),
        Column("Prompt $/M", 10, ">"),
        Column("Compl $/M", 10, ">"),
        Column("Read $/M", 9, ">"),
        Column("Context", 10, ">"),
        Column("Frontier", 12),
        Column("Niche / Advantage", 28),
    ]
    rows = [
        [
            c.model_id[:34],
            f"${c.turn_cost_usd:.6f}",
            f"${c.prompt_price_per_m:.4f}",
            f"${c.completion_price_per_m:.4f}",
            f"${c.cache_read_price_per_m:.4f}" if c.cache_read_price_per_m is not None else "--",
            fmt_context(c.context_length),
            OPTIMAL if c.is_pareto else DOMINATED,
            c.niche if c.is_pareto else "--",
        ]
        for c in candidates
    ]
    print_table(
        cols, rows,
        title="OpenRouter Catalog Model Pareto Frontier",
        subtitle_lines=[
            f"{sum(c.is_pareto for c in candidates)} Pareto-optimal of {total} priced models"
            f"  •  Turn: {cfg.prompt_tokens} prompt + {cfg.completion_tokens} completion tokens",
            "Objectives: Turn Cost ↓  Context ↑  Cache Read $/M ↓  Completion $/M ↓",
        ],
        footer=f"{OPTIMAL} marks models no other model beats on cost, context, and cache pricing simultaneously.",
    )


def build_model_candidates(cfg: ScoringConfig) -> List[ModelCandidate]:
    out = []
    for m in get_all_models():
        p = m.get("pricing", {})
        prompt = price_per_million(p.get("prompt")) or 0.0
        compl = price_per_million(p.get("completion")) or 0.0
        read = price_per_million(p.get("input_cache_read"))
        ctx = int(m.get("context_length") or 0)
        if prompt <= 0 or compl <= 0 or ctx <= 0:
            continue
        out.append(
            ModelCandidate(
                model_id=m["id"],
                model_name=m.get("name") or m["id"],
                prompt_price_per_m=prompt,
                completion_price_per_m=compl,
                cache_read_price_per_m=read,
                context_length=ctx,
                turn_cost_usd=(cfg.prompt_tokens * prompt + cfg.completion_tokens * compl) / 1e6,
            )
        )
    return out


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pareto frontier across the providers of one model, or across the whole catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", nargs="?", default=None, help="Model slug or shorthand (or 'models' for catalog mode)")
    parser.add_argument("--models", action="store_true", help="Catalog-wide model frontier instead of a provider frontier")
    parser.add_argument("-c", "--prompt-tokens", type=int, default=2000, help="Prompt tokens per turn (C)")
    parser.add_argument("-o", "--completion-tokens", type=int, default=500, help="Completion tokens per turn (O)")
    parser.add_argument("--quant", type=str, default="auto", help="Provider mode quantization filter: auto, all, or e.g. fp8")
    parser.add_argument("--all-quants", action="store_true", help="Shorthand for --quant all")
    parser.add_argument("--optimal-only", action="store_true", help="Show only Pareto-optimal candidates")
    parser.add_argument("--top", type=int, default=25, help="Catalog mode: dominated models to show after the frontier")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    cfg = ScoringConfig(prompt_tokens=args.prompt_tokens, completion_tokens=args.completion_tokens)

    if args.models or (args.model and args.model.lower() in ("models", "all", "catalog")):
        candidates = compute_model_pareto(build_model_candidates(cfg))
        total = len(candidates)
        candidates.sort(key=lambda c: (not c.is_pareto, c.turn_cost_usd))
        n_optimal = sum(c.is_pareto for c in candidates)
        shown = candidates[:n_optimal] if args.optimal_only else candidates[: n_optimal + max(0, args.top)]
        if args.json:
            print(json.dumps([asdict(c) for c in shown], indent=2))
        else:
            print_model_table(shown, cfg, total)
        return

    model_id, canonical_slug, display_name = resolve_model(args.model or "z-ai/glm-5.3-flash")
    scores = score_model_providers(canonical_slug, config=cfg)

    quant = "all" if args.all_quants else args.quant.lower()
    if quant == "auto":
        scores = filter_primary_quantization(scores)
        filter_desc = "Quantization: primary"
    elif quant == "all":
        filter_desc = "Quantization: ALL"
    else:
        scores = [s for s in scores if (s.quantization or "unknown").lower() == quant]
        filter_desc = f"Quantization: {quant.upper()}"

    candidates = [
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
            quantization=s.quantization or "unknown",
        )
        for s in scores
    ]
    candidates.sort(key=lambda c: c.scored_cost_usd)
    candidates = compute_provider_pareto(candidates)
    if args.optimal_only:
        candidates = [c for c in candidates if c.is_pareto]

    if args.json:
        print(json.dumps([asdict(c) for c in candidates], indent=2))
        return
    print_provider_table(candidates, f"{display_name} ({model_id})", cfg, filter_desc)


if __name__ == "__main__":
    main()
