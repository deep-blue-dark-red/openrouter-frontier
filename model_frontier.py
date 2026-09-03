#!/usr/bin/env python3
"""
model_frontier.py - Catalog-Wide Model Pareto Frontier for OpenRouter.

Evaluates and identifies Pareto-optimal AI models across OpenRouter's catalog, balancing:
  - Turn Cost in USD per turn (lower is better, given C prompt + O completion tokens)
  - Context Window Length (higher is better)
  - Prompt Price ($/M tokens, lower is better)
  - Completion Price ($/M tokens, lower is better)
  - Cache Read Price ($/M tokens, lower is better)

A model is Pareto-optimal if no other model is strictly cheaper with equal/larger context and better pricing.

Usage:
  ./model_frontier.py [--prompt-tokens C] [--completion-tokens O] [--optimal-only] [-n TOP] [-q QUERY]
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

from openrouter_analytics.resolver import get_all_models


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
    """Computes Pareto frontier across all models."""
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


def print_model_frontier_table(candidates: List[ModelCandidate], C: int, O: int):
    """Prints a clean, modern table using connected dashes and space separation (no vertical bars, no ===)."""
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

    pareto_count = sum(1 for c in candidates if c.is_pareto)
    banner_info = (
        f"Multi-Objective Pareto Analysis  •  Turn: {C} prompt + {O} completion tokens\n"
        f"Evaluation: Turn Cost vs Context Length vs Cache Read Pricing  •  {pareto_count} Pareto-Optimal Models"
    )

    print()
    print(divider)
    print("OpenRouter Catalog-Wide Model Pareto Frontier")
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute Pareto frontier across all models in OpenRouter's catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--prompt-tokens", type=int, default=2000, help="Prompt context tokens per turn (C)")
    parser.add_argument("-o", "--completion-tokens", type=int, default=500, help="Completion tokens per turn (O)")
    parser.add_argument("--optimal-only", action="store_true", help="Display only Pareto-optimal models")
    parser.add_argument("-n", "--top", type=int, default=25, help="Number of rows to display")
    parser.add_argument("-q", "--query", type=str, default="", help="Filter models by keyword or maker")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args()

    raw_models = get_all_models()
    candidates: List[ModelCandidate] = []

    for m in raw_models:
        m_id = m.get("id", "")
        if args.query and args.query.lower() not in m_id.lower() and args.query.lower() not in (m.get("name") or "").lower():
            continue

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
                model_id=m_id,
                model_name=m.get("name") or m_id,
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
        display_candidates = [c for c in candidates if c.is_pareto]
    else:
        pareto_set = [c for c in candidates if c.is_pareto]
        other_set = [c for c in candidates if not c.is_pareto][: max(0, args.top - len(pareto_set))]
        display_candidates = pareto_set + other_set
        display_candidates.sort(key=lambda c: (not c.is_pareto, c.turn_cost_usd))

    if args.json:
        print(json.dumps([c.to_dict() for c in display_candidates[:args.top]], indent=2))
        return

    print_model_frontier_table(display_candidates[:args.top], args.prompt_tokens, args.completion_tokens)


if __name__ == "__main__":
    main()
