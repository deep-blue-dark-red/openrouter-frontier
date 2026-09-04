#!/usr/bin/env python3
"""model_router.py - pick the best model and provider for a required intelligence level.

Given a minimum Artificial Analysis score (the same index ``model_frontier.py`` uses), the
router picks a benchmarked model that meets it and that model's best provider as ranked by
the ProviderScore scorer. Two modes choose the model:

  cheapest   the cheapest model at or above the threshold
  efficient  most quality gained per dollar once the level is met: the efficient point (maximum
             gradient) of the cost/quality frontier built from models at or above it

Models with no active provider are skipped in favour of the next best qualifying one.

Usage:
  ./model_router.py 60
  ./model_router.py 60 --mode efficient
  ./model_router.py 45 --metric coding --time-value 30
  ./model_router.py 70 --json
"""

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import _bootstrap  # noqa: F401

from model_frontier import build_candidates, load_data
from openrouter_frontier._util import filter_primary_quantization
from openrouter_frontier.client import score_model_providers
from openrouter_frontier.pareto import annotate_frontier, cost_quality_frontier, frontier_sort_key
from openrouter_frontier.profile_args import add_task_args, config_from_args
from openrouter_frontier.scoring import ScoreBreakdown, ScoringConfig


@dataclass
class Route:
    """The chosen model and provider for one intelligence requirement."""

    model_id: str
    model_name: str
    permaslug: str
    metric: str
    mode: str
    score: float
    model_cost: float          # model-level price used for ranking, see ``cost_unit``
    cost_unit: str
    provider: ScoreBreakdown   # best endpoint by ProviderScore expected task cost
    skipped: List[str] = field(default_factory=list)  # cheaper qualifying models with no active provider

    def to_dict(self) -> Dict[str, Any]:
        p = self.provider
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "permaslug": self.permaslug,
            "metric": self.metric,
            "mode": self.mode,
            "score": self.score,
            "model_cost": self.model_cost,
            "cost_unit": self.cost_unit,
            "provider": p.provider_name,
            "provider_slug": p.provider_slug,
            "endpoint_id": p.endpoint_id,
            "task_cost_usd": p.task_cost_usd,
            "objective_usd": p.objective_usd,
            "token_cost_usd": p.token_cost_usd,
            "task_cost_per_m": p.task_cost_per_m,
            "quantization": p.quantization,
            "uptime_pct": p.uptime_pct,
            "ttft_seconds": p.ttft_seconds,
            "throughput_tps": p.throughput_tps,
            "skipped": self.skipped,
        }


MODES = ("cheapest", "efficient")


def rank_candidates(qualifying: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """Order qualifying models by preference for ``mode``; the first with a provider wins.

    ``cheapest``: ascending price, higher score first on ties.
    ``efficient``: the efficient point of the frontier over the qualifying models, then the remaining
    frontier models by price, then dominated models by their gap to the frontier. With
    fewer than three frontier points there is no efficient point and the order equals ``cheapest``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    ordered = sorted(qualifying, key=lambda c: (c["cost"], -c["score"]))
    if mode == "cheapest":
        return ordered
    frontier, efficient_idx = cost_quality_frontier(ordered)
    annotate_frontier(ordered, frontier, efficient_idx)
    ordered.sort(key=frontier_sort_key)
    return sorted(ordered, key=lambda c: not c["is_efficient"])  # stable: efficient point first, rest unchanged


def route(
    min_score: float,
    metric: str = "intelligence",
    mode: str = "cheapest",
    price_source: str = "list",
    config: Optional[ScoringConfig] = None,
    all_quants: bool = False,
    force_refresh: bool = False,
) -> Optional[Route]:
    """Return the preferred model scoring at least ``min_score`` and its best provider.

    :param min_score: Required Artificial Analysis score on ``metric`` (0-100 scale).
    :param metric: ``intelligence``, ``coding`` or ``agentic``.
    :param mode: ``cheapest`` or ``efficient``; see :func:`rank_candidates`.
    :param price_source: How models are priced when ranking them; see ``model_frontier.py``.
    :param config: ProviderScore knobs used to rank the chosen model's providers.
    :param all_quants: Consider non-primary quantization variants when picking the provider.
    :returns: ``None`` when no benchmarked model meets the score.
    """
    cfg = config or ScoringConfig()
    catalog, raw_bench = load_data(force_refresh)
    candidates = build_candidates(catalog, raw_bench, metric, price_source, cfg.prompt_tokens, cfg.completion_per_turn)
    qualifying = rank_candidates([c for c in candidates if c["score"] >= min_score], mode)

    skipped: List[str] = []
    for m in qualifying:
        providers = filter_primary_quantization(score_model_providers(m["permaslug"], config=cfg), all_quants)
        if not providers:
            skipped.append(m["id"])
            continue
        return Route(
            model_id=m["id"], model_name=m["name"], permaslug=m["permaslug"], metric=m["metric"],
            mode=mode, score=m["score"], model_cost=m["cost"], cost_unit=m["cost_unit"],
            provider=providers[0], skipped=skipped,
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cheapest model meeting an Artificial Analysis score, and its best provider.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("min_score", type=float, help="Required benchmark score (0-100)")
    parser.add_argument("--metric", choices=["intelligence", "coding", "agentic"], default="intelligence")
    parser.add_argument(
        "--mode", choices=MODES, default="cheapest",
        help="cheapest = cheapest model meeting the score; efficient = most quality per dollar among models meeting it",
    )
    parser.add_argument(
        "--price-source", choices=["list", "weighted", "call"], default="list",
        help="list = catalog prompt $/1M; weighted = traffic-weighted effective prompt $/1M; call = per-turn estimate",
    )
    add_task_args(parser)
    parser.add_argument("--all-quants", action="store_true", help="Consider non-primary quantization variants")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--refresh", action="store_true", help="Bypass the 1-hour cache")
    args = parser.parse_args()

    cfg = config_from_args(args)
    r = route(args.min_score, args.metric, args.mode, args.price_source, cfg, args.all_quants, args.refresh)

    if args.json:
        print(json.dumps(r.to_dict() if r else None, indent=2))
        return
    if r is None:
        raise SystemExit(f"No benchmarked model scores >= {args.min_score:g} on {args.metric}.")

    p = r.provider
    print(f"Model:     {r.model_name} ({r.model_id})")
    print(f"Score:     {r.score:.1f} {r.metric}  (required >= {args.min_score:g}, mode: {r.mode})")
    print(f"Price:     ${r.model_cost:.4f} {r.cost_unit}")
    print(f"Provider:  {p.provider_name} [{p.quantization}]  {p.formatted_task_cost} per task "
          f"({p.formatted_token_cost} tokens + {p.formatted_time_cost} time + {p.formatted_failure_cost} failures; "
          f"{p.turns} turns, routing {p.routing})")
    if r.skipped:
        print(f"Skipped:   {', '.join(r.skipped)} (no active providers)")


if __name__ == "__main__":
    main()
