#!/usr/bin/env python3
"""model_frontier.py - cost vs. quality Pareto frontier across OpenRouter's benchmarked models.

Quality comes from OpenRouter's published Artificial Analysis indices (intelligence, coding,
or agentic); cost from the live catalog list price, the traffic-weighted effective price, or an
estimated per-call cost. Models are on the frontier when no cheaper model scores as high. The
efficient point is the frontier model with the best normalised quality gain per dollar.

Usage:
  ./model_frontier.py
  ./model_frontier.py --metric coding --top 40
  ./model_frontier.py --price-source call -c 2000 -o 500
  ./model_frontier.py --all-models --format markdown
  ./model_frontier.py --format json
"""

import argparse
import concurrent.futures
import json
from typing import Any, Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401

import requests

from openrouter_analytics._util import CACHE_DIR, force_ipv4, load_json_cache, save_json_cache
from openrouter_analytics.pareto import annotate_frontier, cost_quality_frontier, frontier_sort_key
from openrouter_analytics.render import Column, print_table
from openrouter_analytics.resolver import MODELS_CACHE_FILE, get_all_models

force_ipv4()

BENCHMARKS_URL = "https://openrouter.ai/api/frontend/v1/rankings/benchmarks"
BENCHMARKS_CACHE_FILE = CACHE_DIR / "frontend_benchmarks.json"
CACHE_TTL = 3600  # benchmark rankings change rarely

METRIC_ALIASES = {
    "intelligence": "intelligence", "intel": "intelligence",
    "coding": "coding", "code": "coding",
    "agentic": "agentic", "agent": "agentic",
}

_session = requests.Session()
_session.headers.update({"User-Agent": "openrouter-analytics-python", "Accept-Encoding": "gzip, deflate"})


def strip_date_suffix(slug: str) -> str:
    """'z-ai/glm-5.3-flash-20260826' -> 'z-ai/glm-5.3-flash'."""
    base, _, tail = slug.rpartition("-")
    return base if base and tail.isdigit() and len(tail) == 8 else slug


def _fetch_benchmarks() -> Dict[str, Any]:
    resp = _session.get(BENCHMARKS_URL, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", {})


def load_data(force_refresh: bool = False) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Return ``(catalog_by_model_id, raw_benchmarks)``, fetching both concurrently when stale."""
    raw_bench = None if force_refresh else load_json_cache(BENCHMARKS_CACHE_FILE, CACHE_TTL)
    need_models = force_refresh or load_json_cache(MODELS_CACHE_FILE, CACHE_TTL) is None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_models = pool.submit(get_all_models, need_models)
        fut_bench = pool.submit(_fetch_benchmarks) if raw_bench is None else None
        raw_models = fut_models.result()
        if fut_bench is not None:
            try:
                raw_bench = fut_bench.result()
                save_json_cache(BENCHMARKS_CACHE_FILE, raw_bench)
            except Exception as e:
                raw_bench = load_json_cache(BENCHMARKS_CACHE_FILE, ttl=10**12)
                if raw_bench is None:
                    raise RuntimeError(f"Failed to fetch benchmarks: {e}")

    catalog: Dict[str, Dict[str, Any]] = {}
    for m in raw_models:
        pricing = m.get("pricing") or {}
        prompt_tok = float(pricing.get("prompt") or 0.0)
        compl_tok = float(pricing.get("completion") or 0.0)
        catalog[m.get("id", "")] = {
            "name": m.get("name", m.get("id")),
            "canonical_slug": m.get("canonical_slug", m.get("id")),
            "prompt_per_m": prompt_tok * 1_000_000,
            "completion_per_m": compl_tok * 1_000_000,
            "prompt_per_token": prompt_tok,
            "completion_per_token": compl_tok,
        }
    return catalog, raw_bench or {}


def extract_benchmark_items(raw_bench: Dict[str, Any], metric: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Return ``(items, weighted_input_prices)`` for one metric. Each item has ``score``."""
    metric = METRIC_ALIASES.get(metric, "intelligence")
    items = [
        {
            "model_permaslug": e.get("permaslug") or e.get("uid"),
            "display_name": e.get("aa_name") or e.get("uid"),
            "score": e.get("score"),
        }
        for e in raw_bench.get("aaData", {}).get(metric, [])
    ]
    return items, raw_bench.get("weightedInputPrices", {})


def build_candidates(
    catalog: Dict[str, Dict[str, Any]],
    raw_bench: Dict[str, Any],
    metric: str,
    price_source: str = "list",
    prompt_tokens: int = 1000,
    completion_tokens: int = 1000,
) -> List[Dict[str, Any]]:
    """Join benchmark scores with prices and tag each candidate with frontier/efficient/dist."""
    items, weighted_prices = extract_benchmark_items(raw_bench, metric)
    candidates: List[Dict[str, Any]] = []
    seen = set()

    for item in items:
        score = item.get("score")
        if score is None or score <= 0:
            continue
        permaslug = item.get("model_permaslug") or ""
        base_id = strip_date_suffix(permaslug)
        if base_id in seen:
            continue
        cat = catalog.get(base_id) or catalog.get(permaslug)

        cost: Optional[float] = None
        if price_source == "list":
            if cat and cat["prompt_per_m"] > 0:
                cost, unit = cat["prompt_per_m"], "$/1M prompt"
        elif price_source == "weighted":
            wp = weighted_prices.get(permaslug) or weighted_prices.get(base_id)
            if wp is not None and wp > 0:
                cost, unit = wp, "$/1M weighted"
            elif cat and cat["prompt_per_m"] > 0:
                cost, unit = cat["prompt_per_m"], "$/1M weighted"
        elif price_source == "call" and cat:
            c = prompt_tokens * cat["prompt_per_token"] + completion_tokens * cat["completion_per_token"]
            if c > 0:
                cost, unit = c, "$/call"
        if cost is None:
            continue

        candidates.append({
            "id": base_id,
            "permaslug": permaslug,
            "name": item.get("display_name") or (cat["name"] if cat else base_id),
            "score": float(score),
            "cost": float(cost),
            "cost_unit": unit,
            "prompt_p": cat["prompt_per_m"] if cat else None,
            "compl_p": cat["completion_per_m"] if cat else None,
            "metric": metric,
        })
        seen.add(base_id)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cost vs. quality Pareto frontier from OpenRouter's live benchmark rankings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metric", choices=["intelligence", "coding", "agentic"], default="intelligence")
    parser.add_argument(
        "--price-source", choices=["list", "weighted", "call"], default="list",
        help="list = catalog prompt $/1M; weighted = traffic-weighted effective prompt $/1M; call = per-call estimate",
    )
    parser.add_argument("-c", "--prompt-tokens", type=int, default=1000, help="Prompt tokens for --price-source call")
    parser.add_argument("-o", "--completion-tokens", type=int, default=1000, help="Completion tokens for --price-source call")
    parser.add_argument("--top", type=int, default=None, help="Keep only the N highest-scoring models before computing the frontier")
    parser.add_argument("--format", choices=["table", "markdown", "json"], default="table")
    parser.add_argument("--all-models", action="store_true", help="Show dominated models too, sorted by distance to the frontier")
    parser.add_argument("--refresh", action="store_true", help="Bypass the 1-hour cache")
    args = parser.parse_args()

    catalog, raw_bench = load_data(force_refresh=args.refresh)
    candidates = build_candidates(catalog, raw_bench, args.metric, args.price_source, args.prompt_tokens, args.completion_tokens)

    if args.top:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[: args.top]

    frontier, efficient_idx = cost_quality_frontier(candidates)
    annotate_frontier(candidates, frontier, efficient_idx)
    output = sorted(candidates, key=frontier_sort_key) if args.all_models else frontier

    if args.format == "json":
        print(json.dumps({
            "metric": args.metric,
            "price_source": args.price_source,
            "total_evaluated": len(candidates),
            "frontier_count": len(frontier),
            "models": output,
        }, indent=2))
        return

    unit = output[0]["cost_unit"] if output else "USD"
    digits = 4 if args.price_source == "call" else 2
    metric_header = f"{args.metric.capitalize()} Score"

    if args.format == "markdown":
        print(f"| Model | {metric_header} | Cost ({unit}) | Frontier |")
        print("|---|---|---|---|")
        for m in output:
            status = ("Yes (efficient point)" if m["is_efficient"] else "Yes") if m["on_frontier"] else "No"
            print(f"| {m['name']} (`{m['id']}`) | {m['score']:.1f} | ${m['cost']:.{digits}f} | {status} |")
        return

    cols = [Column("Model", 56), Column("ID", 32), Column(metric_header, 14, ">"), Column(f"Cost ({unit})", 16, ">"), Column("Status", 14)]
    rows = [
        [
            m["name"][:56],
            m["id"][:32],
            f"{m['score']:.1f}",
            f"${m['cost']:.{digits}f}",
            "← EFFICIENT" if m["is_efficient"] else ("ON FRONTIER" if m["on_frontier"] else f"-{m['dist']:.1f} pts"),
        ]
        for m in output
    ]
    print_table(
        cols, rows,
        title=f"OpenRouter Cost vs. {args.metric.capitalize()} Pareto Frontier",
        subtitle_lines=[f"{len(frontier)} non-dominated of {len(candidates)} benchmarked models  •  price source: {args.price_source}"],
        footer="← EFFICIENT is the frontier point with the largest normalised score gain per dollar; '-N pts' is a dominated model's gap to the frontier.",
    )


if __name__ == "__main__":
    main()
