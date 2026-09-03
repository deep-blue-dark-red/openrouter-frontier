#!/usr/bin/env python3
"""OpenRouter Pareto Frontier CLI Tool.

Fetches live benchmark scores (Intelligence, Coding, Agentic) and pricing from OpenRouter's API
and computes the cost vs. quality Pareto frontier (non-dominated models) with Knee Point detection.

Prices are resolved against OpenRouter's live catalog (/api/v1/models) to ensure
standard list pricing rather than batch or fallback rates.

High-performance architecture:
  - macOS IPv4 fast socket resolution (zero 10s TCP timeouts)
  - Persistent requests.Session with HTTP Keep-Alive and gzip compression
  - Concurrent background fetching of catalog and benchmark data
  - Multi-tier 1-hour disk caching (~0.04s warm)

Usage:
    ./pareto_frontier.py
    ./pareto_frontier.py --metric coding
    ./pareto_frontier.py --price-source weighted --top 20
    ./pareto_frontier.py --all-models
    ./pareto_frontier.py --format markdown
    ./pareto_frontier.py --format json
"""

import sys
import os
import glob
import time
import json
import socket
import argparse
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Ensure .venv dependencies are available regardless of invocation environment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

possible_venvs = [
    os.path.join(SCRIPT_DIR, ".venv/lib/python*/site-packages"),
    os.path.expanduser("~/git/openrouter-analytics/.venv/lib/python*/site-packages"),
]
for p in possible_venvs:
    matches = glob.glob(p)
    if matches:
        sys.path.insert(0, matches[0])
        break

# Force IPv4 socket resolution across all HTTP libraries on macOS
try:
    import urllib3.util.connection
    urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass

old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

import requests

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_BENCHMARKS_URL = "https://openrouter.ai/api/v1/benchmarks"
OPENROUTER_FRONTEND_BENCHMARKS_URL = "https://openrouter.ai/api/frontend/v1/rankings/benchmarks"

USER_AGENT = "dotOpenRouter-ParetoTool/1.0"
CACHE_DIR = Path.home() / ".cache" / "openrouter_analytics"
CACHE_TTL = 3600  # 1 hour

_session = requests.Session()
_session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
})


def strip_date_suffix(slug: str) -> str:
    """Strip snapshot date suffix (e.g. -20260831) to get base model ID."""
    parts = slug.split("-")
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 8:
        return "-".join(parts[:-1])
    return slug


def fetch_json(url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Fetch JSON with persistent session, gzip, and IPv4."""
    req_headers = {}
    if headers:
        req_headers.update(headers)
    resp = _session.get(url, headers=req_headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def load_data_concurrent(api_key: Optional[str] = None, force_refresh: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads catalog pricing and frontend benchmarks concurrently with disk caching."""
    models_cache = CACHE_DIR / "models.json"
    bench_cache = CACHE_DIR / "frontend_benchmarks.json"

    raw_models = None
    raw_bench = None

    if not force_refresh:
        now = time.time()
        if models_cache.exists():
            try:
                if (now - models_cache.stat().st_mtime) < CACHE_TTL:
                    with open(models_cache, "r", encoding="utf-8") as f:
                        raw_models = json.load(f)
            except Exception:
                pass

        if bench_cache.exists():
            try:
                if (now - bench_cache.stat().st_mtime) < CACHE_TTL:
                    with open(bench_cache, "r", encoding="utf-8") as f:
                        raw_bench = json.load(f)
            except Exception:
                pass

    # If either needs network fetch, fetch concurrently
    to_fetch_models = raw_models is None
    to_fetch_bench = raw_bench is None

    if to_fetch_models or to_fetch_bench:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_models = executor.submit(fetch_json, OPENROUTER_MODELS_URL) if to_fetch_models else None
            bench_url = OPENROUTER_FRONTEND_BENCHMARKS_URL
            fut_bench = executor.submit(fetch_json, bench_url) if to_fetch_bench else None

            if fut_models:
                try:
                    payload = fut_models.result()
                    raw_models = payload.get("data", [])
                    try:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        with open(models_cache, "w", encoding="utf-8") as f:
                            json.dump(raw_models, f)
                    except Exception:
                        pass
                except Exception as e:
                    if raw_models is None:
                        raise RuntimeError(f"Failed to fetch models: {e}")

            if fut_bench:
                try:
                    payload = fut_bench.result()
                    raw_bench = payload.get("data", {})
                    try:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        with open(bench_cache, "w", encoding="utf-8") as f:
                            json.dump(raw_bench, f)
                    except Exception:
                        pass
                except Exception as e:
                    if raw_bench is None:
                        raise RuntimeError(f"Failed to fetch benchmarks: {e}")

    # Build catalog dictionary
    catalog = {}
    for m in raw_models or []:
        m_id = m.get("id", "")
        pricing = m.get("pricing") or {}
        prompt_val = pricing.get("prompt")
        completion_val = pricing.get("completion")
        prompt_per_m = float(prompt_val) * 1_000_000 if prompt_val is not None else 0.0
        completion_per_m = float(completion_val) * 1_000_000 if completion_val is not None else 0.0

        catalog[m_id] = {
            "name": m.get("name", m_id),
            "canonical_slug": m.get("canonical_slug", m_id),
            "prompt_per_m": prompt_per_m,
            "completion_per_m": completion_per_m,
            "prompt_per_token": float(prompt_val) if prompt_val is not None else 0.0,
            "completion_per_token": float(completion_val) if completion_val is not None else 0.0,
        }

    return catalog, raw_bench or {}


def extract_benchmark_items(raw_bench: Dict[str, Any], metric: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    weighted_prices = raw_bench.get("weightedInputPrices", {})
    aa_data = raw_bench.get("aaData", {})

    cat_key = "intelligence"
    if metric in ("coding", "coding_index", "code"):
        cat_key = "coding"
    elif metric in ("agentic", "agentic_index", "agent"):
        cat_key = "agentic"

    raw_list = aa_data.get(cat_key, [])
    items = []
    for entry in raw_list:
        items.append({
            "model_permaslug": entry.get("permaslug") or entry.get("uid"),
            "display_name": entry.get("aa_name") or entry.get("uid"),
            "intelligence_index": entry.get("score") if cat_key == "intelligence" else None,
            "coding_index": entry.get("score") if cat_key == "coding" else None,
            "agentic_index": entry.get("score") if cat_key == "agentic" else None,
        })
    return items, weighted_prices


def compute_pareto(candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Compute Pareto frontier given items with 'cost' and 'score'.

    Objectives: minimize cost, maximize score.
    Returns: (frontier_items, knee_index)
    """
    if not candidates:
        return [], None

    # Sort primarily by cost ascending; break ties by score descending
    sorted_items = sorted(candidates, key=lambda x: (x["cost"], -x["score"]))

    frontier = []
    max_score = -float("inf")
    for item in sorted_items:
        if item["score"] > max_score:
            frontier.append(item)
            max_score = item["score"]

    knee_idx = None
    if len(frontier) >= 3:
        min_cost = frontier[0]["cost"]
        max_cost = frontier[-1]["cost"]
        min_score = frontier[0]["score"]
        max_score = frontier[-1]["score"]
        cost_range = max_cost - min_cost or 1.0
        score_range = max_score - min_score or 1.0

        best_dist = -float("inf")
        for i, item in enumerate(frontier):
            norm_x = (item["cost"] - min_cost) / cost_range
            norm_y = (item["score"] - min_score) / score_range
            gain = norm_y - norm_x
            if gain > best_dist:
                best_dist = gain
                knee_idx = i

    return frontier, knee_idx


def main():
    parser = argparse.ArgumentParser(
        description="Compute OpenRouter cost vs. quality Pareto frontier from live benchmark data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metric",
        choices=["intelligence", "coding", "agentic"],
        default="intelligence",
        help="Benchmark metric to evaluate (default: intelligence).",
    )
    parser.add_argument(
        "--price-source",
        choices=["list", "weighted", "call"],
        default="list",
        help=(
            "'list' = standard catalog prompt price ($/1M tokens); "
            "'weighted' = traffic-weighted effective prompt price from ClickHouse; "
            "'call' = estimated cost per call using --prompt-tokens and --completion-tokens."
        ),
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=1000,
        help="Prompt tokens for 'call' price source (default: 1000).",
    )
    parser.add_argument(
        "--completion-tokens",
        type=int,
        default=1000,
        help="Completion tokens for 'call' price source (default: 1000).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Filter to top N highest-scoring models before computing frontier.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "markdown", "json"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Include all evaluated models in output, tagging frontier models.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (defaults to $OPENROUTER_API_KEY env var).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh data from OpenRouter API.",
    )

    args = parser.parse_args()

    # Load catalog and benchmarks concurrently (with sub-second cache)
    catalog, raw_bench = load_data_concurrent(api_key=args.api_key, force_refresh=args.refresh)
    bench_items, weighted_prices = extract_benchmark_items(raw_bench, args.metric)

    score_key = f"{args.metric}_index"
    candidates = []
    seen_slugs = set()

    for item in bench_items:
        score = item.get(score_key)
        if score is None or score <= 0:
            continue

        permaslug = item.get("model_permaslug") or item.get("uid") or ""
        base_id = strip_date_suffix(permaslug)

        if base_id in seen_slugs:
            continue

        cat_entry = catalog.get(base_id) or catalog.get(permaslug)

        if args.price_source == "list":
            if not cat_entry or cat_entry["prompt_per_m"] <= 0:
                continue
            cost = cat_entry["prompt_per_m"]
            cost_unit = "$/1M prompt"
        elif args.price_source == "weighted":
            wp = weighted_prices.get(permaslug) or weighted_prices.get(base_id)
            if wp is not None and wp > 0:
                cost = wp
            elif cat_entry and cat_entry["prompt_per_m"] > 0:
                cost = cat_entry["prompt_per_m"]
            else:
                continue
            cost_unit = "$/1M weighted"
        elif args.price_source == "call":
            if not cat_entry:
                continue
            cost = (
                args.prompt_tokens * cat_entry["prompt_per_token"]
                + args.completion_tokens * cat_entry["completion_per_token"]
            )
            if cost <= 0:
                continue
            cost_unit = "$/call"

        display_name = item.get("display_name") or (cat_entry["name"] if cat_entry else base_id)

        candidates.append({
            "id": base_id,
            "permaslug": permaslug,
            "name": display_name,
            "score": float(score),
            "cost": float(cost),
            "cost_unit": cost_unit,
        })
        seen_slugs.add(base_id)

    if args.top is not None and args.top > 0:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:args.top]

    # Compute Pareto frontier
    frontier, knee_idx = compute_pareto(candidates)
    frontier_ids = {m["id"] for m in frontier}

    for item in candidates:
        item["on_frontier"] = item["id"] in frontier_ids
        item["is_knee"] = knee_idx is not None and item["id"] == frontier[knee_idx]["id"]

    # Calculate distance to frontier for dominated models
    if frontier:
        for item in candidates:
            if item["on_frontier"]:
                item["dist"] = 0.0
            else:
                better_scores = [f["score"] for f in frontier if f["cost"] <= item["cost"]]
                if better_scores:
                    item["dist"] = max(better_scores) - item["score"]
                else:
                    item["dist"] = 1.0

    output_list = candidates if args.all_models else frontier
    if args.all_models:
        output_list.sort(key=lambda x: (not x["on_frontier"], x["cost"] if x["on_frontier"] else x["dist"]))

    if args.format == "json":
        result = {
            "metric": args.metric,
            "price_source": args.price_source,
            "total_evaluated": len(candidates),
            "frontier_count": len(frontier),
            "models": output_list,
        }
        print(json.dumps(result, indent=2))
        return

    metric_header = f"{args.metric.capitalize()} Score"
    cost_header = f"Cost ({output_list[0]['cost_unit'] if output_list else 'USD'})"

    if args.format == "markdown":
        print(f"| Model | {metric_header} | {cost_header} | Frontier |")
        print("|---|---|---|---|")
        for m in output_list:
            knee_tag = " (knee)" if m.get("is_knee") else ""
            on_front = f"Yes{knee_tag}" if m["on_frontier"] else "No"
            cost_str = f"${m['cost']:.4f}" if args.price_source == "call" else f"${m['cost']:.2f}"
            print(f"| {m['name']} (`{m['id']}`) | {m['score']:.1f} | {cost_str} | {on_front} |")
    else:
        cols = [
            ("MODEL", 56, "<"),
            ("ID", 32, "<"),
            (metric_header, 12, ">"),
            (cost_header, 16, ">"),
            ("STATUS", 14, "<"),
        ]
        header_parts = [f"{name:>{w}}" if a == ">" else f"{name:<{w}}" for name, w, a in cols]
        header_str = "  ".join(header_parts)
        divider = "─" * len(header_str)

        print()
        print(divider)
        print(f"OpenRouter Pareto Frontier [{args.metric.upper()}] ({len(frontier)} non-dominated of {len(candidates)} models)")
        print(f"Price metric: {args.price_source.upper()} ({output_list[0]['cost_unit'] if output_list else ''})")
        print(divider)
        print(header_str)
        print(divider)

        for m in output_list:
            tag = "← KNEE" if m.get("is_knee") else ("ON FRONTIER" if m["on_frontier"] else "")
            cost_str = f"${m['cost']:<10.4f}" if args.price_source == "call" else f"${m['cost']:<10.2f}"
            row_vals = [
                m["name"][:56],
                m["id"][:32],
                f"{m['score']:.1f}",
                cost_str.strip(),
                tag,
            ]
            print("  ".join(f"{val:>{w}}" if a == ">" else f"{val:<{w}}" for val, (_, w, a) in zip(row_vals, cols)))

        print(divider)
        print("← KNEE indicates the optimal trade-off point with maximum quality score gain per dollar.\n")


if __name__ == "__main__":
    main()
