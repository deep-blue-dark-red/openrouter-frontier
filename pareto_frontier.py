#!/usr/bin/env python3
"""OpenRouter Pareto Frontier CLI Tool.

Fetches live benchmark scores (Intelligence, Coding, Agentic) and pricing from OpenRouter's API
and computes the cost vs. quality Pareto frontier (non-dominated models) with Knee Point detection.

Prices are resolved against OpenRouter's live catalog (/api/v1/models) to ensure
standard list pricing rather than batch or fallback rates.

Usage:
    ./pareto_frontier.py
    ./pareto_frontier.py --metric coding
    ./pareto_frontier.py --price-source weighted --top 20
    ./pareto_frontier.py --all-models
    ./pareto_frontier.py --format markdown
    ./pareto_frontier.py --format json
"""

import argparse
import json
import os
import sys
import time
import socket
import urllib.error
import urllib.request
from pathlib import Path

# Optimize connection on macOS (avoid IPv6 timeout)
try:
    import urllib3.util.connection
    urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass

# Patch socket.getaddrinfo to force IPv4
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_BENCHMARKS_URL = "https://openrouter.ai/api/v1/benchmarks"
OPENROUTER_FRONTEND_BENCHMARKS_URL = "https://openrouter.ai/api/frontend/v1/rankings/benchmarks"

USER_AGENT = "dotOpenRouter-ParetoTool/1.0"
CACHE_DIR = Path.home() / ".cache" / "openrouter_analytics"
CACHE_TTL = 3600  # 1 hour


def http_get(url: str, headers: dict | None = None) -> dict:
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        error_body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code} from {url}: {error_body}") from err
    except Exception as err:
        raise RuntimeError(f"Failed to fetch {url}: {err}") from err


def strip_date_suffix(slug: str) -> str:
    """Strip snapshot date suffix (e.g. -20260831) to get base model ID."""
    parts = slug.split("-")
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 8:
        return "-".join(parts[:-1])
    return slug


def load_catalog_pricing(force_refresh: bool = False) -> dict[str, dict]:
    """Fetch catalog pricing from /api/v1/models with disk caching."""
    cache_file = CACHE_DIR / "models.json"
    if not force_refresh and cache_file.exists():
        try:
            if (time.time() - cache_file.stat().st_mtime) < CACHE_TTL:
                with open(cache_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    catalog = {}
                    for m in raw_data:
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
                    return catalog
        except Exception:
            pass

    payload = http_get(OPENROUTER_MODELS_URL)
    raw_data = payload.get("data", [])
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(raw_data, f)
    except Exception:
        pass

    catalog = {}
    for m in raw_data:
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
    return catalog


def load_benchmarks(api_key: str | None, metric: str, force_refresh: bool = False) -> tuple[list[dict], dict[str, float]]:
    """Load benchmark data and optional weighted input prices with disk caching."""
    cache_file = CACHE_DIR / "frontend_benchmarks.json"
    fe_payload = None

    if not force_refresh and cache_file.exists():
        try:
            if (time.time() - cache_file.stat().st_mtime) < CACHE_TTL:
                with open(cache_file, "r", encoding="utf-8") as f:
                    fe_payload = json.load(f)
        except Exception:
            pass

    if fe_payload is None:
        try:
            fe_payload = http_get(OPENROUTER_FRONTEND_BENCHMARKS_URL).get("data", {})
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(fe_payload, f)
        except Exception:
            fe_payload = {}

    weighted_prices = fe_payload.get("weightedInputPrices", {})
    aa_data = fe_payload.get("aaData", {})

    items = []
    cat_key = "intelligence"
    if metric in ("coding", "coding_index", "code"):
        cat_key = "coding"
    elif metric in ("agentic", "agentic_index", "agent"):
        cat_key = "agentic"

    raw_list = aa_data.get(cat_key, [])
    for entry in raw_list:
        items.append({
            "model_permaslug": entry.get("permaslug") or entry.get("uid"),
            "display_name": entry.get("aa_name") or entry.get("uid"),
            "intelligence_index": entry.get("score") if cat_key == "intelligence" else None,
            "coding_index": entry.get("score") if cat_key == "coding" else None,
            "agentic_index": entry.get("score") if cat_key == "agentic" else None,
        })

    return items, weighted_prices


def compute_pareto(candidates: list[dict]) -> tuple[list[dict], int | None]:
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

    # Calculate knee point (best trade-off: max normalized distance above line between extremes)
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

    args = parser.parse_args()

    # 1. Fetch live catalog pricing (cached)
    catalog = load_catalog_pricing()

    # 2. Fetch live benchmark entries (cached)
    bench_items, weighted_prices = load_benchmarks(args.api_key, args.metric)

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
                # How much score is lost compared to closest cost on frontier
                better_scores = [f["score"] for f in frontier if f["cost"] <= item["cost"]]
                if better_scores:
                    item["dist"] = max(better_scores) - item["score"]
                else:
                    item["dist"] = 1.0

    output_list = candidates if args.all_models else frontier
    if args.all_models:
        # Sort: frontier first by cost, then dominated by distance to frontier
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
        divider = "─" * 125
        print()
        print(divider)
        print(f"OpenRouter Pareto Frontier [{args.metric.upper()}] ({len(frontier)} non-dominated of {len(candidates)} models)")
        print(f"Price metric: {args.price_source.upper()} ({output_list[0]['cost_unit'] if output_list else ''})")
        print(divider)
        print(f"{'MODEL':<56} {'ID':<30} {metric_header:>12} {cost_header:>16}  STATUS")
        print(divider)
        for m in output_list:
            tag = "← KNEE" if m.get("is_knee") else ("ON FRONTIER" if m["on_frontier"] else "")
            cost_str = f"${m['cost']:<10.4f}" if args.price_source == "call" else f"${m['cost']:<10.2f}"
            print(f"{m['name'][:56]:<56} {m['id'][:30]:<30} {m['score']:>12.1f} {cost_str:>16}  {tag}")
        print(divider)
        print("← KNEE indicates the optimal trade-off point with maximum quality score gain per dollar.\n")


if __name__ == "__main__":
    main()
