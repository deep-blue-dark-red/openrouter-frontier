#!/usr/bin/env python3
"""
get_models.py - Fast model catalog search and inspector for OpenRouter.

Query, inspect, and filter 400+ models from OpenRouter with punctuation-agnostic
fuzzy matching:
  - Supports 'zai' vs 'z.ai' vs 'z-ai'
  - Subsequence matching: 'glm53', 'claude37', 'gpt45'
  - Typo tolerance: 'flsh' -> 'flash'
  - Filter by prompt caching support, creator, or modality
  - Sort by prompt price, completion price, context length, or relevance

Usage:
  ./get_models.py [QUERY]                  # Search models (fuzzy) or inspect single model
  ./get_models.py zai                      # Matches all Z.ai models (same as 'z.ai' or 'z-ai')
  ./get_models.py -q flash                 # Keyword search
  ./get_models.py --creator zai            # Filter by maker
  ./get_models.py --caching                # Only models supporting prompt caching
  ./get_models.py --sort price             # Sort by prompt price
  ./get_models.py --sort context           # Sort by context window
  ./get_models.py --refresh                # Force refresh from OpenRouter API
  ./get_models.py --json                   # Output raw JSON
"""

import sys
import os
import re
import glob
import json
import socket
import difflib
import argparse
from typing import Optional, List, Dict, Any, Tuple

# Optimize socket resolution on macOS (avoid IPv6 timeout)
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

from openrouter_analytics.resolver import get_all_models, resolve_model


def clean_str(s: str) -> str:
    """Removes all non-alphanumeric characters and lowercases the string."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def calculate_match_score(query: str, model_id: str, model_name: str) -> float:
    """
    Calculates fuzzy relevance score (0.0 to 1.0) between query and a model.
    Punctuation/delimiter-agnostic: 'zai' matches 'z.ai' and 'z-ai'.
    """
    if not query:
        return 1.0

    q_clean = clean_str(query)
    if not q_clean:
        return 0.0

    m_id_clean = clean_str(model_id)
    m_name_clean = clean_str(model_name)

    short_id = model_id.split("/")[-1] if "/" in model_id else model_id
    short_clean = clean_str(short_id)

    creator = model_id.split("/")[0] if "/" in model_id else ""
    creator_clean = clean_str(creator)

    # 1. Exact clean match on ID or short slug
    if q_clean == m_id_clean or q_clean == short_clean:
        return 1.0

    # 2. Exact match on creator (e.g. 'zai' or 'z.ai' matching 'z-ai')
    if q_clean == creator_clean:
        return 0.95

    # 3. Clean alphanumeric prefix match
    if short_clean.startswith(q_clean) or m_id_clean.startswith(q_clean) or creator_clean.startswith(q_clean):
        return 0.90

    # 4. Clean alphanumeric substring match (e.g. 'zai' in 'zaiglm53flash' or 'glm53')
    if q_clean in m_id_clean or q_clean in m_name_clean or q_clean in short_clean or q_clean in creator_clean:
        return 0.85

    # 5. Fuzzy ratio on clean query vs short slug, ID, or creator
    r1 = difflib.SequenceMatcher(None, q_clean, short_clean).ratio()
    r2 = difflib.SequenceMatcher(None, q_clean, m_id_clean).ratio()
    r3 = difflib.SequenceMatcher(None, q_clean, creator_clean).ratio()
    max_r = max(r1, r2, r3)
    if max_r >= 0.70:
        return max_r * 0.8

    return 0.0


def format_price(val: Any) -> str:
    if val is None:
        return "--"
    try:
        f = float(val)
        if f <= 0:
            return "Free"
        p_m = f * 1_000_000.0 if f < 0.01 else f
        return f"${p_m:.4f}"
    except (ValueError, TypeError):
        return "--"


def print_single_model(m: Dict[str, Any]):
    p = m.get("pricing", {})
    arch = m.get("architecture", {})
    top_p = m.get("top_provider", {})
    ctx = m.get("context_length", 0)

    divider = "─" * 80
    print()
    print(divider)
    print(f"Model: {m.get('name') or m.get('id')}")
    print(f"ID:    {m.get('id')}")
    if m.get("canonical_slug") and m.get("canonical_slug") != m.get("id"):
        print(f"Slug:  {m.get('canonical_slug')}")
    print(divider)

    print(f"  Context Window:        {ctx:,} tokens" if ctx else "  Context Window:        --")
    if top_p.get("max_completion_tokens"):
        print(f"  Max Output Tokens:     {top_p['max_completion_tokens']:,}")
    if arch.get("modality"):
        print(f"  Modality:              {arch['modality']}")
    if arch.get("instruct_type"):
        print(f"  Instruct Format:       {arch['instruct_type']}")

    print("\n  Pricing (USD per million tokens):")
    print(f"    Prompt (Input):      {format_price(p.get('prompt'))} / M")
    print(f"    Completion (Output): {format_price(p.get('completion'))} / M")
    print(f"    Cache Read:          {format_price(p.get('input_cache_read'))} / M")
    print(f"    Cache Write:         {format_price(p.get('input_cache_write'))} / M")
    if p.get("request"):
        print(f"    Request Fee:         ${float(p['request']):.6f}")
    if p.get("discount"):
        print(f"    Discount:            {float(p['discount']) * 100:.0f}% off")

    desc = m.get("description", "").strip()
    if desc:
        print("\n  Description:")
        for line in desc.split("\n")[:4]:
            if line.strip():
                print(f"    {line.strip()[:90]}")

    print(divider)
    print(f"Tip: run `./score_providers.py {m['id']}` to evaluate provider utility.")
    print(f"Tip: run `./provider_frontier.py {m['id']}` to view the Pareto frontier.\n")


def print_models_table(models: List[Dict[str, Any]], title_suffix: str = ""):
    cols = [
        ("Model ID", 34, "<"),
        ("Model Name", 28, "<"),
        ("Prompt $/M", 11, ">"),
        ("Compl $/M", 11, ">"),
        ("Read $/M", 10, ">"),
        ("Context", 10, ">"),
        ("Modality", 12, "<"),
    ]

    header_parts = [f"{name:>{w}}" if a == ">" else f"{name:<{w}}" for name, w, a in cols]
    header_line = "  ".join(header_parts)
    divider = "─" * len(header_line)

    print()
    print(divider)
    print(f"OpenRouter Models ({len(models)} models){title_suffix}")
    print(divider)
    print(header_line)
    print(divider)

    for m in models:
        p = m.get("pricing", {})
        arch = m.get("architecture", {})
        ctx = m.get("context_length", 0)
        ctx_str = f"{ctx // 1000}k" if ctx >= 1000 else (str(ctx) if ctx else "--")
        mod_str = arch.get("modality") or "text"

        row_vals = [
            m.get("id", "")[:34],
            (m.get("name") or m.get("id", ""))[:28],
            format_price(p.get("prompt")),
            format_price(p.get("completion")),
            format_price(p.get("input_cache_read")),
            ctx_str,
            mod_str[:12],
        ]

        print("  ".join(f"{val:>{w}}" if a == ">" else f"{val:<{w}}" for val, (_, w, a) in zip(row_vals, cols)))

    print(divider)
    print("Tip: pass model ID to inspect details (e.g. `./get_models.py z-ai/glm-5.3-flash`).\n")


def main():
    parser = argparse.ArgumentParser(
        description="Query and inspect models from OpenRouter catalog with fuzzy matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", nargs="?", default=None, help="Model ID, search keyword, or creator slug (e.g. zai, z.ai, glm-5.3)")
    parser.add_argument("-q", "--search", type=str, default="", help="Keyword search across model ID and name")
    parser.add_argument("-m", "--creator", type=str, default="", help="Filter by creator (e.g. zai, z.ai, anthropic, openai)")
    parser.add_argument("--caching", action="store_true", help="Show only models supporting prompt cache read")
    parser.add_argument("--modality", type=str, default="", help="Filter by modality (e.g. text->text, multimodal)")
    parser.add_argument(
        "--sort",
        choices=["relevance", "name", "price", "completion", "context", "created"],
        default="relevance",
        help="Sort order",
    )
    parser.add_argument("-n", "--top", type=int, default=30, help="Max models to display")
    parser.add_argument("--refresh", action="store_true", help="Force refresh models from OpenRouter API")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args()

    raw_models = get_all_models(force_refresh=args.refresh)

    search_query = (args.query or args.search).strip()

    # Exact single model lookup if requested by exact slug/id
    if args.query and not args.search:
        clean_q = clean_str(args.query)
        exact_matches = [
            m for m in raw_models
            if clean_str(m.get("id", "")) == clean_q or clean_str(m.get("canonical_slug", "")) == clean_q
        ]
        if len(exact_matches) == 1 and ("/" in args.query or len(exact_matches[0].get("id", "").split("/")) == 1):
            if args.json:
                print(json.dumps(exact_matches[0], indent=2))
            else:
                print_single_model(exact_matches[0])
            return

    # Filter models and calculate scores
    scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

    creator_clean = clean_str(args.creator) if args.creator else ""

    for m in raw_models:
        m_id = m.get("id", "")
        m_name = m.get("name") or m_id
        m_creator = m_id.split("/")[0] if "/" in m_id else ""

        if creator_clean:
            if creator_clean not in clean_str(m_creator) and creator_clean not in clean_str(m_name):
                continue

        p = m.get("pricing", {})
        if args.caching:
            if not p.get("input_cache_read"):
                continue

        if args.modality:
            mod = (m.get("architecture", {}).get("modality") or "").lower()
            if args.modality.lower() not in mod:
                continue

        score = calculate_match_score(search_query, m_id, m_name)
        if search_query and score <= 0.0:
            continue

        scored_candidates.append((score, m))

    # Sort candidates
    def sort_key(item: Tuple[float, Dict[str, Any]]):
        score, m = item
        p = m.get("pricing", {})
        if args.sort == "price":
            raw = float(p.get("prompt") or 0.0)
            return raw * 1e6 if raw < 0.01 else raw
        elif args.sort == "completion":
            raw = float(p.get("completion") or 0.0)
            return raw * 1e6 if raw < 0.01 else raw
        elif args.sort == "context":
            return -int(m.get("context_length") or 0)
        elif args.sort == "created":
            return -int(m.get("created") or 0)
        elif args.sort == "name":
            return (m.get("name") or m.get("id", "")).lower()
        # Default: relevance score first (higher is better), then name
        return (-score, (m.get("name") or m.get("id", "")).lower())

    scored_candidates.sort(key=sort_key)
    filtered = [m for _, m in scored_candidates]

    if args.json:
        print(json.dumps(filtered[:args.top], indent=2))
        return

    # If exactly 1 result matched a specific query, print detailed card
    if len(filtered) == 1 and search_query:
        print_single_model(filtered[0])
        return

    title_suffix = ""
    if search_query:
        title_suffix += f" matching '{search_query}'"
    if args.creator:
        title_suffix += f" by '{args.creator}'"
    if args.caching:
        title_suffix += " with Prompt Caching"
    if args.sort not in ("relevance", "name"):
        title_suffix += f" (sorted by {args.sort})"

    print_models_table(filtered[:args.top], title_suffix=title_suffix)


if __name__ == "__main__":
    main()
