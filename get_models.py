#!/usr/bin/env python3
"""
get_models.py - Fast model catalog search and inspector for OpenRouter.

Query, inspect, and filter 400+ models from OpenRouter:
  - Search by name, slug, or maker
  - Filter by prompt caching support, creator, or modality
  - Sort by prompt price, completion price, or context length
  - Inspect full pricing and architecture specs for any individual model

Usage:
  ./get_models.py [QUERY]                  # Search models or inspect single model
  ./get_models.py -q flash                 # Keyword search
  ./get_models.py --creator anthropic      # Filter by maker
  ./get_models.py --caching                # Only models supporting prompt caching
  ./get_models.py --sort price             # Sort by prompt price
  ./get_models.py --sort context           # Sort by context window
  ./get_models.py --refresh                # Force refresh from OpenRouter API
  ./get_models.py --json                   # Output raw JSON
"""

import sys
import os
import glob
import json
import socket
import argparse
from typing import Optional, List, Dict, Any

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
        description="Query and inspect models from OpenRouter catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", nargs="?", default=None, help="Model ID, search keyword, or creator slug")
    parser.add_argument("-q", "--search", type=str, default="", help="Keyword search across model ID and name")
    parser.add_argument("-m", "--creator", type=str, default="", help="Filter by creator (e.g. anthropic, openai, meta, z-ai)")
    parser.add_argument("--caching", action="store_true", help="Show only models supporting prompt cache read")
    parser.add_argument("--modality", type=str, default="", help="Filter by modality (e.g. text->text, multimodal)")
    parser.add_argument("--sort", choices=["name", "price", "completion", "context", "created"], default="name", help="Sort order")
    parser.add_argument("-n", "--top", type=int, default=30, help="Max models to display")
    parser.add_argument("--refresh", action="store_true", help="Force refresh models from OpenRouter API")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args()

    raw_models = get_all_models(force_refresh=args.refresh)

    search_term = (args.query or args.search).strip().lower()

    # If exact single model match requested, inspect directly
    if args.query and not args.search:
        exact_matches = [m for m in raw_models if m.get("id", "").lower() == search_term or m.get("canonical_slug", "").lower() == search_term]
        if exact_matches:
            if args.json:
                print(json.dumps(exact_matches[0], indent=2))
            else:
                print_single_model(exact_matches[0])
            return

    # Filter models
    filtered = []
    for m in raw_models:
        m_id = m.get("id", "").lower()
        m_name = (m.get("name") or "").lower()

        if search_term:
            if search_term not in m_id and search_term not in m_name:
                continue

        if args.creator:
            creator = args.creator.lower().rstrip("/")
            if not m_id.startswith(creator + "/") and creator not in m_name:
                continue

        p = m.get("pricing", {})
        if args.caching:
            if not p.get("input_cache_read"):
                continue

        if args.modality:
            mod = (m.get("architecture", {}).get("modality") or "").lower()
            if args.modality.lower() not in mod:
                continue

        filtered.append(m)

    # Sort
    def sort_key(m):
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
        return (m.get("name") or m.get("id", "")).lower()

    filtered.sort(key=sort_key)

    if args.json:
        print(json.dumps(filtered[:args.top], indent=2))
        return

    # If only 1 matched, print details
    if len(filtered) == 1 and search_term:
        print_single_model(filtered[0])
        return

    title_suffix = ""
    if search_term:
        title_suffix += f" matching '{search_term}'"
    if args.creator:
        title_suffix += f" by '{args.creator}'"
    if args.caching:
        title_suffix += " with Prompt Caching"
    if args.sort != "name":
        title_suffix += f" (sorted by {args.sort})"

    print_models_table(filtered[:args.top], title_suffix=title_suffix)


if __name__ == "__main__":
    main()
