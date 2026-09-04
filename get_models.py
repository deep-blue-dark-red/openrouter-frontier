#!/usr/bin/env python3
"""get_models.py - search and inspect the OpenRouter model catalog.

Matching is punctuation-agnostic ('zai' == 'z.ai' == 'z-ai'; 'glm53' matches 'glm-5.3-flash')
and tolerates the 'flsh' typo.

Usage:
  ./get_models.py [QUERY]                  # fuzzy search, or a detail card for an exact id
  ./get_models.py --creator zai --sort context
  ./get_models.py --caching --sort price --top 10
  ./get_models.py --refresh                # bypass the 1-hour catalog cache
  ./get_models.py -q flash --json
"""

import argparse
import difflib
import json
import re
from typing import Any, Dict, List, Tuple

import _bootstrap  # noqa: F401

from openrouter_frontier._util import price_per_million
from openrouter_frontier.render import Column, fmt_context, print_table
from openrouter_frontier.resolver import get_all_models


def clean_str(s: str) -> str:
    """Lowercase and strip everything except letters and digits."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_score(query: str, model_id: str, model_name: str) -> float:
    """Relevance in 0..1 between a query and a model, ignoring punctuation."""
    if not query:
        return 1.0
    q = clean_str(query)
    if not q:
        return 0.0

    creator, _, short = model_id.rpartition("/")
    full_c, short_c, creator_c, name_c = clean_str(model_id), clean_str(short), clean_str(creator), clean_str(model_name)

    if q in (full_c, short_c):
        return 1.0
    if q == creator_c:
        return 0.95
    if any(t.startswith(q) for t in (short_c, full_c, creator_c)):
        return 0.90
    if any(q in t for t in (full_c, name_c, short_c, creator_c)):
        return 0.85
    best = max(difflib.SequenceMatcher(None, q, t).ratio() for t in (short_c, full_c, creator_c))
    return best * 0.8 if best >= 0.70 else 0.0


def format_price(val: Any) -> str:
    p = price_per_million(val)
    if p is None:
        return "--"
    return "Free" if p <= 0 else f"${p:.4f}"


def print_single_model(m: Dict[str, Any]) -> None:
    p = m.get("pricing", {})
    arch = m.get("architecture", {})
    top_p = m.get("top_provider", {})
    ctx = m.get("context_length", 0)
    rule = "─" * 80

    print()
    print(rule)
    print(f"Model: {m.get('name') or m.get('id')}")
    print(f"ID:    {m.get('id')}")
    if m.get("canonical_slug") and m["canonical_slug"] != m.get("id"):
        print(f"Slug:  {m['canonical_slug']}")
    print(rule)
    print(f"  Context Window:        {ctx:,} tokens" if ctx else "  Context Window:        --")
    if top_p.get("max_completion_tokens"):
        print(f"  Max Output Tokens:     {top_p['max_completion_tokens']:,}")
    if arch.get("modality"):
        print(f"  Modality:              {arch['modality']}")
    if arch.get("instruct_type"):
        print(f"  Instruct Format:       {arch['instruct_type']}")

    print("\n  Pricing (USD per million tokens):")
    print(f"    Prompt (Input):      {format_price(p.get('prompt'))}")
    print(f"    Completion (Output): {format_price(p.get('completion'))}")
    print(f"    Cache Read:          {format_price(p.get('input_cache_read'))}")
    print(f"    Cache Write:         {format_price(p.get('input_cache_write'))}")
    if p.get("request"):
        print(f"    Request Fee:         ${float(p['request']):.6f}")
    if p.get("discount"):
        print(f"    Discount:            {float(p['discount']) * 100:.0f}% off")

    desc = (m.get("description") or "").strip()
    if desc:
        print("\n  Description:")
        for line in [l.strip() for l in desc.split("\n") if l.strip()][:4]:
            print(f"    {line[:90]}")

    print(rule)
    print(f"Tip: ./score_providers.py {m['id']}     ranks this model's providers by cost per turn")
    print(f"Tip: ./provider_frontier.py {m['id']}   shows the provider Pareto frontier\n")


def print_models_table(models: List[Dict[str, Any]], title_suffix: str = "") -> None:
    cols = [
        Column("Model ID", 34),
        Column("Model Name", 28),
        Column("Prompt $/M", 11, ">"),
        Column("Compl $/M", 11, ">"),
        Column("Read $/M", 10, ">"),
        Column("Context", 10, ">"),
        Column("Modality", 12),
    ]
    rows = []
    for m in models:
        p = m.get("pricing", {})
        rows.append([
            m.get("id", "")[:34],
            (m.get("name") or m.get("id", ""))[:28],
            format_price(p.get("prompt")),
            format_price(p.get("completion")),
            format_price(p.get("input_cache_read")),
            fmt_context(int(m.get("context_length") or 0)),
            (m.get("architecture", {}).get("modality") or "text")[:12],
        ])
    print_table(
        cols, rows,
        title=f"OpenRouter Models ({len(models)} models){title_suffix}",
        footer="Tip: pass an exact model ID to see its detail card (e.g. ./get_models.py z-ai/glm-5.3-flash).",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search and inspect the OpenRouter model catalog with fuzzy matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", nargs="?", default=None, help="Model ID, keyword, or creator (e.g. zai, z.ai, glm-5.3)")
    parser.add_argument("-q", "--search", type=str, default="", help="Keyword search across model ID and name")
    parser.add_argument("-m", "--creator", type=str, default="", help="Filter by creator (e.g. zai, anthropic, openai)")
    parser.add_argument("--caching", action="store_true", help="Only models with a prompt cache read price")
    parser.add_argument("--modality", type=str, default="", help="Filter by modality substring (e.g. text->text, image)")
    parser.add_argument("--sort", choices=["relevance", "name", "price", "completion", "context", "created"], default="relevance")
    parser.add_argument("-n", "--top", type=int, default=30, help="Max models to display")
    parser.add_argument("--refresh", action="store_true", help="Force refresh from the OpenRouter API")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    models = get_all_models(force_refresh=args.refresh)
    query = (args.query or args.search).strip()

    # An exact id (punctuation-insensitive) shows the detail card directly.
    if args.query and not args.search:
        q = clean_str(args.query)
        exact = [m for m in models if q in (clean_str(m.get("id", "")), clean_str(m.get("canonical_slug", "")))]
        if len(exact) == 1 and ("/" in args.query or "/" not in exact[0].get("id", "")):
            if args.json:
                print(json.dumps(exact[0], indent=2))
            else:
                print_single_model(exact[0])
            return

    creator_q = clean_str(args.creator)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for m in models:
        m_id = m.get("id", "")
        m_name = m.get("name") or m_id
        creator = m_id.split("/")[0] if "/" in m_id else ""
        if creator_q and creator_q not in clean_str(creator) and creator_q not in clean_str(m_name):
            continue
        if args.caching and not m.get("pricing", {}).get("input_cache_read"):
            continue
        if args.modality and args.modality.lower() not in (m.get("architecture", {}).get("modality") or "").lower():
            continue
        score = match_score(query, m_id, m_name)
        if query and score <= 0.0:
            continue
        scored.append((score, m))

    def sort_key(item: Tuple[float, Dict[str, Any]]):
        score, m = item
        p = m.get("pricing", {})
        name = (m.get("name") or m.get("id", "")).lower()
        if args.sort == "price":
            return price_per_million(p.get("prompt")) or 0.0
        if args.sort == "completion":
            return price_per_million(p.get("completion")) or 0.0
        if args.sort == "context":
            return -int(m.get("context_length") or 0)
        if args.sort == "created":
            return -int(m.get("created") or 0)
        if args.sort == "name":
            return name
        return (-score, name)

    scored.sort(key=sort_key)
    filtered = [m for _, m in scored]

    if args.json:
        print(json.dumps(filtered[: args.top], indent=2))
        return
    if len(filtered) == 1 and query:
        print_single_model(filtered[0])
        return

    suffix = ""
    if query:
        suffix += f" matching '{query}'"
    if args.creator:
        suffix += f" by '{args.creator}'"
    if args.caching:
        suffix += " with prompt caching"
    if args.sort not in ("relevance", "name"):
        suffix += f" (sorted by {args.sort})"
    print_models_table(filtered[: args.top], suffix)


if __name__ == "__main__":
    main()
