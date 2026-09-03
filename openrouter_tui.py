#!/usr/bin/env python3
"""
openrouter-tui - Interactive Terminal UI for OpenRouter Analytics.

Features:
  - Model Explorer sorted by the Cost vs. Quality Pareto Frontier
    (powered by live OpenRouter / Artificial Analysis benchmark scores).
    Top models are on the frontier from lowest cost to highest, highlighting the KNEE point,
    followed by dominated models ordered by distance from the frontier.
  - Press 'm' to toggle benchmark metric: Intelligence Index vs. Coding Index.
  - Real-time delimiter-agnostic fuzzy searching ('zai' matches 'z-ai' and 'z.ai',
    'glm53' matches 'glm-5.3-flash', 'claude' matches 'anthropic/claude-*').
  - Provider scoring view when a model is selected, showing all endpoints ranked by
    ProviderUtility scored cost per turn (cache hit rates, shrinkage, latency, TPS, uptime).
  - Keyboard & mouse navigation (Up/Down, PgUp/PgDn, Ctrl-P/N, Mouse wheel/clicks).
"""

import sys
import os
import re
import math
import glob
import json
import tty
import termios
import select
import shutil
from typing import Optional, List, Dict, Any, Tuple

# Optimize connection on macOS (avoid IPv6 timeout)
try:
    import urllib3.util.connection
    import socket
    urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass

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

from openrouter_analytics.resolver import resolve_model
from openrouter_analytics.client import score_model_providers
from openrouter_analytics.scoring import ScoringConfig, ScoreBreakdown
from pareto_frontier import load_catalog_pricing, load_benchmarks, compute_pareto, strip_date_suffix


# ==============================================================================
# Helper Math & Data
# ==============================================================================

def clean_str(s: str) -> str:
    """Removes all non-alphanumeric characters and lowercases the string."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def matches_query(query: str, m: Dict[str, Any]) -> bool:
    """Fuzzy punctuation-agnostic word match: all query words must match."""
    if not query.strip():
        return True
    words = query.strip().lower().split()
    target_raw = f"{m['id']} {m['name']}".lower()
    target_clean = clean_str(target_raw)

    for w in words:
        if "flsh" in w:
            w = w.replace("flsh", "flash")
        w_clean = clean_str(w)
        if (w not in target_raw) and (not w_clean or w_clean not in target_clean):
            return False
    return True


def build_pareto_model_catalog(metric: str = "intelligence") -> Tuple[List[Dict[str, Any]], int]:
    """Loads catalog pricing and live benchmarks, then computes the Cost vs Quality Pareto frontier."""
    catalog = load_catalog_pricing()
    bench_items, weighted_prices = load_benchmarks(None, metric)

    score_key = f"{metric}_index"
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
        if not cat_entry or cat_entry["prompt_per_m"] <= 0:
            continue

        candidates.append({
            "id": base_id,
            "permaslug": permaslug,
            "name": item.get("display_name") or cat_entry["name"],
            "score": float(score),
            "cost": cat_entry["prompt_per_m"],
            "prompt_p": cat_entry["prompt_per_m"],
            "compl_p": cat_entry["completion_per_m"],
            "metric": metric,
        })
        seen_slugs.add(base_id)

    frontier, knee_idx = compute_pareto(candidates)
    frontier_ids = {m["id"] for m in frontier}

    # Mark frontier status and knee
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

    # Sort: frontier first by cost ascending, then dominated by distance to frontier ascending
    candidates.sort(key=lambda x: (not x["on_frontier"], x["cost"] if x["on_frontier"] else x["dist"]))
    return candidates, len(frontier)


# ==============================================================================
# Raw Terminal Keyboard Input
# ==============================================================================

def get_key() -> str:
    """Reads interactive keypresses and full escape sequences from raw terminal mode."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        select.select([fd], [], [])
        ch = os.read(fd, 1).decode("utf-8", errors="ignore")
        if ch == "\x1b":
            seq = ch
            while True:
                r, _, _ = select.select([fd], [], [], 0.02)
                if r:
                    more = os.read(fd, 64).decode("utf-8", errors="ignore")
                    if not more:
                        break
                    seq += more
                else:
                    break
            return seq
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# ==============================================================================
# Terminal Interactive TUI Loop
# ==============================================================================

def run_tui():
    current_metric = "intelligence"  # "intelligence" or "coding"
    models, frontier_count = build_pareto_model_catalog(current_metric)

    current_view = "MODELS"  # "MODELS" or "PROVIDERS" or "DETAIL"
    query = ""
    selected_idx = 0
    scroll_offset = 0
    first_draw = True
    last_total_lines = 0

    # Selected model state
    selected_model_dict: Optional[Dict[str, Any]] = None
    provider_scores: List[ScoreBreakdown] = []
    provider_selected_idx = 0
    provider_scroll_offset = 0
    filter_all_quants = False

    # Hide cursor and enable SGR mouse tracking
    sys.stdout.write("\x1b[?25l\x1b[?1000h\x1b[?1006h")
    sys.stdout.flush()

    try:
        while True:
            term_width, term_height = shutil.get_terminal_size((100, 30))

            # ==================================================================
            # VIEW 1: MODEL SELECTION LIST
            # ==================================================================
            if current_view == "MODELS":
                filtered_models = [m for m in models if matches_query(query, m)] if query else models
                selected_idx = max(0, min(selected_idx, len(filtered_models) - 1)) if filtered_models else 0

                num_visible = min(len(filtered_models), max(2, term_height - 7))

                if selected_idx < scroll_offset:
                    scroll_offset = selected_idx
                elif selected_idx >= scroll_offset + num_visible:
                    scroll_offset = max(0, selected_idx - num_visible + 1)

                total_lines = num_visible + 6

                if not first_draw:
                    sys.stdout.write(f"\x1b[{last_total_lines}A\r")
                else:
                    first_draw = False

                last_total_lines = total_lines
                sys.stdout.write("\x1b[J")

                # 1. Search Bar
                search_prompt = f"Search Models: {query}█"
                count_str = f"[{len(filtered_models)}/{len(models)} models | {frontier_count} on Frontier | Metric: {current_metric.upper()}]"
                space = max(2, term_width - len(search_prompt) - len(count_str) - 2)
                top_bar = f"\x1b[1;36m{search_prompt}\x1b[0m{' ' * space}\x1b[2m{count_str}\x1b[0m"
                sys.stdout.write(top_bar[:term_width + 15] + "\n")

                # 2. Table Header
                metric_label = "Intel Score" if current_metric == "intelligence" else "Coding Score"
                cols = [
                    ("Status", 14, "<"),
                    ("Model Name", 32, "<"),
                    ("Model ID", 30, "<"),
                    (metric_label, 12, ">"),
                    ("Cost ($/1M)", 12, ">"),
                    ("Output $/M", 11, ">"),
                ]
                header_parts = [f"{name:>{w}}" if a == ">" else f"{name:<{w}}" for name, w, a in cols]
                header_str = "  ".join(header_parts)
                sys.stdout.write(f"\x1b[1m{header_str[:term_width - 1]}\x1b[0m\n")
                sys.stdout.write("─" * min(term_width - 1, len(header_str)) + "\n")

                # 3. Model Rows
                end_idx = min(len(filtered_models), scroll_offset + num_visible)
                for i in range(scroll_offset, end_idx):
                    m = filtered_models[i]
                    on_f = m["on_frontier"]
                    is_k = m.get("is_knee", False)

                    if is_k:
                        status_str = "★ KNEE"
                    elif on_f:
                        status_str = "★ OPTIMAL"
                    else:
                        status_str = f"-{m['dist']:.1f} pts"

                    cost_str = f"${m['cost']:.4f}" if m['cost'] < 0.1 else f"${m['cost']:.2f}"
                    compl_str = f"${m['compl_p']:.4f}" if m['compl_p'] < 0.1 else f"${m['compl_p']:.2f}"

                    row_vals = [
                        status_str,
                        m["name"][:32],
                        m["id"][:30],
                        f"{m['score']:.1f}",
                        cost_str,
                        compl_str,
                    ]

                    line_str = "  ".join(f"{val:>{w}}" if a == ">" else f"{val:<{w}}" for val, (_, w, a) in zip(row_vals, cols))
                    line_clipped = line_str[:term_width - 1]

                    if i == selected_idx:
                        if is_k:
                            sys.stdout.write(f"\x1b[48;5;237;1;33m{line_clipped}\x1b[0m\n")
                        elif on_f:
                            sys.stdout.write(f"\x1b[48;5;237;1;32m{line_clipped}\x1b[0m\n")
                        else:
                            sys.stdout.write(f"\x1b[48;5;237;1m{line_clipped}\x1b[0m\n")
                    else:
                        if is_k:
                            sys.stdout.write(f"\x1b[1;33m★ KNEE\x1b[0m         {line_clipped[15:]}\n")
                        elif on_f:
                            sys.stdout.write(f"\x1b[32m★ OPTIMAL\x1b[0m      {line_clipped[15:]}\n")
                        else:
                            sys.stdout.write(f"\x1b[2m{line_clipped}\x1b[0m\n")

                rendered_rows = end_idx - scroll_offset
                if rendered_rows < num_visible:
                    for _ in range(num_visible - rendered_rows):
                        sys.stdout.write("\n")

                # 4. Footer Help
                footer1 = "Enter: Inspect Providers  •  [m]: Toggle Metric (Intel/Coding)  •  Up/Down: Move  •  Esc: Exit"
                footer2 = "Cost vs. Quality Pareto Frontier: Top models are non-dominated. ★ KNEE = highest quality gain per $."
                sys.stdout.write(f"\n\x1b[2m{footer1[:term_width - 1]}\x1b[0m\n")
                sys.stdout.write(f"\x1b[2m{footer2[:term_width - 1]}\x1b[0m")
                sys.stdout.flush()

                # Input event
                key = get_key()
                if key in ("\x1b[A", "\x10"):  # Up
                    selected_idx = max(0, selected_idx - 1)
                elif key in ("\x1b[B", "\x0e"):  # Down
                    selected_idx = min(len(filtered_models) - 1, selected_idx + 1)
                elif key in ("\x1b[5~", "\x1b[1;5A"):  # PgUp
                    selected_idx = max(0, selected_idx - 5)
                elif key in ("\x1b[6~", "\x1b[1;5B"):  # PgDn
                    selected_idx = min(len(filtered_models) - 1, selected_idx + 5)
                elif key.startswith("\x1b[<"):  # Mouse event
                    try:
                        is_release = key.endswith("m")
                        body = key[3:-1]
                        parts = body.split(";")
                        if len(parts) >= 3:
                            cb, cx, cy = int(parts[0]), int(parts[1]), int(parts[2])
                            if cb in (64, 68):
                                selected_idx = max(0, selected_idx - 5)
                            elif cb in (65, 69):
                                selected_idx = min(len(filtered_models) - 1, selected_idx + 5)
                            elif cb == 0 and not is_release:
                                if 4 <= cy < 4 + num_visible:
                                    clicked_idx = scroll_offset + (cy - 4)
                                    if 0 <= clicked_idx < len(filtered_models):
                                        if clicked_idx == selected_idx:
                                            selected_model_dict = filtered_models[selected_idx]
                                            current_view = "PROVIDERS"
                                            first_draw = True
                                        else:
                                            selected_idx = clicked_idx
                    except Exception:
                        pass
                elif key in ("m", "M") and len(query) == 0:  # Toggle metric when search empty
                    current_metric = "coding" if current_metric == "intelligence" else "intelligence"
                    models, frontier_count = build_pareto_model_catalog(current_metric)
                    selected_idx = 0
                    scroll_offset = 0
                elif key in ("\r", "\n"):  # Enter
                    if filtered_models:
                        selected_model_dict = filtered_models[selected_idx]
                        current_view = "PROVIDERS"
                        first_draw = True
                        provider_selected_idx = 0
                        provider_scroll_offset = 0
                elif key in ("\x1b", "\x03"):  # Esc / Ctrl-C
                    break
                elif key in ("\x7f", "\x08"):  # Backspace
                    if len(query) > 0:
                        query = query[:-1]
                        selected_idx = 0
                elif len(key) == 1 and 32 <= ord(key) <= 126:  # Printable character
                    query += key
                    selected_idx = 0

            # ==================================================================
            # VIEW 2: PROVIDER SCORING VIEW
            # ==================================================================
            elif current_view == "PROVIDERS":
                if selected_model_dict and not provider_scores:
                    sys.stdout.write(f"\x1b[{last_total_lines}A\r\x1b[J")
                    sys.stdout.write(f"Fetching provider analytics for {selected_model_dict['id']}...\n")
                    sys.stdout.flush()
                    cfg = ScoringConfig(prompt_tokens=2000, completion_tokens=500, time_value_usd_per_hour=0.0, price_failures=True)
                    provider_scores = score_model_providers(selected_model_dict["permaslug"], config=cfg)
                    last_total_lines = 1

                # Apply quantization filter (default: auto primary fp8 matching website)
                active_scores = []
                primary_quant = "fp8" if any(getattr(s, "quantization", "") == "fp8" for s in provider_scores) else None
                for s in provider_scores:
                    q = getattr(s, "quantization", "unknown") or "unknown"
                    if not filter_all_quants and primary_quant and q != primary_quant:
                        continue
                    active_scores.append(s)

                provider_selected_idx = max(0, min(provider_selected_idx, len(active_scores) - 1)) if active_scores else 0
                num_visible = min(len(active_scores), max(2, term_height - 7))

                if provider_selected_idx < provider_scroll_offset:
                    provider_scroll_offset = provider_selected_idx
                elif provider_selected_idx >= provider_scroll_offset + num_visible:
                    provider_scroll_offset = max(0, provider_selected_idx - num_visible + 1)

                total_lines = num_visible + 6

                if not first_draw:
                    sys.stdout.write(f"\x1b[{last_total_lines}A\r")
                else:
                    first_draw = False

                last_total_lines = total_lines
                sys.stdout.write("\x1b[J")

                # Header Banner
                m_title = selected_model_dict.get("name", selected_model_dict["id"])
                quant_tag = "[Filter: ALL QUANTS]" if filter_all_quants else (f"[Filter: {primary_quant.upper()}]" if primary_quant else "[Filter: ALL]")
                banner = f"Providers for: \x1b[1;36m{m_title}\x1b[0m ({selected_model_dict['id']})  •  {quant_tag}"
                sys.stdout.write(f"{banner[:term_width + 15]}\n")

                # Table Header
                cols = [
                    ("Provider", 16, "<"),
                    ("Scored Cost", 12, ">"),
                    ("Token Cost", 11, ">"),
                    ("Fail Risk", 10, ">"),
                    ("h(used)", 8, ">"),
                    ("Latency", 8, ">"),
                    ("TPS", 5, ">"),
                    ("Uptime", 7, ">"),
                    ("Hit $/M", 9, ">"),
                    ("Miss $/M", 9, ">"),
                ]
                header_parts = [f"{name:>{w}}" if a == ">" else f"{name:<{w}}" for name, w, a in cols]
                header_str = "  ".join(header_parts)
                sys.stdout.write(f"\x1b[1m{header_str[:term_width - 1]}\x1b[0m\n")
                sys.stdout.write("─" * min(term_width - 1, len(header_str)) + "\n")

                # Provider Rows
                end_idx = min(len(active_scores), provider_scroll_offset + num_visible)
                for i in range(provider_scroll_offset, end_idx):
                    s = active_scores[i]
                    lat_str = f"{s.ttft_seconds:.2f}s" if s.ttft_seconds else "--"
                    tps_str = f"{s.throughput_tps:.0f}" if s.throughput_tps else "--"
                    upt_str = f"{s.uptime_pct:.1f}%" if s.uptime_pct else "--"

                    row_vals = [
                        s.provider_name[:16],
                        f"${s.total_cost_usd:.6f}",
                        f"${s.token_cost_usd:.6f}",
                        f"${s.failure_risk_cost_usd:.6f}",
                        f"{s.h_used * 100:.1f}%",
                        lat_str,
                        tps_str,
                        upt_str,
                        f"${s.hit_price:.4f}",
                        f"${s.miss_price:.4f}",
                    ]

                    line_str = "  ".join(f"{val:>{w}}" if a == ">" else f"{val:<{w}}" for val, (_, w, a) in zip(row_vals, cols))
                    line_clipped = line_str[:term_width - 1]

                    if i == provider_selected_idx:
                        if i == 0:
                            sys.stdout.write(f"\x1b[48;5;237;1;32m{line_clipped}\x1b[0m\n")
                        else:
                            sys.stdout.write(f"\x1b[48;5;237;1m{line_clipped}\x1b[0m\n")
                    else:
                        if i == 0:
                            sys.stdout.write(f"\x1b[32m{line_clipped}\x1b[0m\n")
                        else:
                            sys.stdout.write(f"{line_clipped}\n")

                rendered_rows = end_idx - provider_scroll_offset
                if rendered_rows < num_visible:
                    for _ in range(num_visible - rendered_rows):
                        sys.stdout.write("\n")

                # Footer Help
                footer1 = "Esc / Backspace / q: Back to Models  •  Tab / a: Toggle Quants  •  Enter: Provider Spec Card"
                footer2 = "Providers ranked by ProviderUtility Scored Cost. #1 provides optimal economic turn utility."
                sys.stdout.write(f"\n\x1b[2m{footer1[:term_width - 1]}\x1b[0m\n")
                sys.stdout.write(f"\x1b[2m{footer2[:term_width - 1]}\x1b[0m")
                sys.stdout.flush()

                # Input event
                key = get_key()
                if key in ("\x1b[A", "\x10"):
                    provider_selected_idx = max(0, provider_selected_idx - 1)
                elif key in ("\x1b[B", "\x0e"):
                    provider_selected_idx = min(len(active_scores) - 1, provider_selected_idx + 1)
                elif key in ("\x1b[5~", "\x1b[1;5A"):
                    provider_selected_idx = max(0, provider_selected_idx - 5)
                elif key in ("\x1b[6~", "\x1b[1;5B"):
                    provider_selected_idx = min(len(active_scores) - 1, provider_selected_idx + 5)
                elif key.startswith("\x1b[<"):
                    try:
                        is_release = key.endswith("m")
                        body = key[3:-1]
                        parts = body.split(";")
                        if len(parts) >= 3:
                            cb, cx, cy = int(parts[0]), int(parts[1]), int(parts[2])
                            if cb in (64, 68):
                                provider_selected_idx = max(0, provider_selected_idx - 5)
                            elif cb in (65, 69):
                                provider_selected_idx = min(len(active_scores) - 1, provider_selected_idx + 5)
                            elif cb == 0 and not is_release:
                                if 4 <= cy < 4 + num_visible:
                                    clicked_idx = provider_scroll_offset + (cy - 4)
                                    if 0 <= clicked_idx < len(active_scores):
                                        provider_selected_idx = clicked_idx
                    except Exception:
                        pass
                elif key in ("\t", "a", "A"):
                    filter_all_quants = not filter_all_quants
                    provider_selected_idx = 0
                    provider_scroll_offset = 0
                elif key in ("\x1b", "\x7f", "\x08", "q", "Q", "\x1b[D"):
                    current_view = "MODELS"
                    provider_scores = []
                    first_draw = True
                elif key in ("\r", "\n"):
                    if active_scores:
                        current_view = "DETAIL"
                        first_draw = True
                elif key == "\x03":
                    break

            # ==================================================================
            # VIEW 3: PROVIDER SPEC DETAIL POPUP
            # ==================================================================
            elif current_view == "DETAIL":
                p_stat = active_scores[provider_selected_idx]
                lines_to_draw = 16

                if not first_draw:
                    sys.stdout.write(f"\x1b[{last_total_lines}A\r")
                else:
                    first_draw = False

                last_total_lines = lines_to_draw
                sys.stdout.write("\x1b[J")

                divider = "─" * min(term_width - 1, 78)
                sys.stdout.write(f"\x1b[1;36mProvider Detail: {p_stat.provider_name}\x1b[0m  (for {selected_model_dict['id']})\n")
                sys.stdout.write(divider + "\n")
                sys.stdout.write(f"  Scored Cost per Turn:   \x1b[1;32m${p_stat.total_cost_usd:.6f}\x1b[0m\n")
                sys.stdout.write(f"  Token Cost per Turn:    ${p_stat.token_cost_usd:.6f}\n")
                sys.stdout.write(f"  Failure Risk Penalty:   ${p_stat.failure_risk_cost_usd:.6f}\n")
                sys.stdout.write(f"  Time Opportunity Cost:  ${p_stat.time_cost_usd:.6f}\n")
                sys.stdout.write(f"  Prompt Cache Hit Rate:  Used: \x1b[1m{p_stat.h_used * 100:.1f}%\x1b[0m  (Observed 24h: {p_stat.h_raw * 100:.1f}%)\n")
                sys.stdout.write(f"  Cache Read (Hit) Price: ${p_stat.hit_price:.4f} / M tokens\n")
                sys.stdout.write(f"  Cache Write/Miss Price: ${p_stat.miss_price:.4f} / M tokens\n")
                sys.stdout.write(f"  Completion Price:       ${p_stat.out_price:.4f} / M tokens\n")
                lat_str = f"{p_stat.ttft_seconds:.2f}s" if p_stat.ttft_seconds else "--"
                tps_str = f"{p_stat.throughput_tps:.0f} TPS" if p_stat.throughput_tps else "--"
                upt_str = f"{p_stat.uptime_pct:.1f}%" if p_stat.uptime_pct else "--"
                sys.stdout.write(f"  Latency (TTFT):         {lat_str}  |  Throughput: {tps_str}  |  Uptime: {upt_str}\n")
                q_str = getattr(p_stat, "quantization", "unknown")
                sys.stdout.write(f"  Quantization Variant:   {q_str.upper()}\n")
                sys.stdout.write(divider + "\n")
                sys.stdout.write("\x1b[2mPress any key, Esc, or Enter to return to providers table...\x1b[0m\n")
                sys.stdout.flush()

                key = get_key()
                if key == "\x03":
                    break
                current_view = "PROVIDERS"
                first_draw = True

    finally:
        sys.stdout.write(f"\x1b[{last_total_lines}A\r\x1b[J\x1b[?1000l\x1b[?1006l\x1b[?25h")
        sys.stdout.flush()


def main():
    try:
        run_tui()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
