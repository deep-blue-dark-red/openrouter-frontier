#!/usr/bin/env python3
"""openrouter-tui - interactive terminal explorer for OpenRouter models and providers.

Views:
  1. Models     - the catalog ranked by the cost vs. quality Pareto frontier (frontier models
                  first, cheapest to most expensive, with the knee highlighted; then dominated
                  models by distance to the frontier). Type to fuzzy-filter; Tab toggles the
                  benchmark metric between intelligence and coding.
  2. Providers  - Enter on a model lists its endpoints ranked by ProviderUtility scored cost.
                  Tab or 'a' toggles between the primary quantization and all variants.
  3. Detail     - Enter on a provider shows the full cost breakdown and pricing.

Keys: Up/Down or Ctrl-P/N move, PgUp/PgDn jump 5, mouse wheel/click, Enter select,
Esc/Backspace back (Esc clears the search first), Ctrl-C quit.

Runs in the alternate screen buffer so the normal terminal scrollback is untouched.
"""

import os
import select
import shutil
import sys
import termios
import tty
from typing import Any, Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401

from model_frontier import build_candidates, load_data
from openrouter_analytics._util import filter_primary_quantization
from openrouter_analytics.client import score_model_providers
from openrouter_analytics.pareto import annotate_frontier, cost_quality_frontier, frontier_sort_key
from openrouter_analytics.render import Column, fmt_pct, fmt_seconds, fmt_tps, format_row, header_line
from openrouter_analytics.scoring import ScoreBreakdown, ScoringConfig

from get_models import clean_str

# ANSI helpers
ALT_SCREEN_ON = "\x1b[?1049h\x1b[?25l\x1b[?1000h\x1b[?1006h"
ALT_SCREEN_OFF = "\x1b[?1000l\x1b[?1006l\x1b[?25h\x1b[?1049l"
HOME = "\x1b[H"
CLEAR_BELOW = "\x1b[J"
RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
GREEN = "\x1b[32m"
BOLD_YELLOW = "\x1b[1;33m"
HILITE = "\x1b[48;5;237;1m"

KEYS_UP = ("\x1b[A", "\x1bOA", "\x10")
KEYS_DOWN = ("\x1b[B", "\x1bOB", "\x0e")
KEYS_PGUP = ("\x1b[1;5A", "\x1b[1;6A", "\x1b[1;2A", "\x1b[1;3A", "\x1b[1;9A", "\x1b\x1b[A", "\x1b[5~")
KEYS_PGDN = ("\x1b[1;5B", "\x1b[1;6B", "\x1b[1;2B", "\x1b[1;3B", "\x1b[1;9B", "\x1b\x1b[B", "\x1b[6~")
KEYS_ENTER = ("\r", "\n")
KEYS_BACKSPACE = ("\x7f", "\x08")
KEY_ESC = "\x1b"
KEY_CTRL_C = "\x03"
PAGE = 5
TABLE_FIRST_ROW = 4  # 1-based terminal row of the first data row (bar, header, rule above it)


# ------------------------------------------------------------------ data

def matches_query(query: str, m: Dict[str, Any]) -> bool:
    """Every whitespace-separated word must appear in the id+name, ignoring punctuation."""
    if not query.strip():
        return True
    raw = f"{m['id']} {m['name']}".lower()
    cleaned = clean_str(raw)
    for w in query.lower().split():
        w = w.replace("flsh", "flash")
        if w not in raw and clean_str(w) not in cleaned:
            return False
    return True


def build_model_list(metric: str) -> Tuple[List[Dict[str, Any]], int]:
    """Benchmarked models sorted by frontier position; returns ``(models, frontier_size)``."""
    catalog, raw_bench = load_data()
    candidates = build_candidates(catalog, raw_bench, metric, price_source="list")
    frontier, knee = cost_quality_frontier(candidates)
    annotate_frontier(candidates, frontier, knee)
    candidates.sort(key=frontier_sort_key)
    return candidates, len(frontier)


# ------------------------------------------------------------------ terminal I/O

def get_key() -> str:
    """Read one keypress, returning full escape sequences as a single string."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        select.select([fd], [], [])
        seq = os.read(fd, 1).decode("utf-8", errors="ignore")
        if seq == KEY_ESC:
            while select.select([fd], [], [], 0.02)[0]:
                more = os.read(fd, 64).decode("utf-8", errors="ignore")
                if not more:
                    break
                seq += more
        return seq
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def clear_screen() -> None:
    sys.stdout.write("\x1b[2J" + HOME)
    sys.stdout.flush()


def write(s: str = "") -> None:
    sys.stdout.write(s + "\n")


def parse_mouse(key: str) -> Optional[Tuple[int, int, int, bool]]:
    """Decode an SGR mouse report into ``(button, col, row, is_release)``."""
    if not key.startswith("\x1b[<"):
        return None
    try:
        parts = key[3:-1].split(";")
        return int(parts[0]), int(parts[1]), int(parts[2]), key.endswith("m")
    except (ValueError, IndexError):
        return None


def navigate(key: str, idx: int, count: int, scroll: int, visible: int) -> Tuple[Optional[int], bool]:
    """Apply a navigation key. Returns ``(new_index_or_None, clicked_current_row)``."""
    last = max(0, count - 1)
    if key in KEYS_UP:
        return max(0, idx - 1), False
    if key in KEYS_DOWN:
        return min(last, idx + 1), False
    if key in KEYS_PGUP:
        return max(0, idx - PAGE), False
    if key in KEYS_PGDN:
        return min(last, idx + PAGE), False
    mouse = parse_mouse(key)
    if mouse:
        button, _, row, released = mouse
        if button in (64, 68):
            return max(0, idx - PAGE), False
        if button in (65, 69):
            return min(last, idx + PAGE), False
        if button == 0 and not released and TABLE_FIRST_ROW <= row < TABLE_FIRST_ROW + visible:
            clicked = scroll + (row - TABLE_FIRST_ROW)
            if 0 <= clicked < count:
                return clicked, clicked == idx
        return idx, False
    return None, False


def scroll_window(idx: int, scroll: int, count: int, viewport: int) -> Tuple[int, int]:
    """Return ``(scroll_offset, num_visible)`` keeping ``idx`` inside the viewport."""
    visible = min(count, viewport)
    if idx < scroll:
        scroll = idx
    elif idx >= scroll + visible:
        scroll = max(0, idx - visible + 1)
    return scroll, visible


def pad_rows(rendered: int, viewport: int) -> None:
    for _ in range(viewport - rendered):
        write()


# ------------------------------------------------------------------ views

MODEL_COLS_BASE = [Column("Status", 14), Column("Model Name", 32), Column("Model ID", 30)]
PROVIDER_COLS = [
    Column("Provider", 16),
    Column("Scored Cost", 12, ">"),
    Column("Token Cost", 11, ">"),
    Column("Fail Risk", 10, ">"),
    Column("CacheHit", 8, ">"),
    Column("Latency", 8, ">"),
    Column("TPS", 5, ">"),
    Column("Uptime", 7, ">"),
    Column("Hit $/M", 9, ">"),
    Column("Miss $/M", 9, ">"),
]


def _usd(v: float) -> str:
    return f"${v:.4f}" if v < 0.1 else f"${v:.2f}"


def draw_models(models, query, metric, frontier_count, idx, scroll, width, viewport) -> Tuple[List[Dict[str, Any]], int, int]:
    filtered = [m for m in models if matches_query(query, m)] if query else models
    idx = min(idx, len(filtered) - 1) if filtered else 0
    scroll, visible = scroll_window(idx, scroll, len(filtered), viewport)

    sys.stdout.write(HOME)
    prompt = f"Search Models: {query}█"
    status = f"[{len(filtered)}/{len(models)} models | {frontier_count} on frontier | metric: {metric}]"
    gap = " " * max(2, width - len(prompt) - len(status) - 2)
    write(f"\x1b[1;36m{prompt}{RESET}{gap}{DIM}{status}{RESET}")

    cols = MODEL_COLS_BASE + [Column("Intel Score" if metric == "intelligence" else "Coding Score", 12, ">"),
                              Column("Prompt $/M", 12, ">"), Column("Output $/M", 11, ">")]
    head = header_line(cols)
    write(f"{BOLD}{head[:width - 1]}{RESET}")
    write("─" * min(width - 1, len(head)))

    if not filtered:
        write(f"{DIM}  (no models matching '{query}'){RESET}")
        pad_rows(1, viewport)
    else:
        end = min(len(filtered), scroll + visible)
        for i in range(scroll, end):
            m = filtered[i]
            knee, front = m["is_knee"], m["on_frontier"]
            status = "★ KNEE" if knee else "★ OPTIMAL" if front else f"-{m['dist']:.1f} pts"
            line = format_row(
                [status, m["name"][:32], m["id"][:30], f"{m['score']:.1f}", _usd(m["cost"]), _usd(m["compl_p"] or 0.0)],
                cols,
            )[: width - 1]
            if i == idx:
                colour = HILITE + ("\x1b[33m" if knee else "\x1b[32m" if front else "")
                write(f"{colour}{line}{RESET}")
            elif knee:
                write(f"{BOLD_YELLOW}{line[:14]}{RESET}{line[14:]}")
            elif front:
                write(f"{GREEN}{line[:14]}{RESET}{line[14:]}")
            else:
                write(f"{DIM}{line}{RESET}")
        pad_rows(end - scroll, viewport)

    write(f"\n{DIM}Enter: providers  •  Tab: toggle metric  •  Up/Down: move  •  Esc: clear / exit{RESET}"[: width + 8])
    write(f"{DIM}Cost vs. quality Pareto frontier. ★ KNEE = best quality gain per dollar; '-N pts' = gap to frontier.{RESET}"[: width + 8])
    sys.stdout.write(CLEAR_BELOW)
    sys.stdout.flush()
    return filtered, idx, scroll


def draw_providers(model, scores, all_quants, idx, scroll, width, viewport) -> Tuple[List[ScoreBreakdown], int, int]:
    active = filter_primary_quantization(scores, all_quants)
    idx = min(idx, len(active) - 1) if active else 0
    scroll, visible = scroll_window(idx, scroll, len(active), viewport)

    sys.stdout.write(HOME)
    quant_tag = "[quants: ALL]" if all_quants else "[quants: primary]"
    write(f"Providers for: \x1b[1;36m{model['name']}{RESET} ({model['id']})  •  {quant_tag}"[: width + 12])
    head = header_line(PROVIDER_COLS)
    write(f"{BOLD}{head[:width - 1]}{RESET}")
    write("─" * min(width - 1, len(head)))

    if not active:
        write(f"{DIM}  (no active providers found for this model){RESET}")
        pad_rows(1, viewport)
    else:
        end = min(len(active), scroll + visible)
        for i in range(scroll, end):
            s = active[i]
            line = format_row(
                [
                    s.provider_name[:16], s.formatted_total_cost, s.formatted_token_cost, s.formatted_failure_cost,
                    s.formatted_h_used, fmt_seconds(s.ttft_seconds), fmt_tps(s.throughput_tps), fmt_pct(s.uptime_pct),
                    f"${s.hit_price:.4f}", f"${s.miss_price:.4f}",
                ],
                PROVIDER_COLS,
            )[: width - 1]
            best = "\x1b[32m" if i == 0 else ""
            if i == idx:
                write(f"{HILITE}{best}{line}{RESET}")
            else:
                write(f"{best}{line}{RESET}")
        pad_rows(end - scroll, viewport)

    write(f"\n{DIM}Esc / Backspace / q: back  •  Tab / a: toggle quants  •  Enter: provider detail{RESET}"[: width + 8])
    write(f"{DIM}Ranked by ProviderUtility scored cost per turn (2000 prompt + 500 completion tokens). #1 is cheapest.{RESET}"[: width + 8])
    sys.stdout.write(CLEAR_BELOW)
    sys.stdout.flush()
    return active, idx, scroll


def draw_detail(s: ScoreBreakdown, model_id: str, width: int) -> None:
    sys.stdout.write(HOME)
    rule = "─" * min(width - 1, 78)
    write(f"\x1b[1;36mProvider Detail: {s.provider_name}{RESET}  (for {model_id})")
    write(rule)
    write(f"  Scored Cost per Turn:   \x1b[1;32m{s.formatted_total_cost}{RESET}")
    write(f"  Token Cost per Turn:    {s.formatted_token_cost}")
    write(f"  Failure Risk Penalty:   {s.formatted_failure_cost}")
    write(f"  Time Opportunity Cost:  {s.formatted_time_cost}")
    write(f"  Prompt Cache Hit Rate:  used {BOLD}{s.formatted_h_used}{RESET}  (published 24h: {s.formatted_h_raw})")
    write(f"  Cache Read (Hit) Price: ${s.hit_price:.4f} / M tokens")
    write(f"  Cache Write/Miss Price: ${s.miss_price:.4f} / M tokens")
    write(f"  Completion Price:       ${s.out_price:.4f} / M tokens")
    write(f"  Latency (TTFT):         {fmt_seconds(s.ttft_seconds)}  |  Throughput: {fmt_tps(s.throughput_tps, ' TPS')}  |  Uptime: {fmt_pct(s.uptime_pct)}")
    write(f"  Quantization Variant:   {s.quantization.upper()}")
    write(rule)
    write(f"{DIM}Press any key to return to the providers table.{RESET}")
    sys.stdout.write(CLEAR_BELOW)
    sys.stdout.flush()


# ------------------------------------------------------------------ main loop

def run_tui() -> None:
    metric = "intelligence"
    models, frontier_count = build_model_list(metric)

    view = "MODELS"
    query = ""
    m_idx = m_scroll = 0
    selected: Optional[Dict[str, Any]] = None
    scores: List[ScoreBreakdown] = []
    active: List[ScoreBreakdown] = []
    p_idx = p_scroll = 0
    all_quants = False

    sys.stdout.write(ALT_SCREEN_ON)
    clear_screen()
    try:
        while True:
            width, height = shutil.get_terminal_size((100, 30))
            viewport = max(2, height - 7)

            if view == "MODELS":
                filtered, m_idx, m_scroll = draw_models(models, query, metric, frontier_count, m_idx, m_scroll, width, viewport)
                key = get_key()
                new_idx, clicked = navigate(key, m_idx, len(filtered), m_scroll, min(len(filtered), viewport))
                if new_idx is not None:
                    m_idx = new_idx
                    if clicked and filtered:
                        selected, view, scores, p_idx, p_scroll = filtered[m_idx], "PROVIDERS", [], 0, 0
                        clear_screen()
                elif key == "\t":
                    metric = "coding" if metric == "intelligence" else "intelligence"
                    models, frontier_count = build_model_list(metric)
                    m_idx = m_scroll = 0
                    clear_screen()
                elif key in KEYS_ENTER and filtered:
                    selected, view, scores, p_idx, p_scroll = filtered[m_idx], "PROVIDERS", [], 0, 0
                    clear_screen()
                elif key in (KEY_ESC, KEY_CTRL_C):
                    if query and key == KEY_ESC:
                        query, m_idx = "", 0
                    else:
                        break
                elif key in KEYS_BACKSPACE:
                    query, m_idx = query[:-1], 0
                elif len(key) == 1 and 32 <= ord(key) <= 126:
                    query, m_idx = query + key, 0
                # any other escape sequence is ignored so it never lands in the search box

            elif view == "PROVIDERS":
                assert selected is not None
                if not scores:
                    clear_screen()
                    write(f"Fetching provider analytics for {selected['id']}...")
                    sys.stdout.flush()
                    scores = score_model_providers(selected["permaslug"], config=ScoringConfig())
                    clear_screen()
                active, p_idx, p_scroll = draw_providers(selected, scores, all_quants, p_idx, p_scroll, width, viewport)
                key = get_key()
                new_idx, clicked = navigate(key, p_idx, len(active), p_scroll, min(len(active), viewport))
                if new_idx is not None:
                    p_idx = new_idx
                    if clicked and active:
                        view = "DETAIL"
                        clear_screen()
                elif key in ("\t", "a", "A"):
                    all_quants = not all_quants
                    p_idx = p_scroll = 0
                    clear_screen()
                elif key in (KEY_ESC, "q", "Q", "\x1b[D", "\x1bOD") or key in KEYS_BACKSPACE:
                    view, scores = "MODELS", []
                    clear_screen()
                elif key in KEYS_ENTER and active:
                    view = "DETAIL"
                    clear_screen()
                elif key == KEY_CTRL_C:
                    break

            else:  # DETAIL
                assert selected is not None
                draw_detail(active[p_idx], selected["id"], width)
                if get_key() == KEY_CTRL_C:
                    break
                view = "PROVIDERS"
                clear_screen()
    finally:
        sys.stdout.write(ALT_SCREEN_OFF)
        sys.stdout.flush()


def main() -> None:
    try:
        run_tui()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
