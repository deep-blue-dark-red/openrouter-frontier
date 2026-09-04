#!/usr/bin/env python3
"""openrouter-tui - interactive terminal explorer for OpenRouter models and providers.

Views:
  1. Models     - the catalog ranked by the cost vs. quality Pareto frontier (frontier models
                  first, cheapest to most expensive, with the efficient point highlighted; then dominated
                  models by distance to the frontier). Type to fuzzy-filter; Tab toggles the
                  benchmark metric between intelligence and coding.
  2. Providers  - Enter on a model lists its endpoints ranked by ProviderScore scored cost.
                  Tab or 'a' toggles between the primary quantization and all variants.
  3. Detail     - Enter on a provider shows the full cost breakdown and pricing.

Keys: Up/Down or Ctrl-P/N move, PgUp/PgDn jump 5, mouse wheel/click, Enter select,
Esc/Backspace back (Esc clears the search first), Ctrl-C quit.

Runs in the alternate screen buffer so the normal terminal scrollback is untouched.
"""

import os
import re
import select
import shutil
import sys
import termios
import tty
from typing import Any, Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401

from model_frontier import build_candidates, load_data
from openrouter_frontier._util import filter_primary_quantization
from openrouter_frontier.client import score_model_providers
from openrouter_frontier.pareto import annotate_frontier, cost_quality_frontier, frontier_sort_key
from openrouter_frontier.render import Column, fmt_pct, fmt_seconds, fmt_tps, format_row, header_line
from openrouter_frontier.scoring import ScoreBreakdown, ScoringConfig

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
CYAN = "\x1b[36m"
BOLD_CYAN = "\x1b[1;36m"
KEY = "\x1b[1;33m"          # key names in the help line
BADGE = KEY                  # the active filter / metric mode
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
MODEL_NAME_MIN_W = 32  # Model Name column never shrinks below this


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
    frontier, efficient = cost_quality_frontier(candidates)
    annotate_frontier(candidates, frontier, efficient)
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
                more = os.read(fd, 256).decode("utf-8", errors="ignore")
                if not more:
                    break
                seq += more
        return seq
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def clear_screen() -> None:
    """Home the cursor and erase everything below it, so each frame starts on a blank buffer."""
    sys.stdout.write(HOME + CLEAR_BELOW)


def write(s: str = "") -> None:
    sys.stdout.write(s + "\n")


_MOUSE_RE = re.compile(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])")


def parse_mouse(key: str) -> List[Tuple[int, int, int, bool]]:
    """Decode every SGR mouse report in the buffer into ``(button, col, row, is_release)``.

    Rapid wheel ticks arrive concatenated in a single read; decoding all of them
    keeps fast scrolling from dropping events.
    """
    events = []
    for button, col, row, kind in _MOUSE_RE.findall(key):
        events.append((int(button), int(col), int(row), kind == "m"))
    return events


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
    events = parse_mouse(key)
    if events:
        orig = idx
        clicked = False
        for button, _, row, released in events:
            if button >= 64 and (button - 64) % 4 == 0:  # wheel up, incl. modifier variants
                idx = max(0, idx - PAGE)
            elif button >= 65 and (button - 65) % 4 == 0:  # wheel down
                idx = min(last, idx + PAGE)
            elif button == 0 and not released and TABLE_FIRST_ROW <= row < TABLE_FIRST_ROW + visible:
                hit = scroll + (row - TABLE_FIRST_ROW)
                if 0 <= hit < count:
                    idx = hit
                    clicked = hit == orig
        return idx, clicked
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


def styled(segments: List[Tuple[str, str]], width: int) -> str:
    """Join ``(style, text)`` segments, clipping the *visible* text to ``width`` columns."""
    out, room = [], width
    for style, text in segments:
        if room <= 0:
            break
        out.append(f"{style}{text[:room]}{RESET}" if style else text[:room])
        room -= len(text)
    return "".join(out)


def help_line(pairs: List[Tuple[str, str]], width: int) -> str:
    """Render ``(key, action)`` pairs as a single help line with the keys highlighted."""
    segments: List[Tuple[str, str]] = []
    for i, (key, action) in enumerate(pairs):
        if i:
            segments.append(("", "   "))
        segments.extend([(KEY, key), ("", f" {action}")])
    return styled(segments, width)


# ------------------------------------------------------------------ views

MODEL_STATUS_COL = Column("Status", 14)
MODEL_ID_COL = Column("Model ID", 30)
PROVIDER_COLS = [
    Column("Provider", 16),
    Column("Task $", 9, ">"),
    Column("Tokens $", 9, ">"),
    Column("Time $", 9, ">"),
    Column("Miss $", 8, ">"),
    Column("CacheHit", 8, ">"),
    Column("E[TTFT]", 8, ">"),
    Column("TPS", 5, ">"),
    Column("Uptime", 7, ">"),
    Column("$/M", 8, ">"),
]


def _usd(v: float) -> str:
    return f"${v:.4f}" if v < 0.1 else f"${v:.2f}"


def draw_models(models, query, metric, frontier_count, idx, scroll, width, viewport) -> Tuple[List[Dict[str, Any]], int, int]:
    filtered = [m for m in models if matches_query(query, m)] if query else models
    idx = min(idx, len(filtered) - 1) if filtered else 0
    scroll, visible = scroll_window(idx, scroll, len(filtered), viewport)

    clear_screen()
    write(styled([
        (BOLD_CYAN, f"Search Models: {query}█"),
        ("", "   "),
        (BOLD, f"{len(filtered)}/{len(models)}"), ("", " models   "),
        (BOLD, str(frontier_count)), ("", " on frontier   "),
        ("", "metric: "), (BADGE, metric),
    ], width - 1))

    name_w = max([MODEL_NAME_MIN_W] + [len(m["name"]) for m in models])
    cols = [MODEL_STATUS_COL, Column("Model Name", name_w), MODEL_ID_COL,
            Column("Intel Score" if metric == "intelligence" else "Coding Score", 12, ">"),
            Column("Prompt $/M", 12, ">"), Column("Output $/M", 11, ">")]
    head = header_line(cols)
    write(f"{BOLD}{head[:width - 1]}{RESET}")
    write("─" * min(width - 1, len(head)))

    if not filtered:
        pad_rows(0, viewport)
    else:
        end = min(len(filtered), scroll + visible)
        for i in range(scroll, end):
            m = filtered[i]
            efficient, front = m["is_efficient"], m["on_frontier"]
            status = "★ EFFICIENT" if efficient else "★ OPTIMAL" if front else f"-{m['dist']:.1f} pts"
            line = format_row(
                [status, m["name"][:name_w], m["id"][:30], f"{m['score']:.1f}", _usd(m["cost"]), _usd(m["compl_p"] or 0.0)],
                cols,
            )[: width - 1]
            if i == idx:
                colour = HILITE + ("\x1b[33m" if efficient else "\x1b[32m" if front else "")
                write(f"{colour}{line}{RESET}")
            elif efficient:
                write(f"{BOLD_YELLOW}{line[:14]}{RESET}{line[14:]}")
            elif front:
                write(f"{GREEN}{line[:14]}{RESET}{line[14:]}")
            else:
                write(f"{DIM}{line}{RESET}")
        pad_rows(end - scroll, viewport)

    write()
    write(help_line([("Enter", "providers"), ("Tab", "toggle metric"), ("↑/↓", "move"), ("Esc", "clear / exit")], width - 1))
    sys.stdout.write(styled([
        (BOLD_YELLOW, "★ EFFICIENT"), ("", " best quality gain per dollar   "),
        (GREEN, "★ OPTIMAL"), ("", " on the cost/quality frontier   "),
        ("", "-N pts gap to frontier"),
    ], width - 1))
    sys.stdout.flush()
    return filtered, idx, scroll


def draw_providers(model, scores, all_quants, idx, scroll, width, viewport) -> Tuple[List[ScoreBreakdown], int, int]:
    active = filter_primary_quantization(scores, all_quants)
    idx = min(idx, len(active) - 1) if active else 0
    scroll, visible = scroll_window(idx, scroll, len(active), viewport)

    clear_screen()
    write(styled([
        ("", "Providers for: "), (BOLD_CYAN, model["name"]), ("", f" ({model['id']})   "),
        ("", "quants: "), (BADGE, "all" if all_quants else "primary"),
    ], width - 1))
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
                    s.provider_name[:16], s.formatted_task_cost, s.formatted_token_cost, s.formatted_time_cost,
                    s.formatted_miss_premium, s.formatted_cache_hit_rate, fmt_seconds(s.ttft_seconds),
                    fmt_tps(s.throughput_tps), fmt_pct(s.uptime_pct), f"${s.task_cost_per_m:.4f}",
                ],
                PROVIDER_COLS,
            )[: width - 1]
            best = "\x1b[32m" if i == 0 else ""
            if i == idx:
                write(f"{HILITE}{best}{line}{RESET}")
            else:
                write(f"{best}{line}{RESET}")
        pad_rows(end - scroll, viewport)

    write()
    write(help_line([("Enter", "provider detail"), ("Tab", "toggle quants"), ("↑/↓", "move"), ("Esc", "back")], width - 1))
    cfg = ScoringConfig()
    sys.stdout.write(
        f"Ranked by expected task cost: {cfg.n_turns} turns × ({cfg.new_tokens_per_turn}+{cfg.completion_per_turn} tok) "
        f"→ {cfg.transcript_tokens // 1000}k, time ${cfg.time_value_usd_per_hour:.0f}/hr, {cfg.routing} routing. #1 is cheapest."[: width - 1]
    )
    sys.stdout.flush()
    return active, idx, scroll


def draw_detail(s: ScoreBreakdown, model_id: str, width: int) -> None:
    clear_screen()
    rule = "─" * min(width - 1, 78)
    write(f"\x1b[1;36mProvider Detail: {s.provider_name}{RESET}  (for {model_id})")
    write(rule)
    write(f"  Task: {s.turns} turns × ({s.new_tokens} new + {s.completion_tokens} out tokens), routing {s.routing}")
    write(f"  Expected Task Cost:         \x1b[1;32m{s.formatted_task_cost}{RESET}   (per turn {_usd(s.mean_turn_cost_usd)}, per 1M submitted tok ${s.task_cost_per_m:.4f})")
    write(f"    fixed (new tok + output):  {_usd(s.fixed_cost_usd)}")
    write(f"    time:                      {_usd(s.time_cost_usd)}")
    write(f"    cached-read baseline:      {_usd(s.read_baseline_usd)}")
    write(f"    cache-miss premium:        {_usd(s.miss_premium_usd)}")
    write(f"    failure premium + return:  {_usd(s.failure_premium_usd + s.return_penalty_usd)}")
    write(f"  Bounds: perfect cache {_usd(s.perfect_cache_cost_usd)}  |  cold cache {_usd(s.cold_cache_cost_usd)}")
    write(f"  Risk: σ_proc {_usd(s.sigma_proc_usd)}  |  σ_par {_usd(s.sigma_par_usd)}  |  P(migrate to fallback) {s.migration_probability:.0%}")
    write(f"  Cache Hit Rate (24h):       {BOLD}{s.formatted_cache_hit_rate}{RESET}" + (f"   imputed: {', '.join(s.imputed)}" if s.imputed else ""))
    write(f"  Prices per 1M tok:          input ${s.input_price:.4f}  |  read ${s.read_price:.4f}  |  write ${s.write_price:.4f}  |  miss ${s.miss_price:.4f}  |  out ${s.out_price:.4f}")
    write(f"  E[TTFT]: {fmt_seconds(s.ttft_seconds)}  |  Throughput p50: {fmt_tps(s.throughput_tps, ' TPS')}  |  Uptime: {fmt_pct(s.uptime_pct)}")
    write(f"  Quantization Variant:       {s.quantization.upper()}")
    write(rule)
    write(help_line([("Any key", "back to providers")], width - 1))
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
    try:
        while True:
            width, height = shutil.get_terminal_size((100, 30))
            viewport = max(2, height - 6)

            if view == "MODELS":
                filtered, m_idx, m_scroll = draw_models(models, query, metric, frontier_count, m_idx, m_scroll, width, viewport)
                key = get_key()
                new_idx, clicked = navigate(key, m_idx, len(filtered), m_scroll, min(len(filtered), viewport))
                if new_idx is not None:
                    m_idx = new_idx
                    if clicked and filtered:
                        selected, view, scores, p_idx, p_scroll = filtered[m_idx], "PROVIDERS", [], 0, 0
                elif key == "\t":
                    metric = "coding" if metric == "intelligence" else "intelligence"
                    models, frontier_count = build_model_list(metric)
                    m_idx = m_scroll = 0
                elif key in KEYS_ENTER and filtered:
                    selected, view, scores, p_idx, p_scroll = filtered[m_idx], "PROVIDERS", [], 0, 0
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
                active, p_idx, p_scroll = draw_providers(selected, scores, all_quants, p_idx, p_scroll, width, viewport)
                key = get_key()
                new_idx, clicked = navigate(key, p_idx, len(active), p_scroll, min(len(active), viewport))
                if new_idx is not None:
                    p_idx = new_idx
                    if clicked and active:
                        view = "DETAIL"
                elif key in ("\t", "a", "A"):
                    all_quants = not all_quants
                    p_idx = p_scroll = 0
                elif key in (KEY_ESC, "q", "Q", "\x1b[D", "\x1bOD") or key in KEYS_BACKSPACE:
                    view, scores = "MODELS", []
                elif key in KEYS_ENTER and active:
                    view = "DETAIL"
                elif key == KEY_CTRL_C:
                    break

            else:  # DETAIL
                assert selected is not None
                draw_detail(active[p_idx], selected["id"], width)
                if get_key() == KEY_CTRL_C:
                    break
                view = "PROVIDERS"
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
