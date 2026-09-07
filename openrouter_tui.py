#!/usr/bin/env python3
"""openrouter-tui - interactive terminal explorer for OpenRouter models and providers.

Views:
  1. Models     - the catalog ranked by the cost vs. quality Pareto frontier (frontier models
                  first, cheapest to most expensive, with the knee point highlighted; then dominated
                  models by distance to the frontier). Type to fuzzy-filter; Tab toggles the
                  benchmark metric between intelligence and coding.
  2. Providers  - Enter on a model lists its endpoints ranked by ProviderScore scored cost.
                  Tab or 'a' toggles between the primary quantization and all variants.
  3. Detail     - Enter on a provider shows the full cost breakdown and pricing.

Keys: Up/Down or Ctrl-P/N move, PgUp/PgDn jump 5, mouse wheel/click, Enter select,
Esc/Backspace back (Esc clears the search first), Ctrl-C quit.

Runs in the alternate screen buffer so the normal terminal scrollback is untouched.
"""

import atexit
import os
import re
import select
import shutil
import signal
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


def _cleanup_terminal() -> None:
    sys.stdout.write(ALT_SCREEN_OFF)
    sys.stdout.flush()


atexit.register(_cleanup_terminal)


def _signal_handler(signum: int, frame: Any) -> None:
    _cleanup_terminal()
    sys.exit(0)


signal.signal(signal.SIGTERM, _signal_handler)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _signal_handler)
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
WHEEL = 1  # mouse wheel scroll step
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
        raw = os.read(fd, 1)
        if not raw:
            raise EOFError("stdin closed")
        seq = raw.decode("utf-8", errors="ignore")
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
            if 64 <= button < 80 and (button - 64) % 4 == 0:  # wheel up, incl. modifier variants
                idx = max(0, idx - WHEEL)
            elif 65 <= button < 80 and (button - 65) % 4 == 0:  # wheel down
                idx = min(last, idx + WHEEL)
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
    Column("Token $", 9, ">"),
    Column("Fail $", 8, ">"),
    Column("Time $", 8, ">"),
    Column("Cache Hit", 9, ">"),
    Column("TTFT", 7, ">"),
    Column("TPS", 5, ">"),
    Column("Turn Time", 9, ">"),
    Column("Task Time", 9, ">"),
    Column("Uptime", 7, ">"),
    Column("Input $/M", 11, ">"),
    Column("Output $/M", 11, ">"),
    Column("Cache $/M", 11, ">"),
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

    if width < 120:
        name_w = max(18, min(28, width // 4))
    else:
        name_w = min(max([MODEL_NAME_MIN_W] + [len(m["name"]) for m in models]), 50)
    cols = [MODEL_STATUS_COL, Column("Model Name", name_w), MODEL_ID_COL,
            Column("Intel Score" if metric == "intelligence" else "Coding Score", 12, ">"),
            Column("Input $/M", 12, ">"), Column("Output $/M", 11, ">"),
            Column("Cache $/M", 11, ">"), Column("Blend $/M", 11, ">")]
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
            status = "★ OPTIMAL KNEE" if efficient else "★ OPTIMAL" if front else f"-{m['dist']:.1f} pts"
            line = format_row(
                [status, m["name"][:name_w], m["id"][:30], f"{m['score']:.1f}", _usd(m["cost"]), _usd(m["compl_p"] or 0.0),
                 _usd(m["cache_p"]) if m["cache_p"] is not None else "-",
                 _usd(m["blend_p"]) if m["blend_p"] is not None else "-"],
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
        (BOLD_YELLOW, "★ OPTIMAL KNEE"), ("", " best quality gain per dollar   "),
        (GREEN, "★ OPTIMAL"), ("", " on the cost/quality frontier   "),
        ("", "-N pts is distance from frontier, score units"),
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
            cache_p = _usd(s.read_price) if (s.has_cache or s.read_price < s.input_price) else "-"
            line = format_row(
                [
                    s.provider_name[:16], s.formatted_task_cost, s.formatted_token_cost, s.formatted_failure_cost,
                    s.formatted_time_cost, s.formatted_cache_hit_rate, fmt_seconds(s.ttft_seconds),
                    fmt_tps(s.throughput_tps), s.formatted_turn_time, s.formatted_task_time, fmt_pct(s.uptime_pct),
                    _usd(s.input_price), _usd(s.out_price), cache_p,
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

    # Split the task cost into the model's variables and normalise each to
    # USD per 1M submitted-or-generated tokens (the denominator used for
    # task_cost_per_m), so the breakdown is comparable across providers.
    submitted = s.submitted_tokens or 1
    per_m = lambda usd: f"${usd / submitted * 1e6:.4f}"

    fixed_m = per_m(s.fixed_cost_usd)        # write new tokens + generate output (+ request fee)
    read_m = per_m(s.read_baseline_usd)      # cache-read baseline (prefix @ cache-read price)
    miss_m = per_m(s.miss_premium_usd)       # cache-miss premium (missed prefix @ cache-write price)
    time_m = per_m(s.time_cost_usd)          # decode + prefill + overhead
    fail_m = per_m(s.failure_cost_usd)       # failure premium + return penalty

    # weighted /M: idealized blended price = pure token billing assuming perfect cache
    # (no misses, no time charge, no failures). It is the sum of the token-only terms.
    weighted_m = per_m(s.fixed_cost_usd + s.read_baseline_usd)
    # adj weighted/M: the realised task cost per 1M tokens (== task_cost_per_m).
    adj_m = per_m(s.task_cost_usd)

    # Aligned label/value rows; every value shares the same starting column.
    rows = []
    def row(label, value, note="", indent=2):
        rows.append((indent, label, value, note))

    row("Expected Task Cost:", f"\x1b[1;32m{s.formatted_task_cost}{RESET}",
        f"(per turn {_usd(s.mean_turn_cost_usd)}, per 1M submitted tok ${s.task_cost_per_m:.4f})")
    row("Expected Task Time:", s.formatted_task_time,
        f"(per turn {s.formatted_turn_time}; published TTFT {fmt_seconds(s.ttft_seconds)} shown, not charged)")
    row("weighted/M (idealized):", weighted_m,
        "(= fixed/M + cache-read/M; no misses, no time, no failures)")
    row("adj weighted/M:", adj_m, f"(task {s.formatted_task_cost} / {submitted / 1e6:.2f}M submitted tok)")
    row("fixed/M:", fixed_m, "(write new tokens + generate output)", indent=4)
    row("cache-read/M:", read_m, "(prefix @ cache-read price)", indent=4)
    row("cache-miss/M:", miss_m, f"(real {s.formatted_cache_hit_rate} hit rate; missed prefix @ cache-write)", indent=4)
    row("time/M:", time_m, "(decode + prefill + overhead)", indent=4)
    row("failure/M:", fail_m, f"(uptime {fmt_pct(s.uptime_pct)})", indent=4)
    row("fixed (new tok + output):", _usd(s.fixed_cost_usd), indent=4)
    time_note = (
        f"(gen {_usd(s.decode_cost_usd)}, prefill {_usd(s.prefill_new_cost_usd)}, miss {_usd(s.prefill_miss_cost_usd)})"
        if width < 110 else
        f"(gen {_usd(s.decode_cost_usd)}, prefill new tokens {_usd(s.prefill_new_cost_usd)}, "
        f"re-prefill on cache miss {_usd(s.prefill_miss_cost_usd)}, overhead {_usd(s.ttft_cost_usd)})"
    )
    row("time:", _usd(s.time_cost_usd), time_note, indent=4)
    row("cached-read baseline:", _usd(s.read_baseline_usd), indent=4)
    row("cache-miss premium:", _usd(s.miss_premium_usd), indent=4)
    row("failure premium + return:", _usd(s.failure_cost_usd), indent=4)
    row("Cache Hit Rate (24h):", f"{BOLD}{s.formatted_cache_hit_rate}{RESET}"
        + (f"   imputed: {', '.join(s.imputed)}" if s.imputed else ""))
    row("Quantization Variant:", s.quantization.upper())

    # value column = one past the longest (indent + label) prefix, so every value lines up
    value_col = max(len(" " * indent + label) for indent, label, _, _ in rows) + 2

    write(f"\x1b[1;36mProvider Detail: {s.provider_name}{RESET}  (for {model_id})"[: width - 1])
    write(rule[: width - 1])
    write(f"  Task: {s.turns} turns × ({s.new_tokens} new + {s.completion_tokens} out tokens), routing {s.routing}"[: width - 1])
    write(f"  per 1M token:  input/M: ${s.input_price:.4f}  |  output/M: ${s.out_price:.4f}  |  "
          f"cache/M: ${s.read_price:.4f}  |  weighted/M: {weighted_m}"[: width - 1])
    for indent, label, value, note in rows:
        prefix = " " * indent + label
        line = prefix + " " * (value_col - len(prefix)) + value
        if note:
            line += f"  {note}"
        write(line[: width - 1])
    write(f"  Bounds: perfect cache {_usd(s.perfect_cache_cost_usd)}  |  cold cache {_usd(s.cold_cache_cost_usd)}"[: width - 1])
    write(f"  Risk: σ_proc {_usd(s.sigma_proc_usd)}  |  σ_par {_usd(s.sigma_par_usd)}  |  P(migrate to fallback) {s.migration_probability:.0%}"[: width - 1])
    write(f"  E[TTFT]: {fmt_seconds(s.ttft_seconds)}  |  Throughput p50: {fmt_tps(s.throughput_tps, ' TPS')}  |  Uptime: {fmt_pct(s.uptime_pct)}"[: width - 1])
    write(rule[: width - 1])
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
    scores: Optional[List[ScoreBreakdown]] = None
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
                        selected, view, scores, p_idx, p_scroll = filtered[m_idx], "PROVIDERS", None, 0, 0
                elif key == "\t":
                    metric = "coding" if metric == "intelligence" else "intelligence"
                    models, frontier_count = build_model_list(metric)
                    m_idx = m_scroll = 0
                elif key in KEYS_ENTER and filtered:
                    selected, view, scores, p_idx, p_scroll = filtered[m_idx], "PROVIDERS", None, 0, 0
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
                if scores is None:
                    clear_screen()
                    write(f"Fetching provider analytics for {selected['id']}...")
                    sys.stdout.flush()
                    try:
                        scores = score_model_providers(selected["permaslug"], config=ScoringConfig())
                    except Exception as e:
                        clear_screen()
                        write(f"{BOLD}Failed to fetch provider analytics for {selected['id']}{RESET}")
                        write(f"{DIM}{e}{RESET}")
                        write()
                        write(help_line([("Any key", "back to models")], width - 1))
                        sys.stdout.flush()
                        get_key()
                        view, scores = "MODELS", None
                        continue
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
                    view, scores = "MODELS", None
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
    if not sys.stdin.isatty():
        sys.stderr.write("Error: openrouter-tui requires an interactive terminal.\n")
        sys.exit(1)
    try:
        run_tui()
    except (KeyboardInterrupt, SystemExit, EOFError):
        pass


if __name__ == "__main__":
    main()
