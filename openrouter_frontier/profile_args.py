"""Shared argparse options for the task profile (docs/task_cost_model.tex, Section 1.4)."""

import argparse

from .scoring import ScoringConfig


def add_task_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("task profile")
    g.add_argument("-a", "--new-tokens", "-c", "--prompt-tokens", dest="new_tokens", type=int, default=2000,
                   help="New prompt tokens appended per turn, a (user text / tool results)")
    g.add_argument("--task-tokens", type=int, default=300_000,
                   help="Transcript size the task grows to (context at the last turn)")
    g.add_argument("--output-tokens", type=int, default=10_000,
                   help="Total completion tokens over the whole task; o = output_tokens / N")
    g.add_argument("-o", "--completion-tokens", type=int, default=None,
                   help="Completion tokens per turn, o (overrides --output-tokens)")
    g.add_argument("-N", "--turns", type=int, default=None,
                   help="Explicit number of turns (default: (task_tokens - output_tokens) / a)")
    g.add_argument("-t", "--time-value", type=float, default=20.0, help="Value of your time, USD per hour (0 disables)")
    g.add_argument("--prefill-multiplier", type=float, default=100.0,
                   help="Prompt-processing speed as a multiple of decode throughput; a cache miss re-prefills the prefix at this rate")
    g.add_argument("--no-failures", action="store_true", help="Do not price failures from 24h uptime")
    g.add_argument("--routing", choices=["sticky", "order"], default="sticky",
                   help="sticky = OpenRouter default, a fallback becomes sticky; order = explicit provider order, return to primary")
    g.add_argument("--miss-policy", choices=["rewrite", "process"], default="rewrite",
                   help="On a cache miss the prefix is rewritten (write price) or processed (input price)")
    g.add_argument("--cache", dest="cache_mode", choices=["aggregate", "cold", "assumed"], default="aggregate",
                   help="Hit rate source: published aggregate, cold (0), or --assumed-hit-rate")
    g.add_argument("--assumed-hit-rate", type=float, default=0.0, help="Reusable-prefix hit rate for --cache assumed")
    g.add_argument("--sigma-h", type=float, default=0.0, help="Epistemic std. dev. of the hit rate (drift / mismatch)")
    g.add_argument("--lambda-proc", type=float, default=0.0, help="Risk aversion to process variance")
    g.add_argument("--lambda-par", type=float, default=0.0, help="Risk aversion to parameter variance")
    g.add_argument("--no-discount", action="store_true", help="Use list pricing, ignoring endpoint discounts")


def config_from_args(args: argparse.Namespace) -> ScoringConfig:
    return ScoringConfig(
        new_tokens_per_turn=args.new_tokens,
        task_tokens=args.task_tokens,
        output_tokens=args.output_tokens,
        completion_tokens=args.completion_tokens,
        turns=args.turns,
        time_value_usd_per_hour=args.time_value,
        prefill_multiplier=args.prefill_multiplier,
        price_failures=not args.no_failures,
        routing=args.routing,
        miss_policy=args.miss_policy,
        cache_mode=args.cache_mode,
        assumed_hit_rate=args.assumed_hit_rate,
        sigma_h=args.sigma_h,
        lambda_proc=args.lambda_proc,
        lambda_par=args.lambda_par,
        apply_discount=not args.no_discount,
    )


def describe_profile(cfg: ScoringConfig) -> str:
    """One line summarising the task profile for table headers."""
    k = cfg.transcript_tokens
    size = f"{k / 1000:.0f}k" if k < 1_000_000 else f"{k / 1e6:.2f}M"
    parts = [
        f"Task: {cfg.n_turns} turns × ({cfg.new_tokens_per_turn} new + {cfg.completion_per_turn} out) → {size} context, "
        f"{cfg.n_turns * cfg.completion_per_turn / 1000:.0f}k output",
        f"Time: ${cfg.time_value_usd_per_hour:.0f}/hr",
        f"Routing: {cfg.routing}",
        f"Miss: {cfg.miss_policy}",
        f"Cache: {cfg.cache_mode}" + (f" ({cfg.assumed_hit_rate:.0%})" if cfg.cache_mode == "assumed" else ""),
    ]
    if cfg.sigma_h or cfg.lambda_proc or cfg.lambda_par:
        parts.append(f"σ_h={cfg.sigma_h:.2f} λ_proc={cfg.lambda_proc:g} λ_par={cfg.lambda_par:g}")
    if not cfg.price_failures:
        parts.append("Failures: ignored")
    if not cfg.apply_discount:
        parts.append("List prices")
    return "  •  ".join(parts)
