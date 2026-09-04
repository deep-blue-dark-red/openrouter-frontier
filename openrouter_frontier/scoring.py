"""ProviderScore: expected cost of a whole *task* on one endpoint.

Implements docs/task_cost_model.tex. A task is N turns; turn k appends ``a`` new prompt
tokens and ``o`` completion tokens, so the reusable prefix is ``S_k = (k-1)(a+o)``.

Per successful turn (whole-prefix caching, cached fraction H_k with mean h):

    X_ok  = f + w·a + c·o + v'(E[l] + o·E[s]) + S_k·[m − H_k·(m − r)]

Prices (USD/token): b ordinary input, r cache read, w cache write (= b if unpublished),
c completion, f request fee. Miss price m = w (prefix rewritten on a miss) or b (processed).

On failure (prob q = 1 − uptime) the turn is served cold by a fallback B, and under an
explicit provider *order* the next turn returns to A with a small return penalty; under
*sticky* routing (OpenRouter's default) the task stays on B from then on.

Task cost = sum of turn costs. The module reports its expectation and decomposition, a
process-variance bound (Bernoulli misses), an epistemic parameter term for uncertainty in h
(O(N^2) in standard deviation), and the objective J = E + λ_proc·σ_proc + λ_par·σ_par.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ._util import price_per_million

_Z90 = 1.2815515655446004  # Φ^{-1}(0.9)


# ----------------------------------------------------------------------------- config

@dataclass
class ScoringConfig:
    """Task profile and caller preferences.

    :param new_tokens_per_turn: ``a`` — prompt tokens appended each turn (user text, tool results).
    :param task_tokens: total transcript size the task grows to (context at the last turn).
    :param output_tokens: total completion tokens generated over the whole task. Output is a
                          small share of an agentic transcript: most of it is tool results.
    :param completion_tokens: ``o`` — completion tokens per turn. Derived from ``output_tokens``
                              and the number of turns unless given explicitly.
    :param turns: ``N`` — explicit number of turns. Derived as
                  ``(task_tokens - output_tokens) / a`` unless given explicitly.
    :param time_value_usd_per_hour: ``v`` — value of the caller's wall-clock time.
    :param prefill_multiplier: prompt-processing speed as a multiple of the endpoint's decode
                               throughput (prefill is ~100× faster than generation). A cache
                               miss re-prefills the whole prefix at this rate; a hit skips it.
    :param price_failures: charge for failures using the endpoint's 24h uptime.
    :param routing: ``"sticky"`` (OpenRouter default: a fallback becomes the new sticky
                    provider) or ``"order"`` (explicit provider order: return to the primary).
    :param miss_policy: ``"rewrite"`` (miss billed at the write price) or ``"process"``
                        (miss billed at the ordinary input price).
    :param cache_mode: ``"aggregate"`` (published hit rate), ``"cold"`` (h = 0), or
                       ``"assumed"`` (use ``assumed_hit_rate``).
    :param assumed_hit_rate: caller-supplied reusable-prefix hit rate for ``cache_mode="assumed"``.
    :param sigma_h: epistemic standard deviation of the hit rate (drift / workload mismatch).
    :param lambda_proc: risk aversion to process variance (realised misses and failures).
    :param lambda_par: risk aversion to parameter variance (being wrong about h).
    :param apply_discount: use net (discounted) prices; ``False`` recovers list prices.
    """

    new_tokens_per_turn: int = 2000
    task_tokens: int = 300_000
    output_tokens: int = 10_000
    completion_tokens: Optional[int] = None
    turns: Optional[int] = None
    time_value_usd_per_hour: float = 20.0
    prefill_multiplier: float = 100.0
    price_failures: bool = True
    routing: str = "sticky"
    miss_policy: str = "rewrite"
    cache_mode: str = "aggregate"
    assumed_hit_rate: float = 0.0
    sigma_h: float = 0.0
    lambda_proc: float = 0.0
    lambda_par: float = 0.0
    apply_discount: bool = True

    def __post_init__(self) -> None:
        if self.new_tokens_per_turn < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.completion_tokens is not None and self.completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if self.new_tokens_per_turn + self.completion_per_turn == 0:
            raise ValueError("a turn must contain at least one token")
        if self.routing not in ("sticky", "order"):
            raise ValueError("routing must be 'sticky' or 'order'")
        if self.miss_policy not in ("rewrite", "process"):
            raise ValueError("miss_policy must be 'rewrite' or 'process'")
        if self.cache_mode not in ("aggregate", "cold", "assumed"):
            raise ValueError("cache_mode must be 'aggregate', 'cold', or 'assumed'")
        if not 0.0 <= self.assumed_hit_rate <= 1.0:
            raise ValueError("assumed_hit_rate must be in [0, 1]")
        if self.sigma_h < 0 or self.lambda_proc < 0 or self.lambda_par < 0:
            raise ValueError("sigma_h and lambdas must be non-negative")
        if self.time_value_usd_per_hour < 0:
            raise ValueError("time value must be non-negative")
        if self.prefill_multiplier <= 0:
            raise ValueError("prefill_multiplier must be positive")
        if self.turns is not None and self.turns < 1:
            raise ValueError("turns must be >= 1")
        if self.task_tokens < 1:
            raise ValueError("task_tokens must be >= 1")

    # -- derived profile

    @property
    def n_turns(self) -> int:
        """``N``: explicit, else the number of turns that grows the transcript to ``task_tokens``.

        With ``o`` derived from the task's total output, ``N·a + output = task`` gives
        ``N = (task_tokens − output_tokens) / a``. With ``o`` given explicitly,
        ``N = task_tokens / (a + o)``.
        """
        if self.turns is not None:
            return self.turns
        if self.completion_tokens is not None:
            return max(1, round(self.task_tokens / (self.new_tokens_per_turn + self.completion_tokens)))
        if self.new_tokens_per_turn <= 0:
            return 1
        return max(1, round(max(0, self.task_tokens - self.output_tokens) / self.new_tokens_per_turn))

    @property
    def completion_per_turn(self) -> int:
        """``o``: explicit, else the task's total output spread evenly over the turns."""
        if self.completion_tokens is not None:
            return self.completion_tokens
        return max(0, round(self.output_tokens / self.n_turns))

    @property
    def growth_per_turn(self) -> int:
        """``d = a + o``."""
        return self.new_tokens_per_turn + self.completion_per_turn

    @property
    def time_value_per_second(self) -> float:
        return self.time_value_usd_per_hour / 3600.0 if self.time_value_usd_per_hour > 0 else 0.0

    @property
    def transcript_tokens(self) -> int:
        """Final transcript length ``N·d``."""
        return self.n_turns * self.growth_per_turn

    @property
    def submitted_tokens(self) -> int:
        """Total tokens submitted or generated over the task: ``Σ_k (S_k + a + o)``."""
        n, d = self.n_turns, self.growth_per_turn
        return n * d + d * n * (n - 1) // 2

    # -- backwards-compatible aliases (the old single-turn scorer's names)

    @property
    def prompt_tokens(self) -> int:
        return self.new_tokens_per_turn


# ----------------------------------------------------------------------------- pricing

@dataclass
class EndpointPricing:
    """Per-endpoint prices in USD per million tokens (net of discount unless told otherwise)."""

    prompt: float                              # b: ordinary uncached input
    completion: float                          # c
    input_cache_read: Optional[float] = None   # r; absent => endpoint has no prompt cache
    input_cache_write: Optional[float] = None  # w; absent => writes billed as ordinary input
    request_fee: float = 0.0                   # f, fixed USD per request
    discount: float = 0.0                      # informational fraction, e.g. 0.5 for 50% off

    @classmethod
    def from_api_dict(cls, p: Dict[str, Any], apply_discount: bool = True) -> "EndpointPricing":
        """Parse a ``pricing`` object from the OpenRouter endpoints API.

        The API quotes prices per token and **already net of any discount**; ``discount`` is
        informational. With ``apply_discount=True`` the prices are used as-is. With
        ``apply_discount=False`` they are divided by ``1 - discount`` to recover the list price.
        """
        prompt = price_per_million(p.get("prompt")) or 0.0
        completion = price_per_million(p.get("completion")) or 0.0
        read = price_per_million(p.get("input_cache_read"))
        write = price_per_million(p.get("input_cache_write"))
        request_fee = float(p.get("request") or 0.0)
        discount = float(p.get("discount") or 0.0)

        if not apply_discount and 0 < discount < 1:
            mult = 1.0 / (1.0 - discount)
            prompt *= mult
            completion *= mult
            read = read * mult if read is not None else None
            write = write * mult if write is not None else None

        return cls(prompt, completion, read, write, request_fee, discount)

    @property
    def has_cache(self) -> bool:
        return self.input_cache_read is not None

    @property
    def read_price(self) -> float:
        """r ($/M): cache read, or the input price when there is no cache."""
        return self.input_cache_read if self.input_cache_read is not None else self.prompt

    @property
    def write_price(self) -> float:
        """w ($/M): cache write, or the input price when unpublished / no cache."""
        if not self.has_cache:
            return self.prompt
        return self.input_cache_write if self.input_cache_write is not None else self.prompt

    def miss_price(self, policy: str) -> float:
        """m ($/M): what an uncached prefix token costs on a miss under ``policy``."""
        if not self.has_cache:
            return self.prompt
        return self.write_price if policy == "rewrite" else self.prompt


# ----------------------------------------------------------------------------- telemetry

@dataclass
class EndpointInputs:
    """Telemetry for one endpoint after the paper's input treatment (Section 3).

    Latencies in seconds, throughput in tokens/s, uptime in 0..100. ``None`` means missing
    and is filled by :func:`impute_missing` before scoring.
    """

    cache_hit_rate: Optional[float] = None    # h^obs, 0..1
    uptime_pct: Optional[float] = None        # u·100
    ttft_p50: Optional[float] = None          # seconds
    ttft_p90: Optional[float] = None
    tps_p50: Optional[float] = None
    tps_p90: Optional[float] = None
    imputed: List[str] = field(default_factory=list)

    @property
    def expected_ttft(self) -> Optional[float]:
        """E[l] from a lognormal fitted to the p50 and p90 (median if only one is known)."""
        return _lognormal_mean(self.ttft_p50, self.ttft_p90)

    @property
    def expected_seconds_per_token(self) -> Optional[float]:
        """E[s], s = 1/g, from a lognormal fitted to the throughput quantiles."""
        if not self.tps_p50 or self.tps_p50 <= 0:
            return None
        s50 = 1.0 / self.tps_p50
        s_other = 1.0 / self.tps_p90 if self.tps_p90 and self.tps_p90 > 0 else None
        return _lognormal_mean(s50, s_other)


def _lognormal_mean(median: Optional[float], p90_or_p10: Optional[float]) -> Optional[float]:
    """Mean of a lognormal with the given median and one other decile.

    σ = |ln(q / median)| / Φ^{-1}(0.9) works whether the second quantile is the 90th or the
    10th percentile: the model-page stats publish the *fast* tail for throughput and the
    *slow* tail for latency, and the log-ratio is symmetric.
    """
    if median is None or median <= 0:
        return None
    if p90_or_p10 is None or p90_or_p10 <= 0:
        return median
    sigma = abs(math.log(p90_or_p10 / median)) / _Z90
    return median * math.exp(sigma * sigma / 2.0)


def impute_missing(inputs: List[EndpointInputs]) -> None:
    """Assumption 5: missing latency / throughput / uptime take the worst observed value
    among the model's endpoints, in place. A cacheable endpoint with no hit rate is cold."""
    worst_ttft50 = max((i.ttft_p50 for i in inputs if i.ttft_p50 is not None), default=None)
    worst_ttft90 = max((i.ttft_p90 for i in inputs if i.ttft_p90 is not None), default=None)
    worst_tps50 = min((i.tps_p50 for i in inputs if i.tps_p50 is not None and i.tps_p50 > 0), default=None)
    worst_tps90 = min((i.tps_p90 for i in inputs if i.tps_p90 is not None and i.tps_p90 > 0), default=None)
    worst_up = min((i.uptime_pct for i in inputs if i.uptime_pct is not None), default=None)
    for i in inputs:
        if i.ttft_p50 is None and worst_ttft50 is not None:
            i.ttft_p50, i.ttft_p90 = worst_ttft50, worst_ttft90
            i.imputed.append("latency")
        if i.tps_p50 is None and worst_tps50 is not None:
            i.tps_p50, i.tps_p90 = worst_tps50, worst_tps90
            i.imputed.append("throughput")
        if i.uptime_pct is None and worst_up is not None:
            i.uptime_pct = worst_up
            i.imputed.append("uptime")
        if i.cache_hit_rate is None:
            i.cache_hit_rate = 0.0
            i.imputed.append("cache_hit_rate")


# ----------------------------------------------------------------------------- result

@dataclass
class ScoreBreakdown:
    """Expected cost of the task on one endpoint. All ``*_usd`` values are per task."""

    provider_name: str
    provider_slug: str
    endpoint_id: str

    # prices actually used, $/M
    input_price: float      # b
    read_price: float       # r
    write_price: float      # w
    miss_price: float       # m
    out_price: float        # c
    request_fee: float      # f, USD

    # inputs actually used
    cache_hit_rate: float           # h, 0..1 (after cache_mode)
    uptime_pct: Optional[float]     # u·100
    ttft_seconds: Optional[float]   # E[l]
    seconds_per_token: Optional[float]  # E[s]
    prefill_tps: Optional[float]        # prompt-processing rate assumed for misses
    throughput_tps: Optional[float]     # p50, for display
    imputed: List[str]

    # profile
    new_tokens: int
    completion_tokens: int
    turns: int
    routing: str

    # expected task cost and decomposition (sums to task_cost_usd)
    task_cost_usd: float
    fixed_cost_usd: float
    time_cost_usd: float
    read_baseline_usd: float
    miss_premium_usd: float
    failure_premium_usd: float
    return_penalty_usd: float

    # bounds and risk
    perfect_cache_cost_usd: float
    cold_cache_cost_usd: float
    sigma_proc_usd: float
    sigma_par_usd: float
    objective_usd: float
    migration_probability: float    # P(task ends on the fallback) under sticky routing

    # expected token totals over the task
    ordinary_tokens: float
    read_tokens: float
    write_tokens: float
    completion_tokens_total: float
    submitted_tokens: int

    quantization: str = "unknown"
    rank: int = 0

    # -- convenience views

    @property
    def mean_turn_cost_usd(self) -> float:
        return self.task_cost_usd / self.turns if self.turns else 0.0

    @property
    def task_cost_per_m(self) -> float:
        """Secondary: task cost normalised per 1M submitted-or-generated tokens."""
        return self.task_cost_usd / self.submitted_tokens * 1e6 if self.submitted_tokens else 0.0

    @property
    def token_cost_usd(self) -> float:
        """Everything that is billed for tokens (no time, no failure surcharge)."""
        return self.fixed_cost_usd + self.read_baseline_usd + self.miss_premium_usd

    @property
    def failure_cost_usd(self) -> float:
        return self.failure_premium_usd + self.return_penalty_usd

    # legacy names used by older callers
    @property
    def total_cost_usd(self) -> float:
        return self.task_cost_usd

    @property
    def total_cost_per_m(self) -> float:
        return self.task_cost_per_m

    @property
    def hit_price(self) -> float:
        return self.read_price

    @property
    def formatted_task_cost(self) -> str:
        return _fmt_usd(self.task_cost_usd)

    @property
    def formatted_total_cost(self) -> str:
        return self.formatted_task_cost

    @property
    def formatted_objective(self) -> str:
        return _fmt_usd(self.objective_usd)

    @property
    def formatted_token_cost(self) -> str:
        return _fmt_usd(self.token_cost_usd)

    @property
    def formatted_time_cost(self) -> str:
        return _fmt_usd(self.time_cost_usd)

    @property
    def formatted_failure_cost(self) -> str:
        return _fmt_usd(self.failure_cost_usd)

    @property
    def formatted_miss_premium(self) -> str:
        return _fmt_usd(self.miss_premium_usd)

    @property
    def formatted_cache_hit_rate(self) -> str:
        return f"{self.cache_hit_rate * 100.0:.1f}%"

    def to_dict(self) -> Dict[str, Any]:
        r6 = lambda x: None if x is None else round(x, 6)
        return {
            "provider_name": self.provider_name,
            "provider_slug": self.provider_slug,
            "endpoint_id": self.endpoint_id,
            "rank": self.rank,
            "quantization": self.quantization,
            "profile": {
                "new_tokens_per_turn": self.new_tokens,
                "completion_tokens": self.completion_tokens,
                "turns": self.turns,
                "transcript_tokens": self.turns * (self.new_tokens + self.completion_tokens),
                "submitted_tokens": self.submitted_tokens,
                "routing": self.routing,
            },
            "prices_per_m": {
                "input": r6(self.input_price), "cache_read": r6(self.read_price),
                "cache_write": r6(self.write_price), "miss": r6(self.miss_price),
                "completion": r6(self.out_price), "request_fee_usd": r6(self.request_fee),
            },
            "inputs": {
                "cache_hit_rate": round(self.cache_hit_rate, 4),
                "uptime_pct": self.uptime_pct,
                "expected_ttft_seconds": r6(self.ttft_seconds),
                "expected_seconds_per_token": r6(self.seconds_per_token),
                "prefill_tps": self.prefill_tps,
                "throughput_p50_tps": self.throughput_tps,
                "imputed": list(self.imputed),
            },
            "task_cost_usd": r6(self.task_cost_usd),
            "decomposition_usd": {
                "fixed": r6(self.fixed_cost_usd), "time": r6(self.time_cost_usd),
                "read_baseline": r6(self.read_baseline_usd), "miss_premium": r6(self.miss_premium_usd),
                "failure_premium": r6(self.failure_premium_usd), "return_penalty": r6(self.return_penalty_usd),
            },
            "perfect_cache_cost_usd": r6(self.perfect_cache_cost_usd),
            "cold_cache_cost_usd": r6(self.cold_cache_cost_usd),
            "sigma_proc_usd": r6(self.sigma_proc_usd),
            "sigma_par_usd": r6(self.sigma_par_usd),
            "objective_usd": r6(self.objective_usd),
            "migration_probability": round(self.migration_probability, 4),
            "expected_tokens": {
                "ordinary": round(self.ordinary_tokens), "cache_read": round(self.read_tokens),
                "cache_write": round(self.write_tokens), "completion": round(self.completion_tokens_total),
            },
            "mean_turn_cost_usd": r6(self.mean_turn_cost_usd),
            "task_cost_per_m": r6(self.task_cost_per_m),
        }


def _fmt_usd(x: float) -> str:
    if abs(x) >= 100:
        return f"${x:,.0f}"
    if abs(x) >= 1:
        return f"${x:,.2f}"
    return f"${x:.4f}"


# ----------------------------------------------------------------------------- the model

@dataclass
class _Ep:
    """Resolved per-endpoint scalars, USD per *token*."""
    b: float; r: float; w: float; m: float; c: float; f: float
    h: float; q: float; El: float; Es: float
    Ep: float  # seconds per prefilled prompt token (0 if unknown)


def _resolve(pricing: EndpointPricing, inp: EndpointInputs, cfg: ScoringConfig) -> _Ep:
    per_tok = 1e-6
    b = pricing.prompt * per_tok
    r = pricing.read_price * per_tok
    w = pricing.write_price * per_tok
    m = pricing.miss_price(cfg.miss_policy) * per_tok
    c = pricing.completion * per_tok
    f = pricing.request_fee

    if not pricing.has_cache:
        h = 0.0
    elif cfg.cache_mode == "cold":
        h = 0.0
    elif cfg.cache_mode == "assumed":
        h = cfg.assumed_hit_rate
    else:
        h = 0.0 if inp.cache_hit_rate is None else max(0.0, min(1.0, float(inp.cache_hit_rate)))

    q = 0.0
    if cfg.price_failures and inp.uptime_pct is not None:
        q = 1.0 - max(0.0, min(1.0, inp.uptime_pct / 100.0))

    El = inp.expected_ttft or 0.0
    Es = inp.expected_seconds_per_token or 0.0
    Ep = 1.0 / (cfg.prefill_multiplier * inp.tps_p50) if inp.tps_p50 and inp.tps_p50 > 0 else 0.0
    return _Ep(b, r, w, m, c, f, h, q, El, Es, Ep)


@dataclass
class _Acc:
    fixed: float = 0.0; time: float = 0.0; read: float = 0.0; miss: float = 0.0
    fail: float = 0.0; ret: float = 0.0
    ordinary: float = 0.0; reads: float = 0.0; writes: float = 0.0; comp: float = 0.0
    var: float = 0.0; par: float = 0.0

    @property
    def total(self) -> float:
        return self.fixed + self.time + self.read + self.miss + self.fail + self.ret


def _run(A: _Ep, B: _Ep, cfg: ScoringConfig, h_override: Optional[float] = None) -> _Acc:
    """Accumulate the expected task cost and its decomposition turn by turn.

    Direct evaluation of Proposition 1 (routing 'order') and Section 7 (routing 'sticky');
    the loop is exact and avoids the geometric-sum special cases.
    """
    a, o, N = cfg.new_tokens_per_turn, cfg.completion_per_turn, cfg.n_turns
    d = a + o
    vps = cfg.time_value_per_second
    hA = A.h if h_override is None else h_override
    hB = B.h if h_override is None else h_override
    q = A.q
    x = 1.0 - q
    acc = _Acc()

    # per-turn constants
    fixA = A.f + A.w * a + A.c * o
    timeA = vps * (A.El + o * A.Es)
    fixB_cold = B.f + B.w * a + B.c * o          # cold turn on B: new tokens written on B
    timeFail = vps * A.El + vps * (B.El + o * B.Es)  # wasted wait on A + the retry on B
    timeB = vps * (B.El + o * B.Es)
    piA = hA * A.r + (1.0 - hA) * A.m
    piB = hB * B.r + (1.0 - hB) * B.m
    prefA = vps * A.Ep                            # $ of wall-clock per re-prefilled prefix token
    prefB = vps * B.Ep
    ret_per = hA * (A.m - A.r + prefA) * d        # return penalty under 'order' (money + prefill time)
    var_missA = (A.m - A.r + prefA) ** 2 * hA * (1.0 - hA)
    var_missB = (B.m - B.r + prefB) ** 2 * hB * (1.0 - hB)

    on_A = 1.0  # P(still on A at turn k) under sticky; always 1 under order
    for k in range(1, N + 1):
        S = (k - 1) * d
        if cfg.routing == "order":
            p_ok, p_fail, p_B = (1.0 - q), q, 0.0
        else:
            p_ok, p_fail, p_B = on_A * x, on_A * q, 1.0 - on_A

        # --- successful warm turn on A
        acc.fixed += p_ok * fixA
        acc.time += p_ok * (timeA + (1.0 - hA) * S * prefA)   # missed prefix is re-prefilled
        acc.read += p_ok * A.r * S
        acc.miss += p_ok * (1.0 - hA) * (A.m - A.r) * S
        acc.reads += p_ok * hA * S
        acc.writes += p_ok * (a + ((1.0 - hA) * S if cfg.miss_policy == "rewrite" else 0.0))
        acc.ordinary += p_ok * ((1.0 - hA) * S if cfg.miss_policy == "process" else 0.0)
        acc.comp += p_ok * o
        acc.var += p_ok * var_missA * S * S
        acc.par += p_ok * (A.m - A.r + prefA) * S

        # --- failed on A, served cold by B
        acc.fixed += p_fail * fixB_cold
        acc.time += p_fail * (timeFail + S * prefB)        # whole prefix prefilled cold on B
        acc.read += p_fail * A.r * S                       # baseline at A's read price ...
        acc.fail += p_fail * (B.w - A.r) * S               # ... plus the cold surcharge
        acc.writes += p_fail * (S + a)
        acc.comp += p_fail * o
        if cfg.routing == "order" and k < N:
            acc.ret += p_fail * ret_per
        # mixture term of the law of total variance: (μ_fail − μ_ok)^2
        mu_ok = fixA + timeA + piA * S + (1.0 - hA) * S * prefA
        mu_fail = fixB_cold + timeFail + B.w * S + S * prefB + (ret_per if (cfg.routing == "order" and k < N) else 0.0)
        if cfg.routing == "order":
            acc.var += q * (1.0 - q) * (mu_fail - mu_ok) ** 2
        else:
            acc.var += on_A * q * x * (mu_fail - mu_ok) ** 2

        # --- warm turn on B (sticky only, after migration)
        if p_B > 0:
            acc.fixed += p_B * (B.f + B.w * a + B.c * o)
            acc.time += p_B * (timeB + (1.0 - hB) * S * prefB)
            acc.read += p_B * B.r * S
            acc.miss += p_B * (1.0 - hB) * (B.m - B.r) * S
            acc.reads += p_B * hB * S
            acc.writes += p_B * (a + ((1.0 - hB) * S if cfg.miss_policy == "rewrite" else 0.0))
            acc.ordinary += p_B * ((1.0 - hB) * S if cfg.miss_policy == "process" else 0.0)
            acc.comp += p_B * o
            acc.var += p_B * var_missB * S * S
            acc.par += p_B * (B.m - B.r + prefB) * S

        if cfg.routing == "sticky":
            on_A *= x
    return acc


def evaluate_endpoint(
    pricing: EndpointPricing,
    inputs: Optional[EndpointInputs] = None,
    config: Optional[ScoringConfig] = None,
    fallback_pricing: Optional[EndpointPricing] = None,
    fallback_inputs: Optional[EndpointInputs] = None,
    provider_name: str = "Unknown",
    provider_slug: str = "unknown",
    endpoint_id: str = "",
    quantization: str = "unknown",
    # legacy keyword arguments from the single-turn scorer
    cache_hit_rate: Optional[float] = None,
    ttft_seconds: Optional[float] = None,
    throughput_tps: Optional[float] = None,
    uptime_pct: Optional[float] = None,
) -> ScoreBreakdown:
    """Expected cost of the configured task on this endpoint.

    The fallback defaults to the endpoint itself, served cold (the paper's proxy for
    "retried later"); pass ``fallback_pricing``/``fallback_inputs`` for a real chain.
    """
    cfg = config or ScoringConfig()
    inp = inputs or EndpointInputs(
        cache_hit_rate=cache_hit_rate, uptime_pct=uptime_pct,
        ttft_p50=ttft_seconds, tps_p50=throughput_tps,
    )
    A = _resolve(pricing, inp, cfg)
    B = _resolve(fallback_pricing, fallback_inputs, cfg) if fallback_pricing is not None else A
    if fallback_pricing is None and fallback_inputs is not None:
        B = _resolve(pricing, fallback_inputs, cfg)

    acc = _run(A, B, cfg)
    perfect = _run(A, B, cfg, h_override=1.0).total
    cold = _run(A, B, cfg, h_override=0.0).total
    sigma_proc = math.sqrt(max(0.0, acc.var))
    sigma_par = cfg.sigma_h * acc.par
    objective = acc.total + cfg.lambda_proc * sigma_proc + cfg.lambda_par * sigma_par
    migration = 0.0 if cfg.routing == "order" else 1.0 - (1.0 - A.q) ** cfg.n_turns

    return ScoreBreakdown(
        provider_name=provider_name, provider_slug=provider_slug, endpoint_id=endpoint_id,
        input_price=pricing.prompt, read_price=pricing.read_price, write_price=pricing.write_price,
        miss_price=pricing.miss_price(cfg.miss_policy), out_price=pricing.completion,
        request_fee=pricing.request_fee,
        cache_hit_rate=A.h, uptime_pct=inp.uptime_pct,
        ttft_seconds=inp.expected_ttft, seconds_per_token=inp.expected_seconds_per_token,
        prefill_tps=(cfg.prefill_multiplier * inp.tps_p50) if inp.tps_p50 else None,
        throughput_tps=inp.tps_p50, imputed=list(inp.imputed),
        new_tokens=cfg.new_tokens_per_turn, completion_tokens=cfg.completion_per_turn,
        turns=cfg.n_turns, routing=cfg.routing,
        task_cost_usd=acc.total, fixed_cost_usd=acc.fixed, time_cost_usd=acc.time,
        read_baseline_usd=acc.read, miss_premium_usd=acc.miss, failure_premium_usd=acc.fail,
        return_penalty_usd=acc.ret,
        perfect_cache_cost_usd=perfect, cold_cache_cost_usd=cold,
        sigma_proc_usd=sigma_proc, sigma_par_usd=sigma_par, objective_usd=objective,
        migration_probability=migration,
        ordinary_tokens=acc.ordinary, read_tokens=acc.reads, write_tokens=acc.writes,
        completion_tokens_total=acc.comp, submitted_tokens=cfg.submitted_tokens,
        quantization=quantization,
    )
