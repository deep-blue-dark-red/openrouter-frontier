"""ProviderUtility scoring model: expected cost per conversation turn for one endpoint.

    tokenCost   = [ C·(h_used·hitPrice + (1 − h_used)·missPrice) + O·out ] / 1e6 + requestFee
    timeCost    = (timeValue$/hr / 3600) · (ttft + O / throughput)
    failureCost = (1 − uptime) · [ C·h_used·(missPrice − hitPrice) / 1e6 + (timeValue$/hr / 3600)·ttft ]
    totalCost   = tokenCost + timeCost + failureCost

where ``C`` is prompt tokens per turn, ``O`` completion tokens per turn, and ``h_used`` the
Bayesian-shrunk cache hit rate ``(h·T + prior·W) / (T + W)``.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ._util import price_per_million


@dataclass
class ScoringConfig:
    """Knobs for the ProviderUtility model.

    :param prompt_tokens: Prompt/context tokens per turn (C).
    :param completion_tokens: Completion tokens per turn (O).
    :param time_value_usd_per_hour: Opportunity cost of wall-clock time. 0 disables timeCost.
    :param price_failures: Charge for expected retries using the endpoint's 24h uptime.
    :param prior: Prior belief for the cache hit rate before observing traffic.
    :param prior_weight_tokens: Pseudo-count W (in tokens) behind the prior. Endpoints that
                                served fewer than W tokens are pulled toward the prior.
    :param apply_discount: Apply the endpoint's advertised discount to all prices.
    """

    prompt_tokens: int = 2000
    completion_tokens: int = 500
    time_value_usd_per_hour: float = 0.0
    price_failures: bool = True
    prior: float = 0.5
    prior_weight_tokens: float = 1e9
    apply_discount: bool = True

    @property
    def time_value_per_second(self) -> float:
        return self.time_value_usd_per_hour / 3600.0 if self.time_value_usd_per_hour > 0 else 0.0


@dataclass
class EndpointPricing:
    """Per-endpoint list prices in USD per million tokens."""

    prompt: float
    completion: float
    input_cache_read: Optional[float] = None   # absent => endpoint has no prompt cache
    input_cache_write: Optional[float] = None  # absent => cache writes billed as normal input
    request_fee: float = 0.0                   # fixed USD per request
    discount: float = 0.0                      # fraction, e.g. 0.5 for 50% off

    @classmethod
    def from_api_dict(cls, p: Dict[str, Any], apply_discount: bool = True) -> "EndpointPricing":
        """Parse a ``pricing`` object from the OpenRouter endpoints API."""
        prompt = price_per_million(p.get("prompt")) or 0.0
        completion = price_per_million(p.get("completion")) or 0.0
        read = price_per_million(p.get("input_cache_read"))
        write = price_per_million(p.get("input_cache_write"))
        request_fee = float(p.get("request") or 0.0)
        discount = float(p.get("discount") or 0.0)

        if apply_discount and discount > 0:
            mult = max(0.0, 1.0 - discount)
            prompt *= mult
            completion *= mult
            read = read * mult if read is not None else None
            write = write * mult if write is not None else None

        return cls(prompt, completion, read, write, request_fee, discount)


@dataclass
class ScoreBreakdown:
    """Result of evaluating one endpoint. All ``*_usd`` values are per turn."""

    provider_name: str
    provider_slug: str
    endpoint_id: str

    hit_price: float   # $/M charged for cached prompt tokens
    miss_price: float  # $/M charged for uncached prompt tokens
    h_raw: float       # published 24h cache hit rate, 0..1
    h_used: float      # shrunk cache hit rate actually used

    token_cost_usd: float
    time_cost_usd: float
    failure_cost_usd: float
    total_cost_usd: float

    prompt_tokens: int
    completion_tokens: int
    uptime_pct: Optional[float] = None
    ttft_seconds: Optional[float] = None
    throughput_tps: Optional[float] = None
    out_price: float = 0.0
    quantization: str = "unknown"
    rank: int = 0

    @property
    def formatted_token_cost(self) -> str:
        return f"${self.token_cost_usd:.6f}"

    @property
    def formatted_time_cost(self) -> str:
        return f"${self.time_cost_usd:.6f}"

    @property
    def formatted_failure_cost(self) -> str:
        return f"${self.failure_cost_usd:.6f}"

    @property
    def formatted_total_cost(self) -> str:
        return f"${self.total_cost_usd:.6f}"

    @property
    def formatted_h_raw(self) -> str:
        return f"{self.h_raw * 100.0:.1f}%"

    @property
    def formatted_h_used(self) -> str:
        return f"{self.h_used * 100.0:.1f}%"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "provider_slug": self.provider_slug,
            "endpoint_id": self.endpoint_id,
            "rank": self.rank,
            "quantization": self.quantization,
            "h_raw": round(self.h_raw, 4),
            "h_used": round(self.h_used, 4),
            "hit_price_per_m": round(self.hit_price, 6),
            "miss_price_per_m": round(self.miss_price, 6),
            "completion_price_per_m": round(self.out_price, 6),
            "token_cost_usd": round(self.token_cost_usd, 8),
            "time_cost_usd": round(self.time_cost_usd, 8),
            "failure_cost_usd": round(self.failure_cost_usd, 8),
            "total_cost_usd": round(self.total_cost_usd, 8),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_seconds": self.ttft_seconds,
            "throughput_tps": self.throughput_tps,
            "uptime_pct": self.uptime_pct,
        }


def shrink_hit_rate(h: float, total_tokens: float, prior: float, prior_weight_tokens: float) -> float:
    """Bayesian shrinkage of an observed hit rate toward ``prior``.

    ``h_used = (h·T + prior·W) / (T + W)``. With ``T = 0`` this returns the prior; as ``T``
    grows past ``W`` the observed rate dominates.
    """
    T = max(0.0, float(total_tokens))
    W = max(0.0, float(prior_weight_tokens))
    if T + W <= 0:
        return prior
    return (h * T + prior * W) / (T + W)


def evaluate_endpoint(
    pricing: EndpointPricing,
    cache_hit_rate: Optional[float] = None,
    total_tokens: int = 0,
    ttft_seconds: Optional[float] = None,
    throughput_tps: Optional[float] = None,
    uptime_pct: Optional[float] = None,
    config: Optional[ScoringConfig] = None,
    provider_name: str = "Unknown",
    provider_slug: str = "unknown",
    endpoint_id: str = "",
    quantization: str = "unknown",
) -> ScoreBreakdown:
    """Score one endpoint with the ProviderUtility model (see module docstring).

    :param cache_hit_rate: Published 24h hit rate in 0..1. ``None`` falls back to the prior.
    :param total_tokens: 24h tokens served by this endpoint; drives shrinkage weight.
    :param ttft_seconds: Median time to first token.
    :param throughput_tps: Median output tokens per second.
    :param uptime_pct: 24h uptime in 0..100.
    """
    cfg = config or ScoringConfig()
    C = cfg.prompt_tokens
    O = cfg.completion_tokens

    # Derived prices. Without a cache-read price the endpoint has no cache: every prompt
    # token is a miss at the input price and the hit rate is forced to zero.
    if pricing.input_cache_read is None:
        hit_price = miss_price = pricing.prompt
        h_raw = h_used = 0.0
    else:
        hit_price = pricing.input_cache_read
        miss_price = pricing.input_cache_write if pricing.input_cache_write is not None else pricing.prompt
        h_raw = cfg.prior if cache_hit_rate is None else max(0.0, min(1.0, float(cache_hit_rate)))
        h_used = shrink_hit_rate(h_raw, total_tokens, cfg.prior, cfg.prior_weight_tokens)

    # Token cost
    effective_prompt_price = h_used * hit_price + (1.0 - h_used) * miss_price
    token_cost = (C * effective_prompt_price + O * pricing.completion) / 1_000_000.0 + pricing.request_fee

    # Time cost: waiting for the first token plus streaming the completion.
    per_sec = cfg.time_value_per_second
    ttft = max(0.0, ttft_seconds or 0.0)
    time_cost = 0.0
    if per_sec > 0:
        gen_time = (O / throughput_tps) if throughput_tps and throughput_tps > 0 else 0.0
        time_cost = per_sec * (ttft + gen_time)

    # Failure cost: with probability (1 − uptime) the request fails and is retried elsewhere,
    # losing the cached prefix (pay miss instead of hit on the cached share) and the wait.
    failure_cost = 0.0
    if cfg.price_failures and uptime_pct is not None:
        failure_prob = 1.0 - max(0.0, min(1.0, uptime_pct / 100.0))
        prefix_loss = C * h_used * (miss_price - hit_price) / 1_000_000.0
        failure_cost = failure_prob * (prefix_loss + per_sec * ttft)

    return ScoreBreakdown(
        provider_name=provider_name,
        provider_slug=provider_slug,
        endpoint_id=endpoint_id,
        hit_price=hit_price,
        miss_price=miss_price,
        h_raw=h_raw,
        h_used=h_used,
        token_cost_usd=token_cost,
        time_cost_usd=time_cost,
        failure_cost_usd=failure_cost,
        total_cost_usd=token_cost + time_cost + failure_cost,
        prompt_tokens=C,
        completion_tokens=O,
        uptime_pct=uptime_pct,
        ttft_seconds=ttft_seconds,
        throughput_tps=throughput_tps,
        out_price=pricing.completion,
        quantization=quantization,
    )
