"""ProviderScore scoring model: expected cost per conversation turn for one endpoint.

    tokenCost   = [ C·(h·hitPrice + (1 − h)·missPrice) + O·out ] / 1e6 + requestFee
    timeCost    = (timeValue$/hr / 3600) · (ttft + O / throughput)
    failureCost = (1 − uptime) · [ C·h·(missPrice − hitPrice) / 1e6 + (timeValue$/hr / 3600)·ttft ]
    totalCost   = tokenCost + timeCost + failureCost

where ``C`` is prompt tokens per turn, ``O`` completion tokens per turn, and ``h`` the
endpoint's published 24-hour token-weighted cache hit rate, used as observed.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ._util import price_per_million


@dataclass
class ScoringConfig:
    """Knobs for the ProviderScore model.

    :param prompt_tokens: Prompt/context tokens per turn (C).
    :param completion_tokens: Completion tokens per turn (O).
    :param time_value_usd_per_hour: Opportunity cost of wall-clock time. 0 disables timeCost.
    :param price_failures: Charge for expected retries using the endpoint's 24h uptime.
    :param apply_discount: Apply the endpoint's advertised discount to all prices.
    """

    prompt_tokens: int = 2000
    completion_tokens: int = 500
    time_value_usd_per_hour: float = 0.0
    price_failures: bool = True
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
    discount: float = 0.0                      # fraction, e.g. 0.5 for 50% off (informational)

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


@dataclass
class ScoreBreakdown:
    """Result of evaluating one endpoint. All ``*_usd`` values are per turn."""

    provider_name: str
    provider_slug: str
    endpoint_id: str

    hit_price: float   # $/M charged for cached prompt tokens
    miss_price: float  # $/M charged for uncached prompt tokens
    cache_hit_rate: float  # published 24h token-weighted cache hit rate, 0..1 (0 if no cache)

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

    # -- per-million-token views: the same costs normalised by the turn's token count, so
    #    they read on the same scale as the catalog's $/M prices instead of as tiny per-turn sums.

    def _per_m(self, usd: float) -> float:
        tokens = self.prompt_tokens + self.completion_tokens
        return usd / tokens * 1_000_000 if tokens else 0.0

    @property
    def token_cost_per_m(self) -> float:
        return self._per_m(self.token_cost_usd)

    @property
    def time_cost_per_m(self) -> float:
        return self._per_m(self.time_cost_usd)

    @property
    def failure_cost_per_m(self) -> float:
        return self._per_m(self.failure_cost_usd)

    @property
    def total_cost_per_m(self) -> float:
        return self._per_m(self.total_cost_usd)

    @property
    def formatted_token_cost(self) -> str:
        return f"${self.token_cost_per_m:.4f}"

    @property
    def formatted_time_cost(self) -> str:
        return f"${self.time_cost_per_m:.4f}"

    @property
    def formatted_failure_cost(self) -> str:
        return f"${self.failure_cost_per_m:.4f}"

    @property
    def formatted_total_cost(self) -> str:
        return f"${self.total_cost_per_m:.4f}"

    @property
    def formatted_cache_hit_rate(self) -> str:
        return f"{self.cache_hit_rate * 100.0:.1f}%"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "provider_slug": self.provider_slug,
            "endpoint_id": self.endpoint_id,
            "rank": self.rank,
            "quantization": self.quantization,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "hit_price_per_m": round(self.hit_price, 6),
            "miss_price_per_m": round(self.miss_price, 6),
            "completion_price_per_m": round(self.out_price, 6),
            "token_cost_usd": round(self.token_cost_usd, 8),
            "time_cost_usd": round(self.time_cost_usd, 8),
            "failure_cost_usd": round(self.failure_cost_usd, 8),
            "total_cost_usd": round(self.total_cost_usd, 8),
            "token_cost_per_m": round(self.token_cost_per_m, 6),
            "time_cost_per_m": round(self.time_cost_per_m, 6),
            "failure_cost_per_m": round(self.failure_cost_per_m, 6),
            "total_cost_per_m": round(self.total_cost_per_m, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_seconds": self.ttft_seconds,
            "throughput_tps": self.throughput_tps,
            "uptime_pct": self.uptime_pct,
        }


def evaluate_endpoint(
    pricing: EndpointPricing,
    cache_hit_rate: Optional[float] = None,
    ttft_seconds: Optional[float] = None,
    throughput_tps: Optional[float] = None,
    uptime_pct: Optional[float] = None,
    config: Optional[ScoringConfig] = None,
    provider_name: str = "Unknown",
    provider_slug: str = "unknown",
    endpoint_id: str = "",
    quantization: str = "unknown",
) -> ScoreBreakdown:
    """Score one endpoint with the ProviderScore model (see module docstring).

    :param cache_hit_rate: Published 24h token-weighted hit rate in 0..1, used as observed.
                           ``None`` (no observation) is treated as a cold cache.
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
        h = 0.0
    else:
        hit_price = pricing.input_cache_read
        miss_price = pricing.input_cache_write if pricing.input_cache_write is not None else pricing.prompt
        h = 0.0 if cache_hit_rate is None else max(0.0, min(1.0, float(cache_hit_rate)))

    # Token cost
    effective_prompt_price = h * hit_price + (1.0 - h) * miss_price
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
        prefix_loss = C * h * (miss_price - hit_price) / 1_000_000.0
        failure_cost = failure_prob * (prefix_loss + per_sec * ttft)

    return ScoreBreakdown(
        provider_name=provider_name,
        provider_slug=provider_slug,
        endpoint_id=endpoint_id,
        hit_price=hit_price,
        miss_price=miss_price,
        cache_hit_rate=h,
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
