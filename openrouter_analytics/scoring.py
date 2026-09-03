from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ScoringConfig:
    """
    Configuration for provider cost and utility scoring.
    
    :param prompt_tokens: Number of prompt/context tokens per turn (C). Default 2000.
    :param completion_tokens: Number of completion/output tokens per turn (O). Default 500.
    :param time_value_usd_per_hour: Economic value of user/agent time in USD/hr.
                                    0 leaves the pure token cost model. Default 0.0.
    :param price_failures: Whether to factor in failure risk based on endpoint uptime. Default True.
    :param prior: Prior belief for cache hit rate shrinkage. Default 0.5.
    :param prior_weight_tokens: Weight of the prior belief in tokens (W). Default 1e9 (1 billion tokens).
    :param apply_discount: Whether to apply provider discounts listed in pricing. Default True.
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
    """Pricing inputs per endpoint in USD per million tokens (from public endpoints API)."""
    prompt: float                       # Input price ($/M tokens)
    completion: float                   # Output price ($/M tokens)
    input_cache_read: Optional[float] = None   # Cache read price ($/M tokens)
    input_cache_write: Optional[float] = None  # Cache write price ($/M tokens)
    request_fee: float = 0.0            # Fixed fee per request ($)
    discount: float = 0.0               # Discount fraction (e.g. 0.5 for 50% off)

    @classmethod
    def from_api_dict(cls, p_dict: Dict[str, Any], apply_discount: bool = True) -> "EndpointPricing":
        """
        Parse pricing dictionary from OpenRouter endpoints API.
        Automatically scales $/token to $/million tokens if values are < 0.01.
        """
        def _scale(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                f = float(val)
                # OpenRouter API prices are per token (e.g. 0.000000075). Scale to $/M.
                return f * 1_000_000.0 if f < 0.01 else f
            except (ValueError, TypeError):
                return None

        prompt = _scale(p_dict.get("prompt")) or 0.0
        completion = _scale(p_dict.get("completion")) or 0.0
        read = _scale(p_dict.get("input_cache_read"))
        write = _scale(p_dict.get("input_cache_write"))
        req_fee = float(p_dict.get("request") or 0.0)
        discount = float(p_dict.get("discount") or 0.0)

        if apply_discount and discount > 0:
            mult = max(0.0, 1.0 - discount)
            prompt *= mult
            completion *= mult
            if read is not None:
                read *= mult
            if write is not None:
                write *= mult

        return cls(
            prompt=prompt,
            completion=completion,
            input_cache_read=read,
            input_cache_write=write,
            request_fee=req_fee,
            discount=discount,
        )


@dataclass
class ScoreBreakdown:
    """Detailed result of evaluating an endpoint's token cost and utility."""
    provider_name: str
    provider_slug: str
    endpoint_id: str

    # Derived caching parameters
    hit_price: float                    # $/M
    miss_price: float                   # $/M
    h_raw: float                        # Published cache hit rate (0.0 to 1.0)
    h_used: float                       # Shrunken cache hit rate used in evaluation

    # Cost components in USD per turn
    token_cost_usd: float
    time_cost_usd: float
    failure_cost_usd: float
    total_cost_usd: float

    # Raw metrics for reference
    prompt_tokens: int
    completion_tokens: int
    uptime_pct: Optional[float] = None
    ttft_seconds: Optional[float] = None
    throughput_tps: Optional[float] = None
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
            "h_raw": round(self.h_raw, 4),
            "h_used": round(self.h_used, 4),
            "hit_price_per_m": round(self.hit_price, 6),
            "miss_price_per_m": round(self.miss_price, 6),
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
    """
    Evaluate an endpoint using the ProviderUtility scoring model.

    Token-cost model:
      in        = pricing.prompt
      out       = pricing.completion
      read      = pricing.input_cache_read
      write     = pricing.input_cache_write
      hitPrice  = read ?? in
      missPrice = read is absent ? in : (write ?? in)
      h         = read is absent ? 0 : clamp(cacheHitRate, 0, 1)
      h_used    = (h · T + prior · W) / (T + W)  [or prior directly if no published rate]
      tokenCost = [ C · ( h_used · hitPrice + (1 − h_used) · missPrice ) + O · out ] / 1_000_000 + requestFee

    Full utility adds:
      timeCost    = (TimeValueUsdPerHour / 3600) · (ttft + O / throughput)
      failureCost = (1 − uptime) · ( C · h_used · (missPrice − hitPrice) / 1e6 + timeValuePerSecond · ttft )
      totalCost   = tokenCost + timeCost + failureCost
    """
    cfg = config or ScoringConfig()
    C = cfg.prompt_tokens
    O = cfg.completion_tokens

    inp = pricing.prompt
    out = pricing.completion
    read = pricing.input_cache_read
    write = pricing.input_cache_write
    request_fee = pricing.request_fee

    # 1. Derived prices
    hit_price = read if read is not None else inp
    miss_price = inp if read is None else (write if write is not None else inp)

    # 2. Raw cache hit rate h
    if read is None:
        # Without a cache-read price the endpoint has no cache
        h_raw = 0.0
        h_used = 0.0
    else:
        if cache_hit_rate is not None:
            h_raw = max(0.0, min(1.0, float(cache_hit_rate)))
        else:
            h_raw = cfg.prior

        # 3. Shrinkage before use: h_used = (h · T + prior · W) / (T + W)
        T = max(0, total_tokens)
        W = cfg.prior_weight_tokens
        if (T + W) > 0:
            h_used = (h_raw * T + cfg.prior * W) / (T + W)
        else:
            h_used = cfg.prior

    # 4. Token cost per turn
    prompt_effective_price = h_used * hit_price + (1.0 - h_used) * miss_price
    token_cost = (C * prompt_effective_price + O * out) / 1_000_000.0 + request_fee

    # 5. Time cost
    time_val_per_sec = cfg.time_value_per_second
    if time_val_per_sec > 0:
        ttft = max(0.0, ttft_seconds or 0.0)
        tps = throughput_tps if throughput_tps and throughput_tps > 0 else 0.0
        gen_time = (O / tps) if tps > 0 else 0.0
        time_cost = time_val_per_sec * (ttft + gen_time)
    else:
        time_cost = 0.0

    # 6. Failure risk cost
    if cfg.price_failures and uptime_pct is not None:
        uptime = max(0.0, min(1.0, uptime_pct / 100.0))
        failure_prob = 1.0 - uptime
        prefix_loss_usd = C * h_used * (miss_price - hit_price) / 1_000_000.0
        ttft = max(0.0, ttft_seconds or 0.0)
        wasted_wait_usd = time_val_per_sec * ttft
        failure_cost = failure_prob * (prefix_loss_usd + wasted_wait_usd)
    else:
        failure_cost = 0.0

    total_cost = token_cost + time_cost + failure_cost

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
        total_cost_usd=total_cost,
        prompt_tokens=C,
        completion_tokens=O,
        uptime_pct=uptime_pct,
        ttft_seconds=ttft_seconds,
        throughput_tps=throughput_tps,
        quantization=quantization,
    )
