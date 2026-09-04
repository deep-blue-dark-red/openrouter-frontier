"""Data models for per-provider and per-model 24h statistics."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .scoring import EndpointPricing, ScoreBreakdown, ScoringConfig, evaluate_endpoint


def format_tokens(n: int) -> str:
    """Compact token count: 1.2B, 3.4M, 5.6K."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_latency(ms: Optional[float]) -> str:
    if ms is None:
        return "--"
    return f"{ms / 1000.0:.2f}s" if ms >= 1000 else f"{ms:.0f}ms"


def _fmt_tps(tps: Optional[float]) -> str:
    return f"{tps:.0f} tps" if tps is not None else "--"


def _fmt_pct(pct: Optional[float]) -> str:
    return f"{pct:.1f}%" if pct is not None else "--"


@dataclass
class ProviderStats:
    """One endpoint serving a model, with observed 24h metrics."""

    endpoint_id: str
    name: str
    slug: str
    effective_input_price: float   # $/M, traffic-weighted, from effective-pricing
    effective_output_price: float  # $/M
    cache_hit_rate: float          # 0..1
    total_tokens: int
    token_share: float = 0.0       # 0..1 share of the model's total tokens

    latency_p50_ms: Optional[float] = None
    latency_p90_ms: Optional[float] = None
    throughput_p50_tps: Optional[float] = None
    throughput_p90_tps: Optional[float] = None
    uptime_1d_pct: Optional[float] = None

    pricing: Optional[EndpointPricing] = None
    quantization: str = "unknown"

    @property
    def cache_hit_rate_pct(self) -> float:
        return self.cache_hit_rate * 100.0

    @property
    def formatted_cache_hit_rate(self) -> str:
        return f"{self.cache_hit_rate_pct:.1f}%"

    @property
    def formatted_input_price(self) -> str:
        return f"${self.effective_input_price:.4f}"

    @property
    def formatted_output_price(self) -> str:
        return f"${self.effective_output_price:.4f}"

    @property
    def formatted_tokens(self) -> str:
        return format_tokens(self.total_tokens)

    @property
    def formatted_token_share(self) -> str:
        return f"{self.token_share * 100.0:.1f}%"

    @property
    def formatted_latency(self) -> str:
        return _fmt_latency(self.latency_p50_ms)

    @property
    def formatted_tps(self) -> str:
        return _fmt_tps(self.throughput_p50_tps)

    @property
    def formatted_uptime(self) -> str:
        return _fmt_pct(self.uptime_1d_pct)

    def evaluate_score(self, config: Optional[ScoringConfig] = None) -> Optional[ScoreBreakdown]:
        """Score this endpoint with the ProviderScore model. ``None`` if pricing is unknown."""
        if not self.pricing:
            return None
        ttft = self.latency_p50_ms / 1000.0 if self.latency_p50_ms is not None else None
        return evaluate_endpoint(
            pricing=self.pricing,
            cache_hit_rate=self.cache_hit_rate,
            ttft_seconds=ttft,
            throughput_tps=self.throughput_p50_tps,
            uptime_pct=self.uptime_1d_pct,
            config=config or ScoringConfig(),
            provider_name=self.name,
            provider_slug=self.slug,
            endpoint_id=self.endpoint_id,
            quantization=self.quantization,
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "endpoint_id": self.endpoint_id,
            "name": self.name,
            "slug": self.slug,
            "quantization": self.quantization,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_hit_rate_pct": round(self.cache_hit_rate_pct, 2),
            "effective_input_price_per_m": self.effective_input_price,
            "effective_output_price_per_m": self.effective_output_price,
            "total_tokens": self.total_tokens,
            "token_share": round(self.token_share, 4),
            "token_share_pct": round(self.token_share * 100.0, 2),
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "throughput_p50_tps": self.throughput_p50_tps,
            "throughput_p90_tps": self.throughput_p90_tps,
            "uptime_1d_pct": self.uptime_1d_pct,
        }
        if self.pricing:
            d["pricing"] = {
                "prompt": self.pricing.prompt,
                "completion": self.pricing.completion,
                "input_cache_read": self.pricing.input_cache_read,
                "input_cache_write": self.pricing.input_cache_write,
                "request_fee": self.pricing.request_fee,
                "discount": self.pricing.discount,
            }
        return d


# Sort specs: canonical name -> (aliases, key, descending_by_default).
# Keys return tuples so that missing values always sort last regardless of direction.
_SORT_SPECS: Dict[str, Any] = {
    "cache": (("cache", "cache_hit_rate", "hit_rate"), lambda p: p.cache_hit_rate, True),
    "latency": (("latency", "lat"), lambda p: p.latency_p50_ms, False),
    "tps": (("tps", "throughput"), lambda p: p.throughput_p50_tps, True),
    "uptime": (("uptime", "up"), lambda p: p.uptime_1d_pct, True),
    "input": (("input", "input_price"), lambda p: p.effective_input_price, False),
    "output": (("output", "output_price"), lambda p: p.effective_output_price, False),
    "tokens": (("tokens", "volume", "total_tokens"), lambda p: p.total_tokens, True),
    "share": (("share", "token_share"), lambda p: p.token_share, True),
    "name": (("name", "provider"), lambda p: p.name.lower(), False),
}
_SORT_ALIASES = {alias: canon for canon, (aliases, _, _) in _SORT_SPECS.items() for alias in aliases}


@dataclass
class ModelStats:
    """All providers serving one model plus model-level aggregates."""

    model_id: str
    permaslug: str
    model_name: Optional[str]
    providers: List[ProviderStats]
    weighted_cache_hit_rate: float
    weighted_input_price: float
    weighted_output_price: float
    total_tokens: int
    input_chart_data: List[Dict[str, Any]]
    output_chart_data: List[Dict[str, Any]]

    avg_latency_p50_ms: Optional[float] = None
    avg_throughput_p50_tps: Optional[float] = None
    avg_uptime_1d_pct: Optional[float] = None

    @property
    def provider_count(self) -> int:
        return len(self.providers)

    @property
    def formatted_weighted_cache_hit_rate(self) -> str:
        return f"{self.weighted_cache_hit_rate * 100.0:.1f}%"

    @property
    def formatted_weighted_input_price(self) -> str:
        return f"${self.weighted_input_price:.4f}"

    @property
    def formatted_weighted_output_price(self) -> str:
        return f"${self.weighted_output_price:.4f}"

    @property
    def formatted_total_tokens(self) -> str:
        return format_tokens(self.total_tokens)

    @property
    def formatted_avg_latency(self) -> str:
        return _fmt_latency(self.avg_latency_p50_ms)

    @property
    def formatted_avg_tps(self) -> str:
        return _fmt_tps(self.avg_throughput_p50_tps)

    @property
    def formatted_avg_uptime(self) -> str:
        return _fmt_pct(self.avg_uptime_1d_pct)

    def get_provider(self, query: str) -> Optional[ProviderStats]:
        """Find a provider by exact slug/name, then by substring."""
        q = query.strip().lower()
        for p in self.providers:
            if q in (p.slug.lower(), p.name.lower()):
                return p
        for p in self.providers:
            if q in p.slug.lower() or q in p.name.lower():
                return p
        return None

    def score_providers(self, config: Optional[ScoringConfig] = None) -> List[ScoreBreakdown]:
        """Score every provider and return them ranked by total cost, cheapest first."""
        scores = [sb for sb in (p.evaluate_score(config) for p in self.providers) if sb is not None]
        scores.sort(key=lambda s: s.total_cost_usd)
        for idx, s in enumerate(scores, 1):
            s.rank = idx
        return scores

    def sort_by(
        self,
        field: str = "cache",
        descending: Optional[bool] = None,
        config: Optional[ScoringConfig] = None,
    ) -> List[ProviderStats]:
        """Return providers sorted by one field.

        Fields: ``cache``, ``score``/``cost``, ``token_cost``, ``latency``, ``tps``, ``uptime``,
        ``input``, ``output``, ``tokens``, ``share``, ``name``. Each field has a natural
        direction (best first); pass ``descending`` to override it. Providers with a missing
        value for the field always sort last.
        """
        f = field.lower()

        if f in ("score", "cost", "total_cost", "token_cost", "tokencost"):
            attr = "token_cost_usd" if f.startswith("token") else "total_cost_usd"
            cache: Dict[str, Optional[float]] = {}
            for p in self.providers:
                sb = p.evaluate_score(config)
                cache[p.endpoint_id] = getattr(sb, attr) if sb else None
            key: Callable[[ProviderStats], Any] = lambda p: cache[p.endpoint_id]
            desc_default = False
        else:
            canon = _SORT_ALIASES.get(f, "cache")
            _, key, desc_default = _SORT_SPECS[canon]

        desc = desc_default if descending is None else descending

        def sort_key(p: ProviderStats):
            v = key(p)
            if v is None:
                return (1, 0)
            if isinstance(v, str):
                return (0, v)
            return (0, -v if desc else v)

        return sorted(self.providers, key=sort_key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "permaslug": self.permaslug,
            "model_name": self.model_name,
            "provider_count": self.provider_count,
            "total_tokens": self.total_tokens,
            "weighted_cache_hit_rate": self.weighted_cache_hit_rate,
            "weighted_cache_hit_rate_pct": round(self.weighted_cache_hit_rate * 100.0, 2),
            "weighted_input_price_per_m": self.weighted_input_price,
            "weighted_output_price_per_m": self.weighted_output_price,
            "avg_latency_p50_ms": self.avg_latency_p50_ms,
            "avg_throughput_p50_tps": self.avg_throughput_p50_tps,
            "avg_uptime_1d_pct": self.avg_uptime_1d_pct,
            "providers": [p.to_dict() for p in self.providers],
        }
