from dataclasses import dataclass
from typing import Optional, List, Dict, Any


def format_tokens(n: int) -> str:
    """Format token count in human-readable compact notation (B, M, K)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


@dataclass
class ProviderStats:
    endpoint_id: str
    name: str
    slug: str
    effective_input_price: float   # in $ per million tokens
    effective_output_price: float  # in $ per million tokens
    cache_hit_rate: float          # 0.0 to 1.0
    total_tokens: int
    token_share: float = 0.0       # 0.0 to 1.0 relative to model total

    # Latency, Throughput (TPS), and Uptime
    latency_p50_ms: Optional[float] = None
    latency_p90_ms: Optional[float] = None
    throughput_p50_tps: Optional[float] = None
    throughput_p90_tps: Optional[float] = None
    uptime_1d_pct: Optional[float] = None

    @property
    def cache_hit_rate_pct(self) -> float:
        return self.cache_hit_rate * 100.0

    @property
    def formatted_cache_hit_rate(self) -> str:
        return f"{self.cache_hit_rate * 100.0:.1f}%"

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
        if self.latency_p50_ms is not None:
            if self.latency_p50_ms >= 1000:
                return f"{self.latency_p50_ms / 1000.0:.2f}s"
            return f"{self.latency_p50_ms:.0f}ms"
        return "--"

    @property
    def formatted_tps(self) -> str:
        if self.throughput_p50_tps is not None:
            return f"{self.throughput_p50_tps:.0f} tps"
        return "--"

    @property
    def formatted_uptime(self) -> str:
        if self.uptime_1d_pct is not None:
            return f"{self.uptime_1d_pct:.1f}%"
        return "--"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "name": self.name,
            "slug": self.slug,
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


@dataclass
class ModelStats:
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

    # Model-level aggregated latency/tps/uptime if available
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
        if self.avg_latency_p50_ms is not None:
            if self.avg_latency_p50_ms >= 1000:
                return f"{self.avg_latency_p50_ms / 1000.0:.2f}s"
            return f"{self.avg_latency_p50_ms:.0f}ms"
        return "--"

    @property
    def formatted_avg_tps(self) -> str:
        if self.avg_throughput_p50_tps is not None:
            return f"{self.avg_throughput_p50_tps:.0f} tps"
        return "--"

    @property
    def formatted_avg_uptime(self) -> str:
        if self.avg_uptime_1d_pct is not None:
            return f"{self.avg_uptime_1d_pct:.1f}%"
        return "--"

    def get_provider(self, query: str) -> Optional[ProviderStats]:
        """Find a provider by name, slug, or substring match."""
        q = query.strip().lower()
        for p in self.providers:
            if p.slug.lower() == q or p.name.lower() == q:
                return p
        for p in self.providers:
            if q in p.slug.lower() or q in p.name.lower():
                return p
        return None

    def sort_by(self, field: str = "cache", descending: bool = True) -> List[ProviderStats]:
        """
        Sort providers by:
        - 'cache' or 'hit_rate': cache hit rate
        - 'latency': p50 latency (lower is better, so default asc if requested)
        - 'tps' or 'throughput': p50 throughput (higher is better)
        - 'uptime': 1d uptime %
        - 'input_price': effective input price
        - 'output_price': effective output price
        - 'tokens' or 'volume': total tokens served
        - 'share': token share
        - 'name': provider name
        """
        f = field.lower()
        if f in ("cache", "cache_hit_rate", "hit_rate"):
            key = lambda p: p.cache_hit_rate
        elif f in ("latency", "lat"):
            # Put None at the end
            key = lambda p: (p.latency_p50_ms is None, p.latency_p50_ms or 999999)
            return sorted(self.providers, key=key, reverse=not descending)
        elif f in ("tps", "throughput"):
            key = lambda p: p.throughput_p50_tps or 0.0
        elif f in ("uptime", "up"):
            key = lambda p: p.uptime_1d_pct or 0.0
        elif f in ("input_price", "input"):
            key = lambda p: p.effective_input_price
        elif f in ("output_price", "output"):
            key = lambda p: p.effective_output_price
        elif f in ("tokens", "volume", "total_tokens"):
            key = lambda p: p.total_tokens
        elif f in ("share", "token_share"):
            key = lambda p: p.token_share
        elif f in ("name", "provider"):
            key = lambda p: p.name.lower()
            return sorted(self.providers, key=key, reverse=not descending)
        else:
            key = lambda p: p.cache_hit_rate

        return sorted(self.providers, key=key, reverse=descending)

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
