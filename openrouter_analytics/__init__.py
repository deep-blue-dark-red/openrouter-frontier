"""
OpenRouter Analytics: A Python tool suite to inspect 24h provider stats, cache hit rates,
effective pricing, latency, throughput (TPS), uptime, and cost/utility scoring.
"""

from .models import ProviderStats, ModelStats
from .resolver import resolve_model, search_models, ModelResolutionError
from .scoring import (
    ScoringConfig,
    EndpointPricing,
    ScoreBreakdown,
    evaluate_endpoint,
)
from .client import (
    OpenRouterAnalytics,
    OpenRouterAnalyticsError,
    get_model_stats,
    get_cache_hit_rate,
    score_model_providers,
)

__version__ = "0.2.0"
__all__ = [
    "ProviderStats",
    "ModelStats",
    "resolve_model",
    "search_models",
    "ModelResolutionError",
    "ScoringConfig",
    "EndpointPricing",
    "ScoreBreakdown",
    "evaluate_endpoint",
    "OpenRouterAnalytics",
    "OpenRouterAnalyticsError",
    "get_model_stats",
    "get_cache_hit_rate",
    "score_model_providers",
]
