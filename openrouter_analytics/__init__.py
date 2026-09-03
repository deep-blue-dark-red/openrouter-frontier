"""
OpenRouter Analytics: A Python tool suite to inspect 24h provider stats, cache hit rates, and effective pricing.
"""

from .models import ProviderStats, ModelStats
from .resolver import resolve_model, search_models, ModelResolutionError
from .client import (
    OpenRouterAnalytics,
    OpenRouterAnalyticsError,
    get_model_stats,
    get_cache_hit_rate,
)

__version__ = "0.1.0"
__all__ = [
    "ProviderStats",
    "ModelStats",
    "resolve_model",
    "search_models",
    "ModelResolutionError",
    "OpenRouterAnalytics",
    "OpenRouterAnalyticsError",
    "get_model_stats",
    "get_cache_hit_rate",
]
