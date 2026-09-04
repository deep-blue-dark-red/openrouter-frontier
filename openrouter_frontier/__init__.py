"""OpenRouter Frontier: 24h provider stats, cache hit rates, pricing, and cost/utility scoring."""

from .client import (
    OpenRouterAnalytics,
    OpenRouterFrontierError,
    get_cache_hit_rate,
    get_model_stats,
    score_model_providers,
)
from .models import ModelStats, ProviderStats
from .pareto import Objective, annotate_frontier, cost_quality_frontier, pareto_mask
from .resolver import ModelResolutionError, get_all_models, resolve_model, search_models
from .scoring import EndpointPricing, ScoreBreakdown, ScoringConfig, evaluate_endpoint

__version__ = "0.2.0"
__all__ = [
    "OpenRouterAnalytics",
    "OpenRouterFrontierError",
    "get_model_stats",
    "get_cache_hit_rate",
    "score_model_providers",
    "ModelStats",
    "ProviderStats",
    "Objective",
    "pareto_mask",
    "cost_quality_frontier",
    "annotate_frontier",
    "ModelResolutionError",
    "get_all_models",
    "resolve_model",
    "search_models",
    "EndpointPricing",
    "ScoreBreakdown",
    "ScoringConfig",
    "evaluate_endpoint",
]
