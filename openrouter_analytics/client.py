from typing import Optional, List, Dict, Any
import requests

from .models import ProviderStats, ModelStats
from .resolver import resolve_model


class OpenRouterAnalyticsError(Exception):
    """Base exception for analytics errors."""
    pass


class OpenRouterAnalytics:
    """Client for querying OpenRouter stats, cache hit rates, and effective pricing."""

    BASE_FRONTEND_URL = "https://openrouter.ai/api/frontend/v1"
    BASE_API_URL = "https://openrouter.ai/api/v1"

    def __init__(self, management_key: Optional[str] = None, user_agent: str = "openrouter-analytics-python/0.1"):
        self.management_key = management_key
        self.headers = {"User-Agent": user_agent}

    def get_model_stats(self, model: str) -> ModelStats:
        """
        Fetch full 24h stats (cache hit rates, effective pricing, tokens served)
        for all providers serving the requested model.

        :param model: Model name, ID, or slug (e.g. 'z-ai/glm-5.3-flash', 'glm-5.3-flash').
        :return: ModelStats object containing list of ProviderStats.
        """
        model_id, canonical_slug, display_name = resolve_model(model)

        url = f"{self.BASE_FRONTEND_URL}/stats/effective-pricing"
        params = {"permaslug": canonical_slug, "shape": "v7"}

        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=15)
            resp.raise_for_status()
            res_data = resp.json().get("data", {})
        except Exception as e:
            raise OpenRouterAnalyticsError(f"Failed to fetch stats for model '{model}' ({canonical_slug}): {e}")

        raw_summaries = res_data.get("providerSummaries", [])
        total_tokens_all = sum(s.get("totalTokens", 0) for s in raw_summaries)

        providers: List[ProviderStats] = []
        for s in raw_summaries:
            p_tokens = s.get("totalTokens", 0)
            token_share = (p_tokens / total_tokens_all) if total_tokens_all > 0 else 0.0
            providers.append(
                ProviderStats(
                    endpoint_id=s.get("endpointId", ""),
                    name=s.get("providerName", "Unknown"),
                    slug=s.get("providerSlug", ""),
                    effective_input_price=s.get("effectiveInputPrice", 0.0),
                    effective_output_price=s.get("effectiveOutputPrice", 0.0),
                    cache_hit_rate=s.get("cacheHitRate", 0.0),
                    total_tokens=p_tokens,
                    token_share=token_share,
                )
            )

        return ModelStats(
            model_id=model_id,
            permaslug=canonical_slug,
            model_name=display_name,
            providers=providers,
            weighted_cache_hit_rate=res_data.get("weightedCacheHitRate", 0.0),
            weighted_input_price=res_data.get("weightedInputPrice", 0.0),
            weighted_output_price=res_data.get("weightedOutputPrice", 0.0),
            total_tokens=total_tokens_all,
            input_chart_data=res_data.get("inputChartData", []),
            output_chart_data=res_data.get("outputChartData", []),
        )

    def get_provider_stats(self, model: str, provider: str) -> Optional[ProviderStats]:
        """
        Fetch stats for a single provider on a given model.
        Returns None if provider is not found.
        """
        model_stats = self.get_model_stats(model)
        return model_stats.get_provider(provider)

    def query_account_analytics(
        self,
        metrics: List[str],
        start_time: str,
        end_time: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        dimensions: Optional[List[str]] = None,
        granularity: str = "day",
    ) -> Dict[str, Any]:
        """
        Query account-specific analytics using an OpenRouter Management Key.
        Requires management_key to be set.
        """
        if not self.management_key:
            raise OpenRouterAnalyticsError(
                "A Management API Key is required to query private account analytics."
            )

        url = f"{self.BASE_API_URL}/analytics/query"
        headers = {
            **self.headers,
            "Authorization": f"Bearer {self.management_key}",
            "Content-Type": "application/json",
        }

        filters = []
        if model:
            model_id, _, _ = resolve_model(model)
            filters.append({"field": "model", "operator": "eq", "value": model_id})
        if provider:
            filters.append({"field": "provider", "operator": "eq", "value": provider})

        payload: Dict[str, Any] = {
            "metrics": metrics,
            "time_range": {"start": start_time, "end": end_time},
            "granularity": granularity,
        }
        if dimensions:
            payload["dimensions"] = dimensions
        if filters:
            payload["filters"] = filters

        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()


# Standalone shortcut functions
_default_client = OpenRouterAnalytics()


def get_model_stats(model: str) -> ModelStats:
    """Convenience function: Get all provider stats and cache hit rates for a model."""
    return _default_client.get_model_stats(model)


def get_cache_hit_rate(model: str, provider: str) -> Optional[float]:
    """Convenience function: Get just the cache hit rate (0.0 to 1.0) for a model and provider."""
    p = _default_client.get_provider_stats(model, provider)
    return p.cache_hit_rate if p else None
