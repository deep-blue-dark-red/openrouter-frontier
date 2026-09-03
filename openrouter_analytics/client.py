import re
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import concurrent.futures
import requests

from .models import ProviderStats, ModelStats
from .resolver import resolve_model
from .scoring import EndpointPricing, ScoringConfig, ScoreBreakdown, evaluate_endpoint

CACHE_DIR = Path.home() / ".cache" / "openrouter_analytics"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = 300  # 5 minutes


class OpenRouterAnalyticsError(Exception):
    """Base exception for analytics errors."""
    pass


class OpenRouterAnalytics:
    """Client for querying OpenRouter stats, cache hit rates, latency, TPS, uptime, and scoring."""

    BASE_FRONTEND_URL = "https://openrouter.ai/api/frontend/v1"
    BASE_API_URL = "https://openrouter.ai/api/v1"
    BASE_SITE_URL = "https://openrouter.ai"

    def __init__(self, management_key: Optional[str] = None, user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"):
        self.management_key = management_key
        self.headers = {"User-Agent": user_agent}

    def _fetch_endpoint_performance(self, model_id: str, canonical_slug: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """
        Extract latency, throughput (TPS), and uptime for endpoints.
        Attempts to read from cache, then model page RSC flight data, falling back to public endpoints API.
        """
        safe_slug = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", canonical_slug)
        perf_cache_file = CACHE_DIR / f"perf_{safe_slug}.json"

        if perf_cache_file.exists():
            try:
                mtime = perf_cache_file.stat().st_mtime
                if time.time() - mtime < CACHE_TTL:
                    with open(perf_cache_file, "r") as f:
                        cached = json.load(f)
                        return cached.get("stats", {}), cached.get("uptime", {})
            except Exception:
                pass

        stats_by_endpoint: Dict[str, Dict[str, Any]] = {}
        uptime_by_endpoint: Dict[str, float] = {}

        candidates = [canonical_slug] if model_id == canonical_slug else [canonical_slug, model_id]
        for slug_candidate in candidates:
            url = f"{self.BASE_SITE_URL}/models/{slug_candidate}"
            try:
                resp = requests.get(url, headers=self.headers, timeout=6)
                if resp.status_code == 200:
                    html = resp.text
                    rsc_raw = ""
                    for m in re.finditer(r'self\.__next_f\.push\(\[\d+,\"(.*)\"\]\)', html):
                        chunk = m.group(1)
                        try:
                            rsc_raw += json.loads('"' + chunk + '"')
                        except Exception:
                            pass

                    # Extract stats (latency, throughput)
                    for m in re.finditer(r'\"stats\":(\{[^}]+\})', rsc_raw):
                        try:
                            s = json.loads(m.group(1))
                            ep_id = s.get("endpoint_id")
                            if ep_id and ep_id not in stats_by_endpoint:
                                stats_by_endpoint[ep_id] = s
                        except Exception:
                            pass

                    # Extract uptime data
                    for m in re.finditer(r'\"([a-f0-9\-]{36})\":\[(\{\"date\":[^\]]+\})\]', rsc_raw):
                        ep_id = m.group(1)
                        try:
                            arr = json.loads("[" + m.group(2) + "]")
                            valid = [x["uptime"] for x in arr if x.get("uptime") is not None]
                            if valid and ep_id not in uptime_by_endpoint:
                                uptime_by_endpoint[ep_id] = valid[0]
                        except Exception:
                            pass

                    if stats_by_endpoint:
                        break
            except Exception:
                pass

        if not uptime_by_endpoint:
            ep_url = f"{self.BASE_API_URL}/models/{canonical_slug}/endpoints"
            try:
                resp = requests.get(ep_url, headers=self.headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json().get("data", {})
                    for ep in data.get("endpoints", []):
                        ep_tag = ep.get("tag") or ep.get("name")
                        upt = ep.get("uptime_last_1d")
                        if upt is not None and ep_tag:
                            uptime_by_endpoint[ep_tag] = upt
            except Exception:
                pass

        # Save to disk cache
        try:
            with open(perf_cache_file, "w") as f:
                json.dump({"stats": stats_by_endpoint, "uptime": uptime_by_endpoint}, f)
        except Exception:
            pass

        return stats_by_endpoint, uptime_by_endpoint

    def _fetch_endpoints_pricing(self, canonical_slug: str, apply_discount: bool = True) -> Dict[str, EndpointPricing]:
        """Fetch list pricing per endpoint from the public API (with caching)."""
        safe_slug = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", canonical_slug)
        pricing_cache_file = CACHE_DIR / f"pricing_{safe_slug}.json"

        raw_endpoints = []
        if pricing_cache_file.exists():
            try:
                mtime = pricing_cache_file.stat().st_mtime
                if time.time() - mtime < CACHE_TTL:
                    with open(pricing_cache_file, "r") as f:
                        raw_endpoints = json.load(f)
            except Exception:
                pass

        if not raw_endpoints:
            url = f"{self.BASE_API_URL}/models/{canonical_slug}/endpoints"
            try:
                resp = requests.get(url, headers=self.headers, timeout=6)
                if resp.status_code == 200:
                    raw_endpoints = resp.json().get("data", {}).get("endpoints", [])
                    try:
                        with open(pricing_cache_file, "w") as f:
                            json.dump(raw_endpoints, f)
                    except Exception:
                        pass
            except Exception:
                pass

        pricing_map: Dict[str, EndpointPricing] = {}
        quant_map: Dict[str, str] = {}
        for ep in raw_endpoints:
            raw_p = ep.get("pricing", {})
            parsed = EndpointPricing.from_api_dict(raw_p, apply_discount=apply_discount)
            tag = (ep.get("tag") or "").lower()
            p_name = (ep.get("provider_name") or "").lower()
            q = ep.get("quantization") or "unknown"
            if tag:
                pricing_map[tag] = parsed
                quant_map[tag] = q
                if "/" in tag:
                    pricing_map[tag.split("/")[0]] = parsed
                    quant_map[tag.split("/")[0]] = q
            if p_name:
                pricing_map[p_name] = parsed
                quant_map[p_name] = q

        self._last_quant_map = quant_map
        return pricing_map

    def get_model_stats(self, model: str, apply_discount: bool = True) -> ModelStats:
        """
        Fetch full 24h stats (cache hit rates, effective pricing, endpoint pricing, latency, TPS, uptime, tokens)
        for all providers serving the requested model.

        :param model: Model name, ID, or slug (e.g. 'z-ai/glm-5.3-flash', 'glm-5.3-flash').
        :param apply_discount: Whether to factor in endpoint discount rates into pricing.
        :return: ModelStats object containing list of ProviderStats.
        """
        model_id, canonical_slug, display_name = resolve_model(model)

        def _get_pricing():
            pricing_url = f"{self.BASE_FRONTEND_URL}/stats/effective-pricing"
            params = {"permaslug": canonical_slug, "shape": "v7"}
            resp = requests.get(pricing_url, params=params, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json().get("data", {})

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            fut_pricing = executor.submit(_get_pricing)
            fut_perf = executor.submit(self._fetch_endpoint_performance, model_id, canonical_slug)
            fut_ep_pricing = executor.submit(self._fetch_endpoints_pricing, canonical_slug, apply_discount)

            try:
                res_data = fut_pricing.result()
            except Exception as e:
                raise OpenRouterAnalyticsError(f"Failed to fetch stats for model '{model}' ({canonical_slug}): {e}")

            stats_map, uptime_map = fut_perf.result()
            pricing_map = fut_ep_pricing.result()

        raw_summaries = res_data.get("providerSummaries", [])
        total_tokens_all = sum(s.get("totalTokens", 0) for s in raw_summaries)

        providers: List[ProviderStats] = []
        valid_latencies = []
        valid_tps = []
        valid_uptimes = []

        for s in raw_summaries:
            ep_id = s.get("endpointId", "")
            p_name = s.get("providerName", "Unknown")
            p_slug = s.get("providerSlug", "")
            p_tokens = s.get("totalTokens", 0)
            token_share = (p_tokens / total_tokens_all) if total_tokens_all > 0 else 0.0

            # Merge latency and throughput
            perf = stats_map.get(ep_id, {})
            lat_p50 = perf.get("p50_latency")
            lat_p90 = perf.get("p90_latency")
            tps_p50 = perf.get("p50_throughput")
            tps_p90 = perf.get("p90_throughput")

            # Merge uptime
            upt = uptime_map.get(ep_id)
            if upt is None:
                upt = uptime_map.get(p_slug)

            if lat_p50 is not None:
                valid_latencies.append(lat_p50)
            if tps_p50 is not None:
                valid_tps.append(tps_p50)
            if upt is not None:
                valid_uptimes.append(upt)

            # Match EndpointPricing
            p_pricing = pricing_map.get(p_slug.lower())
            if not p_pricing:
                p_pricing = pricing_map.get(p_name.lower())
            if not p_pricing:
                for k, v in pricing_map.items():
                    if p_slug.lower() in k or k in p_slug.lower() or p_name.lower() in k or k in p_name.lower():
                        p_pricing = v
                        break
            if not p_pricing:
                p_pricing = EndpointPricing(
                    prompt=s.get("effectiveInputPrice", 0.0),
                    completion=s.get("effectiveOutputPrice", 0.0),
                    input_cache_read=None,
                    input_cache_write=None,
                )

            q_val = getattr(self, "_last_quant_map", {}).get(p_slug.lower()) or getattr(self, "_last_quant_map", {}).get(p_name.lower()) or "unknown"
            providers.append(
                ProviderStats(
                    endpoint_id=ep_id,
                    name=p_name,
                    slug=p_slug,
                    effective_input_price=s.get("effectiveInputPrice", 0.0),
                    effective_output_price=s.get("effectiveOutputPrice", 0.0),
                    cache_hit_rate=s.get("cacheHitRate", 0.0),
                    total_tokens=p_tokens,
                    token_share=token_share,
                    latency_p50_ms=lat_p50,
                    latency_p90_ms=lat_p90,
                    throughput_p50_tps=tps_p50,
                    throughput_p90_tps=tps_p90,
                    uptime_1d_pct=upt,
                    pricing=p_pricing,
                    quantization=q_val,
                )
            )

        avg_lat = sum(valid_latencies) / len(valid_latencies) if valid_latencies else None
        avg_tps = sum(valid_tps) / len(valid_tps) if valid_tps else None
        avg_upt = sum(valid_uptimes) / len(valid_uptimes) if valid_uptimes else None

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
            avg_latency_p50_ms=avg_lat,
            avg_throughput_p50_tps=avg_tps,
            avg_uptime_1d_pct=avg_upt,
        )

    def get_provider_stats(self, model: str, provider: str) -> Optional[ProviderStats]:
        """Fetch stats for a single provider on a given model."""
        model_stats = self.get_model_stats(model)
        return model_stats.get_provider(provider)

    def score_model_providers(self, model: str, config: Optional[ScoringConfig] = None) -> List[ScoreBreakdown]:
        """Evaluate and rank all providers serving a model using the utility scoring model."""
        cfg = config or ScoringConfig()
        model_stats = self.get_model_stats(model, apply_discount=cfg.apply_discount)
        return model_stats.score_providers(cfg)

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
        """Query account-specific analytics using an OpenRouter Management Key."""
        if not self.management_key:
            raise OpenRouterAnalyticsError("A Management API Key is required to query private account analytics.")

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


_default_client = OpenRouterAnalytics()


def get_model_stats(model: str, apply_discount: bool = True) -> ModelStats:
    """Get all provider stats, cache hit rates, latency, TPS, and uptime for a model."""
    return _default_client.get_model_stats(model, apply_discount=apply_discount)


def get_cache_hit_rate(model: str, provider: str) -> Optional[float]:
    """Get cache hit rate (0.0 to 1.0) for a model and provider."""
    p = _default_client.get_provider_stats(model, provider)
    return p.cache_hit_rate if p else None


def score_model_providers(model: str, config: Optional[ScoringConfig] = None) -> List[ScoreBreakdown]:
    """Evaluate and rank all providers serving a model using the utility scoring model."""
    return _default_client.score_model_providers(model, config=config)
