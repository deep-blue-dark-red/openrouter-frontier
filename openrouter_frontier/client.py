"""HTTP client for OpenRouter's public and frontend stats endpoints."""

import concurrent.futures
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from ._util import CACHE_DIR, DEFAULT_CACHE_TTL, force_ipv4, load_json_cache, safe_filename, save_json_cache
from .models import ModelStats, ProviderStats
from .resolver import resolve_model
from .scoring import EndpointPricing, ScoreBreakdown, ScoringConfig

force_ipv4()

# Regexes for pulling data out of the Next.js RSC flight payload embedded in a model page.
_RSC_CHUNK_RE = re.compile(r'self\.__next_f\.push\(\[\d+,"(.*)"\]\)')
_RSC_STATS_RE = re.compile(r'"stats":(\{[^}]+\})')
_RSC_UPTIME_RE = re.compile(r'"([a-f0-9\-]{36})":\[(\{"date":[^\]]+\})\]')


class OpenRouterFrontierError(Exception):
    """Raised when OpenRouter data cannot be fetched."""


class OpenRouterAnalytics:
    """Fetches 24h provider stats (cache hit rate, pricing, latency, TPS, uptime) for a model.

    Three sources are merged per endpoint and cached on disk for :data:`cache_ttl` seconds:

    * ``/api/frontend/v1/stats/effective-pricing`` - cache hit rate, effective prices, tokens
    * the public model page's RSC payload - p50/p90 latency and throughput, uptime
    * ``/api/v1/models/{slug}/endpoints`` - list pricing, quantization, uptime fallback
    """

    BASE_FRONTEND_URL = "https://openrouter.ai/api/frontend/v1"
    BASE_API_URL = "https://openrouter.ai/api/v1"
    BASE_SITE_URL = "https://openrouter.ai"

    def __init__(
        self,
        management_key: Optional[str] = None,
        user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        cache_ttl: int = DEFAULT_CACHE_TTL,
    ):
        self.management_key = management_key
        self.cache_ttl = cache_ttl
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    # ------------------------------------------------------------------ fetchers

    def _get_json(self, url: str, timeout: float = 6, **params: Any) -> Optional[Dict[str, Any]]:
        try:
            resp = self.session.get(url, params=params or None, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def _fetch_effective_pricing(self, canonical_slug: str) -> Dict[str, Any]:
        cache_file = CACHE_DIR / f"eff_{safe_filename(canonical_slug)}.json"
        cached = load_json_cache(cache_file, self.cache_ttl)
        if cached is not None:
            return cached
        resp = self.session.get(
            f"{self.BASE_FRONTEND_URL}/stats/effective-pricing",
            params={"permaslug": canonical_slug, "shape": "v7"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        save_json_cache(cache_file, data)
        return data

    def _fetch_endpoint_performance(
        self, model_id: str, canonical_slug: str
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """Return ``(stats_by_endpoint_id, uptime_by_endpoint_id_or_tag)``.

        Latency and throughput only appear in the model page's RSC payload. Uptime is read
        from the same payload, falling back to the public endpoints API keyed by tag.
        """
        cache_file = CACHE_DIR / f"perf_{safe_filename(canonical_slug)}.json"
        cached = load_json_cache(cache_file, self.cache_ttl)
        if cached is not None:
            return cached.get("stats", {}), cached.get("uptime", {})

        stats: Dict[str, Dict[str, Any]] = {}
        uptime: Dict[str, float] = {}

        slugs = [canonical_slug] if model_id == canonical_slug else [canonical_slug, model_id]
        for slug in slugs:
            try:
                resp = self.session.get(f"{self.BASE_SITE_URL}/models/{slug}", timeout=6)
            except Exception:
                continue
            if resp.status_code != 200:
                continue
            rsc = "".join(_decode_rsc_chunk(m.group(1)) for m in _RSC_CHUNK_RE.finditer(resp.text))
            for m in _RSC_STATS_RE.finditer(rsc):
                try:
                    s = json.loads(m.group(1))
                except Exception:
                    continue
                ep_id = s.get("endpoint_id")
                if ep_id:
                    stats.setdefault(ep_id, s)
            for m in _RSC_UPTIME_RE.finditer(rsc):
                try:
                    series = json.loads("[" + m.group(2) + "]")
                except Exception:
                    continue
                values = [x["uptime"] for x in series if x.get("uptime") is not None]
                if values:
                    uptime.setdefault(m.group(1), values[0])
            if stats:
                break

        if not uptime:
            data = self._get_json(f"{self.BASE_API_URL}/models/{canonical_slug}/endpoints", timeout=5)
            for ep in (data or {}).get("data", {}).get("endpoints", []):
                tag = ep.get("tag") or ep.get("name")
                if tag and ep.get("uptime_last_1d") is not None:
                    uptime[tag] = ep["uptime_last_1d"]

        save_json_cache(cache_file, {"stats": stats, "uptime": uptime})
        return stats, uptime

    def _fetch_endpoints_pricing(
        self, canonical_slug: str, apply_discount: bool
    ) -> Tuple[Dict[str, EndpointPricing], Dict[str, str]]:
        """Return ``(pricing_by_key, quantization_by_key)`` keyed by tag, tag prefix, and provider name."""
        cache_file = CACHE_DIR / f"pricing_{safe_filename(canonical_slug)}.json"
        raw = load_json_cache(cache_file, self.cache_ttl)
        if raw is None:
            data = self._get_json(f"{self.BASE_API_URL}/models/{canonical_slug}/endpoints")
            raw = (data or {}).get("data", {}).get("endpoints", [])
            if raw:
                save_json_cache(cache_file, raw)

        pricing: Dict[str, EndpointPricing] = {}
        quant: Dict[str, str] = {}
        for ep in raw:
            parsed = EndpointPricing.from_api_dict(ep.get("pricing", {}), apply_discount=apply_discount)
            q = ep.get("quantization") or "unknown"
            keys = []
            tag = (ep.get("tag") or "").lower()
            if tag:
                keys.append(tag)
                if "/" in tag:
                    keys.append(tag.split("/", 1)[0])
            if ep.get("provider_name"):
                keys.append(ep["provider_name"].lower())
            for k in keys:
                pricing[k] = parsed
                quant[k] = q
        return pricing, quant

    # ------------------------------------------------------------------ public API

    def get_model_stats(self, model: str, apply_discount: bool = True) -> ModelStats:
        """Fetch and merge all 24h stats for every provider serving ``model``.

        :param model: Model name, id, or permaslug; resolved fuzzily.
        :param apply_discount: Apply advertised endpoint discounts to list pricing.
        """
        model_id, canonical_slug, display_name = resolve_model(model)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            fut_eff = pool.submit(self._fetch_effective_pricing, canonical_slug)
            fut_perf = pool.submit(self._fetch_endpoint_performance, model_id, canonical_slug)
            fut_price = pool.submit(self._fetch_endpoints_pricing, canonical_slug, apply_discount)
            try:
                eff = fut_eff.result()
            except Exception as e:
                raise OpenRouterFrontierError(f"Failed to fetch stats for '{model}' ({canonical_slug}): {e}")
            stats_map, uptime_map = fut_perf.result()
            pricing_map, quant_map = fut_price.result()

        summaries = eff.get("providerSummaries", [])
        total_tokens = sum(s.get("totalTokens", 0) for s in summaries)

        providers: List[ProviderStats] = []
        for s in summaries:
            ep_id = s.get("endpointId", "")
            name = s.get("providerName", "Unknown")
            slug = s.get("providerSlug", "")
            tokens = s.get("totalTokens", 0)
            perf = stats_map.get(ep_id, {})

            key = _match_key(pricing_map, slug, name)
            pricing = pricing_map.get(key) if key else None
            if pricing is None:
                pricing = EndpointPricing(
                    prompt=s.get("effectiveInputPrice", 0.0),
                    completion=s.get("effectiveOutputPrice", 0.0),
                )

            providers.append(
                ProviderStats(
                    endpoint_id=ep_id,
                    name=name,
                    slug=slug,
                    effective_input_price=s.get("effectiveInputPrice", 0.0),
                    effective_output_price=s.get("effectiveOutputPrice", 0.0),
                    cache_hit_rate=s.get("cacheHitRate", 0.0),
                    total_tokens=tokens,
                    token_share=(tokens / total_tokens) if total_tokens > 0 else 0.0,
                    latency_p50_ms=perf.get("p50_latency"),
                    latency_p90_ms=perf.get("p90_latency"),
                    throughput_p50_tps=perf.get("p50_throughput"),
                    throughput_p90_tps=perf.get("p90_throughput"),
                    uptime_1d_pct=uptime_map.get(ep_id, uptime_map.get(slug)),
                    pricing=pricing,
                    quantization=quant_map.get(key, "unknown") if key else "unknown",
                )
            )

        return ModelStats(
            model_id=model_id,
            permaslug=canonical_slug,
            model_name=display_name,
            providers=providers,
            weighted_cache_hit_rate=eff.get("weightedCacheHitRate", 0.0),
            weighted_input_price=eff.get("weightedInputPrice", 0.0),
            weighted_output_price=eff.get("weightedOutputPrice", 0.0),
            total_tokens=total_tokens,
            input_chart_data=eff.get("inputChartData", []),
            output_chart_data=eff.get("outputChartData", []),
            avg_latency_p50_ms=_mean([p.latency_p50_ms for p in providers]),
            avg_throughput_p50_tps=_mean([p.throughput_p50_tps for p in providers]),
            avg_uptime_1d_pct=_mean([p.uptime_1d_pct for p in providers]),
        )

    def get_provider_stats(self, model: str, provider: str) -> Optional[ProviderStats]:
        """Stats for a single provider of ``model``, or ``None`` if it isn't serving it."""
        return self.get_model_stats(model).get_provider(provider)

    def score_model_providers(self, model: str, config: Optional[ScoringConfig] = None) -> List[ScoreBreakdown]:
        """Rank every provider of ``model`` by ProviderUtility total cost, cheapest first."""
        cfg = config or ScoringConfig()
        return self.get_model_stats(model, apply_discount=cfg.apply_discount).score_providers(cfg)

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
        """Query private account analytics. Requires a management API key."""
        if not self.management_key:
            raise OpenRouterFrontierError("A management API key is required for account analytics.")

        filters = []
        if model:
            filters.append({"field": "model", "operator": "eq", "value": resolve_model(model)[0]})
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

        resp = self.session.post(
            f"{self.BASE_API_URL}/analytics/query",
            headers={"Authorization": f"Bearer {self.management_key}"},
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------- helpers

def _decode_rsc_chunk(chunk: str) -> str:
    try:
        return json.loads('"' + chunk + '"')
    except Exception:
        return ""


def _match_key(pricing_map: Dict[str, Any], slug: str, name: str) -> Optional[str]:
    """Find the endpoints-API key for a provider: exact slug, exact name, then substring."""
    slug, name = slug.lower(), name.lower()
    for k in (slug, name):
        if k in pricing_map:
            return k
    for k in pricing_map:
        if slug and (slug in k or k in slug):
            return k
        if name and (name in k or k in name):
            return k
    return None


def _mean(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


# ---------------------------------------------------------------------- module-level API

_default_client = OpenRouterAnalytics()


def get_model_stats(model: str, apply_discount: bool = True) -> ModelStats:
    """Fetch 24h provider stats for a model using a shared default client."""
    return _default_client.get_model_stats(model, apply_discount=apply_discount)


def get_cache_hit_rate(model: str, provider: str) -> Optional[float]:
    """Published 24h cache hit rate (0..1) for one provider of a model, or ``None``."""
    p = _default_client.get_provider_stats(model, provider)
    return p.cache_hit_rate if p else None


def score_model_providers(model: str, config: Optional[ScoringConfig] = None) -> List[ScoreBreakdown]:
    """Rank providers of a model by ProviderUtility total cost using a shared default client."""
    return _default_client.score_model_providers(model, config=config)
