"""Model catalog retrieval and fuzzy model-name resolution."""

import difflib
import re
from typing import Any, Dict, List, Tuple

import requests

from ._util import CACHE_DIR, force_ipv4, load_json_cache, save_json_cache

force_ipv4()

MODELS_URL = "https://openrouter.ai/api/v1/models"
MODELS_CACHE_FILE = CACHE_DIR / "models.json"
MODELS_CACHE_TTL = 3600  # the catalog changes rarely; 1 hour


class ModelResolutionError(Exception):
    """Raised when a query cannot be matched to any OpenRouter model."""


def _fetch_models_from_api() -> List[Dict[str, Any]]:
    resp = requests.get(MODELS_URL, headers={"User-Agent": "openrouter-analytics-python"}, timeout=15)
    resp.raise_for_status()
    return resp.json().get("data", [])


def get_all_models(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Return the full model catalog, served from a 1-hour disk cache when possible.

    If the network request fails, a stale cache is returned rather than raising.
    """
    if not force_refresh:
        cached = load_json_cache(MODELS_CACHE_FILE, MODELS_CACHE_TTL)
        if cached is not None:
            return cached
    try:
        models = _fetch_models_from_api()
    except Exception as e:
        stale = load_json_cache(MODELS_CACHE_FILE, ttl=10**12)
        if stale is not None:
            return stale
        raise ModelResolutionError(f"Failed to fetch model list from OpenRouter: {e}")
    save_json_cache(MODELS_CACHE_FILE, models)
    return models


_CREATOR_FIXES = [
    (re.compile(r"^z[\.\-_]?ai(?=/|$)"), "z-ai"),
    (re.compile(r"^meta[\.\-_]llama(?=/|$)"), "meta-llama"),
]


def _normalize_query(q: str) -> str:
    """Lowercase and repair common creator-prefix and typo variants (z.ai, zai, flsh)."""
    q = q.strip().lower()
    for pattern, repl in _CREATOR_FIXES:
        q = pattern.sub(repl, q)
    return q.replace("flsh", "flash")


def _describe(m: Dict[str, Any]) -> Tuple[str, str, str]:
    return m["id"], m.get("canonical_slug") or m["id"], m.get("name") or m["id"]


def resolve_model(query: str) -> Tuple[str, str, str]:
    """Resolve a user-supplied model string to ``(model_id, canonical_permaslug, display_name)``.

    Matching order: exact id or permaslug; exact short name without creator prefix;
    substring (preferring ids that end with the query, then the shortest id); finally a
    difflib fuzzy match against full and short ids.
    """
    norm = _normalize_query(query)
    models = get_all_models()

    def mid(m: Dict[str, Any]) -> str:
        return m.get("id", "").lower()

    def slug(m: Dict[str, Any]) -> str:
        return (m.get("canonical_slug") or m.get("id", "")).lower()

    def short(m: Dict[str, Any]) -> str:
        return mid(m).rsplit("/", 1)[-1]

    for m in models:
        if norm in (mid(m), slug(m)):
            return _describe(m)

    for m in models:
        if norm == short(m):
            return _describe(m)

    candidates = [m for m in models if norm in mid(m) or norm in slug(m) or norm in (m.get("name") or "").lower()]
    if candidates:
        candidates.sort(key=lambda m: (not mid(m).endswith(norm), len(mid(m))))
        return _describe(candidates[0])

    by_id = {mid(m): m for m in models}
    close = difflib.get_close_matches(norm, by_id.keys(), n=1, cutoff=0.5)
    if close:
        return _describe(by_id[close[0]])

    by_short = {short(m): m for m in models if "/" in mid(m)}
    close = difflib.get_close_matches(norm, by_short.keys(), n=1, cutoff=0.5)
    if close:
        return _describe(by_short[close[0]])

    raise ModelResolutionError(
        f"Could not resolve model '{query}'. Try 'openrouter-analytics search {query}' to list candidates."
    )


def search_models(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """Substring search over id, name, and permaslug."""
    norm = _normalize_query(query)
    results = []
    for m in get_all_models():
        m_id = m.get("id", "")
        name = m.get("name", "")
        c_slug = m.get("canonical_slug") or m_id
        if norm in m_id.lower() or norm in name.lower() or norm in c_slug.lower():
            results.append({
                "id": m_id,
                "canonical_slug": c_slug,
                "name": name,
                "context_length": str(m.get("context_length", "")),
            })
            if len(results) >= limit:
                break
    return results
