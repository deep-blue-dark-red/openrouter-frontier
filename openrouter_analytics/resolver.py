import os
import json
import time
import difflib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import requests

CACHE_DIR = Path.home() / ".cache" / "openrouter_analytics"
CACHE_FILE = CACHE_DIR / "models.json"
CACHE_TTL = 3600  # 1 hour


class ModelResolutionError(Exception):
    """Raised when a model cannot be resolved to an OpenRouter permaslug."""
    pass


def _fetch_models_from_api() -> List[Dict[str, Any]]:
    url = "https://openrouter.ai/api/v1/models"
    headers = {"User-Agent": "openrouter-analytics-python/0.1"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json().get("data", [])


def get_all_models(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Retrieve all models from OpenRouter with disk-based caching."""
    if not force_refresh and CACHE_FILE.exists():
        try:
            mtime = CACHE_FILE.stat().st_mtime
            if (time.time() - mtime) < CACHE_TTL:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass

    try:
        models = _fetch_models_from_api()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(models, f)
        return models
    except Exception as e:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ModelResolutionError(f"Failed to fetch model list from OpenRouter: {e}")


def _normalize_query(q: str) -> str:
    """Normalize input strings, correcting common domain/separator substitutions."""
    q = q.strip().lower()
    # Replace domain dot with hyphen (e.g. z.ai -> z-ai)
    q = q.replace("z.ai", "z-ai")
    q = q.replace("meta.llama", "meta-llama")
    # Common typo: flsh -> flash
    if "flsh" in q:
        q = q.replace("flsh", "flash")
    return q


def resolve_model(query: str) -> Tuple[str, str, str]:
    """
    Resolve any query string into:
    (model_id, canonical_permaslug, display_name)

    Accepts:
    - Full slugs (e.g. 'z-ai/glm-5.3-flash', 'anthropic/claude-3.7-sonnet')
    - Canonical slugs (e.g. 'z-ai/glm-5.3-flash-20260826')
    - Short names (e.g. 'glm-5.3-flash', 'claude-3.7-sonnet', 'gpt-4o')
    - Fuzzy/typo matches (e.g. 'z.ai/glm-5.3-flsh')
    """
    norm = _normalize_query(query)
    models = get_all_models()

    # 1. Exact matches on id or canonical_slug
    for m in models:
        m_id = m.get("id", "").lower()
        c_slug = (m.get("canonical_slug") or m_id).lower()
        if norm in (m_id, c_slug):
            return m["id"], m.get("canonical_slug") or m["id"], m.get("name", m["id"])

    # 2. Match without author prefix (e.g. 'glm-5.3-flash' matches 'z-ai/glm-5.3-flash')
    for m in models:
        m_id = m.get("id", "").lower()
        c_slug = (m.get("canonical_slug") or m_id).lower()
        short_id = m_id.split("/")[-1] if "/" in m_id else m_id
        if norm == short_id:
            return m["id"], m.get("canonical_slug") or m["id"], m.get("name", m["id"])

    # 3. Substring match on id
    candidates = []
    for m in models:
        m_id = m.get("id", "").lower()
        c_slug = (m.get("canonical_slug") or m_id).lower()
        name = m.get("name", "").lower()
        if norm in m_id or norm in c_slug or norm in name:
            candidates.append(m)

    if len(candidates) == 1:
        m = candidates[0]
        return m["id"], m.get("canonical_slug") or m["id"], m.get("name", m["id"])

    if len(candidates) > 1:
        # Prefer exact short suffix match or shortest ID
        candidates.sort(key=lambda x: (
            not x.get("id", "").lower().endswith(norm),
            len(x.get("id", ""))
        ))
        m = candidates[0]
        return m["id"], m.get("canonical_slug") or m["id"], m.get("name", m["id"])

    # 4. Fuzzy match using difflib
    all_slugs = {m.get("id", ""): m for m in models}
    close_matches = difflib.get_close_matches(norm, all_slugs.keys(), n=3, cutoff=0.5)
    if close_matches:
        m = all_slugs[close_matches[0]]
        return m["id"], m.get("canonical_slug") or m["id"], m.get("name", m["id"])

    # Also try matching against short slugs
    short_map = {(m.get("id", "").split("/")[-1]): m for m in models if "/" in m.get("id", "")}
    close_short = difflib.get_close_matches(norm, short_map.keys(), n=3, cutoff=0.5)
    if close_short:
        m = short_map[close_short[0]]
        return m["id"], m.get("canonical_slug") or m["id"], m.get("name", m["id"])

    raise ModelResolutionError(
        f"Could not resolve model '{query}'. Try searching for available models with 'openrouter-analytics search {query}'."
    )


def search_models(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """Search for models matching query string."""
    norm = _normalize_query(query)
    models = get_all_models()
    results = []

    for m in models:
        m_id = m.get("id", "")
        name = m.get("name", "")
        c_slug = m.get("canonical_slug", m_id)
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
