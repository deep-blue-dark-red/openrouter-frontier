"""Shared helpers: disk cache, network tuning, price normalisation, and quantization filtering."""

import json
import re
import socket
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, TypeVar

CACHE_DIR = Path.home() / ".cache" / "openrouter_analytics"
DEFAULT_CACHE_TTL = 300  # seconds


def force_ipv4() -> None:
    """Make urllib3 resolve hostnames to IPv4 only.

    On macOS, connecting to openrouter.ai over IPv6 frequently stalls for ~10s before
    falling back to IPv4. Restricting the address family avoids that stall entirely.
    """
    try:
        import urllib3.util.connection as _conn

        _conn.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[assignment]
    except Exception:
        pass


def safe_filename(slug: str) -> str:
    """Turn a model slug like 'z-ai/glm-5.3-flash' into a filesystem-safe fragment."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", slug)


def load_json_cache(path: Path, ttl: int = DEFAULT_CACHE_TTL) -> Optional[Any]:
    """Return the JSON payload at ``path`` if it exists and is younger than ``ttl`` seconds."""
    try:
        if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def save_json_cache(path: Path, data: Any) -> None:
    """Write ``data`` as JSON to ``path``; cache failures are never fatal."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def price_per_million(val: Any) -> Optional[float]:
    """Normalise an OpenRouter price to USD per million tokens.

    The public APIs quote prices per token (e.g. ``"0.000000075"``), while the frontend
    effective-pricing endpoint already quotes per million. Anything below $0.01 is
    treated as per-token and scaled up; no real per-million price is that small.
    Returns ``None`` for missing or unparsable values.
    """
    if val is None or val == "":
        return None
    try:
        f = float(val)
    except (ValueError, TypeError):
        return None
    return f * 1_000_000.0 if f < 0.01 else f


T = TypeVar("T")

PRIMARY_QUANTIZATION = "fp8"


def filter_primary_quantization(items: Iterable[T], all_quants: bool = False) -> List[T]:
    """Keep only the model's primary quantization variant unless ``all_quants`` is set.

    OpenRouter's web pricing page shows the official (usually ``fp8``) variant. Items are
    expected to expose a ``quantization`` attribute. If no item is ``fp8`` the list is
    returned unchanged, since the model has no primary variant to filter to.
    """
    items = list(items)
    if all_quants:
        return items
    quants = [(getattr(i, "quantization", None) or "unknown").lower() for i in items]
    if PRIMARY_QUANTIZATION not in quants:
        return items
    return [i for i, q in zip(items, quants) if q == PRIMARY_QUANTIZATION]
