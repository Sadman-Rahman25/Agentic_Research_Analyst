"""src/cache.py

Disk-backed cache for search results.

Why: every dev iteration re-runs queries against Tavily (1000-credit/month free
tier), arXiv (polite 3 req/sec), and GitHub (5000 req/hr with PAT). Without
caching, you burn budgets just from running the eval twice. With caching, the
second eval is nearly free.

Cache keys are deterministic hashes of (source, query, kwargs). Values are
whatever the researcher returns — usually a list of raw API result dicts,
NOT typed SearchResult objects (we reconstruct those fresh on each call so
sub_question_id tagging stays current).

Default TTL: 24 hours. Override per-call if needed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from diskcache import Cache

from src.config import CACHE_DIR

_cache = Cache(str(CACHE_DIR / "search"))


def _key(source: str, query: str, **kwargs: Any) -> str:
    """Deterministic 16-char cache key from source + query + sorted kwargs."""
    payload = json.dumps(
        {"source": source, "query": query, **kwargs},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get(source: str, query: str, **kwargs: Any) -> Optional[Any]:
    """Return cached value or None."""
    return _cache.get(_key(source, query, **kwargs))


def set_(source: str, query: str, value: Any, ttl: int = 86400, **kwargs: Any) -> None:
    """Cache a value with a TTL (default 24h)."""
    _cache.set(_key(source, query, **kwargs), value, expire=ttl)


def clear_all() -> int:
    """Wipe the entire cache. Returns count cleared. Use during prompt iteration."""
    return _cache.clear()


def stats() -> dict[str, Any]:
    """Return cache statistics for debugging."""
    return {
        "size": len(_cache),
        "directory": str(_cache.directory),
        "volume_bytes": _cache.volume(),
    }