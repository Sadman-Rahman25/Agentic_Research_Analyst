"""src/researchers/web.py

Tavily web search wrapper.

Pure tool call — NO LLM. Takes a query, returns ranked SearchResult objects
with title/url/snippet. Caches the raw API response so re-runs are free.

Tavily free tier: 1000 credits/month. Basic search = 1 credit. Stay on basic
during dev; advanced is 2 credits and the quality bump isn't worth it for
synthesis-grade snippets.
"""

from __future__ import annotations

from tavily import AsyncTavilyClient

from src import cache
from src.config import TAVILY_API_KEY
from src.schemas import SearchResult


# Lazy-initialized singleton so import-time doesn't fail when key is missing
_client: AsyncTavilyClient | None = None


def _get_client() -> AsyncTavilyClient:
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY not set. Check .env.")
        _client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
    return _client


async def web_search(
    sub_question_id: str,
    query: str,
    top_k: int = 5,
) -> list[SearchResult]:
    """Search the web via Tavily. Returns up to `top_k` ranked results.

    Cached. Empty list on failure (logged, never raises).
    """
    # Cache check — raw API results, not typed SearchResults
    raw = cache.get("web", query, top_k=top_k)
    if raw is None:
        try:
            client = _get_client()
            response = await client.search(
                query=query,
                search_depth="basic",
                max_results=top_k,
            )
            raw = response.get("results", [])
            cache.set_("web", query, raw, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            print(f"  [web]  WARN: Tavily call failed for {query!r}: {e}")
            return []

    return [
        SearchResult(
            sub_question_id=sub_question_id,
            source_type="web",
            title=r.get("title", "(no title)"),
            url=r.get("url", ""),
            snippet=(r.get("content") or "")[:500],
            metadata={"score": r.get("score")},
            rank=i + 1,
        )
        for i, r in enumerate(raw[:top_k])
    ]