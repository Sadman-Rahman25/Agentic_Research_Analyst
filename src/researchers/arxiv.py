"""src/researchers/arxiv.py

arXiv search wrapper.

Pure tool call — NO LLM. The `arxiv` package is sync; we wrap with
`asyncio.to_thread()` so the orchestrator can run all researchers in parallel.

arXiv rate-limits at 3 requests/sec per IP. The arxiv.Client below sets
delay_seconds=3 to be a polite citizen.

For Day 2 we use the abstract as the snippet. Full PDF extraction is a Day 3
optimization (slow: 3-5s per paper) and we only do it for top-2 papers per
sub-question when synthesizer quality demands it.
"""

from __future__ import annotations

import asyncio

import arxiv

from src import cache
from src.schemas import SearchResult


# Module-level client — arxiv package recommends one client per process
_arxiv_client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=2)


def _search_sync(query: str, top_k: int) -> list[dict]:
    """Sync arXiv search. Returns plain dicts for clean pickling/caching."""
    search = arxiv.Search(
        query=query,
        max_results=top_k,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    out: list[dict] = []
    for paper in _arxiv_client.results(search):
        out.append(
            {
                "title": paper.title,
                "url": paper.entry_id,  # e.g. https://arxiv.org/abs/2401.12345v1
                "summary": paper.summary,
                "arxiv_id": paper.get_short_id(),
                "authors": [str(a) for a in paper.authors],
                "published": paper.published.isoformat() if paper.published else None,
                "primary_category": paper.primary_category,
            }
        )
    return out


async def arxiv_search(
    sub_question_id: str,
    query: str,
    top_k: int = 5,
) -> list[SearchResult]:
    """Search arXiv. Returns up to `top_k` ranked results.

    Cached. Empty list on failure (logged, never raises).
    """
    raw = cache.get("arxiv", query, top_k=top_k)
    if raw is None:
        try:
            raw = await asyncio.to_thread(_search_sync, query, top_k)
            cache.set_("arxiv", query, raw, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            print(f"  [arxiv] WARN: arXiv call failed for {query!r}: {e}")
            return []

    return [
        SearchResult(
            sub_question_id=sub_question_id,
            source_type="arxiv",
            title=p["title"].strip().replace("\n", " "),
            url=p["url"],
            snippet=(p["summary"] or "")[:500].strip().replace("\n", " "),
            metadata={
                "arxiv_id": p["arxiv_id"],
                "authors": p["authors"][:5],  # cap to avoid huge metadata blobs
                "published": p["published"],
                "primary_category": p["primary_category"],
            },
            rank=i + 1,
        )
        for i, p in enumerate(raw)
    ]