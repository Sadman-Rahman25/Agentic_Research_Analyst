"""tests/test_researchers.py

Light integration tests for the three researchers.

These hit live APIs (Tavily, arXiv, GitHub). Each suite costs:
  - web: 1 Tavily credit
  - arxiv: 1 polite request
  - github: 1 search call

We test that each researcher returns valid SearchResult objects on a known
query. We do NOT test result quality.

Run:
    pytest tests/test_researchers.py -v
"""

import pytest

from src.researchers.arxiv import arxiv_search
from src.researchers.github import github_search
from src.researchers.web import web_search
from src.schemas import SearchResult


# ---------------------------------------------------------------------------
# Web (Tavily)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_returns_results():
    results = await web_search("sq_test", "What is machine learning?", top_k=3)
    assert len(results) >= 1
    assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_web_tags_source_and_sub_question():
    results = await web_search("sq_abc", "vector databases", top_k=2)
    for r in results:
        assert r.source_type == "web"
        assert r.sub_question_id == "sq_abc"


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arxiv_returns_results():
    results = await arxiv_search("sq_test", "attention is all you need", top_k=3)
    assert len(results) >= 1
    assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_arxiv_metadata_includes_arxiv_id():
    results = await arxiv_search("sq_test", "BERT pretraining", top_k=2)
    assert len(results) >= 1
    assert all("arxiv_id" in r.metadata for r in results)
    assert all(r.source_type == "arxiv" for r in results)


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_github_returns_results():
    results = await github_search("sq_test", "langchain", top_k=3)
    assert len(results) >= 1
    assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_github_metadata_includes_stars():
    results = await github_search("sq_test", "rag retrieval augmented", top_k=2)
    assert len(results) >= 1
    for r in results:
        assert "stars" in r.metadata
        assert isinstance(r.metadata["stars"], int)
        assert r.source_type == "github"


# ---------------------------------------------------------------------------
# Ranks are 1-indexed and sequential
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ranks_are_sequential():
    results = await web_search("sq_test", "transformers neural network", top_k=4)
    ranks = [r.rank for r in results]
    assert ranks == list(range(1, len(ranks) + 1))