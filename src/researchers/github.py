"""src/researchers/github.py

GitHub repository search wrapper.

Pure tool call — NO LLM. Uses PyGithub. Wrapped with asyncio.to_thread().

GitHub rate limits: 60 req/hr unauthenticated (hostile), 5000 req/hr with PAT
(comfortable). We assume PAT is set; the lazy init enforces it.

KEYWORD EXTRACTION: GitHub's repo search ANDs all query terms across repo
name + description + README. Free-form questions from the Planner have 8-15
words, most of which are noise; ANDing them all guarantees zero hits.

We aggressively strip noise and cap the query at 3 substantive terms. Concrete
example:

    "Which open-source implementations of sparse retrieval are most widely used?"
        ↓
    "sparse retrieval rag"   ← 3 terms, GitHub-friendly

Truly niche sub-questions ("LangGraph deployment failures") may still return
zero — those are recall limits of GitHub's name/description index, not bugs
in this code. The pipeline tolerates empty results gracefully downstream.
"""

from __future__ import annotations

import asyncio
import re

from github import Github, GithubException

from src import cache
from src.config import GITHUB_TOKEN
from src.schemas import SearchResult


# Words that GitHub's repo search treats as noise. The list is intentionally
# aggressive — including words that ARE technically content-bearing in English
# but that ALMOST NEVER appear in a useful repo's name/description (e.g.,
# "frequently", "widely", "compare"). Lossy on purpose: we'd rather lose
# nuance and get hits than preserve nuance and get zeros.
_NOISE_WORDS: set[str] = {
    # WH-words, copulas, modals
    "what", "which", "how", "why", "when", "where", "who", "whose",
    "are", "is", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "doing",
    "can", "could", "should", "would", "may", "might", "will", "shall",
    "have", "has", "had", "having",
    # Articles, prepositions, conjunctions
    "the", "a", "an",
    "of", "in", "on", "for", "to", "with", "and", "or", "but", "by",
    "at", "from", "as", "into", "than", "that", "this", "these", "those",
    "between", "across", "about", "through",
    # Generic qualifiers / intensifiers
    "most", "common", "best", "known", "current", "available", "real",
    "world", "modern", "recent", "latest", "various", "different", "new",
    "popular", "well", "widely", "used", "using",
    "frequently", "reported", "current",
    # Meta-words about GitHub itself (every repo is open, public, source code)
    "public", "private", "repositories", "repository", "repos", "repo",
    "open", "source", "opensource",
    "project", "projects", "library", "libraries", "code", "codebase",
    # Generic nouns that match too broadly
    "implementations", "implementation", "frameworks", "framework",
    "approach", "approaches", "system", "systems",
    "method", "methods", "way", "ways", "tools", "tool",
    "solutions", "solution", "examples", "example",
    # Vague state / activity nouns
    "production", "deployment", "deployments", "performance",
    "configuration", "configurations",
    "errors", "error", "issues", "issue", "failures", "failure",
    "challenges", "challenge", "limitations", "limitation",
    "practices", "practice",
    # Vague verbs
    "compare", "comparison", "comparing", "evaluating", "evaluate",
    "handle", "handling", "deploy", "deploying",
}


def _to_keyword_query(question: str) -> str:
    """Convert an English question into a 2-3 token GitHub keyword query.

    Strategy:
      1. Lowercase + replace all non-word/space chars (including hyphens) with spaces
      2. Drop noise words and very short tokens (<3 chars)
      3. Cap to top 3 tokens (GitHub's AND semantics punish more)
      4. Fall back to the original question if nothing useful survives
    """
    cleaned = re.sub(r"[^\w\s]", " ", question.lower())
    tokens = [
        t for t in cleaned.split()
        if t and t not in _NOISE_WORDS and len(t) > 2
    ]
    return " ".join(tokens[:3]) if tokens else question


_gh: Github | None = None


def _get_client() -> Github:
    global _gh
    if _gh is None:
        if not GITHUB_TOKEN:
            raise RuntimeError("GITHUB_TOKEN not set. Check .env.")
        _gh = Github(GITHUB_TOKEN, per_page=10)
    return _gh


def _search_sync(query: str, top_k: int) -> list[dict]:
    """Sync GitHub search. Extracts everything to dicts to avoid lazy loads later."""
    gh = _get_client()
    out: list[dict] = []
    keyword_query = _to_keyword_query(query)
    try:
        # Sort by stars (popularity proxy); descending.
        repos = gh.search_repositories(query=keyword_query, sort="stars", order="desc")
        for i, repo in enumerate(repos):
            if i >= top_k:
                break
            out.append(
                {
                    "full_name": repo.full_name,
                    "html_url": repo.html_url,
                    "description": repo.description or "",
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "language": repo.language,
                    "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                    "topics": list(repo.topics) if repo.topics else [],
                }
            )
    except GithubException as e:
        # Rate-limit hits and bad queries land here; let the caller turn it into []
        raise RuntimeError(f"GitHub API error (status={e.status}): {e.data}") from e
    return out


async def github_search(
    sub_question_id: str,
    query: str,
    top_k: int = 5,
) -> list[SearchResult]:
    """Search GitHub repositories. Returns up to `top_k` ranked by stars.

    The raw English `query` is sent through `_to_keyword_query()` internally
    before hitting the API. The cache key uses the original query string so
    semantically identical sub-questions still dedupe correctly.

    Cached. Empty list on failure (logged, never raises).
    """
    raw = cache.get("github", query, top_k=top_k)
    if raw is None:
        try:
            raw = await asyncio.to_thread(_search_sync, query, top_k)
            cache.set_("github", query, raw, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            print(f"  [github] WARN: GitHub call failed for {query!r}: {e}")
            return []

    return [
        SearchResult(
            sub_question_id=sub_question_id,
            source_type="github",
            title=r["full_name"],
            url=r["html_url"],
            snippet=(r["description"] or "")[:500],
            metadata={
                "stars": r["stars"],
                "forks": r["forks"],
                "language": r["language"],
                "updated_at": r["updated_at"],
                "topics": r["topics"][:5],
            },
            rank=i + 1,
        )
        for i, r in enumerate(raw)
    ]