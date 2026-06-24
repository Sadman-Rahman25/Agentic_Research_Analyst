"""Pydantic schemas — the data contracts between agents.

Every agent has a clear input schema and clear output schema. These models
ARE the API surface of the pipeline. If you change them, you change the
contract between stages — do it deliberately.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Planner output
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """A single sub-question produced by the Planner."""

    id: str = Field(description="Stable identifier like 'sq_1', 'sq_2'.")
    question: str = Field(description="The sub-question itself, phrased clearly.")
    rationale: str = Field(description="Why this sub-question matters for the user query.")
    source: Literal["web", "arxiv", "github"] = Field(
        description="Which researcher should answer this. "
        "web=current events/blogs/docs, arxiv=academic papers, github=code/implementations."
    )
    priority: int = Field(ge=1, le=3, description="1=critical, 2=important, 3=nice-to-have.")


class ResearchPlan(BaseModel):
    """The full research plan emitted by the Planner."""

    user_query: str
    sub_questions: list[SubQuestion]
    expected_report_sections: list[str] = Field(
        description="The Writer will use these as section headers."
    )


# ---------------------------------------------------------------------------
# Researcher output (shared across web / arxiv / github)
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """One result from a researcher tool. Researchers do NOT use LLMs."""

    sub_question_id: str
    source_type: Literal["web", "arxiv", "github"]
    title: str
    url: str
    snippet: str = Field(description="Short summary returned by the search API.")
    full_text: Optional[str] = Field(
        default=None,
        description="Full extracted text. Populated only for the top-N results to save tokens.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Source-specific fields: arxiv_id, authors, stars, etc.",
    )
    rank: int = Field(description="1-indexed rank within this researcher's response.")


# ---------------------------------------------------------------------------
# Synthesizer output
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """An atomic factual claim extracted from one source, with citation."""

    sub_question_id: str
    claim: str = Field(description="One factual statement, one sentence.")
    source_url: str
    source_quote: str = Field(
        max_length=300,
        description="Exact passage (≤30 words) from the source supporting the claim.",
    )
    confidence: Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Verifier output
# ---------------------------------------------------------------------------


class FlaggedClaim(BaseModel):
    """A claim in the report that the Verifier flagged as problematic."""

    claim_text: str
    cited_source: str
    issue: Literal["unsupported", "contradicted", "partial_support"]
    explanation: str


class VerificationResult(BaseModel):
    total_claims: int
    verified: int
    flagged: list[FlaggedClaim]
    overall_grounding_score: float = Field(
        ge=0.0, le=1.0, description="verified / total_claims"
    )
