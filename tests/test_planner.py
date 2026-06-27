"""tests/test_planner.py

Light schema-validation tests for the Planner.

These are INTEGRATION tests: they make real Gemini API calls. Each test costs
~1 request against your daily quota. Run them when you change the agent, not
on every save.

We deliberately do NOT test output *quality* here — that's what the manual
eval pass (`python -m src.agents.planner`) is for. We test that the function
returns a well-formed ResearchPlan on known inputs.

Run:
    pytest tests/test_planner.py -v
"""

import pytest

from src.agents.planner import plan
from src.schemas import ResearchPlan, SubQuestion


# A reusable query that should produce a clean plan
SAFE_QUERY = "What is retrieval-augmented generation?"


def test_returns_research_plan():
    """Planner returns a ResearchPlan instance."""
    result = plan(SAFE_QUERY)
    assert isinstance(result, ResearchPlan)


def test_has_reasonable_number_of_sub_questions():
    """1-8 sub-questions; never empty, never absurd."""
    result = plan(SAFE_QUERY)
    assert 1 <= len(result.sub_questions) <= 8
    assert all(isinstance(sq, SubQuestion) for sq in result.sub_questions)


def test_sub_questions_have_valid_sources():
    """Every sub-question is tagged web / arxiv / github."""
    result = plan("How does FAISS handle large-scale vector search?")
    valid = {"web", "arxiv", "github"}
    for sq in result.sub_questions:
        assert sq.source in valid, f"Invalid source: {sq.source!r}"


def test_sub_questions_have_valid_priorities():
    """Priorities are 1, 2, or 3."""
    result = plan("What are recent advances in transformer architecture?")
    for sq in result.sub_questions:
        assert 1 <= sq.priority <= 3, f"Invalid priority: {sq.priority}"


def test_sub_question_ids_are_unique():
    """No duplicate sub-question IDs (would break the orchestrator)."""
    result = plan("Compare PyTorch and TensorFlow.")
    ids = [sq.id for sq in result.sub_questions]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"


def test_preserves_user_query():
    """The plan echoes back the exact user query (verbatim)."""
    query = "What is the role of attention in neural networks?"
    result = plan(query)
    assert result.user_query == query


def test_has_report_sections():
    """The plan defines at least one expected report section."""
    result = plan(SAFE_QUERY)
    assert len(result.expected_report_sections) > 0


def test_raises_on_empty_query():
    """Empty queries are rejected before any LLM call."""
    with pytest.raises(ValueError):
        plan("")
    with pytest.raises(ValueError):
        plan("   ")


def test_respects_max_sub_questions():
    """The max_sub_questions parameter is honored (soft upper bound)."""
    result = plan("Give me a broad overview of modern NLP.", max_sub_questions=4)
    # Allow some slack — model occasionally returns one over. But not 7.
    assert len(result.sub_questions) <= 5