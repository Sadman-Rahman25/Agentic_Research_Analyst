"""Day 1 morning smoke test.

Goal: confirm `langchain-google-genai` works inside a minimal LangGraph node
with structured output, BEFORE building any agents on top.

If this script returns a valid `Hello` object, your SDK versions are
compatible and you can proceed to Planner work.

Run:
    python -m src.smoke_test
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.llm import get_llm


class Hello(BaseModel):
    """Trivial schema to verify structured output works end-to-end."""

    greeting: str = Field(description="A friendly greeting.")
    language: str = Field(description="The language the greeting is in.")
    word_count: int = Field(description="Number of words in the greeting.")


class GraphState(TypedDict):
    """LangGraph state — flows between nodes."""

    user_input: str
    response: Hello | None


def greeter_node(state: GraphState) -> GraphState:
    """One LangGraph node that calls Gemini with structured output."""
    llm = get_llm(provider="gemini", structured=True)
    structured_llm = llm.with_structured_output(Hello)
    result = structured_llm.invoke(
        f"Greet me in any language. Input was: {state['user_input']}"
    )
    return {"user_input": state["user_input"], "response": result}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("greeter", greeter_node)
    graph.add_edge(START, "greeter")
    graph.add_edge("greeter", END)
    return graph.compile()


def main() -> None:
    print("=" * 60)
    print("Day 1 smoke test — Gemini + LangGraph + structured output")
    print("=" * 60)

    app = build_graph()
    result = app.invoke({"user_input": "Hi there", "response": None})

    print(f"\nGreeting: {result['response'].greeting}")
    print(f"Language: {result['response'].language}")
    print(f"Word count: {result['response'].word_count}")
    print("\n✅ Smoke test passed. SDKs are compatible. Proceed to Planner.\n")


if __name__ == "__main__":
    main()
