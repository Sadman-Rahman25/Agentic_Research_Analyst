"""LLM factory. One function returns the right chat model per provider.

Why a factory: each agent declares its provider preference (gemini by default,
groq for the Synthesizer if the live trace feels sluggish). A factory means
Day 5 provider swaps are a one-line change in the agent, not a refactor.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel

from src.config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL

Provider = Literal["gemini", "groq"]


def get_llm(
    provider: Provider = "gemini",
    *,
    structured: bool = True,
    temperature: float = 0.2,
) -> BaseChatModel:
    """Return a chat model configured for the requested provider.

    Args:
        provider: 'gemini' (default) or 'groq'.
        structured: If True and provider is gemini, request JSON output mode.
            Ignored for groq (use `with_structured_output` instead).
        temperature: Sampling temperature. 0.2 is a sane default for agents.
    """
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set. Check .env.")

        kwargs: dict = {
            "model": GEMINI_MODEL,
            "temperature": temperature,
            "google_api_key": GEMINI_API_KEY,
        }
        if structured:
            # Native JSON mode — Gemini's structured-output enforcement.
            # Use together with .with_structured_output(MyModel) on the model.
            kwargs["model_kwargs"] = {"response_mime_type": "application/json"}
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "groq":
        from langchain_groq import ChatGroq

        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set. Check .env.")

        return ChatGroq(
            model=GROQ_MODEL,
            temperature=temperature,
            api_key=GROQ_API_KEY,
        )

    raise ValueError(f"Unknown provider: {provider!r}. Use 'gemini' or 'groq'.")
