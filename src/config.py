"""Central config. Loads .env and exposes typed constants.

Keep all magic strings (model names, defaults, paths) here so swapping
providers or tuning limits is a one-file change.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import time
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- API keys ---
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "agentic-research-analyst")

# --- Models ---
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")

# --- Rate limiting ---
# Gemini 2.5 Flash free tier RPM is somewhere between 10-15 depending on the
# December 2025 quota changes. Cap concurrency conservatively and verify
# real limits in AI Studio for your project.
GEMINI_MAX_CONCURRENT_CALLS: int = int(os.getenv("GEMINI_MAX_CONCURRENT_CALLS", "5"))

# --- Paths ---
PROJECT_ROOT: Path = _PROJECT_ROOT
CACHE_DIR: Path = _PROJECT_ROOT / os.getenv("CACHE_DIR", ".cache")
TRACE_DIR: Path = _PROJECT_ROOT / "trace_logs"

# Ensure runtime directories exist
CACHE_DIR.mkdir(exist_ok=True)
TRACE_DIR.mkdir(exist_ok=True)


def assert_required_keys() -> None:
    """Raise if any critical key is missing. Call this from app entry points."""
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill them in."
        )
