# Agentic Research Analyst

A multi-agent research assistant that decomposes a research question, executes parallel searches across the web, arXiv, and GitHub, synthesizes findings with citations, drafts a structured report, and verifies every claim against its source.

Built with **LangGraph**, **Gemini 2.5 Flash** (primary), and **Groq Llama-3.3-70B** (fallback for streaming).

> **Status:** Day 0 — scaffolding complete. Building.

---

## Why this exists

Demonstrates agentic orchestration, multi-tool reasoning, parallel research, and verifiable structured output — the capabilities that matter in 2026 AI engineering and that production systems actually need.

## Architecture

```
User Query
   ↓
Planner (Gemini 2.5 Flash) → ResearchPlan (5–8 sub-questions)
   ↓
Parallel Researchers (Tavily / arXiv / GitHub) → SearchResult[]
   ↓
Synthesizer (Gemini) → Finding[] (one claim, one quote, one URL)
   ↓
Writer (Gemini, 1M context) → markdown report with [1]-style citations
   ↓
Verifier (Gemini) → grounding score + flagged claims
```

Five agents. Strict separation of extract (Synthesizer) from compose (Writer) so the writer can't hallucinate beyond its sources.

## Quick start

```bash
# Clone and enter
git clone https://github.com/Sadman-Rahman25/agentic-research-analyst
cd agentic-research-analyst

# Virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows PowerShell
# source venv/bin/activate      # macOS/Linux

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env             # or copy .env.example .env on Windows
# Fill in: GEMINI_API_KEY, GROQ_API_KEY, TAVILY_API_KEY, GITHUB_TOKEN

# Smoke test
python -m src.smoke_test

# Run UI
streamlit run app.py
```

## Project layout

```
agentic-research-analyst/
├── app.py                  # Streamlit entry point
├── src/
│   ├── config.py           # Env vars, model names, paths
│   ├── schemas.py          # All Pydantic models
│   ├── llm.py              # get_llm() factory: gemini | groq
│   ├── smoke_test.py       # Day 1 morning sanity check
│   ├── state.py            # AgentState TypedDict for LangGraph
│   ├── graph.py            # StateGraph wiring
│   ├── agents/             # Planner, Synthesizer, Writer, Verifier
│   └── researchers/        # Web (Tavily), arXiv, GitHub
├── tests/
│   └── test_queries.json   # 10 hand-crafted eval queries
├── examples/               # Polished example reports
└── docs/                   # Architecture diagram, writeup
```

## Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph |
| Primary LLM | Gemini 2.5 Flash (`langchain-google-genai`) |
| Fallback LLM | Groq Llama-3.3-70B-versatile |
| Structured output | Pydantic v2 + Gemini native JSON mode |
| Web search | Tavily |
| arXiv | `arxiv` Python package |
| GitHub | PyGithub |
| UI | Streamlit |
| Deployment | HuggingFace Spaces (CPU free tier) |

## Evaluation

Three configurations × 10 queries × 3 metrics. Table goes here once Day 4 runs.

| Config | Groundedness | Coverage (1–5) | Latency | Tokens |
|---|---|---|---|---|
| Single-call baseline | — | — | — | — |
| Single-agent + tools | — | — | — | — |
| Full pipeline | — | — | — | — |

## Honest limitations

To be written on Day 6. Expected: full pipeline costs more tokens and is slower; for simple queries the single-call baseline is competitive; the verifier is necessary but adds latency.

## License

MIT
