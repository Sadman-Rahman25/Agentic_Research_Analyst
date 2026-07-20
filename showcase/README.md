---
title: Agentic Research Analyst Showcase
emoji: microscope
colorFrom: red
colorTo: pink
sdk: static
pinned: false
license: mit
short_description: 5-agent LangGraph pipeline - browse 10 evaluation reports
---

# Agentic Research Analyst - Showcase

Static browse-only showcase of 10 pre-computed reports from the Day 5 benchmark evaluation of the [Agentic Research Analyst](https://github.com/Sadman-Rahman25/Agentic_Research_Analyst) pipeline.

Each report was produced by a 5-agent LangGraph pipeline (Planner -> Researchers -> Synthesizer -> Writer -> Verifier) running on Groq Llama-3.3-70B with Llama-3.1-8B as fallback. Every claim in every report carries an inline `[N]` citation, and every citation is verified against its source with a machine-readable grounding score.

To run new queries live: clone the [GitHub repo](https://github.com/Sadman-Rahman25/Agentic_Research_Analyst) and follow the setup instructions.

## Files in this Space

- `index.html` - the showcase itself (coral-on-black bento UI, client-side query switching)
- `reports.json` - pre-computed data for all 10 queries (query text, plan, findings, report, verification, timings)

No Python runtime, no API calls, no rate limits.