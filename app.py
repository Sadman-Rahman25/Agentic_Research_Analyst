"""app.py — Streamlit UI for Agentic Research Analyst.

Run locally:
    streamlit run app.py

Design: coral-on-black bento dashboard.
- Primary accent: coral #FF5C39 (citations, buttons, brand)
- Grounding: lime green #C5F560
- Partial-support flags: gold yellow #FFD557
- Unsupported flags: hot pink #FF3F8B
- arXiv source pills: violet #B794F4

Layout:
- Sidebar: query input, display toggles, Run pipeline button
- Header: title + Download report button
- Hero row: big grounding card + 3 metric tiles
- Per-agent trace (timings + plan)
- Findings (preview 5, "View all" to expand)
- Report (with secondary Download .md button)
- Verification (preview 3 flagged, "View all" to expand)
- Pipeline errors (collapsed expander)

Architecture: LangGraph from src/graph.py invoked via asyncio.run().
State persists in st.session_state so toggles don't re-trigger the pipeline.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import streamlit as st

from src.graph import build_graph
from src.schemas import Finding, VerificationResult, FlaggedClaim


# ============================================================
# 1. Page config — MUST be the first Streamlit call
# ============================================================

st.set_page_config(
    page_title="Agentic research analyst",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. Tabler icons + custom CSS — dark coral-on-black theme
# ============================================================

st.markdown(
    """
<link href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/tabler-icons.min.css" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

CUSTOM_CSS = """
<style>
/* ============ Base palette ============ */
.stApp {
    background: #0A0A0A;
    color: #FFFFFF;
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* Hide Streamlit chrome — but keep the header IN the DOM so the sidebar toggle stays clickable */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent !important;
    height: auto !important;
}
/* The sidebar collapse/expand chevron — make sure it stays visible in coral */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] {
    color: #FF5C39 !important;
    visibility: visible !important;
}
[data-testid="stSidebarCollapseButton"] button svg,
[data-testid="collapsedControl"] svg {
    fill: #FF5C39 !important;
    color: #FF5C39 !important;
}

/* ============ Sidebar ============ */
[data-testid="stSidebar"] {
    background: #050505;
    border-right: 1px solid #1F1F1F;
}
[data-testid="stSidebar"] * { color: #C8C8C8; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 10px !important;
    color: #5A5A5A !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ============ Text inputs ============ */
.stTextArea textarea {
    background: #0A0A0A !important;
    color: #FFFFFF !important;
    border: 1px solid #1F1F1F !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #FF5C39 !important;
    box-shadow: 0 0 0 1px #FF5C39 !important;
}

/* ============ Checkboxes ============ */
.stCheckbox > label {
    color: #C8C8C8 !important;
    font-size: 13px !important;
}
.stCheckbox [data-baseweb="checkbox"] [data-checked="true"] {
    background-color: #FF5C39 !important;
    border-color: #FF5C39 !important;
}

/* ============ Buttons ============ */
/* Primary buttons (Run, Download) — solid coral pill */
.stButton > button[kind="primary"],
.stDownloadButton > button {
    background: #FF5C39 !important;
    color: #0A0A0A !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 9px 18px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: background 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button:hover {
    background: #FF7252 !important;
    color: #0A0A0A !important;
}

/* Secondary buttons (View more) — outlined coral pill */
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #FF5C39 !important;
    border: 1px solid #FF5C39 !important;
    border-radius: 999px !important;
    padding: 6px 14px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #2A1108 !important;
    color: #FF5C39 !important;
}

/* ============ Status widget ============ */
.stStatusWidget, [data-testid="stStatusWidget"] {
    background: #141414 !important;
    border: 1px solid #1F1F1F !important;
    border-radius: 12px !important;
}

/* ============ Expanders ============ */
.stExpander {
    background: #141414 !important;
    border: 1px solid #1F1F1F !important;
    border-radius: 12px !important;
}
.stExpander summary, .stExpander summary p {
    color: #C8C8C8 !important;
    font-size: 12px !important;
}

/* ============ Markdown rendering (the report body) ============ */
.report-body { color: #C8C8C8; line-height: 1.7; font-size: 14px; }
.report-body h1, .report-body h2, .report-body h3 {
    color: #FFFFFF;
    font-weight: 500;
    letter-spacing: -0.01em;
}
.report-body h1 { font-size: 22px; margin: 0 0 12px; }
.report-body h2 {
    font-size: 17px; margin: 20px 0 10px;
}
.report-body h2::after {
    content: ".";
    color: #FF5C39;
}
.report-body h3 { font-size: 14px; margin: 14px 0 6px; }
.report-body a { color: #FF5C39; text-decoration: none; }
.report-body a:hover { text-decoration: underline; }
.report-body strong { color: #FFFFFF; font-weight: 500; }
.report-body code {
    background: #1A1A1A;
    color: #FF5C39;
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 13px;
}
.report-body ul, .report-body ol { color: #C8C8C8; padding-left: 1.5rem; }

/* Coral citation numbers [N] inside the report */
.report-body p {
    color: #C8C8C8;
}

/* ============ Bento cards (custom HTML) ============ */
.bento {
    background: #141414;
    border: 1px solid #1F1F1F;
    border-radius: 12px;
    padding: 14px 16px;
}
.label-uc {
    font-size: 10px;
    color: #5A5A5A;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.tag {
    font-family: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    white-space: nowrap;
    font-weight: 500;
}
.cite {
    color: #FF5C39;
    font-family: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace;
    font-size: 12px;
    font-weight: 500;
}
.coral-dot { color: #FF5C39; }

/* ============ Scrollbar ============ */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0A0A0A; }
::-webkit-scrollbar-thumb { background: #2E2E2E; border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: #FF5C39; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# 3. Graph compilation (cached for the process lifetime)
# ============================================================


@st.cache_resource
def get_compiled_graph():
    return build_graph()


_APP = get_compiled_graph()


# ============================================================
# 4. Helpers
# ============================================================


def slugify(text: str, max_len: int = 60) -> str:
    """Make a filename-safe slug from query text."""
    slug = re.sub(r"[^a-zA-Z0-9\s-]", "", text.lower())
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug[:max_len] or "report"


async def run_with_progress(query: str, status_widget) -> dict[str, Any]:
    """Stream node-completion events into the status widget. Accumulate to final state."""
    final_state: dict[str, Any] = {"user_query": query}
    stage_labels = {
        "planner": "Planning sub-questions",
        "researchers": "Searching web, arXiv, GitHub",
        "synthesizer": "Extracting findings",
        "writer": "Composing report",
        "verifier": "Verifying claims",
    }
    async for chunk in _APP.astream({"user_query": query}, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        for node_name, node_state in chunk.items():
            if not isinstance(node_state, dict):
                continue
            final_state = {**final_state, **node_state}
            label = stage_labels.get(node_name, node_name)
            status_widget.write(f"✓ **{label}** complete")
    return final_state


# ============================================================
# 5. HTML render helpers — bento sections
# ============================================================


def render_header_html(query: str) -> str:
    """Page title + tagline. Download button rendered separately as native widget."""
    return """
<div style="margin-bottom: 4px;">
  <h1 style="font-size: 24px; margin: 0 0 4px; font-weight: 500; color: #FFFFFF; letter-spacing: -0.01em;">Agentic research analyst<span class="coral-dot">.</span></h1>
  <p style="font-size: 13px; color: #A8A8A8; margin: 0; line-height: 1.5;">Multi-agent LLM pipeline that produces citation-grounded research reports with a measurable grounding score.</p>
</div>
"""


def render_metrics_row_html(state: dict[str, Any]) -> str:
    """Hero grounding card + 3 metric tiles (sub-questions, findings, wall clock)."""
    findings = state.get("findings", []) or []
    timings = state.get("timings", {}) or {}
    plan = state.get("plan")
    n_sq = len(plan.sub_questions) if plan else 0
    total_time = sum(timings.values()) if timings else 0
    verification = state.get("verification")

    if isinstance(verification, VerificationResult) and verification.total_claims > 0:
        pct = int(round(verification.overall_grounding_score * 100))
        grounding_inner = f"""
<div style="font-size: 56px; font-weight: 500; color: #C5F560; line-height: 1; letter-spacing: -0.02em;">{pct}<span style="font-size: 28px;">%</span></div>
<div style="height: 4px; background: #1F1F1F; border-radius: 4px; margin: 10px 0 8px; overflow: hidden;">
  <div style="width: {pct}%; height: 100%; background: #C5F560; border-radius: 4px;"></div>
</div>
<div style="font-size: 11px; color: #C5F560;">{verification.verified} verified · {len(verification.flagged)} flagged · {verification.total_claims} total</div>
"""
    else:
        grounding_inner = """
<div style="font-size: 36px; font-weight: 500; color: #5A5A5A; line-height: 1;">—</div>
<div style="font-size: 11px; color: #5A5A5A; margin-top: 8px;">No verifiable claims</div>
"""

    return f"""
<div style="display: grid; grid-template-columns: 280px 1fr; gap: 10px; margin-bottom: 14px;">
  <div style="background: #0F0F0F; border: 1px solid #1F2A0A; border-radius: 12px; padding: 16px; display: flex; flex-direction: column; justify-content: space-between; min-height: 160px;">
    <div style="display: flex; align-items: center; gap: 6px;">
      <i class="ti ti-shield-check" style="font-size: 14px; color: #C5F560;"></i>
      <span style="font-size: 10px; color: #C5F560; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;">Grounding</span>
    </div>
    <div>{grounding_inner}</div>
  </div>
  <div style="display: grid; grid-template-rows: 1fr 1fr 1fr; gap: 8px;">
    <div class="bento" style="display: flex; align-items: center; justify-content: space-between;">
      <div>
        <div style="font-size: 10px; color: #5A5A5A; text-transform: uppercase; letter-spacing: 0.08em;">Sub-questions</div>
        <div style="font-size: 24px; font-weight: 500; color: #FFFFFF;">{n_sq}</div>
      </div>
      <i class="ti ti-list-numbers" style="font-size: 20px; color: #FF5C39;"></i>
    </div>
    <div class="bento" style="display: flex; align-items: center; justify-content: space-between;">
      <div>
        <div style="font-size: 10px; color: #5A5A5A; text-transform: uppercase; letter-spacing: 0.08em;">Findings</div>
        <div style="font-size: 24px; font-weight: 500; color: #FFFFFF;">{len(findings)}</div>
      </div>
      <i class="ti ti-quote" style="font-size: 20px; color: #FF5C39;"></i>
    </div>
    <div class="bento" style="display: flex; align-items: center; justify-content: space-between;">
      <div>
        <div style="font-size: 10px; color: #5A5A5A; text-transform: uppercase; letter-spacing: 0.08em;">Wall clock</div>
        <div style="font-size: 24px; font-weight: 500; color: #FFFFFF;">{total_time:.1f}s</div>
      </div>
      <i class="ti ti-clock" style="font-size: 20px; color: #FF5C39;"></i>
    </div>
  </div>
</div>
"""


def _source_tag_html(source: str, priority: int) -> str:
    """Color-coded source/priority pill."""
    bg_color = {
        "web": "background: #2A1108; color: #FF5C39;",
        "arxiv": "background: #1F1429; color: #B794F4;",
        "github": "background: #0F0F0F; color: #A8A8A8; border: 1px solid #1F1F1F;",
    }.get(source, "background: #0F0F0F; color: #A8A8A8;")
    return f'<span class="tag" style="{bg_color}">{source} P{priority}</span>'


def render_trace_html(state: dict[str, Any]) -> str:
    """Per-agent trace: timings on left, plan on right."""
    timings = state.get("timings", {}) or {}
    plan = state.get("plan")

    timings_rows = ""
    for stage in ("plan", "search", "synthesize", "write", "verify"):
        if stage in timings:
            timings_rows += f"""
<div style="display: flex; justify-content: space-between;">
  <span style="color: #A8A8A8;">{stage}</span>
  <span style="color: #FFFFFF;">{timings[stage]:.2f}s</span>
</div>"""

    plan_rows = ""
    if plan and plan.sub_questions:
        for sq in plan.sub_questions:
            plan_rows += f"""
<div style="display: flex; gap: 8px; align-items: flex-start;">
  {_source_tag_html(sq.source, sq.priority)}
  <span>{sq.question}</span>
</div>"""
    else:
        plan_rows = '<div style="color: #5A5A5A; font-size: 11px;">No plan available</div>'

    return f"""
<section class="bento" style="margin-bottom: 14px;">
  <div class="label-uc"><i class="ti ti-route" style="font-size: 13px; color: #FF5C39;"></i>Per-agent trace</div>
  <div style="display: grid; grid-template-columns: 140px 1fr; gap: 18px;">
    <div>
      <div style="font-size: 10px; color: #5A5A5A; margin-bottom: 8px; letter-spacing: 0.08em;">TIMINGS</div>
      <div style="font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.9;">
        {timings_rows}
      </div>
    </div>
    <div>
      <div style="font-size: 10px; color: #5A5A5A; margin-bottom: 8px; letter-spacing: 0.08em;">PLAN · {len(plan.sub_questions) if plan else 0} SUB-QUESTIONS</div>
      <div style="display: flex; flex-direction: column; gap: 6px; font-size: 12px; line-height: 1.5; color: #C8C8C8;">
        {plan_rows}
      </div>
    </div>
  </div>
</section>
"""


def _finding_card_html(finding: Finding, idx: int) -> str:
    """One finding card with confidence pill + claim + quote + URL."""
    conf_styles = {
        "high": "background: #1F2A0A; color: #C5F560;",
        "medium": "background: #2E2308; color: #FFD557;",
        "low": "background: #0F0F0F; color: #A8A8A8; border: 1px solid #2E2E2E;",
    }
    conf_style = conf_styles.get(finding.confidence, conf_styles["medium"])
    conf_label = finding.confidence[:4]  # "high", "med", "low"
    # Show URL host + truncated path for readability
    url_display = finding.source_url.replace("https://", "").replace("http://", "")
    if len(url_display) > 60:
        url_display = url_display[:57] + "..."
    return f"""
<div style="border: 1px solid #1F1F1F; border-radius: 8px; padding: 12px 14px; background: #0F0F0F;">
  <div style="display: flex; gap: 10px; align-items: flex-start;">
    <span class="tag" style="{conf_style}">[{idx}] {conf_label}</span>
    <div style="font-size: 13px; line-height: 1.5; flex: 1; color: #FFFFFF;">{finding.claim}</div>
  </div>
  <div style="font-size: 12px; color: #A8A8A8; padding-left: 12px; border-left: 2px solid #FF5C39; font-style: italic; line-height: 1.6; margin-top: 8px;">"{finding.source_quote}"</div>
  <div style="font-size: 11px; color: #FF5C39; margin-top: 8px; font-family: ui-monospace, monospace; display: flex; align-items: center; gap: 4px;">
    <i class="ti ti-external-link" style="font-size: 11px;"></i> {url_display}
  </div>
</div>
"""


def render_findings_section_html(findings: list[Finding], show_all: bool, preview_count: int = 5) -> str:
    """Findings section with preview + view-more pattern."""
    if not findings:
        return """
<section class="bento" style="margin-bottom: 14px;">
  <div class="label-uc"><i class="ti ti-quote" style="font-size: 13px; color: #FF5C39;"></i>Extracted findings</div>
  <div style="color: #5A5A5A; font-size: 12px; text-align: center; padding: 20px;">No findings extracted</div>
</section>
"""

    to_show = findings if show_all else findings[:preview_count]
    cards = "".join(
        _finding_card_html(f, idx) for idx, f in enumerate(to_show, start=1)
    )

    header = f"""
<section class="bento" style="margin-bottom: 14px;">
  <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
    <div class="label-uc" style="margin-bottom: 0;"><i class="ti ti-quote" style="font-size: 13px; color: #FF5C39;"></i>Extracted findings</div>
    <span style="font-size: 12px; color: #A8A8A8; font-family: ui-monospace, monospace;">{len(findings)} total</span>
  </div>
  <div style="display: flex; flex-direction: column; gap: 10px;">
    {cards}
  </div>
"""

    # The View more button is rendered as a native Streamlit widget OUTSIDE this HTML
    # We close the section here, the button will be appended below by the main flow.
    footer = "</section>"
    return header + footer


def render_report_html(report_md: str) -> str:
    """Wrap the markdown report in a styled container.
    Note: report_md is already markdown text from the Writer; Streamlit will render it.
    """
    return f'<div class="bento" style="margin-bottom: 14px; padding: 18px 20px;"><div class="label-uc" style="margin-bottom: 14px;"><i class="ti ti-file-text" style="font-size: 13px; color: #FF5C39;"></i>Research report</div><div class="report-body">'


def render_report_close() -> str:
    return "</div></div>"


def _flagged_card_html(fc: FlaggedClaim) -> str:
    """One flagged-claim tile, color-coded by verdict."""
    if fc.issue == "partial_support":
        bg = "#1A1408"
        border = "#2E2308"
        pill_bg = "#2E2308"
        text_main = "#FFFFFF"
        text_sub = "#FFD557"
        accent = "#FFD557"
    elif fc.issue == "contradicted":
        bg = "#1A0810"
        border = "#3D0D20"
        pill_bg = "#3D0D20"
        text_main = "#FFFFFF"
        text_sub = "#FF3F8B"
        accent = "#FF3F8B"
    else:  # unsupported
        bg = "#1A0810"
        border = "#3D0D20"
        pill_bg = "#3D0D20"
        text_main = "#FFFFFF"
        text_sub = "#FF3F8B"
        accent = "#FF3F8B"

    return f"""
<div style="background: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 12px 14px;">
  <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
    <span class="tag" style="background: {pill_bg}; color: {accent};">{fc.issue}</span>
    <span style="color: {accent}; font-family: ui-monospace, monospace; font-size: 11px;">→ {fc.cited_source.replace('https://', '').replace('http://', '')[:60]}</span>
  </div>
  <div style="font-size: 13px; color: {text_main}; line-height: 1.5; margin-bottom: 6px;">{fc.claim_text}</div>
  <div style="font-size: 12px; color: {text_sub}; line-height: 1.6;">{fc.explanation}</div>
</div>
"""


def render_verification_section_html(
    verification: VerificationResult, show_all_flagged: bool, preview_count: int = 3
) -> str:
    """Verification section header + verified/flagged tiles + flagged list."""
    if verification.total_claims == 0:
        return """
<section class="bento" style="margin-bottom: 14px;">
  <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
    <i class="ti ti-shield-check" style="font-size: 15px; color: #5A5A5A;"></i>
    <h2 style="font-size: 16px; margin: 0; font-weight: 500; color: #FFFFFF;">Verification<span class="coral-dot">.</span></h2>
  </div>
  <div style="color: #5A5A5A; font-size: 12px; text-align: center; padding: 16px;">No verifiable claims in this report (no inline citations found)</div>
</section>
"""

    flagged = verification.flagged
    to_show = flagged if show_all_flagged else flagged[:preview_count]
    flagged_cards = "".join(_flagged_card_html(fc) for fc in to_show)

    if not flagged:
        flagged_section = '<div style="font-size: 12px; color: #C5F560; text-align: center; padding: 10px; font-style: italic;">All checkable claims verified against their cited findings.</div>'
    else:
        flagged_section = f"""
<div style="display: flex; flex-direction: column; gap: 10px;">
  {flagged_cards}
</div>
"""

    return f"""
<section class="bento" style="margin-bottom: 14px;">
  <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
    <i class="ti ti-shield-check" style="font-size: 16px; color: #C5F560;"></i>
    <h2 style="font-size: 17px; margin: 0; font-weight: 500; color: #FFFFFF;">Verification<span class="coral-dot">.</span></h2>
  </div>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
    <div style="background: #0F0F0F; border: 1px solid #1F2A0A; border-radius: 8px; padding: 12px 14px;">
      <div style="font-size: 10px; color: #C5F560; letter-spacing: 0.08em; margin-bottom: 6px;">VERIFIED</div>
      <div style="font-size: 24px; font-weight: 500; color: #C5F560; line-height: 1.1;">{verification.verified}<span style="font-size: 14px; color: #A8A8A8;"> / {verification.total_claims}</span></div>
    </div>
    <div style="background: #0F0F0F; border: 1px solid #2E2308; border-radius: 8px; padding: 12px 14px;">
      <div style="font-size: 10px; color: #FFD557; letter-spacing: 0.08em; margin-bottom: 6px;">FLAGGED</div>
      <div style="font-size: 24px; font-weight: 500; color: #FFD557; line-height: 1.1;">{len(flagged)}</div>
    </div>
  </div>
  {flagged_section}
</section>
"""


def render_welcome_html() -> str:
    """First-load screen explaining what the system does."""
    return """
<div style="background: #0F0F0F; border: 1px solid #1F1F1F; border-radius: 12px; padding: 28px 32px; max-width: 900px; margin: 40px auto;">
  <div style="text-align: center; margin-bottom: 24px;">
    <h1 style="font-size: 28px; margin: 0 0 8px; font-weight: 500; color: #FFFFFF; letter-spacing: -0.01em;">Agentic research analyst<span class="coral-dot">.</span></h1>
    <p style="color: #A8A8A8; font-size: 14px; margin: 0; line-height: 1.6;">A 5-agent LangGraph pipeline that decomposes your question, searches the web/arXiv/GitHub in parallel, extracts atomic claims with verbatim quotes, composes a citation-grounded report, and verifies every claim against its source.</p>
  </div>
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 24px 0;">
    <div style="background: #141414; border: 1px solid #1F1F1F; border-radius: 8px; padding: 12px 10px; text-align: center;">
      <div style="font-size: 10px; color: #5A5A5A; letter-spacing: 0.08em; margin-bottom: 6px;">STEP 1</div>
      <div style="font-size: 12px; color: #FFFFFF; font-weight: 500;">Plan</div>
      <div style="font-size: 10px; color: #A8A8A8; margin-top: 4px;">Llama-3.3-70B</div>
    </div>
    <div style="background: #141414; border: 1px solid #1F1F1F; border-radius: 8px; padding: 12px 10px; text-align: center;">
      <div style="font-size: 10px; color: #5A5A5A; letter-spacing: 0.08em; margin-bottom: 6px;">STEP 2</div>
      <div style="font-size: 12px; color: #FFFFFF; font-weight: 500;">Search</div>
      <div style="font-size: 10px; color: #A8A8A8; margin-top: 4px;">Tavily · arXiv · GitHub</div>
    </div>
    <div style="background: #141414; border: 1px solid #1F1F1F; border-radius: 8px; padding: 12px 10px; text-align: center;">
      <div style="font-size: 10px; color: #5A5A5A; letter-spacing: 0.08em; margin-bottom: 6px;">STEP 3</div>
      <div style="font-size: 12px; color: #FFFFFF; font-weight: 500;">Extract</div>
      <div style="font-size: 10px; color: #A8A8A8; margin-top: 4px;">Llama-3.1-8B</div>
    </div>
    <div style="background: #141414; border: 1px solid #1F1F1F; border-radius: 8px; padding: 12px 10px; text-align: center;">
      <div style="font-size: 10px; color: #5A5A5A; letter-spacing: 0.08em; margin-bottom: 6px;">STEP 4</div>
      <div style="font-size: 12px; color: #FFFFFF; font-weight: 500;">Compose</div>
      <div style="font-size: 10px; color: #A8A8A8; margin-top: 4px;">Llama-3.3-70B</div>
    </div>
    <div style="background: #141414; border: 1px solid #1F2A0A; border-radius: 8px; padding: 12px 10px; text-align: center;">
      <div style="font-size: 10px; color: #C5F560; letter-spacing: 0.08em; margin-bottom: 6px;">STEP 5</div>
      <div style="font-size: 12px; color: #C5F560; font-weight: 500;">Verify</div>
      <div style="font-size: 10px; color: #A8A8A8; margin-top: 4px;">Grounding score</div>
    </div>
  </div>
  <div style="color: #A8A8A8; font-size: 13px; line-height: 1.7; padding: 16px; background: #0A0A0A; border-radius: 8px; border-left: 3px solid #FF5C39;">
    Enter your research question in the sidebar and press <span style="color: #FF5C39; font-weight: 500;">Run pipeline</span>. Queries take 60-180 seconds. The pipeline produces a markdown report with inline <span class="cite">[N]</span> citations and a verification footer showing which claims were verified or flagged.
  </div>
</div>
"""


# ============================================================
# 6. Sidebar
# ============================================================

with st.sidebar:
    st.markdown(
        """
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px; padding-bottom: 14px; border-bottom: 1px solid #1F1F1F;">
  <div style="width: 32px; height: 32px; border-radius: 8px; background: #FF5C39; color: #0A0A0A; display: flex; align-items: center; justify-content: center;">
    <i class="ti ti-flask-2" style="font-size: 17px;"></i>
  </div>
  <div style="font-size: 14px; font-weight: 500; color: #FFFFFF;">Research<span class="coral-dot">.</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "Question",
        value=st.session_state.get("last_query", ""),
        height=120,
        placeholder="e.g. What are the latest techniques for sparse retrieval in RAG systems in 2025-2026?",
        label_visibility="visible",
    )

    st.markdown('<div style="margin-top: 4px;"></div>', unsafe_allow_html=True)

    show_trace = st.checkbox("Per-agent trace", value=True)
    show_findings = st.checkbox("Extracted findings", value=True)
    show_errors = st.checkbox("Pipeline warnings", value=True)

    st.markdown('<div style="margin-top: 4px;"></div>', unsafe_allow_html=True)

    run = st.button(
        "▶  Run pipeline",
        type="primary",
        use_container_width=True,
        disabled=not query.strip(),
    )

    st.markdown(
        """
<div style="border-top: 1px solid #1F1F1F; padding-top: 14px; margin-top: 16px;">
  <div style="font-size: 11px; color: #5A5A5A; line-height: 1.7;">
    Queries take 60-180 seconds. The pipeline calls Groq Llama-3.3-70B and Llama-3.1-8B with automatic fallback on rate-limit.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# 7. Run pipeline if button clicked
# ============================================================

if run and query.strip():
    st.session_state["last_query"] = query
    st.session_state.pop("results", None)
    st.session_state["show_all_findings"] = False
    st.session_state["show_all_flagged"] = False

    with st.status("Running multi-agent pipeline...", expanded=True) as status:
        try:
            final_state = asyncio.run(run_with_progress(query, status))
            st.session_state["results"] = final_state
            status.update(
                label="Pipeline complete",
                state="complete",
                expanded=False,
            )
        except Exception as e:  # noqa: BLE001
            status.update(
                label=f"Pipeline failed: {type(e).__name__}",
                state="error",
                expanded=True,
            )
            st.error(f"**Pipeline error:** `{e}`")
            st.stop()


# ============================================================
# 8. Main area — welcome OR results
# ============================================================

results: dict[str, Any] | None = st.session_state.get("results")

if results is None:
    st.markdown(render_welcome_html(), unsafe_allow_html=True)
else:
    report_md = results.get("report_md", "") or ""
    findings = results.get("findings", []) or []
    verification = results.get("verification")
    errors = results.get("errors", []) or []
    user_query = results.get("user_query", "")

    # --- Header row: title on left, download button on right ---
    header_col, dl_col = st.columns([5, 1.2])
    with header_col:
        st.markdown(render_header_html(user_query), unsafe_allow_html=True)
    with dl_col:
        if report_md:
            st.markdown('<div style="margin-top: 6px;"></div>', unsafe_allow_html=True)
            st.download_button(
                "⬇  Download report (.md)",
                data=report_md,
                file_name=f"{slugify(user_query)}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # --- Metrics row: grounding hero + 3 tiles ---
    st.markdown(render_metrics_row_html(results), unsafe_allow_html=True)

    # --- Per-agent trace (collapsible via sidebar checkbox) ---
    if show_trace:
        st.markdown(render_trace_html(results), unsafe_allow_html=True)

    # --- Findings with view-more pattern ---
    if show_findings and findings:
        show_all_findings = st.session_state.get("show_all_findings", False)
        st.markdown(
            render_findings_section_html(findings, show_all_findings),
            unsafe_allow_html=True,
        )
        # View more / Show less button
        if len(findings) > 5:
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                if show_all_findings:
                    if st.button(
                        f"Show less (preview 5 of {len(findings)})",
                        key="findings_less",
                        type="secondary",
                        use_container_width=True,
                    ):
                        st.session_state["show_all_findings"] = False
                        st.rerun()
                else:
                    if st.button(
                        f"View all {len(findings)} findings",
                        key="findings_more",
                        type="secondary",
                        use_container_width=True,
                    ):
                        st.session_state["show_all_findings"] = True
                        st.rerun()

    # --- Report card with secondary download button ---
    if report_md:
        # Report header with inline download (smaller, outlined)
        rep_col1, rep_col2 = st.columns([5, 1])
        with rep_col1:
            st.markdown(
                """<div class="label-uc" style="margin-top: 10px; margin-bottom: 4px;">
<i class="ti ti-file-text" style="font-size: 13px; color: #FF5C39;"></i>Research report
</div>""",
                unsafe_allow_html=True,
            )
        with rep_col2:
            st.download_button(
                "⬇  Download .md",
                data=report_md,
                file_name=f"{slugify(user_query)}.md",
                mime="text/markdown",
                key="dl_in_report",
                use_container_width=True,
            )

        st.markdown('<div class="bento report-body" style="padding: 18px 22px;">', unsafe_allow_html=True)
        st.markdown(report_md, unsafe_allow_html=False)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div style="margin-bottom: 14px;"></div>', unsafe_allow_html=True)

    # --- Verification section ---
    if isinstance(verification, VerificationResult):
        show_all_flagged = st.session_state.get("show_all_flagged", False)
        st.markdown(
            render_verification_section_html(verification, show_all_flagged),
            unsafe_allow_html=True,
        )
        # View more / less for flagged
        if len(verification.flagged) > 3:
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                if show_all_flagged:
                    if st.button(
                        f"Show less (preview 3 of {len(verification.flagged)})",
                        key="flagged_less",
                        type="secondary",
                        use_container_width=True,
                    ):
                        st.session_state["show_all_flagged"] = False
                        st.rerun()
                else:
                    if st.button(
                        f"View all {len(verification.flagged)} flagged claims",
                        key="flagged_more",
                        type="secondary",
                        use_container_width=True,
                    ):
                        st.session_state["show_all_flagged"] = True
                        st.rerun()

    # --- Pipeline warnings (collapsed by default) ---
    if show_errors and errors:
        with st.expander(f"⚠ Pipeline warnings ({len(errors)})", expanded=False):
            for e in errors:
                st.code(e, language=None)

    # --- Footer ---
    st.markdown(
        """
<div style="font-size: 11px; color: #5A5A5A; text-align: center; padding: 24px 0 8px; line-height: 1.6; border-top: 1px solid #1F1F1F; margin-top: 16px;">
  Built with LangGraph · Groq Llama-3.3-70B + Llama-3.1-8B · Tavily · arXiv · GitHub · Streamlit
</div>
""",
        unsafe_allow_html=True,
    )