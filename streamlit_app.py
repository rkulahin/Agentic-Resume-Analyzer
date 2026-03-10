"""Streamlit UI for Agentic Resume Analyzer."""

from __future__ import annotations

import json
import re

import streamlit as st

from app import config
from app.graph.main_graph import build_main_graph
from app.utils.text_extraction import (
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_url,
)

st.set_page_config(page_title="Agentic Resume Analyzer", page_icon="📄", layout="wide")

DEMO_QUERIES = [
    "Looking for a Senior Python developer with FastAPI and PostgreSQL experience, Kyiv or Remote",
    "Need a Middle React developer, 3+ years, Lviv",
    "Looking for a Junior Data Analyst with SQL and Python, Kyiv",
    "Find me a QA engineer with Selenium and Cypress, Remote",
    "My resume: 4 years of Python backend, FastAPI, Django, PostgreSQL, Docker. "
    "Looking for Senior Python positions, Remote or Kyiv.",
    "I'm a React developer with 2 years of experience, TypeScript, Redux, Next.js. "
    "Looking for a job in Lviv.",
    "Junior Data Analyst, 1 year experience with SQL, Excel, Power BI. "
    "Looking for my first full-time position in Kyiv.",
]

_URL_PATTERN = re.compile(r"https?://[^\s]+")


def _load_resume_json(source_file: str) -> dict | None:
    """Load a raw resume JSON by source_file name."""
    path = config.RAW_RESUMES_DIR / source_file
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _expand_urls_in_text(text: str) -> str:
    """Find URLs in text and replace them with fetched content."""
    urls = _URL_PATTERN.findall(text)
    if not urls:
        return text

    result = text
    for url in urls:
        try:
            fetched = extract_text_from_url(url)
            if fetched:
                result = result.replace(url, f"\n\n[Content from {url}]:\n{fetched}\n")
        except Exception:
            pass
    return result


def _render_candidate_cards(match_results: list, recommendations: str) -> None:
    """Render candidate results as structured cards with download buttons."""
    st.subheader("Candidate Search Results")

    for i, m in enumerate(match_results, 1):
        title = m.full_name or m.entity_id
        with st.container():
            st.markdown(f"### {i}. {title} (`{m.entity_id}`)")

            col_score, col_download = st.columns([3, 1])

            with col_score:
                st.markdown(
                    f"- **Score:** {m.final_score}\n"
                    f"- **Skill:** {m.score_breakdown.skill_score} | "
                    f"**Semantic:** {m.score_breakdown.semantic_score} | "
                    f"**Experience:** {m.score_breakdown.experience_score} | "
                    f"**Location:** {m.score_breakdown.location_score} | "
                    f"**Seniority:** {m.score_breakdown.seniority_score}\n"
                    f"- **Matched:** {', '.join(m.matched_skills) or 'none'}\n"
                    f"- **Missing:** {', '.join(m.missing_skills) or 'none'}\n"
                    f"- {m.short_explanation}"
                )

            with col_download:
                if m.source_file:
                    resume_data = _load_resume_json(m.source_file)
                    if resume_data:
                        st.download_button(
                            label="Download Resume",
                            data=json.dumps(resume_data, indent=2, ensure_ascii=False),
                            file_name=m.source_file,
                            mime="application/json",
                            key=f"download_{m.entity_id}",
                        )

            st.markdown("---")

    if recommendations:
        st.markdown(f"### Recommendations\n{recommendations}")


def _render_vacancy_results(final_response: str) -> None:
    """Render vacancy search results as markdown."""
    st.subheader("Results")
    st.markdown(final_response)


def _display_results(result: dict) -> None:
    """Display agent results from session state."""
    col_flow, col_results = st.columns([1, 2])

    with col_flow:
        st.subheader("Agent Flow")

        task_type = result.get("task_type", "unknown")
        st.markdown(f"**Task type:** `{task_type}`")

        criteria = result.get("parsed_criteria", {})
        if criteria:
            skills = criteria.get("extracted_skills", [])
            st.markdown(f"**Extracted skills:** {', '.join(skills) if skills else 'none'}")
            if criteria.get("location"):
                st.markdown(f"**Location:** {criteria['location']}")
            if criteria.get("seniority_level"):
                st.markdown(f"**Seniority:** {criteria['seniority_level']}")
            if criteria.get("role"):
                st.markdown(f"**Role:** {criteria['role']}")
            if criteria.get("years_experience"):
                st.markdown(f"**Experience:** {criteria['years_experience']} years")

        retrieved = result.get("retrieved_ids", [])
        st.markdown(f"**Retrieved entities:** {len(retrieved)}")

        match_results = result.get("match_results", [])
        st.markdown(f"**Scored matches:** {len(match_results)}")

    with col_results:
        if result.get("error"):
            st.error(result["error"])
        elif task_type == "candidate_search":
            _render_candidate_cards(
                result.get("match_results", []),
                result.get("recommendations", ""),
            )
        else:
            _render_vacancy_results(
                result.get("final_response", "No response generated.")
            )

    with st.expander("Raw agent state (debug)"):
        debug_state = {}
        for k, v in result.items():
            if k == "match_results":
                debug_state[k] = [
                    m.model_dump() if hasattr(m, "model_dump") else m for m in v
                ]
            else:
                debug_state[k] = v
        st.json(debug_state)


def main() -> None:
    st.title("Agentic Resume Analyzer")
    st.markdown(
        "An agentic RAG chatbot that matches **candidates** with **vacancies** "
        "using LangGraph, Chroma, and structured scoring. "
        "The bot automatically detects whether you are searching for candidates or vacancies."
    )

    st.markdown("---")

    demo = st.selectbox(
        "Try a demo query:",
        ["(type your own)"] + DEMO_QUERIES,
    )
    default_text = demo if demo != "(type your own)" else ""
    user_input = st.text_area(
        "Describe the vacancy, paste a resume, or enter a search query. "
        "You can also paste a URL — its content will be fetched automatically.",
        value=default_text,
        height=120,
    )

    uploaded_file = st.file_uploader(
        "Or attach a file (PDF / TXT):",
        type=["pdf", "txt"],
    )

    if st.button("Analyze", type="primary", use_container_width=True):
        text_input = user_input.strip()

        if _URL_PATTERN.search(text_input):
            with st.spinner("Fetching content from URL..."):
                try:
                    text_input = _expand_urls_in_text(text_input)
                except Exception as e:
                    st.error(f"Failed to fetch URL: {e}")
                    return

        attachment_text = ""
        if uploaded_file is not None:
            with st.spinner("Extracting text from file..."):
                try:
                    raw_bytes = uploaded_file.read()
                    if uploaded_file.name.lower().endswith(".pdf"):
                        attachment_text = extract_text_from_pdf(raw_bytes)
                    else:
                        attachment_text = extract_text_from_txt(raw_bytes)
                except Exception as e:
                    st.error(f"Failed to extract text from file: {e}")
                    return

        parts = [p for p in [text_input, attachment_text] if p]
        final_input = "\n\n".join(parts)

        if not final_input:
            st.warning("Please enter a query or attach a file.")
            return

        with st.spinner("Running agent workflow..."):
            graph = build_main_graph()
            result = graph.invoke({"user_input": final_input})

        st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        _display_results(st.session_state["last_result"])


if __name__ == "__main__":
    main()
