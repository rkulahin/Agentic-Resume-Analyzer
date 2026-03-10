"""Main LangGraph workflow with 5+ nodes, conditional routing, and RAG subgraph."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from app import config
from app.graph.rag_subgraph import build_rag_subgraph
from app.graph.state import AgentState
from app.ingestion.indexer import get_chroma_client, _get_embedding_fn
from app.models import CandidateProfile, VacancyProfile
from app.tools.scoring_tool import compute_match


logger = logging.getLogger(__name__)

_CLASSIFY_AND_EXTRACT_PROMPT = """\
You are an expert classifier and information extractor for a recruiting / job-search system.
Analyze the user input and return ONLY a valid JSON object with these fields:

- "task_type": either "candidate_search" or "vacancy_search".
  Use "candidate_search" when the user is a RECRUITER looking for candidates/employees \
(e.g. "looking for a developer", "need an engineer", "find me a QA", "hire a backend dev").
  Use "vacancy_search" when the user is a JOB SEEKER looking for jobs/vacancies \
(e.g. "I'm looking for a job", "my resume", "my experience is", "open to new roles").
  If the input IS a resume/CV of a person → "vacancy_search".
  If the input IS a job posting or describes a desired employee → "candidate_search".
- "skills": list of technical skills (canonical English lowercase, e.g. "python", "react"). Translate from any language.
- "location": city name in English (e.g. "Lviv", "Kyiv", "Remote") or "".
- "seniority_level": one of "Junior", "Middle", "Senior", "Lead", or "".
- "years_experience": numeric years as float, or 0.
- "role": job role / title in English (e.g. "Python Developer") or "".

Rules:
- Translate everything to English.
- If a field is not mentioned, use empty default ("", [], or 0).
- Return ONLY valid JSON. No markdown, no explanation.

Example:
Input: "Looking for a Python developer in Lviv office"
Output: {"task_type": "candidate_search", "skills": ["python"], "location": "Lviv", "seniority_level": "", "years_experience": 0, "role": "Python Developer"}

Input: "Need a senior React developer in Kyiv, 5+ years"
Output: {"task_type": "candidate_search", "skills": ["react"], "location": "Kyiv", "seniority_level": "Senior", "years_experience": 5, "role": "React Developer"}

Input: "I am a QA engineer with 3 years of experience, looking for a job in Lviv"
Output: {"task_type": "vacancy_search", "skills": ["qa"], "location": "Lviv", "seniority_level": "", "years_experience": 3, "role": "QA Engineer"}\
"""

_EMPTY_CRITERIA: dict = {
    "task_type": "candidate_search",
    "skills": [],
    "location": "",
    "seniority_level": "",
    "years_experience": 0,
    "role": "",
}


def _classify_and_extract_with_llm(text: str, llm: ChatOllama) -> dict:
    """Single LLM call to classify intent AND extract structured criteria."""
    try:
        response = llm.invoke([
            SystemMessage(content=_CLASSIFY_AND_EXTRACT_PROMPT),
            HumanMessage(content=text),
        ])
        raw = response.content.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("LLM extraction returned no JSON: %s", raw[:200])
            return {**_EMPTY_CRITERIA}
        parsed = json.loads(raw[start:end])

        task_type = parsed.get("task_type", "").strip().lower()
        if task_type not in ("candidate_search", "vacancy_search"):
            task_type = "candidate_search"

        return {
            "task_type": task_type,
            "skills": [s.lower().strip() for s in parsed.get("skills", []) if s],
            "location": parsed.get("location", "").strip(),
            "seniority_level": parsed.get("seniority_level", "").strip(),
            "years_experience": float(parsed.get("years_experience", 0)),
            "role": parsed.get("role", "").strip(),
        }
    except Exception:
        logger.exception("LLM classify+extract failed")
        return {**_EMPTY_CRITERIA}


def _get_llm(num_predict: int = 512) -> ChatOllama:
    return ChatOllama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL,
        temperature=0,
        num_predict=num_predict,
    )


def _load_profiles_by_ids(
    ids: list[str], task_type: str
) -> list[CandidateProfile] | list[VacancyProfile]:
    """Load full profiles from raw JSON files by IDs."""
    if task_type == "candidate_search":
        directory = config.RAW_RESUMES_DIR
        model_cls = CandidateProfile
    else:
        directory = config.RAW_VACANCIES_DIR
        model_cls = VacancyProfile

    profiles = []
    for path in directory.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        id_field = "candidate_id" if task_type == "candidate_search" else "vacancy_id"
        if data.get(id_field) in ids:
            data.setdefault("source_file", path.name)
            profiles.append(model_cls(**data))
    return profiles


# ── Node 1: Intake ──


def intake_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "").strip()
    if not user_input:
        return {"error": "Empty input provided."}
    return {"user_input": user_input}


# ── Node 2: Classify + Extract (single LLM call) ──


_CANDIDATE_PATTERNS = [
    "looking for", "hire", "recruit", "find me",
    "need a", "search for", "find a", "find candidate",
    "responsibilities", "we offer", "required skills",
    "developer", "engineer", "analyst", "tester",
    "we need", "we are looking", "seeking",
]
_VACANCY_PATTERNS = [
    "vacancy", "job", "position", "apply",
    "my resume", "my experience", "my cv",
    "looking for a job", "career", "job search",
    "open to", "available for", "hire me",
    "work experience", "education", "summary",
    "years of experience",
]


def _classify_with_keywords(text: str) -> str:
    """Fast keyword-based classification fallback."""
    text_lower = text.lower()
    candidate_score = sum(1 for kw in _CANDIDATE_PATTERNS if kw in text_lower)
    vacancy_score = sum(1 for kw in _VACANCY_PATTERNS if kw in text_lower)
    if vacancy_score > candidate_score:
        return "vacancy_search"
    return "candidate_search"


def classify_node(state: AgentState) -> dict:
    """Classify intent AND extract criteria in a single LLM call."""
    user_input = state.get("user_input", "")

    try:
        llm = _get_llm(num_predict=200)
        result = _classify_and_extract_with_llm(user_input, llm)
    except Exception:
        logger.exception("classify_node LLM call failed, using keyword fallback")
        result = {**_EMPTY_CRITERIA}

    task_type = result.pop("task_type", None)
    if not task_type:
        task_type = _classify_with_keywords(user_input)

    return {"task_type": task_type, "parsed_criteria": result}


# ── Node 3: Route decision (conditional) ──


def route_decision(state: AgentState) -> str:
    return state.get("task_type", "candidate_search")


# ── Node 4a: Candidate search ──


def _build_query_text(user_input: str, criteria: dict) -> str:
    """Build enriched query text from user input and extracted criteria."""
    parts = [user_input]
    if criteria.get("skills"):
        parts.append(" ".join(criteria["skills"]))
    if criteria.get("role"):
        parts.append(criteria["role"])
    return " ".join(parts)


def candidate_search_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    criteria = state.get("parsed_criteria", {})

    criteria["query_text"] = _build_query_text(user_input, criteria)
    criteria["extracted_skills"] = criteria.get("skills", [])
    return {"parsed_criteria": criteria}


# ── Node 4b: Vacancy search ──


def vacancy_search_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    criteria = state.get("parsed_criteria", {})

    criteria["query_text"] = _build_query_text(user_input, criteria)
    criteria["extracted_skills"] = criteria.get("skills", [])
    return {
        "task_type": "vacancy_search",
        "parsed_criteria": criteria,
    }


# ── Node 5: Scoring ──


def scoring_node(state: AgentState) -> dict:
    task_type = state.get("task_type", "candidate_search")
    retrieved_ids = state.get("retrieved_ids", [])
    retrieved_context = state.get("retrieved_context", [])

    knowledge_ids = {
        "resume_best_practices", "react_developer_guide",
        "python_backend_guide", "data_analyst_guide", "qa_engineer_guide",
        "skill_gap_analysis", "interview_preparation", "career_growth_tips",
    }
    entity_ids = [rid for rid in retrieved_ids if rid not in knowledge_ids]

    if not entity_ids:
        return {"match_results": [], "career_advice": "\n".join(retrieved_context[-3:])}

    profiles = _load_profiles_by_ids(entity_ids, task_type)

    if task_type == "candidate_search":
        criteria = state.get("parsed_criteria", {})
        skills = criteria.get("extracted_skills", [])
        dummy_vacancy = VacancyProfile(
            vacancy_id="query",
            title=criteria.get("role", "Search query") or "Search query",
            required_skills=skills,
            location=criteria.get("location", ""),
            seniority_level=criteria.get("seniority_level", ""),
            years_experience_required=criteria.get("years_experience", 0),
        )

        chroma_client = get_chroma_client()
        ef = _get_embedding_fn()
        col = chroma_client.get_or_create_collection(
            config.COLLECTION_CANDIDATES, embedding_function=ef
        )
        query_text = criteria.get("query_text", "")
        sim_results = col.query(
            query_texts=[query_text], n_results=10, include=["distances"]
        )
        distance_map = {}
        if sim_results["ids"] and sim_results["distances"]:
            for rid, dist in zip(sim_results["ids"][0], sim_results["distances"][0]):
                distance_map[rid] = max(0.0, 1.0 - dist)

        results = []
        for profile in profiles:
            sem_score = distance_map.get(profile.candidate_id, 0.0)
            match = compute_match(profile, dummy_vacancy, semantic_sim=sem_score)
            results.append(match)
    else:
        criteria = state.get("parsed_criteria", {})
        skills = criteria.get("extracted_skills", [])
        dummy_candidate = CandidateProfile(
            candidate_id="query",
            full_name="Query user",
            skills=skills,
            location=criteria.get("location", ""),
            seniority_level=criteria.get("seniority_level", ""),
            years_experience=criteria.get("years_experience", 0),
        )

        chroma_client = get_chroma_client()
        ef = _get_embedding_fn()
        col = chroma_client.get_or_create_collection(
            config.COLLECTION_VACANCIES, embedding_function=ef
        )
        query_text = criteria.get("query_text", "")
        sim_results = col.query(
            query_texts=[query_text], n_results=10, include=["distances"]
        )
        distance_map = {}
        if sim_results["ids"] and sim_results["distances"]:
            for rid, dist in zip(sim_results["ids"][0], sim_results["distances"][0]):
                distance_map[rid] = max(0.0, 1.0 - dist)

        results = []
        for profile in profiles:
            sem_score = distance_map.get(profile.vacancy_id, 0.0)
            match = compute_match(dummy_candidate, profile, semantic_sim=sem_score)
            match.entity_id = profile.vacancy_id
            match.entity_type = "vacancy"
            match.full_name = profile.title
            match.source_file = profile.source_file
            results.append(match)

    results.sort(key=lambda r: r.final_score, reverse=True)

    knowledge_context = [
        ctx for rid, ctx in zip(retrieved_ids, retrieved_context)
        if rid in knowledge_ids
    ]

    return {
        "match_results": results[: config.TOP_K_RESULTS],
        "career_advice": "\n".join(knowledge_context),
    }


# ── Node 6: Recommendation ──


def recommendation_node(state: AgentState) -> dict:
    match_results = state.get("match_results", [])
    task_type = state.get("task_type", "candidate_search")
    career_advice = state.get("career_advice", "")

    if not match_results:
        return {
            "recommendations": "No matches found for your query. "
            "Try broadening your search criteria."
        }

    try:
        llm = _get_llm(num_predict=300)

        results_text = ""
        for i, m in enumerate(match_results, 1):
            results_text += (
                f"\n{i}. ID: {m.entity_id}, Score: {m.final_score}\n"
                f"   Matched skills: {', '.join(m.matched_skills)}\n"
                f"   Missing skills: {', '.join(m.missing_skills)}\n"
                f"   {m.short_explanation}\n"
            )

        if task_type == "candidate_search":
            prompt = (
                f"You are a recruiting assistant. Based on these candidate matches:\n"
                f"{results_text}\n"
                f"User query: {state.get('user_input', '')}\n\n"
                f"Provide a brief recommendation summary explaining why the top candidates "
                f"are a good fit and what to consider."
            )
        else:
            prompt = (
                f"You are a career advisor. Based on these vacancy matches:\n"
                f"{results_text}\n"
                f"User input: {state.get('user_input', '')}\n\n"
                f"Relevant career advice:\n{career_advice}\n\n"
                f"Provide a brief recommendation explaining which positions are "
                f"the best fit and actionable advice to improve chances."
            )

        response = llm.invoke([
            SystemMessage(content="You are a helpful career/recruiting assistant. "
                          "Give concise, actionable advice."),
            HumanMessage(content=prompt),
        ])
        return {"recommendations": response.content}

    except Exception:
        parts = []
        for i, m in enumerate(match_results, 1):
            parts.append(
                f"{i}. {m.entity_id} (score: {m.final_score}) — {m.short_explanation}"
            )
        fallback = "\n".join(parts)
        if career_advice:
            fallback += f"\n\nCareer advice:\n{career_advice[:500]}"
        return {"recommendations": fallback}


# ── Node 7: Response ──


def response_node(state: AgentState) -> dict:
    task_type = state.get("task_type", "candidate_search")
    match_results = state.get("match_results", [])
    recommendations = state.get("recommendations", "")

    if state.get("error"):
        return {"final_response": f"Error: {state['error']}"}

    header = (
        "## Candidate Search Results\n"
        if task_type == "candidate_search"
        else "## Vacancy Search Results\n"
    )

    parts = [header]

    for i, m in enumerate(match_results, 1):
        title = m.full_name or m.entity_id
        parts.append(
            f"### {i}. {title} ({m.entity_id})\n"
            f"- **Score:** {m.final_score}\n"
            f"- **Skill:** {m.score_breakdown.skill_score} | "
            f"**Semantic:** {m.score_breakdown.semantic_score} | "
            f"**Experience:** {m.score_breakdown.experience_score} | "
            f"**Location:** {m.score_breakdown.location_score} | "
            f"**Seniority:** {m.score_breakdown.seniority_score}\n"
            f"- **Matched:** {', '.join(m.matched_skills) or 'none'}\n"
            f"- **Missing:** {', '.join(m.missing_skills) or 'none'}\n"
            f"- {m.short_explanation}\n"
        )

    if recommendations:
        parts.append(f"\n### Recommendations\n{recommendations}")

    return {"final_response": "\n".join(parts)}


# ── Build graph ──


def build_main_graph() -> StateGraph:
    rag = build_rag_subgraph()

    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("classify", classify_node)
    graph.add_node("candidate_search", candidate_search_node)
    graph.add_node("vacancy_search", vacancy_search_node)
    graph.add_node("rag", rag)
    graph.add_node("scoring", scoring_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "classify")
    graph.add_conditional_edges(
        "classify",
        route_decision,
        {
            "candidate_search": "candidate_search",
            "vacancy_search": "vacancy_search",
        },
    )
    graph.add_edge("candidate_search", "rag")
    graph.add_edge("vacancy_search", "rag")
    graph.add_edge("rag", "scoring")
    graph.add_edge("scoring", "recommendation")
    graph.add_edge("recommendation", "response")
    graph.add_edge("response", END)

    return graph.compile()
