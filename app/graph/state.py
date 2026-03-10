from __future__ import annotations

from typing import TypedDict

from app.models import MatchResult


class AgentState(TypedDict, total=False):
    # User input
    user_input: str
    task_type: str  # "candidate_search" | "vacancy_search"

    # Parsed criteria from recruiter or resume
    parsed_criteria: dict

    # RAG retrieval results
    retrieved_context: list[str]
    retrieved_ids: list[str]

    # Scoring results
    match_results: list[MatchResult]

    # Recommendations
    recommendations: str
    career_advice: str

    # Final output
    final_response: str

    # Metadata
    error: str
