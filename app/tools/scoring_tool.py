"""Non-retrieval tool: computes structured match scores between candidates and vacancies."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from app import config
from app.models import CandidateProfile, MatchResult, ScoreBreakdown, VacancyProfile


def _skill_score(candidate_skills: list[str], required: list[str], optional: list[str]) -> float:
    if not required:
        return 1.0
    c_lower = {s.lower() for s in candidate_skills}
    req_lower = {s.lower() for s in required}
    opt_lower = {s.lower() for s in optional}

    req_match = len(c_lower & req_lower) / len(req_lower)
    opt_match = len(c_lower & opt_lower) / len(opt_lower) if opt_lower else 0
    return min(1.0, req_match * 0.85 + opt_match * 0.15)


def _experience_score(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 1.0
    if candidate_years >= required_years:
        return 1.0
    ratio = candidate_years / required_years
    if ratio >= 0.7:
        return ratio
    return 0.0


def _location_score(c_location: str, c_preferred: list[str], v_location: str) -> float:
    c_loc = c_location.lower()
    v_loc = v_location.lower()
    pref = {p.lower() for p in c_preferred}

    if c_loc == v_loc or v_loc == "remote" or c_loc == "remote":
        return 1.0
    if v_loc in pref:
        return 0.8
    if "remote" in pref or "hybrid" in v_loc:
        return 0.5
    return 0.0


_SENIORITY_MAP = {"junior": 0, "middle": 1, "senior": 2, "lead": 3}


def _seniority_score(c_level: str, v_level: str) -> float:
    c_val = _SENIORITY_MAP.get(c_level.lower(), 1)
    v_val = _SENIORITY_MAP.get(v_level.lower(), 1)
    diff = abs(c_val - v_val)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    return 0.0


def compute_match(
    candidate: CandidateProfile,
    vacancy: VacancyProfile,
    semantic_sim: float = 0.0,
) -> MatchResult:
    weights = config.SCORING_WEIGHTS

    sk = _skill_score(candidate.skills, vacancy.required_skills, vacancy.optional_skills)
    ex = _experience_score(candidate.years_experience, vacancy.years_experience_required)
    loc = _location_score(candidate.location, candidate.preferred_locations, vacancy.location)
    sen = _seniority_score(candidate.seniority_level, vacancy.seniority_level)
    sem = min(1.0, max(0.0, semantic_sim))

    final = (
        weights["skill"] * sk
        + weights["semantic"] * sem
        + weights["experience"] * ex
        + weights["location"] * loc
        + weights["seniority"] * sen
    )

    c_lower = {s.lower() for s in candidate.skills}
    req_lower = {s.lower() for s in vacancy.required_skills}
    matched = sorted(c_lower & req_lower)
    missing = sorted(req_lower - c_lower)

    breakdown = ScoreBreakdown(
        skill_score=round(sk, 3),
        semantic_score=round(sem, 3),
        experience_score=round(ex, 3),
        location_score=round(loc, 3),
        seniority_score=round(sen, 3),
    )

    explanation_parts = []
    if loc < 1.0 and candidate.location and vacancy.location:
        explanation_parts.append(f"Location: {candidate.location} (wanted {vacancy.location}).")
    if sen < 1.0 and candidate.seniority_level and vacancy.seniority_level:
        explanation_parts.append(
            f"Seniority: {candidate.seniority_level} (wanted {vacancy.seniority_level})."
        )
    if ex < 1.0:
        explanation_parts.append(
            f"Experience: {candidate.years_experience}y (wanted {vacancy.years_experience_required}y)."
        )
    if not explanation_parts:
        if sk >= 0.8:
            explanation_parts.append("Strong skill match.")
        elif sk >= 0.5:
            explanation_parts.append("Partial skill match.")
        else:
            explanation_parts.append("Low skill overlap.")

    return MatchResult(
        entity_id=candidate.candidate_id,
        entity_type="candidate",
        full_name=candidate.full_name,
        source_file=candidate.source_file,
        final_score=round(final, 3),
        score_breakdown=breakdown,
        matched_skills=matched,
        missing_skills=missing,
        short_explanation=" ".join(explanation_parts),
    )


@tool
def score_candidate_vacancy(candidate_json: str, vacancy_json: str) -> str:
    """Score how well a candidate matches a vacancy.

    Args:
        candidate_json: JSON string of candidate profile.
        vacancy_json: JSON string of vacancy profile.

    Returns:
        JSON string with match score and breakdown.
    """
    candidate = CandidateProfile(**json.loads(candidate_json))
    vacancy = VacancyProfile(**json.loads(vacancy_json))
    result = compute_match(candidate, vacancy)
    return result.model_dump_json(indent=2)
