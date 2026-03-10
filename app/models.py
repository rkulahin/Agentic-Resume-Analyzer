from __future__ import annotations

from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    candidate_id: str
    full_name: str
    location: str = ""
    preferred_locations: list[str] = Field(default_factory=list)
    years_experience: float = 0.0
    seniority_level: str = ""
    skills: list[str] = Field(default_factory=list)
    primary_stack: str = ""
    languages: list[str] = Field(default_factory=list)
    education: str = ""
    work_history: str = ""
    target_roles: list[str] = Field(default_factory=list)
    summary: str = ""
    raw_text: str = ""
    source_file: str = ""


class VacancyProfile(BaseModel):
    vacancy_id: str
    title: str
    company: str = ""
    location: str = ""
    employment_type: str = ""
    years_experience_required: float = 0.0
    seniority_level: str = ""
    required_skills: list[str] = Field(default_factory=list)
    optional_skills: list[str] = Field(default_factory=list)
    responsibilities: str = ""
    requirements_text: str = ""
    benefits: str = ""
    summary: str = ""
    raw_text: str = ""
    source_file: str = ""


class ScoreBreakdown(BaseModel):
    skill_score: float = 0.0
    semantic_score: float = 0.0
    experience_score: float = 0.0
    location_score: float = 0.0
    seniority_score: float = 0.0


class MatchResult(BaseModel):
    entity_id: str
    entity_type: str  # "candidate" | "vacancy"
    full_name: str = ""
    source_file: str = ""
    final_score: float = 0.0
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    short_explanation: str = ""
