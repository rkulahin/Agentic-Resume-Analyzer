from __future__ import annotations

import sqlite3
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app import config
from app.models import CandidateProfile, VacancyProfile


def _get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)


def get_chroma_client() -> chromadb.ClientAPI:
    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))


def _candidate_embed_text(c: CandidateProfile) -> str:
    parts = [c.summary, "Skills: " + ", ".join(c.skills)]
    if c.work_history:
        parts.append(c.work_history)
    if c.target_roles:
        parts.append("Target roles: " + ", ".join(c.target_roles))
    return "\n".join(parts)


def _vacancy_embed_text(v: VacancyProfile) -> str:
    parts = [
        v.title,
        v.summary,
        "Required skills: " + ", ".join(v.required_skills),
    ]
    if v.optional_skills:
        parts.append("Optional skills: " + ", ".join(v.optional_skills))
    if v.responsibilities:
        parts.append(v.responsibilities)
    if v.requirements_text:
        parts.append(v.requirements_text)
    return "\n".join(parts)


def index_candidates(profiles: list[CandidateProfile]) -> None:
    client = get_chroma_client()
    ef = _get_embedding_fn()
    col = client.get_or_create_collection(config.COLLECTION_CANDIDATES, embedding_function=ef)

    col.upsert(
        ids=[c.candidate_id for c in profiles],
        documents=[_candidate_embed_text(c) for c in profiles],
        metadatas=[
            {
                "candidate_id": c.candidate_id,
                "location": c.location,
                "years_experience": c.years_experience,
                "seniority_level": c.seniority_level,
                "primary_stack": c.primary_stack,
            }
            for c in profiles
        ],
    )


def index_vacancies(profiles: list[VacancyProfile]) -> None:
    client = get_chroma_client()
    ef = _get_embedding_fn()
    col = client.get_or_create_collection(config.COLLECTION_VACANCIES, embedding_function=ef)

    col.upsert(
        ids=[v.vacancy_id for v in profiles],
        documents=[_vacancy_embed_text(v) for v in profiles],
        metadatas=[
            {
                "vacancy_id": v.vacancy_id,
                "location": v.location,
                "employment_type": v.employment_type,
                "years_experience_required": v.years_experience_required,
                "seniority_level": v.seniority_level,
            }
            for v in profiles
        ],
    )


def index_knowledge_base(docs: list[dict[str, str]]) -> None:
    client = get_chroma_client()
    ef = _get_embedding_fn()
    col = client.get_or_create_collection(config.COLLECTION_KNOWLEDGE, embedding_function=ef)

    col.upsert(
        ids=[d["doc_id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source_file": d["source_file"]} for d in docs],
    )


def init_sqlite() -> None:
    config.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.SQLITE_DB_PATH))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            full_name TEXT,
            location TEXT,
            years_experience REAL,
            seniority_level TEXT,
            primary_stack TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS vacancies (
            vacancy_id TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            location TEXT,
            employment_type TEXT,
            years_experience_required REAL,
            seniority_level TEXT
        )
    """)

    conn.commit()
    conn.close()


def populate_sqlite_candidates(profiles: list[CandidateProfile]) -> None:
    conn = sqlite3.connect(str(config.SQLITE_DB_PATH))
    cur = conn.cursor()
    for c in profiles:
        cur.execute(
            """
            INSERT OR REPLACE INTO candidates
            (candidate_id, full_name, location, years_experience, seniority_level, primary_stack)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (c.candidate_id, c.full_name, c.location, c.years_experience,
             c.seniority_level, c.primary_stack),
        )
    conn.commit()
    conn.close()


def populate_sqlite_vacancies(profiles: list[VacancyProfile]) -> None:
    conn = sqlite3.connect(str(config.SQLITE_DB_PATH))
    cur = conn.cursor()
    for v in profiles:
        cur.execute(
            """
            INSERT OR REPLACE INTO vacancies
            (vacancy_id, title, company, location, employment_type,
             years_experience_required, seniority_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (v.vacancy_id, v.title, v.company, v.location, v.employment_type,
             v.years_experience_required, v.seniority_level),
        )
    conn.commit()
    conn.close()
