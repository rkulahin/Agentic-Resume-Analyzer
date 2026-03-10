"""Full ingestion pipeline: load raw data -> normalize -> index into Chroma + SQLite."""

from __future__ import annotations

from app import config
from app.ingestion.indexer import (
    index_candidates,
    index_knowledge_base,
    index_vacancies,
    init_sqlite,
    populate_sqlite_candidates,
    populate_sqlite_vacancies,
)
from app.ingestion.loader import load_knowledge_base, load_resumes, load_vacancies
from app.ingestion.normalizer import normalize_resumes, normalize_vacancies


def run_ingestion() -> None:
    print("Loading raw data...")
    resumes = load_resumes(config.RAW_RESUMES_DIR)
    vacancies = load_vacancies(config.RAW_VACANCIES_DIR)
    knowledge_docs = load_knowledge_base(config.RAW_KNOWLEDGE_DIR)

    print(f"  Loaded {len(resumes)} resumes, {len(vacancies)} vacancies, "
          f"{len(knowledge_docs)} knowledge docs")

    print("Normalizing...")
    normalize_resumes(resumes, config.PROCESSED_DIR / "resumes.jsonl")
    normalize_vacancies(vacancies, config.PROCESSED_DIR / "vacancies.jsonl")

    print("Initializing SQLite...")
    init_sqlite()
    populate_sqlite_candidates(resumes)
    populate_sqlite_vacancies(vacancies)

    print("Indexing into Chroma...")
    index_candidates(resumes)
    index_vacancies(vacancies)
    index_knowledge_base(knowledge_docs)

    print("Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()
