from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- LLM ---
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

# --- Embeddings ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# --- Chroma ---
CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", BASE_DIR / "data" / "chroma_db"))

COLLECTION_CANDIDATES = "candidate_profiles"
COLLECTION_VACANCIES = "vacancy_profiles"
COLLECTION_KNOWLEDGE = "career_knowledge"

# --- SQLite ---
SQLITE_DB_PATH: Path = Path(os.getenv("SQLITE_DB_PATH", BASE_DIR / "data" / "metadata.db"))

# --- Data paths ---
RAW_RESUMES_DIR: Path = Path(os.getenv("RAW_RESUMES_DIR", BASE_DIR / "data" / "raw" / "resumes"))
RAW_VACANCIES_DIR: Path = Path(
    os.getenv("RAW_VACANCIES_DIR", BASE_DIR / "data" / "raw" / "vacancies")
)
RAW_KNOWLEDGE_DIR: Path = Path(
    os.getenv("RAW_KNOWLEDGE_DIR", BASE_DIR / "data" / "raw" / "knowledge_base")
)
PROCESSED_DIR: Path = Path(os.getenv("PROCESSED_DIR", BASE_DIR / "data" / "processed"))

# --- Scoring weights ---
SCORING_WEIGHTS = {
    "skill": 0.35,
    "semantic": 0.25,
    "experience": 0.20,
    "location": 0.10,
    "seniority": 0.10,
}

TOP_K_RESULTS = 3
