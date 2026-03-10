from __future__ import annotations

import json
from pathlib import Path

from app.models import CandidateProfile, VacancyProfile


def load_resumes(directory: Path) -> list[CandidateProfile]:
    profiles: list[CandidateProfile] = []
    for path in sorted(directory.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("source_file", path.name)
        profiles.append(CandidateProfile(**data))
    return profiles


def load_vacancies(directory: Path) -> list[VacancyProfile]:
    profiles: list[VacancyProfile] = []
    for path in sorted(directory.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("source_file", path.name)
        profiles.append(VacancyProfile(**data))
    return profiles


def load_knowledge_base(directory: Path) -> list[dict[str, str]]:
    """Return list of {"doc_id": ..., "text": ..., "source_file": ...}."""
    docs: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({
                "doc_id": path.stem,
                "text": text,
                "source_file": path.name,
            })
    return docs
