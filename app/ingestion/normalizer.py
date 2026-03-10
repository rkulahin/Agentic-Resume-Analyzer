from __future__ import annotations

import json
from pathlib import Path

from app.models import CandidateProfile, VacancyProfile


def save_jsonl(items: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")


def normalize_resumes(profiles: list[CandidateProfile], output_path: Path) -> None:
    save_jsonl(profiles, output_path)


def normalize_vacancies(profiles: list[VacancyProfile], output_path: Path) -> None:
    save_jsonl(profiles, output_path)
