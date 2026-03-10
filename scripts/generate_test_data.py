from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_RESUMES_DIR = BASE_DIR / "data" / "raw" / "resumes"
RAW_VACANCIES_DIR = BASE_DIR / "data" / "raw" / "vacancies"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@dataclass(frozen=True)
class RoleTemplate:
    """Configuration for one synthetic role family."""

    role_key: str
    candidate_title: str
    vacancy_title: str
    primary_stack: str
    education_focus: str
    core_skills: tuple[str, ...]
    extra_skills: tuple[str, ...]
    optional_skills: tuple[str, ...]
    candidate_roles: tuple[str, ...]
    vacancy_titles: tuple[str, ...]
    work_focus: str
    responsibilities: tuple[str, ...]


ROLE_TEMPLATES: tuple[RoleTemplate, ...] = (
    RoleTemplate(
        role_key="react",
        candidate_title="React Developer",
        vacancy_title="React Engineer",
        primary_stack="React",
        education_focus="software engineering",
        core_skills=("react", "javascript", "typescript", "html", "css"),
        extra_skills=("redux", "next.js", "graphql", "jest", "cypress", "tailwind", "vite"),
        optional_skills=("storybook", "figma", "webpack", "accessibility", "performance optimization"),
        candidate_roles=("Frontend Developer", "React Developer", "UI Engineer"),
        vacancy_titles=("React Developer", "Frontend Engineer", "Next.js Developer"),
        work_focus="building responsive product interfaces and reusable UI components",
        responsibilities=(
            "build reusable frontend components",
            "integrate REST and GraphQL APIs",
            "improve page performance and accessibility",
        ),
    ),
    RoleTemplate(
        role_key="python",
        candidate_title="Python Backend Developer",
        vacancy_title="Python Backend Engineer",
        primary_stack="Python",
        education_focus="computer science",
        core_skills=("python", "fastapi", "django", "postgresql", "docker"),
        extra_skills=("aws", "redis", "celery", "sqlalchemy", "pytest", "flask", "kubernetes"),
        optional_skills=("rabbitmq", "terraform", "prometheus", "microservices", "linux"),
        candidate_roles=("Python Developer", "Backend Engineer", "API Developer"),
        vacancy_titles=("Python Backend Engineer", "FastAPI Developer", "Backend Python Engineer"),
        work_focus="developing backend services, APIs, and data-driven platform features",
        responsibilities=(
            "design and maintain backend APIs",
            "optimize database queries and service performance",
            "ship reliable services with tests and monitoring",
        ),
    ),
    RoleTemplate(
        role_key="data",
        candidate_title="Data Analyst",
        vacancy_title="Data Analyst",
        primary_stack="Analytics",
        education_focus="analytics, economics, or applied mathematics",
        core_skills=("sql", "python", "excel", "tableau"),
        extra_skills=("power bi", "pandas", "statistics", "airflow", "looker", "bigquery", "a/b testing"),
        optional_skills=("dbt", "forecasting", "product analytics", "ga4", "data storytelling"),
        candidate_roles=("Data Analyst", "BI Analyst", "Product Analyst"),
        vacancy_titles=("Data Analyst", "BI Analyst", "Product Data Analyst"),
        work_focus="analyzing business metrics, building dashboards, and explaining trends",
        responsibilities=(
            "prepare datasets and business reports",
            "build dashboards for product and operations teams",
            "translate analytical findings into action items",
        ),
    ),
    RoleTemplate(
        role_key="qa",
        candidate_title="QA Engineer",
        vacancy_title="QA Engineer",
        primary_stack="QA Automation",
        education_focus="computer science or quality engineering",
        core_skills=("testing", "jira", "sql", "postman"),
        extra_skills=("selenium", "cypress", "api testing", "playwright", "python", "java", "test automation"),
        optional_skills=("performance testing", "mobile testing", "ci/cd", "bug reporting", "test planning"),
        candidate_roles=("QA Engineer", "Automation QA", "Test Engineer"),
        vacancy_titles=("QA Engineer", "Automation QA Engineer", "Software Test Engineer"),
        work_focus="validating product quality through manual and automated testing",
        responsibilities=(
            "create and maintain test cases",
            "automate regression and smoke checks",
            "document defects and collaborate with engineering teams",
        ),
    ),
    RoleTemplate(
        role_key="marketing",
        candidate_title="Marketing Specialist",
        vacancy_title="Marketing Specialist",
        primary_stack="Marketing",
        education_focus="marketing, communications, or business",
        core_skills=("digital marketing", "content strategy", "seo", "google analytics", "campaign management"),
        extra_skills=("email marketing", "smm", "copywriting", "hubspot", "meta ads", "google ads", "crm"),
        optional_skills=("brand strategy", "market research", "canva", "landing pages", "a/b testing"),
        candidate_roles=("Marketing Specialist", "Growth Marketer", "Digital Marketing Manager"),
        vacancy_titles=("Marketing Specialist", "Digital Marketer", "Growth Marketing Specialist"),
        work_focus="planning and executing acquisition, retention, and content campaigns",
        responsibilities=(
            "launch and optimize multichannel campaigns",
            "analyze funnel metrics and campaign performance",
            "coordinate content, design, and sales stakeholders",
        ),
    ),
    RoleTemplate(
        role_key="csharp",
        candidate_title="C# .NET Developer",
        vacancy_title="C# .NET Engineer",
        primary_stack=".NET",
        education_focus="software engineering or computer science",
        core_skills=("c#", ".net", "asp.net core", "sql server", "entity framework"),
        extra_skills=("azure", "docker", "microservices", "xunit", "rest api", "redis", "blazor"),
        optional_skills=("kubernetes", "message queues", "clean architecture", "ci/cd", "gRPC"),
        candidate_roles=("C# Developer", ".NET Engineer", "Backend .NET Developer"),
        vacancy_titles=("C# Developer", ".NET Backend Engineer", "ASP.NET Core Developer"),
        work_focus="building enterprise backend systems and internal business applications",
        responsibilities=(
            "develop and maintain .NET services",
            "integrate databases and internal systems",
            "improve code quality, observability, and deployment flow",
        ),
    ),
    RoleTemplate(
        role_key="audit",
        candidate_title="Audit Specialist",
        vacancy_title="Audit Specialist",
        primary_stack="Audit",
        education_focus="accounting, finance, or auditing",
        core_skills=("audit", "excel", "financial reporting", "internal controls", "risk assessment"),
        extra_skills=("ifrs", "gaap", "sap", "documentation", "process review", "compliance", "data analysis"),
        optional_skills=("power bi", "sox", "erp", "stakeholder management", "policy review"),
        candidate_roles=("Audit Specialist", "Internal Auditor", "Audit Associate"),
        vacancy_titles=("Audit Specialist", "Internal Auditor", "Audit Associate"),
        work_focus="reviewing financial processes, controls, and compliance activities",
        responsibilities=(
            "perform audit procedures and control testing",
            "document findings and prepare audit reports",
            "recommend improvements for compliance and risk mitigation",
        ),
    ),
    RoleTemplate(
        role_key="tax",
        candidate_title="Tax Consultant",
        vacancy_title="Tax Consultant",
        primary_stack="Tax Consulting",
        education_focus="tax, finance, accounting, or law",
        core_skills=("tax compliance", "tax planning", "excel", "accounting", "tax legislation"),
        extra_skills=("vat", "corporate tax", "transfer pricing", "reporting", "risk analysis", "advisory", "1c"),
        optional_skills=("ifrs", "erp", "policy drafting", "client communication", "financial modeling"),
        candidate_roles=("Tax Consultant", "Tax Advisor", "Tax Associate"),
        vacancy_titles=("Tax Consultant", "Tax Advisor", "Tax Compliance Specialist"),
        work_focus="supporting businesses with tax compliance, planning, and advisory work",
        responsibilities=(
            "prepare tax calculations and compliance documents",
            "analyze legislation changes and client impact",
            "advise stakeholders on tax risks and optimization opportunities",
        ),
    ),
)

FIRST_NAMES = (
    "Olena",
    "Andrii",
    "Iryna",
    "Maksym",
    "Sofiia",
    "Dmytro",
    "Kateryna",
    "Yaroslav",
    "Anastasiia",
    "Roman",
    "Nataliia",
    "Bohdan",
    "Tetiana",
    "Viktor",
    "Alina",
    "Denys",
    "Marta",
    "Artem",
    "Yuliia",
    "Taras",
)

LAST_NAMES = (
    "Shevchenko",
    "Koval",
    "Melnyk",
    "Tkachenko",
    "Bondarenko",
    "Kravets",
    "Polishchuk",
    "Savchenko",
    "Mazur",
    "Hrytsenko",
    "Oliinyk",
    "Marchenko",
    "Kushnir",
    "Pavlenko",
    "Sydorenko",
    "Rudenko",
    "Lysenko",
    "Tymoshenko",
    "Korniienko",
    "Shapoval",
)

COMPANIES = (
    "NovaSoft",
    "DataBridge",
    "BrightLabs",
    "CloudForge",
    "VectorApps",
    "InsightFlow",
    "CodeHarbor",
    "TalentGrid",
    "PixelCraft",
    "CoreStack",
    "MetricWave",
    "LaunchPoint",
)

LOCATIONS = ("Kyiv", "Lviv", "Remote", "Dnipro", "Kharkiv", "Odesa", "Warsaw")
EMPLOYMENT_TYPES = ("full-time", "hybrid", "contract")
LANGUAGE_SETS = (
    ["Ukrainian C2", "English B2"],
    ["Ukrainian C2", "English C1"],
    ["Ukrainian C2", "English B1"],
)
TOTAL_PROFILES = 100


def seniority_for_role_index(role_index: int) -> str:
    """Return a predictable seniority distribution for 25 items."""

    if role_index < 8:
        return "Junior"
    if role_index < 17:
        return "Middle"
    return "Senior"


def years_for_seniority(seniority: str, role_index: int) -> float:
    """Create stable experience ranges by seniority."""

    if seniority == "Junior":
        return round(0.8 + (role_index % 8) * 0.2, 1)
    if seniority == "Middle":
        return round(2.0 + ((role_index - 8) % 9) * 0.4, 1)
    return round(5.0 + ((role_index - 17) % 8) * 0.5, 1)


def experience_required(seniority: str, role_index: int) -> float:
    """Return vacancy experience requirements aligned with seniority."""

    if seniority == "Junior":
        return round(0.5 + (role_index % 8) * 0.2, 1)
    if seniority == "Middle":
        return round(2.0 + ((role_index - 8) % 9) * 0.3, 1)
    return round(4.5 + ((role_index - 17) % 8) * 0.4, 1)


def location_for_index(index: int) -> str:
    """Cycle through a realistic set of locations."""

    return LOCATIONS[index % len(LOCATIONS)]


def preferred_locations(location: str, index: int) -> list[str]:
    """Build candidate preferences around the main location."""

    if location == "Remote":
        return ["Remote", LOCATIONS[(index + 1) % len(LOCATIONS)]]
    return [location, "Remote"]


def select_skills(template: RoleTemplate, role_index: int) -> list[str]:
    """Combine core and rotating extra skills without duplicates."""

    extra_count = 2 + (role_index % 3)
    extras = [
        template.extra_skills[(role_index + offset) % len(template.extra_skills)]
        for offset in range(extra_count)
    ]
    return list(dict.fromkeys([*template.core_skills, *extras]))


def select_optional_skills(template: RoleTemplate, role_index: int) -> list[str]:
    """Rotate optional vacancy skills for additional variety."""

    count = 2 + (role_index % 2)
    values = [
        template.optional_skills[(role_index + offset) % len(template.optional_skills)]
        for offset in range(count)
    ]
    return list(dict.fromkeys(values))


def build_full_name(index: int) -> str:
    """Create a stable synthetic Ukrainian-style full name."""

    first_name = FIRST_NAMES[index % len(FIRST_NAMES)]
    last_name = LAST_NAMES[(index * 3) % len(LAST_NAMES)]
    return f"{first_name} {last_name}"


def build_candidate(template: RoleTemplate, global_index: int, role_index: int) -> dict[str, object]:
    """Generate one candidate profile matching the project schema."""

    candidate_id = f"C{global_index + 1:03d}"
    seniority = seniority_for_role_index(role_index)
    years_experience = years_for_seniority(seniority, role_index)
    location = location_for_index(global_index)
    skills = select_skills(template, role_index)
    target_role = template.candidate_roles[role_index % len(template.candidate_roles)]
    full_name = build_full_name(global_index)
    languages = LANGUAGE_SETS[global_index % len(LANGUAGE_SETS)]
    summary = (
        f"{seniority} {template.candidate_title} with {years_experience} years of experience in "
        f"{template.work_focus}."
    )
    work_history = (
        f"Delivered projects focused on {template.work_focus}; worked with "
        f"{', '.join(skills[:5])}; collaborated with product, design, and QA teams."
    )
    education = (
        f"Bachelor degree in {template.education_focus}; continued learning in "
        f"{template.primary_stack.lower()} tooling and engineering practices."
    )
    raw_text = (
        f"{full_name} is a {seniority.lower()} {template.candidate_title.lower()} based in {location}. "
        f"Experience: {years_experience} years. Skills: {', '.join(skills)}. "
        f"Target roles: {target_role}, {template.candidate_roles[(role_index + 1) % len(template.candidate_roles)]}. "
        f"Focus: {template.work_focus}."
    )

    return {
        "candidate_id": candidate_id,
        "full_name": full_name,
        "location": location,
        "preferred_locations": preferred_locations(location, global_index),
        "years_experience": years_experience,
        "seniority_level": seniority,
        "skills": skills,
        "primary_stack": template.primary_stack,
        "languages": languages,
        "education": education,
        "work_history": work_history,
        "target_roles": [
            target_role,
            template.candidate_roles[(role_index + 1) % len(template.candidate_roles)],
        ],
        "summary": summary,
        "raw_text": raw_text,
        "source_file": f"candidate_{candidate_id}.json",
    }


def build_vacancy(template: RoleTemplate, global_index: int, role_index: int) -> dict[str, object]:
    """Generate one vacancy profile matching the project schema."""

    vacancy_id = f"V{global_index + 1:03d}"
    seniority = seniority_for_role_index(role_index)
    location = location_for_index(global_index + 2)
    employment_type = EMPLOYMENT_TYPES[global_index % len(EMPLOYMENT_TYPES)]
    required_skills = select_skills(template, role_index)
    optional_skills = select_optional_skills(template, role_index)
    title = template.vacancy_titles[role_index % len(template.vacancy_titles)]
    company = COMPANIES[global_index % len(COMPANIES)]
    years_required = experience_required(seniority, role_index)
    responsibilities = "; ".join(template.responsibilities)
    requirements_text = (
        f"We are looking for a {seniority.lower()} specialist with hands-on experience in "
        f"{', '.join(required_skills[:5])}. Strong communication and delivery ownership are expected."
    )
    benefits = (
        "Flexible schedule, paid vacation, learning budget, mentorship, and access to modern tooling."
    )
    summary = (
        f"{company} is hiring a {seniority.lower()} {title.lower()} to support "
        f"{template.work_focus}."
    )
    raw_text = (
        f"Vacancy {vacancy_id}: {title} at {company}. Location: {location}. "
        f"Employment: {employment_type}. Required experience: {years_required} years. "
        f"Required skills: {', '.join(required_skills)}. Optional skills: {', '.join(optional_skills)}. "
        f"Responsibilities: {responsibilities}."
    )

    return {
        "vacancy_id": vacancy_id,
        "title": title,
        "company": company,
        "location": location,
        "employment_type": employment_type,
        "years_experience_required": years_required,
        "seniority_level": seniority,
        "required_skills": required_skills,
        "optional_skills": optional_skills,
        "responsibilities": responsibilities,
        "requirements_text": requirements_text,
        "benefits": benefits,
        "summary": summary,
        "raw_text": raw_text,
        "source_file": f"vacancy_{vacancy_id}.json",
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Persist one JSON payload with pretty formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
    """Persist a list of payloads in JSONL format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for payload in payloads:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def profiles_per_template(total_profiles: int, template_count: int) -> list[int]:
    """Split the requested total across all role families."""

    base_count = total_profiles // template_count
    remainder = total_profiles % template_count
    return [
        base_count + (1 if template_index < remainder else 0)
        for template_index in range(template_count)
    ]


def generate_candidates() -> list[dict[str, object]]:
    """Create synthetic candidate profiles across all role families."""

    candidates: list[dict[str, object]] = []
    counts = profiles_per_template(TOTAL_PROFILES, len(ROLE_TEMPLATES))
    global_index = 0
    for template, count in zip(ROLE_TEMPLATES, counts, strict=True):
        for role_index in range(count):
            candidate = build_candidate(template, global_index, role_index)
            candidates.append(candidate)
            write_json(RAW_RESUMES_DIR / candidate["source_file"], candidate)
            global_index += 1
    return candidates


def generate_vacancies() -> list[dict[str, object]]:
    """Create synthetic vacancy profiles across all role families."""

    vacancies: list[dict[str, object]] = []
    counts = profiles_per_template(TOTAL_PROFILES, len(ROLE_TEMPLATES))
    global_index = 0
    for template, count in zip(ROLE_TEMPLATES, counts, strict=True):
        for role_index in range(count):
            vacancy = build_vacancy(template, global_index, role_index)
            vacancies.append(vacancy)
            write_json(RAW_VACANCIES_DIR / vacancy["source_file"], vacancy)
            global_index += 1
    return vacancies


def main() -> None:
    """Generate synthetic raw and processed datasets for local testing."""

    candidates = generate_candidates()
    vacancies = generate_vacancies()

    write_jsonl(PROCESSED_DIR / "resumes.jsonl", candidates)
    write_jsonl(PROCESSED_DIR / "vacancies.jsonl", vacancies)

    print(f"Generated {len(candidates)} resumes in {RAW_RESUMES_DIR}")
    print(f"Generated {len(vacancies)} vacancies in {RAW_VACANCIES_DIR}")
    print(f"Wrote processed files to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
