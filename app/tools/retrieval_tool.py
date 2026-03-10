"""Retrieval tool: searches Chroma for candidates or vacancies matching a query."""

from __future__ import annotations

from langchain_core.tools import tool

from app import config
from app.ingestion.indexer import _get_embedding_fn, get_chroma_client


@tool
def search_candidates(query: str, n_results: int = 5) -> str:
    """Search the candidate database for profiles matching the query.

    Args:
        query: Natural language description of desired candidate profile.
        n_results: Number of results to return.

    Returns:
        Formatted string with matching candidate profiles.
    """
    client = get_chroma_client()
    ef = _get_embedding_fn()
    col = client.get_or_create_collection(config.COLLECTION_CANDIDATES, embedding_function=ef)

    results = col.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        return "No matching candidates found."

    output_parts = []
    for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metadatas), 1):
        output_parts.append(
            f"--- Candidate {i}: {doc_id} ---\n"
            f"Location: {meta.get('location', 'N/A')}\n"
            f"Experience: {meta.get('years_experience', 'N/A')} years\n"
            f"Level: {meta.get('seniority_level', 'N/A')}\n"
            f"Stack: {meta.get('primary_stack', 'N/A')}\n"
            f"Profile:\n{doc}\n"
        )
    return "\n".join(output_parts)


@tool
def search_vacancies(query: str, n_results: int = 5) -> str:
    """Search the vacancy database for positions matching the query.

    Args:
        query: Natural language description of desired position or skills.
        n_results: Number of results to return.

    Returns:
        Formatted string with matching vacancy profiles.
    """
    client = get_chroma_client()
    ef = _get_embedding_fn()
    col = client.get_or_create_collection(config.COLLECTION_VACANCIES, embedding_function=ef)

    results = col.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        return "No matching vacancies found."

    output_parts = []
    for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metadatas), 1):
        output_parts.append(
            f"--- Vacancy {i}: {doc_id} ---\n"
            f"Location: {meta.get('location', 'N/A')}\n"
            f"Type: {meta.get('employment_type', 'N/A')}\n"
            f"Experience required: {meta.get('years_experience_required', 'N/A')} years\n"
            f"Level: {meta.get('seniority_level', 'N/A')}\n"
            f"Description:\n{doc}\n"
        )
    return "\n".join(output_parts)
