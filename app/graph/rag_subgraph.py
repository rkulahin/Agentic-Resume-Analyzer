"""RAG subgraph: retrieves relevant documents from Chroma collections."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app import config
from app.graph.state import AgentState
from app.ingestion.indexer import _get_embedding_fn, get_chroma_client


def _retrieve_from_collection(
    query: str,
    collection_name: str,
    n_results: int = 5,
    where_filter: dict | None = None,
) -> dict[str, Any]:
    client = get_chroma_client()
    ef = _get_embedding_fn()
    col = client.get_or_create_collection(collection_name, embedding_function=ef)

    kwargs: dict[str, Any] = {"query_texts": [query], "n_results": n_results}
    if where_filter:
        kwargs["where"] = where_filter
    results = col.query(**kwargs)

    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    return {"documents": docs, "ids": ids, "distances": distances}


def retrieve_node(state: AgentState) -> dict:
    task_type = state.get("task_type", "candidate_search")
    criteria = state.get("parsed_criteria", {})

    query_text = criteria.get("query_text", state.get("user_input", ""))

    if task_type == "candidate_search":
        collection = config.COLLECTION_CANDIDATES
    else:
        collection = config.COLLECTION_VACANCIES

    results = _retrieve_from_collection(query_text, collection, n_results=10)

    knowledge_results = _retrieve_from_collection(
        query_text, config.COLLECTION_KNOWLEDGE, n_results=3
    )

    all_docs = results["documents"] + knowledge_results["documents"]
    all_ids = results["ids"] + knowledge_results["ids"]

    return {
        "retrieved_context": all_docs,
        "retrieved_ids": all_ids,
    }


def build_rag_subgraph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", END)
    return graph.compile()
