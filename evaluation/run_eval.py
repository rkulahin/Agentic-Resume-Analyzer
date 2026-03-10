"""Functional evaluation script: runs eval queries and checks result quality."""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.graph.main_graph import build_main_graph


def load_eval_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation() -> dict:
    dataset_path = Path(__file__).parent / "eval_dataset.json"
    dataset = load_eval_dataset(dataset_path)

    graph = build_main_graph()

    results = []
    total_hit = 0
    total_queries = len(dataset)

    for item in dataset:
        query_id = item["id"]
        query = item["query"]
        expected_ids = {eid.upper() for eid in item.get("expected_top_ids", [])}

        start = time.time()
        try:
            state = graph.invoke({"user_input": query})
            latency = time.time() - start

            match_results = state.get("match_results", [])
            returned_ids = {m.entity_id.upper() for m in match_results}
            hit = bool(returned_ids & expected_ids)

            if hit:
                total_hit += 1

            results.append({
                "query_id": query_id,
                "query": query[:80],
                "task_type": state.get("task_type", ""),
                "expected": sorted(expected_ids),
                "returned": sorted(returned_ids),
                "hit": hit,
                "top_score": match_results[0].final_score if match_results else 0,
                "latency_s": round(latency, 3),
                "error": state.get("error", ""),
            })
        except Exception as e:
            results.append({
                "query_id": query_id,
                "query": query[:80],
                "hit": False,
                "error": str(e),
                "latency_s": round(time.time() - start, 3),
            })

    hit_rate = total_hit / total_queries if total_queries else 0
    avg_latency = sum(r.get("latency_s", 0) for r in results) / len(results) if results else 0

    summary = {
        "total_queries": total_queries,
        "hits": total_hit,
        "hit_rate": round(hit_rate, 3),
        "avg_latency_s": round(avg_latency, 3),
        "results": results,
    }

    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete: {total_hit}/{total_queries} hits ({hit_rate:.1%})")
    print(f"Average latency: {avg_latency:.3f}s")
    print(f"Results saved to {output_path}")

    return summary


if __name__ == "__main__":
    run_evaluation()
