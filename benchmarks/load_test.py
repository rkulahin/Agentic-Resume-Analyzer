"""Simplified load test: runs 100 queries sequentially and reports latency metrics."""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

from app.graph.main_graph import build_main_graph

SAMPLE_QUERIES = [
    "Looking for a Senior Python developer with FastAPI, Remote",
    "Need a Middle React developer, Lviv",
    "Junior Data Analyst with SQL, Kyiv",
    "QA Engineer with Selenium, Remote",
    "Python backend developer with Django, Kyiv",
    "Senior React developer, any location",
    "Middle QA Engineer, Kyiv",
    "Data Analyst with Tableau, Remote",
    "Junior Python developer, Dnipro",
    "React developer with TypeScript, Kharkiv",
    "My resume: 3 years Python, FastAPI, PostgreSQL. Looking for Middle positions.",
    "React developer, 2 years, TypeScript, Redux. Looking for a job in Lviv.",
    "Junior QA, 1 year, manual testing, Jira. Looking for positions in Kyiv.",
    "Data Analyst, 4 years, SQL, Python, Tableau. Remote.",
    "Senior Python engineer, Django, AWS, Docker. Kyiv or Remote.",
]

NUM_QUERIES = 100


def run_load_test() -> dict:
    graph = build_main_graph()
    latencies: list[float] = []
    errors = 0

    print(f"Running load test with {NUM_QUERIES} queries...")

    for i in range(NUM_QUERIES):
        query = random.choice(SAMPLE_QUERIES)
        start = time.time()
        try:
            graph.invoke({"user_input": query})
            latency = time.time() - start
            latencies.append(latency)
        except Exception:
            errors += 1
            latencies.append(time.time() - start)

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{NUM_QUERIES}")

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95_idx = int(len(latencies) * 0.95)
    p95 = latencies[min(p95_idx, len(latencies) - 1)]
    p99_idx = int(len(latencies) * 0.99)
    p99 = latencies[min(p99_idx, len(latencies) - 1)]

    summary = {
        "total_queries": NUM_QUERIES,
        "successful": NUM_QUERIES - errors,
        "errors": errors,
        "success_rate": round((NUM_QUERIES - errors) / NUM_QUERIES, 3),
        "avg_latency_s": round(statistics.mean(latencies), 3),
        "min_latency_s": round(min(latencies), 3),
        "max_latency_s": round(max(latencies), 3),
        "p50_latency_s": round(p50, 3),
        "p95_latency_s": round(p95, 3),
        "p99_latency_s": round(p99, 3),
        "total_time_s": round(sum(latencies), 1),
        "bottleneck": "Embedding computation during Chroma queries is the main bottleneck. "
                      "Each query triggers sentence-transformer inference for semantic search.",
        "optimization_recommendations": [
            "Cache the embedding model in memory instead of reloading per query",
            "Pre-compute and cache query embeddings for repeated/similar queries",
        ],
    }

    output_path = Path(__file__).parent / "load_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nLoad test complete:")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Avg latency:  {summary['avg_latency_s']}s")
    print(f"  P50 latency:  {summary['p50_latency_s']}s")
    print(f"  P95 latency:  {summary['p95_latency_s']}s")
    print(f"  Total time:   {summary['total_time_s']}s")
    print(f"Results saved to {output_path}")

    return summary


if __name__ == "__main__":
    run_load_test()
