#!/usr/bin/env python
"""Benchmark 1: USearch frontier — parameter sweep for recall vs latency.

Forces exact and HNSW separately. Sweeps expansion_add and expansion_search.
Reports build time, query latency, recall@20, peak RSS.
Emits a machine-readable chosen default per metric.

Data:
  - Jaccard: data/chembl/chembl_200k_morgan.npy (real molecular fingerprints)
  - Cosine:  synthetic d=768 (protein-embedding scale)

Usage:
  python scripts/bench_usearch_frontier.py                    # full sweep
  python scripts/bench_usearch_frontier.py --metric jaccard   # jaccard only
  python scripts/bench_usearch_frontier.py --metric cosine    # cosine only
  python scripts/bench_usearch_frontier.py --sizes 10000,50000

Results:
  benchmarks/results_paper/usearch_frontier_{metric}.csv
  benchmarks/results_paper/usearch_paper_defaults.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results_paper"

SEED = 42
K = 20
N_QUERIES = 1000  # held-out queries for recall and latency


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else r.ru_maxrss / 1024


def _hardware_meta() -> dict:
    import os
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 1)
        if hasattr(os, "sysconf") else None,
        "python": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_chembl_fps(n: int | None = None) -> np.ndarray:
    X = np.load(ROOT / "data" / "chembl" / "chembl_200k_morgan.npy")
    if n is not None and n < X.shape[0]:
        X = X[np.random.default_rng(SEED).choice(X.shape[0], n, replace=False)]
    return X.astype(np.uint8)


def make_dense(n: int, d: int = 768) -> np.ndarray:
    return np.random.default_rng(SEED).standard_normal((n, d)).astype(np.float32)


# ---------------------------------------------------------------------------
# Exact kNN ground truth
# ---------------------------------------------------------------------------

def exact_knn(X: np.ndarray, metric: str, k: int = K, n_queries: int = N_QUERIES) -> np.ndarray:
    """Exact brute-force kNN for query sample X[:n_queries]. Returns (n_queries, k)."""
    n_q = min(n_queries, len(X))
    Q = X[:n_q]
    idx = (
        NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
        .fit(X)
        .kneighbors(Q, return_distance=False)
    )
    clean = np.zeros((n_q, k), dtype=np.int32)
    for i in range(n_q):
        others = [j for j in idx[i] if j != i][:k]
        clean[i, : len(others)] = others
    return clean


def recall_at_k(pred: np.ndarray, exact: np.ndarray, k: int = K) -> float:
    n = len(exact)
    return sum(len(set(pred[i]) & set(exact[i])) for i in range(n)) / (n * k)


# ---------------------------------------------------------------------------
# Benchmark one config
# ---------------------------------------------------------------------------

def bench_one(
    X: np.ndarray,
    metric: str,
    mode: str,
    exact_idx: np.ndarray,
    expansion_add: int,
    expansion_search: int,
    dataset: str,
    dim: int,
) -> dict:
    from tmap.index.usearch_index import USearchIndex

    n = X.shape[0]
    is_binary = metric == "jaccard"

    idx = USearchIndex(
        seed=SEED,
        mode=mode,
        expansion_add=expansion_add,
        expansion_search=expansion_search,
    )

    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    if is_binary:
        idx.build_from_binary(X)
    else:
        idx.build_from_vectors(X, metric=metric)
    build_s = time.perf_counter() - t0
    rss_after = _peak_rss_mb()

    # Batch query latency on held-out sample (median of 3 runs)
    n_q = min(N_QUERIES, n)
    Q = X[:n_q]
    latencies = []
    pred_idx = None
    for _ in range(3):
        t0 = time.perf_counter()
        pred_idx, _ = idx.query_batch(Q, k=K + 1)
        latencies.append((time.perf_counter() - t0) / n_q * 1000)
    batch_query_ms = float(np.median(latencies))

    # Recall (strip self)
    pred_clean = np.zeros((n_q, K), dtype=np.int32)
    for i in range(n_q):
        others = [int(j) for j in pred_idx[i] if j != i][:K]
        pred_clean[i, : len(others)] = others
    rec = recall_at_k(pred_clean, exact_idx[:n_q], k=K)

    return {
        # identifiers
        "n": n,
        "metric": metric,
        "dataset": dataset,
        "dim": dim,
        # params (structured, not in label)
        "mode": mode,
        "expansion_add": expansion_add if mode == "hnsw" else "",
        "expansion_search": expansion_search if mode == "hnsw" else "",
        "k": K,
        "n_queries": n_q,
        "seed": SEED,
        # timings
        "build_s": round(build_s, 3),
        "batch_query_ms_per_query": round(batch_query_ms, 4),
        "build_points_per_s": round(n / build_s) if build_s > 0 else "",
        # quality
        "recall_at_20": round(rec, 4),
        # memory
        "rss_before_mb": round(rss_before),
        "rss_after_mb": round(rss_after),
        "peak_rss_mb": round(rss_after),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(metric: str, sizes: list[int]) -> list[dict]:
    results = []
    expansion_adds = [128, 256, 512]
    expansion_searches = [100, 200, 400, 800]

    is_binary = metric == "jaccard"
    dataset = "chembl_200k_morgan" if is_binary else "synthetic_normal"
    dim = 2048 if is_binary else 768

    for n in sizes:
        print(f"\n{'=' * 70}")
        print(f" metric={metric}  n={n:,}  dataset={dataset}  dim={dim}")
        print(f"{'=' * 70}")

        X = load_chembl_fps(n) if is_binary else make_dense(n)

        n_q = min(N_QUERIES, n)
        print(f"  Exact kNN ground truth (n_queries={n_q})...", end=" ", flush=True)
        t0 = time.perf_counter()
        exact_idx = exact_knn(X, metric=metric, k=K, n_queries=n_q)
        print(f"done ({time.perf_counter() - t0:.1f}s)")

        # Exact mode (skip at large n)
        if n <= 100_000:
            print("  exact: ", end="", flush=True)
            r = bench_one(X, metric, "exact", exact_idx, 0, 0, dataset, dim)
            print(f"build={r['build_s']:.1f}s  query={r['batch_query_ms_per_query']:.3f}ms  "
                  f"recall={r['recall_at_20']:.4f}  RSS={r['peak_rss_mb']:.0f}MB")
            results.append(r)
        else:
            print(f"  Skipping exact at n={n:,} (too slow)")

        # HNSW sweep
        for ea in expansion_adds:
            for es in expansion_searches:
                print(f"  hnsw ea={ea:>3} es={es:>3}: ", end="", flush=True)
                r = bench_one(X, metric, "hnsw", exact_idx, ea, es, dataset, dim)
                print(
                    f"build={r['build_s']:>7.1f}s  "
                    f"query={r['batch_query_ms_per_query']:>7.3f}ms  "
                    f"recall={r['recall_at_20']:.4f}  "
                    f"RSS={r['peak_rss_mb']:>5.0f}MB"
                )
                results.append(r)

    return results


# ---------------------------------------------------------------------------
# Paper default selection
# ---------------------------------------------------------------------------

def select_defaults(results: list[dict], metric: str, min_recall: float = 0.95) -> dict | None:
    """Select the cheapest HNSW config with recall >= min_recall at the largest n."""
    hnsw = [r for r in results if r["mode"] == "hnsw" and r["metric"] == metric]
    if not hnsw:
        return None
    max_n = max(r["n"] for r in hnsw)
    candidates = [r for r in hnsw if r["n"] == max_n and r["recall_at_20"] >= min_recall]
    if not candidates:
        # Relax: pick best recall at max_n
        candidates = sorted(hnsw, key=lambda r: (-r["n"], -r["recall_at_20"]))[:1]
    # Pick lowest build time among qualifying
    best = min(candidates, key=lambda r: r["build_s"])
    return {
        "metric": metric,
        "expansion_add": best["expansion_add"],
        "expansion_search": best["expansion_search"],
        "recall_at_20": best["recall_at_20"],
        "n": best["n"],
        "selection_rule": f"cheapest HNSW with recall >= {min_recall} at n={max_n:,}",
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], metric: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"usearch_frontier_{metric}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}")
    return path


def save_defaults(defaults: dict, all_defaults: dict) -> None:
    path = RESULTS_DIR / "usearch_paper_defaults.json"
    all_defaults[defaults["metric"]] = defaults
    all_defaults["_hardware"] = _hardware_meta()
    with open(path, "w") as f:
        json.dump(all_defaults, f, indent=2)
    print(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser(description="USearch frontier benchmark")
    p.add_argument("--metric", choices=["jaccard", "cosine", "euclidean", "all"], default="all")
    p.add_argument("--sizes", default="10000,50000,100000,200000")
    p.add_argument(
        "--min-recall", type=float, default=0.95,
        help="Target recall for default selection",
    )
    args = p.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    metrics = ["jaccard", "cosine"] if args.metric == "all" else [args.metric]

    all_defaults = {}
    for metric in metrics:
        results = run_sweep(metric, sizes)
        save_csv(results, metric)
        chosen = select_defaults(results, metric, min_recall=args.min_recall)
        if chosen:
            save_defaults(chosen, all_defaults)
            print(
                f"\n  Chosen default for {metric}: "
                f"ea={chosen['expansion_add']}, "
                f"es={chosen['expansion_search']} "
                f"(recall={chosen['recall_at_20']:.4f} "
                f"at n={chosen['n']:,})"
            )


if __name__ == "__main__":
    main()
