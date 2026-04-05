#!/usr/bin/env python
"""Benchmark: add_points() vs full refit.

Measures the cost of incremental insertion compared to rebuilding from scratch.
Reports time, peak RSS, and placement consistency (kNN agreement between
the incremental and refit embeddings for the original points).

Data: ChEMBL Morgan fingerprints (Jaccard) and synthetic (cosine).

Usage:
  python scripts/bench_addpoints.py --metric jaccard
  python scripts/bench_addpoints.py --metric cosine
  python scripts/bench_addpoints.py  # both

Results: benchmarks/results_paper/addpoints_{metric}.csv
"""

from __future__ import annotations

import argparse
import csv
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


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return r.ru_maxrss / (1024 * 1024)
    return r.ru_maxrss / 1024


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_chembl(n: int) -> np.ndarray:
    X = np.load(ROOT / "data" / "chembl" / "chembl_200k_morgan.npy")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(X.shape[0], min(n, X.shape[0]), replace=False)
    return X[idx].astype(np.uint8)


def make_dense(n: int, d: int = 768) -> np.ndarray:
    return np.random.default_rng(SEED).standard_normal(
        (n, d)
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Placement consistency: do original points keep their 2D neighbors?
# ---------------------------------------------------------------------------

def knn_agreement(emb_a: np.ndarray, emb_b: np.ndarray, k: int = K) -> float:
    """Fraction of k-NN in emb_a that are also k-NN in emb_b (same points)."""
    n = min(len(emb_a), len(emb_b))
    emb_a, emb_b = emb_a[:n], emb_b[:n]

    nn_a = (
        NearestNeighbors(n_neighbors=k, metric="euclidean")
        .fit(emb_a)
        .kneighbors(emb_a, return_distance=False)
    )
    nn_b = (
        NearestNeighbors(n_neighbors=k, metric="euclidean")
        .fit(emb_b)
        .kneighbors(emb_b, return_distance=False)
    )
    total = sum(
        len(set(nn_a[i]) & set(nn_b[i])) for i in range(n)
    )
    return total / (n * k)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_metric(
    metric: str, base_n: int, insert_sizes: list[int],
) -> list[dict]:
    from tmap import TMAP

    results = []
    is_binary = metric == "jaccard"

    # Load all data upfront
    total_n = base_n + max(insert_sizes)
    if is_binary:
        X_all = load_chembl(total_n)
    else:
        X_all = make_dense(total_n)

    X_base = X_all[:base_n]
    dataset = "chembl_200k_morgan" if is_binary else "synthetic_d768"

    print(f"\n{'=' * 70}")
    print(f" add_points vs refit: metric={metric}  base_n={base_n:,}")
    print(f" dataset={dataset}  insert_sizes={insert_sizes}")
    print(f"{'=' * 70}")

    # Fit base model
    print(f"\n  Fitting base model (n={base_n:,})...", end=" ", flush=True)
    t0 = time.perf_counter()
    # store_index=True required for add_points with dense metrics
    needs_index = metric in ("cosine", "euclidean")
    base_model = TMAP(
        n_neighbors=K, metric=metric, seed=SEED,
        store_index=needs_index,
    )
    base_model.fit(X_base)
    base_fit_s = time.perf_counter() - t0
    base_emb = base_model.embedding_.copy()
    base_rss = _peak_rss_mb()
    print(f"{base_fit_s:.1f}s  RSS={base_rss:.0f}MB")

    for ins_n in insert_sizes:
        X_new = X_all[base_n: base_n + ins_n]
        X_combined = X_all[: base_n + ins_n]

        print(f"\n  --- insert {ins_n:,} points ---")

        # Method 1: add_points (incremental)
        # Re-fit base to get a fresh model (add_points mutates)
        model_inc = TMAP(
            n_neighbors=K, metric=metric, seed=SEED,
            store_index=needs_index,
        )
        model_inc.fit(X_base)

        t0 = time.perf_counter()
        model_inc.add_points(X_new)
        add_s = time.perf_counter() - t0
        add_rss = _peak_rss_mb()
        emb_inc = model_inc.embedding_[:base_n]  # original points only
        print(
            f"    add_points: {add_s:>7.2f}s  "
            f"RSS={add_rss:>6.0f}MB"
        )

        # Method 2: full refit from scratch
        t0 = time.perf_counter()
        model_refit = TMAP(n_neighbors=K, metric=metric, seed=SEED)
        model_refit.fit(X_combined)
        refit_s = time.perf_counter() - t0
        refit_rss = _peak_rss_mb()
        _ = model_refit.embedding_[:base_n]  # ensure layout computed
        print(
            f"    full refit: {refit_s:>7.2f}s  "
            f"RSS={refit_rss:>6.0f}MB"
        )

        # Placement consistency
        agreement = knn_agreement(emb_inc, base_emb, k=K)
        speedup = refit_s / add_s if add_s > 0 else float("inf")
        print(
            f"    speedup: {speedup:>5.1f}x  "
            f"kNN agreement (vs base): {agreement:.4f}"
        )

        results.append({
            "metric": metric,
            "dataset": dataset,
            "base_n": base_n,
            "insert_n": ins_n,
            "total_n": base_n + ins_n,
            "k": K,
            "seed": SEED,
            "base_fit_s": round(base_fit_s, 2),
            "add_points_s": round(add_s, 2),
            "refit_s": round(refit_s, 2),
            "speedup": round(speedup, 2),
            "knn_agreement_vs_base": round(agreement, 4),
            "add_rss_mb": round(add_rss),
            "refit_rss_mb": round(refit_rss),
        })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def save(rows: list[dict], metric: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"addpoints_{metric}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}")


def main():
    p = argparse.ArgumentParser(description="add_points vs refit benchmark")
    p.add_argument(
        "--metric", choices=["jaccard", "cosine", "all"],
        default="all",
    )
    p.add_argument("--base-n", type=int, default=100_000)
    p.add_argument(
        "--insert-sizes", default="1000,5000,10000,50000",
    )
    args = p.parse_args()

    insert_sizes = [int(x) for x in args.insert_sizes.split(",")]
    metrics = (
        ["jaccard", "cosine"] if args.metric == "all"
        else [args.metric]
    )

    for metric in metrics:
        results = bench_metric(metric, args.base_n, insert_sizes)
        save(results, metric)


if __name__ == "__main__":
    main()
