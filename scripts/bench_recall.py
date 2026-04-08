#!/usr/bin/env python
"""Benchmark: Fair recall comparison across backends.

All backends compared on the SAME real data, SAME k, SAME kc.
Ground truth: sklearn brute-force exact kNN on 1000 query points.

Datasets:
  - Jaccard:   ChEMBL/20M-SMILES Morgan FPs (10K to 5M)
  - Cosine:    MNIST 70K + synthetic d=768 beyond (10K to 5M)
  - Euclidean: synthetic d=768 (10K to 5M)

Backends:
  - USearch exact (reference, n <= 200K)
  - USearch HNSW (sweep: ea in [128, 256, 512])
  - TMAP2 Numba LSH d=512 (Jaccard only)
  - TMAP1 C++ LSH d=512 (Jaccard only)

All LSH methods use kc=50. USearch uses frozen paper defaults.

Usage:
  python scripts/bench_recall.py --metric jaccard
  python scripts/bench_recall.py --metric cosine
  python scripts/bench_recall.py --metric euclidean
  python scripts/bench_recall.py  # all three

Results: benchmarks/results_paper/recall_{metric}.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results_paper"
CACHE_DIR = ROOT / "benchmarks" / "cache"

SEED = 42
K = 20
KC = 50  # fixed across all LSH methods
N_QUERIES = 1000
N_PERM = 512  # MinHash permutations for LSH methods


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_chembl(n: int) -> tuple[np.ndarray, str]:
    X = np.load(ROOT / "data" / "chembl" / "chembl_200k_morgan.npy")
    if n <= X.shape[0]:
        if n < X.shape[0]:
            X = X[np.random.default_rng(SEED).choice(X.shape[0], n, replace=False)]
        return X.astype(np.uint8), "chembl_200k_morgan"
    # For larger sizes, use 20M SMILES
    return load_smiles_fps(n)


def load_smiles_fps(n: int, n_bits: int = 2048) -> tuple[np.ndarray, str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"smiles_morgan_{n}_{n_bits}.npy"
    if cache.exists():
        return np.load(cache).astype(np.uint8), f"20M_smiles_morgan_{n}"
    from tmap import fingerprints_from_smiles
    smiles_path = ROOT / "data" / "20M_smiles.txt"
    with open(smiles_path) as f:
        smiles = [line.strip() for _, line in zip(range(n), f) if line.strip()]
    print(f"  Generating {n:,} Morgan FP...", end=" ", flush=True)
    import time as _t
    t0 = _t.perf_counter()
    X = fingerprints_from_smiles(smiles[:n], fp_type="morgan", n_bits=n_bits)
    print(f"done ({_t.perf_counter() - t0:.1f}s)")
    np.save(cache, X.astype(np.uint8))
    return X.astype(np.uint8), f"20M_smiles_morgan_{n}"


def load_mnist(n: int) -> tuple[np.ndarray, str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "mnist_784.npz"
    if cache.exists():
        X = np.load(cache)["X"]
    else:
        from sklearn.datasets import fetch_openml
        X, _ = fetch_openml(
            "mnist_784", version=1, return_X_y=True,
            as_frame=False, parser="auto",
        )
        X = X.astype(np.float32)
        np.savez_compressed(cache, X=X)
    if n < X.shape[0]:
        X = X[np.random.default_rng(SEED).choice(X.shape[0], n, replace=False)]
    return X.astype(np.float32), "mnist_784"


def load_dense(n: int, d: int = 768, metric_name: str = "cosine") -> tuple[np.ndarray, str]:
    """MNIST for n<=70K (cosine only), synthetic beyond."""
    if metric_name == "cosine" and n <= 70_000:
        return load_mnist(n)
    return (
        np.random.default_rng(SEED).standard_normal((n, d)).astype(np.float32),
        f"synthetic_d{d}",
    )


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def exact_knn(
    X: np.ndarray, metric: str, k: int = K, n_queries: int = N_QUERIES,
) -> np.ndarray:
    """Brute-force exact kNN. Returns (n_queries, k) index array."""
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


def recall_at_k(pred: np.ndarray, truth: np.ndarray) -> float:
    n = len(truth)
    k = truth.shape[1]
    return sum(
        len(set(pred[i]) & set(truth[i])) for i in range(n)
    ) / (n * k)


# ---------------------------------------------------------------------------
# Backend: USearch
# ---------------------------------------------------------------------------

def recall_usearch(
    X: np.ndarray, truth: np.ndarray, metric: str,
    mode: str, ea: int, es: int,
) -> dict:
    from tmap.index.usearch_index import USearchIndex

    n_q = len(truth)
    is_binary = metric == "jaccard"

    idx = USearchIndex(
        seed=SEED, mode=mode,
        expansion_add=ea, expansion_search=es,
    )
    t0 = time.perf_counter()
    if is_binary:
        idx.build_from_binary(X)
    else:
        idx.build_from_vectors(X, metric=metric)
    build_s = time.perf_counter() - t0

    Q = X[:n_q]
    pred_raw, _ = idx.query_batch(Q, k=K + 1)
    pred = np.zeros((n_q, K), dtype=np.int32)
    for i in range(n_q):
        others = [int(j) for j in pred_raw[i] if j != i][:K]
        pred[i, : len(others)] = others

    rec = recall_at_k(pred, truth)
    label = f"usearch_{mode}"
    if mode == "hnsw":
        label = f"usearch_hnsw(ea={ea},es={es})"

    return {
        "backend": label,
        "mode": mode,
        "expansion_add": ea if mode == "hnsw" else "",
        "expansion_search": es if mode == "hnsw" else "",
        "n_perm": "",
        "kc": "",
        "build_s": round(build_s, 2),
        "recall_at_20": round(rec, 4),
    }


# ---------------------------------------------------------------------------
# Backend: TMAP2 LSH (Numba)
# ---------------------------------------------------------------------------

def recall_tmap2_lsh(
    X: np.ndarray, truth: np.ndarray,
    n_perm: int = N_PERM,
) -> dict:
    from tmap.index.encoders.minhash import MinHash
    from tmap.index.lsh_forest import LSHForest

    n_q = len(truth)

    t0 = time.perf_counter()
    encoder = MinHash(num_perm=n_perm, seed=SEED)
    sigs = encoder.batch_from_binary_array(X)
    l = max(8, n_perm // 4)
    forest = LSHForest(d=n_perm, l=l)
    forest.batch_add(sigs)
    forest.index()
    knn = forest.get_knn_graph(k=K, kc=KC)
    build_s = time.perf_counter() - t0

    pred = knn.indices[:n_q]
    rec = recall_at_k(pred, truth)

    return {
        "backend": f"tmap2_lsh(d={n_perm},kc={KC})",
        "mode": "lsh",
        "expansion_add": "",
        "expansion_search": "",
        "n_perm": n_perm,
        "kc": KC,
        "build_s": round(build_s, 2),
        "recall_at_20": round(rec, 4),
    }


# ---------------------------------------------------------------------------
# Backend: TMAP1 C++ LSH
# ---------------------------------------------------------------------------

def recall_tmap1_lsh(
    X: np.ndarray, truth: np.ndarray,
    n_perm: int = N_PERM,
) -> dict | None:
    import glob as g
    import importlib.util
    import sysconfig

    so_dir = Path(sysconfig.get_paths()["purelib"]) / "tmap"
    candidates = g.glob(str(so_dir / "_tmap*.so"))
    if not candidates:
        print("      TMAP1 not installed, skipping")
        return None
    spec = importlib.util.spec_from_file_location("_tmap", candidates[0])
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    n = X.shape[0]
    n_q = len(truth)

    # Use same l as TMAP2 for fair comparison: l = max(8, d//4)
    l = max(8, n_perm // 4)
    t0 = time.perf_counter()
    enc = tm.Minhash(n_perm)
    lf = tm.LSHForest(n_perm, l)
    for i in range(n):
        fp = tm.VectorUchar(X[i].tolist())
        lf.add(enc.from_binary_array(fp))
    lf.index()

    # Extract kNN graph with fixed kc
    v_from = tm.VectorUint()
    v_to = tm.VectorUint()
    v_weight = tm.VectorFloat()
    lf.get_knn_graph(v_from, v_to, v_weight, K, KC)
    build_s = time.perf_counter() - t0

    # Build per-node neighbor lists
    from collections import defaultdict
    adj: dict[int, list[int]] = defaultdict(list)
    for src, dst in zip(v_from, v_to):
        adj[int(src)].append(int(dst))

    pred = np.full((n_q, K), -1, dtype=np.int32)
    for i in range(n_q):
        neighbors = adj.get(i, [])[:K]
        pred[i, : len(neighbors)] = neighbors

    rec = recall_at_k(pred, truth)

    return {
        "backend": f"tmap1_cpp(d={n_perm},kc={KC})",
        "mode": "lsh",
        "expansion_add": "",
        "expansion_search": "",
        "n_perm": n_perm,
        "kc": KC,
        "build_s": round(build_s, 2),
        "recall_at_20": round(rec, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def bench_metric(metric: str, sizes: list[int]) -> list[dict]:
    results = []
    is_jaccard = metric == "jaccard"

    for n in sizes:
        if is_jaccard:
            X, dataset = load_chembl(n)
        else:
            X, dataset = load_dense(n, metric_name=metric)
        dim = X.shape[1]

        print(f"\n{'=' * 70}")
        print(f" Recall comparison: metric={metric}  n={n:,}  [{dataset}]")
        print(f"{'=' * 70}")

        # Ground truth (brute-force only feasible up to ~200K)
        if n <= 200_000:
            print(
                f"  Exact kNN (brute force, {N_QUERIES} queries)...",
                end=" ", flush=True,
            )
            t0 = time.perf_counter()
            truth = exact_knn(X, metric=metric, k=K, n_queries=N_QUERIES)
            print(f"done ({time.perf_counter() - t0:.1f}s)")
        else:
            # For n > 200K, use USearch exact on query sample as ground truth
            print(
                f"  Ground truth via USearch exact ({N_QUERIES} queries)...",
                end=" ", flush=True,
            )
            t0 = time.perf_counter()
            truth = _exact_knn_usearch(X, metric=metric, k=K, n_queries=N_QUERIES)
            print(f"done ({time.perf_counter() - t0:.1f}s)")

        base = {
            "n": n,
            "metric": metric,
            "dataset": dataset,
            "dim": dim,
            "k": K,
            "n_queries": N_QUERIES,
            "seed": SEED,
        }

        # USearch exact (reference, only feasible at moderate sizes)
        if n <= 200_000:
            print("  USearch exact...", end=" ", flush=True)
            r = recall_usearch(X, truth, metric, "exact", 0, 0)
            print(f"recall={r['recall_at_20']:.4f}")
            results.append({**base, **r})

        # USearch HNSW at multiple ea (es fixed at paper default)
        es = 400 if is_jaccard else 800
        for ea in [128, 256, 512]:
            print(
                f"  USearch HNSW ea={ea} es={es}...",
                end=" ", flush=True,
            )
            r = recall_usearch(X, truth, metric, "hnsw", ea, es)
            print(f"recall={r['recall_at_20']:.4f}  ({r['build_s']:.1f}s)")
            results.append({**base, **r})

        # LSH backends (Jaccard only)
        if is_jaccard:
            print(
                f"  TMAP2 LSH (d={N_PERM}, kc={KC})...",
                end=" ", flush=True,
            )
            r = recall_tmap2_lsh(X, truth, n_perm=N_PERM)
            print(f"recall={r['recall_at_20']:.4f}  ({r['build_s']:.1f}s)")
            results.append({**base, **r})

            print(
                f"  TMAP1 C++ (d={N_PERM}, kc={KC})...",
                end=" ", flush=True,
            )
            r = recall_tmap1_lsh(X, truth, n_perm=N_PERM)
            if r:
                print(f"recall={r['recall_at_20']:.4f}  ({r['build_s']:.1f}s)")
                results.append({**base, **r})

    return results


def _exact_knn_usearch(
    X: np.ndarray, metric: str, k: int = K, n_queries: int = N_QUERIES,
) -> np.ndarray:
    """Ground truth via USearch exact mode for large datasets where sklearn
    brute-force is too slow/memory-intensive."""
    from tmap.index.usearch_index import USearchIndex

    n_q = min(n_queries, len(X))
    Q = X[:n_q]

    idx = USearchIndex(seed=SEED, mode="exact")
    if metric == "jaccard":
        idx.build_from_binary(X)
    else:
        idx.build_from_vectors(X, metric=metric)

    pred_raw, _ = idx.query_batch(Q, k=k + 1)
    clean = np.zeros((n_q, k), dtype=np.int32)
    for i in range(n_q):
        others = [int(j) for j in pred_raw[i] if j != i][:k]
        clean[i, : len(others)] = others
    return clean


def save(rows: list[dict], metric: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"recall_{metric}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}")


def main():
    p = argparse.ArgumentParser(
        description="Fair recall comparison across backends",
    )
    p.add_argument(
        "--metric", choices=["jaccard", "cosine", "euclidean", "all"],
        default="all",
    )
    p.add_argument(
        "--sizes-jaccard",
        default="10000,50000,100000,200000,500000,1000000,2000000,5000000",
    )
    p.add_argument(
        "--sizes-dense",
        default="10000,50000,100000,200000,500000,1000000,2000000,5000000",
    )
    args = p.parse_args()

    metrics = (
        ["jaccard", "cosine", "euclidean"] if args.metric == "all"
        else [args.metric]
    )
    for metric in metrics:
        if metric == "jaccard":
            sizes = [int(x) for x in args.sizes_jaccard.split(",")]
        else:
            sizes = [int(x) for x in args.sizes_dense.split(",")]
        results = bench_metric(metric, sizes)
        save(results, metric)


if __name__ == "__main__":
    main()
