#!/usr/bin/env python
"""Benchmark 2: Index scaling — pure index build+query, no OGDF.

Measures how each backend scales with N for build time, kNN graph construction,
interactive query latency, and peak RSS. No layout step.

Backends:
  - USearch HNSW (Jaccard/Cosine/Euclidean) — up to 5M
  - TMAP2 MinHash+LSH (Numba JIT) — up to 1M
  - TMAP1 MinHash+LSH (C++) — up to 1M

Data provenance is explicit in every row: dataset column records source.
kc is fixed to 50 across ALL LSH methods for fair comparison.

Usage:
  python scripts/bench_index_scale.py usearch --metric jaccard [--sizes ...]
  python scripts/bench_index_scale.py lsh     [--sizes ...]
  python scripts/bench_index_scale.py all

Results:
  benchmarks/results_paper/index_scale_usearch_{metric}.csv
  benchmarks/results_paper/index_scale_lsh.csv
  benchmarks/results_paper/index_scale_usearch_{metric}_queries.csv  (interactive query companion)
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

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results_paper"
CACHE_DIR = ROOT / "benchmarks" / "cache"

SEED = 42
K = 20
KC = 50  # FIXED across all LSH methods for fair comparison
INTERACTIVE_QUERIES = 1000  # held-out queries for the companion CSV


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else r.ru_maxrss / 1024


def _hardware_meta() -> dict:
    import os
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# Data loading (with explicit provenance)
# ---------------------------------------------------------------------------

def load_chembl_fps(n: int) -> tuple[np.ndarray, str]:
    """Returns (X, dataset_name)."""
    X = np.load(ROOT / "data" / "chembl" / "chembl_200k_morgan.npy")
    if n < X.shape[0]:
        X = X[np.random.default_rng(SEED).choice(X.shape[0], n, replace=False)]
    return X.astype(np.uint8), "chembl_200k_morgan"


def load_smiles_fps(n: int, n_bits: int = 2048) -> tuple[np.ndarray, str]:
    """Returns (X, dataset_name)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"smiles_morgan_{n}_{n_bits}.npy"
    if cache.exists():
        return np.load(cache).astype(np.uint8), f"20M_smiles_morgan_{n}"

    from tmap import fingerprints_from_smiles
    with open(ROOT / "data" / "20M_smiles.txt") as f:
        smiles = [line.strip() for _, line in zip(range(n), f) if line.strip()]
    if len(smiles) < n:
        raise ValueError(f"Only {len(smiles)} SMILES available, requested {n}")

    print(f"    Generating {n:,} Morgan FP (cached)...", end=" ", flush=True)
    t0 = time.perf_counter()
    X = fingerprints_from_smiles(smiles[:n], fp_type="morgan", n_bits=n_bits)
    print(f"done ({time.perf_counter() - t0:.1f}s)")
    np.save(cache, X.astype(np.uint8))
    return X.astype(np.uint8), f"20M_smiles_morgan_{n}"


def load_jaccard_data(n: int) -> tuple[np.ndarray, str]:
    """ChEMBL up to 200K, 20M_smiles.txt beyond. Returns (X, dataset)."""
    if n <= 200_000:
        return load_chembl_fps(n)
    return load_smiles_fps(n)


def make_dense(n: int, d: int = 768) -> tuple[np.ndarray, str]:
    X = np.random.default_rng(SEED).standard_normal((n, d)).astype(np.float32)
    return X, f"synthetic_normal_d{d}"


# ---------------------------------------------------------------------------
# Common row builder
# ---------------------------------------------------------------------------

def _base_row(n: int, metric: str, backend: str, dataset: str, dim: int, **params) -> dict:
    """Columns common to all rows."""
    return {
        "n": n,
        "metric": metric,
        "backend": backend,
        "dataset": dataset,
        "dim": dim,
        "k": K,
        "seed": SEED,
        **params,
    }


# ---------------------------------------------------------------------------
# USearch benchmark
# ---------------------------------------------------------------------------

def bench_usearch(sizes: list[int], metric: str, ea: int, es: int) -> tuple[list[dict], list[dict]]:
    """Returns (scaling_rows, query_companion_rows)."""
    from tmap.index.usearch_index import USearchIndex

    scaling = []
    queries = []
    is_binary = metric == "jaccard"

    for n in sizes:
        X, dataset = load_jaccard_data(n) if is_binary else make_dense(n)
        dim = X.shape[1]
        print(f"\n  USearch {metric} n={n:,} (ea={ea}, es={es}) [{dataset}]", flush=True)

        idx = USearchIndex(seed=SEED, mode="hnsw", expansion_add=ea, expansion_search=es)

        # Build
        rss_before = _peak_rss_mb()
        t0 = time.perf_counter()
        if is_binary:
            idx.build_from_binary(X)
        else:
            idx.build_from_vectors(X, metric=metric)
        build_s = time.perf_counter() - t0

        # Full kNN graph
        t0 = time.perf_counter()
        idx.query_knn(k=K)
        knn_graph_s = time.perf_counter() - t0

        rss_after = _peak_rss_mb()
        knn_ms_per_point = knn_graph_s / n * 1000

        print(f"    build={build_s:>8.1f}s  knn_graph={knn_graph_s:>8.1f}s  "
              f"({knn_ms_per_point:.3f}ms/pt)  RSS={rss_after:>6.0f}MB", flush=True)

        row = _base_row(n, metric, "usearch_hnsw", dataset, dim,
                        mode="hnsw", expansion_add=ea, expansion_search=es)
        row.update({
            "build_s": round(build_s, 2),
            "knn_graph_s": round(knn_graph_s, 2),
            "knn_graph_ms_per_point": round(knn_ms_per_point, 4),
            "total_s": round(build_s + knn_graph_s, 2),
            "build_points_per_s": round(n / build_s) if build_s > 0 else "",
            "knn_graph_points_per_s": round(n / knn_graph_s) if knn_graph_s > 0 else "",
            "neighbors_per_s": round(n * K / knn_graph_s) if knn_graph_s > 0 else "",
            "rss_before_mb": round(rss_before),
            "rss_after_mb": round(rss_after),
            "peak_rss_mb": round(rss_after),
        })
        scaling.append(row)

        # Interactive query companion (held-out queries against index)
        n_q = min(INTERACTIVE_QUERIES, n)
        Q = X[:n_q]
        latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            idx.query_batch(Q, k=K)
            latencies.append((time.perf_counter() - t0) / n_q * 1000)
        interactive_ms = float(np.median(latencies))

        qrow = _base_row(n, metric, "usearch_hnsw", dataset, dim,
                          mode="hnsw", expansion_add=ea, expansion_search=es)
        qrow.update({
            "n_queries": n_q,
            "query_type": "batch_query_held_out",
            "batch_query_ms_per_query": round(interactive_ms, 4),
        })
        queries.append(qrow)
        print(f"    interactive: {interactive_ms:.3f}ms/query ({n_q} queries)", flush=True)

    return scaling, queries


# ---------------------------------------------------------------------------
# LSH benchmarks
# ---------------------------------------------------------------------------

def bench_tmap2_lsh(sizes: list[int], n_perm: int = 512) -> list[dict]:
    from tmap.index.encoders.minhash import MinHash
    from tmap.index.lsh_forest import LSHForest

    results = []
    for n in sizes:
        X, dataset = load_jaccard_data(n)
        dim = X.shape[1]
        l = max(8, n_perm // 4)
        print(f"\n  TMAP2 LSH (d={n_perm}, l={l}, kc={KC}) n={n:,} [{dataset}]", flush=True)

        # Encode
        t0 = time.perf_counter()
        encoder = MinHash(num_perm=n_perm, seed=SEED)
        sigs = encoder.batch_from_binary_array(X)
        encode_s = time.perf_counter() - t0

        # Index
        t0 = time.perf_counter()
        forest = LSHForest(d=n_perm, l=l)
        forest.batch_add(sigs)
        forest.index()
        index_s = time.perf_counter() - t0

        # kNN graph
        t0 = time.perf_counter()
        forest.get_knn_graph(k=K, kc=KC)
        knn_graph_s = time.perf_counter() - t0

        rss = _peak_rss_mb()
        total = encode_s + index_s + knn_graph_s
        knn_ms_per_point = knn_graph_s / n * 1000

        print(f"    encode={encode_s:>6.1f}s  index={index_s:>6.1f}s  "
              f"knn_graph={knn_graph_s:>6.1f}s  total={total:>6.1f}s  "
              f"RSS={rss:>6.0f}MB", flush=True)

        row = _base_row(n, "jaccard", "tmap2_lsh", dataset, dim,
                        n_perm=n_perm, l=l, kc=KC)
        row.update({
            "encode_s": round(encode_s, 2),
            "index_s": round(index_s, 2),
            "build_s": round(encode_s + index_s, 2),
            "knn_graph_s": round(knn_graph_s, 2),
            "knn_graph_ms_per_point": round(knn_ms_per_point, 4),
            "total_s": round(total, 2),
            "build_points_per_s": round(n / (encode_s + index_s))
            if (encode_s + index_s) > 0 else "",
            "knn_graph_points_per_s": round(n / knn_graph_s) if knn_graph_s > 0 else "",
            "neighbors_per_s": round(n * K / knn_graph_s) if knn_graph_s > 0 else "",
            "peak_rss_mb": round(rss),
        })
        results.append(row)

    return results


def bench_tmap1_lsh(sizes: list[int], n_perm: int = 512, l: int | None = None) -> list[dict]:
    import glob as g
    import importlib.util
    import sysconfig

    so_dir = Path(sysconfig.get_paths()["purelib"]) / "tmap"
    candidates = g.glob(str(so_dir / "_tmap*.so"))
    if not candidates:
        print("  TMAP1 not installed, skipping")
        return []
    spec = importlib.util.spec_from_file_location("_tmap", candidates[0])
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    # Use same l as TMAP2 for fair comparison (default: d//4, min 8)
    if l is None:
        l = max(8, n_perm // 4)

    results = []
    for n in sizes:
        X, dataset = load_jaccard_data(n)
        dim = X.shape[1]
        print(f"\n  TMAP1 C++ LSH (d={n_perm}, l={l}, kc={KC}) n={n:,} [{dataset}]", flush=True)

        # Encode + add (row-by-row, this is the bottleneck)
        t0 = time.perf_counter()
        enc = tm.Minhash(n_perm)
        lf = tm.LSHForest(n_perm, l)
        for i in range(n):
            fp = tm.VectorUchar(X[i].tolist())
            lf.add(enc.from_binary_array(fp))
        encode_s = time.perf_counter() - t0

        # Index
        t0 = time.perf_counter()
        lf.index()
        index_s = time.perf_counter() - t0

        # kNN graph (with fixed kc)
        t0 = time.perf_counter()
        v_from = tm.VectorUint()
        v_to = tm.VectorUint()
        v_weight = tm.VectorFloat()
        lf.get_knn_graph(v_from, v_to, v_weight, K, KC)
        knn_graph_s = time.perf_counter() - t0

        rss = _peak_rss_mb()
        total = encode_s + index_s + knn_graph_s
        knn_ms_per_point = knn_graph_s / n * 1000

        print(f"    encode={encode_s:>6.1f}s  index={index_s:>6.1f}s  "
              f"knn_graph={knn_graph_s:>6.1f}s  total={total:>6.1f}s  "
              f"RSS={rss:>6.0f}MB", flush=True)

        row = _base_row(n, "jaccard", "tmap1_cpp", dataset, dim,
                        n_perm=n_perm, l=l, kc=KC)
        row.update({
            "encode_s": round(encode_s, 2),
            "index_s": round(index_s, 2),
            "build_s": round(encode_s + index_s, 2),
            "knn_graph_s": round(knn_graph_s, 2),
            "knn_graph_ms_per_point": round(knn_ms_per_point, 4),
            "total_s": round(total, 2),
            "build_points_per_s": round(n / (encode_s + index_s))
            if (encode_s + index_s) > 0 else "",
            "knn_graph_points_per_s": round(n / knn_graph_s) if knn_graph_s > 0 else "",
            "neighbors_per_s": round(n * K / knn_graph_s) if knn_graph_s > 0 else "",
            "peak_rss_mb": round(rss),
        })
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], filename: str) -> None:
    if not rows:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}")


def save_meta() -> None:
    path = RESULTS_DIR / "hardware.json"
    with open(path, "w") as f:
        json.dump(_hardware_meta(), f, indent=2)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Index scaling benchmark (no OGDF)")
    sub = p.add_subparsers(dest="suite", required=True)

    u = sub.add_parser("usearch", help="USearch HNSW scaling")
    u.add_argument("--metric", choices=["jaccard", "cosine", "euclidean"], default="jaccard")
    u.add_argument("--sizes", default="10000,100000,500000,1000000,2000000,5000000")
    u.add_argument("--ea", type=int, default=512, help="expansion_add (frozen)")
    u.add_argument("--es", type=int, default=400, help="expansion_search (frozen)")

    l = sub.add_parser("lsh", help="LSH backends scaling (TMAP2 Numba + TMAP1 C++)")
    l.add_argument("--sizes", default="10000,50000,100000,200000,500000,1000000,2000000,5000000")
    l.add_argument("--n-perm", type=int, default=512)
    l.add_argument("--l", type=int, default=None, help="Number of prefix trees (default: d//4)")

    a = sub.add_parser("all", help="Run all index benchmarks")
    a.add_argument("--sizes-usearch", default="10000,100000,500000,1000000,2000000,5000000")
    a.add_argument(
        "--sizes-lsh",
        default="10000,50000,100000,200000,500000,1000000,2000000,5000000",
    )
    a.add_argument("--ea", type=int, default=512)
    a.add_argument("--es", type=int, default=400)
    a.add_argument("--n-perm", type=int, default=512)
    a.add_argument("--l", type=int, default=None, help="Number of prefix trees (default: d//4)")

    args = p.parse_args()
    save_meta()

    if args.suite == "usearch":
        sizes = [int(x) for x in args.sizes.split(",")]
        scaling, queries = bench_usearch(sizes, args.metric, ea=args.ea, es=args.es)
        save_csv(scaling, f"index_scale_usearch_{args.metric}.csv")
        save_csv(queries, f"index_scale_usearch_{args.metric}_queries.csv")

    elif args.suite == "lsh":
        sizes = [int(x) for x in args.sizes.split(",")]
        lsh_l = getattr(args, "l", None)
        r2 = bench_tmap2_lsh(sizes, n_perm=args.n_perm)
        r1 = bench_tmap1_lsh(sizes, n_perm=args.n_perm, l=lsh_l)
        save_csv(r2 + r1, "index_scale_lsh.csv")

    elif args.suite == "all":
        sizes_u = [int(x) for x in args.sizes_usearch.split(",")]
        sizes_l = [int(x) for x in args.sizes_lsh.split(",")]
        lsh_l = getattr(args, "l", None)

        for metric in ["jaccard", "cosine"]:
            scaling, queries = bench_usearch(sizes_u, metric, ea=args.ea, es=args.es)
            save_csv(scaling, f"index_scale_usearch_{metric}.csv")
            save_csv(queries, f"index_scale_usearch_{metric}_queries.csv")

        r2 = bench_tmap2_lsh(sizes_l, n_perm=args.n_perm)
        r1 = bench_tmap1_lsh(sizes_l, n_perm=args.n_perm, l=lsh_l)
        save_csv(r2 + r1, "index_scale_lsh.csv")


if __name__ == "__main__":
    main()
