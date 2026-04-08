#!/usr/bin/env python
"""Benchmark: End-to-end pipeline — TMAP2 vs UMAP, with external kNN backends.

Full pipeline comparison: data in → 2D embedding out.
TMAP2 uses frozen HNSW params (no auto-switching). UMAP uses defaults.
Also benchmarks TMAP2 with external kNN from FAISS and PyNNDescent to show
that TMAP accepts precomputed kNN and benefits from faster backends.
Subprocess isolation for fair memory measurement.

Datasets:
  - Jaccard: ChEMBL/20M-SMILES Morgan FPs (10K to 5M)
  - Cosine:  MNIST 70K + synthetic beyond (10K to 1M)

Methods:
  - tmap2: Default TMAP2 (USearch HNSW)
  - tmap2_faiss: TMAP2 with FAISS kNN (cosine/euclidean only)
  - tmap2_pynndescent: TMAP2 with PyNNDescent kNN
  - umap: UMAP defaults

Usage:
  python scripts/bench_pipeline.py --metric jaccard
  python scripts/bench_pipeline.py --metric cosine
  python scripts/bench_pipeline.py  # both

Results: benchmarks/results_paper/pipeline_{metric}.csv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results_paper"
CACHE_DIR = ROOT / "benchmarks" / "cache"

SEED = 42
K = 20


# ---------------------------------------------------------------------------
# Data (real only)
# ---------------------------------------------------------------------------

def load_jaccard_data(n: int) -> tuple[np.ndarray, str]:
    """ChEMBL up to 200K, 20M_smiles beyond."""
    chembl_path = ROOT / "data" / "chembl" / "chembl_200k_morgan.npy"
    if n <= 200_000 and chembl_path.exists():
        X = np.load(chembl_path)
        if n < X.shape[0]:
            X = X[np.random.default_rng(SEED).choice(X.shape[0], n, replace=False)]
        return X.astype(np.uint8), "chembl_200k_morgan"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"smiles_morgan_{n}_2048.npy"
    if cache.exists():
        return np.load(cache).astype(np.uint8), f"20M_smiles_morgan_{n}"
    from tmap import fingerprints_from_smiles
    with open(ROOT / "data" / "20M_smiles.txt") as f:
        smiles = [line.strip() for _, line in zip(range(n), f) if line.strip()]
    import time as _t
    print(f"  Generating {n:,} Morgan FP...", end=" ", flush=True)
    t0 = _t.perf_counter()
    X = fingerprints_from_smiles(smiles[:n], fp_type="morgan", n_bits=2048)
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


def make_dense(n: int, d: int = 1280) -> tuple[np.ndarray, str]:
    X = np.random.default_rng(SEED).standard_normal(
        (n, d),
    ).astype(np.float32)
    return X, f"synthetic_d{d}"


# ---------------------------------------------------------------------------
# Subprocess runner (isolates memory measurement)
# ---------------------------------------------------------------------------

_WORKER = """
import sys, time, resource, platform, numpy as np
from pathlib import Path

def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return r.ru_maxrss / (1024 * 1024)
    return r.ru_maxrss / 1024

data = np.load(sys.argv[1])
X = data["X"]
method = sys.argv[2]
metric = sys.argv[3]
k = int(sys.argv[4])
seed = int(sys.argv[5])

t0 = time.perf_counter()

if method == "tmap2":
    from tmap import TMAP
    model = TMAP(
        n_neighbors=k, metric=metric, seed=seed,
        store_index=(metric in ("cosine", "euclidean")),
    )
    model.fit(X)

elif method == "tmap2_faiss":
    # TMAP2 with FAISS kNN (cosine/euclidean only)
    import faiss
    from tmap import TMAP
    from tmap.index.types import KNNGraph

    n, d = X.shape
    if metric == "cosine":
        faiss.normalize_L2(X)
    index = faiss.IndexFlatL2(d) if metric != "cosine" else faiss.IndexFlatIP(d)
    index.add(X)
    distances, indices = index.search(X, k)
    if metric == "cosine":
        distances = 1.0 - distances  # IP → cosine distance
    knn = KNNGraph(indices=indices.astype(np.int32), distances=distances.astype(np.float32))
    model = TMAP(n_neighbors=k, metric="precomputed", seed=seed)
    model.fit(knn_graph=knn)

elif method == "tmap2_pynndescent":
    # TMAP2 with PyNNDescent kNN
    from pynndescent import NNDescent
    from tmap import TMAP
    from tmap.index.types import KNNGraph

    pynn_metric = metric if metric != "jaccard" else "jaccard"
    nnd = NNDescent(X, metric=pynn_metric, n_neighbors=k, random_state=seed)
    indices, distances = nnd.neighbor_graph
    knn = KNNGraph(indices=indices.astype(np.int32), distances=distances.astype(np.float32))
    model = TMAP(n_neighbors=k, metric="precomputed", seed=seed)
    model.fit(knn_graph=knn)

elif method == "umap":
    import umap
    umap.UMAP(
        n_components=2, n_neighbors=k, min_dist=0.1,
        metric=metric, random_state=seed,
    ).fit_transform(X)

runtime_s = time.perf_counter() - t0
rss = peak_rss_mb()

out = Path(sys.argv[6])
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, runtime_s=np.float64(runtime_s), peak_rss_mb=np.float64(rss))
"""


def run_isolated(
    method: str, X: np.ndarray, metric: str,
) -> dict:
    """Run one method in a subprocess for isolated memory measurement."""
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.npz"
        out = Path(tmp) / "out.npz"
        worker = Path(tmp) / "worker.py"

        np.savez_compressed(inp, X=X)
        worker.write_text(_WORKER)

        cmd = [
            sys.executable, str(worker),
            str(inp), method, metric, str(K), str(SEED), str(out),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{method} failed (rc={proc.returncode}):\n"
                f"{proc.stderr[-2000:]}"
            )

        data = np.load(out)
        return {
            "runtime_s": float(data["runtime_s"]),
            "peak_rss_mb": float(data["peak_rss_mb"]),
        }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _get_methods(metric: str) -> list[str]:
    """Which methods to benchmark for a given metric."""
    methods = ["tmap2", "umap"]
    # FAISS only supports dense metrics (cosine/euclidean)
    if metric in ("cosine", "euclidean"):
        methods.append("tmap2_faiss")
    # PyNNDescent supports all metrics
    methods.append("tmap2_pynndescent")
    return methods


def bench_metric(
    metric: str, sizes: list[int], cosine_dataset: str = "mnist",
    dim: int = 784,
) -> list[dict]:
    results = []
    is_jaccard = metric == "jaccard"
    methods = _get_methods(metric)

    for n in sizes:
        if is_jaccard:
            X, dataset = load_jaccard_data(n)
        elif cosine_dataset == "mnist" and n <= 70_000:
            X, dataset = load_mnist(n)
        else:
            X, dataset = make_dense(n, d=dim)
        dim = X.shape[1]

        print(f"\n{'=' * 70}")
        print(f" Pipeline: metric={metric}  n={n:,}  [{dataset}]")
        print(f"{'=' * 70}")

        for method in methods:
            if method == "umap" and metric == "jaccard":
                X_run = X.astype(np.float32)
            else:
                X_run = X

            print(f"  {method:>20}: ", end="", flush=True)
            try:
                r = run_isolated(method, X_run, metric)
                print(
                    f"{r['runtime_s']:>8.1f}s  "
                    f"RSS={r['peak_rss_mb']:>6.0f}MB"
                )
            except Exception as e:
                print(f"FAILED: {e}")
                r = {"runtime_s": "", "peak_rss_mb": ""}

            results.append({
                "n": n,
                "metric": metric,
                "dataset": dataset,
                "dim": dim,
                "method": method,
                "k": K,
                "seed": SEED,
                "runtime_s": round(r["runtime_s"], 2)
                if r["runtime_s"] else "",
                "peak_rss_mb": round(r["peak_rss_mb"])
                if r["peak_rss_mb"] else "",
            })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def save(rows: list[dict], metric: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"pipeline_{metric}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}")


def main():
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: TMAP2 vs UMAP",
    )
    p.add_argument(
        "--metric", choices=["jaccard", "cosine", "all"],
        default="all",
    )
    p.add_argument(
        "--sizes-jaccard",
        default="10000,50000,100000,200000,500000,1000000,2000000,5000000",
    )
    p.add_argument("--sizes-cosine", default="10000,50000,100000,200000,500000,1000000")
    p.add_argument(
        "--cosine-dataset", choices=["mnist", "synthetic"],
        default="mnist",
    )
    p.add_argument("--dim", type=int, default=1280, help="Dim for synthetic")
    args = p.parse_args()

    metrics = (
        ["jaccard", "cosine"] if args.metric == "all"
        else [args.metric]
    )
    for metric in metrics:
        sizes = (
            [int(x) for x in args.sizes_jaccard.split(",")]
            if metric == "jaccard"
            else [int(x) for x in args.sizes_cosine.split(",")]
        )
        results = bench_metric(
            metric, sizes,
            cosine_dataset=args.cosine_dataset,
            dim=args.dim,
        )
        save(results, metric)


if __name__ == "__main__":
    main()
