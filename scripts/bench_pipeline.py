#!/usr/bin/env python
"""Benchmark: End-to-end pipeline — TMAP2 vs UMAP.

Full pipeline comparison: data in → 2D embedding out.
TMAP2 uses frozen HNSW params (no auto-switching). UMAP uses defaults.
Subprocess isolation for fair memory measurement.

Datasets (real data only):
  - Jaccard: ChEMBL 200K Morgan fingerprints (10K, 50K, 100K, 200K)
  - Cosine:  MNIST 70K digits d=784 (10K, 30K, 70K)

Metrics: runtime, peak RSS (via subprocess isolation).

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

def load_chembl(n: int) -> tuple[np.ndarray, str]:
    X = np.load(ROOT / "data" / "chembl" / "chembl_200k_morgan.npy")
    if n < X.shape[0]:
        X = X[np.random.default_rng(SEED).choice(X.shape[0], n, replace=False)]
    return X.astype(np.uint8), "chembl_200k_morgan"


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
    # Frozen HNSW mode: no auto-switching
    model = TMAP(
        n_neighbors=k, metric=metric, seed=seed,
        store_index=(metric in ("cosine", "euclidean")),
    )
    model.fit(X)

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

def bench_metric(metric: str, sizes: list[int]) -> list[dict]:
    results = []
    is_jaccard = metric == "jaccard"

    for n in sizes:
        X, dataset = load_chembl(n) if is_jaccard else load_mnist(n)
        dim = X.shape[1]

        print(f"\n{'=' * 70}")
        print(f" Pipeline: metric={metric}  n={n:,}  [{dataset}]")
        print(f"{'=' * 70}")

        for method in ["tmap2", "umap"]:
            # UMAP doesn't support jaccard on uint8 directly;
            # it needs the metric name to match its internal handling.
            umap_metric = metric
            if method == "umap" and metric == "jaccard":
                # UMAP accepts 'jaccard' for binary data but expects
                # float input; convert.
                X_run = X.astype(np.float32)
            else:
                X_run = X

            print(f"  {method:>6}: ", end="", flush=True)
            try:
                r = run_isolated(method, X_run, umap_metric)
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
    p.add_argument("--sizes-jaccard", default="10000,50000,100000,200000")
    p.add_argument("--sizes-cosine", default="10000,30000,70000")
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
        results = bench_metric(metric, sizes)
        save(results, metric)


if __name__ == "__main__":
    main()
