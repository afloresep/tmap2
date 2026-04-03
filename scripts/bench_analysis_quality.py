#!/usr/bin/env python
"""Benchmark: Analysis quality — where TMAP shines vs UMAP.

Measures tree-specific quality metrics on labeled datasets:
  - same_label_edge_fraction: fraction of MST edges connecting same-label nodes
  - subtree_purity: average label purity of subtrees (min_size=10)
  - boundary_edge_fraction: fraction of edges at label boundaries
  - path_label_transitions: avg label changes along random paths
  - path_monotonicity: Spearman correlation of tree distance vs label ordering
    (for ordinal labels like digits or pseudotime)

Datasets:
  - MNIST digits (cosine, 10 classes, ordinal)
  - ChEMBL scaffolds (jaccard, categorical)

Usage:
  python scripts/bench_analysis_quality.py --dataset mnist
  python scripts/bench_analysis_quality.py --dataset chembl
  python scripts/bench_analysis_quality.py  # both

Results: benchmarks/results_paper/analysis_quality_{dataset}.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results_paper"
CACHE_DIR = ROOT / "benchmarks" / "cache"

SEED = 42
K = 20
N_PATHS = 500  # random paths for path metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist(n: int = 70_000) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X float32, labels int)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "mnist_784.npz"
    if cache.exists():
        data = np.load(cache)
        X, y = data["X"], data["y"]
    else:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml(
            "mnist_784", version=1, return_X_y=True,
            as_frame=False, parser="auto",
        )
        X = X.astype(np.float32)
        y = y.astype(int)
        np.savez_compressed(cache, X=X, y=y)

    if n < X.shape[0]:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(X.shape[0], n, replace=False)
        X, y = X[idx], y[idx]
    return X.astype(np.float32), y.astype(int)


def load_chembl_with_scaffolds(
    n: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X uint8, scaffold_labels int)."""
    X = np.load(ROOT / "data" / "chembl" / "chembl_200k_morgan.npy")

    # Generate scaffold labels (Murcko)
    cache = CACHE_DIR / "chembl_200k_scaffolds.npy"
    if cache.exists():
        scaffolds = np.load(cache, allow_pickle=True)
    else:
        import pandas as pd

        from tmap import murcko_scaffolds

        meta = pd.read_parquet(
            ROOT / "data" / "chembl" / "chembl_200k_meta.parquet"
        )
        if "smiles" in meta.columns:
            smiles = meta["smiles"].tolist()
        elif "canonical_smiles" in meta.columns:
            smiles = meta["canonical_smiles"].tolist()
        else:
            raise ValueError(
                f"No SMILES column found. Columns: {meta.columns.tolist()}"
            )
        scaffolds = np.array(murcko_scaffolds(smiles))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, scaffolds, allow_pickle=True)

    # Convert scaffold strings to integer labels
    unique_scaffolds, labels = np.unique(scaffolds, return_inverse=True)

    if n < X.shape[0]:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(X.shape[0], n, replace=False)
        X, labels = X[idx], labels[idx]

    return X.astype(np.uint8), labels


# ---------------------------------------------------------------------------
# Tree quality metrics
# ---------------------------------------------------------------------------

def same_label_edge_fraction(tree, labels: np.ndarray) -> float:
    """Fraction of MST edges connecting nodes with the same label."""
    same = sum(
        1 for s, t in zip(tree.edges[:, 0], tree.edges[:, 1])
        if labels[s] == labels[t]
    )
    return same / len(tree.edges)


def boundary_edge_fraction(tree, labels: np.ndarray) -> float:
    """Fraction of MST edges at label boundaries (different labels)."""
    return 1.0 - same_label_edge_fraction(tree, labels)


def subtree_purity(tree, labels: np.ndarray, min_size: int = 10) -> float:
    """Average label purity of subtrees rooted at each node."""
    from tmap.graph.analysis import subtree_purity as _sp
    purities = _sp(tree, labels, min_size=min_size)
    valid = purities[purities >= 0]
    return float(valid.mean()) if len(valid) > 0 else 0.0


def path_label_transitions(
    tree, labels: np.ndarray, n_paths: int = N_PATHS,
) -> float:
    """Average number of label changes along random tree paths."""
    rng = np.random.default_rng(SEED)
    n = tree.n_nodes
    transitions = []

    for _ in range(n_paths):
        a, b = rng.integers(0, n, size=2)
        if a == b:
            continue
        path = tree.path(a, b)
        if len(path) < 2:
            continue
        path_labels = labels[path]
        changes = sum(
            1 for i in range(1, len(path_labels))
            if path_labels[i] != path_labels[i - 1]
        )
        transitions.append(changes / (len(path) - 1))

    return float(np.mean(transitions)) if transitions else 0.0


def path_monotonicity(
    tree, labels: np.ndarray, n_paths: int = N_PATHS,
) -> float:
    """Spearman correlation of tree-path position vs ordinal label.

    Meaningful for ordinal labels (digits 0-9, pseudotime).
    Higher = better ordering along paths.
    """
    rng = np.random.default_rng(SEED)
    n = tree.n_nodes
    correlations = []

    for _ in range(n_paths):
        a, b = rng.integers(0, n, size=2)
        if a == b:
            continue
        path = tree.path(a, b)
        if len(path) < 5:
            continue
        path_labels = labels[path].astype(float)
        positions = np.arange(len(path), dtype=float)
        rho, _ = spearmanr(positions, path_labels)
        if not np.isnan(rho):
            correlations.append(abs(rho))

    return float(np.mean(correlations)) if correlations else 0.0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_dataset(dataset: str, n: int) -> dict:
    from tmap import TMAP

    print(f"\n{'=' * 70}")
    print(f" Analysis quality: dataset={dataset}  n={n:,}")
    print(f"{'=' * 70}")

    if dataset == "mnist":
        X, labels = load_mnist(n)
        metric = "cosine"
        is_ordinal = True
    elif dataset == "chembl":
        X, labels = load_chembl_with_scaffolds(n)
        metric = "jaccard"
        is_ordinal = False
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_classes = len(np.unique(labels))
    print(
        f"  Data: {X.shape}, metric={metric}, "
        f"classes={n_classes}, ordinal={is_ordinal}"
    )

    # Fit TMAP
    print("  Fitting TMAP...", end=" ", flush=True)
    t0 = time.perf_counter()
    model = TMAP(n_neighbors=K, metric=metric, seed=SEED)
    model.fit(X)
    fit_s = time.perf_counter() - t0
    print(f"{fit_s:.1f}s")

    tree = model.tree_

    # Compute metrics
    print("  Computing metrics...", flush=True)

    slef = same_label_edge_fraction(tree, labels)
    print(f"    same_label_edge_fraction: {slef:.4f}")

    bef = boundary_edge_fraction(tree, labels)
    print(f"    boundary_edge_fraction:   {bef:.4f}")

    sp = subtree_purity(tree, labels, min_size=10)
    print(f"    subtree_purity (min=10):  {sp:.4f}")

    plt = path_label_transitions(tree, labels, n_paths=N_PATHS)
    print(f"    path_label_transitions:   {plt:.4f}")

    row = {
        "dataset": dataset,
        "n": n,
        "metric": metric,
        "k": K,
        "seed": SEED,
        "n_classes": n_classes,
        "is_ordinal": is_ordinal,
        "fit_s": round(fit_s, 2),
        "same_label_edge_fraction": round(slef, 4),
        "boundary_edge_fraction": round(bef, 4),
        "subtree_purity_min10": round(sp, 4),
        "path_label_transitions": round(plt, 4),
    }

    if is_ordinal:
        pm = path_monotonicity(tree, labels, n_paths=N_PATHS)
        print(f"    path_monotonicity:        {pm:.4f}")
        row["path_monotonicity"] = round(pm, 4)

    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def save(rows: list[dict], filename: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}")


def main():
    p = argparse.ArgumentParser(
        description="Analysis quality benchmark",
    )
    p.add_argument(
        "--dataset", choices=["mnist", "chembl", "all"],
        default="all",
    )
    p.add_argument("--n-mnist", type=int, default=50_000)
    p.add_argument("--n-chembl", type=int, default=50_000)
    args = p.parse_args()

    datasets = (
        ["mnist", "chembl"] if args.dataset == "all"
        else [args.dataset]
    )

    all_rows = []
    for ds in datasets:
        n = args.n_mnist if ds == "mnist" else args.n_chembl
        row = bench_dataset(ds, n)
        all_rows.append(row)

    save(all_rows, "analysis_quality.csv")


if __name__ == "__main__":
    main()
