#!/usr/bin/env python
"""Generate paper figures from benchmark CSV results.

Usage:
    python scripts/plot_paper.py            # all figures
    python scripts/plot_paper.py table2     # TMAP2 vs UMAP only
    python scripts/plot_paper.py table1     # TMAP2 vs TMAP1 only
    python scripts/plot_paper.py table3     # USearch latency only
    python scripts/plot_paper.py scale      # scaling figure only

Reads from benchmarks/results_paper/*.csv
Writes to   benchmarks/results_paper/figures/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "benchmarks" / "results_paper"
FIGURES = RESULTS / "figures"

# Style
COLORS = {
    "tmap2": "#2563eb",
    "tmap2_lsh_d128": "#7c3aed",
    "tmap2_lsh_d512": "#a855f7",
    "umap": "#dc2626",
    "tmap1": "#f59e0b",
    "tmap1_d128": "#f59e0b",
    "tmap1_d512": "#ea580c",
    "cosine": "#2563eb",
    "euclidean": "#16a34a",
}
MARKERS = {
    "tmap2": "o", "tmap2_lsh_d128": "p", "tmap2_lsh_d512": "P",
    "umap": "s", "tmap1": "D", "tmap1_d128": "D", "tmap1_d512": "^",
}
LABELS = {
    "tmap2": "TMAP2 (USearch)",
    "tmap2_lsh_d128": "TMAP2 LSH (d=128)",
    "tmap2_lsh_d512": "TMAP2 LSH (d=512)",
    "umap": "UMAP",
    "tmap1": "TMAP1 (LSH)",
    "tmap1_d128": "TMAP1 C++ (d=128)",
    "tmap1_d512": "TMAP1 C++ (d=512)",
}


def _setup():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.1,
    })


# ---------------------------------------------------------------------------
# Table 2: TMAP2 vs UMAP
# ---------------------------------------------------------------------------

def plot_table2():
    csv = RESULTS / "table2_tmap2_vs_umap.csv"
    if not csv.exists():
        print(f"  Skipping table2 — {csv} not found")
        return
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Runtime
    ax = axes[0]
    for method in ["tmap2", "umap"]:
        d = df[df.method == method]
        ax.plot(d.n, d.runtime_s, marker=MARKERS[method], color=COLORS[method],
                label=LABELS[method], linewidth=2, markersize=7)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime")
    ax.legend()
    ax.set_xscale("log")

    # Trustworthiness
    ax = axes[1]
    for method in ["tmap2", "umap"]:
        d = df[(df.method == method) & (df.trustworthiness != "")]
        if not d.empty:
            ax.plot(d.n, d.trustworthiness.astype(float), marker=MARKERS[method],
                    color=COLORS[method], label=LABELS[method], linewidth=2, markersize=7)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Trustworthiness")
    ax.set_title("Trustworthiness")
    ax.legend()
    ax.set_xscale("log")
    ax.set_ylim(0.85, 1.0)

    # kNN Preservation
    ax = axes[2]
    for method in ["tmap2", "umap"]:
        d = df[(df.method == method) & (df.knn_preservation != "")]
        if not d.empty:
            ax.plot(d.n, d.knn_preservation.astype(float), marker=MARKERS[method],
                    color=COLORS[method], label=LABELS[method], linewidth=2, markersize=7)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("kNN preservation")
    ax.set_title("k-NN Preservation (k=20)")
    ax.legend()
    ax.set_xscale("log")

    fig.suptitle("TMAP2 vs UMAP — MNIST, cosine metric", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES / "table2_tmap2_vs_umap.png"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Table 1: TMAP2 vs TMAP1
# ---------------------------------------------------------------------------

def plot_table1():
    csv = RESULTS / "table1_tmap2_vs_tmap1.csv"
    if not csv.exists():
        print(f"  Skipping table1 — {csv} not found")
        return
    df = pd.read_csv(csv)

    has_recall = "recall_at_20" in df.columns
    ncols = 3 if has_recall else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    methods = sorted(df.method.unique(), key=lambda x: x)

    # Runtime
    ax = axes[0]
    for method in methods:
        d = df[df.method == method]
        ax.plot(d.n, d.runtime_s, marker=MARKERS.get(method, "o"),
                color=COLORS.get(method, "#666"), label=LABELS.get(method, method),
                linewidth=2, markersize=7)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime")
    ax.legend()
    ax.set_xscale("log")

    # Memory
    ax = axes[1]
    for method in methods:
        d = df[df.method == method]
        ax.plot(d.n, d.peak_rss_mb, marker=MARKERS.get(method, "o"),
                color=COLORS.get(method, "#666"), label=LABELS.get(method, method),
                linewidth=2, markersize=7)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("Memory")
    ax.legend()
    ax.set_xscale("log")

    # Recall
    if has_recall:
        ax = axes[2]
        for method in methods:
            d = df[(df.method == method) & (df.recall_at_20 != "")]
            if not d.empty:
                ax.plot(d.n, d.recall_at_20.astype(float),
                        marker=MARKERS.get(method, "o"),
                        color=COLORS.get(method, "#666"),
                        label=LABELS.get(method, method),
                        linewidth=2, markersize=7)
        ax.set_xlabel("Dataset size")
        ax.set_ylabel("Recall@20")
        ax.set_title("kNN Recall (vs exact)")
        ax.legend()
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)

    fig.suptitle("TMAP2 vs TMAP (original) — Jaccard, binary fingerprints", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES / "table1_tmap2_vs_tmap1.png"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Table 3: USearch latency
# ---------------------------------------------------------------------------

def plot_table3():
    csv = RESULTS / "table3_usearch_latency.csv"
    if not csv.exists():
        print(f"  Skipping table3 — {csv} not found")
        return
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Build time
    ax = axes[0]
    for metric in ["cosine", "euclidean"]:
        d = df[df.metric == metric]
        ax.plot(d.n, d.build_s, marker="o", color=COLORS[metric],
                label=metric, linewidth=2, markersize=7)
    ax.set_xlabel("Index size (points)")
    ax.set_ylabel("Build time (s)")
    ax.set_title("Index Build Time")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Query latency
    ax = axes[1]
    for metric in ["cosine", "euclidean"]:
        d = df[df.metric == metric]
        ax.plot(d.n, d.query_1nn_ms, marker="o", color=COLORS[metric],
                label=f"{metric} (1-NN)", linewidth=2, markersize=7)
        ax.plot(d.n, d.query_20nn_ms, marker="s", color=COLORS[metric],
                label=f"{metric} (20-NN)", linewidth=2, markersize=7, linestyle="--")
    ax.set_xlabel("Index size (points)")
    ax.set_ylabel("Query latency (ms/query)")
    ax.set_title("Query Latency")
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Recall
    ax = axes[2]
    for metric in ["cosine", "euclidean"]:
        d = df[(df.metric == metric) & (df.recall_at_20 != "")]
        if not d.empty:
            ax.plot(d.n, d.recall_at_20.astype(float), marker="o", color=COLORS[metric],
                    label=metric, linewidth=2, markersize=7)
    ax.set_xlabel("Index size (points)")
    ax.set_ylabel("Recall@20")
    ax.set_title("kNN Recall (vs exact)")
    ax.legend()
    ax.set_xscale("log")
    ax.set_ylim(0.9, 1.01)

    fig.suptitle("USearch HNSW Index — d=768", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES / "table3_usearch_latency.png"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Scaling figure
# ---------------------------------------------------------------------------

def plot_scale():
    csv = RESULTS / "figure_scaling.csv"
    if not csv.exists():
        print(f"  Skipping scale — {csv} not found")
        return
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Runtime
    ax = axes[0]
    for method in ["tmap2", "umap"]:
        d = df[df.method == method]
        ax.plot(d.n, d.runtime_s, marker=MARKERS[method], color=COLORS[method],
                label=LABELS[method], linewidth=2, markersize=7)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime vs Scale")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Memory
    ax = axes[1]
    for method in ["tmap2", "umap"]:
        d = df[df.method == method]
        ax.plot(d.n, d.peak_rss_mb, marker=MARKERS[method], color=COLORS[method],
                label=LABELS[method], linewidth=2, markersize=7)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("Memory vs Scale")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.suptitle("TMAP2 vs UMAP — Scaling (synthetic, cosine)", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES / "figure_scaling.png"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SUITES = {
    "table1": plot_table1,
    "table2": plot_table2,
    "table3": plot_table3,
    "scale": plot_scale,
}


def main():
    _setup()
    targets = sys.argv[1:] or list(SUITES.keys())
    for name in targets:
        if name not in SUITES:
            print(f"Unknown suite: {name}. Choose from: {', '.join(SUITES)}")
            sys.exit(1)
        print(f"Plotting {name}...")
        SUITES[name]()
    print("Done.")


if __name__ == "__main__":
    main()
