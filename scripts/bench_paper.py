#!/usr/bin/env python
"""Paper benchmarks for TMAP2.

Generates four deliverables:
  Table 1  — TMAP2 vs TMAP1 (Jaccard, binary fingerprints)
  Table 2  — TMAP2 vs UMAP  (cosine, MNIST)
  Figure   — Scaling curves  (time + memory vs N)
  Table 3  — USearch index query latency

Usage:
  python scripts/bench_paper.py umap   [--sizes ...] [--repeats 3]
  python scripts/bench_paper.py tmap1  [--sizes ...] [--repeats 3]
  python scripts/bench_paper.py index  [--sizes ...]
  python scripts/bench_paper.py scale  [--sizes ...] [--repeats 3]

Results saved to benchmarks/results_paper/
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "benchmarks" / "workers" / "bench_worker.py"
RESULTS_DIR = ROOT / "benchmarks" / "results_paper"
CACHE_DIR = ROOT / "benchmarks" / "cache"

SEED = 42
K = 20  # n_neighbors for all methods


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_mnist(n: int | None = None) -> np.ndarray:
    """MNIST-784 (70K x 784), float32, cached to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "mnist_784.npz"
    if cache.exists():
        X = np.load(cache)["X"]
    else:
        from sklearn.datasets import fetch_openml

        X, _ = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
        )
        X = X.astype(np.float32)
        np.savez_compressed(cache, X=X)
    if n is not None and n < X.shape[0]:
        X = X[np.random.default_rng(SEED).choice(X.shape[0], n, replace=False)]
    return X


def make_synthetic(n: int, d: int = 784, clusters: int = 50) -> np.ndarray:
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=n, n_features=d, centers=clusters, random_state=SEED)
    return X.astype(np.float32)


def make_binary(n: int, d: int = 2048, density: float = 0.1) -> np.ndarray:
    return (np.random.default_rng(SEED).random((n, d)) < density).astype(np.uint8)


def load_molecular_fps(n: int | None = None, n_bits: int = 2048) -> np.ndarray:
    """Morgan fingerprints: ChEMBL 200K for n<=200K, 20M SMILES beyond."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Use ChEMBL 200K if sufficient
    chembl_path = ROOT / "data" / "chembl" / "chembl_200k_morgan.npy"
    if n is not None and n <= 200_000 and chembl_path.exists():
        X = np.load(chembl_path)
        if n < X.shape[0]:
            X = X[np.random.default_rng(SEED).choice(X.shape[0], n, replace=False)]
        return X.astype(np.uint8)

    # For larger n, use 20M_smiles.txt
    cache = CACHE_DIR / f"smiles_morgan_{n}_{n_bits}.npy"
    if cache.exists():
        return np.load(cache).astype(np.uint8)

    smiles_path = ROOT / "data" / "20M_smiles.txt"
    if not smiles_path.exists():
        raise FileNotFoundError(
            f"Need {smiles_path} for n={n:,}. "
            "Download or symlink your SMILES file."
        )

    from tmap import fingerprints_from_smiles

    with open(smiles_path) as f:
        smiles = [line.strip() for _, line in zip(range(n), f) if line.strip()]
    if len(smiles) < n:
        raise ValueError(f"Only {len(smiles)} SMILES available, requested {n}")

    print(f"  Generating {n:,} Morgan FP (cached)...", end=" ", flush=True)
    import time as _t
    t0 = _t.perf_counter()
    X = fingerprints_from_smiles(smiles[:n], fp_type="morgan", n_bits=n_bits)
    print(f"done ({_t.perf_counter() - t0:.1f}s)")
    np.save(cache, X.astype(np.uint8))
    return X.astype(np.uint8)


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_worker(method: str, X: np.ndarray, metric: str = "cosine", n_perm: int = 128) -> dict:
    """Run one embedding in an isolated subprocess. Returns dict with
    embedding (n,2), runtime_s, peak_rss_mb."""
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.npz"
        out = Path(tmp) / "out.npz"
        np.savez_compressed(inp, X=X)

        cmd = [
            sys.executable,
            str(WORKER),
            "--method", method,
            "--input", str(inp),
            "--output", str(out),
            "--n-neighbors", str(K),
            "--metric", metric,
            "--n-perm", str(n_perm),
            "--seed", str(SEED),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{method} worker failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}"
            )

        data = np.load(out)
        return {
            "embedding": np.array(data["embedding"]),
            "runtime_s": float(data["runtime_s"]),
            "peak_rss_mb": float(data["peak_rss_mb"]),
        }


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def knn_preservation(X: np.ndarray, emb: np.ndarray, k: int = K, metric: str = "cosine") -> float:
    """Fraction of k-NN in original space also present in 2D embedding."""
    from sklearn.neighbors import NearestNeighbors

    idx_orig = (
        NearestNeighbors(n_neighbors=k, metric=metric, algorithm="brute")
        .fit(X)
        .kneighbors(X, return_distance=False)
    )
    idx_emb = (
        NearestNeighbors(n_neighbors=k, metric="euclidean")
        .fit(emb)
        .kneighbors(emb, return_distance=False)
    )
    n = len(X)
    preserved = sum(len(set(idx_orig[i]) & set(idx_emb[i])) for i in range(n))
    return preserved / (n * k)


def trustworthiness_safe(
    X: np.ndarray, emb: np.ndarray, k: int = K, metric: str = "cosine", max_n: int = 5000
) -> float:
    """Trustworthiness, subsampled if n > max_n to avoid O(n^2) memory."""
    from sklearn.manifold import trustworthiness

    if len(X) > max_n:
        idx = np.random.default_rng(SEED).choice(len(X), max_n, replace=False)
        X, emb = X[idx], emb[idx]
    return float(trustworthiness(X, emb, n_neighbors=k, metric=metric))


def compute_exact_knn(X: np.ndarray, k: int = K, metric: str = "cosine") -> np.ndarray:
    """Precompute exact k-NN indices for X (reused across methods)."""
    from sklearn.neighbors import NearestNeighbors

    return (
        NearestNeighbors(n_neighbors=k, metric=metric, algorithm="brute")
        .fit(X)
        .kneighbors(X, return_distance=False)
    )


def knn_preservation_precomputed(
    idx_exact: np.ndarray, emb: np.ndarray, k: int = K
) -> float:
    """kNN preservation given precomputed exact neighbors."""
    from sklearn.neighbors import NearestNeighbors

    idx_emb = (
        NearestNeighbors(n_neighbors=k, metric="euclidean")
        .fit(emb)
        .kneighbors(emb, return_distance=False)
    )
    n = len(emb)
    preserved = sum(len(set(idx_exact[i]) & set(idx_emb[i])) for i in range(n))
    return preserved / (n * k)


# ---------------------------------------------------------------------------
# Suite: TMAP2 vs UMAP  (Table 2 + Figure)
# ---------------------------------------------------------------------------

def bench_umap(
    sizes: list[int], repeats: int, quality: bool = True, output: str = "table2_tmap2_vs_umap.csv"
) -> list[dict]:
    results = []

    for n in sizes:
        is_mnist = n <= 70_000
        X = load_mnist(n) if is_mnist else make_synthetic(n, d=784)
        source = "mnist" if is_mnist else "synthetic"

        print(f"\n{'=' * 60}")
        print(f" n={n:>8,}  data={source}  metric=cosine  k={K}")
        print(f"{'=' * 60}")

        # Precompute exact kNN once for quality metrics (up to 100K)
        idx_exact = None
        if quality and n <= 100_000:
            print("  Precomputing exact kNN...", end=" ", flush=True)
            t0 = time.perf_counter()
            idx_exact = compute_exact_knn(X, k=K, metric="cosine")
            print(f"done ({time.perf_counter() - t0:.1f}s)")

        for method in ["tmap2", "umap"]:
            runtimes, memories = [], []
            emb = None

            for rep in range(repeats):
                print(f"  {method:>6} [{rep + 1}/{repeats}] ", end="", flush=True)
                r = run_worker(method, X, metric="cosine")
                runtimes.append(r["runtime_s"])
                memories.append(r["peak_rss_mb"])
                emb = r["embedding"]
                print(f"{r['runtime_s']:>8.1f}s  {r['peak_rss_mb']:>7.0f} MB")

            tw, kp = "", ""
            if quality and n <= 100_000 and emb is not None:
                print(f"  {'':>6} quality: ", end="", flush=True)
                tw = trustworthiness_safe(X, emb, k=K, metric="cosine")
                kp = knn_preservation_precomputed(idx_exact, emb, k=K)
                print(f"trust={tw:.4f}  kNN_pres={kp:.4f}")
                tw = round(tw, 4)
                kp = round(kp, 4)

            results.append({
                "n": n,
                "data": source,
                "method": method,
                "runtime_s": round(float(np.median(runtimes)), 2),
                "runtime_std": round(float(np.std(runtimes)), 2),
                "peak_rss_mb": round(float(np.median(memories))),
                "trustworthiness": tw,
                "knn_preservation": kp,
            })

    _save(results, output)
    _print_table(results)
    return results


# ---------------------------------------------------------------------------
# Suite: TMAP2 vs TMAP1  (Table 1)
# ---------------------------------------------------------------------------

def _recall_tmap2_jaccard(X_bin: np.ndarray, exact_idx: np.ndarray, k: int = K) -> float:
    """kNN recall for TMAP2's USearch Jaccard index."""
    from tmap.index.usearch_index import USearchIndex

    idx = USearchIndex()
    idx.build_from_binary(X_bin)
    knn = idx.query_knn(k)
    pred = knn.indices  # (n, k)

    n = len(X_bin)
    total = sum(len(set(pred[i]) & set(exact_idx[i])) for i in range(n))
    return total / (n * k)


def _recall_tmap2_lsh_jaccard(
    X_bin: np.ndarray, exact_idx: np.ndarray, k: int = K, n_perm: int = 128
) -> float:
    """kNN recall for TMAP2's MinHash+LSH Numba path."""
    from tmap.index.encoders.minhash import MinHash
    from tmap.index.lsh_forest import LSHForest

    encoder = MinHash(num_perm=n_perm, seed=SEED)
    sigs = encoder.batch_from_binary_array(X_bin)

    l = max(8, n_perm // 4)
    forest = LSHForest(d=n_perm, l=l)
    forest.batch_add(sigs)
    forest.index()
    knn = forest.get_knn_graph(k=k, kc=50)
    pred = knn.indices  # (n, k)

    n = len(X_bin)
    total = sum(len(set(pred[i]) & set(exact_idx[i])) for i in range(n))
    return total / (n * k)


def _recall_tmap1_jaccard(
    X_bin: np.ndarray, exact_idx: np.ndarray, k: int = K, n_perm: int = 128,
) -> float:
    """kNN recall for original TMAP's MinHash+LSH."""
    import glob as globmod
    import importlib.util
    import sysconfig

    so_dir = Path(sysconfig.get_paths()["purelib"]) / "tmap"
    candidates = globmod.glob(str(so_dir / "_tmap*.so"))
    if not candidates:
        raise ImportError("tmap-silicon C extension not found")

    spec = importlib.util.spec_from_file_location("_tmap", candidates[0])
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    n, _d = X_bin.shape
    # Use same l as TMAP2 for fair comparison: l = max(8, d//4)
    l = max(8, n_perm // 4)
    enc = tm.Minhash(n_perm)
    lf = tm.LSHForest(n_perm, l)

    for i in range(n):
        fp = tm.VectorUchar(X_bin[i].tolist())
        lf.add(enc.from_binary_array(fp))
    lf.index()

    # get_knn_graph writes into pre-allocated vectors
    v_from = tm.VectorUint()
    v_to = tm.VectorUint()
    v_weight = tm.VectorFloat()
    lf.get_knn_graph(v_from, v_to, v_weight, k, 10)

    from collections import defaultdict
    adj = defaultdict(set)
    for src, dst in zip(v_from, v_to):
        adj[src].add(dst)

    total = 0
    for i in range(n):
        total += len(adj[i] & set(exact_idx[i]))
    return total / (n * k)


def bench_tmap1(sizes: list[int], repeats: int) -> list[dict]:
    results = []

    for n in sizes:
        X_bin = load_molecular_fps(n)
        source = "molecular"
        print(f"\n{'=' * 60}")
        print(f" n={n:>8,}  data={source}  metric=jaccard  k={K}")
        print(f"{'=' * 60}")

        # Compute exact kNN for recall (subsample if too large)
        recall_n = min(n, 10_000)
        X_recall = X_bin[:recall_n]
        print(f"  Computing exact kNN (n={recall_n:,})...", end=" ", flush=True)
        t0 = time.perf_counter()
        from sklearn.neighbors import NearestNeighbors
        exact_idx = (
            NearestNeighbors(n_neighbors=K + 1, metric="jaccard", algorithm="brute")
            .fit(X_recall)
            .kneighbors(X_recall, return_distance=False)
        )
        # Exclude self
        exact_idx_clean = np.zeros((recall_n, K), dtype=np.int32)
        for i in range(recall_n):
            others = [j for j in exact_idx[i] if j != i][:K]
            exact_idx_clean[i, :len(others)] = others
        exact_idx = exact_idx_clean
        print(f"done ({time.perf_counter() - t0:.1f}s)")

        # Measure recall for each index backend
        print(f"  TMAP2 (USearch)   recall@{K}: ", end="", flush=True)
        r_usearch = _recall_tmap2_jaccard(X_recall, exact_idx, k=K)
        print(f"{r_usearch:.4f}")

        # TMAP2 LSH path (Numba JIT) recall
        recalls_lsh2 = {}
        for d_perm in [128, 512]:
            print(f"  TMAP2 LSH (d={d_perm:>3}) recall@{K}: ", end="", flush=True)
            r_lsh2 = _recall_tmap2_lsh_jaccard(X_recall, exact_idx, k=K, n_perm=d_perm)
            recalls_lsh2[d_perm] = r_lsh2
            print(f"{r_lsh2:.4f}")

        # TMAP1 (original C++) recall
        recalls_lsh1 = {}
        for d_perm in [128, 512]:
            print(f"  TMAP1 (LSH d={d_perm:>3}) recall@{K}: ", end="", flush=True)
            r_lsh1 = _recall_tmap1_jaccard(X_recall, exact_idx, k=K, n_perm=d_perm)
            recalls_lsh1[d_perm] = r_lsh1
            print(f"{r_lsh1:.4f}")

        # All methods: tmap2 (USearch), tmap2_lsh (Numba), tmap1 (original C++)
        configs = [
            ("tmap2", "tmap2", "jaccard", 128, r_usearch),
            ("tmap2_lsh_d128", "tmap2_lsh", "jaccard", 128, recalls_lsh2[128]),
            ("tmap2_lsh_d512", "tmap2_lsh", "jaccard", 512, recalls_lsh2[512]),
            ("tmap1_d128", "tmap1", "jaccard", 128, recalls_lsh1[128]),
            ("tmap1_d512", "tmap1", "jaccard", 512, recalls_lsh1[512]),
        ]

        for label, method, metric, d_perm, recall in configs:
            runtimes, memories = [], []

            for rep in range(repeats):
                print(f"  {label:>12} [{rep + 1}/{repeats}] ", end="", flush=True)
                r = run_worker(method, X_bin, metric=metric, n_perm=d_perm)
                runtimes.append(r["runtime_s"])
                memories.append(r["peak_rss_mb"])
                print(f"{r['runtime_s']:>8.1f}s  {r['peak_rss_mb']:>7.0f} MB")

            results.append({
                "n": n,
                "method": label,
                "runtime_s": round(float(np.median(runtimes)), 2),
                "runtime_std": round(float(np.std(runtimes)), 2),
                "peak_rss_mb": round(float(np.median(memories))),
                "recall_at_20": round(recall, 4),
            })

    _save(results, "table1_tmap2_vs_tmap1.csv")
    _print_table(results)
    return results


# ---------------------------------------------------------------------------
# Suite: USearch index latency  (Table 3)
# ---------------------------------------------------------------------------

def bench_index(sizes: list[int], mmap_mode: bool = False) -> list[dict]:
    """USearch index latency benchmark.

    With --mmap: build index, save to disk, then query via memory-mapped view().
    This allows benchmarking indexes that don't fit in RAM (e.g., 100M vectors).
    Data is generated in chunks to avoid materializing the full matrix at once.
    """
    from usearch.index import Index as UsearchIndex

    results = []
    d = 768  # typical embedding dim (BERT/ESM)
    n_queries = 100

    for n in sizes:
        mode_label = "mmap" if mmap_mode else "ram"
        print(f"\n--- n={n:,}  d={d}  mode={mode_label} ---")
        rng = np.random.default_rng(SEED)
        queries = rng.standard_normal((n_queries, d)).astype(np.float32)

        for metric in ["cosine", "euclidean"]:
            usearch_metric = "cos" if metric == "cosine" else "l2sq"

            if mmap_mode:
                import tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    idx_path = Path(tmp) / "index.usearch"

                    # Build in chunks to limit peak memory
                    idx = UsearchIndex(ndim=d, metric=usearch_metric)
                    chunk_size = min(n, 500_000)
                    t0 = time.perf_counter()
                    offset = 0
                    while offset < n:
                        batch_n = min(chunk_size, n - offset)
                        batch = rng.standard_normal((batch_n, d)).astype(np.float32)
                        keys = np.arange(offset, offset + batch_n, dtype=np.int64)
                        idx.add(keys, batch)
                        offset += batch_n
                    build_s = time.perf_counter() - t0

                    # Save and reopen as mmap
                    idx.save(str(idx_path))
                    del idx

                    idx_mmap = UsearchIndex(ndim=d, metric=usearch_metric)
                    idx_mmap.view(str(idx_path))

                    # Query latency on mmap index
                    times_1, times_20 = [], []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        idx_mmap.search(queries, 1)
                        times_1.append((time.perf_counter() - t0) / n_queries * 1000)
                    for _ in range(3):
                        t0 = time.perf_counter()
                        idx_mmap.search(queries, 20)
                        times_20.append((time.perf_counter() - t0) / n_queries * 1000)

                    q1_ms = float(np.median(times_1))
                    q20_ms = float(np.median(times_20))
                    del idx_mmap
            else:
                # In-memory mode (original path)
                X = rng.standard_normal((n, d)).astype(np.float32)

                idx = UsearchIndex(ndim=d, metric=usearch_metric)
                t0 = time.perf_counter()
                idx.add(np.arange(n, dtype=np.int64), X)
                build_s = time.perf_counter() - t0

                times_1, times_20 = [], []
                for _ in range(3):
                    t0 = time.perf_counter()
                    idx.search(queries, 1)
                    times_1.append((time.perf_counter() - t0) / n_queries * 1000)
                for _ in range(3):
                    t0 = time.perf_counter()
                    idx.search(queries, 20)
                    times_20.append((time.perf_counter() - t0) / n_queries * 1000)

                q1_ms = float(np.median(times_1))
                q20_ms = float(np.median(times_20))
                del idx, X

            # Recall@20 (expensive — only for n <= 100K in RAM mode)
            recall = ""
            if not mmap_mode and n <= 100_000:
                from sklearn.neighbors import NearestNeighbors

                X_recall = rng.standard_normal((n, d)).astype(np.float32)
                sample = min(1000, n)
                Q = X_recall[:sample]

                idx_r = UsearchIndex(ndim=d, metric=usearch_metric)
                idx_r.add(np.arange(n, dtype=np.int64), X_recall)
                results_r = idx_r.search(Q, 21)
                pred_idx = np.array(results_r.keys)

                exact_idx = (
                    NearestNeighbors(n_neighbors=21, metric=metric, algorithm="brute")
                    .fit(X_recall)
                    .kneighbors(Q, return_distance=False)
                )
                total = 0
                for i in range(sample):
                    pred = set(int(j) for j in pred_idx[i]) - {i}
                    truth = set(int(j) for j in exact_idx[i]) - {i}
                    total += len(pred & truth)
                recall = round(total / (sample * 20), 4)
                del X_recall, idx_r

            line = (
                f"  {metric:>10}: build={build_s:>6.2f}s  "
                f"1-NN={q1_ms:>7.3f}ms/q  20-NN={q20_ms:>7.3f}ms/q"
            )
            if recall:
                line += f"  recall@20={recall:.4f}"
            print(line)

            results.append({
                "n": n,
                "d": d,
                "metric": metric,
                "mode": mode_label,
                "build_s": round(build_s, 3),
                "query_1nn_ms": round(q1_ms, 4),
                "query_20nn_ms": round(q20_ms, 4),
                "recall_at_20": recall,
            })

    _save(results, "table3_usearch_latency.csv")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(rows: list[dict], filename: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {path}")


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'Method':>8} {'n':>8} {'Runtime':>10} {'Memory':>10}", end="")
    has_quality = any(r.get("trustworthiness") for r in rows)
    if has_quality:
        print(f" {'Trust':>8} {'kNN':>8}", end="")
    print()
    print("-" * (50 + (18 if has_quality else 0)))
    for r in rows:
        print(
            f"{r['method']:>8} {r['n']:>8,} {r['runtime_s']:>8.1f}s {r['peak_rss_mb']:>8.0f}MB",
            end="",
        )
        if has_quality:
            tw = f"{r['trustworthiness']:.4f}" if r.get("trustworthiness") else "    -"
            kp = f"{r['knn_preservation']:.4f}" if r.get("knn_preservation") else "    -"
            print(f" {tw:>8} {kp:>8}", end="")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="TMAP2 paper benchmarks")
    sub = p.add_subparsers(dest="suite", required=True)

    u = sub.add_parser("umap", help="Table 2: TMAP2 vs UMAP")
    u.add_argument("--sizes", default="10000,50000,100000,200000,500000,1000000")
    u.add_argument("--repeats", type=int, default=3)

    t = sub.add_parser("tmap1", help="Table 1: TMAP2 vs TMAP1")
    t.add_argument("--sizes", default="10000,50000,100000,200000,500000,1000000,2000000,5000000")
    t.add_argument("--repeats", type=int, default=3)

    i = sub.add_parser("index", help="Table 3: USearch query latency")
    i.add_argument(
        "--sizes",
        default="10000,100000,500000,1000000,5000000,10000000,50000000,100000000",
    )
    i.add_argument(
        "--mmap", action="store_true",
        help="Use mmap mode: build, save, query via view()",
    )

    s = sub.add_parser("scale", help="Figure: scaling curves (time + memory only)")
    s.add_argument("--sizes", default="10000,50000,100000,200000,500000,1000000")
    s.add_argument("--repeats", type=int, default=3)

    args = p.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]

    if args.suite == "umap":
        bench_umap(sizes, args.repeats, quality=True)
    elif args.suite == "tmap1":
        bench_tmap1(sizes, args.repeats)
    elif args.suite == "index":
        bench_index(sizes, mmap_mode=getattr(args, "mmap", False))
    elif args.suite == "scale":
        bench_umap(sizes, args.repeats, quality=False, output="figure_scaling.csv")


if __name__ == "__main__":
    main()
