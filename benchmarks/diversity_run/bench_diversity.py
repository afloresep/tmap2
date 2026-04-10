#!/usr/bin/env python
"""Benchmark final-tree quality across diverse chemical datasets.

Adapted from scripts/bench_final_map_quality.py for arbitrary .npy datasets.
Instead of hardcoded ChEMBL paths, accepts --datasets as a comma-separated list
of .npy fingerprint files.  For each dataset x size combination, it subsamples
and runs all backend/profile configs.

Evaluation is identical to the original:
  - recall@20 against exact Jaccard neighbors
  - exact MST tree weight
  - topological nearest-neighbor quality (tree hop curve)

Each backend/profile runs in a subprocess for clean memory measurement.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from tmap.graph.types import Tree

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = Path(__file__).resolve().parent / "cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SEED = 42
K = 20
KC = 50
DEFAULT_QUERIES = 100
DEFAULT_EXACT_CHUNK_SIZE = 50_000
DEFAULT_EDGE_CHUNK_SIZE = 50_000

BYTE_POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)

USEARCH_PROFILES = {
    "low": {"connectivity": 16, "expansion_add": 128, "expansion_search": 64},
    "medium": {"connectivity": 32, "expansion_add": 128, "expansion_search": 100},
    "high": {"connectivity": 32, "expansion_add": 256, "expansion_search": 200},
}

LSH_PROFILES = {
    "low": {"n_perm": 256, "l": 128},
    "medium": {"n_perm": 512, "l": 256},
    "high": {"n_perm": 1024, "l": 512},
}


# ---------------------------------------------------------------------------
# System diagnostics
# ---------------------------------------------------------------------------

def _log_system_info() -> None:
    print(f"System: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor() or 'unknown'}")
    print(f"Cores: {os.cpu_count()}")
    print(f"Python: {sys.version.split()[0]}")
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        print(f"CPU model: {line.split(':',1)[1].strip()}")
                        break
        except OSError:
            pass
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(flush=True)


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


# ---------------------------------------------------------------------------
# TMAP1 extension loader
# ---------------------------------------------------------------------------

def _load_tmap1_extension() -> Any:
    so_dir = Path(sysconfig.get_paths()["purelib"]) / "tmap"
    candidates = glob.glob(str(so_dir / "_tmap*.so"))
    if not candidates:
        raise ImportError(
            "TMAP1 C++ extension not found in site-packages/tmap. "
            "Install it with `pip install tmap-silicon`."
        )
    spec = importlib.util.spec_from_file_location("_tmap", candidates[0])
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load TMAP1 extension from {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_binary_matrix(
    data_file: Path, n: int, seed: int = SEED,
) -> tuple[np.ndarray, str]:
    """Load a .npy fingerprint matrix and subsample to *n* rows."""
    if not data_file.exists():
        raise FileNotFoundError(f"Missing dataset: {data_file}")
    X = np.load(data_file, mmap_mode="r")
    dataset_name = data_file.stem

    if n > X.shape[0]:
        raise ValueError(
            f"Requested n={n:,}, but {data_file.name} only has {X.shape[0]:,} rows."
        )
    if n == X.shape[0]:
        return np.asarray(X, dtype=np.uint8), dataset_name

    rng = np.random.default_rng(seed)
    subset = np.sort(rng.choice(X.shape[0], size=n, replace=False))
    return np.asarray(X[subset], dtype=np.uint8), f"{dataset_name}_sampled_{n}"


# ---------------------------------------------------------------------------
# Exact Jaccard ground truth
# ---------------------------------------------------------------------------

def _packed_binary(X: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(X, dtype=np.uint8), axis=1)


def _truth_cache_path(dataset: str, n: int, n_queries: int, k: int, seed: int) -> Path:
    return CACHE_DIR / f"quality_exact_jaccard_{dataset}_n{n}_q{n_queries}_k{k}_seed{seed}.npz"


def _exact_jaccard_truth(
    packed: np.ndarray, query_ids: np.ndarray, k: int, chunk_size: int,
) -> np.ndarray:
    n = packed.shape[0]
    truth = np.empty((len(query_ids), k), dtype=np.int32)

    for out_row, query_id in enumerate(query_ids):
        query = packed[query_id]
        distances = np.empty(n, dtype=np.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = packed[start:end]
            inter = BYTE_POPCOUNT[np.bitwise_and(chunk, query)].sum(axis=1, dtype=np.uint16)
            union = BYTE_POPCOUNT[np.bitwise_or(chunk, query)].sum(axis=1, dtype=np.uint16)
            block = np.empty(end - start, dtype=np.float32)
            valid = union != 0
            block[valid] = 1.0 - inter[valid] / union[valid]
            block[~valid] = 0.0
            distances[start:end] = block

        distances[query_id] = np.inf
        best = np.argpartition(distances, k - 1)[:k]
        order = np.lexsort((best, distances[best]))
        truth[out_row] = best[order].astype(np.int32, copy=False)

    return truth


def _load_or_compute_truth(
    X: np.ndarray, dataset: str, n: int, k: int, n_queries: int,
    seed: int, chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = _truth_cache_path(dataset, n, n_queries, k, seed)
    if cache_path.exists():
        cached = np.load(cache_path)
        return (
            np.asarray(cached["query_ids"], dtype=np.int32),
            np.asarray(cached["truth"], dtype=np.int32),
        )

    rng = np.random.default_rng(seed)
    query_ids = np.sort(
        rng.choice(len(X), size=min(n_queries, len(X)), replace=False)
    ).astype(np.int32)

    print(
        f"  exact Jaccard neighbors: n={n:,}, queries={len(query_ids)}, "
        f"chunk={chunk_size:,}",
        flush=True,
    )
    t0 = time.perf_counter()
    truth = _exact_jaccard_truth(_packed_binary(X), query_ids, k=k, chunk_size=chunk_size)
    print(f"    built in {time.perf_counter() - t0:.1f}s", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, query_ids=query_ids, truth=truth)
    return query_ids, truth


# ---------------------------------------------------------------------------
# Recall helpers
# ---------------------------------------------------------------------------

def _strip_self_from_batch(
    raw_indices: np.ndarray, query_ids: np.ndarray, k: int,
) -> np.ndarray:
    clean = np.full((len(query_ids), k), -1, dtype=np.int32)
    for row_idx, query_id in enumerate(query_ids):
        keep = [int(idx) for idx in raw_indices[row_idx] if int(idx) != int(query_id)]
        take = keep[:k]
        if take:
            clean[row_idx, : len(take)] = np.asarray(take, dtype=np.int32)
    return clean


def _neighbors_from_result_tuples(results: list[Any], k: int) -> np.ndarray:
    out = np.full(k, -1, dtype=np.int32)
    ids = [int(item[1]) for item in results[:k]]
    if ids:
        out[: len(ids)] = np.asarray(ids, dtype=np.int32)
    return out


def _recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    total = 0
    for row_idx in range(len(truth)):
        valid = pred[row_idx][pred[row_idx] >= 0]
        total += len(set(valid.tolist()) & set(truth[row_idx].tolist()))
    return total / (len(truth) * k)


# ---------------------------------------------------------------------------
# Tree evaluation
# ---------------------------------------------------------------------------

def _component_count(tree: Tree) -> int:
    seen = np.zeros(tree.n_nodes, dtype=bool)
    components = 0
    for node in range(tree.n_nodes):
        if seen[node]:
            continue
        components += 1
        queue: deque[int] = deque([node])
        seen[node] = True
        while queue:
            cur = queue.popleft()
            for neighbor, _ in tree._adjacency[cur]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
    return components


def _exact_tree_edge_weights(
    packed: np.ndarray, edges: np.ndarray, chunk_size: int,
) -> np.ndarray:
    weights = np.empty(len(edges), dtype=np.float32)
    if len(edges) == 0:
        return weights

    for start in range(0, len(edges), chunk_size):
        end = min(start + chunk_size, len(edges))
        src = packed[edges[start:end, 0]]
        dst = packed[edges[start:end, 1]]
        inter = BYTE_POPCOUNT[np.bitwise_and(src, dst)].sum(axis=1, dtype=np.uint16)
        union = BYTE_POPCOUNT[np.bitwise_or(src, dst)].sum(axis=1, dtype=np.uint16)
        block = np.empty(end - start, dtype=np.float32)
        valid = union != 0
        block[valid] = 1.0 - inter[valid] / union[valid]
        block[~valid] = 0.0
        weights[start:end] = block

    return weights


def _tree_hop_curve(
    tree: Tree, query_ids: np.ndarray, truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    curve = np.full((len(query_ids), truth.shape[1]), np.nan, dtype=np.float32)

    for row_idx, query_id in enumerate(query_ids):
        target_pos = {int(target): idx for idx, target in enumerate(truth[row_idx])}
        remaining = set(target_pos)
        hop = np.full(tree.n_nodes, -1, dtype=np.int32)
        queue: deque[int] = deque([int(query_id)])
        hop[int(query_id)] = 0

        while queue and remaining:
            node = queue.popleft()
            next_hop = hop[node] + 1
            for neighbor, _ in tree._adjacency[node]:
                if hop[neighbor] != -1:
                    continue
                hop[neighbor] = next_hop
                if neighbor in target_pos:
                    curve[row_idx, target_pos[neighbor]] = next_hop
                    remaining.remove(neighbor)
                    if not remaining:
                        break
                queue.append(neighbor)

    means = np.full(truth.shape[1], np.nan, dtype=np.float32)
    coverage = np.zeros(truth.shape[1], dtype=np.float32)
    for idx in range(truth.shape[1]):
        valid = np.isfinite(curve[:, idx])
        coverage[idx] = float(valid.mean())
        if np.any(valid):
            means[idx] = float(curve[valid, idx].mean())
    return means, coverage


# ---------------------------------------------------------------------------
# Worker backends (run in subprocess)
# ---------------------------------------------------------------------------

def _worker_tmap2_usearch(
    X: np.ndarray, profile: str, query_ids: np.ndarray, k: int, seed: int,
) -> dict[str, Any]:
    from tmap.index.usearch_index import USearchIndex
    from tmap.layout import LayoutConfig, layout_from_knn_graph

    params = USEARCH_PROFILES[profile]
    config = LayoutConfig()
    config.k = k

    t0 = time.perf_counter()
    index = USearchIndex(
        seed=seed,
        connectivity=params["connectivity"],
        expansion_add=params["expansion_add"],
        expansion_search=params["expansion_search"],
    )
    index.build_from_binary(X)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    knn = index.query_knn(k=k)
    knn_graph_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    x, y, s, t = layout_from_knn_graph(knn, config=config, create_mst=True)
    layout_s = time.perf_counter() - t0

    raw_pred, _ = index.query_batch(np.asarray(X[query_ids]), k=k + 1)
    pred = _strip_self_from_batch(raw_pred, query_ids, k)

    return {
        "backend": "tmap2_usearch",
        "profile": profile,
        "pred_indices": pred,
        "embedding": np.column_stack([x, y]).astype(np.float32, copy=False),
        "edges": np.column_stack(
            [s.astype(np.int32, copy=False), t.astype(np.int32, copy=False)]
        ),
        "build_s": build_s,
        "knn_graph_s": knn_graph_s,
        "pre_layout_s": build_s + knn_graph_s,
        "layout_from_knn_graph_s": layout_s,
        "total_s": build_s + knn_graph_s + layout_s,
        "peak_rss_pipeline_mb": _peak_rss_mb(),
        "peak_rss_mb": _peak_rss_mb(),
        **params,
    }


def _worker_tmap2_lsh(
    X: np.ndarray, profile: str, query_ids: np.ndarray,
    k: int, kc: int, seed: int,
) -> dict[str, Any]:
    from tmap.index.encoders.minhash import MinHash
    from tmap.index.lsh_forest import LSHForest
    from tmap.layout import LayoutConfig, layout_from_lsh_forest

    n_perm = LSH_PROFILES[profile]["n_perm"]
    l = LSH_PROFILES[profile]["l"]
    config = LayoutConfig()
    config.k = k
    config.kc = kc

    t0 = time.perf_counter()
    encoder = MinHash(num_perm=n_perm, seed=seed)
    signatures = encoder.batch_from_binary_array(X)
    encode_s = time.perf_counter() - t0

    forest = LSHForest(d=n_perm, l=l)

    t0 = time.perf_counter()
    forest.batch_add(signatures)
    add_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    forest.index()
    index_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    x, y, s, t = layout_from_lsh_forest(forest, config=config, create_mst=True)
    layout_s = time.perf_counter() - t0

    pred = np.vstack(
        [
            _neighbors_from_result_tuples(forest.query_linear_scan_by_id(int(qid), k, kc), k)
            for qid in query_ids
        ]
    )

    return {
        "backend": "tmap2_lsh",
        "profile": profile,
        "pred_indices": pred,
        "embedding": np.column_stack([x, y]).astype(np.float32, copy=False),
        "edges": np.column_stack(
            [s.astype(np.int32, copy=False), t.astype(np.int32, copy=False)]
        ),
        "encode_s": encode_s,
        "add_s": add_s,
        "index_s": index_s,
        "pre_layout_s": encode_s + add_s + index_s,
        "layout_from_lsh_forest_s": layout_s,
        "total_s": encode_s + add_s + index_s + layout_s,
        "peak_rss_pipeline_mb": _peak_rss_mb(),
        "peak_rss_mb": _peak_rss_mb(),
        "n_perm": n_perm,
        "l": l,
        "kc": kc,
    }


def _worker_tmap1_lsh(
    X: np.ndarray, profile: str, query_ids: np.ndarray,
    k: int, kc: int, seed: int,
) -> dict[str, Any]:
    tm = _load_tmap1_extension()

    n_perm = LSH_PROFILES[profile]["n_perm"]
    l = LSH_PROFILES[profile]["l"]
    config = tm.LayoutConfiguration()
    config.k = k
    config.kc = kc

    t0 = time.perf_counter()
    encoder = tm.Minhash(n_perm, seed)
    signatures = encoder.batch_from_binary_array(np.asarray(X, dtype=np.uint8))
    encode_s = time.perf_counter() - t0

    forest = tm.LSHForest(n_perm, l)

    t0 = time.perf_counter()
    forest.batch_add(signatures)
    add_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    forest.index()
    index_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    x, y, s, t, _gp = tm.layout_from_lsh_forest(forest, config)
    layout_s = time.perf_counter() - t0

    pred = np.vstack(
        [
            _neighbors_from_result_tuples(forest.query_linear_scan_by_id(int(qid), k, kc), k)
            for qid in query_ids
        ]
    )

    return {
        "backend": "tmap1_lsh",
        "profile": profile,
        "pred_indices": pred,
        "embedding": np.column_stack([list(x), list(y)]).astype(np.float32, copy=False),
        "edges": np.column_stack(
            [
                np.asarray(list(s), dtype=np.int32),
                np.asarray(list(t), dtype=np.int32),
            ]
        ),
        "encode_s": encode_s,
        "add_s": add_s,
        "index_s": index_s,
        "pre_layout_s": encode_s + add_s + index_s,
        "layout_from_lsh_forest_s": layout_s,
        "total_s": encode_s + add_s + index_s + layout_s,
        "peak_rss_pipeline_mb": _peak_rss_mb(),
        "peak_rss_mb": _peak_rss_mb(),
        "n_perm": n_perm,
        "l": l,
        "kc": kc,
    }


# ---------------------------------------------------------------------------
# Subprocess worker entry point
# ---------------------------------------------------------------------------

def _worker_main(args: argparse.Namespace) -> None:
    X, dataset = _load_binary_matrix(args.data_file, args.n, seed=args.seed)
    query_ids = np.load(args.query_path).astype(np.int32, copy=False)

    if args.backend == "tmap2_usearch":
        result = _worker_tmap2_usearch(X, args.profile, query_ids, args.k, args.seed)
    elif args.backend == "tmap2_lsh":
        result = _worker_tmap2_lsh(X, args.profile, query_ids, args.k, args.kc, args.seed)
    else:
        result = _worker_tmap1_lsh(X, args.profile, query_ids, args.k, args.kc, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scalar_json = json.dumps(
        {
            key: value
            for key, value in result.items()
            if key not in {"pred_indices", "embedding", "edges"}
        }
    )
    np.savez_compressed(
        args.output,
        pred_indices=np.asarray(result["pred_indices"], dtype=np.int32),
        embedding=np.asarray(result["embedding"], dtype=np.float32),
        edges=np.asarray(result["edges"], dtype=np.int32),
        scalars=np.asarray(scalar_json),
        dataset=np.asarray(dataset),
        n=np.int64(args.n),
    )


def _run_worker(
    backend: str,
    profile: str,
    data_file: Path,
    n: int,
    query_ids: np.ndarray,
    k: int,
    kc: int,
    seed: int,
    timeout_s: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        query_path = tmp / "query_ids.npy"
        output_path = tmp / "result.npz"
        np.save(query_path, query_ids)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--backend", backend,
            "--profile", profile,
            "--data-file", str(data_file),
            "--n", str(n),
            "--query-path", str(query_path),
            "--output", str(output_path),
            "--k", str(k),
            "--kc", str(kc),
            "--seed", str(seed),
        ]
        wall_start = time.strftime("%H:%M:%S")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        wall_end = time.strftime("%H:%M:%S")

        if proc.returncode != 0:
            raise RuntimeError(
                f"{backend}/{profile} failed for n={n:,} (rc={proc.returncode})\n"
                f"STDOUT:\n{proc.stdout[-4000:]}\n"
                f"STDERR:\n{proc.stderr[-4000:]}"
            )

        data = np.load(output_path, allow_pickle=False)
        result = json.loads(str(data["scalars"].item()))
        result["pred_indices"] = np.asarray(data["pred_indices"], dtype=np.int32)
        result["embedding"] = np.asarray(data["embedding"], dtype=np.float32)
        result["edges"] = np.asarray(data["edges"], dtype=np.int32)
        result["dataset"] = str(data["dataset"].item())
        result["n"] = int(data["n"])
        result["wall_start"] = wall_start
        result["wall_end"] = wall_end
        return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _curve_columns(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_r{rank + 1:02d}": round(float(value), 4)
        for rank, value in enumerate(values)
    }


def _print_summary(row: dict[str, Any]) -> None:
    if row["backend"] == "tmap2_usearch":
        layout_s = row["layout_from_knn_graph_s"]
    else:
        layout_s = row["layout_from_lsh_forest_s"]
    print(
        f"    {row['backend']}/{row['profile']:>8}  "
        f"time={row['total_s']:>8.1f}s  "
        f"pre={row['pre_layout_s']:>8.1f}s  "
        f"layout={layout_s:>8.1f}s  "
        f"peak={row['peak_rss_pipeline_mb']:>7.0f}MB  "
        f"recall@20={row['recall_at_20']:.4f}  "
        f"mst_w={row['mst_true_weight_sum']:.1f}  "
        f"hops@1={row['tree_hops_r01']:.2f}  "
        f"[{row['wall_start']}-{row['wall_end']}]",
        flush=True,
    )


def _save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def _benchmark(args: argparse.Namespace) -> None:
    _log_system_info()

    backends = [b.strip() for b in args.backends.split(",")]
    profiles = [p.strip() for p in args.profiles.split(",")]
    configs = [(b, p) for b in backends for p in profiles]

    dataset_paths = [Path(p.strip()) for p in args.datasets.split(",")]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    rows: list[dict[str, Any]] = []
    output_path = RESULTS_DIR / args.output

    for data_file in dataset_paths:
        X_full, _ = _load_binary_matrix(data_file, n=np.load(data_file, mmap_mode="r").shape[0])
        max_n = X_full.shape[0]

        for n in sizes:
            if n > max_n:
                print(f"\nSkipping n={n:,} for {data_file.name} (only {max_n:,} rows)")
                continue

            X, dataset = _load_binary_matrix(data_file, n=n, seed=args.seed)
            packed = _packed_binary(X)
            query_ids, truth = _load_or_compute_truth(
                X=X,
                dataset=dataset,
                n=n,
                k=args.k,
                n_queries=args.queries,
                seed=args.seed,
                chunk_size=args.exact_chunk_size,
            )

            print(f"\n{'=' * 78}")
            print(
                f"dataset={dataset}  n={n:,}  queries={len(query_ids)}  "
                f"file={data_file.name}"
            )
            print(f"{'=' * 78}")

            for backend, profile in configs:
                try:
                    result = _run_worker(
                        backend=backend,
                        profile=profile,
                        data_file=data_file,
                        n=n,
                        query_ids=query_ids,
                        k=args.k,
                        kc=args.kc,
                        seed=args.seed,
                        timeout_s=args.timeout_s,
                    )
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    print(f"    {backend}/{profile}: FAILED — {exc}", flush=True)
                    continue

                edges = np.asarray(result["edges"], dtype=np.int32)
                pred = np.asarray(result["pred_indices"], dtype=np.int32)

                exact_edge_weights = _exact_tree_edge_weights(
                    packed, edges, args.edge_chunk_size,
                )
                tree = Tree(n_nodes=n, edges=edges, weights=exact_edge_weights, root=0)
                tree_hops, tree_hops_cov = _tree_hop_curve(tree, query_ids, truth)

                row: dict[str, Any] = {
                    "n": n,
                    "dataset": dataset,
                    "source_file": data_file.name,
                    "backend": backend,
                    "profile": profile,
                    "k": args.k,
                    "queries": len(query_ids),
                    "kc": result.get("kc", ""),
                    "connectivity": result.get("connectivity", ""),
                    "expansion_add": result.get("expansion_add", ""),
                    "expansion_search": result.get("expansion_search", ""),
                    "n_perm": result.get("n_perm", ""),
                    "l": result.get("l", ""),
                    "build_s": round(float(result["build_s"]), 3)
                    if "build_s" in result else "",
                    "encode_s": round(float(result["encode_s"]), 3)
                    if "encode_s" in result else "",
                    "add_s": round(float(result["add_s"]), 3)
                    if "add_s" in result else "",
                    "index_s": round(float(result["index_s"]), 3)
                    if "index_s" in result else "",
                    "knn_graph_s": round(float(result["knn_graph_s"]), 3)
                    if "knn_graph_s" in result else "",
                    "pre_layout_s": round(float(result["pre_layout_s"]), 3),
                    "layout_from_knn_graph_s": round(
                        float(result["layout_from_knn_graph_s"]), 3
                    ) if "layout_from_knn_graph_s" in result else "",
                    "layout_from_lsh_forest_s": round(
                        float(result["layout_from_lsh_forest_s"]), 3
                    ) if "layout_from_lsh_forest_s" in result else "",
                    "total_s": round(float(result["total_s"]), 3),
                    "wall_start": result.get("wall_start", ""),
                    "wall_end": result.get("wall_end", ""),
                    "peak_rss_pipeline_mb": round(float(result["peak_rss_pipeline_mb"]), 1),
                    "peak_rss_mb": round(float(result["peak_rss_mb"]), 1),
                    "recall_at_20": round(_recall_at_k(pred, truth, args.k), 4),
                    "tree_edges": int(len(edges)),
                    "tree_components": _component_count(tree),
                    "mst_true_weight_sum": round(
                        float(exact_edge_weights.sum(dtype=np.float64)), 4
                    ),
                    "mst_true_weight_mean": round(float(exact_edge_weights.mean()), 6),
                    "tree_hops_mean_r20": round(float(np.nanmean(tree_hops)), 4),
                    "tree_hops_coverage_mean_r20": round(float(np.mean(tree_hops_cov)), 6),
                }
                row.update(_curve_columns("tree_hops", tree_hops))
                row.update(_curve_columns("tree_hops_cov", tree_hops_cov))

                rows.append(row)
                _save_rows(rows, output_path)
                _print_summary(row)

    print(f"\nSaved {len(rows)} rows to {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode")

    # Internal subprocess worker
    worker = sub.add_parser("worker", help="internal subprocess worker")
    worker.add_argument(
        "--backend",
        choices=["tmap2_usearch", "tmap2_lsh", "tmap1_lsh"],
        required=True,
    )
    worker.add_argument("--profile", choices=["low", "medium", "high"], required=True)
    worker.add_argument("--data-file", type=Path, required=True)
    worker.add_argument("--n", type=int, required=True)
    worker.add_argument("--query-path", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--k", type=int, default=K)
    worker.add_argument("--kc", type=int, default=KC)
    worker.add_argument("--seed", type=int, default=SEED)

    # Main benchmark arguments
    parser.add_argument(
        "--datasets", required=True,
        help="Comma-separated paths to .npy fingerprint files.",
    )
    parser.add_argument(
        "--sizes", required=True,
        help="Comma-separated subsample sizes (e.g. 20000,100000,500000,1000000).",
    )
    parser.add_argument(
        "--backends",
        default="tmap2_usearch,tmap2_lsh,tmap1_lsh",
        help="Comma-separated backends to run.",
    )
    parser.add_argument(
        "--profiles",
        default="low,medium,high",
        help="Comma-separated profiles to run.",
    )
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--kc", type=int, default=KC)
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--exact-chunk-size", type=int, default=DEFAULT_EXACT_CHUNK_SIZE)
    parser.add_argument("--edge-chunk-size", type=int, default=DEFAULT_EDGE_CHUNK_SIZE)
    parser.add_argument("--timeout-s", type=int, default=6 * 60 * 60)
    parser.add_argument(
        "--output",
        default="diversity_tree_quality.csv",
        help="Output filename relative to benchmarks/diversity_run/results/.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "worker":
        _worker_main(args)
        return

    _benchmark(args)


if __name__ == "__main__":
    main()
