"""Benchmark worker — runs one embedding method in an isolated subprocess.

Called by scripts/bench_paper.py. Reports embedding, wall-clock time, and
peak RSS so the orchestrator can collect fair, isolated measurements.
"""

from __future__ import annotations

import argparse
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np


def _peak_rss_mb() -> float:
    """Peak resident set size of this process in MB."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return r.ru_maxrss / (1024 * 1024)  # bytes on macOS
    return r.ru_maxrss / 1024  # KB on Linux


def run_tmap2(X: np.ndarray, n_neighbors: int, metric: str, seed: int) -> np.ndarray:
    from tmap import TMAP

    model = TMAP(n_neighbors=n_neighbors, metric=metric, seed=seed)
    x, y, _s, _t = model.fit_transform(X)
    return np.column_stack([x, y])


def run_umap(X: np.ndarray, n_neighbors: int, metric: str, seed: int) -> np.ndarray:
    import umap

    return umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric=metric,
        random_state=seed,
    ).fit_transform(X)


def _load_tmap_silicon():
    """Load the original tmap-silicon C extension, bypassing the tmap2 namespace."""
    import importlib.util
    import sysconfig

    so_path = Path(sysconfig.get_paths()["purelib"]) / "tmap" / (
        f"_tmap.cpython-{sys.version_info.major}{sys.version_info.minor}"
        f"-{sysconfig.get_config_var('MULTIARCH') or 'darwin'}.so"
    )
    if not so_path.exists():
        # macOS arm64 fallback
        import glob
        candidates = glob.glob(
            str(Path(sysconfig.get_paths()["purelib"]) / "tmap" / "_tmap*.so")
        )
        if not candidates:
            raise ImportError("tmap-silicon not found. Install with: pip install tmap-silicon")
        so_path = Path(candidates[0])

    spec = importlib.util.spec_from_file_location("_tmap", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_tmap2_lsh(X: np.ndarray, n_neighbors: int, n_perm: int, seed: int) -> np.ndarray:
    """TMAP2 via MinHash+LSH Numba path. X must be binary uint8.

    Converts binary matrix to sparse index lists to trigger the LSH path
    instead of USearch.
    """
    import scipy.sparse as sp

    from tmap import TMAP

    X_sparse = sp.csr_matrix(X.astype(np.float32))
    model = TMAP(
        n_neighbors=n_neighbors,
        metric="jaccard",
        n_permutations=n_perm,
        seed=seed,
    )
    x, y, _s, _t = model.fit_transform(X_sparse)
    return np.column_stack([x, y])


def run_tmap1(X: np.ndarray, n_neighbors: int, n_perm: int = 128) -> np.ndarray:
    """Original TMAP via tmap-silicon. X must be binary uint8."""
    tm = _load_tmap_silicon()

    n, _d = X.shape
    enc = tm.Minhash(n_perm)
    lf = tm.LSHForest(n_perm)

    for i in range(n):
        fp = tm.VectorUchar(X[i].tolist())
        lf.add(enc.from_binary_array(fp))
    lf.index()

    cfg = tm.LayoutConfiguration()
    cfg.k = n_neighbors
    x, y, _s, _t, _gp = tm.layout_from_lsh_forest(lf, cfg)
    return np.column_stack([list(x), list(y)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["tmap2", "tmap2_lsh", "umap", "tmap1"])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--n-perm", type=int, default=128, help="MinHash permutations (tmap1 only)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = np.load(args.input)
    X = np.asarray(data["X"])
    if args.method not in ("tmap1", "tmap2_lsh") and X.dtype != np.float32:
        X = X.astype(np.float32)

    t0 = time.perf_counter()
    if args.method == "tmap2":
        emb = run_tmap2(X, args.n_neighbors, args.metric, args.seed)
    elif args.method == "tmap2_lsh":
        emb = run_tmap2_lsh(X, args.n_neighbors, args.n_perm, args.seed)
    elif args.method == "umap":
        emb = run_umap(X, args.n_neighbors, args.metric, args.seed)
    else:
        emb = run_tmap1(X, args.n_neighbors, n_perm=args.n_perm)
    runtime = time.perf_counter() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embedding=emb.astype(np.float32, copy=False),
        runtime_s=np.float64(runtime),
        peak_rss_mb=np.float64(_peak_rss_mb()),
    )


if __name__ == "__main__":
    main()
