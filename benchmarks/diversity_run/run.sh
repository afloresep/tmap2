#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- Configuration -----------------------------------------------------------
# Override these env vars or edit the defaults below.

# Comma-separated paths to .npy fingerprint files (uint8, shape N x bits).
DATASETS="${DATASETS:-benchmarks/diversity_run/data/enamine_1M_morgan_2048.npy,benchmarks/diversity_run/data/gdb20s_1M_morgan_2048.npy,benchmarks/diversity_run/data/mixed_1M_morgan_2048.npy}"

# Comma-separated subsample sizes to evaluate.
SIZES="${SIZES:-20000,100000,200000,500000,1000000}"

# Backends and profiles (all combinations are run).
BACKENDS="${BACKENDS:-tmap2_usearch,tmap2_lsh,tmap1_lsh}"
PROFILES="${PROFILES:-low,medium,high}"

# Output CSV name (written to benchmarks/diversity_run/results/).
OUTPUT="${OUTPUT:-diversity_tree_quality.csv}"

# --- Smoke test ---------------------------------------------------------------
if [[ "${1:-}" == "smoke" ]]; then
    echo "=== Smoke test ==="
    python "$SCRIPT_DIR/bench_diversity.py" \
        --datasets "$DATASETS" \
        --sizes 1000 \
        --backends "$BACKENDS" \
        --profiles "$PROFILES" \
        --queries 10 \
        --exact-chunk-size 500 \
        --edge-chunk-size 500 \
        --output "smoke_${OUTPUT}"
    exit 0
fi

# --- Full run -----------------------------------------------------------------
echo "=== Diversity benchmark ==="
echo "Datasets: $DATASETS"
echo "Sizes:    $SIZES"
echo "Backends: $BACKENDS"
echo "Profiles: $PROFILES"
echo "Output:   $OUTPUT"
echo ""

python "$SCRIPT_DIR/bench_diversity.py" \
    --datasets "$DATASETS" \
    --sizes "$SIZES" \
    --backends "$BACKENDS" \
    --profiles "$PROFILES" \
    --queries 100 \
    --exact-chunk-size 50000 \
    --edge-chunk-size 50000 \
    --output "$OUTPUT"
