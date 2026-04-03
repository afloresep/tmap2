#!/bin/bash
# Paper benchmarks — sequential, isolated runs.
# Close all other heavy apps. Don't close the laptop lid.
#
# Usage:
#   bash scripts/bench_all_sequential.sh          # all benchmarks
#   bash scripts/bench_all_sequential.sh frontier  # frontier sweep only
#   bash scripts/bench_all_sequential.sh usearch   # USearch scaling only
#   bash scripts/bench_all_sequential.sh lsh       # LSH scaling only
#
# Estimated times (Apple M-series, 32 GB):
#   frontier:  ~30 min  (param sweep on ≤200K)
#   usearch:   ~40 min  (Jaccard to 5M + cosine to 1M)
#   lsh:       ~30 min  (TMAP2 Numba + TMAP1 C++ to 1M)
#   total:     ~1.5 hr
#
# Results: benchmarks/results_paper/

set -euo pipefail
cd "$(dirname "$0")/.."

SUITE="${1:-all}"

echo "============================================"
echo " TMAP2 Paper Benchmarks (sequential)"
echo " Started: $(date)"
echo " Suite:   $SUITE"
echo " Host:    $(hostname)"
echo "============================================"

run_step() {
    local name="$1"
    shift
    echo ""
    echo "--- $name ---"
    echo "    $(date)"
    time python "$@"
    echo ""
}

# 1. USearch frontier sweep (defines frozen paper defaults)
if [[ "$SUITE" == "all" || "$SUITE" == "frontier" ]]; then
    run_step "USearch frontier: Jaccard (ChEMBL ≤200K)" \
        scripts/bench_usearch_frontier.py --metric jaccard --sizes 10000,50000,100000,200000

    run_step "USearch frontier: Cosine (synthetic d=768, ≤100K)" \
        scripts/bench_usearch_frontier.py --metric cosine --sizes 10000,50000,100000
fi

# 2. USearch index scaling (frozen HNSW params, no OGDF)
if [[ "$SUITE" == "all" || "$SUITE" == "usearch" ]]; then
    run_step "USearch scaling: Jaccard to 5M (ea=256, es=200)" \
        scripts/bench_index_scale.py usearch --metric jaccard \
        --sizes 10000,100000,500000,1000000,2000000,5000000 \
        --ea 256 --es 200

    run_step "USearch scaling: Cosine to 1M (ea=256, es=200)" \
        scripts/bench_index_scale.py usearch --metric cosine \
        --sizes 10000,100000,500000,1000000 \
        --ea 256 --es 200
fi

# 3. LSH scaling (TMAP2 Numba + TMAP1 C++, fixed kc=50)
if [[ "$SUITE" == "all" || "$SUITE" == "lsh" ]]; then
    run_step "LSH scaling: TMAP2 Numba + TMAP1 C++ (d=512, kc=50, to 1M)" \
        scripts/bench_index_scale.py lsh \
        --sizes 10000,50000,100000,200000,500000,1000000 \
        --n-perm 512
fi

echo ""
echo "============================================"
echo " Done: $(date)"
echo "============================================"
ls -lh benchmarks/results_paper/*.csv benchmarks/results_paper/*.json 2>/dev/null
