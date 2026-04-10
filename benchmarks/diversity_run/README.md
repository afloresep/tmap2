# Diversity Run — Cross-Dataset Tree Quality Benchmark

Same evaluation as `scripts/bench_final_map_quality.py` but parameterized on
arbitrary `.npy` fingerprint files instead of hardcoded ChEMBL paths.

## Expected input data

Place `.npy` files (uint8, shape `N x 2048`) in `benchmarks/diversity_run/data/`:

- `enamine_1M_morgan_2048.npy` — 1M Morgan FPs from Enamine
- `gdb20s_1M_morgan_2048.npy` — 1M Morgan FPs from GDB20s
- `mixed_1M_morgan_2048.npy` — 500K Enamine + 500K GDB20s (shuffled)

Any `.npy` binary fingerprint matrix works. Pass paths via `--datasets`.

## Commands

Smoke test (fast, 1K rows):

```bash
./benchmarks/diversity_run/run.sh smoke
```

Full run:

```bash
./benchmarks/diversity_run/run.sh
```

Override defaults via environment variables:

```bash
DATASETS="path/to/fps_a.npy,path/to/fps_b.npy" \
SIZES="50000,200000,1000000" \
BACKENDS="tmap2_usearch,tmap2_lsh" \
./benchmarks/diversity_run/run.sh
```

Or call the script directly:

```bash
python benchmarks/diversity_run/bench_diversity.py \
  --datasets data/enamine.npy,data/gdb.npy,data/mixed.npy \
  --sizes 20000,100000,200000,500000,1000000 \
  --queries 100 \
  --output diversity_tree_quality.csv
```

## Output

Results CSV written to `benchmarks/diversity_run/results/`.
Exact-Jaccard ground truth cached in `benchmarks/diversity_run/cache/`.

Columns are identical to the original benchmark output plus:

- `source_file` — which `.npy` file the data came from
- `wall_start`, `wall_end` — wall-clock timestamps per run (for throttling diagnosis)

## Timing diagnostics

System info (CPU, cores, Python version) is printed at the start of each run.
Wall-clock timestamps are recorded per backend/profile to detect thermal
throttling on sustained workloads. Compare `wall_start`/`wall_end` across
sequential runs to check for slowdowns later in the sequence.

## Profile definitions

Same as the original benchmark:

LSH: `low` (d=256, l=128), `medium` (d=512, l=256), `high` (d=1024, l=512)

USearch: `low` (M=16, ef_add=128, ef_search=64), `medium` (M=32, ef_add=128,
ef_search=100), `high` (M=32, ef_add=256, ef_search=200)
