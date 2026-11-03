# Understanding MinHash: A Visual Guide

This guide explains how MinHash encoding works in TMAP and helps you choose the right method for your data.

**When TMAP uses MinHash:** `TMAP(metric="jaccard").fit(X)` sends a dense 0/1 matrix (such as Morgan fingerprints) to USearch, which compares the raw bits and skips MinHash entirely (see the [USearch guide](usearch_backend.md)). MinHash + LSHForest is used for lists of feature indices, lists of string tokens, and SciPy sparse matrices. The estimator then encodes with 512 permutations and `minhash_seed=42`. Everything below is about the lower-level `MinHash` class, which you can use directly on any of these inputs.

---

## What MinHash Does

MinHash is a **locality-sensitive hashing** technique that compresses your data into compact "signatures" while preserving similarity information.

If two items are similar (high Jaccard similarity), their MinHash signatures are similar too.

```txt
Your Data (2048-bit fingerprint)
        ↓
    [MinHash Encoding]
        ↓
Compact Signature (128 values)
```

TMAP uses this compression because:

1. **Fixed length**: every item becomes the same number of values, whether it has five elements or five thousand. Sets of tokens and sparse features can't be packed into a fixed bit vector, so this is what makes them searchable.
2. **Indexing**: LSH Forest indexes these signatures, so similar items can be found without comparing every pair.

A signature takes 8 bytes per permutation (1 KB at 128 permutations), which is more than a bit-packed 2048-bit fingerprint (256 bytes). For dense fingerprints, compression is not the reason to use MinHash.

---

## Quick Start

```python
from tmap import MinHash

# Create encoder
mh = MinHash(num_perm=128, seed=42)

# Encode fingerprints (most common case)
fingerprints = load_your_data()  # shape: (n_samples, n_bits)
signatures = mh.batch_from_binary_array(fingerprints)

# Now use with LSHForest
from tmap import LSHForest
lsh = LSHForest(d=128, l=32)
lsh.batch_add(signatures)
lsh.index()
```

---

## Choosing the Right Method

`MinHash` has one encoding method per input type, each in a single-item and a batch version.

### Decision Tree

```txt
What type of data do you have?
│
├─ Binary fingerprints (0/1 arrays)?
│  │
│  ├─ One fingerprint at a time? → from_binary_array()
│  │
│  └─ Many fingerprints? → batch_from_binary_array()  (most common)
│
├─ Sparse indices (list of "on" positions)?
│  │
│  ├─ One sample? → from_sparse_binary_array()
│  │
│  └─ Many samples? → batch_from_sparse_binary_array()
│
├─ Text/string data (words, tokens)?
│  │
│  ├─ One document? → from_string_array()
│  │
│  └─ Many documents? → batch_from_string_array()
│
└─ Auto-detect input type? → encode()
```

### Method Comparison

| Method | Input | Backend | When to Use |
|--------|-------|---------|-------------|
| `batch_from_binary_array()` | 2D array (n, bits) | Numba | Many dense fingerprints |
| `from_binary_array()` | 1D array (bits,) | Numba | Single fingerprint |
| `batch_from_sparse_binary_array()` | List of index lists | Numba | Data already stored as "on" positions |
| `from_sparse_binary_array()` | List of indices | Numba | Single sparse fingerprint |
| `batch_from_string_array()` | List of string lists | xxhash64 + Numba | Text data, tokens, n-grams |
| `from_string_array()` | List of strings | xxhash64 + Numba | Single text document |
| `encode()` | Auto-detect | Auto | General purpose, flexible |

---

## Method Details

### `batch_from_binary_array()`: Dense Fingerprints

**Use this when your fingerprints are a 0/1 matrix.**

```python
import numpy as np
from tmap import MinHash

mh = MinHash(num_perm=128, seed=42)

# Your fingerprints: n_samples × n_bits
fingerprints = np.random.randint(0, 2, size=(10000, 2048), dtype=np.uint8)

# Encode all at once (fastest)
signatures = mh.batch_from_binary_array(fingerprints)
print(signatures.shape)  # (10000, 128)
```

**Performance:** the time goes into the set bits, so speed depends mostly on how many bits are 1. Sparse fingerprints such as Morgan (a few percent of bits set) encode several times faster than the half-full random matrix above. If fewer than 30% of the bits are set on average, the method switches to the sparse kernel by itself.

**When to use:**

- Molecular fingerprints (ECFP, MACCS, etc.)
- Any dense binary data
- Processing datasets of any size

---

### `from_binary_array()`: Single Fingerprint

**For encoding one fingerprint at a time.**

```python
# Single fingerprint
fp = np.array([1, 0, 1, 1, 0, 0, 1, 0, ...], dtype=np.uint8)
sig = mh.from_binary_array(fp)
print(sig.shape)  # (128,)
```

**When to use:**

- Processing one item at a time (streaming)
- Querying a new point against existing index
- Testing and debugging

**Note:** For multiple fingerprints, use `batch_from_binary_array()` instead. It encodes rows in parallel with Numba.

---

### `batch_from_sparse_binary_array()`: Sparse Data

**For fingerprints stored as lists of "on" indices.**

```python
# Sparse format: each item is a list of indices where bit = 1
sparse_fingerprints = [
    [0, 5, 42, 100, 256],      # First fingerprint has bits 0, 5, 42, 100, 256 set
    [1, 5, 43, 101, 257],      # Second fingerprint
    [0, 6, 44, 102, 258],      # Third fingerprint
]

signatures = mh.batch_from_sparse_binary_array(sparse_fingerprints)
```

**When to use:**

- Your data is already stored as lists of "on" positions
- The full 0/1 matrix would be too large to build in memory

Each row must be a list, tuple or array of non-negative integers. Python sets work with `encode()` but not with this method.

**Important:** Sparse and dense representations produce **identical** signatures:

```python
# These produce the same signature
indices = [0, 5, 10]
sig_sparse = mh.from_sparse_binary_array(indices)

dense = np.zeros(100, dtype=np.uint8)
dense[indices] = 1
sig_dense = mh.from_binary_array(dense)

np.testing.assert_array_equal(sig_sparse, sig_dense)  # same signature
```

---

### `batch_from_string_array()`: Text Data

**For text data like words, tokens, or n-grams.**

```python
# Each document is a list of strings (tokens, words, etc.)
documents = [
    ["apple", "banana", "cherry"],
    ["banana", "date", "elderberry"],
    ["apple", "fig", "grape"],
]

signatures = mh.batch_from_string_array(documents)
```

**When to use:**

- Text documents (bag-of-words)
- Chemical SMILES treated as character n-grams
- Any string-based set data

**Important:** Each string token is first turned into a 63-bit integer with `xxhash64`, and then goes through the same MinHash as the binary methods. Those integers have nothing to do with bit positions, so a string signature can't be compared with a binary one, even when the tokens are `"0"`, `"5"`, `"42"`.

---

### `encode()`: Auto-Detection

**Flexible method that auto-detects input type.**

```python
# Works with numpy arrays
data = np.random.randint(0, 2, size=(100, 50), dtype=np.uint8)
sigs = mh.encode(data)

# Also works with sets
data = [{1, 2, 3}, {2, 3, 4}, {3, 4, 5}]
sigs = mh.encode(data)

# And lists
data = [[1, 2, 3], [2, 3, 4]]
sigs = mh.encode(data)
```

**When to use:**

- Writing generic code that handles multiple input types
- Quick prototyping
- When input type varies at runtime

**Behavior:**

- NumPy array → Uses Numba (fast)
- List of integer sets/lists → Uses the sparse Numba path
- List of string sets/lists → Uses `xxhash64` token hashing + Numba

---

## WeightedMinHash: For Float Vectors

When each feature has a weight, such as a count or a TF-IDF score, use `WeightedMinHash`:

```python
from tmap import WeightedMinHash

# Must specify dimension upfront
wmh = WeightedMinHash(dim=100, num_perm=128, seed=42)

# Weight vectors (e.g., TF-IDF, count vectors)
weights = np.random.uniform(0.1, 10, size=(1000, 100))  # Must be > 0

# Encode
signatures = wmh.batch_from_weight_array(weights, n_jobs=1)
print(signatures.shape)  # (1000, 128, 2): two values per hash

# Distance calculation is different
sig1 = wmh.from_weight_array(weights[0])
sig2 = wmh.from_weight_array(weights[1])
distance = wmh.get_weighted_distance(sig1, sig2)
```

**Key differences from MinHash:**

1. Requires knowing `dim` at construction time
2. Every value must be greater than zero; a zero raises `ValueError`
3. Output has shape `(n, num_perm, 2)`, two values per hash
4. Use `get_weighted_distance()` for comparisons
5. It runs a Python loop over rows (built on `datasketch`), so it is much slower than `MinHash`. `batch_from_weight_array` starts a process pool when `n_jobs` is left at `None` and there are 100 rows or more. That needs an `if __name__ == "__main__":` guard in scripts, and for a few thousand rows `n_jobs=1` is often faster.

---

## Parameters

### `num_perm` - Signature Size

Controls the tradeoff between accuracy and size.

| Value | Accuracy | Memory | Speed | Use Case |
|-------|----------|--------|-------|----------|
| 32 | Low | 256 bytes | Fastest | Quick exploration |
| 64 | Medium | 512 bytes | Fast | Development |
| 128 | Good | 1 KB | Good | `MinHash` class default, most cases |
| 256 | High | 2 KB | Slower | High precision needed |
| 512 | Very High | 4 KB | Slowest | `TMAP` default, publication quality |

**Rule of thumb:** Start with 128. Increase only if you need better Jaccard estimation accuracy.

### `seed` - Reproducibility

```python
# Same seed = same signatures for same input
mh1 = MinHash(num_perm=128, seed=42)
mh2 = MinHash(num_perm=128, seed=42)

sig1 = mh1.encode(data)
sig2 = mh2.encode(data)
np.testing.assert_array_equal(sig1, sig2)  # identical
```

The default seed is fixed (`seed=1`), so the same input always gives the same signatures even if you don't pass one. To reproduce the signatures `TMAP` builds internally, use `MinHash(num_perm=512, seed=42)`.

---

## Computing Distances

### Jaccard Distance (MinHash)

```python
mh = MinHash(num_perm=128, seed=42)
sig1 = mh.from_binary_array(fp1)
sig2 = mh.from_binary_array(fp2)

# Method 1: Static method
distance = MinHash.get_distance(sig1, sig2)

# Method 2: Alias
distance = MinHash.jaccard_distance(sig1, sig2)

# Distance is in [0, 1]
# 0 = identical, 1 = completely different
```

### Weighted Jaccard Distance (WeightedMinHash)

```python
wmh = WeightedMinHash(dim=100, num_perm=128, seed=42)
sig1 = wmh.from_weight_array(w1)
sig2 = wmh.from_weight_array(w2)

distance = WeightedMinHash.get_weighted_distance(sig1, sig2)
```

### Accuracy vs True Jaccard

MinHash **estimates** Jaccard distance. The estimate improves with more permutations:

```python
# True Jaccard for sets: 1 - |A ∩ B| / |A ∪ B|
true_distance = 1 - len(set1 & set2) / len(set1 | set2)

# MinHash estimate (more accurate with higher num_perm)
mh = MinHash(num_perm=512, seed=42)  # High for accuracy
sig1 = mh.encode([set1])[0]
sig2 = mh.encode([set2])[0]
estimated_distance = mh.get_distance(sig1, sig2)

# Typically within 0.05 of true value with 512 permutations
```

---

## Common Pitfalls

### 1. Mixing Binary and String Signatures

**Problem:** String tokens are hashed to integer IDs first, so they don't line up with bit positions.

```python
# Wrong: these signatures are not comparable
sig_binary = mh.from_binary_array(fingerprint)
sig_string = mh.from_string_array(["1", "2", "3"])
distance = mh.get_distance(sig_binary, sig_string)  # meaningless
```

**Solution:** Use the same encoding method for all data you want to compare.

### 2. Using Batch Methods for Single Items

**Problem:** Overhead from method dispatch.

```python
# Slow: calling the batch method in a loop
for fp in fingerprints:
    sig = mh.batch_from_binary_array(fp.reshape(1, -1))[0]

# Fast: one batch call
sigs = mh.batch_from_binary_array(fingerprints)
```

### 3. Wrong Shape for from_binary_array

**Problem:** Passing 2D array to single-item method.

```python
# Wrong: raises ValueError
fp = np.array([[1, 0, 1], [0, 1, 1]])  # 2D
sig = mh.from_binary_array(fp)

# Right
sig = mh.batch_from_binary_array(fp)  # For multiple
sig = mh.from_binary_array(fp[0])     # For single
```

### 4. Weighted MinHash with Zero Values

**Problem:** WeightedMinHash requires strictly positive values.

```python
# Wrong: raises ValueError
weights = np.array([1.0, 0.0, 2.0])  # contains 0
sig = wmh.from_weight_array(weights)

# Right: add a small epsilon
weights = np.maximum(weights, 1e-10)
sig = wmh.from_weight_array(weights)
```

### 5. Mismatched num_perm

**Problem:** Comparing signatures with different `num_perm`.

```python
mh1 = MinHash(num_perm=64)
mh2 = MinHash(num_perm=128)

sig1 = mh1.from_binary_array(fp)  # Shape: (64,)
sig2 = mh2.from_binary_array(fp)  # Shape: (128,)

# Raises ValueError: mismatched lengths
distance = mh1.get_distance(sig1, sig2)
```

---

## Performance Tips

### 1. Use Batch Methods

```python
# Fast
sigs = mh.batch_from_binary_array(all_fingerprints)

# Slow
sigs = [mh.from_binary_array(fp) for fp in all_fingerprints]
```

### 2. Ensure Correct dtype

```python
fps = fingerprints.astype(np.uint8)   # or bool
sigs = mh.batch_from_binary_array(fps)
```

Other dtypes are converted with `astype(np.uint8)`. That costs a copy, and it truncates: a value of 0.5 becomes 0, so the bit counts as unset. Make sure float input holds only 0.0 and 1.0.

### 3. Skip Manual Sparse Conversion

`batch_from_binary_array` already switches to the sparse kernel when fewer than 30% of the bits are set, so converting a dense matrix to index lists yourself doesn't speed anything up. Use `batch_from_sparse_binary_array` when your data starts out as index lists.

### 4. Numba JIT Compilation

The first call after installing TMAP compiles the Numba kernels, which takes about a second. The compiled code is cached on disk, so later calls and later Python sessions skip that step.

---

## Complete Example

```python
import numpy as np
from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest, LayoutConfig

# 1. Load or generate binary fingerprints
n_molecules = 10000
n_bits = 2048
fingerprints = (np.random.rand(n_molecules, n_bits) < 0.1).astype(np.uint8)

# 2. Create MinHash encoder
mh = MinHash(num_perm=128, seed=42)

# 3. Encode all fingerprints (batch is fastest)
signatures = mh.batch_from_binary_array(fingerprints)
print(f"Encoded {len(signatures)} molecules to shape {signatures.shape}")

# 4. Build LSH Forest index
lsh = LSHForest(d=128, l=32)  # d must match num_perm
lsh.batch_add(signatures)
lsh.index()

# 5. Create layout
cfg = LayoutConfig()
cfg.k = 20
cfg.kc = 50
cfg.deterministic = True
cfg.seed = 42

x, y, s, t = layout_from_lsh_forest(lsh, cfg)
print(f"Layout: {len(x)} points, {len(s)} edges")
```

---

## Next Steps

- See [LSHForest Guide](lshforest_guide.md) for indexing and querying
- See [Layout Guide](layout_guide.md) for visualization parameters
- Work through [notebook 07](../notebooks/07_minhash_deep_dive.ipynb) for the encoders and [notebook 12](../notebooks/12_lower_level_pipeline.ipynb) for the full lower-level pipeline
