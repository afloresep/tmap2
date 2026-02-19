"""Parallel chemistry utilities for fingerprint and property computation.
Requires ``rdkit`` (install via ``pip install tmap[chemistry]``).
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

# default params
_FP_RADIUS: int = 2
_FP_NBITS: int = 1024 


def _init_fp_worker(radius: int, n_bits: int) -> None:
    global _FP_RADIUS, _FP_NBITS
    _FP_RADIUS = radius
    _FP_NBITS = n_bits


def _morgan_fp_batch(smiles_batch: list[str]) -> tuple[NDArray[np.uint8], list[int]]:
    """Process a batch of SMILES, return (fps_array, valid_local_indices)."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=_FP_RADIUS, fpSize=_FP_NBITS)
    fps = []
    valid = []
    for i, smi in enumerate(smiles_batch):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(gen.GetFingerprintAsNumPy(mol).astype(np.uint8))
            valid.append(i)
    if fps:
        return np.stack(fps), valid
    return np.empty((0, _FP_NBITS), dtype=np.uint8), valid


def _mqn_fp_batch(smiles_batch: list[str]) -> tuple[NDArray[np.uint8], list[int]]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    fps = []
    valid = []
    for i, smi in enumerate(smiles_batch):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mqn = np.array(Descriptors.MQNs_(mol), dtype=np.float32) #type: ignore
            fps.append((mqn > 0).astype(np.uint8))
            valid.append(i)
    if fps:
        return np.stack(fps), valid
    return np.empty((0, 42), dtype=np.uint8), valid

#TODO: Probably add more fp

def _mol_props_batch(smiles_batch: list[str]) -> NDArray[np.float64]:
    """Process a batch, return (len(batch), 3) array. Invalid rows are NaN."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    out = np.full((len(smiles_batch), 3), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles_batch):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out[i, 0] = Descriptors.ExactMolWt(mol) #type:ignore
            out[i, 1] = Descriptors.MolLogP(mol) #type: ignore
            out[i, 2] = Chem.rdMolDescriptors.CalcNumRings(mol) #type:ignore
    return out

def _get_mp_context() -> multiprocessing.context.BaseContext: #type:ignore
    if sys.platform == "darwin":
        return multiprocessing.get_context("spawn")
    return multiprocessing.get_context("forkserver")


def _default_n_workers() -> int:
    return min(os.cpu_count() or 1, 12)


def _split_into_chunks(lst: list[Any], n_chunks: int) -> list[list[Any]]:
    """Split list into roughly equal chunks."""
    k, m = divmod(len(lst), n_chunks)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def fingerprints_from_smiles(
    smiles: list[str],
    fp_type: str = "morgan",
    n_workers: int | None = None,
    **kwargs: Any,
) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """Compute molecular fingerprints in parallel.

    Parameters
    ----------
    smiles : list[str]
        Input SMILES strings.
    fp_type : str, default ``'morgan'``
        Fingerprint type. ``'morgan'`` uses Morgan circular fingerprints
        (kwargs: ``radius=2``, ``n_bits=2048``). ``'mqn'`` uses Molecular
        Quantum Numbers (42-dim binarized vector, no kwargs).
    n_workers : int or None
        Number of parallel workers. Defaults to ``min(cpu_count, 12)``.
    **kwargs
        Passed to the fingerprint generator (Morgan only).

    Returns
    -------
    fps : ndarray of shape ``(n_valid, n_bits)``, dtype ``uint8``
        Fingerprint matrix for valid SMILES.
    valid_indices : ndarray of shape ``(n_valid,)``, dtype ``int64``
        Indices into the original *smiles* list for each valid row.
    """
    n = len(smiles)
    if n == 0:
        return np.empty((0, 0), dtype=np.uint8), np.empty(0, dtype=np.int64)

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = min(n_workers, n)

    ctx = _get_mp_context()
    chunks = _split_into_chunks(smiles, n_workers)

    if fp_type == "morgan":
        radius = kwargs.get("radius", 2)
        n_bits = kwargs.get("n_bits", 2048)
        with ctx.Pool(n_workers, initializer=_init_fp_worker, initargs=(radius, n_bits)) as pool:
            batch_results = pool.map(_morgan_fp_batch, chunks)
    elif fp_type == "mqn":
        with ctx.Pool(n_workers) as pool:
            batch_results = pool.map(_mqn_fp_batch, chunks)
    else:
        raise ValueError(f"Unsupported fp_type={fp_type!r}. Use 'morgan' or 'mqn'.")

    # each batch returns (fps_array, local_valid_indices)
    fps_parts: list[NDArray[np.uint8]] = []
    valid_idx: list[int] = []
    offset = 0
    for fps_arr, local_valid in batch_results:
        if len(local_valid) > 0:
            fps_parts.append(fps_arr)
            for li in local_valid:
                valid_idx.append(offset + li)
        offset += len(chunks[len(fps_parts) - 1]) if len(local_valid) > 0 else 0

    # recompute properly
    fps_parts2: list[NDArray[np.uint8]] = []
    valid_idx2: list[int] = []
    offset = 0
    for chunk_i, (fps_arr, local_valid) in enumerate(batch_results):
        if len(local_valid) > 0:
            fps_parts2.append(fps_arr)
            for li in local_valid:
                valid_idx2.append(offset + li)
        offset += len(chunks[chunk_i])

    if not fps_parts2:
        return np.empty((0, 0), dtype=np.uint8), np.empty(0, dtype=np.int64)

    print(f"  [FP] {len(valid_idx2):,}/{n:,} valid")
    fps = np.concatenate(fps_parts2)
    return fps, np.array(valid_idx2, dtype=np.int64)


#TODO: Add more properties and select by passing a list
def molecular_properties(
    smiles: list[str],
    n_workers: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute molecular properties in parallel.

    Computes molecular weight (MW), LogP, and number of rings for each
    SMILES string. Invalid SMILES produce ``NaN``.

    Parameters
    ----------
    smiles : list[str]
        Input SMILES strings.
    n_workers : int or None
        Number of parallel workers. Defaults to ``min(cpu_count, 12)``.

    Returns
    -------
    dict with keys ``'mw'``, ``'logp'``, ``'n_rings'``
        Each value is an ndarray of length ``len(smiles)``.
    """
    n = len(smiles)
    if n == 0:
        return {
            "mw": np.empty(0, dtype=np.float64),
            "logp": np.empty(0, dtype=np.float64),
            "n_rings": np.empty(0, dtype=np.float64),
        }

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = min(n_workers, n)

    ctx = _get_mp_context()
    chunks = _split_into_chunks(smiles, n_workers)

    with ctx.Pool(n_workers) as pool:
        batch_results = pool.map(_mol_props_batch, chunks)

    # Concatenate all (chunk_size, 3) arrays
    props = np.concatenate(batch_results)  # (n, 3)
    print(f"  [Props] {n:,} done, {np.isnan(props[:, 0]).sum():,} invalid")

    return {
        "mw": props[:, 0],
        "logp": props[:, 1],
        "n_rings": props[:, 2],
    }
