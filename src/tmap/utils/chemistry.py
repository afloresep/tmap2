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
_PROP_NAMES: list[str] = []

AVAILABLE_PROPERTIES: list[str] = [
    "mw",
    "logp",
    "n_rings",
    "hba",
    "hbd",
    "tpsa",
    "n_rotatable_bonds",
    "n_heavy_atoms",
    "fraction_csp3",
    "n_aromatic_rings",
    "n_heteroatoms",
    "formal_charge",
    "qed",
]


def _init_fp_worker(radius: int, n_bits: int) -> None:
    global _FP_RADIUS, _FP_NBITS
    _FP_RADIUS = radius
    _FP_NBITS = n_bits


def _init_props_worker(prop_names: list[str]) -> None:
    global _PROP_NAMES
    _PROP_NAMES = prop_names


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
            mqn = np.array(Descriptors.MQNs_(mol), dtype=np.float32)  # type: ignore
            fps.append((mqn > 0).astype(np.uint8))
            valid.append(i)
    if fps:
        return np.stack(fps), valid
    return np.empty((0, 42), dtype=np.uint8), valid


# TODO: Probably add more fp


def _mol_props_batch(smiles_batch: list[str]) -> NDArray[np.float64]:
    """Process a batch, return (len(batch), n_props) array. Invalid rows are NaN."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, rdMolDescriptors

    computers = {
        "mw": lambda mol: Descriptors.ExactMolWt(mol),
        "logp": lambda mol: Descriptors.MolLogP(mol),
        "n_rings": lambda mol: rdMolDescriptors.CalcNumRings(mol),
        "hba": lambda mol: rdMolDescriptors.CalcNumHBA(mol),
        "hbd": lambda mol: rdMolDescriptors.CalcNumHBD(mol),
        "tpsa": lambda mol: rdMolDescriptors.CalcTPSA(mol),
        "n_rotatable_bonds": lambda mol: rdMolDescriptors.CalcNumRotatableBonds(mol),
        "n_heavy_atoms": lambda mol: float(mol.GetNumHeavyAtoms()),
        "fraction_csp3": lambda mol: rdMolDescriptors.CalcFractionCSP3(mol),
        "n_aromatic_rings": lambda mol: rdMolDescriptors.CalcNumAromaticRings(mol),
        "n_heteroatoms": lambda mol: float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        "formal_charge": lambda mol: float(Chem.GetFormalCharge(mol)),
        "qed": lambda mol: QED.qed(mol),
    }

    props = _PROP_NAMES
    out = np.full((len(smiles_batch), len(props)), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles_batch):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            for j, name in enumerate(props):
                out[i, j] = computers[name](mol)
    return out


def _get_mp_context() -> multiprocessing.context.BaseContext:  # type:ignore
    if sys.platform == "darwin":
        return multiprocessing.get_context("spawn")
    return multiprocessing.get_context("forkserver")


def _default_n_workers() -> int:
    return min(os.cpu_count() or 1, 12)


def _split_into_chunks(lst: list[Any], n_chunks: int) -> list[list[Any]]:
    """Split list into roughly equal chunks."""
    k, m = divmod(len(lst), n_chunks)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


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


def _scaffolds_batch(smiles_batch: list[str]) -> list[str]:
    """Return Murcko scaffold SMILES for a batch. Invalid SMILES → empty string."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    out: list[str] = []
    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
            except Exception:
                out.append("")
        else:
            out.append("")
    return out


def murcko_scaffolds(
    smiles: list[str],
    n_workers: int | None = None,
) -> NDArray[np.object_]:
    """Compute Murcko scaffolds in parallel.

    The Murcko scaffold is the core ring system plus linkers of a
    molecule — useful for grouping/coloring by chemical series.

    Parameters
    ----------
    smiles : list[str]
        Input SMILES strings.
    n_workers : int or None
        Number of parallel workers. Defaults to ``min(cpu_count, 12)``.

    Returns
    -------
    ndarray of shape ``(len(smiles),)``, dtype ``object``
        Scaffold SMILES for each input. Invalid SMILES produce ``""``.
    """
    n = len(smiles)
    if n == 0:
        return np.empty(0, dtype=object)

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = min(n_workers, n)

    ctx = _get_mp_context()
    chunks = _split_into_chunks(smiles, n_workers)

    with ctx.Pool(n_workers) as pool:
        batch_results = pool.map(_scaffolds_batch, chunks)

    flat: list[str] = []
    for batch in batch_results:
        flat.extend(batch)

    result = np.empty(n, dtype=object)
    result[:] = flat
    n_valid = sum(1 for s in flat if s)
    print(f"  [Scaffolds] {n_valid:,}/{n:,} valid")
    return result


def molecular_properties(
    smiles: list[str],
    properties: list[str] | None = None,
    n_workers: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute molecular properties in parallel.

    Parameters
    ----------
    smiles : list[str]
        Input SMILES strings.
    properties : list[str] or None
        Which properties to compute. Defaults to all in
        ``AVAILABLE_PROPERTIES``. Options: ``'mw'``, ``'logp'``,
        ``'n_rings'``, ``'hba'``, ``'hbd'``, ``'tpsa'``,
        ``'n_rotatable_bonds'``, ``'n_heavy_atoms'``,
        ``'fraction_csp3'``, ``'n_aromatic_rings'``,
        ``'n_heteroatoms'``, ``'formal_charge'``.
    n_workers : int or None
        Number of parallel workers. Defaults to ``min(cpu_count, 12)``.

    Returns
    -------
    dict[str, ndarray]
        Each key is a property name, each value is an ndarray of
        length ``len(smiles)``. Invalid SMILES produce ``NaN``.
    """
    if properties is None:
        properties = list(AVAILABLE_PROPERTIES)
    else:
        bad = [p for p in properties if p not in AVAILABLE_PROPERTIES]
        if bad:
            raise ValueError(
                f"Unknown properties: {bad}. "
                f"Available: {AVAILABLE_PROPERTIES}"
            )

    n = len(smiles)
    if n == 0:
        return {p: np.empty(0, dtype=np.float64) for p in properties}

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = min(n_workers, n)

    ctx = _get_mp_context()
    chunks = _split_into_chunks(smiles, n_workers)

    with ctx.Pool(
        n_workers, initializer=_init_props_worker, initargs=(properties,)
    ) as pool:
        batch_results = pool.map(_mol_props_batch, chunks)

    props = np.concatenate(batch_results)  # (n, len(properties))
    print(f"  [Props] {n:,} done, {np.isnan(props[:, 0]).sum():,} invalid")

    return {name: props[:, j] for j, name in enumerate(properties)}
