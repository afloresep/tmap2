"""Public top-level API for TMAP."""

from __future__ import annotations

import sysconfig
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any


def _extend_package_path_for_extensions() -> None:
    """Include platform-specific install path for editable OGDF extension builds."""
    platlib_tmap = Path(sysconfig.get_paths()["platlib"]) / "tmap"
    if platlib_tmap.is_dir() and str(platlib_tmap) not in __path__:
        __path__.append(str(platlib_tmap))


_extend_package_path_for_extensions()

if TYPE_CHECKING:
    from tmap.estimator import TMAP
    from tmap.index.encoders.minhash import MinHash, WeightedMinHash
    from tmap.index.lsh_forest import LSHForest
    from tmap.utils.chemistry import (
        AVAILABLE_PROPERTIES,
        AVAILABLE_REACTION_PROPERTIES,
        fingerprints_from_smiles,
        molecular_properties,
        murcko_scaffolds,
        reaction_properties,
    )
    from tmap.utils.proteins import (
        AVAILABLE_SEQUENCE_PROPERTIES,
        fetch_alphafold,
        fetch_uniprot,
        parse_alignment,
        read_fasta,
        read_id_list,
        read_pdb,
        read_pdb_dir,
        read_protein_csv,
        sequence_properties,
    )
    from tmap.utils.singlecell import (
        cell_metadata,
        from_anndata,
        marker_scores,
        obs_to_numeric,
        sample_obs_indices,
        subset_anndata,
    )

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "MinHash",
    "WeightedMinHash",
    "LSHForest",
    "TMAP",
    "AVAILABLE_PROPERTIES",
    "AVAILABLE_REACTION_PROPERTIES",
    "AVAILABLE_SEQUENCE_PROPERTIES",
    "cell_metadata",
    "fingerprints_from_smiles",
    "from_anndata",
    "marker_scores",
    "molecular_properties",
    "obs_to_numeric",
    "murcko_scaffolds",
    "reaction_properties",
    "parse_alignment",
    "sequence_properties",
    "fetch_uniprot",
    "fetch_alphafold",
    "read_fasta",
    "read_id_list",
    "read_pdb",
    "read_pdb_dir",
    "read_protein_csv",
    "sample_obs_indices",
    "subset_anndata",
]

_LAZY_IMPORTS: dict[str, str] = {
    "TMAP": "tmap.estimator",
    "LSHForest": "tmap.index.lsh_forest",
    "MinHash": "tmap.index.encoders.minhash",
    "WeightedMinHash": "tmap.index.encoders.minhash",
    "AVAILABLE_PROPERTIES": "tmap.utils.chemistry",
    "AVAILABLE_REACTION_PROPERTIES": "tmap.utils.chemistry",
    "AVAILABLE_SEQUENCE_PROPERTIES": "tmap.utils.proteins",
    "fingerprints_from_smiles": "tmap.utils.chemistry",
    "cell_metadata": "tmap.utils.singlecell",
    "from_anndata": "tmap.utils.singlecell",
    "marker_scores": "tmap.utils.singlecell",
    "obs_to_numeric": "tmap.utils.singlecell",
    "sample_obs_indices": "tmap.utils.singlecell",
    "subset_anndata": "tmap.utils.singlecell",
    "molecular_properties": "tmap.utils.chemistry",
    "murcko_scaffolds": "tmap.utils.chemistry",
    "reaction_properties": "tmap.utils.chemistry",
    "sequence_properties": "tmap.utils.proteins",
    "fetch_uniprot": "tmap.utils.proteins",
    "fetch_alphafold": "tmap.utils.proteins",
    "parse_alignment": "tmap.utils.proteins",
    "read_fasta": "tmap.utils.proteins",
    "read_id_list": "tmap.utils.proteins",
    "read_pdb": "tmap.utils.proteins",
    "read_pdb_dir": "tmap.utils.proteins",
    "read_protein_csv": "tmap.utils.proteins",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module(module_path)
    except ModuleNotFoundError as exc:
        if name in {"MinHash", "WeightedMinHash"} and exc.name in {"datasketch", "xxhash"}:
            raise ModuleNotFoundError(
                f"Optional dependencies 'datasketch' and 'xxhash' are required "
                f"for `tmap.{name}`. Install them with "
                "`pip install datasketch xxhash`."
            ) from exc
        raise

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
