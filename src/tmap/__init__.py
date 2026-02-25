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
    from tmap.utils.chemistry import fingerprints_from_smiles, molecular_properties

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "MinHash",
    "WeightedMinHash",
    "LSHForest",
    "TMAP",
    "fingerprints_from_smiles",
    "molecular_properties",
]

_LAZY_IMPORTS: dict[str, str] = {
    "TMAP": "tmap.estimator",
    "LSHForest": "tmap.index.lsh_forest",
    "MinHash": "tmap.index.encoders.minhash",
    "WeightedMinHash": "tmap.index.encoders.minhash",
    "FaissIndex": "tmap.index.faiss_index",
    "fingerprints_from_smiles": "tmap.utils.chemistry",
    "molecular_properties": "tmap.utils.chemistry",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module(module_path)
    except ModuleNotFoundError as exc:
        if name in {"MinHash", "WeightedMinHash"} and exc.name == "datasketch":
            raise ModuleNotFoundError(
                f"Optional dependency 'datasketch' is required for `tmap.{name}`. "
                "Install it with `pip install datasketch`."
            ) from exc
        raise

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
