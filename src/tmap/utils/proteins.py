"""Protein sequence utilities, file readers, and UniProt/AlphaFold data fetching.

File readers:

- ``read_fasta`` — parse FASTA files into (ids, sequences)
- ``read_protein_csv`` — read a CSV/TSV with ID and sequence columns
- ``read_id_list`` — read a plain-text file of UniProt accessions
- ``read_pdb`` — extract sequence, B-factors, and properties from a PDB file
- ``read_pdb_dir`` — batch-read PDB files from a directory

Alignment parsers:

- ``parse_alignment`` — parse MMseqs2/Foldseek/BLAST m8 output into KNNGraph

Sequence analysis:

- ``sequence_properties`` — physicochemical properties (offline, no deps)

API fetchers:

- ``fetch_uniprot`` — batch-fetch annotations from UniProt REST API
- ``fetch_alphafold`` — batch-fetch structural metadata from AlphaFold DB
"""

from __future__ import annotations

import csv
import json
import logging
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Amino acid lookup tables
# ---------------------------------------------------------------------------

# Standard 20 amino acids
_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Monoisotopic residue masses (Da) — residue mass (loses water on peptide bond)
_AA_WEIGHTS: dict[str, float] = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841,
}

_WATER_MASS: float = 18.01056

# Kyte-Doolittle hydrophobicity scale
_KD_HYDROPHOBICITY: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "E": -3.5,
    "Q": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

# pKa values for isoelectric point calculation
# Format: (pKa, charge_when_protonated)
# Positive residues gain +1 when protonated (below pKa)
# Negative residues/groups lose -1 when deprotonated (above pKa)
_PKA_NTERM = 9.69   # alpha-amino (positive when protonated)
_PKA_CTERM = 2.34   # alpha-carboxyl (negative when deprotonated)

# Side-chain pKa values and charge contribution
_PKA_SIDECHAINS: dict[str, tuple[float, int]] = {
    "D": (3.65, -1),   # Asp — negative when deprotonated
    "E": (4.25, -1),   # Glu
    "C": (8.18, -1),   # Cys
    "Y": (10.07, -1),  # Tyr
    "H": (6.00, 1),    # His — positive when protonated
    "K": (10.54, 1),   # Lys
    "R": (12.48, 1),   # Arg
}

AVAILABLE_SEQUENCE_PROPERTIES: list[str] = [
    "length",
    "molecular_weight",
    "isoelectric_point",
    "gravy",
    "charge_at_ph7",
    "aromaticity",
    "aliphatic_index",
    "frac_charged",
    "frac_hydrophobic",
    "frac_polar",
    "frac_acidic",
    "frac_basic",
    "n_cysteines",
]

# Three-letter to one-letter amino acid code mapping (for PDB parsing)
_AA3_TO_1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ---------------------------------------------------------------------------
# UniProt constants
# ---------------------------------------------------------------------------

_UNIPROT_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")

_UNIPROT_BASE_URL = "https://rest.uniprot.org/uniprotkb/search"

DEFAULT_FIELDS: tuple[str, ...] = (
    "accession",
    "protein_name",
    "organism_name",
    "annotation_score",
    "length",
    "ec",
    "cc_subcellular_location",
    "go_f",
    "go_p",
)

_NUMERIC_FIELDS: frozenset[str] = frozenset({"annotation_score", "length"})


# ---------------------------------------------------------------------------
# Internal helpers — sequence properties
# ---------------------------------------------------------------------------

def _is_valid_sequence(seq: str) -> bool:
    """Check if a sequence contains only standard amino acid characters."""
    return len(seq) > 0 and all(c in _STANDARD_AA for c in seq)


def _net_charge_at_ph(ph: float, seq: str) -> float:
    """Compute net charge of a protein at a given pH."""
    # N-terminus: positive when protonated
    charge = 1.0 / (1.0 + 10.0 ** (ph - _PKA_NTERM))
    # C-terminus: negative when deprotonated
    charge -= 1.0 / (1.0 + 10.0 ** (_PKA_CTERM - ph))

    for aa in seq:
        if aa in _PKA_SIDECHAINS:
            pka, sign = _PKA_SIDECHAINS[aa]
            if sign == 1:  # positive when protonated (H, K, R)
                charge += 1.0 / (1.0 + 10.0 ** (ph - pka))
            else:  # negative when deprotonated (D, E, C, Y)
                charge -= 1.0 / (1.0 + 10.0 ** (pka - ph))
    return charge


def _compute_pI(seq: str) -> float:
    """Estimate isoelectric point by bisection."""
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        charge = _net_charge_at_ph(mid, seq)
        if charge > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


_HYDROPHOBIC_AA = frozenset("AVILMFW")
_AROMATIC_AA = frozenset("FWY")
_CHARGED_AA = frozenset("DEKR")
_ACIDIC_AA = frozenset("DE")
_BASIC_AA = frozenset("KRH")
_POLAR_AA = frozenset("STNQYC")


def _frac_of(seq: str, residue_set: frozenset[str]) -> float:
    return sum(1 for aa in seq if aa in residue_set) / len(seq)


def _aliphatic_index(seq: str) -> float:
    """Ikai (1980) aliphatic index."""
    n = len(seq)
    xa = seq.count("A") / n
    xv = seq.count("V") / n
    xi = seq.count("I") / n
    xl = seq.count("L") / n
    return 100.0 * (xa + 2.9 * xv + 3.9 * (xi + xl))


def _compute_prop(name: str, seq: str) -> float:
    """Compute a single named property for a valid sequence."""
    if name == "length":
        return float(len(seq))
    if name == "molecular_weight":
        return sum(_AA_WEIGHTS[aa] for aa in seq) + _WATER_MASS
    if name == "isoelectric_point":
        return _compute_pI(seq)
    if name == "gravy":
        return sum(_KD_HYDROPHOBICITY[aa] for aa in seq) / len(seq)
    if name == "charge_at_ph7":
        return _net_charge_at_ph(7.0, seq)
    if name == "aromaticity":
        return _frac_of(seq, _AROMATIC_AA)
    if name == "aliphatic_index":
        return _aliphatic_index(seq)
    if name == "frac_charged":
        return _frac_of(seq, _CHARGED_AA)
    if name == "frac_hydrophobic":
        return _frac_of(seq, _HYDROPHOBIC_AA)
    if name == "frac_polar":
        return _frac_of(seq, _POLAR_AA)
    if name == "frac_acidic":
        return _frac_of(seq, _ACIDIC_AA)
    if name == "frac_basic":
        return _frac_of(seq, _BASIC_AA)
    if name == "n_cysteines":
        return float(seq.count("C"))
    raise ValueError(f"Unknown property: {name!r}")

def _fetch_uniprot_chunk(
    ids: list[str],
    fields: tuple[str, ...],
) -> list[dict[str, str]]:
    """Fetch a single chunk of UniProt IDs. Returns list of row dicts."""
    query = " OR ".join(f"accession:{uid}" for uid in ids)
    fields_str = ",".join(fields)
    params = urllib.parse.urlencode({
        "query": query,
        "fields": fields_str,
        "format": "tsv",
        "size": str(len(ids)),
    })
    url = f"{_UNIPROT_BASE_URL}?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "tmap2/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    lines = text.strip().split("\n")
    if len(lines) < 2:
        return []

    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        values = line.split("\t")
        row = dict(zip(headers, values))
        rows.append(row)
    return rows

def read_fasta(
    path: str | Path,
    max_seqs: int | None = None,
) -> tuple[list[str], list[str]]:
    """Parse a FASTA file into IDs and sequences.

    Parameters
    ----------
    path : str or Path
        Path to a ``.fa`` / ``.fasta`` file.
    max_seqs : int or None
        Stop after this many sequences. ``None`` reads all.

    Returns
    -------
    ids : list[str]
        Header lines (everything after ``>`` up to the first whitespace).
    sequences : list[str]
        Protein sequences (uppercased, multi-line concatenated).
    """
    ids: list[str] = []
    sequences: list[str] = []
    current_seq: list[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq).upper())
                    current_seq = []
                if max_seqs is not None and len(ids) >= max_seqs:
                    break
                # ID = first whitespace-delimited token after '>'
                ids.append(line[1:].split()[0] if line[1:].strip() else "")
            else:
                current_seq.append(line)
        if current_seq and (max_seqs is None or len(sequences) < max_seqs):
            sequences.append("".join(current_seq).upper())

    return ids, sequences


def read_protein_csv(
    path: str | Path,
    id_col: str = "id",
    seq_col: str = "sequence",
) -> tuple[list[str], list[str]]:
    """Read protein IDs and sequences from a CSV or TSV file.

    Auto-detects delimiter (comma vs. tab).

    Parameters
    ----------
    path : str or Path
        Path to a ``.csv`` or ``.tsv`` file with a header row.
    id_col : str
        Column name for protein IDs.
    seq_col : str
        Column name for amino acid sequences.

    Returns
    -------
    ids : list[str]
        Protein identifiers.
    sequences : list[str]
        Protein sequences (uppercased).
    """
    path = Path(path)
    with open(path, newline="") as f:
        sample = f.read(4096)
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t")

    ids: list[str] = []
    sequences: list[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if id_col not in (reader.fieldnames or []):
            raise ValueError(f"Column {id_col!r} not found. Available: {reader.fieldnames}")
        if seq_col not in (reader.fieldnames or []):
            raise ValueError(f"Column {seq_col!r} not found. Available: {reader.fieldnames}")
        for row in reader:
            ids.append(row[id_col])
            sequences.append(row[seq_col].upper())

    return ids, sequences


def read_id_list(
    path: str | Path,
) -> list[str]:
    """Read UniProt accession IDs from a plain-text file (one per line).

    Blank lines and lines starting with ``#`` are skipped.
    IDs that don't match UniProt accession format are logged as warnings
    but still included.

    Parameters
    ----------
    path : str or Path
        Path to a text file with one accession per line.

    Returns
    -------
    list[str]
        UniProt accession IDs.
    """
    ids: list[str] = []
    n_invalid = 0
    with open(path) as f:
        for line in f:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            # Take first whitespace-delimited token (handles trailing comments)
            uid = token.split()[0]
            if not _UNIPROT_RE.match(uid):
                n_invalid += 1
                logger.warning("read_id_list: %r doesn't match UniProt format", uid)
            ids.append(uid)

    if n_invalid:
        logger.warning("read_id_list: %d/%d IDs don't match UniProt format", n_invalid, len(ids))
    return ids

def read_pdb(
    path: str | Path,
    chain: str | None = None,
) -> dict:
    """Extract sequence and structural properties from a PDB file.

    Parses ``ATOM`` records for CA (alpha-carbon) atoms to extract per-residue
    information. B-factor values correspond to pLDDT in AlphaFold structures.

    Parameters
    ----------
    path : str or Path
        Path to a ``.pdb`` or ``.ent`` file.
    chain : str or None
        If given, only parse this chain ID (e.g. ``"A"``). If ``None``,
        use all chains.

    Returns
    -------
    dict with keys:
        - ``id`` — PDB ID from HEADER record, or filename stem
        - ``sequence`` — one-letter amino acid sequence from CA atoms
        - ``length`` — number of residues (int)
        - ``mean_bfactor`` — mean B-factor across CA atoms (float)
        - ``frac_bfactor_high`` — fraction of residues with B-factor > 90
        - ``frac_bfactor_low`` — fraction of residues with B-factor < 50
    """
    path = Path(path)
    pdb_id = path.stem
    residues: list[str] = []
    bfactors: list[float] = []

    with open(path) as f:
        for line in f:
            # Extract PDB ID from HEADER record
            if line.startswith("HEADER"):
                header_id = line[62:66].strip()
                if header_id:
                    pdb_id = header_id

            if not line.startswith("ATOM"):
                continue

            # PDB fixed-width: atom name at 12-16, residue name 17-20,
            # chain ID 21, B-factor 60-66
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            chain_id = line[21]
            if chain is not None and chain_id != chain:
                continue

            resname = line[17:20].strip()
            aa = _AA3_TO_1.get(resname)
            if aa is None:
                continue  # skip non-standard residues

            try:
                bfactor = float(line[60:66])
            except (ValueError, IndexError):
                bfactor = float("nan")

            residues.append(aa)
            bfactors.append(bfactor)

    sequence = "".join(residues)
    length = len(residues)

    if length == 0:
        return {
            "id": pdb_id,
            "sequence": "",
            "length": 0,
            "mean_bfactor": float("nan"),
            "frac_bfactor_high": float("nan"),
            "frac_bfactor_low": float("nan"),
        }

    bf = np.array(bfactors, dtype=np.float64)
    valid = ~np.isnan(bf)
    n_valid = int(valid.sum())

    return {
        "id": pdb_id,
        "sequence": sequence,
        "length": length,
        "mean_bfactor": float(np.nanmean(bf)) if n_valid else float("nan"),
        "frac_bfactor_high": float(np.sum(bf[valid] > 90) / n_valid) if n_valid else float("nan"),
        "frac_bfactor_low": float(np.sum(bf[valid] < 50) / n_valid) if n_valid else float("nan"),
    }


def read_pdb_dir(
    directory: str | Path,
    pattern: str = "*.pdb",
    chain: str | None = None,
) -> tuple[list[str], list[str], dict[str, NDArray[np.float64]]]:
    """Batch-read PDB files from a directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing PDB files.
    pattern : str
        Glob pattern for PDB files. Default ``"*.pdb"``.
    chain : str or None
        Chain filter passed to :func:`read_pdb`.

    Returns
    -------
    ids : list[str]
        PDB IDs (one per file).
    sequences : list[str]
        Amino acid sequences.
    properties : dict[str, NDArray[np.float64]]
        Arrays for ``mean_bfactor``, ``frac_bfactor_high``,
        ``frac_bfactor_low``, ``length``. Compatible with
        :meth:`~tmap.visualization.TmapViz.add_metadata`.
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))

    ids: list[str] = []
    sequences: list[str] = []
    lengths: list[float] = []
    mean_bf: list[float] = []
    frac_high: list[float] = []
    frac_low: list[float] = []

    for f in files:
        entry = read_pdb(f, chain=chain)
        ids.append(entry["id"])
        sequences.append(entry["sequence"])
        lengths.append(float(entry["length"]))
        mean_bf.append(entry["mean_bfactor"])
        frac_high.append(entry["frac_bfactor_high"])
        frac_low.append(entry["frac_bfactor_low"])

    props = {
        "length": np.array(lengths, dtype=np.float64),
        "mean_bfactor": np.array(mean_bf, dtype=np.float64),
        "frac_bfactor_high": np.array(frac_high, dtype=np.float64),
        "frac_bfactor_low": np.array(frac_low, dtype=np.float64),
    }

    return ids, sequences, props

# PUblic API
def parse_alignment(
    path: str | Path,
    k: int = 20,
    score_col: str = "bitscore",
    as_distance: bool = True,
) -> tuple:
    """Parse BLAST/MMseqs2/Foldseek tabular output into a KNNGraph.

    Reads standard BLAST6 / m8 format (12 tab-separated columns, no header):
    ``qseqid sseqid pident length mismatch gapopen qstart qend sstart send
    evalue bitscore``.

    Parameters
    ----------
    path : str or Path
        Path to a tab-separated m8 file.
    k : int
        Number of nearest neighbors to retain per query.
    score_col : str
        Column to use for ranking: ``"bitscore"`` (default), ``"pident"``,
        or ``"evalue"``.
    as_distance : bool
        If True, convert similarity to distance. For bitscore/pident:
        ``1 / (1 + score)``. For evalue: ``-log10(evalue + 1e-300)``.

    Returns
    -------
    knn : KNNGraph
        k-nearest neighbor graph suitable for ``TMAP().fit(knn_graph=knn)``.
    id_order : list[str]
        Protein IDs in the order they appear as row indices in the KNNGraph.

    Examples
    --------
    >>> knn, ids = parse_alignment("mmseqs_results.m8", k=20)
    >>> model = TMAP().fit(knn_graph=knn)
    """
    from tmap.index.types import KNNGraph

    _COL_MAP = {"pident": 2, "evalue": 10, "bitscore": 11}
    if score_col not in _COL_MAP:
        raise ValueError(f"score_col must be one of {list(_COL_MAP)}, got {score_col!r}")
    col_idx = _COL_MAP[score_col]

    # Parse: collect best hit per (query, subject) pair
    hits: dict[str, dict[str, float]] = {}  # query -> {subject -> best_score}
    id_set: dict[str, int] = {}  # id -> integer index (insertion order)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 12:
                continue

            qid, sid = cols[0], cols[1]
            if qid == sid:  # skip self-hits
                continue

            try:
                score = float(cols[col_idx])
            except ValueError:
                continue

            # Register IDs in first-seen order
            if qid not in id_set:
                id_set[qid] = len(id_set)
            if sid not in id_set:
                id_set[sid] = len(id_set)

            # Keep best score per (query, subject) pair
            q_hits = hits.setdefault(qid, {})
            if score_col == "evalue":
                # Lower evalue = better hit
                if sid not in q_hits or score < q_hits[sid]:
                    q_hits[sid] = score
            else:
                # Higher bitscore/pident = better hit
                if sid not in q_hits or score > q_hits[sid]:
                    q_hits[sid] = score

    n = len(id_set)
    if n == 0:
        raise ValueError("No valid hits found in alignment file")

    id_order = list(id_set.keys())

    # Build kNN arrays
    indices = np.full((n, k), -1, dtype=np.int32)
    distances = np.full((n, k), np.inf, dtype=np.float32)

    for qid, q_hits in hits.items():
        qi = id_set[qid]
        # Sort by score
        if score_col == "evalue":
            sorted_hits = sorted(q_hits.items(), key=lambda x: x[1])  # ascending
        else:
            sorted_hits = sorted(q_hits.items(), key=lambda x: -x[1])  # descending

        for j, (sid, score) in enumerate(sorted_hits[:k]):
            si = id_set[sid]
            if as_distance:
                if score_col == "evalue":
                    dist = -np.log10(score + 1e-300)
                else:
                    dist = 1.0 / (1.0 + score)
            else:
                dist = score
            indices[qi, j] = si
            distances[qi, j] = np.float32(dist)

    knn = KNNGraph.from_arrays(indices, distances)
    return knn, id_order


# Public API for sequence analysis

def sequence_properties(
    sequences: Sequence[str],
    properties: list[str] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute physicochemical properties from amino acid sequences.

    Parameters
    ----------
    sequences : list[str]
        Amino acid sequences using standard single-letter codes.
    properties : list[str] or None
        Which properties to compute. Defaults to all in
        ``AVAILABLE_SEQUENCE_PROPERTIES``. Options: ``'length'``,
        ``'molecular_weight'``, ``'isoelectric_point'``, ``'gravy'``,
        ``'charge_at_ph7'``, ``'aromaticity'``, ``'aliphatic_index'``,
        ``'frac_charged'``, ``'frac_hydrophobic'``, ``'frac_polar'``,
        ``'frac_acidic'``, ``'frac_basic'``, ``'n_cysteines'``.

    Returns
    -------
    dict[str, ndarray]
        Each key is a property name, each value is an ndarray of
        length ``len(sequences)``. Invalid sequences (non-standard
        characters or empty) produce ``NaN``.
    """
    if properties is None:
        properties = list(AVAILABLE_SEQUENCE_PROPERTIES)
    else:
        bad = [p for p in properties if p not in AVAILABLE_SEQUENCE_PROPERTIES]
        if bad:
            raise ValueError(
                f"Unknown properties: {bad}. "
                f"Available: {AVAILABLE_SEQUENCE_PROPERTIES}"
            )

    n = len(sequences)
    if n == 0:
        return {k: np.empty(0, dtype=np.float64) for k in properties}

    n_props = len(properties)
    out = np.full((n, n_props), np.nan, dtype=np.float64)
    n_invalid = 0

    for i, seq in enumerate(sequences):
        seq_upper = seq.upper()
        if _is_valid_sequence(seq_upper):
            for j, name in enumerate(properties):
                out[i, j] = _compute_prop(name, seq_upper)
        else:
            n_invalid += 1

    if n_invalid > 0:
        logger.warning("sequence_properties: %d/%d sequences invalid (non-standard AAs or empty)", n_invalid, n)

    return {k: out[:, j] for j, k in enumerate(properties)}


def fetch_uniprot(
    uniprot_ids: Sequence[str],
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    chunk_size: int = 50,
    max_workers: int = 4,
) -> dict[str, NDArray]:
    """Batch-fetch annotations from the UniProt REST API.

    Parameters
    ----------
    uniprot_ids : list[str]
        UniProt accession IDs (e.g. ``["P12345", "Q9NZC2"]``).
    fields : tuple[str, ...]
        UniProt return fields. See https://www.uniprot.org/help/return_fields .
    chunk_size : int
        Number of IDs per API request. Default 50 to stay within URL length
        limits for GET requests.
    max_workers : int
        Number of concurrent HTTP requests.

    Returns
    -------
    dict mapping field names to arrays
        Numeric fields (``annotation_score``, ``length``) → ``float64`` (NaN
        for missing). Text fields → ``object`` array of strings (empty string
        for missing). Arrays are ordered to match the input *uniprot_ids*.
    """
    n = len(uniprot_ids)
    if n == 0:
        return {f: np.empty(0, dtype=np.float64 if f in _NUMERIC_FIELDS else object) for f in fields}

    # Validate IDs
    valid_ids: list[str] = []
    valid_mask = np.zeros(n, dtype=bool)
    for i, uid in enumerate(uniprot_ids):
        if _UNIPROT_RE.match(uid):
            valid_ids.append(uid)
            valid_mask[i] = True
        else:
            logger.warning("fetch_uniprot: skipping invalid ID %r", uid)

    if not valid_ids:
        logger.warning("fetch_uniprot: no valid UniProt IDs provided")
        return {f: np.empty(0, dtype=np.float64 if f in _NUMERIC_FIELDS else object) for f in fields}

    # Chunk the valid IDs
    chunks = [valid_ids[i : i + chunk_size] for i in range(0, len(valid_ids), chunk_size)]
    n_chunks = len(chunks)

    # Fetch in parallel
    all_rows: list[dict[str, str]] = []
    n_failed = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, n_chunks)) as executor:
        futures = {
            executor.submit(_fetch_uniprot_chunk, chunk, fields): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
            except Exception as exc:
                n_failed += 1
                logger.warning(
                    "fetch_uniprot: chunk %d/%d failed: %s",
                    chunk_idx + 1, n_chunks, exc,
                )

    if n_failed:
        logger.warning("fetch_uniprot: %d/%d chunks failed", n_failed, n_chunks)

    print(f"  [UniProt] fetched {len(all_rows):,}/{len(valid_ids):,} entries ({n_failed} chunk failures)")

    # Build accession → row lookup from TSV headers
    # UniProt TSV uses human-readable column names; map them back
    acc_to_row: dict[str, dict[str, str]] = {}
    for row in all_rows:
        # The accession column appears as "Entry" in TSV output
        acc = row.get("Entry", row.get("accession", ""))
        if acc:
            acc_to_row[acc] = row

    # Map UniProt field names to TSV column names
    _FIELD_TO_COL: dict[str, str] = {
        "accession": "Entry",
        "protein_name": "Protein names",
        "organism_name": "Organism",
        "annotation_score": "Annotation",
        "length": "Length",
        "sequence": "Sequence",
        "ec": "EC number",
        "cc_subcellular_location": "Subcellular location [CC]",
        "go_f": "Gene Ontology (molecular function)",
        "go_p": "Gene Ontology (biological process)",
    }

    # Build output arrays matching original input order
    result: dict[str, NDArray] = {}
    for field in fields:
        col_name = _FIELD_TO_COL.get(field, field)
        if field in _NUMERIC_FIELDS:
            arr = np.full(n, np.nan, dtype=np.float64)
            valid_j = 0
            for i in range(n):
                if valid_mask[i]:
                    uid = uniprot_ids[i]
                    row = acc_to_row.get(uid, {})
                    val_str = row.get(col_name, "")
                    if val_str:
                        try:
                            arr[i] = float(val_str)
                        except ValueError:
                            pass
                    valid_j += 1
            result[field] = arr
        else:
            arr = np.empty(n, dtype=object)
            arr[:] = ""
            valid_j = 0
            for i in range(n):
                if valid_mask[i]:
                    uid = uniprot_ids[i]
                    row = acc_to_row.get(uid, {})
                    arr[i] = row.get(col_name, "")
                    valid_j += 1
            result[field] = arr

    return result


# AlphaFold DB helpers
_ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction"


def _fetch_one_alphafold(uniprot_id: str) -> tuple[str, dict[str, float] | None]:
    """Fetch structural metadata for a single protein from AlphaFold DB."""
    url = f"{_ALPHAFOLD_API_URL}/{uniprot_id}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            entry = data[0] if isinstance(data, list) else data
            return uniprot_id, {
                "length": float(entry.get("sequenceEnd", 0)),
                "plddt": float(entry.get("globalMetricValue", 0)) or float("nan"),
                "frac_disordered": float(entry.get("fractionPlddtVeryLow", 0)),
                "frac_confident": float(entry.get("fractionPlddtVeryHigh", 0)),
            }
    except Exception:
        return uniprot_id, None


def fetch_alphafold(
    uniprot_ids: Sequence[str],
    max_workers: int = 20,
) -> dict[str, NDArray[np.float32]]:
    """Batch-fetch structural metadata from the AlphaFold DB REST API.

    Parameters
    ----------
    uniprot_ids : list[str]
        UniProt accession IDs (e.g. ``["P12345", "Q9NZC2"]``).
    max_workers : int
        Number of concurrent HTTP requests. Default 20 (I/O-bound).

    Returns
    -------
    dict with keys ``'length'``, ``'plddt'``, ``'frac_disordered'``,
    ``'frac_confident'``
        Each value is a ``float32`` ndarray of length ``len(uniprot_ids)``.
        Missing entries (404 / network error) produce ``NaN``.
    """
    keys = ("length", "plddt", "frac_disordered", "frac_confident")
    n = len(uniprot_ids)
    if n == 0:
        return {k: np.empty(0, dtype=np.float32) for k in keys}

    out = np.full((n, 4), np.nan, dtype=np.float32)
    id_to_idx = {uid: i for i, uid in enumerate(uniprot_ids)}

    n_ok = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as executor:
        futures = {
            executor.submit(_fetch_one_alphafold, uid): uid
            for uid in uniprot_ids
        }
        for future in as_completed(futures):
            uid, result = future.result()
            if result is not None:
                i = id_to_idx[uid]
                out[i] = [result[k] for k in keys]
                n_ok += 1
            if (n_ok + 1) % 1000 == 0 or n_ok == n:
                print(f"  [AlphaFold] fetched {n_ok:,}/{n:,}")

    n_missing = n - n_ok
    if n_missing:
        logger.info("fetch_alphafold: %d/%d proteins had no AlphaFold entry", n_missing, n)
    print(f"  [AlphaFold] done — {n_ok:,}/{n:,} entries fetched")

    return {k: out[:, j] for j, k in enumerate(keys)}
