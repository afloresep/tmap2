"""Binary container format for large-scale TMAP visualization.

This module implements a binary format optimized for 5M-20M+ points:
- Uint16 quantized coordinates (4x smaller than Float32 JSON)
- Gzip-compressed typed arrays for numeric columns
- Block-wise string encoding for SMILES (lazy loading)

Format Structure:
    [Header: 32 bytes fixed]
    [Section Table: variable]
    [Metadata JSON: variable, optional]
    [Coords Section: gzipped Uint16]
    [Numeric Columns: gzipped Float32/Int32]
    [String Columns: block-compressed]
"""

from __future__ import annotations

import gzip
import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Magic bytes: "TMAP" in ASCII
MAGIC = b"TMAP"
VERSION = 1


# Section types
class SectionType(IntEnum):
    COORDS = 1
    NUMERIC_COLUMN = 2
    STRING_COLUMN = 3
    METADATA = 4


@dataclass
class SectionEntry:
    """Entry in the section table."""

    section_type: SectionType
    name: str  # Column name (empty for coords/metadata)
    offset: int  # Byte offset from start of file
    compressed_size: int
    uncompressed_size: int
    dtype: str  # 'uint16', 'float32', 'int32', 'string'


def quantize_coords(coords: NDArray[np.float64], bits: int = 16) -> NDArray[Any]:
    """Quantize normalized [-1, 1] coordinates to unsigned integers.

    Args:
        coords: Shape (n, 2) array of normalized coordinates in [-1, 1]
        bits: Quantization bits (16 or 32)

    Returns:
        Quantized coordinates as uint16 or uint32
    """
    if bits == 16:
        max_val = 65535
        return ((coords.astype(np.float64) + 1.0) * (max_val / 2.0)).astype(np.uint16)
    if bits == 32:
        max_val = 4294967295
        return ((coords.astype(np.float64) + 1.0) * (max_val / 2.0)).astype(np.uint32)
    raise ValueError(f"bits must be 16 or 32, got {bits}")


def dequantize_coords(quantized: NDArray[Any], bits: int = 16) -> NDArray[np.float32]:
    """Dequantize coordinates back to float32.

    This is primarily for verification; the JS worker does this.
    """
    if bits == 16:
        max_val = 65535
    else:
        max_val = 4294967295

    # Map [0, max_val] back to [-1, 1]
    return (quantized.astype(np.float64) / (max_val / 2.0) - 1.0).astype(np.float32)


def pack_coords(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bits: int = 16,
) -> tuple[bytes, int]:
    """Pack coordinates as gzip-compressed quantized integers.

    Args:
        x: X coordinates (normalized to [-1, 1])
        y: Y coordinates (normalized to [-1, 1])
        bits: Quantization bits (16 default)

    Returns:
        (compressed_bytes, uncompressed_size)
    """
    # Stack as interleaved [x0, y0, x1, y1, ...]
    coords = np.column_stack([x, y]).astype(np.float64)
    quantized = quantize_coords(coords, bits)

    # Flatten to interleaved format
    raw = quantized.flatten().tobytes()
    compressed = gzip.compress(raw, compresslevel=6)

    return compressed, len(raw)


def pack_numeric_column(
    values: NDArray[Any],
    dtype: str = "float32",
) -> tuple[bytes, int]:
    """Pack a numeric column as gzip-compressed typed array.

    Args:
        values: Column values
        dtype: Target dtype ('float32', 'int32')

    Returns:
        (compressed_bytes, uncompressed_size)
    """
    if dtype == "float32":
        arr = values.astype(np.float32)
    elif dtype == "int32":
        arr = values.astype(np.int32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    raw = arr.tobytes()
    compressed = gzip.compress(raw, compresslevel=6)

    return compressed, len(raw)


def pack_categorical_column(
    values: list[Any],
) -> tuple[bytes, int, list[str]]:
    """Pack a categorical column using dictionary encoding.

    Args:
        values: Column values (will be converted to strings)

    Returns:
        (compressed_indices, uncompressed_size, dictionary)
    """
    # Build dictionary
    unique_values: list[str] = []
    value_to_idx: dict[str, int] = {}
    indices: NDArray[np.uint32] = np.empty(len(values), dtype=np.uint32)

    for i, v in enumerate(values):
        s = str(v)
        if s not in value_to_idx:
            value_to_idx[s] = len(unique_values)
            unique_values.append(s)
        indices[i] = value_to_idx[s]

    raw = indices.tobytes()
    compressed = gzip.compress(raw, compresslevel=6)

    return compressed, len(raw), unique_values


class BinaryContainerWriter:
    """Writes TMAP binary container format."""

    def __init__(self) -> None:
        self.sections: list[tuple[SectionEntry, bytes]] = []
        self.metadata: dict[str, Any] = {}

    def add_coords(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        bits: int = 16,
    ) -> None:
        """Add coordinate section."""
        compressed, uncompressed = pack_coords(x, y, bits)
        dtype = "uint16" if bits == 16 else "uint32"

        entry = SectionEntry(
            section_type=SectionType.COORDS,
            name="",
            offset=0,  # Will be set during write
            compressed_size=len(compressed),
            uncompressed_size=uncompressed,
            dtype=dtype,
        )
        self.sections.append((entry, compressed))

    def add_numeric_column(
        self,
        name: str,
        values: NDArray[Any],
        dtype: str = "float32",
    ) -> None:
        """Add a numeric column section."""
        compressed, uncompressed = pack_numeric_column(values, dtype)

        entry = SectionEntry(
            section_type=SectionType.NUMERIC_COLUMN,
            name=name,
            offset=0,
            compressed_size=len(compressed),
            uncompressed_size=uncompressed,
            dtype=dtype,
        )
        self.sections.append((entry, compressed))

    def add_categorical_column(
        self,
        name: str,
        values: list[Any],
    ) -> None:
        """Add a categorical column with dictionary encoding."""
        compressed, uncompressed, dictionary = pack_categorical_column(values)

        # Store dictionary in metadata
        self.metadata[f"dict_{name}"] = dictionary

        entry = SectionEntry(
            section_type=SectionType.NUMERIC_COLUMN,  # Indices are numeric
            name=name,
            offset=0,
            compressed_size=len(compressed),
            uncompressed_size=uncompressed,
            dtype="uint32",  # Dictionary indices
        )
        self.sections.append((entry, compressed))

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Set visualization metadata (title, colormaps, etc.)."""
        self.metadata.update(metadata)

    def write(self) -> bytes:
        """Write the complete binary container."""
        buffer = BytesIO()

        # Prepare metadata section
        metadata_json = json.dumps(self.metadata, separators=(",", ":")).encode("utf-8")
        metadata_compressed = gzip.compress(metadata_json, compresslevel=6)

        # Calculate section table
        # Header: 32 bytes
        # Section count: 4 bytes
        # Section entries: 64 bytes each (type:4, name_len:4, name:32, offset:8, comp:8, uncomp:8)
        header_size = 32
        section_count = len(self.sections) + 1  # +1 for metadata
        section_table_size = 4 + section_count * 64

        # Calculate offsets
        data_start = header_size + section_table_size
        current_offset = data_start

        # Metadata section entry
        metadata_entry = SectionEntry(
            section_type=SectionType.METADATA,
            name="",
            offset=current_offset,
            compressed_size=len(metadata_compressed),
            uncompressed_size=len(metadata_json),
            dtype="json",
        )
        current_offset += len(metadata_compressed)

        # Update offsets for data sections
        for entry, data in self.sections:
            entry.offset = current_offset
            current_offset += len(data)

        # Write header
        buffer.write(MAGIC)  # 4 bytes
        buffer.write(struct.pack("<I", VERSION))  # 4 bytes
        buffer.write(struct.pack("<Q", len(self.sections)))  # 8 bytes: n_points placeholder
        buffer.write(struct.pack("<Q", data_start))  # 8 bytes: data start offset
        buffer.write(b"\x00" * 8)  # 8 bytes: reserved

        # Write section count
        buffer.write(struct.pack("<I", section_count))

        # Write section table
        def write_section_entry(entry: SectionEntry) -> None:
            buffer.write(struct.pack("<I", entry.section_type))
            name_bytes = entry.name.encode("utf-8")[:32].ljust(32, b"\x00")
            buffer.write(struct.pack("<I", len(entry.name)))
            buffer.write(name_bytes)
            buffer.write(struct.pack("<Q", entry.offset))
            buffer.write(struct.pack("<Q", entry.compressed_size))
            buffer.write(struct.pack("<Q", entry.uncompressed_size))

        write_section_entry(metadata_entry)
        for entry, _ in self.sections:
            write_section_entry(entry)

        # Write metadata
        buffer.write(metadata_compressed)

        # Write data sections
        for _, data in self.sections:
            buffer.write(data)

        return buffer.getvalue()

    def write_chunked(self) -> dict[str, bytes]:
        """Write sections as separate chunks for HTML embedding.

        Returns dict with keys: 'header', 'metadata', 'coords', 'col_<name>', ...
        """
        chunks: dict[str, bytes] = {}

        # Metadata
        metadata_json = json.dumps(self.metadata, separators=(",", ":")).encode("utf-8")
        chunks["metadata"] = gzip.compress(metadata_json, compresslevel=6)

        # Sections
        for entry, data in self.sections:
            if entry.section_type == SectionType.COORDS:
                chunks["coords"] = data
            else:
                chunks[f"col_{entry.name}"] = data

        # Header (section info for JS decoder)
        header = {
            "version": VERSION,
            "sections": [
                {
                    "type": int(entry.section_type),
                    "name": entry.name,
                    "dtype": entry.dtype,
                    "compressed_size": len(data),
                    "uncompressed_size": entry.uncompressed_size,
                }
                for entry, data in self.sections
            ],
        }
        chunks["header"] = json.dumps(header, separators=(",", ":")).encode("utf-8")

        return chunks


def create_binary_payload(
    points: NDArray[np.float64],
    columns: dict[str, tuple[NDArray[Any] | list[Any], str]],  # name -> (values, dtype)
    metadata: dict[str, Any],
    coord_bits: int = 16,
) -> dict[str, bytes]:
    """Create binary payload chunks for HTML embedding.

    Args:
        points: Shape (n, 2) normalized coordinates
        columns: Dict of column_name -> (values, dtype)
                 dtype is 'float32', 'int32', or 'categorical'
        metadata: Visualization metadata (title, colormaps, etc.)
        coord_bits: Quantization bits for coordinates

    Returns:
        Dict of chunk_name -> compressed_bytes
    """
    writer = BinaryContainerWriter()

    # Add coordinates
    writer.add_coords(points[:, 0], points[:, 1], bits=coord_bits)

    # Add columns
    for name, (values, dtype) in columns.items():
        if dtype == "categorical":
            writer.add_categorical_column(name, list(values))
        else:
            arr = np.asarray(values)
            writer.add_numeric_column(name, arr, dtype)

    # Add metadata
    writer.set_metadata(metadata)

    return writer.write_chunked()
