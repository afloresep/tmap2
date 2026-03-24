"""Tests for tmap.utils.proteins module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tmap.utils.proteins import (
    AVAILABLE_SEQUENCE_PROPERTIES,
    DEFAULT_FIELDS,
    _compute_pI,
    _fetch_one_alphafold,
    _is_valid_sequence,
    _net_charge_at_ph,
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

# ---------------------------------------------------------------------------
# Known sequences for validation
# ---------------------------------------------------------------------------

# Human insulin B chain (30 residues)
INSULIN_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"

# Human ubiquitin (76 residues) — well-characterized protein
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


class TestIsValidSequence:
    def test_valid(self):
        assert _is_valid_sequence("ACDEFGHIKLMNPQRSTVWY")

    def test_invalid_character(self):
        assert not _is_valid_sequence("ACXDEF")

    def test_empty(self):
        assert not _is_valid_sequence("")

    def test_lowercase_not_valid(self):
        # The function checks exact characters; caller uppercases
        assert not _is_valid_sequence("acde")

    def test_numbers(self):
        assert not _is_valid_sequence("AC1DE")


class TestNetCharge:
    def test_neutral_at_pI(self):
        """At the pI, charge should be ~0."""
        pi = _compute_pI(UBIQUITIN)
        charge = _net_charge_at_ph(pi, UBIQUITIN)
        assert abs(charge) < 0.01

    def test_positive_at_low_ph(self):
        charge = _net_charge_at_ph(2.0, UBIQUITIN)
        assert charge > 0

    def test_negative_at_high_ph(self):
        charge = _net_charge_at_ph(12.0, UBIQUITIN)
        assert charge < 0


class TestComputePI:
    def test_ubiquitin_pI(self):
        """Ubiquitin pI varies by pKa table (~6.6-7.7). Ours uses textbook values."""
        pi = _compute_pI(UBIQUITIN)
        assert 6.0 < pi < 8.5

    def test_insulin_b_pI(self):
        """Insulin B chain — basic protein, pI around 6-8."""
        pi = _compute_pI(INSULIN_B)
        assert 5.0 < pi < 9.0


class TestSequenceProperties:
    def test_basic_output_shape(self):
        props = sequence_properties(["ACDE", "FGHI"])
        assert set(props.keys()) == set(AVAILABLE_SEQUENCE_PROPERTIES)
        for v in props.values():
            assert len(v) == 2
            assert v.dtype == np.float64

    def test_length(self):
        props = sequence_properties(["ACDE", "FGHIKLMN"])
        np.testing.assert_array_equal(props["length"], [4, 8])

    def test_molecular_weight_positive(self):
        props = sequence_properties([UBIQUITIN])
        mw = props["molecular_weight"][0]
        # Ubiquitin MW is ~8565 Da
        assert 8400 < mw < 8700

    def test_molecular_weight_glycine(self):
        """Single glycine: residue mass (57.02146) + water (18.01056) ≈ 75.03."""
        props = sequence_properties(["G"])
        mw = props["molecular_weight"][0]
        assert abs(mw - 75.032) < 0.01

    def test_gravy(self):
        """GRAVY for all-Isoleucine should be 4.5 (highest KD value)."""
        props = sequence_properties(["IIIII"])
        assert abs(props["gravy"][0] - 4.5) < 1e-10

    def test_gravy_all_arg(self):
        """GRAVY for all-Arginine should be -4.5 (lowest KD value)."""
        props = sequence_properties(["RRRRR"])
        assert abs(props["gravy"][0] - (-4.5)) < 1e-10

    def test_invalid_sequence_nan(self):
        props = sequence_properties(["ACDE", "INVALID_X_SEQ", "FGHI"])
        assert np.isnan(props["molecular_weight"][1])
        assert np.isnan(props["isoelectric_point"][1])
        assert np.isnan(props["gravy"][1])
        # Valid sequences should NOT be NaN
        assert not np.isnan(props["molecular_weight"][0])
        assert not np.isnan(props["molecular_weight"][2])

    def test_empty_sequence_nan(self):
        props = sequence_properties(["", "ACDE"])
        assert np.isnan(props["molecular_weight"][0])
        assert not np.isnan(props["molecular_weight"][1])

    def test_empty_input(self):
        props = sequence_properties([])
        assert len(props["length"]) == 0
        assert len(props["molecular_weight"]) == 0

    def test_lowercase_accepted(self):
        """Lowercase input should be uppercased internally."""
        props = sequence_properties(["acde"])
        assert not np.isnan(props["molecular_weight"][0])

    def test_charge_at_ph7_sign(self):
        """All-lysine peptide should be very positive at pH 7."""
        props = sequence_properties(["KKKKK"])
        assert props["charge_at_ph7"][0] > 4.0

    def test_charge_at_ph7_acidic(self):
        """All-aspartate peptide should be very negative at pH 7."""
        props = sequence_properties(["DDDDD"])
        assert props["charge_at_ph7"][0] < -4.0


class TestSequencePropertiesExport:
    def test_import_from_utils(self):
        from tmap.utils import sequence_properties as sp

        assert callable(sp)

    def test_import_from_top_level(self):
        from tmap import sequence_properties as sp

        assert callable(sp)


# ---------------------------------------------------------------------------
# fetch_uniprot tests (mocked)
# ---------------------------------------------------------------------------

_MOCK_TSV_RESPONSE = (
    "Entry\tProtein names\tOrganism\tAnnotation\tLength\t"
    "EC number\tSubcellular location [CC]\t"
    "Gene Ontology (molecular function)\tGene Ontology (biological process)\n"
    "P12345\tAspartate aminotransferase\tOryctolagus cuniculus\t5\t413\t"
    "2.6.1.1\tMitochondrion matrix\t"
    "L-aspartate:2-oxoglutarate aminotransferase activity [GO:0004069]\t"
    "aspartate catabolic process [GO:0006533]\n"
)


class TestFetchUniprot:
    def test_empty_input(self):
        result = fetch_uniprot([])
        for field in DEFAULT_FIELDS:
            assert len(result[field]) == 0

    def test_invalid_ids_skipped(self):
        """Invalid IDs should be skipped with a warning."""
        with patch("tmap.utils.proteins.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = _MOCK_TSV_RESPONSE.encode("utf-8")
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = fetch_uniprot(["INVALID", "P12345"])
            # Should still have results for P12345
            assert len(result["accession"]) == 2
            # First entry (invalid) should be empty string
            assert result["accession"][0] == ""
            # Second entry should be P12345
            assert result["accession"][1] == "P12345"

    def test_numeric_fields_are_float(self):
        with patch("tmap.utils.proteins.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = _MOCK_TSV_RESPONSE.encode("utf-8")
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = fetch_uniprot(["P12345"])
            assert result["annotation_score"].dtype == np.float64
            assert result["length"].dtype == np.float64
            assert result["annotation_score"][0] == 5.0
            assert result["length"][0] == 413.0

    def test_text_fields_are_object(self):
        with patch("tmap.utils.proteins.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = _MOCK_TSV_RESPONSE.encode("utf-8")
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = fetch_uniprot(["P12345"])
            assert result["organism_name"].dtype == object
            assert result["organism_name"][0] == "Oryctolagus cuniculus"

    def test_chunk_failure_graceful(self):
        """If a chunk fails, other chunks should still succeed."""
        call_count = 0

        def mock_urlopen_side_effect(req, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise urllib.error.URLError("Connection refused")
            mock_resp = MagicMock()
            mock_resp.read.return_value = _MOCK_TSV_RESPONSE.encode("utf-8")
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        import urllib.error

        with patch(
            "tmap.utils.proteins.urllib.request.urlopen", side_effect=mock_urlopen_side_effect
        ):
            # Two chunks of 1 each — first fails, second succeeds
            result = fetch_uniprot(["P12345", "Q9NZC2"], chunk_size=1)
            # Should have partial results
            assert len(result["accession"]) == 2

    def test_export_from_utils(self):
        from tmap.utils import fetch_uniprot as fu

        assert callable(fu)

    def test_export_from_top_level(self):
        from tmap import fetch_uniprot as fu

        assert callable(fu)


# ---------------------------------------------------------------------------
# fetch_alphafold tests (mocked)
# ---------------------------------------------------------------------------

_MOCK_ALPHAFOLD_JSON = json.dumps(
    [
        {
            "sequenceEnd": 350,
            "globalMetricValue": 92.5,
            "fractionPlddtVeryLow": 0.03,
            "fractionPlddtVeryHigh": 0.78,
        }
    ]
)


class TestFetchOneAlphafold:
    def test_success(self):
        with patch("tmap.utils.proteins.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = _MOCK_ALPHAFOLD_JSON.encode("utf-8")
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            uid, result = _fetch_one_alphafold("P12345")
            assert uid == "P12345"
            assert result is not None
            assert result["length"] == 350.0
            assert result["plddt"] == 92.5
            assert abs(result["frac_disordered"] - 0.03) < 1e-6
            assert abs(result["frac_confident"] - 0.78) < 1e-6

    def test_failure_returns_none(self):
        with patch("tmap.utils.proteins.urllib.request.urlopen", side_effect=Exception("timeout")):
            uid, result = _fetch_one_alphafold("P99999")
            assert uid == "P99999"
            assert result is None


class TestFetchAlphafold:
    def test_empty_input(self):
        result = fetch_alphafold([])
        assert set(result.keys()) == {"length", "plddt", "frac_disordered", "frac_confident"}
        for v in result.values():
            assert len(v) == 0
            assert v.dtype == np.float32

    def test_output_dtypes(self):
        with patch("tmap.utils.proteins._fetch_one_alphafold") as mock_fetch:
            mock_fetch.return_value = (
                "P12345",
                {
                    "length": 350.0,
                    "plddt": 92.5,
                    "frac_disordered": 0.03,
                    "frac_confident": 0.78,
                },
            )
            result = fetch_alphafold(["P12345"])
            for v in result.values():
                assert v.dtype == np.float32
                assert len(v) == 1

    def test_failed_fetch_gives_nan(self):
        with patch("tmap.utils.proteins._fetch_one_alphafold") as mock_fetch:
            mock_fetch.return_value = ("P99999", None)
            result = fetch_alphafold(["P99999"])
            for v in result.values():
                assert np.isnan(v[0])

    def test_mixed_success_failure(self):
        def mock_side_effect(uid):
            if uid == "P12345":
                return uid, {
                    "length": 200.0,
                    "plddt": 85.0,
                    "frac_disordered": 0.1,
                    "frac_confident": 0.6,
                }
            return uid, None

        with patch("tmap.utils.proteins._fetch_one_alphafold", side_effect=mock_side_effect):
            result = fetch_alphafold(["P12345", "BADID"])
            assert not np.isnan(result["length"][0])
            assert np.isnan(result["length"][1])

    def test_export_from_utils(self):
        from tmap.utils import fetch_alphafold as fa

        assert callable(fa)

    def test_export_from_top_level(self):
        from tmap import fetch_alphafold as fa

        assert callable(fa)


# ---------------------------------------------------------------------------
# File reader tests
# ---------------------------------------------------------------------------


class TestReadFasta:
    def test_basic(self, tmp_path):
        fasta = tmp_path / "test.fa"
        fasta.write_text(">sp|P12345|PROT1 Some protein\nACDE\nFGHI\n>P99999\nKLMN\n")
        ids, seqs = read_fasta(fasta)
        assert ids == ["sp|P12345|PROT1", "P99999"]
        assert seqs == ["ACDEFGHI", "KLMN"]

    def test_max_seqs(self, tmp_path):
        fasta = tmp_path / "test.fa"
        fasta.write_text(">A\nACDE\n>B\nFGHI\n>C\nKLMN\n")
        ids, seqs = read_fasta(fasta, max_seqs=2)
        assert len(ids) == 2
        assert len(seqs) == 2

    def test_empty_file(self, tmp_path):
        fasta = tmp_path / "empty.fa"
        fasta.write_text("")
        ids, seqs = read_fasta(fasta)
        assert ids == []
        assert seqs == []

    def test_uppercase(self, tmp_path):
        fasta = tmp_path / "lower.fa"
        fasta.write_text(">id1\nacde\n")
        _, seqs = read_fasta(fasta)
        assert seqs == ["ACDE"]

    def test_blank_lines_skipped(self, tmp_path):
        fasta = tmp_path / "blank.fa"
        fasta.write_text(">id1\nACDE\n\nFGHI\n\n>id2\nKLMN\n")
        ids, seqs = read_fasta(fasta)
        assert ids == ["id1", "id2"]
        assert seqs == ["ACDEFGHI", "KLMN"]

    def test_export(self):
        from tmap.utils import read_fasta as rf

        assert callable(rf)
        from tmap import read_fasta as rf2

        assert callable(rf2)


class TestReadProteinCsv:
    def test_csv(self, tmp_path):
        f = tmp_path / "proteins.csv"
        f.write_text("id,sequence,other\nP12345,ACDE,x\nQ99999,FGHI,y\n")
        ids, seqs = read_protein_csv(f)
        assert ids == ["P12345", "Q99999"]
        assert seqs == ["ACDE", "FGHI"]

    def test_tsv(self, tmp_path):
        f = tmp_path / "proteins.tsv"
        f.write_text("id\tsequence\nP12345\tacde\n")
        ids, seqs = read_protein_csv(f)
        assert seqs == ["ACDE"]  # uppercased

    def test_custom_columns(self, tmp_path):
        f = tmp_path / "custom.csv"
        f.write_text("accession,seq,desc\nP12345,ACDE,test\n")
        ids, seqs = read_protein_csv(f, id_col="accession", seq_col="seq")
        assert ids == ["P12345"]
        assert seqs == ["ACDE"]

    def test_missing_column_raises(self, tmp_path):
        f = tmp_path / "bad.csv"
        f.write_text("accession,seq\nP12345,ACDE\n")
        with pytest.raises(ValueError, match="id"):
            read_protein_csv(f)  # default id_col="id" not found

    def test_export(self):
        from tmap.utils import read_protein_csv as rc

        assert callable(rc)
        from tmap import read_protein_csv as rc2

        assert callable(rc2)


class TestReadIdList:
    def test_basic(self, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text("P12345\nQ9NZC2\nA0A6A4IZ81\n")
        ids = read_id_list(f)
        assert ids == ["P12345", "Q9NZC2", "A0A6A4IZ81"]

    def test_comments_and_blanks(self, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text("# header\nP12345\n\n# comment\nQ9NZC2\n")
        ids = read_id_list(f)
        assert ids == ["P12345", "Q9NZC2"]

    def test_trailing_whitespace(self, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text("P12345  some note\nQ9NZC2\n")
        ids = read_id_list(f)
        assert ids == ["P12345", "Q9NZC2"]

    def test_invalid_ids_still_included(self, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text("P12345\nNOT_VALID\n")
        ids = read_id_list(f)
        assert len(ids) == 2  # both included, warning logged

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        ids = read_id_list(f)
        assert ids == []

    def test_export(self):
        from tmap.utils import read_id_list as rl

        assert callable(rl)
        from tmap import read_id_list as rl2

        assert callable(rl2)


# ---------------------------------------------------------------------------
# read_pdb
# ---------------------------------------------------------------------------

# Minimal synthetic PDB content (3 CA atoms, chain A)
_MINIMAL_PDB = """\
HEADER    TEST PROTEIN                            01-JAN-00   1TST
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 85.00
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 85.00
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 85.00
ATOM      4  N   GLY A   2       4.000   5.000   6.000  1.00 92.00
ATOM      5  CA  GLY A   2       5.000   6.000   7.000  1.00 92.00
ATOM      6  C   GLY A   2       6.000   7.000   8.000  1.00 92.00
ATOM      7  N   VAL A   3       7.000   8.000   9.000  1.00 40.00
ATOM      8  CA  VAL A   3       8.000   9.000  10.000  1.00 40.00
ATOM      9  C   VAL A   3       9.000  10.000  11.000  1.00 40.00
END
"""

_MULTICHAIN_PDB = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00
ATOM      2  CA  GLY A   2       2.000   3.000   4.000  1.00 95.00
ATOM      3  CA  LEU B   1       3.000   4.000   5.000  1.00 30.00
ATOM      4  CA  PRO B   2       4.000   5.000   6.000  1.00 35.00
END
"""


class TestReadPdb:
    def test_basic(self, tmp_path):
        f = tmp_path / "test.pdb"
        f.write_text(_MINIMAL_PDB)
        result = read_pdb(f)
        assert result["id"] == "1TST"
        assert result["sequence"] == "AGV"
        assert result["length"] == 3
        assert abs(result["mean_bfactor"] - (85 + 92 + 40) / 3) < 0.01
        # 1 of 3 residues > 90
        assert abs(result["frac_bfactor_high"] - 1 / 3) < 0.01
        # 1 of 3 residues < 50
        assert abs(result["frac_bfactor_low"] - 1 / 3) < 0.01

    def test_chain_filter(self, tmp_path):
        f = tmp_path / "multi.pdb"
        f.write_text(_MULTICHAIN_PDB)
        result_a = read_pdb(f, chain="A")
        assert result_a["sequence"] == "AG"
        assert result_a["length"] == 2

        result_b = read_pdb(f, chain="B")
        assert result_b["sequence"] == "LP"
        assert result_b["length"] == 2

    def test_all_chains(self, tmp_path):
        f = tmp_path / "multi.pdb"
        f.write_text(_MULTICHAIN_PDB)
        result = read_pdb(f)
        assert result["sequence"] == "AGLP"
        assert result["length"] == 4

    def test_no_header_uses_filename(self, tmp_path):
        pdb_content = "ATOM      1  CA  ALA A   1       1.0   2.0   3.0  1.00 80.00\nEND\n"
        f = tmp_path / "myprotein.pdb"
        f.write_text(pdb_content)
        result = read_pdb(f)
        assert result["id"] == "myprotein"

    def test_empty_pdb(self, tmp_path):
        f = tmp_path / "empty.pdb"
        f.write_text("END\n")
        result = read_pdb(f)
        assert result["sequence"] == ""
        assert result["length"] == 0
        assert np.isnan(result["mean_bfactor"])

    def test_non_standard_residues_skipped(self, tmp_path):
        pdb_content = (
            "ATOM      1  CA  ALA A   1       1.0   2.0   3.0  1.00 80.00\n"
            "ATOM      2  CA  UNK A   2       2.0   3.0   4.0  1.00 70.00\n"
            "ATOM      3  CA  GLY A   3       3.0   4.0   5.0  1.00 60.00\n"
            "END\n"
        )
        f = tmp_path / "test.pdb"
        f.write_text(pdb_content)
        result = read_pdb(f)
        assert result["sequence"] == "AG"  # UNK skipped
        assert result["length"] == 2

    def test_hetatm_ignored(self, tmp_path):
        pdb_content = (
            "ATOM      1  CA  ALA A   1       1.0   2.0   3.0  1.00 80.00\n"
            "HETATM    2  CA  ALA A   2       2.0   3.0   4.0  1.00 70.00\n"
            "END\n"
        )
        f = tmp_path / "test.pdb"
        f.write_text(pdb_content)
        result = read_pdb(f)
        assert result["length"] == 1  # HETATM ignored

    def test_export(self):
        from tmap.utils import read_pdb as rp

        assert callable(rp)
        from tmap import read_pdb as rp2

        assert callable(rp2)


class TestReadPdbDir:
    def test_basic(self, tmp_path):
        for i, (seq_line, bf) in enumerate(
            [
                ("ALA", "80.00"),
                ("GLY", "95.00"),
                ("VAL", "30.00"),
            ]
        ):
            content = f"ATOM      1  CA  {seq_line} A   1       1.0   2.0   3.0  1.00 {bf}\nEND\n"
            (tmp_path / f"prot{i}.pdb").write_text(content)

        ids, seqs, props = read_pdb_dir(tmp_path)
        assert len(ids) == 3
        assert len(seqs) == 3
        assert props["mean_bfactor"].shape == (3,)
        assert props["length"].shape == (3,)

    def test_empty_dir(self, tmp_path):
        ids, seqs, props = read_pdb_dir(tmp_path)
        assert ids == []
        assert seqs == []
        assert props["mean_bfactor"].shape == (0,)

    def test_pattern_filter(self, tmp_path):
        (tmp_path / "a.pdb").write_text(
            "ATOM      1  CA  ALA A   1       1.0   2.0   3.0  1.00 80.00\nEND\n"
        )
        (tmp_path / "b.ent").write_text(
            "ATOM      1  CA  GLY A   1       1.0   2.0   3.0  1.00 90.00\nEND\n"
        )
        ids, seqs, props = read_pdb_dir(tmp_path, pattern="*.ent")
        assert len(ids) == 1
        assert seqs[0] == "G"

    def test_export(self):
        from tmap.utils import read_pdb_dir as rpd

        assert callable(rpd)
        from tmap import read_pdb_dir as rpd2

        assert callable(rpd2)


# ---------------------------------------------------------------------------
# parse_alignment
# ---------------------------------------------------------------------------


def _make_m8_line(
    qid, sid, pident, length, mismatch, gapopen, qstart, qend, sstart, send, evalue, bitscore
):
    """Build a single BLAST6 m8 line."""
    return "\t".join(
        str(x)
        for x in [
            qid,
            sid,
            pident,
            length,
            mismatch,
            gapopen,
            qstart,
            qend,
            sstart,
            send,
            evalue,
            bitscore,
        ]
    )


class TestParseAlignment:
    def _write_m8(self, tmp_path, lines):
        f = tmp_path / "hits.m8"
        f.write_text("\n".join(lines) + "\n")
        return f

    def test_basic(self, tmp_path):
        lines = [
            _make_m8_line("A", "B", 95.0, 100, 5, 0, 1, 100, 1, 100, 1e-50, 200),
            _make_m8_line("A", "C", 80.0, 90, 18, 0, 1, 90, 1, 90, 1e-30, 150),
            _make_m8_line("B", "A", 95.0, 100, 5, 0, 1, 100, 1, 100, 1e-50, 200),
            _make_m8_line("B", "C", 70.0, 80, 24, 0, 1, 80, 1, 80, 1e-20, 100),
        ]
        f = self._write_m8(tmp_path, lines)
        knn, ids = parse_alignment(f, k=2)
        assert len(ids) == 3  # A, B, C
        assert knn.indices.shape[0] == 3
        assert knn.indices.shape[1] == 2

    def test_self_hits_excluded(self, tmp_path):
        lines = [
            _make_m8_line("A", "A", 100.0, 100, 0, 0, 1, 100, 1, 100, 0, 500),
            _make_m8_line("A", "B", 80.0, 90, 18, 0, 1, 90, 1, 90, 1e-30, 150),
        ]
        f = self._write_m8(tmp_path, lines)
        knn, ids = parse_alignment(f, k=5)
        # A→A should be excluded, only A→B remains
        a_idx = ids.index("A")
        # First neighbor of A should be B, rest should be -1
        assert knn.indices[a_idx, 0] == ids.index("B")
        assert knn.indices[a_idx, 1] == -1

    def test_k_truncation(self, tmp_path):
        # A has 5 hits but k=2
        lines = [
            _make_m8_line("A", "B", 90, 100, 10, 0, 1, 100, 1, 100, 1e-40, 300),
            _make_m8_line("A", "C", 85, 100, 15, 0, 1, 100, 1, 100, 1e-35, 250),
            _make_m8_line("A", "D", 80, 100, 20, 0, 1, 100, 1, 100, 1e-30, 200),
            _make_m8_line("A", "E", 75, 100, 25, 0, 1, 100, 1, 100, 1e-25, 150),
            _make_m8_line("A", "F", 70, 100, 30, 0, 1, 100, 1, 100, 1e-20, 100),
        ]
        f = self._write_m8(tmp_path, lines)
        knn, ids = parse_alignment(f, k=2)
        a_idx = ids.index("A")
        # Should only have top-2 by bitscore (B=300, C=250)
        assert knn.indices[a_idx, 0] == ids.index("B")
        assert knn.indices[a_idx, 1] == ids.index("C")

    def test_fewer_than_k_hits_padded(self, tmp_path):
        lines = [
            _make_m8_line("A", "B", 90, 100, 10, 0, 1, 100, 1, 100, 1e-40, 300),
        ]
        f = self._write_m8(tmp_path, lines)
        knn, ids = parse_alignment(f, k=5)
        a_idx = ids.index("A")
        assert knn.indices[a_idx, 0] == ids.index("B")
        assert knn.indices[a_idx, 1] == -1
        assert np.isinf(knn.distances[a_idx, 1])

    def test_score_col_pident(self, tmp_path):
        lines = [
            _make_m8_line("A", "B", 95, 100, 5, 0, 1, 100, 1, 100, 1e-40, 100),
            _make_m8_line("A", "C", 80, 100, 20, 0, 1, 100, 1, 100, 1e-30, 300),
        ]
        f = self._write_m8(tmp_path, lines)
        # By bitscore: C first (300 > 100)
        knn_bs, ids = parse_alignment(f, k=2, score_col="bitscore")
        a_idx = ids.index("A")
        assert knn_bs.indices[a_idx, 0] == ids.index("C")

        # By pident: B first (95 > 80)
        knn_pi, ids2 = parse_alignment(f, k=2, score_col="pident")
        a_idx2 = ids2.index("A")
        assert knn_pi.indices[a_idx2, 0] == ids2.index("B")

    def test_as_distance_conversion(self, tmp_path):
        lines = [
            _make_m8_line("A", "B", 90, 100, 10, 0, 1, 100, 1, 100, 1e-40, 200),
        ]
        f = self._write_m8(tmp_path, lines)

        knn_dist, _ = parse_alignment(f, k=5, as_distance=True)
        knn_raw, _ = parse_alignment(f, k=5, as_distance=False)

        a_idx = 0
        # as_distance: 1/(1+200) ≈ 0.00497
        assert abs(knn_dist.distances[a_idx, 0] - 1.0 / 201.0) < 1e-4
        # raw: 200.0
        assert abs(knn_raw.distances[a_idx, 0] - 200.0) < 1e-4

    def test_duplicate_pairs_keep_best(self, tmp_path):
        lines = [
            _make_m8_line("A", "B", 80, 100, 20, 0, 1, 100, 1, 100, 1e-30, 100),
            _make_m8_line("A", "B", 90, 100, 10, 0, 1, 100, 1, 100, 1e-40, 200),
        ]
        f = self._write_m8(tmp_path, lines)
        knn, ids = parse_alignment(f, k=5, as_distance=False)
        a_idx = ids.index("A")
        # Best bitscore is 200
        assert abs(knn.distances[a_idx, 0] - 200.0) < 1e-4

    def test_invalid_score_col_raises(self, tmp_path):
        f = self._write_m8(tmp_path, [])
        with pytest.raises(ValueError, match="score_col"):
            parse_alignment(f, score_col="invalid")

    def test_empty_file_raises(self, tmp_path):
        f = self._write_m8(tmp_path, [])
        with pytest.raises(ValueError, match="No valid hits"):
            parse_alignment(f, k=5)

    def test_export(self):
        from tmap.utils import parse_alignment as pa

        assert callable(pa)
        from tmap import parse_alignment as pa2

        assert callable(pa2)
