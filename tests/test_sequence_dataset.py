"""Tests for the nucleotide-only training path used by single-modality backbones."""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("datasets")

from minerva.backbones import get_backbone  # noqa: E402
from minerva.finetuning import load_sequence_dataset, read_sequences  # noqa: E402

GENBANK = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "UG27_systems.gb")


def _fasta(tmp, records):
    path = os.path.join(tmp, "seqs.fasta")
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq}\n")
    return path


def test_reads_plain_nucleotides_from_fasta():
    with tempfile.TemporaryDirectory() as tmp:
        path = _fasta(tmp, [("a", "ACGUACGU"), ("b", "GGGGCCCC")])
        assert read_sequences(path) == ["ACGUACGU", "GGGGCCCC"]


@pytest.mark.skipif(not os.path.exists(GENBANK), reason="example GenBank not present")
def test_genbank_yields_sequence_not_translations():
    """GenBank is a fine source: take record.seq, ignore the CDS features."""
    seqs = read_sequences(GENBANK)
    assert seqs
    joined = "".join(seqs).upper()
    assert set(joined) <= set("ACGTUN")          # nucleotides only
    assert "<+>" not in joined and "<-" not in joined


def test_blocks_long_records_instead_of_truncating():
    with tempfile.TemporaryDirectory() as tmp:
        path = _fasta(tmp, [("long", "ACGT" * 100)])       # 400 nt
        ds = load_sequence_dataset(path, block_size=64)
        texts = ds["train"]["text"]
        assert len(texts) > 1                              # tiled, not truncated
        assert sum(len(t) for t in texts) == 400           # nothing dropped


def test_small_datasets_skip_the_validation_split():
    with tempfile.TemporaryDirectory() as tmp:
        ds = load_sequence_dataset(_fasta(tmp, [("a", "ACGT")]))
        assert "validation" not in ds


def test_separate_validation_file_is_used():
    with tempfile.TemporaryDirectory() as tmp:
        train = _fasta(tmp, [("a", "ACGT"), ("b", "TGCA")])
        val = os.path.join(tmp, "val.fasta")
        with open(val, "w") as fh:
            fh.write(">v\nGGGG\n")
        ds = load_sequence_dataset(train, validation_file=val)
        assert ds["validation"]["text"] == ["GGGG"]


def test_nucleotide_backbone_rejects_mixed_modality_text():
    """The guard that stops protein silently encoding as IUPAC bases."""
    rinalmo = get_backbone("rinalmo")
    with pytest.raises(ValueError, match="nucleotide-only"):
        rinalmo.reject_mixed_modality("<+>MALTKVEK<+>acgt")
    rinalmo.reject_mixed_modality("ACGUACGU")          # plain nucleotides are fine


def test_mixed_backbone_accepts_mixed_modality_text():
    get_backbone("minerva").reject_mixed_modality("<+>MALTKVEK<+>acgt")
