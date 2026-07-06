"""Tests for minerva.sequence_utils position-mapping utilities."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from minerva.sequence_utils import (
    _is_dna_segment,
    _parse_segments,
    build_fwd_rc_permutation,
    build_prodigal_mixed_sequence,
    chunk_sequence_with_stride,
    map_features_to_genome,
    mixed_token_length,
    rc_mixed_token_string,
    remap_rc_attention,
    tokenize_genome_window,
    tokenize_mixed_window,
    trim_attention_maps,
    trim_contact_map,
    trim_hidden_states,
    verify_token_mapping,
)


# ---------------------------------------------------------------------------
# Minimal mock tokenizer that reproduces the real mapping:
#   <+> → 33,  a → 29,  c → 31,  g → 32,  t → 30
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Minimal tokenizer matching the Minerva token ID scheme.

    Nucleotides: a→29, c→31, g→32, t→30
    Amino acids:  A→4, C→5, D→6, E→7, F→8, G→9, H→10, I→11, K→12, L→13,
                  M→14, N→15, P→16, Q→17, R→18, S→19, T→20, V→21, W→22, Y→23
    Strand markers: <+>→33, <->→34
    """

    _vocab = {
        "<+>": 33, "<->": 34,
        "a": 29, "c": 31, "g": 32, "t": 30,
        # Amino acids (subset sufficient for tests)
        "A": 4, "C": 5, "D": 6, "E": 7, "F": 8, "G": 9, "H": 10,
        "I": 11, "K": 12, "L": 13, "M": 14, "N": 15, "P": 16,
        "Q": 17, "R": 18, "S": 19, "T": 20, "V": 21, "W": 22, "Y": 23,
    }

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = []
        i = 0
        while i < len(text):
            if text[i:].startswith("<+>"):
                ids.append(self._vocab["<+>"])
                i += 3
            elif text[i:].startswith("<->"):
                ids.append(self._vocab["<->"])
                i += 3
            else:
                ids.append(self._vocab.get(text[i], 0))
                i += 1
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


@pytest.fixture
def tokenizer():
    return _MockTokenizer()


# ---------------------------------------------------------------------------
# tokenize_genome_window
# ---------------------------------------------------------------------------

class TestTokenizeGenomeWindow:
    def test_short_seq(self, tokenizer):
        """'acgt' → 5 tokens: <+>, a, c, g, t."""
        info = tokenize_genome_window("acgt", tokenizer)
        assert info["input_ids"].shape == (1, 5)
        assert info["genome_len"] == 4
        assert info["offset"] == 1
        assert info["input_ids"][0, 0].item() == 33  # <+>

    def test_case_insensitive(self, tokenizer):
        """Upper-case input should be lowered internally."""
        info = tokenize_genome_window("ACGT", tokenizer)
        assert info["input_ids"].shape == (1, 5)
        assert info["genome_len"] == 4

    def test_single_nt(self, tokenizer):
        info = tokenize_genome_window("a", tokenizer)
        assert info["input_ids"].shape == (1, 2)
        assert info["genome_len"] == 1


# ---------------------------------------------------------------------------
# trim helpers
# ---------------------------------------------------------------------------

class TestTrimHiddenStates:
    def test_shape(self):
        h = torch.randn(1, 5, 8)  # batch=1, seq_len+1=5, dim=8
        trimmed = trim_hidden_states(h)
        assert trimmed.shape == (1, 4, 8)

    def test_values(self):
        h = torch.arange(40).reshape(1, 5, 8).float()
        trimmed = trim_hidden_states(h)
        assert torch.equal(trimmed, h[:, 1:, :])


class TestTrimContactMap:
    def test_shape(self):
        c = torch.randn(5, 5)
        trimmed = trim_contact_map(c)
        assert trimmed.shape == (4, 4)

    def test_values(self):
        c = torch.arange(25).reshape(5, 5).float()
        trimmed = trim_contact_map(c)
        assert torch.equal(trimmed, c[1:, 1:])


class TestTrimAttentionMaps:
    def test_shape(self):
        attn = {31: torch.randn(1, 20, 5, 5)}
        trimmed = trim_attention_maps(attn)
        assert 31 in trimmed
        assert trimmed[31].shape == (1, 20, 4, 4)

    def test_values(self):
        raw = torch.arange(500).reshape(1, 20, 5, 5).float()
        attn = {31: raw}
        trimmed = trim_attention_maps(attn)
        assert torch.equal(trimmed[31], raw[:, :, 1:, 1:])

    def test_multi_layer(self):
        attn = {
            31: torch.randn(1, 20, 5, 5),
            32: torch.randn(1, 20, 5, 5),
        }
        trimmed = trim_attention_maps(attn)
        assert trimmed[31].shape == (1, 20, 4, 4)
        assert trimmed[32].shape == (1, 20, 4, 4)


# ---------------------------------------------------------------------------
# verify_token_mapping
# ---------------------------------------------------------------------------

class TestVerifyTokenMapping:
    def test_correct(self, tokenizer):
        info = tokenize_genome_window("acgt", tokenizer)
        # Should not raise
        verify_token_mapping(info["input_ids"], "acgt", tokenizer)

    def test_wrong_length(self, tokenizer):
        ids = torch.tensor([[33, 29, 31]])  # 3 tokens but seq has 4 nt
        with pytest.raises(AssertionError, match="Token length mismatch"):
            verify_token_mapping(ids, "acgt", tokenizer)

    def test_wrong_prefix(self, tokenizer):
        ids = torch.tensor([[29, 29, 31, 32, 30]])  # first is 'a', not <+>
        with pytest.raises(AssertionError, match="First token should be"):
            verify_token_mapping(ids, "acgt", tokenizer)

    def test_wrong_nucleotide(self, tokenizer):
        # Correct prefix but swap a↔c at position 0
        ids = torch.tensor([[33, 31, 31, 32, 30]])  # pos0: c instead of a
        with pytest.raises(AssertionError, match="Token mismatch"):
            verify_token_mapping(ids, "acgt", tokenizer)


# ---------------------------------------------------------------------------
# Roundtrip & windowed position tests
# ---------------------------------------------------------------------------

class TestPositionRoundtrip:
    def test_roundtrip(self, tokenizer):
        """genome_pos i → token i+1 → trim → array index i."""
        seq = "acgtacgt"
        info = tokenize_genome_window(seq, tokenizer)
        offset = info["offset"]

        # Hidden states: index in model output for genome pos i is i + offset
        h = torch.arange(9 * 4).reshape(1, 9, 4).float()  # 8+1 tokens
        trimmed = trim_hidden_states(h, offset)

        for genome_pos in range(len(seq)):
            # After trim, array index == genome_pos
            model_idx = genome_pos + offset
            assert torch.equal(
                trimmed[0, genome_pos, :],
                h[0, model_idx, :],
            )

    def test_windowed_position(self, tokenizer):
        """Window at w_start=1000, genome pos 1005 → array index 5."""
        full_seq = "a" * 2000
        w_start = 1000
        w_len = 100
        window_seq = full_seq[w_start : w_start + w_len]

        info = tokenize_genome_window(window_seq, tokenizer)
        offset = info["offset"]

        # Simulate hidden states for this window
        h = torch.randn(1, w_len + 1, 8)
        trimmed = trim_hidden_states(h, offset)

        genome_pos = 1005
        array_idx = genome_pos - w_start  # 5
        assert array_idx == 5
        assert trimmed.shape[1] == w_len
        assert torch.equal(trimmed[0, array_idx, :], h[0, array_idx + offset, :])


# ---------------------------------------------------------------------------
# build_prodigal_mixed_sequence
# ---------------------------------------------------------------------------

class TestBuildProdigalMixedSequence:
    def test_no_cds(self):
        """No CDS → entire genome is intergenic: <+>acgt."""
        result = build_prodigal_mixed_sequence("acgt", [])
        assert result["token_string"] == "<+>acgt"
        # 3 chars for <+> + 4 nucleotides = 7 chars total
        assert len(result["token_to_genome"]) == 7
        # First 3 are sentinel (strand marker)
        assert result["token_to_genome"][:3] == [(-1, -1)] * 3
        # Next 4 are genome pos 0-3
        assert result["token_to_genome"][3] == (0, 1)
        assert result["token_to_genome"][6] == (3, 4)
        # genome_to_token: each pos maps to its char index
        g2t = result["genome_to_token"]
        assert g2t[0] == 3
        assert g2t[3] == 6

    def test_single_fwd_cds(self):
        """Single forward-strand CDS flanked by intergenic."""
        # Build a sequence with a 9-bp CDS (3 codons) at positions 3-12
        # ATG GCT TAA → M A * → "MA" (stop removed)
        seq = "aaa" + "ATGGCTCAG" + "ttt"  # 15 bp total
        cds = [{"start": 3, "end": 12, "strand": 1}]
        result = build_prodigal_mixed_sequence(seq, cds)

        ts = result["token_string"]
        # Should be: <+>aaa<+>MAS<+>ttt
        # (ATGGCTTAA → translate → MAS after removing stop)
        assert ts.startswith("<+>aaa")
        assert "<+>ttt" in ts
        # The middle part should contain the protein
        assert "MA" in ts  # First two AAs at minimum

    def test_single_rev_cds(self):
        """Reverse-strand CDS uses <-> marker."""
        # 9 bp CDS on reverse strand at positions 3-12
        # RC of ATGGCTCAG = CTGAGCCAT → translate → LSH (or similar)
        seq = "aaa" + "ATGGCTCAG" + "ttt"
        cds = [{"start": 3, "end": 12, "strand": -1}]
        result = build_prodigal_mixed_sequence(seq, cds)
        ts = result["token_string"]
        assert "<->" in ts

    def test_genome_to_token_intergenic_1to1(self):
        """Intergenic nucleotides map 1:1."""
        seq = "acgtacgt"
        result = build_prodigal_mixed_sequence(seq, [])
        g2t = result["genome_to_token"]
        # Each genome pos should map to a unique char index
        assert len(set(g2t)) == 8

    def test_genome_to_token_cds_codon_mapping(self):
        """Each codon's 3 positions map to the same AA char index."""
        # 9 bp forward CDS
        seq = "ATGGCTCAG"
        cds = [{"start": 0, "end": 9, "strand": 1}]
        result = build_prodigal_mixed_sequence(seq, cds)
        g2t = result["genome_to_token"]
        # Positions 0,1,2 (first codon) should all point to same token
        assert g2t[0] == g2t[1] == g2t[2]
        # Positions 3,4,5 (second codon) should all point to same token
        assert g2t[3] == g2t[4] == g2t[5]
        # But first and second codons should be different tokens
        assert g2t[0] != g2t[3]

    def test_token_to_genome_aa_spans_3bp(self):
        """AA chars in token_to_genome should span 3 bp."""
        seq = "ATGGCTCAG"
        cds = [{"start": 0, "end": 9, "strand": 1}]
        result = build_prodigal_mixed_sequence(seq, cds)
        t2g = result["token_to_genome"]
        # Find AA entries (non-sentinel, span > 1)
        aa_entries = [(s, e) for s, e in t2g if s != -1 and e - s == 3]
        assert len(aa_entries) >= 2  # At least 2 AAs (MAQ from ATGGCTCAG)

    def test_coverage_all_positions(self):
        """Every genome position should appear in genome_to_token."""
        seq = "aaa" + "ATGGCTCAG" + "ttt"
        cds = [{"start": 3, "end": 12, "strand": 1}]
        result = build_prodigal_mixed_sequence(seq, cds)
        g2t = result["genome_to_token"]
        for pos in range(len(seq)):
            assert g2t[pos] >= 0, f"genome pos {pos} not mapped"

    def test_multiple_cds(self):
        """Multiple CDS with intergenic gaps."""
        seq = "aaa" + "ATGGCTCAG" + "ccc" + "ATGATGCAG" + "ggg"
        cds = [
            {"start": 3, "end": 12, "strand": 1},
            {"start": 15, "end": 24, "strand": 1},
        ]
        result = build_prodigal_mixed_sequence(seq, cds)
        g2t = result["genome_to_token"]
        # All positions should be mapped
        for pos in range(len(seq)):
            assert g2t[pos] >= 0


class TestTokenizeMixedWindow:
    def test_basic(self, tokenizer):
        """Tokenize a simple mixed-token string."""
        result = tokenize_mixed_window("<+>acgt", tokenizer)
        assert result["input_ids"].shape == (1, 5)  # <+>, a, c, g, t
        assert result["n_tokens"] == 5

    def test_with_aa(self, tokenizer):
        """Tokenize string with amino acids."""
        result = tokenize_mixed_window("<+>ac<+>MA<+>gt", tokenizer)
        # <+>, a, c, <+>, M, A, <+>, g, t = 9 tokens
        assert result["n_tokens"] == 9


class TestMapFeaturesToGenome:
    def test_intergenic_only(self, tokenizer):
        """Pure intergenic: features map 1:1 to genome positions."""
        seq = "acgt"
        result = build_prodigal_mixed_sequence(seq, [])
        tok = tokenize_mixed_window(result["token_string"], tokenizer)

        n_tokens = tok["n_tokens"]
        features = np.arange(n_tokens * 2, dtype=np.float32).reshape(n_tokens, 2)

        mapped, counts = map_features_to_genome(
            features, result["token_to_genome"], len(seq),
            tokenizer, tok["input_ids"],
        )
        assert mapped.shape == (4, 2)
        # Each genome pos should have count 1
        assert np.all(counts == 1.0)

    def test_cds_replication(self, tokenizer):
        """AA token features are replicated to all 3 codon positions."""
        seq = "ATGGCTCAG"  # 9 bp, 3 codons
        cds = [{"start": 0, "end": 9, "strand": 1}]
        result = build_prodigal_mixed_sequence(seq, cds)
        tok = tokenize_mixed_window(result["token_string"], tokenizer)

        n_tokens = tok["n_tokens"]
        features = np.ones((n_tokens, 1), dtype=np.float32)

        mapped, counts = map_features_to_genome(
            features, result["token_to_genome"], len(seq),
            tokenizer, tok["input_ids"],
        )
        # All 9 positions should be covered
        assert mapped.shape == (9, 1)
        assert np.all(counts > 0)


# ---------------------------------------------------------------------------
# _parse_segments
# ---------------------------------------------------------------------------

class TestParseSegments:
    def test_single_intergenic(self):
        segs = _parse_segments("<+>aaa")
        assert segs == [("<+>", "aaa")]

    def test_mixed_segments(self):
        segs = _parse_segments("<+>aaa<+>MAQ<->LSH<+>ttt")
        assert segs == [
            ("<+>", "aaa"),
            ("<+>", "MAQ"),
            ("<->", "LSH"),
            ("<+>", "ttt"),
        ]

    def test_empty_string(self):
        assert _parse_segments("") == []

    def test_marker_only(self):
        segs = _parse_segments("<+>")
        assert segs == [("<+>", "")]

    def test_adjacent_markers(self):
        """Adjacent CDS with no intergenic gap."""
        segs = _parse_segments("<+>MAQ<->LSH")
        assert segs == [("<+>", "MAQ"), ("<->", "LSH")]

    def test_is_dna_segment(self):
        assert _is_dna_segment("acgtn") is True
        assert _is_dna_segment("MAQ") is False
        assert _is_dna_segment("") is True
        assert _is_dna_segment("acgtN") is False  # uppercase N is not DNA


# ---------------------------------------------------------------------------
# rc_mixed_token_string
# ---------------------------------------------------------------------------

class TestRcMixedTokenString:
    def test_basic_example(self):
        """The canonical example from the plan."""
        result = rc_mixed_token_string("<+>aaa<+>MAQ<->LSH<+>ttt")
        assert result == "<+>aaa<+>LSH<->MAQ<+>ttt"

    def test_pure_dna(self):
        """Pure DNA is reverse-complemented."""
        # aaaccc → complement = tttggg → reversed = gggttt
        assert rc_mixed_token_string("<+>aaaccc") == "<+>gggttt"

    def test_single_forward_cds(self):
        """Forward CDS becomes reverse CDS."""
        assert rc_mixed_token_string("<+>MAQ") == "<->MAQ"

    def test_single_reverse_cds(self):
        """Reverse CDS becomes forward CDS."""
        assert rc_mixed_token_string("<->LSH") == "<+>LSH"

    def test_double_rc_identity(self):
        """RC of RC must recover the original string (critical invariant)."""
        cases = [
            "<+>aaa<+>MAQ<->LSH<+>ttt",
            "<+>aaaccc",
            "<+>MAQ",
            "<->LSH",
            "<+>acgt<+>MAQVV<->LSHK<+>nnn<->GDEF<+>tgca",
            "<+>aaa<+>MAQ<->LSH",
        ]
        for s in cases:
            assert rc_mixed_token_string(rc_mixed_token_string(s)) == s, (
                f"Double RC failed for: {s}"
            )

    def test_intergenic_markers_always_plus(self):
        """DNA segments always use <+> regardless of neighbours."""
        result = rc_mixed_token_string("<->LSH<+>acgt<+>MAQ")
        # reversed: MAQ, acgt, LSH
        # MAQ was <+> → becomes <->
        # acgt is DNA → <+> + rc(acgt) = <+>acgt (palindromic!)
        # LSH was <-> → becomes <+>
        assert result == "<->MAQ<+>acgt<+>LSH"  # acgt is its own RC

    def test_adjacent_cds_no_gap(self):
        """Adjacent CDS without intergenic gap."""
        result = rc_mixed_token_string("<+>MAQ<->LSH")
        assert result == "<+>LSH<->MAQ"

    def test_ambiguous_bases(self):
        """Ambiguous base 'n' complements to 'n'."""
        assert rc_mixed_token_string("<+>nnn") == "<+>nnn"
        # anc → complement = tng → reversed = gnt
        assert rc_mixed_token_string("<+>anc") == "<+>gnt"

    def test_empty_string(self):
        """Empty string returns empty."""
        assert rc_mixed_token_string("") == ""

    def test_long_mixed(self):
        """Longer mixed string with multiple CDS and intergenic regions."""
        s = "<+>acgt<+>MAQVV<->LSHK<+>nnn<->GDEF<+>tgca"
        rc = rc_mixed_token_string(s)
        # Reversed segments: tgca, GDEF, nnn, LSHK, MAQVV, acgt
        # rc(tgca) = tgca (palindromic), GDEF(<->→<+>), rc(nnn)=nnn,
        # LSHK(<->→<+>), MAQVV(<+>→<->), rc(acgt)=acgt (palindromic)
        assert rc == "<+>tgca<+>GDEF<+>nnn<+>LSHK<->MAQVV<+>acgt"


# ---------------------------------------------------------------------------
# RC integration tests (extract_and_tokenize_gb + reverse_complement)
# ---------------------------------------------------------------------------

def _create_test_genbank(locus_name: str, sequence: str, features: list[dict]) -> str:
    """Create a minimal GenBank file string for testing."""
    seq_len = len(sequence)
    formatted_seq_lines = []
    pos = 0
    while pos < seq_len:
        line_start = pos + 1
        line_seq = sequence[pos:pos + 60]
        groups = [line_seq[i:i + 10] for i in range(0, len(line_seq), 10)]
        formatted_seq_lines.append(f"{line_start:>9} {' '.join(groups)}")
        pos += 60
    formatted_seq = "\n".join(formatted_seq_lines)

    feature_lines = []
    for feat in features:
        strand = feat.get("strand", 1)
        start = feat["start"]
        end = feat["end"]
        gene_name = feat.get("gene_name", "unknown")
        if strand == 1:
            location = f"{start + 1}..{end}"
        else:
            location = f"complement({start + 1}..{end})"
        feature_lines.append(f"     CDS             {location}")
        feature_lines.append(f'                     /gene="{gene_name}"')

    lines = [
        f"LOCUS       {locus_name:<16} {seq_len} bp    DNA     linear   UNK 01-JAN-2024",
        "DEFINITION  Test sequence.",
        f"ACCESSION   {locus_name}",
        f"VERSION     {locus_name}.1",
        "KEYWORDS    .",
        "SOURCE      synthetic construct",
        "  ORGANISM  synthetic construct",
        "            other sequences; artificial sequences.",
        "FEATURES             Location/Qualifiers",
        f"     source          1..{seq_len}",
        '                     /organism="synthetic construct"',
        '                     /mol_type="genomic DNA"',
    ]
    for line in feature_lines:
        lines.append(line)
    lines.append("ORIGIN")
    lines.extend(formatted_seq.split("\n"))
    lines.append("//")
    return "\n".join(lines) + "\n"


def _write_temp_genbank(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".gb")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _extract(sequence, features, rc=False):
    """Helper: build GenBank, extract fwd and optionally RC record."""
    from minerva.data import extract_and_tokenize_gb

    gb = _create_test_genbank("TEST", sequence, features)
    path = _write_temp_genbank(gb)
    try:
        records = extract_and_tokenize_gb(
            path, overlap_mode="expand", reverse_complement=rc,
        )
        return records[0]
    finally:
        os.unlink(path)


class TestRcIntegration:
    """Integration tests: extract_and_tokenize_gb(reverse_complement=True)."""

    # -- A. Token string correctness -----------------------------------

    def test_rc_sequence_matches_rc_function(self):
        """RC via extract matches rc_mixed_token_string on the forward seq."""
        sequence = "ATGAAAGCGTTTCCCGGGAAATAG" + "NNNNNN" + "ATGCCCGGGAAATTTCCCGGGTAG" + "NNNNNN"
        feats = [
            {"start": 0, "end": 24, "strand": 1, "gene_name": "g1"},
            {"start": 30, "end": 54, "strand": 1, "gene_name": "g2"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)
        assert rc_rec["sequence"] == rc_mixed_token_string(fwd["sequence"])

    def test_rc_partial_overlap_same_strand(self):
        """Overlapping genes (same strand): RC preserves AA and flips orient."""
        sequence = "ATG" * 40  # 120bp
        feats = [
            {"start": 0, "end": 60, "strand": 1, "gene_name": "long_gene"},
            {"start": 30, "end": 90, "strand": 1, "gene_name": "short_gene"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)

        # Sequence matches standalone rc function
        assert rc_rec["sequence"] == rc_mixed_token_string(fwd["sequence"])
        # AA sequences preserved (compare by gene name)
        fwd_seqs = {f["gene_name"]: f["seq"] for f in fwd["features"]}
        rc_seqs = {f["gene_name"]: f["seq"] for f in rc_rec["features"]}
        assert fwd_seqs == rc_seqs
        # Both orientations flipped
        for f in rc_rec["features"]:
            assert f["orientation"] is False

    def test_rc_opposite_strand_overlap(self):
        """Overlapping genes on opposite strands: markers swap correctly."""
        sequence = "ATG" * 30  # 90bp
        feats = [
            {"start": 0, "end": 60, "strand": 1, "gene_name": "plus_gene"},
            {"start": 30, "end": 90, "strand": -1, "gene_name": "minus_gene"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)

        assert rc_rec["sequence"] == rc_mixed_token_string(fwd["sequence"])
        # plus_gene was orient=True → becomes False
        # minus_gene was orient=False → becomes True
        rc_by_name = {f["gene_name"]: f for f in rc_rec["features"]}
        assert rc_by_name["plus_gene"]["orientation"] is False
        assert rc_by_name["minus_gene"]["orientation"] is True

    def test_rc_complete_containment(self):
        """Outer gene fully contains inner gene: both present after RC."""
        sequence = "ATG" * 30  # 90bp
        feats = [
            {"start": 0, "end": 90, "strand": 1, "gene_name": "outer"},
            {"start": 24, "end": 54, "strand": 1, "gene_name": "inner"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)

        assert rc_rec["sequence"] == rc_mixed_token_string(fwd["sequence"])
        assert len(rc_rec["features"]) == 2
        # Inner gene appears first in RC (was last in forward)
        assert rc_rec["features"][0]["gene_name"] == "inner"
        assert rc_rec["features"][1]["gene_name"] == "outer"

    def test_rc_chain_overlapping(self):
        """Chain of 3 overlapping genes: all present, fully reversed."""
        sequence = "ATG" * 40  # 120bp
        feats = [
            {"start": 0, "end": 30, "strand": 1, "gene_name": "g1"},
            {"start": 15, "end": 45, "strand": 1, "gene_name": "g2"},
            {"start": 30, "end": 60, "strand": 1, "gene_name": "g3"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)

        assert rc_rec["sequence"] == rc_mixed_token_string(fwd["sequence"])
        assert len(rc_rec["features"]) == 3
        assert [f["gene_name"] for f in rc_rec["features"]] == ["g3", "g2", "g1"]
        for f in rc_rec["features"]:
            assert f["orientation"] is False

    def test_rc_genes_at_boundaries(self):
        """Genes at position 0 and sequence end: no off-by-one errors."""
        # Gene at start (0-30) and gene at end (90-120)
        sequence = "ATG" * 40  # 120bp
        feats = [
            {"start": 0, "end": 30, "strand": 1, "gene_name": "start_gene"},
            {"start": 90, "end": 120, "strand": 1, "gene_name": "end_gene"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)

        assert rc_rec["sequence"] == rc_mixed_token_string(fwd["sequence"])
        # end_gene appears first in RC
        assert rc_rec["features"][0]["gene_name"] == "end_gene"
        assert rc_rec["features"][1]["gene_name"] == "start_gene"

    # -- B. Invariant tests --------------------------------------------

    def test_rc_double_rc_identity(self):
        """RC(RC(sequence)) == original for overlapping genes."""
        sequence = "ATG" * 40  # 120bp
        feats = [
            {"start": 0, "end": 60, "strand": 1, "gene_name": "long_gene"},
            {"start": 30, "end": 90, "strand": -1, "gene_name": "short_gene"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)
        assert rc_mixed_token_string(rc_rec["sequence"]) == fwd["sequence"]

    def test_rc_tokens_match_sequence(self):
        """tokens joined == sequence for all overlap scenarios."""
        cases = [
            # Partial overlap
            ("ATG" * 40, [
                {"start": 0, "end": 60, "strand": 1, "gene_name": "a"},
                {"start": 30, "end": 90, "strand": 1, "gene_name": "b"},
            ]),
            # Opposite strands
            ("ATG" * 30, [
                {"start": 0, "end": 60, "strand": 1, "gene_name": "a"},
                {"start": 30, "end": 90, "strand": -1, "gene_name": "b"},
            ]),
            # Containment
            ("ATG" * 30, [
                {"start": 0, "end": 90, "strand": 1, "gene_name": "a"},
                {"start": 24, "end": 54, "strand": 1, "gene_name": "b"},
            ]),
        ]
        for seq, feats in cases:
            rc_rec = _extract(seq, feats, rc=True)
            joined = "".join(rc_rec["tokens"])
            assert joined == rc_rec["sequence"], (
                f"Token join mismatch for feats={[f['gene_name'] for f in feats]}"
            )

    def test_rc_feature_count_preserved(self):
        """Same number of features in forward and RC records."""
        sequence = "ATG" * 40
        feats = [
            {"start": 0, "end": 30, "strand": 1, "gene_name": "g1"},
            {"start": 15, "end": 45, "strand": 1, "gene_name": "g2"},
            {"start": 30, "end": 60, "strand": 1, "gene_name": "g3"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)
        assert len(fwd["features"]) == len(rc_rec["features"])

    def test_rc_aa_sequences_preserved(self):
        """Each gene's AA seq is identical between forward and RC."""
        sequence = "ATG" * 30  # 90bp
        feats = [
            {"start": 0, "end": 60, "strand": 1, "gene_name": "plus"},
            {"start": 30, "end": 90, "strand": -1, "gene_name": "minus"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)
        fwd_aa = {f["gene_name"]: f["seq"] for f in fwd["features"]}
        rc_aa = {f["gene_name"]: f["seq"] for f in rc_rec["features"]}
        assert fwd_aa == rc_aa

    # -- C. Feature metadata tests -------------------------------------

    def test_rc_features_reversed_order(self):
        """features[0] in RC == features[-1] in forward (by gene_name)."""
        sequence = "ATG" * 40
        feats = [
            {"start": 0, "end": 30, "strand": 1, "gene_name": "first"},
            {"start": 60, "end": 90, "strand": 1, "gene_name": "second"},
            {"start": 90, "end": 120, "strand": 1, "gene_name": "third"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)
        fwd_names = [f["gene_name"] for f in fwd["features"]]
        rc_names = [f["gene_name"] for f in rc_rec["features"]]
        assert rc_names == list(reversed(fwd_names))

    def test_rc_orientations_all_flipped(self):
        """Every feature's orientation is flipped after RC."""
        sequence = "ATG" * 30
        feats = [
            {"start": 0, "end": 60, "strand": 1, "gene_name": "plus"},
            {"start": 30, "end": 90, "strand": -1, "gene_name": "minus"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)
        fwd_orient = {f["gene_name"]: f["orientation"] for f in fwd["features"]}
        rc_orient = {f["gene_name"]: f["orientation"] for f in rc_rec["features"]}
        for name in fwd_orient:
            assert rc_orient[name] is not fwd_orient[name], (
                f"Orientation not flipped for {name}"
            )

    def test_rc_intergenic_regions_reversed(self):
        """Intergenic regions list is reversed with same count and dicts."""
        sequence = "ATGAAAGCGTTTCCCGGGAAATAG" + "NNNNNN" + "ATGCCCGGGAAATTTCCCGGGTAG" + "NNNNNN"
        feats = [
            {"start": 0, "end": 24, "strand": 1, "gene_name": "g1"},
            {"start": 30, "end": 54, "strand": 1, "gene_name": "g2"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)
        assert len(fwd["intergenic_regions"]) == len(rc_rec["intergenic_regions"])
        assert rc_rec["intergenic_regions"] == list(reversed(fwd["intergenic_regions"]))

    # -- D. Edge cases -------------------------------------------------

    def test_rc_no_cds(self):
        """Pure intergenic: RC produces complement of DNA."""
        sequence = "AAACCC"  # 6bp
        fwd = _extract(sequence, [], rc=False)
        rc_rec = _extract(sequence, [], rc=True)
        assert fwd["sequence"] == "<+>aaaccc"
        assert rc_rec["sequence"] == "<+>gggttt"
        assert rc_mixed_token_string(rc_rec["sequence"]) == fwd["sequence"]

    def test_rc_single_gene(self):
        """Single CDS: marker flips, AA preserved."""
        sequence = "ATG" * 10  # 30bp
        feats = [{"start": 0, "end": 30, "strand": 1, "gene_name": "solo"}]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)
        assert len(rc_rec["features"]) == 1
        assert rc_rec["features"][0]["orientation"] is False
        assert rc_rec["features"][0]["seq"] == fwd["features"][0]["seq"]
        # Forward has <+>, RC has <->
        assert fwd["sequence"].startswith("<+>")
        assert rc_rec["sequence"].startswith("<->")

    def test_rc_all_reverse_strand(self):
        """All genes on minus strand: all become plus after RC."""
        sequence = "ATG" * 40  # 120bp
        feats = [
            {"start": 0, "end": 30, "strand": -1, "gene_name": "r1"},
            {"start": 60, "end": 90, "strand": -1, "gene_name": "r2"},
        ]
        fwd = _extract(sequence, feats, rc=False)
        rc_rec = _extract(sequence, feats, rc=True)
        # All were minus (orient=False), should become plus (orient=True)
        for f in rc_rec["features"]:
            assert f["orientation"] is True


# ---------------------------------------------------------------------------
# build_fwd_rc_permutation
# ---------------------------------------------------------------------------

class TestBuildFwdRcPermutation:
    def test_pure_dna_permutation(self):
        """<+>acgt → 5 tokens, DNA content reverses."""
        perm = build_fwd_rc_permutation("<+>acgt")
        # Single segment: marker stays at 0, content [1,2,3,4] reverses
        assert perm[0] == 0  # marker
        assert perm[1] == 4
        assert perm[2] == 3
        assert perm[3] == 2
        assert perm[4] == 1

    def test_mixed_permutation_basic(self):
        """The canonical example from the plan."""
        perm = build_fwd_rc_permutation("<+>aaa<+>MAQ<->LSH<+>ttt")
        expected = [12, 15, 14, 13, 8, 9, 10, 11, 4, 5, 6, 7, 0, 3, 2, 1]
        assert perm == expected

    def test_permutation_length(self):
        """len(perm) == number of tokens in the string."""
        ts = "<+>aaa<+>MAQ<->LSH<+>ttt"
        perm = build_fwd_rc_permutation(ts)
        # 4 markers + 3+3+3+3 content = 16 tokens
        assert len(perm) == 16

    def test_permutation_is_valid(self):
        """Permutation contains each index exactly once."""
        ts = "<+>acgt<+>MAQVV<->LSHK<+>nnn<->GDEF<+>tgca"
        perm = build_fwd_rc_permutation(ts)
        assert sorted(perm) == list(range(len(perm)))

    def test_permutation_roundtrip(self):
        """P_rc[P_fwd[i]] == i: composing fwd→rc and rc→fwd gives identity."""
        cases = [
            "<+>acgt",
            "<+>aaa<+>MAQ<->LSH<+>ttt",
            "<+>acgt<+>MAQVV<->LSHK<+>nnn<->GDEF<+>tgca",
            "<+>MAQ<->LSH",
        ]
        for ts in cases:
            rc_ts = rc_mixed_token_string(ts)
            p_fwd = build_fwd_rc_permutation(ts)
            p_rc = build_fwd_rc_permutation(rc_ts)
            roundtrip = [p_rc[p_fwd[i]] for i in range(len(p_fwd))]
            assert roundtrip == list(range(len(p_fwd))), (
                f"Roundtrip failed for: {ts}"
            )

    def test_adjacent_cds_no_gap(self):
        """<+>MAQ<->LSH → correct permutation."""
        perm = build_fwd_rc_permutation("<+>MAQ<->LSH")
        # Fwd: [<+>, M, A, Q, <->, L, S, H] = indices 0..7
        # RC:  [<+>, L, S, H, <->, M, A, Q] = indices 0..7
        # Segment 0 (<+>MAQ) ↔ RC segment 1 (<->MAQ): marker 0↔4, CDS same order
        # Segment 1 (<->LSH) ↔ RC segment 0 (<+>LSH): marker 4↔0, CDS same order
        expected = [4, 5, 6, 7, 0, 1, 2, 3]
        assert perm == expected

    def test_single_segment(self):
        """Single DNA segment: <+>acgt."""
        perm = build_fwd_rc_permutation("<+>acgt")
        assert len(perm) == 5
        # Marker maps to marker (only 1 segment)
        assert perm[0] == 0
        # DNA reverses
        assert perm == [0, 4, 3, 2, 1]


# ---------------------------------------------------------------------------
# remap_rc_attention
# ---------------------------------------------------------------------------

class TestRemapRcAttention:
    def test_remap_identity(self):
        """Identity permutation doesn't change attention."""
        attn = torch.randn(1, 2, 4, 4)
        perm = [0, 1, 2, 3]
        result = remap_rc_attention(attn, perm)
        assert torch.equal(result, attn)

    def test_remap_reversal(self):
        """Full reversal perm is equivalent to [::-1,::-1] flip."""
        attn = torch.randn(1, 2, 4, 4)
        perm = [3, 2, 1, 0]
        result = remap_rc_attention(attn, perm)
        expected = attn[:, :, torch.arange(3, -1, -1), :][:, :, :, torch.arange(3, -1, -1)]
        assert torch.allclose(result, expected)

    def test_remap_known_values(self):
        """Construct small attention, verify specific values after remap."""
        # 3x3 attention, perm = [2, 0, 1] (rotate)
        attn = torch.tensor([[1., 2., 3.],
                              [4., 5., 6.],
                              [7., 8., 9.]])
        perm = [2, 0, 1]
        result = remap_rc_attention(attn, perm)
        # result[i, j] = attn[perm[i], perm[j]]
        # result[0, 0] = attn[2, 2] = 9
        # result[0, 1] = attn[2, 0] = 7
        # result[1, 0] = attn[0, 2] = 3
        assert result[0, 0].item() == 9.0
        assert result[0, 1].item() == 7.0
        assert result[1, 0].item() == 3.0
        assert result[1, 1].item() == 1.0
        assert result[2, 2].item() == 5.0


# ---------------------------------------------------------------------------
# Combined fwd+rc attention
# ---------------------------------------------------------------------------

class TestCombineFwdRcAttention:
    def test_upper_tri_from_fwd(self):
        """Upper triangle of combined matches forward attention."""
        L = 6
        fwd = torch.randn(1, 2, L, L)
        rc_aligned = torch.randn(1, 2, L, L)
        mask_upper = torch.triu(torch.ones(L, L), diagonal=1)
        diag = torch.eye(L)
        mask_lower = torch.tril(torch.ones(L, L), diagonal=-1)
        combined = fwd * (mask_upper + diag) + rc_aligned * mask_lower

        # Check upper triangle (strictly above diagonal)
        for i in range(L):
            for j in range(i + 1, L):
                assert torch.equal(combined[:, :, i, j], fwd[:, :, i, j])

    def test_lower_tri_from_rc(self):
        """Lower triangle of combined matches rc_aligned attention."""
        L = 6
        fwd = torch.randn(1, 2, L, L)
        rc_aligned = torch.randn(1, 2, L, L)
        mask_upper = torch.triu(torch.ones(L, L), diagonal=1)
        diag = torch.eye(L)
        mask_lower = torch.tril(torch.ones(L, L), diagonal=-1)
        combined = fwd * (mask_upper + diag) + rc_aligned * mask_lower

        # Check lower triangle (strictly below diagonal)
        for i in range(L):
            for j in range(i):
                assert torch.equal(combined[:, :, i, j], rc_aligned[:, :, i, j])

    def test_diagonal_from_fwd(self):
        """Diagonal of combined comes from forward attention."""
        L = 6
        fwd = torch.randn(1, 2, L, L)
        rc_aligned = torch.randn(1, 2, L, L)
        mask_upper = torch.triu(torch.ones(L, L), diagonal=1)
        diag = torch.eye(L)
        mask_lower = torch.tril(torch.ones(L, L), diagonal=-1)
        combined = fwd * (mask_upper + diag) + rc_aligned * mask_lower

        for i in range(L):
            assert torch.equal(combined[:, :, i, i], fwd[:, :, i, i])


# ---------------------------------------------------------------------------
# Token counting, context-length capping, and windowing
# ---------------------------------------------------------------------------

_SENTINEL_PLUS = ""
_SENTINEL_MINUS = ""


class TestMixedTokenLength:
    def test_counts_markers_as_one(self):
        # 2 markers + 3 aa + 4 nt = 9 tokens
        assert mixed_token_length("<+>MKL<->acgt") == 9

    def test_empty(self):
        assert mixed_token_length("") == 0

    def test_no_markers(self):
        assert mixed_token_length("acgtACGT") == 8


class TestBuildProdigalCapping:
    def _cds(self):
        # A single + strand gene followed by intergenic DNA
        seq = "ATG" + "AAACGT" * 20 + "TAA" + "acgt" * 10
        cds = [{"start": 0, "end": 3 + 6 * 20 + 3, "strand": 1}]
        return seq, cds

    def test_uncapped_matches_len(self):
        seq, cds = self._cds()
        out = build_prodigal_mixed_sequence(seq, cds)
        assert mixed_token_length(out["token_string"]) == len(out["token_to_genome"]) - 2 * (
            out["token_string"].count("<+>") + out["token_string"].count("<->")
        )

    @pytest.mark.parametrize("cap", [1, 2, 5, 20, 50])
    def test_cap_never_exceeded(self, cap):
        seq, cds = self._cds()
        out = build_prodigal_mixed_sequence(seq, cds, max_tokens=cap)
        ts = out["token_string"]
        assert mixed_token_length(ts) <= cap
        # never ends on a dangling strand marker
        assert not ts.endswith("<+>") and not ts.endswith("<->")
        # markers always complete
        assert ts.count("<") == ts.count(">")

    def test_cap_prefix_consistent(self):
        seq, cds = self._cds()
        full = build_prodigal_mixed_sequence(seq, cds)
        capped = build_prodigal_mixed_sequence(seq, cds, max_tokens=15)
        # coord maps of the capped output are a prefix of the uncapped ones
        n = len(capped["token_to_genome"])
        assert capped["token_to_genome"] == full["token_to_genome"][:n]

    def test_cap_larger_than_content_is_noop(self):
        seq, cds = self._cds()
        full = build_prodigal_mixed_sequence(seq, cds)
        big = build_prodigal_mixed_sequence(seq, cds, max_tokens=10_000)
        assert big["token_string"] == full["token_string"]


class TestChunkSequenceWithStride:
    def test_short_sequence_single_chunk(self):
        s = "<+>MKLV"
        assert chunk_sequence_with_stride(s, 100, 50) == [s]

    def test_windows_within_chunk_size(self):
        s = "<+>" + "M" * 100 + "<->" + "acgt" * 50 + "<+>" + "K" * 80
        for c in chunk_sequence_with_stride(s, 100, 50):
            assert mixed_token_length(c) <= 100

    def test_non_overlap_reconstructs(self):
        # stride == chunk_size tiles without overlap and must be lossless
        s = "<+>" + "M" * 100 + "<->" + "acgt" * 50 + "<+>" + "K" * 80
        assert "".join(chunk_sequence_with_stride(s, 50, 50)) == s

    def test_marker_never_split(self):
        s = "M" * 49 + "<+>" + "K" * 60
        for c in chunk_sequence_with_stride(s, 50, 25):
            assert c.count("<") == c.count(">")
            assert _SENTINEL_PLUS not in c and _SENTINEL_MINUS not in c

    def test_overlap_covers_all_tokens(self):
        s = "<+>" + "M" * 200
        chunks = chunk_sequence_with_stride(s, 50, 25)
        # every window is a real substring and the last reaches the end
        assert chunks[-1].endswith("M")

    @pytest.mark.parametrize("bad", [0, -1])
    def test_invalid_args(self, bad):
        with pytest.raises((ValueError, TypeError)):
            chunk_sequence_with_stride("<+>MK", bad, 1)
        with pytest.raises((ValueError, TypeError)):
            chunk_sequence_with_stride("<+>MK", 10, bad)
