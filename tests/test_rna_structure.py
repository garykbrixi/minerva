"""Tests for minerva.rna_structure: pair calling, notation and exports."""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np
import pytest

from minerva.rna_structure import (
    RnaStructure,
    call_base_pairs,
    match_candidates,
    pair_candidates,
    naview_layout,
    call_structure,
    call_structures,
    nucleotide_regions,
    dot_bracket_to_pairs,
    pairs_to_dot_bracket,
    plot_secondary_structure,
)

# The README quick-start sequence: a real 77 nt tRNA. Used where only length
# and token handling matter -- these tests make no claim about its true fold.
README_TRNA = ("cgcggggtggagcagcctggtagctcgtcgggctcataacccgaagatcgtcggttcaaat"
               "ccggcccccgcaacca")

# A real tRNA and its real structure: Rfam RF00005 record X61068.1/760-831,
# taken from the covarval Rfam evaluation set. The dot-bracket below is what
# this module recovers from that record's `inferred_contacts` ground-truth
# matrix, and every one of its 21 pairs is Watson-Crick or G-U wobble -- so it
# doubles as a check that pair calling and notation agree with curated Rfam.
CLOVERLEAF_SEQ = ("GGGGCUUUAGCUCAGCUGGGAGAGCGCCUGCCUUGCACGCAGGAGG"
                  "UCAGCGGUUCGAUCCGCUAAGCUCCA")
CLOVERLEAF = "(((((((..((((........)))).(((((.......))))).....(((((......))))))))))))."


def _map_from_pairs(pairs, n, score=0.95, noise=0.01, seed=0):
    """A synthetic contact map that puts `score` on each pair, noise elsewhere."""
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.0, noise, size=(n, n))
    for i, j, *_ in pairs:
        m[i, j] = m[j, i] = score
    return m


# ---------------------------------------------------------------------------
# Notation round-trips
# ---------------------------------------------------------------------------
def test_dot_bracket_round_trip_nested():
    pairs = dot_bracket_to_pairs(CLOVERLEAF)
    assert len(pairs) == CLOVERLEAF.count("(")
    assert pairs_to_dot_bracket(pairs, len(CLOVERLEAF)).structure == CLOVERLEAF


def test_dot_bracket_round_trip_pseudoknot():
    ss = "((((....[[[[....))))....]]]]"
    pairs = dot_bracket_to_pairs(ss)
    out = pairs_to_dot_bracket(pairs, len(ss), max_pages=3)
    assert out.structure == ss
    assert len(out.nested_pairs) == 4
    assert len(out.pseudoknot_pairs) == 4
    assert out.dropped_pairs == []


def test_dot_bracket_rejects_unbalanced():
    with pytest.raises(ValueError):
        dot_bracket_to_pairs("((..)")
    with pytest.raises(ValueError):
        dot_bracket_to_pairs("(..))")


# ---------------------------------------------------------------------------
# Pair calling
# ---------------------------------------------------------------------------
def test_call_base_pairs_recovers_trna():
    truth = dot_bracket_to_pairs(CLOVERLEAF)
    called = call_base_pairs(_map_from_pairs(truth, len(CLOVERLEAF)), threshold=0.6)
    assert {(i, j) for i, j, _ in called} == {(i, j) for i, j, _ in truth}


def test_call_base_pairs_enforces_one_partner_and_separation():
    n = 30
    m = np.zeros((n, n))
    m[5, 20] = m[20, 5] = 0.9          # strongest, should win
    m[5, 25] = m[25, 5] = 0.8          # base 5 already used
    m[10, 11] = m[11, 10] = 0.95       # inside min_separation
    called = call_base_pairs(m, threshold=0.6, min_separation=3)
    assert [(i, j) for i, j, _ in called] == [(5, 20)]


# ---------------------------------------------------------------------------
# Pseudoknot pages: nothing is lost, and stems stay whole
# ---------------------------------------------------------------------------
def test_max_pages_one_drops_but_reports_crossing_pairs():
    ss = "((((....[[[[....))))....]]]]"
    pairs = dot_bracket_to_pairs(ss)
    out = pairs_to_dot_bracket(pairs, len(ss), max_pages=1)
    assert set("[]{}").isdisjoint(out.structure)
    assert len(out.dropped_pairs) == 4
    # Every input pair is still accounted for somewhere.
    assert (len(out.nested_pairs) + len(out.pseudoknot_pairs)
            + len(out.dropped_pairs)) == len(pairs)


def test_stems_are_not_split_across_pages():
    ss = "((((....[[[[....))))....]]]]"
    out = pairs_to_dot_bracket(dot_bracket_to_pairs(ss), len(ss), max_pages=3)
    # The longer/first stem keeps '()', the crossing one moves wholesale.
    assert out.structure.count("(") == 4
    assert out.structure.count("[") == 4


# ---------------------------------------------------------------------------
# The structure object and its exports
# ---------------------------------------------------------------------------
def _trna_structure(**kw):
    truth = dot_bracket_to_pairs(CLOVERLEAF)
    return call_structure(_map_from_pairs(truth, len(CLOVERLEAF_SEQ)), CLOVERLEAF_SEQ,
                          name="tRNA", **kw)


def test_call_structure_from_sequence():
    s = _trna_structure()
    assert s.dot_bracket == CLOVERLEAF
    assert len(s.sequence) == len(s.dot_bracket) == 72
    assert s.canonical_fraction() == 1.0   # real tRNA: every pair is WC or G-U


def test_sequence_alphabet_follows_the_rna_flag():
    n = len(README_TRNA)
    rna = call_structure(np.zeros((n, n)), README_TRNA)
    dna = call_structure(np.zeros((n, n)), README_TRNA, rna=False)
    assert rna.sequence == README_TRNA.upper().replace("T", "U")
    assert "T" in dna.sequence and "U" not in dna.sequence


def test_call_structure_crops_to_nucleotide_tokens():
    """The README path: a map over tokens including <+> and amino acids."""
    tokens = ["<+>"] + list(CLOVERLEAF_SEQ.lower()) + ["M", "K", "V"]
    truth = dot_bracket_to_pairs(CLOVERLEAF)
    n = len(tokens)
    shifted = [(i + 1, j + 1, 1.0) for i, j, _ in truth]   # <+> occupies index 0
    s = call_structure(_map_from_pairs(shifted, n), tokens=tokens)
    assert s.sequence == CLOVERLEAF_SEQ
    assert s.dot_bracket == CLOVERLEAF
    assert s.offset == 1                                   # first nucleotide token


def test_call_structure_length_mismatch_is_explicit():
    with pytest.raises(ValueError, match="crop them to the same region"):
        call_structure(np.zeros((10, 10)), "ACGU")


def test_call_structure_handles_the_batch_axis():
    """interactions[...] is [batch, L, L]: one map passes, a real batch does not."""
    assert len(call_structure(np.zeros((1, 10, 10)), "A" * 10).sequence) == 10
    with pytest.raises(ValueError, match="batch"):
        call_structure(np.zeros((4, 10, 10)), "A" * 10)


def test_partners_include_pairs_dropped_from_notation():
    ss = "((((....[[[[....))))....]]]]"
    seq = "GCGC" + "AAAA" + "GCGC" + "AAAA" + "GCGC" + "AAAA" + "GCGC"
    pairs = dot_bracket_to_pairs(ss)
    s = call_structure(_map_from_pairs(pairs, len(ss)), seq, max_pages=1)
    assert len(s.dropped_pairs) == 4
    # partners is built from every called pair, notated or not.
    assert int((s.partners >= 0).sum()) == 2 * len(pairs)


def test_vienna_export():
    s = _trna_structure()
    text = s.to_vienna()
    assert text.splitlines() == [">tRNA", s.sequence, s.dot_bracket]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.fa")
        s.to_vienna(path)
        assert open(path).read() == text


def test_ct_export_is_lossless_for_pseudoknots():
    ss = "((((....[[[[....))))....]]]]"
    seq = "G" * len(ss)
    s = call_structure(_map_from_pairs(dot_bracket_to_pairs(ss), len(ss)), seq,
                       max_pages=1)          # forces 4 pairs out of the notation
    lines = s.to_ct().splitlines()
    assert lines[0].split("\t")[0] == str(len(ss))
    partners = [int(l.split("\t")[4]) for l in lines[1:]]
    assert sum(1 for p in partners if p) == 2 * len(s.pairs)   # nothing lost
    # `offset` shifts only the natural-numbering column.
    assert _trna_structure(offset=1000).to_ct().splitlines()[1].split("\t")[5] == "1001"


# ---------------------------------------------------------------------------
# Renderers (smoke tests -- they must not raise, and must produce a figure)
# ---------------------------------------------------------------------------
def test_plot_secondary_structure():
    import matplotlib
    matplotlib.use("Agg")
    fig = plot_secondary_structure(_trna_structure())
    assert fig.get_axes()
    with tempfile.TemporaryDirectory() as d:
        fig.savefig(os.path.join(d, "s.pdf"))


def test_plot_handles_unpaired_sequence():
    import matplotlib
    matplotlib.use("Agg")
    s = call_structure(np.zeros((20, 20)), "A" * 20)
    assert plot_secondary_structure(s).get_axes()


# ---------------------------------------------------------------------------
# Re-thresholding -- the operation the viewer's slider performs
# ---------------------------------------------------------------------------
def test_candidates_are_threshold_independent():
    """Re-thresholding stored candidates == re-running the whole call."""
    truth = dot_bracket_to_pairs(CLOVERLEAF)
    m = _map_from_pairs(truth, len(CLOVERLEAF_SEQ), score=0.9, noise=0.7, seed=3)
    base = call_structure(m, CLOVERLEAF_SEQ, threshold=0.1)
    for thr in (0.2, 0.4, 0.6, 0.8, 0.95):
        direct = call_structure(m, CLOVERLEAF_SEQ, threshold=thr)
        assert base.at_threshold(thr).pairs == direct.pairs
        assert base.at_threshold(thr).dot_bracket == direct.dot_bracket


# ---------------------------------------------------------------------------
# ViennaRNA supplies geometry ONLY. These tests exist to prove that what gets
# drawn is Minerva's predicted structure and never ViennaRNA's own fold.
# ---------------------------------------------------------------------------
def test_layout_honours_a_structure_viennarna_would_never_fold():
    """The decisive check: lay out a physically impossible structure.

    Every pair here is A-A, which no thermodynamic fold would ever produce. If
    the layout still places those pairs together, the geometry is following
    Minerva's structure rather than anything ViennaRNA computed itself.
    """
    n = 40
    seq = "A" * n
    pairs = [(k, n - 1 - k, 0.9) for k in range(12)]
    s = call_structure(_map_from_pairs(pairs, n), seq, name="impossible")
    assert s.canonical_fraction() == 0.0          # not a foldable structure
    assert len(s.nested_pairs) == 12

    xs, ys = naview_layout(s)
    p = np.column_stack([xs, ys])
    step = float(np.median(np.hypot(*np.diff(p, axis=0).T)))
    for i, j, _ in s.nested_pairs:
        assert np.hypot(*(p[j] - p[i])) < 2.5 * step


def test_layout_ignores_the_sequence_entirely():
    """Same pairs, different sequence -> identical geometry."""
    n = 40
    pairs = [(k, n - 1 - k, 0.9) for k in range(12)]
    m = _map_from_pairs(pairs, n)
    a = naview_layout(call_structure(m, "A" * n))
    b = naview_layout(call_structure(m, "".join("GC"[k % 2] for k in range(n))))
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_module_never_calls_a_folding_routine():
    """No ViennaRNA call other than layout appears in the module.

    Checked on the parsed syntax tree rather than the text, so prose mentioning
    a folding function does not trip it and a real call cannot hide.
    """
    import ast
    import inspect
    from minerva import rna_structure

    allowed = {"naview_xy_coordinates"}
    used = set()
    for node in ast.walk(ast.parse(inspect.getsource(rna_structure))):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "RNA":
                used.add(node.attr)
    assert used <= allowed, (
        f"ViennaRNA is for layout only, but the module calls RNA.{sorted(used - allowed)}; "
        "anything that folds would draw a structure Minerva did not predict"
    )
    assert used, "expected the layout call to be found"


# ---------------------------------------------------------------------------
# Mixed loci: nucleotides arrive in separate chunks between genes, and each
# chunk is its own molecule.
# ---------------------------------------------------------------------------
MIXED_TOKENS = (["<+>"] + list("acgtacgtacgt") + ["<+>"] + list("MKVLAIT")
                + ["<+>"] + list("ttttggggcccc"))


def test_nucleotide_regions_finds_each_run():
    assert nucleotide_regions(MIXED_TOKENS) == [(1, 13), (22, 34)]


def test_call_structure_refuses_to_splice_separate_regions():
    """Folding two intergenic regions as one sequence would invent pairs."""
    n = len(MIXED_TOKENS)
    with pytest.raises(ValueError, match="separate nucleotide regions"):
        call_structure(np.zeros((n, n)), tokens=MIXED_TOKENS)


def test_call_structures_splits_and_keeps_pairs_within_a_region():
    n = len(MIXED_TOKENS)
    m = np.zeros((n, n))
    m[1, 12] = m[12, 1] = 0.9        # inside region 1
    m[22, 33] = m[33, 22] = 0.9      # inside region 2
    m[5, 30] = m[30, 5] = 0.95       # ACROSS the gene: not one molecule's structure

    out = call_structures(m, MIXED_TOKENS, name="locus")
    assert [len(s.sequence) for s in out] == [12, 12]
    assert [s.offset for s in out] == [1, 22]
    assert [(i, j) for i, j, _ in out[0].pairs] == [(0, 11)]
    assert [(i, j) for i, j, _ in out[1].pairs] == [(0, 11)]
    # The cross-region contact belongs to the contact map, not to either structure.
    assert all(len(s.pairs) == 1 for s in out)


def test_call_structures_min_length_skips_short_runs():
    tokens = ["<+>"] + list("acg") + ["<+>"] + list("MK") + ["<+>"] + list("ttttggggcccc")
    n = len(tokens)
    assert len(call_structures(np.zeros((n, n)), tokens)) == 2
    assert len(call_structures(np.zeros((n, n)), tokens, min_length=8)) == 1
