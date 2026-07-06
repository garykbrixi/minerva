"""Call genes from raw DNA with Pyrodigal and build Minerva mixed-token input.

This closes the gap between a bare FASTA / nucleotide string and Minerva's
mixed protein + DNA token format.  Pyrodigal predicts coding sequences (CDS),
and the calls are handed to
:func:`minerva.sequence_utils.build_prodigal_mixed_sequence`, which translates
CDS to amino acids (upper-case) and keeps intergenic DNA as nucleotides
(lower-case), inserting ``<+>`` / ``<->`` strand markers.

Use this when you only have unannotated sequence.  If you already have a
GenBank file with CDS features, use :func:`minerva.data.extract_and_tokenize_gb`
instead; if you already have CDS calls from an external tool, feed them
straight to ``build_prodigal_mixed_sequence``.
"""

from __future__ import annotations

import warnings

import pyrodigal
from Bio import SeqIO

from .sequence_utils import build_prodigal_mixed_sequence

__all__ = [
    "call_genes",
    "build_minerva_input",
    "read_fasta",
    "fasta_to_minerva_inputs",
]

# Prodigal's single-genome training needs a reasonably long sequence; below
# this threshold training is unreliable (Pyrodigal itself will complain), so we
# transparently fall back to the metagenomic (pretrained) mode.
_MIN_TRAIN_LEN = 20000


def call_genes(
    sequence: str,
    *,
    meta: bool = False,
    translation_table: int = 11,
    closed: bool = False,
) -> list[dict]:
    """Predict CDS in a nucleotide sequence with Pyrodigal.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence for a single contig / genome.
    meta : bool
        Use Pyrodigal's metagenomic mode (pretrained profiles spanning many
        translation tables) instead of training a single-genome model.
        Recommended for short contigs, metagenomic assemblies, or mixed input.
        Sequences shorter than ``20000`` bp cannot be used to train a
        single-genome model, so this falls back to meta mode automatically
        (with a warning).
    translation_table : int
        NCBI genetic-code table used when training a single-genome model, and
        used to translate the called CDS into amino acids.  Ignored for
        *training* in meta mode (which selects tables itself), but still used
        for the final translation.  Default 11 (bacterial / archaeal / plant
        plastid).
    closed : bool
        If True, do not allow genes to run off the sequence edges (no partial
        genes at contig boundaries).

    Returns
    -------
    list[dict]
        One dict per predicted CDS with 0-based, half-open ``start`` / ``end``
        and ``strand`` in ``{1, -1}`` -- directly consumable by
        :func:`build_minerva_input` and
        :func:`minerva.sequence_utils.build_prodigal_mixed_sequence`.
    """
    seq = str(sequence)

    use_meta = meta
    if not use_meta and len(seq) < _MIN_TRAIN_LEN:
        warnings.warn(
            f"Sequence length {len(seq)} bp < {_MIN_TRAIN_LEN} bp is too short "
            "for reliable single-genome Prodigal training; falling back to "
            "metagenomic mode. Pass meta=True to silence this warning.",
            stacklevel=2,
        )
        use_meta = True

    if use_meta:
        finder = pyrodigal.GeneFinder(meta=True, closed=closed)
    else:
        finder = pyrodigal.GeneFinder(meta=False, closed=closed)
        finder.train(seq, translation_table=translation_table)

    genes = finder.find_genes(seq)

    cds: list[dict] = []
    for gene in genes:
        # Pyrodigal coordinates are 1-based inclusive; convert to the 0-based,
        # half-open convention build_prodigal_mixed_sequence expects.
        cds.append(
            {
                "start": gene.begin - 1,
                "end": gene.end,
                "strand": gene.strand,
            }
        )
    return cds


def build_minerva_input(
    sequence: str,
    *,
    meta: bool = False,
    translation_table: int = 11,
    closed: bool = False,
    max_tokens: int | None = None,
) -> dict:
    """Sequence-to-Minerva in one call: call genes, then build mixed tokens.

    Returns the dict from
    :func:`minerva.sequence_utils.build_prodigal_mixed_sequence`
    (``token_string`` plus genome<->token coordinate maps).

    Pass ``max_tokens`` (e.g. the checkpoint's context: 4096 for
    ``gbrixi/minerva``, 8192 for ``gbrixi/minerva-8k``) to cap the output at a
    gene boundary so it fits the model context. See
    :func:`minerva.sequence_utils.build_prodigal_mixed_sequence` for the exact
    capping semantics.
    """
    cds = call_genes(
        sequence,
        meta=meta,
        translation_table=translation_table,
        closed=closed,
    )
    return build_prodigal_mixed_sequence(
        sequence, cds, translation_table=translation_table, max_tokens=max_tokens
    )


def read_fasta(path: str) -> list[tuple[str, str]]:
    """Read a FASTA file into ``[(record_id, sequence), ...]``."""
    return [(rec.id, str(rec.seq)) for rec in SeqIO.parse(path, "fasta")]


def fasta_to_minerva_inputs(
    path: str,
    *,
    meta: bool = False,
    translation_table: int = 11,
    closed: bool = False,
    max_tokens: int | None = None,
) -> list[dict]:
    """Call genes on every record in a FASTA file and build Minerva inputs.

    Returns one result dict per FASTA record (the output of
    :func:`build_minerva_input`), each augmented with an ``id`` key holding the
    FASTA record id. Pass ``max_tokens`` to cap each record at the model
    context (records longer than the cap are truncated at a gene boundary; for
    full-genome tiling see the note in the README).
    """
    results: list[dict] = []
    for rec_id, seq in read_fasta(path):
        out = build_minerva_input(
            seq,
            meta=meta,
            translation_table=translation_table,
            closed=closed,
            max_tokens=max_tokens,
        )
        out["id"] = rec_id
        results.append(out)
    return results
