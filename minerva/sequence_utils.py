"""Canonical genome ↔ token ↔ embedding position mapping utilities.

Every function that touches the offset between the ``<+>`` prefix token
and genome positions lives here so the mapping is defined once and
tested rather than reimplemented ad-hoc in notebooks and scripts.

Token layout for raw-sequence tokenisation
------------------------------------------
index 0   → ``<+>`` (strand token, ID 33)
index 1   → first nucleotide of the input sequence
index i+1 → genome position i  (0-indexed)

So model outputs with shape ``[batch, seq_len+1, ...]`` must be trimmed
by *offset=1* to align with the genome coordinate system.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from Bio.Seq import Seq


# ---------------------------------------------------------------------------
# Mixed-token reverse complement
# ---------------------------------------------------------------------------

_DNA_CHARS = frozenset("acgtn")
_COMPLEMENT = str.maketrans("acgtn", "tgcan")
_MARKER_RE = re.compile(r"(<[+-]>)")


def _parse_segments(token_string: str) -> list[tuple[str, str]]:
    """Split a mixed-token string into ``(marker, content)`` pairs.

    >>> _parse_segments("<+>aaa<+>MAQ<->LSH<+>ttt")
    [('<+>', 'aaa'), ('<+>', 'MAQ'), ('<->', 'LSH'), ('<+>', 'ttt')]

    Leading content before any marker is dropped (shouldn't happen in
    well-formed strings).
    """
    parts = _MARKER_RE.split(token_string)
    segments: list[tuple[str, str]] = []
    i = 0
    # Skip any leading text before the first marker
    while i < len(parts) and not _MARKER_RE.fullmatch(parts[i]):
        i += 1
    while i < len(parts):
        marker = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        segments.append((marker, content))
        i += 2
    return segments


def _is_dna_segment(content: str) -> bool:
    """Return True if *content* consists entirely of DNA characters (acgtn)."""
    return all(ch in _DNA_CHARS for ch in content)


def rc_mixed_token_string(token_string: str) -> str:
    """Return the reverse-complement of a mixed-modality token string.

    Rules
    -----
    1. Reverse the segment order (3'→5' becomes 5'→3').
    2. Complement DNA segments (a↔t, c↔g, n stays n).
    3. Keep amino-acid segments left-to-right (N→C preserved).
    4. Flip CDS strand markers (``<+>`` ↔ ``<->``).
    5. Intergenic (DNA) segments always use ``<+>``.

    Example::

        >>> rc_mixed_token_string("<+>aaa<+>MAQ<->LSH<+>ttt")
        '<+>aaa<+>LSH<->MAQ<+>ttt'
    """
    segments = _parse_segments(token_string)
    if not segments:
        return token_string

    rc_segments: list[str] = []
    for marker, content in reversed(segments):
        if _is_dna_segment(content):
            # DNA: reverse-complement the bases, always use <+>
            rc_segments.append("<+>" + content.translate(_COMPLEMENT)[::-1])
        else:
            # CDS (amino acids): flip the strand marker, keep AA order
            flipped = "<->" if marker == "<+>" else "<+>"
            rc_segments.append(flipped + content)

    return "".join(rc_segments)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize_genome_window(
    sequence: str,
    tokenizer,
) -> dict:
    """Tokenize a raw genome subsequence for inference.

    Parameters
    ----------
    sequence : str
        Raw nucleotide string (upper- or lower-case).
    tokenizer
        A Minerva / HuggingFace-compatible tokenizer.

    Returns
    -------
    dict
        ``input_ids``  – ``torch.Tensor`` of shape ``[1, genome_len + 1]``
        ``genome_len`` – ``int``, number of nucleotides
        ``offset``     – ``int``, always 1 (index of first nucleotide token)
    """
    seq_lower = sequence.lower()
    tokenized_str = f"<+>{seq_lower}"
    encoded = tokenizer(tokenized_str, return_tensors="pt")
    input_ids: torch.Tensor = encoded["input_ids"]  # [1, genome_len+1]

    return {
        "input_ids": input_ids,
        "genome_len": len(seq_lower),
        "offset": 1,
    }


# ---------------------------------------------------------------------------
# Mixed-token (Prodigal-annotated) sequence building
# ---------------------------------------------------------------------------

_SENTINEL = (-1, -1)


def build_prodigal_mixed_sequence(
    sequence: str,
    prodigal_cds: list[dict],
    translation_table: int = 11,
    max_tokens: int | None = None,
) -> dict:
    """Build Minerva mixed-token input from raw genome + Prodigal CDS calls.

    Mirrors the logic of ``extract_and_tokenize_gb()`` but works from
    ``prodigal_cds.json`` instead of a GenBank file.  CDS regions are
    translated to amino acids (upper-case); intergenic regions stay as
    lower-case nucleotides.

    Parameters
    ----------
    sequence : str
        Full genome nucleotide string.
    prodigal_cds : list[dict]
        Each dict must have ``start``, ``end``, ``strand`` (1 or -1).
    translation_table : int
        NCBI genetic code table for translation (default 11, bacteria).
    max_tokens : int or None
        Optional context-length cap, counted in *tokens*. Because the tokenizer
        is character-level and adds no special tokens, one token == one amino
        acid, one nucleotide, or one strand marker. When set, segments are
        emitted only as far as the budget allows: a CDS or intergenic run is
        filled up to the cap (a partial protein/DNA run is fine), but never
        emitted as a bare dangling marker, so the result is always a valid,
        self-consistent mixed-token string whose ``token_string`` tokenizes to
        ``<= max_tokens`` tokens. Truncation keeps the left (5') end. Use the
        checkpoint's context here (e.g. 4096 for ``gbrixi/minerva-1``, 8192 for
        ``gbrixi/minerva-1-8k``). Defaults to None (no cap).

    Returns
    -------
    dict with keys:

    ``token_string``
        The full mixed-token string ready for the tokenizer.
    ``token_to_genome``
        list[tuple[int, int]] — per-character (genome_start, genome_end).
        Strand markers get ``(-1, -1)`` sentinel.
        For nucleotide chars: ``(pos, pos + 1)``.
        For amino acid chars: ``(codon_start, codon_start + 3)``.
    ``genome_to_token``
        np.ndarray of shape ``(len(sequence),)`` — maps each genome
        position to its primary token-character index in ``token_string``
        (i.e. the index that the tokenizer will turn into a token id).
    """
    seq_len = len(sequence)

    # Sort CDS by start, then longest first for overlap handling
    cds_sorted = sorted(prodigal_cds, key=lambda c: (c["start"], -(c["end"] - c["start"])))

    # Build list of tokens and mappings
    token_chars: list[str] = []       # individual characters of the token string
    char_to_genome: list[tuple[int, int]] = []  # per-char genome span
    genome_to_char = np.full(seq_len, -1, dtype=np.int64)

    coverage_end = 0  # "high water mark" tracking furthest CDS end seen

    # Context-length budget, counted in tokens (marker/aa/nt each == 1 token).
    # Emit functions return True once the budget is exhausted so the driver
    # loop can stop at a segment boundary.
    budget = float("inf") if max_tokens is None else int(max_tokens)
    token_count = 0

    def _emit_intergenic(start: int, end: int) -> bool:
        """Emit <+> + lowercase nucleotides. Returns True if the budget is full."""
        nonlocal token_count
        if end <= start:
            return False
        # Need room for the marker plus at least one base, else stop cleanly.
        if budget - token_count < 2:
            return True
        # Strand marker (one token)
        for ch in "<+>":
            token_chars.append(ch)
            char_to_genome.append(_SENTINEL)
        token_count += 1
        # Nucleotides
        for pos in range(start, end):
            if budget - token_count < 1:
                return True
            token_chars.append(sequence[pos].lower())
            idx = len(token_chars) - 1
            char_to_genome.append((pos, pos + 1))
            genome_to_char[pos] = idx
            token_count += 1
        return False

    def _emit_cds(cds: dict) -> bool:
        """Emit <+>/<-> + translated protein. Returns True if the budget is full."""
        nonlocal token_count
        s, e = cds["start"], cds["end"]
        strand = cds["strand"]

        # Extract and translate first so we can honor the budget per residue.
        dna = sequence[s:e]
        if strand == -1:
            dna = str(Seq(dna).reverse_complement())

        # Trim to codon-aligned
        remainder = len(dna) % 3
        if remainder:
            dna = dna[:-remainder]
        if len(dna) < 3:
            return False

        protein = str(Seq(dna).translate(table=translation_table)).replace("*", "")
        if not protein:
            return False

        # Need room for the marker plus at least one residue, else stop cleanly.
        if budget - token_count < 2:
            return True

        # Strand marker (one token)
        marker = "<+>" if strand == 1 else "<->"
        for ch in marker:
            token_chars.append(ch)
            char_to_genome.append(_SENTINEL)
        token_count += 1

        # Map each AA to its codon positions in genome coordinates
        for aa_idx, aa_char in enumerate(protein):
            if budget - token_count < 1:
                return True
            if strand == 1:
                codon_start = s + aa_idx * 3
            else:
                # Reverse strand: codons run from the genomic 3' end backwards
                codon_start = e - (aa_idx + 1) * 3

            token_chars.append(aa_char.upper())
            char_idx = len(token_chars) - 1
            char_to_genome.append((codon_start, codon_start + 3))
            token_count += 1

            # Map all 3 codon positions to this char
            for offset in range(3):
                gpos = codon_start + offset
                if 0 <= gpos < seq_len:
                    genome_to_char[gpos] = char_idx
        return False

    # Process regions in order (mirrors extract_and_tokenize_gb logic)
    if not cds_sorted:
        _emit_intergenic(0, seq_len)
    else:
        full = False
        # Before first CDS
        if cds_sorted[0]["start"] > 0:
            full = _emit_intergenic(0, cds_sorted[0]["start"])

        for i, cds in enumerate(cds_sorted):
            if full:
                break
            if _emit_cds(cds):
                break
            coverage_end = max(coverage_end, cds["end"])

            # Intergenic gap after this CDS
            if i < len(cds_sorted) - 1:
                gap_start = coverage_end
                gap_end = cds_sorted[i + 1]["start"]
                if gap_end > gap_start:
                    if _emit_intergenic(gap_start, gap_end):
                        full = True

        # After last CDS
        if not full and coverage_end < seq_len:
            _emit_intergenic(coverage_end, seq_len)

    token_string = "".join(token_chars)

    return {
        "token_string": token_string,
        "token_to_genome": char_to_genome,
        "genome_to_token": genome_to_char,
    }


# Strand markers are single tokens; matching them as one unit lets us count and
# window in *token* space rather than character space.
_TOKEN_RE = re.compile(r"<[+-]>|.")


def mixed_token_length(token_string: str) -> int:
    """Number of tokens in a mixed-token string (each ``<+>``/``<->`` counts as 1).

    Because the tokenizer is character-level and adds no special tokens, this
    equals the length the tokenizer will produce.
    """
    return len(_TOKEN_RE.findall(token_string))


def chunk_sequence_with_stride(
    sequence: str,
    chunk_size: int,
    stride: int,
) -> list[str]:
    """Slide a ``chunk_size``-token window over a mixed-token string.

    This is the same primitive used for genome scanning: it tiles a long
    per-LOCUS mixed-token string into (usually overlapping) windows so a genome
    larger than the model context can be processed — or trained on — without
    dropping the tail. ``<+>`` / ``<->`` markers are treated as single
    characters, so ``chunk_size`` and ``stride`` are counted in **tokens**.

    Parameters
    ----------
    sequence : str
        Mixed-token string (e.g. ``token_string`` from
        :func:`build_prodigal_mixed_sequence` or a record ``sequence`` from
        :func:`minerva.data.extract_and_tokenize_gb`).
    chunk_size : int
        Window length in tokens (use the checkpoint context, e.g. 4096 / 8192).
    stride : int
        Step between window starts in tokens. ``stride < chunk_size`` gives
        overlapping windows (e.g. ``stride = chunk_size // 2`` for 50% overlap),
        so features straddling a boundary still appear whole in some window.

    Returns
    -------
    list[str]
        Windows, in order. A sequence of ``<= chunk_size`` tokens returns a
        single element (the whole sequence).
    """
    if not isinstance(chunk_size, int) or not isinstance(stride, int):
        raise TypeError("chunk_size and stride must be integers")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    # Collapse each strand marker to a single sentinel char so slicing counts
    # markers as one token, then restore them on the way out.
    placeholders = {"<+>": "", "<->": ""}
    modified = sequence
    for marker, placeholder in placeholders.items():
        modified = modified.replace(marker, placeholder)

    chunks = [
        modified[i : i + chunk_size]
        for i in range(0, len(modified), stride)
    ]

    restored = []
    for chunk in chunks:
        for marker, placeholder in placeholders.items():
            chunk = chunk.replace(placeholder, marker)
        restored.append(chunk)
    return restored


def tokenize_mixed_window(
    token_string: str,
    tokenizer,
) -> dict:
    """Tokenize a mixed-token string (from ``build_prodigal_mixed_sequence``).

    Unlike ``tokenize_genome_window``, the string already contains strand
    markers — so we pass it directly to the tokenizer without prepending
    ``<+>``.

    Returns
    -------
    dict
        ``input_ids`` — ``torch.Tensor [1, n_tokens]``
        ``n_tokens``  — int
    """
    encoded = tokenizer(token_string, return_tensors="pt")
    input_ids = encoded["input_ids"]
    return {
        "input_ids": input_ids,
        "n_tokens": input_ids.shape[1],
    }


# ---------------------------------------------------------------------------
# Feature ↔ genome mapping for mixed-token sequences
# ---------------------------------------------------------------------------

def map_features_to_genome(
    features: np.ndarray,
    token_to_genome: list[tuple[int, int]],
    seq_len: int,
    tokenizer,
    input_ids: torch.Tensor,
) -> np.ndarray:
    """Map per-token model outputs back to genome-position resolution.

    Parameters
    ----------
    features : np.ndarray
        Shape ``(n_tokens, D)`` — per-token features from model output.
    token_to_genome : list[tuple[int, int]]
        Per-character mapping from ``build_prodigal_mixed_sequence``.
        Length matches the character count of the token string.
    seq_len : int
        Genome length.
    tokenizer
        The tokenizer (to decode token ids back to characters).
    input_ids : torch.Tensor
        Shape ``[1, n_tokens]`` — the token ids fed to the model.

    Returns
    -------
    mapped : np.ndarray
        Shape ``(seq_len, D)`` — features broadcast to genome positions.
        Positions not covered by any token get zeros.
    counts : np.ndarray
        Shape ``(seq_len,)`` — number of tokens contributing to each position
        (for weighted averaging in overlapping windows).
    """
    n_tokens, feat_dim = features.shape
    mapped = np.zeros((seq_len, feat_dim), dtype=np.float32)
    counts = np.zeros(seq_len, dtype=np.float32)

    # Decode token ids to figure out per-token character spans
    ids = input_ids.squeeze().tolist()

    # Build token_idx → list of char indices in the original token_string
    # Each token maps to 1+ characters. We reconstruct this by decoding.
    vocab = tokenizer.get_vocab()
    id_to_str = {v: k for k, v in vocab.items()}

    char_offset = 0
    for tok_idx, tok_id in enumerate(ids):
        tok_str = id_to_str.get(tok_id, "?")
        tok_len = len(tok_str)

        # Gather genome spans for all characters in this token
        for ci in range(tok_len):
            abs_ci = char_offset + ci
            if abs_ci >= len(token_to_genome):
                break
            gstart, gend = token_to_genome[abs_ci]
            if gstart == -1:  # sentinel (strand marker)
                continue
            # Broadcast feature to all genome positions in this span
            for gpos in range(gstart, min(gend, seq_len)):
                mapped[gpos] += features[tok_idx]
                counts[gpos] += 1.0

        char_offset += tok_len

    # Average where multiple tokens hit the same position
    nonzero = counts > 0
    mapped[nonzero] /= counts[nonzero, None]

    return mapped, counts


def build_token_to_genome_map(
    token_string: str,
    char_to_genome: list[tuple[int, int]],
    tokenizer,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Build a per-token genome span mapping from char-level mapping.

    Parameters
    ----------
    token_string : str
        The full mixed-token string from ``build_prodigal_mixed_sequence``.
    char_to_genome : list[tuple[int, int]]
        Per-character genome mapping (same length as ``token_string``).
    tokenizer
        The Minerva tokenizer.

    Returns
    -------
    token_ids : list[int]
        Token IDs for the full string.
    tok_to_genome : list[tuple[int, int]]
        Per-token ``(genome_start, genome_end)``.  ``(-1, -1)`` for strand
        markers or tokens with no genome coverage.
    """
    encoded = tokenizer(token_string, return_tensors="pt")
    token_ids = encoded["input_ids"].squeeze().tolist()

    vocab = tokenizer.get_vocab()
    id_to_str = {v: k for k, v in vocab.items()}

    tok_to_genome = []
    char_offset = 0

    for tok_id in token_ids:
        tok_str = id_to_str.get(tok_id, "?")
        tok_char_len = len(tok_str)

        gmin, gmax = -1, -1
        for ci in range(tok_char_len):
            abs_ci = char_offset + ci
            if abs_ci < len(char_to_genome):
                gs, ge = char_to_genome[abs_ci]
                if gs != -1:
                    if gmin == -1:
                        gmin, gmax = gs, ge
                    else:
                        gmin = min(gmin, gs)
                        gmax = max(gmax, ge)

        tok_to_genome.append((gmin, gmax))
        char_offset += tok_char_len

    return token_ids, tok_to_genome


# ---------------------------------------------------------------------------
# Trimming helpers
# ---------------------------------------------------------------------------

def trim_hidden_states(
    hidden: torch.Tensor,
    offset: int = 1,
) -> torch.Tensor:
    """Strip prefix token(s) from hidden states.

    ``[batch, seq_len+1, dim]`` → ``[batch, genome_len, dim]``
    """
    return hidden[:, offset:, :]


def trim_contact_map(
    contact_map: torch.Tensor,
    offset: int = 1,
) -> torch.Tensor:
    """Strip prefix token row/column from a contact map.

    ``[seq_len+1, seq_len+1]`` → ``[genome_len, genome_len]``
    """
    return contact_map[offset:, offset:]


def trim_attention_maps(
    attn_maps: Dict[int, torch.Tensor],
    offset: int = 1,
) -> Dict[int, torch.Tensor]:
    """Strip prefix token from attention maps.

    ``{layer: [batch, heads, L+1, L+1]}`` → ``{layer: [batch, heads, L, L]}``
    """
    return {
        layer: attn[:, :, offset:, offset:]
        for layer, attn in attn_maps.items()
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Forward + RC permutation and attention combining
# ---------------------------------------------------------------------------


def _segment_token_ranges(
    segments: list[tuple[str, str]],
) -> list[tuple[int, int, int]]:
    """Compute (marker_idx, content_start, content_end) for each segment.

    Each marker is 1 token, each content character is 1 token.
    """
    ranges = []
    pos = 0
    for _marker, content in segments:
        marker_idx = pos
        pos += 1  # marker token
        content_start = pos
        pos += len(content)
        content_end = pos
        ranges.append((marker_idx, content_start, content_end))
    return ranges


def build_fwd_rc_permutation(token_string: str) -> list[int]:
    """Build a permutation mapping forward token indices to RC token indices.

    For a mixed-token string, computes where each forward-strand token
    ends up in the reverse-complement token string.  ``P[fwd_idx] = rc_idx``
    means the token at position *fwd_idx* in the forward string corresponds
    to the token at position *rc_idx* in the RC string.

    Rules per segment pair (fwd segment *i* ↔ RC segment *n-1-i*):
    - Markers map to each other.
    - DNA content: positions reverse within the segment.
    - CDS content: positions keep the same relative order (N→C preserved).
    """
    fwd_segments = _parse_segments(token_string)
    rc_string = rc_mixed_token_string(token_string)
    rc_segments = _parse_segments(rc_string)

    n_segs = len(fwd_segments)
    fwd_ranges = _segment_token_ranges(fwd_segments)
    rc_ranges = _segment_token_ranges(rc_segments)

    n_tokens = sum(1 + len(content) for _, content in fwd_segments)
    perm = [0] * n_tokens

    for i in range(n_segs):
        j = n_segs - 1 - i  # paired RC segment
        fwd_marker, fwd_start, fwd_end = fwd_ranges[i]
        rc_marker, rc_start, rc_end = rc_ranges[j]

        # Map marker
        perm[fwd_marker] = rc_marker

        # Map content
        content_len = fwd_end - fwd_start
        _marker_str, content = fwd_segments[i]
        if _is_dna_segment(content):
            # DNA: reverse within segment
            for k in range(content_len):
                perm[fwd_start + k] = rc_start + (content_len - 1 - k)
        else:
            # CDS: same relative order
            for k in range(content_len):
                perm[fwd_start + k] = rc_start + k

    return perm


def remap_rc_attention(
    rc_attn: torch.Tensor,
    perm: list[int],
) -> torch.Tensor:
    """Reindex RC attention map to forward-strand token coordinates.

    Given an attention tensor in RC token order and a permutation *perm*
    where ``perm[fwd_idx] = rc_idx``, returns an attention tensor where
    ``result[..., i, j] = rc_attn[..., perm[i], perm[j]]``.

    Works for any ``(..., L, L)`` shape (batch, head dims via ``...``).
    """
    idx = torch.tensor(perm, dtype=torch.long, device=rc_attn.device)
    return rc_attn[..., idx[:, None], idx[None, :]]


def extract_fwd_rc_attention(
    model,
    tokenizer,
    token_string: str,
    layers: list[int] | None = None,
) -> dict:
    """Run model on forward and RC, return aligned and combined attention maps.

    Parameters
    ----------
    model
        A Minerva model with a ``get_attention_maps`` method.
    tokenizer
        The Minerva tokenizer.
    token_string : str
        Forward-strand mixed-token string (e.g. ``<+>aaa<+>MAQ<->LSH<+>ttt``).
    layers : list[int] or None
        Transformer layers to extract.  ``None`` → all layers.

    Returns
    -------
    dict with keys:

    ``fwd_attention``
        ``{layer: (batch, heads, L, L)}`` — forward attention maps.
    ``rc_attention``
        ``{layer: (batch, heads, L, L)}`` — raw RC-coordinate attention maps.
    ``rc_aligned``
        ``{layer: (batch, heads, L, L)}`` — RC attention remapped to forward
        token coordinates.
    ``combined``
        ``{layer: (batch, heads, L, L)}`` — upper triangle from forward,
        lower triangle from RC-aligned, diagonal from forward.
    ``permutation``
        ``list[int]`` — the forward→RC token index map.
    ``rc_token_string``
        ``str`` — the RC token string.
    """
    rc_string = rc_mixed_token_string(token_string)
    perm = build_fwd_rc_permutation(token_string)

    fwd_enc = tokenizer(token_string, return_tensors="pt")
    rc_enc = tokenizer(rc_string, return_tensors="pt")

    device = next(model.parameters()).device
    fwd_ids = fwd_enc["input_ids"].to(device)
    rc_ids = rc_enc["input_ids"].to(device)

    fwd_attn = model.get_attention_maps(fwd_ids, layers=layers)
    rc_attn = model.get_attention_maps(rc_ids, layers=layers)

    rc_aligned = {
        layer: remap_rc_attention(a, perm) for layer, a in rc_attn.items()
    }

    # Combine: upper tri from fwd, lower tri from rc_aligned, diagonal from fwd
    L = fwd_ids.shape[1]
    mask_upper = torch.triu(torch.ones(L, L, device=device), diagonal=1)
    diag = torch.eye(L, device=device)
    mask_lower = torch.tril(torch.ones(L, L, device=device), diagonal=-1)

    combined = {}
    for layer in fwd_attn:
        combined[layer] = (
            fwd_attn[layer] * (mask_upper + diag)
            + rc_aligned[layer] * mask_lower
        )

    return {
        "fwd_attention": fwd_attn,
        "rc_attention": rc_attn,
        "rc_aligned": rc_aligned,
        "combined": combined,
        "permutation": perm,
        "rc_token_string": rc_string,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_token_mapping(
    input_ids: torch.Tensor,
    sequence: str,
    tokenizer,
) -> None:
    """Assert that *input_ids* encode ``<+>`` followed by *sequence*.

    Raises ``AssertionError`` with a descriptive message on failure.

    Parameters
    ----------
    input_ids : torch.Tensor
        Shape ``[1, genome_len+1]`` or ``[genome_len+1]``.
    sequence : str
        The raw nucleotide string that was tokenized.
    tokenizer
        The tokenizer used to produce *input_ids*.
    """
    ids = input_ids.squeeze().tolist()
    seq_lower = sequence.lower()

    # Length check
    expected_len = len(seq_lower) + 1  # +1 for <+>
    assert len(ids) == expected_len, (
        f"Token length mismatch: got {len(ids)}, expected {expected_len} "
        f"(genome_len={len(seq_lower)} + 1 prefix)"
    )

    # Prefix token check
    vocab = tokenizer.get_vocab()
    plus_id = vocab.get("<+>")
    assert plus_id is not None, "Tokenizer vocabulary does not contain '<+>'"
    assert ids[0] == plus_id, (
        f"First token should be <+> (id={plus_id}), got id={ids[0]}"
    )

    # Nucleotide identity check (spot-check first 20 and last 20)
    nuc_map = {nt: vocab[nt] for nt in "acgt" if nt in vocab}
    check_positions = list(range(min(20, len(seq_lower))))
    if len(seq_lower) > 20:
        check_positions += list(range(len(seq_lower) - 20, len(seq_lower)))

    for pos in check_positions:
        expected_id = nuc_map.get(seq_lower[pos])
        if expected_id is None:
            continue  # skip non-ACGT (N, etc.)
        actual_id = ids[pos + 1]  # +1 for prefix offset
        assert actual_id == expected_id, (
            f"Token mismatch at genome pos {pos}: "
            f"expected '{seq_lower[pos]}' (id={expected_id}), "
            f"got id={actual_id}"
        )
