"""GenBank -> Minerva-ready mixed DNA+AA token sequences. Self-contained (Biopython + numpy)."""

from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import re


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
    1. Reverse the segment order (3'->5' becomes 5'->3').
    2. Complement DNA segments (a<->t, c<->g, n stays n).
    3. Keep amino-acid segments left-to-right (N->C preserved).
    4. Flip CDS strand markers (``<+>`` <-> ``<->``).
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


def _translate_dna(dna_seq: str, table: int = 11) -> str:
    """Translate DNA sequence to protein, handling codon alignment."""
    remainder = len(dna_seq) % 3
    if remainder != 0:
        dna_seq = dna_seq[:-remainder]

    if len(dna_seq) < 3:
        return ""

    protein_seq = str(Seq(dna_seq).translate(table=table))
    protein_seq = protein_seq.replace("*", "")
    return protein_seq


def _extract_and_translate_cds(
    feature,
    record_seq: str,
    use_existing_translations: bool = False,
    default_translation_table: int = 11,
) -> str:
    """
    Extract and translate a CDS feature with proper handling of:
    - use_existing_translations: Use translation qualifier if present
    - codon_start: Handle reading frame offset
    - Truncated genes: Handle partial 5'/3' genes
    - translation table: honor the feature's ``/transl_table`` qualifier when
      present, otherwise fall back to ``default_translation_table``
      (11 = bacterial/archaeal/plant-plastid, the standard for prokaryotic
      genomes).

    Returns the protein sequence.
    """
    # Use existing translation if available and requested
    if use_existing_translations and "translation" in feature.qualifiers:
        protein_seq = feature.qualifiers["translation"][0]
        return protein_seq.replace("*", "")

    # Extract and translate the CDS sequence
    dna_seq = feature.extract(record_seq)
    if hasattr(dna_seq, 'seq'):
        dna_seq = str(dna_seq.seq)
    else:
        dna_seq = str(dna_seq)

    remainder = len(dna_seq) % 3

    if remainder != 0:
        is_5prime_truncated = False
        is_3prime_truncated = False

        # Check codon_start qualifier
        if "codon_start" in feature.qualifiers:
            codon_start = int(feature.qualifiers["codon_start"][0])
            if codon_start > 1:
                dna_seq = dna_seq[codon_start-1:]
                remainder = len(dna_seq) % 3
                is_5prime_truncated = True

        if remainder != 0:
            location_str = str(feature.location)
            # Check for position operators in string representation
            if '<' in location_str:
                is_5prime_truncated = True
            if '>' in location_str:
                is_3prime_truncated = True

            # Decide how to trim based on truncation status
            if is_3prime_truncated:
                dna_seq = dna_seq[:-remainder]
            elif is_5prime_truncated:
                dna_seq = dna_seq[remainder:]
            else:
                # Default to trimming from the end
                dna_seq = dna_seq[:-remainder]

    if len(dna_seq) < 3:
        return ""

    # Honor a per-CDS /transl_table qualifier when present; otherwise use the
    # caller-supplied default (11 for prokaryotes).
    table = default_translation_table
    if "transl_table" in feature.qualifiers:
        try:
            table = int(feature.qualifiers["transl_table"][0])
        except (ValueError, TypeError):
            table = default_translation_table

    protein_seq = str(Seq(dna_seq).translate(table=table))
    protein_seq = protein_seq.replace("*", "")
    return protein_seq


def _subtract_interval(interval: tuple[int, int], claimed: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Subtract claimed intervals from a single interval.

    Args:
        interval: (start, end) tuple
        claimed: List of (start, end) tuples representing already-claimed regions

    Returns:
        List of (start, end) tuples representing unclaimed portions of the interval
    """
    if not claimed:
        return [interval]

    start, end = interval
    unclaimed = [(start, end)]

    for claim_start, claim_end in claimed:
        new_unclaimed = []
        for seg_start, seg_end in unclaimed:
            # No overlap
            if claim_end <= seg_start or claim_start >= seg_end:
                new_unclaimed.append((seg_start, seg_end))
            else:
                # Overlap - split the segment
                if seg_start < claim_start:
                    new_unclaimed.append((seg_start, claim_start))
                if claim_end < seg_end:
                    new_unclaimed.append((claim_end, seg_end))
        unclaimed = new_unclaimed

    return unclaimed


def _resolve_overlaps(
    features: list[dict],
    record_seq: str,
    mode: str = "expand"
) -> list[dict]:
    """
    Resolve overlapping CDS by prioritizing longer genes.

    Overlaps are resolved regardless of strand (antisense overlaps are treated the same).
    - mode="expand": Keep all CDS features, even overlapping ones
    - mode="long": Shorter overlapping genes are discarded entirely

    Args:
        features: List of feature dicts with 'start', 'end', 'orientation', 'seq', 'gene_name'
        record_seq: The full genomic sequence string
        mode: "expand" to keep all, "long" to discard overlapping genes

    Returns:
        List of features (possibly modified based on mode)
    """
    if not features:
        return features

    # Expand mode: keep all CDS, just sort by position
    if mode == "expand":
        result = [f.copy() for f in features]
        result.sort(key=lambda x: x["start"])
        return result

    # Long mode: keep longest, discard overlapping shorter ones
    # Sort by length (longest first), then by start position
    sorted_by_length = sorted(
        features,
        key=lambda x: (-(x["end"] - x["start"]), x["start"])
    )

    claimed: list[tuple[int, int]] = []
    result = []

    for feat in sorted_by_length:
        # Check if ANY overlap with claimed intervals - if so, discard entirely
        has_overlap = False
        for claim_start, claim_end in claimed:
            if feat["start"] < claim_end and claim_start < feat["end"]:
                has_overlap = True
                break

        if has_overlap:
            continue

        # No overlap - keep the entire feature with its original translation
        result.append(feat.copy())
        claimed.append((feat["start"], feat["end"]))

    # Sort by start position
    result.sort(key=lambda x: x["start"])
    return result


def extract_and_tokenize_gb(
    gb_file: str,
    use_existing_translations: bool = False,
    overlap_mode: str = "expand",
    other_feature_types_to_track: list = None,
    reverse_complement: bool = False,
    translation_table: int = 11,
):
    """
    Extract features from a GenBank file and tokenize them into one sequence per LOCUS.

    Args:
        gb_file: Path to GenBank file
        use_existing_translations: Use translation qualifier if present
        overlap_mode: How to handle overlapping CDS features
            - "expand": Keep all CDS features, even overlapping ones
            - "long": Only keep longest gene at each position, discard overlapping ones
        other_feature_types_to_track: List of additional feature types to extract and track
            (e.g., ['ncRNA', 'tRNA', 'rRNA', 'tmRNA', 'misc_RNA', 'regulatory']).
            Pass ``"all"`` to auto-discover and track every non-CDS, non-source,
            non-gene feature type present in the file.
            These features are tracked but don't affect tokenization. Defaults to None.
        reverse_complement: If True, reverse-complement the mixed-token
            sequence (complement DNA, reverse segment order, flip CDS strand
            markers).  Feature and intergenic lists are reversed accordingly.
        translation_table: Default NCBI genetic-code table used to translate
            CDS features that do not carry their own ``/transl_table``
            qualifier. Defaults to 11 (bacterial/archaeal/plant plastid). A
            feature's own ``/transl_table`` always takes precedence.
    """
    if overlap_mode not in ("expand", "long"):
        raise ValueError(f"Unknown overlap_mode: {overlap_mode}. Must be 'expand' or 'long'")

    # Handle default parameter
    if other_feature_types_to_track is None:
        other_feature_types_to_track = []

    records = list(SeqIO.parse(gb_file, "genbank"))

    # Auto-discover feature types when "all" is requested
    _skip_types = {"CDS", "source", "gene"}
    if other_feature_types_to_track == "all":
        discovered = set()
        for rec in records:
            for feat in rec.features:
                if feat.type not in _skip_types:
                    discovered.add(feat.type)
        other_feature_types_to_track = sorted(discovered)

    tokenized_records = []

    for record_idx, record in enumerate(records):
        sequence = str(record.seq)
        features = []
        other_features = []

        # Extract all CDS features with full translation
        for feature in record.features:
            if feature.type == "CDS":
                start = int(feature.location.start)
                end = int(feature.location.end)
                strand = feature.location.strand
                orientation = True if strand == 1 else False

                # Translate with proper handling of codon_start, truncation, etc.
                protein_seq = _extract_and_translate_cds(
                    feature, record.seq, use_existing_translations,
                    default_translation_table=translation_table,
                )

                if not protein_seq:
                    continue

                # Get gene name if available
                gene_name = ""
                if "gene" in feature.qualifiers:
                    gene_name = feature.qualifiers["gene"][0]
                elif "locus_tag" in feature.qualifiers:
                    gene_name = feature.qualifiers["locus_tag"][0]
                elif "standard_name" in feature.qualifiers:
                    gene_name = feature.qualifiers["standard_name"][0]
                elif "product" in feature.qualifiers:
                    gene_name = feature.qualifiers["product"][0]
                elif "label" in feature.qualifiers:
                    gene_name = feature.qualifiers["label"][0]

                # Get product separately (may differ from gene_name)
                product = ""
                if "product" in feature.qualifiers:
                    product = feature.qualifiers["product"][0]

                features.append({
                    "type": "CDS",
                    "start": start,
                    "end": end,
                    "orientation": orientation,
                    "seq": protein_seq,
                    "gene_name": gene_name,
                    "product": product
                })

            # Extract other feature types if requested
            elif feature.type in other_feature_types_to_track:
                start = int(feature.location.start)
                end = int(feature.location.end)
                strand = feature.location.strand

                # Extract common qualifiers that might be useful
                feature_name = ""
                feature_product = ""
                feature_note = ""

                # Try various qualifier fields for name
                if "gene" in feature.qualifiers:
                    feature_name = feature.qualifiers["gene"][0]
                elif "locus_tag" in feature.qualifiers:
                    feature_name = feature.qualifiers["locus_tag"][0]
                elif "label" in feature.qualifiers:
                    feature_name = feature.qualifiers["label"][0]
                elif "standard_name" in feature.qualifiers:
                    feature_name = feature.qualifiers["standard_name"][0]
                elif "product" in feature.qualifiers:
                    feature_name = feature.qualifiers["product"][0]

                # Try to get product/function information
                if "product" in feature.qualifiers:
                    feature_product = feature.qualifiers["product"][0]
                elif "function" in feature.qualifiers:
                    feature_product = feature.qualifiers["function"][0]

                # Get notes if available
                if "note" in feature.qualifiers:
                    feature_note = feature.qualifiers["note"][0]

                other_features.append({
                    "type": feature.type,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "name": feature_name,
                    "product": feature_product,
                    "note": feature_note
                })

        # Sort features by position, then by length (longest first for tie-breaking)
        features.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

        # Resolve overlapping CDS
        features = _resolve_overlaps(features, sequence, mode=overlap_mode)

        # Process regions in order
        final_tokens = []
        intergenic_regions = []  # Track intergenic regions

        # Handle the case of no CDS features
        if not features:
            if len(sequence) > 0:
                # The entire sequence is intergenic
                final_tokens.append(f"<+>{sequence.lower()}")
                intergenic_regions.append({"start": 0, "end": len(sequence)})
        else:
            # Handle the region before the first CDS
            if features[0]["start"] > 0:
                igs_seq = sequence[0:features[0]["start"]].lower()
                if igs_seq.strip():  # Only add if not empty
                    final_tokens.append(f"<+>{igs_seq}")
                    intergenic_regions.append({"start": 0, "end": features[0]["start"]})

            # Track the "high water mark" - furthest end we've seen
            # This handles overlapping genes correctly
            coverage_end = 0

            # Process each CDS and the intergenic region that follows it
            for i, feature in enumerate(features):
                # Add the CDS with proper orientation
                orientation_token = "<+>" if feature["orientation"] else "<->"
                final_tokens.append(f"{orientation_token}{feature['seq']}")

                # Update high water mark
                coverage_end = max(coverage_end, feature["end"])

                # Add intergenic region after this CDS if not the last one
                if i < len(features) - 1:
                    igs_start = coverage_end
                    igs_end = features[i+1]["start"]
                    if igs_end > igs_start:
                        igs_seq = sequence[igs_start:igs_end].lower()
                        if igs_seq.strip():  # Only add if not empty
                            final_tokens.append(f"<+>{igs_seq}")
                            intergenic_regions.append({"start": igs_start, "end": igs_end})

            # Handle the region after the last CDS
            if coverage_end < len(sequence):
                igs_seq = sequence[coverage_end:].lower()
                if igs_seq.strip():  # Only add if not empty
                    final_tokens.append(f"<+>{igs_seq}")
                    intergenic_regions.append({"start": coverage_end, "end": len(sequence)})

        tokenized_sequence = "".join([token for token in final_tokens if token.strip()])

        # Get the LOCUS name
        locus_name = record.name

        # Map intergenic regions to overlapping other_features
        intergenic_regions_features = []
        for ig_region in intergenic_regions:
            overlapping_features = []
            for other_feat in other_features:
                # Check if feature overlaps with intergenic region
                # Overlap occurs when: feature_start < ig_end AND ig_start < feature_end
                if other_feat["start"] < ig_region["end"] and ig_region["start"] < other_feat["end"]:
                    overlapping_features.append(other_feat)
            intergenic_regions_features.append(overlapping_features)

        # Apply reverse complement if requested
        if reverse_complement:
            tokenized_sequence = rc_mixed_token_string(tokenized_sequence)
            final_tokens = [
                rc_mixed_token_string(tok) for tok in reversed(final_tokens)
            ]
            features = [
                {**f, "orientation": not f["orientation"]}
                for f in reversed(features)
            ]
            intergenic_regions = list(reversed(intergenic_regions))
            intergenic_regions_features = list(reversed(intergenic_regions_features))

        tokenized_records.append({
            "locus_name": locus_name,
            "sequence": tokenized_sequence,
            "tokens": final_tokens,
            "features": features,  # CDS features with gene names
            "other_features": other_features,  # Additional tracked features
            "intergenic_regions": intergenic_regions,  # Intergenic regions
            "intergenic_regions_features": intergenic_regions_features  # Features overlapping each intergenic region
        })

    return tokenized_records
