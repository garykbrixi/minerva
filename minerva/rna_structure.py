"""Turn Minerva base-pairing maps into RNA secondary structures.

    s = call_structure(outputs.interactions["base_pairing"], tokens=token_list)
    s.dot_bracket        # '((((...[[[.))))......]]].'
    s.to_vienna("x.fa")  # read by RNAfold, forna, VARNA, R2R
    s.to_ct("x.ct")      # connect table, keeps pseudoknots
    s.plot()             # matplotlib Figure

Dot-bracket is the portable bit -- sequence plus structure is what the rest of
the RNA world reads, so predictions travel without any of the code here.

The core needs only numpy; plotting also uses matplotlib and ViennaRNA.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RnaStructure",
    "DotBracket",
    "call_structure",
    "call_structures",
    "nucleotide_regions",
    "call_base_pairs",
    "pair_candidates",
    "match_candidates",
    "pairs_to_dot_bracket",
    "dot_bracket_to_pairs",
    "assign_pair_layers",
    "plot_secondary_structure",
    "naview_layout",
]

Pair = Tuple[int, int, float]

# Colors, matching minerva.visualization.COLORS
BASE_PAIR_COLOR = "#F25560"     # coral, matches COLORS["base_pairing"]
LETTER_COLOR = "#E8232A"        # paired/unpaired nucleotide letters
PAIR_BOX_COLOR = "#FBD9D9"      # shading behind paired bases
RUNG_COLOR = "#111111"          # the bar joining a pair
BACKBONE_COLOR = "#333333"
PSEUDOKNOT_COLOR = "#0072B2"    # crossing pairs: connectors and their bases

# Page 0 is the nested structure, 1+ are pseudoknots. Three is the default:
# it covers every depth seen in Rfam and is what most tools parse.
BRACKET_PAGES = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]

DEFAULT_THRESHOLD = 0.6
DEFAULT_MIN_SEPARATION = 3

_WC_OR_WOBBLE = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# Pair calling
def call_base_pairs(matrix,
                    threshold: float = DEFAULT_THRESHOLD,
                    min_separation: int = DEFAULT_MIN_SEPARATION) -> List[Pair]:
    """Call base pairs from an (L, L) contact map.

    Each base proposes its best partner; proposals are accepted best-score-first
    so nothing ends up with two partners. Returns ``(i, j, score)``, ``i < j``.
    """
    candidates = pair_candidates(matrix, min_separation=min_separation)
    return match_candidates(candidates, threshold=threshold)


def pair_candidates(matrix, min_separation: int = DEFAULT_MIN_SEPARATION) -> List[Pair]:
    """Each base's best partner, unthresholded.

    A base's argmax doesn't depend on the threshold, only whether it survives
    one -- so keeping these (at most L) allows re-thresholding without the map.
    """
    scores = _as_numpy(matrix).astype(float)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"Expected a square (L, L) matrix, got {scores.shape}")

    scores = np.maximum(scores, scores.T).copy()
    n = scores.shape[0]
    bad = np.eye(n, dtype=bool)
    if min_separation > 0:
        bad |= np.abs(np.subtract.outer(np.arange(n), np.arange(n))) < min_separation
    scores[bad] = -np.inf

    best: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        j = int(np.argmax(scores[i]))
        score = float(scores[i, j])
        if not np.isfinite(score):
            continue
        a, b = (i, j) if i < j else (j, i)
        if score > best.get((a, b), -np.inf):
            best[(a, b)] = score
    return sorted((i, j, sc) for (i, j), sc in best.items())


def match_candidates(candidates: Sequence[Pair],
                     threshold: float = DEFAULT_THRESHOLD) -> List[Pair]:
    """Accept candidate pairs above `threshold`, best score first, one partner each."""
    used = set()
    pairs: List[Pair] = []
    for i, j, score in sorted((c for c in candidates if c[2] >= threshold),
                              key=lambda c: c[2], reverse=True):
        if i in used or j in used:
            continue
        used.update((i, j))
        pairs.append((i, j, score))
    return sorted(pairs)


# Pseudoknot layering
def _crosses(a, b) -> bool:
    i, j = a[0], a[1]
    k, l = b[0], b[1]
    return (i < k < j < l) or (k < i < l < j)


def _compatible(pair: Pair, layer: List[Pair]) -> bool:
    i, j = pair[0], pair[1]
    for other in layer:
        if i in (other[0], other[1]) or j in (other[0], other[1]):
            return False
        if _crosses(pair, other):
            return False
    return True


def _group_stems(pairs: List[Pair]) -> List[List[Pair]]:
    """Group pairs into stems -- runs of stacked pairs (i+1, j-1)."""
    stems: List[List[Pair]] = []
    current: List[Pair] = []
    for pair in sorted(pairs, key=lambda p: (p[0], p[1])):
        if current and pair[0] == current[-1][0] + 1 and pair[1] == current[-1][1] - 1:
            current.append(pair)
        else:
            if current:
                stems.append(current)
            current = [pair]
    if current:
        stems.append(current)
    return stems


def _stem_sort_key(stem: List[Pair]):
    mean_score = sum(p[2] for p in stem) / len(stem)
    span = max(p[1] for p in stem) - min(p[0] for p in stem)
    return (-len(stem), -mean_score, -span, stem[0][0], stem[0][1])


def assign_pair_layers(pairs: Sequence[Pair]) -> List[List[Pair]]:
    """Split pairs into non-crossing layers, keeping every pair exactly once.

    Stems move as a unit so a helix never straddles two bracket pages. When two
    stems compete, the longer one takes layer 0 and the other becomes the
    pseudoknot -- the usual reading. Layer 0 is nested, 1+ are pseudoknot pages.
    """
    unique: Dict[Tuple[int, int], float] = {}
    for pair in pairs:
        if len(pair) < 2:
            continue
        i, j = int(pair[0]), int(pair[1])
        if i == j:
            continue
        if i > j:
            i, j = j, i
        score = float(pair[2]) if len(pair) > 2 else 1.0
        if score > unique.get((i, j), -np.inf):
            unique[(i, j)] = score

    stems = _group_stems([(i, j, s) for (i, j), s in unique.items()])
    layers: List[List[Pair]] = []
    for stem in sorted(stems, key=_stem_sort_key):
        for layer in layers:
            if all(_compatible(p, layer) for p in stem):
                layer.extend(stem)
                break
        else:
            layers.append(list(stem))
    return [sorted(layer, key=lambda p: (p[0], p[1])) for layer in layers]


@dataclass
class DotBracket:
    """Extended dot-bracket notation plus what did and did not fit into it."""
    structure: str
    nested_pairs: List[Pair] = field(default_factory=list)
    pseudoknot_pairs: List[Pair] = field(default_factory=list)
    dropped_pairs: List[Pair] = field(default_factory=list)

    def __str__(self) -> str:
        return self.structure


def pairs_to_dot_bracket(pairs: Sequence[Pair], n: int, max_pages: int = 3) -> DotBracket:
    """Render 0-based pairs as extended dot-bracket notation.

    Nested pairs get ``()``, each further crossing layer the next page
    (``[]``, ``{}``, ``<>``). ``max_pages=1`` forces strictly nested output for
    tools that only parse ``()``. Anything that won't fit comes back in
    ``dropped_pairs`` rather than vanishing; ``to_ct`` has no page limit.
    """
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    max_pages = min(max_pages, len(BRACKET_PAGES))

    layers = assign_pair_layers(pairs)
    chars = ["."] * int(n)
    nested: List[Pair] = []
    pseudo: List[Pair] = []
    dropped: List[Pair] = []

    for page, layer in enumerate(layers):
        if page >= max_pages:
            dropped.extend(layer)
            continue
        open_ch, close_ch = BRACKET_PAGES[page]
        for pair in layer:
            i, j = pair[0], pair[1]
            if not (0 <= i < j < n):
                raise ValueError(f"Pair {(i, j)} is outside sequence length {n}")
            chars[i] = open_ch
            chars[j] = close_ch
            (nested if page == 0 else pseudo).append(pair)

    return DotBracket("".join(chars), sorted(nested), sorted(pseudo), sorted(dropped))


# The structure object
@dataclass
class RnaStructure:
    """A called secondary structure, and everything you can get out of it.

    ``pairs`` holds every called pair as ``(i, j, score)``, 0-based with
    ``i < j``. ``nested_pairs`` and ``pseudoknot_pairs`` are the ``()`` and
    ``[]{}`` subsets of that; ``dropped_pairs`` are too knotted for the notation
    and so are missing from ``dot_bracket`` but still present in ``pairs`` and
    ``to_ct()``. ``candidates`` is what :meth:`at_threshold` re-calls from, and
    ``offset`` shifts positions in exports when this is a slice of a genome.
    """
    sequence: str
    pairs: List[Pair] = field(default_factory=list)
    dot_bracket: str = ""
    nested_pairs: List[Pair] = field(default_factory=list)
    pseudoknot_pairs: List[Pair] = field(default_factory=list)
    dropped_pairs: List[Pair] = field(default_factory=list)
    offset: int = 0
    name: str = "minerva"
    candidates: List[Pair] = field(default_factory=list)
    threshold: float = DEFAULT_THRESHOLD
    min_separation: int = DEFAULT_MIN_SEPARATION

    def __len__(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:
        return (f"RnaStructure(name={self.name!r}, length={len(self.sequence)}, "
                f"pairs={len(self.pairs)}, pseudoknot={len(self.pseudoknot_pairs)}, "
                f"dropped={len(self.dropped_pairs)})")

    @property
    def partners(self) -> np.ndarray:
        """Partner index per base, ``-1`` where unpaired.

        Built from every called pair, so knots missing from ``dot_bracket`` are
        still here.
        """
        out = np.full(len(self.sequence), -1, dtype=int)
        for i, j, _ in self.pairs:
            out[i] = j
            out[j] = i
        return out

    def canonical_fraction(self) -> float:
        """Fraction of pairs that are Watson-Crick or G-U wobble.

        Nothing is filtered on this; it's a sanity read on a prediction.
        """
        if not self.pairs:
            return 0.0
        seq = self.sequence.upper().replace("T", "U")
        ok = sum(1 for i, j, _ in self.pairs if (seq[i], seq[j]) in _WC_OR_WOBBLE)
        return ok / len(self.pairs)

    # -- exports ------------------------------------------------------------
    def to_vienna(self, path: Optional[str] = None) -> str:
        """Vienna format: ``>name``, sequence, structure.

        Read by RNAfold, forna, VARNA and R2R. Can't express ``dropped_pairs``
        -- use :meth:`to_ct` if you need those.
        """
        text = f">{self.name}\n{self.sequence}\n{self.dot_bracket}\n"
        if path:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    def to_ct(self, path: Optional[str] = None) -> str:
        """Connect-table format, as used by RNAstructure and mfold.

        One row per base, so pseudoknots of any depth survive. This is the
        lossless export.
        """
        partner = self.partners
        n = len(self.sequence)
        lines = [f"{n}\t{self.name}"]
        for k in range(n):
            j = int(partner[k])
            lines.append("\t".join(str(v) for v in (
                k + 1,                                  # index, 1-based
                self.sequence[k],                       # base
                k,                                      # previous
                k + 2 if k + 1 < n else 0,              # next
                j + 1 if j >= 0 else 0,                 # partner, 0 = unpaired
                k + 1 + self.offset,                    # natural numbering
            )))
        text = "\n".join(lines) + "\n"
        if path:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    def at_threshold(self, threshold: float, *, max_pages: int = 3) -> "RnaStructure":
        """Re-call the same sequence at a different threshold.

        Works off the stored ``candidates``, so the contact map isn't needed.
        """
        if not self.candidates:
            raise ValueError(
                "No candidates stored; re-call with call_structure() to enable "
                "re-thresholding."
            )
        pairs = match_candidates(self.candidates, threshold=threshold)
        db = pairs_to_dot_bracket(pairs, len(self.sequence), max_pages=max_pages)
        return RnaStructure(
            sequence=self.sequence, pairs=pairs, dot_bracket=db.structure,
            nested_pairs=db.nested_pairs, pseudoknot_pairs=db.pseudoknot_pairs,
            dropped_pairs=db.dropped_pairs, offset=self.offset, name=self.name,
            candidates=self.candidates, threshold=float(threshold),
            min_separation=self.min_separation,
        )

    # -- rendering ----------------------------------------------------------
    def plot(self, **kwargs):
        """Static 2D structure drawing. See :func:`plot_secondary_structure`."""
        return plot_secondary_structure(self, **kwargs)


def call_structure(matrix,
                   sequence: Optional[str] = None,
                   *,
                   tokens: Optional[List[str]] = None,
                   threshold: float = DEFAULT_THRESHOLD,
                   min_separation: int = DEFAULT_MIN_SEPARATION,
                   max_pages: int = 3,
                   rna: bool = True,
                   offset: int = 0,
                   name: str = "minerva") -> RnaStructure:
    """Call a secondary structure from a Minerva base-pairing map.

    ``matrix`` is the (L, L) map, or (1, L, L) -- a unit batch axis is squeezed.
    Give it either a ``sequence`` or the model's ``tokens``; with tokens the map
    is cropped to nucleotide positions, dropping ``<+>`` and any amino-acid
    stretch, and the sequence is read off them.

    ``offset`` is the position of base 0 in whatever coordinates you want the
    exports numbered in; it is added to the natural-numbering column of
    :meth:`RnaStructure.to_ct` and nothing else.
    """
    arr = _as_numpy(matrix)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected a single map, got a batch of {arr.shape[0]}; "
                "index it first, e.g. interactions['base_pairing'][0]"
            )
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected a square (L, L) matrix, got {arr.shape}")

    if tokens is not None:
        regions = nucleotide_regions(tokens)
        if len(tokens) != arr.shape[0]:
            raise ValueError(
                f"tokens has length {len(tokens)} but the map is {arr.shape[0]}x{arr.shape[0]}"
            )
        if not regions:
            raise ValueError("No nucleotide tokens found; is this a protein-only locus?")
        if len(regions) > 1:
            spans = ", ".join(f"{a}-{b - 1}" for a, b in regions[:4])
            raise ValueError(
                f"tokens contain {len(regions)} separate nucleotide regions ({spans}"
                f"{', ...' if len(regions) > 4 else ''}). They are different "
                "molecules, so folding them as one sequence would invent pairs "
                "across the gap. Use call_structures() for one structure per "
                "region, or pass a single region's map and sequence."
            )
        start, stop = regions[0]
        arr = arr[start:stop, start:stop]
        if sequence is None:
            sequence = "".join(tokens[start:stop])
        offset = offset + start

    if sequence is None:
        raise ValueError("Provide either `sequence` or `tokens`")

    sequence = "".join(sequence.split()).upper()
    if rna:
        sequence = sequence.replace("T", "U")
    if len(sequence) != arr.shape[0]:
        raise ValueError(
            f"sequence has length {len(sequence)} but the map is "
            f"{arr.shape[0]}x{arr.shape[0]}; crop them to the same region "
            "(or pass `tokens` to crop automatically)"
        )

    candidates = pair_candidates(arr, min_separation=min_separation)
    pairs = match_candidates(candidates, threshold=threshold)
    db = pairs_to_dot_bracket(pairs, len(sequence), max_pages=max_pages)
    return RnaStructure(
        sequence=sequence,
        pairs=pairs,
        dot_bracket=db.structure,
        nested_pairs=db.nested_pairs,
        pseudoknot_pairs=db.pseudoknot_pairs,
        dropped_pairs=db.dropped_pairs,
        offset=offset,
        name=name,
        candidates=candidates,
        threshold=float(threshold),
        min_separation=int(min_separation),
    )


# Splitting a mixed locus
def nucleotide_regions(tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """Contiguous runs of nucleotide tokens, as half-open ``(start, stop)`` spans.

    A mixed-modality locus interleaves DNA with amino-acid stretches, so its
    nucleotides come in separate chunks. Each chunk is its own molecule.
    """
    from .visualization import token_types

    spans: List[Tuple[int, int]] = []
    start = None
    for k, kind in enumerate(token_types(list(tokens))):
        if kind == "nucleotide":
            if start is None:
                start = k
        elif start is not None:
            spans.append((start, k))
            start = None
    if start is not None:
        spans.append((start, len(tokens)))
    return spans


def call_structures(matrix,
                    tokens: Sequence[str],
                    *,
                    min_length: int = 1,
                    offset: int = 0,
                    name: str = "minerva",
                    **kwargs) -> List[RnaStructure]:
    """One structure per contiguous nucleotide region of a mixed locus.

    Intergenic stretches separated by a gene are separate molecules, so each is
    called on its own sub-map with its own ``offset``. Pairs the model predicts
    *between* regions are not secondary structure of any one molecule and are
    left in the contact map rather than forced into a structure here.

    Each region's ``offset`` is its start in TOKEN space (plus ``offset``), not
    a genome coordinate -- in a mixed locus one amino-acid token spans three
    bases. For real genome numbering in exports, work the region starts through
    :func:`minerva.sequence_utils.build_token_to_genome_map` and pass the result
    as ``offset``. ``min_length`` skips regions too short to bother folding.
    """
    arr = _as_numpy(matrix)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    regions = nucleotide_regions(tokens)
    out = []
    for n, (start, stop) in enumerate(regions, 1):
        if stop - start < max(min_length, 1):
            continue
        out.append(call_structure(
            arr[start:stop, start:stop],
            "".join(tokens[start:stop]),
            offset=offset + start,
            name=f"{name} region {n}" if len(regions) > 1 else name,
            **kwargs,
        ))
    return out


# Reading structures back in
def dot_bracket_to_pairs(structure: str) -> List[Pair]:
    """Parse extended dot-bracket into ``(i, j, 1.0)`` pairs.

    Understands every page in :data:`BRACKET_PAGES`, so pseudoknotted notation
    and Rfam ``SS_cons`` lines both work.
    """
    openers = {o: k for k, (o, _) in enumerate(BRACKET_PAGES)}
    closers = {c: k for k, (_, c) in enumerate(BRACKET_PAGES)}
    stacks: Dict[int, List[int]] = {k: [] for k in range(len(BRACKET_PAGES))}
    pairs: List[Pair] = []
    for k, ch in enumerate(structure):
        if ch in openers:
            stacks[openers[ch]].append(k)
        elif ch in closers:
            page = closers[ch]
            if not stacks[page]:
                raise ValueError(f"Unbalanced '{ch}' at position {k}")
            pairs.append((stacks[page].pop(), k, 1.0))
    unclosed = [k for stack in stacks.values() for k in stack]
    if unclosed:
        raise ValueError(f"Unclosed bracket(s) at position(s) {sorted(unclosed)}")
    return sorted(pairs)


# 2D layout (ViennaRNA supplies geometry only)

_VIENNA_MISSING = (
    "ViennaRNA is needed to lay out structures: pip install ViennaRNA\n"
    "It supplies geometry only -- what gets drawn is always Minerva's structure."
)


def _nested_dot_bracket(structure: "RnaStructure") -> str:
    """The nested layer alone, as a ``()``-only string.

    naview is planar, so crossing pairs are drawn afterwards as connectors.
    """
    chars = ["."] * len(structure.sequence)
    for pair in structure.nested_pairs:
        chars[int(pair[0])] = "("
        chars[int(pair[1])] = ")"
    return "".join(chars)


def naview_layout(structure: "RnaStructure"):
    """Coordinates for a structure, via ViennaRNA's naview layout.

    Only ``nested_pairs`` shapes the geometry -- the sequence is never folded.
    Returns ``(xs, ys)`` with y up, scaled to a median backbone step of 1.0.
    """
    try:
        import RNA
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(_VIENNA_MISSING) from exc

    db = _nested_dot_bracket(structure)
    coords = RNA.naview_xy_coordinates(db)
    p = np.array([(coords[i].X, -coords[i].Y) for i in range(len(db))], dtype=float)
    if len(p) > 1:
        step = float(np.median(np.hypot(*np.diff(p, axis=0).T)))
        if np.isfinite(step) and step > 0:
            p /= step
    return p[:, 0].copy(), p[:, 1].copy()


# Static rendering
def _bezier(p0, ctrl, p1, steps: int = 48) -> np.ndarray:
    t = np.linspace(0, 1, steps)[:, None]
    return ((1 - t) ** 2 * np.asarray(p0, dtype=float)
            + 2 * (1 - t) * t * np.asarray(ctrl, dtype=float)
            + t ** 2 * np.asarray(p1, dtype=float))


def plot_secondary_structure(structure: "RnaStructure",
                             *,
                             ax=None,
                             figsize: Tuple[float, float] = (7.0, 7.0),
                             letter_color: str = LETTER_COLOR,
                             box_color: str = PAIR_BOX_COLOR,
                             rung_color: str = RUNG_COLOR,
                             backbone_color: str = BACKBONE_COLOR,
                             pseudoknot_color: str = PSEUDOKNOT_COLOR,
                             show_letters: Optional[bool] = None,
                             show_boxes: bool = True,
                             letter_size: Optional[float] = None,
                             title: Optional[str] = None):
    """Draw the structure as a 2D diagram, returning the Figure.

    Paired bases sit in shaded boxes joined by rungs; crossing pairs, which no
    planar layout can draw as rungs, become curved connectors. Pass ``ax`` to
    draw into an existing axes. ``show_letters`` defaults to on until the
    structure is big enough that the glyphs would collide.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    seq = structure.sequence
    n = len(seq)
    if n == 0:
        raise ValueError("Cannot draw an empty structure")

    xs, ys = naview_layout(structure)
    p = np.column_stack([xs, ys])

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    padx = max((xs.max() - xs.min()) * 0.05, 1.0)
    pady = max((ys.max() - ys.min()) * 0.05, 1.0)
    xlim = (xs.min() - padx, xs.max() + padx)
    ylim = (ys.min() - pady, ys.max() + pady)
    pos = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    pts_per_unit = min(fig_w * 72.0 * pos.width / max(xlim[1] - xlim[0], 1e-6),
                       fig_h * 72.0 * pos.height / max(ylim[1] - ylim[0], 1e-6))
    fs = float(letter_size) if letter_size is not None else float(
        np.clip(pts_per_unit * 0.62, 3.0, 11.0))
    if show_letters is None:
        show_letters = fs >= 4.5

    # Line weights are in points, so they have to track how big a nucleotide
    # actually is on the page or they blob together on a large structure.
    rung_lw = float(np.clip(pts_per_unit * 0.22, 0.5, 2.8))
    bb_lw = float(np.clip(pts_per_unit * 0.075, 0.3, 1.0))
    pk_lw = float(np.clip(pts_per_unit * 0.11, 0.4, 1.4))

    crossing = list(structure.pseudoknot_pairs) + list(structure.dropped_pairs)
    pk_bases = {int(k) for pair in crossing for k in pair[:2]}
    paired = {int(k) for pair in structure.pairs for k in pair[:2]}

    ax.plot(xs, ys, "-", color=backbone_color, linewidth=bb_lw, zorder=2,
            solid_capstyle="round")

    for pair in structure.nested_pairs:                     # bold rungs
        i, j = int(pair[0]), int(pair[1])
        d = p[j] - p[i]
        norm = max(float(np.hypot(*d)), 1e-6)
        u = d / norm
        t = min(0.34, norm / 2.6)
        ax.plot([p[i, 0] + u[0] * t, p[j, 0] - u[0] * t],
                [p[i, 1] + u[1] * t, p[j, 1] - u[1] * t],
                "-", color=rung_color, linewidth=rung_lw, zorder=3,
                solid_capstyle="butt")

    for pair in crossing:                                   # curved connectors
        i, j = int(pair[0]), int(pair[1])
        mid = (p[i] + p[j]) / 2.0
        d = p[j] - p[i]
        ctrl = (mid[0] - d[1] * 0.18, mid[1] + d[0] * 0.18)
        pts = _bezier(p[i], ctrl, p[j], steps=40)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=pseudoknot_color,
                linewidth=pk_lw, alpha=0.9, zorder=4)

    if show_boxes:
        for k in sorted(paired):
            ax.add_patch(FancyBboxPatch(
                (p[k, 0] - 0.30, p[k, 1] - 0.30), 0.60, 0.60,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                fc=box_color, ec="none", zorder=5))

    if show_letters:
        for k, ch in enumerate(seq):
            ax.text(p[k, 0], p[k, 1], ch, ha="center", va="center", fontsize=fs,
                    color=pseudoknot_color if k in pk_bases else letter_color,
                    zorder=6, family="serif", weight="bold")

    ax.annotate("5'", p[0], textcoords="offset points", xytext=(-13, -7),
                fontsize=max(fs, 10), color="#333333")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if title is None:
        title = f"{structure.name}  ({n} nt, {len(structure.pairs)} pairs)"
    if title:
        ax.set_title(title, pad=8)
    return fig
