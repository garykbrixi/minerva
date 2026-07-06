"""Visualization utilities for Minerva multimodal contact fingerprints.

The public plotting path accepts named fingerprint results directly:

    fp = model.get_fingerprints(seq, tokenizer)
    plot_fingerprints(fp)

For notebooks that compose their own panels, use ``render_fingerprints(fp)`` to
produce an RGB image. Legacy RGB helpers remain available for compatibility.
"""

from typing import Dict, List, Optional, Union

import numpy as np


# =============================================================================
# Color scheme
# =============================================================================
COLORS = {
    "base_pairing": "#F25560",  # Coral   - RFAM / RNA base pairing
    "protein":      "#2ED5C5",  # Teal    - PDB protein contacts
    "repeat":       "#4F46E5",  # Deep indigo - DNA repeats
    "other":        "#D1D5DB",  # Light gray - unclassified (usually not rendered)
}

# Channels rendered in the overlay, in argmax-stack order.
OVERLAY_CHANNELS = ["base_pairing", "repeat", "protein"]

# RNA base-pairing wins wherever its normalized value exceeds this, so faint
# base-pairing stays visible against strong protein backgrounds.
DEFAULT_OVERLAY_OVERRIDES = {"base_pairing": 0.5}

# Raw Frobenius-norm contact strengths roughly span 0..10 after centering / APC;
# this vmax matches the UG27 / DRT2 fingerprint convention.
DEFAULT_FP_VMAX = 10.0

# Publication palette used by the Minerva UG27 figure scripts.
PUBLICATION_DATASET_COLORS = {
    "rna": (218 / 255, 56 / 255, 50 / 255),       # #DA3832
    "pdb": (125 / 255, 180 / 255, 224 / 255),     # #7DB4E0
    "repeat": (95 / 255, 48 / 255, 140 / 255),    # #5F308C
}
PUBLICATION_JACOBIAN_COLOR = (224 / 255, 159 / 255, 14 / 255)  # #E09F0E
PUBLICATION_OVERLAY_COLORS = {
    "protein": PUBLICATION_DATASET_COLORS["pdb"],
    "base_pairing": PUBLICATION_DATASET_COLORS["rna"],
    "repeat": PUBLICATION_DATASET_COLORS["repeat"],
    "jacobian": PUBLICATION_JACOBIAN_COLOR,
    "other": PUBLICATION_JACOBIAN_COLOR,
}
PUBLICATION_LINE_WIDTH = 0.8


def _hex_to_rgb(c):
    if isinstance(c, str):
        c = c.lstrip("#")
        return np.array([int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.asarray(c, dtype=float)


def _finite_channel(x, *, nan=0.0, posinf=1.0, neginf=0.0):
    """Float32 array with non-finite values replaced for stable rendering."""
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=nan, posinf=posinf, neginf=neginf)


def _finite_rgb(rgb):
    """RGB image in [0, 1], with invalid pixels rendered as white."""
    return np.clip(_finite_channel(rgb, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def contact_rgb_overlay(
    channels: Dict[str, np.ndarray],
    channel_order: Optional[List[str]] = None,
    colors: Optional[Dict[str, object]] = None,
    vmin: Union[float, Dict[str, float]] = 0.0,
    vmax: Union[float, Dict[str, float]] = 1.0,
    overrides: Optional[Dict[str, float]] = DEFAULT_OVERLAY_OVERRIDES,
) -> np.ndarray:
    """Winner-take-all RGB overlay of contact channels over white.

    1. Each channel is independently min-max rescaled to [0, 1] via
       ``clip((value - vmin) / (vmax - vmin), 0, 1)``.
    2. Per pixel, argmax picks the dominant channel (no overlap blending).
    3. The painted color is ``1 - (1 - color) * normalized`` -- a subtract-from-
       white blend, so 0 stays white and 1 is fully saturated.
    4. ``overrides`` force a channel to win wherever its normalized value exceeds
       the threshold (default: base-pairing at 0.5).

    Args:
        channels: dict channel name -> (H, W) array.
        channel_order: argmax stack order. Defaults to ``channels`` insertion order.
        colors: channel name -> hex/(R,G,B). Defaults to module ``COLORS``.
        vmin, vmax: per-channel normalization bounds (float or {channel: bound}).
        overrides: {channel: threshold} forced wins, or None for pure argmax.

    Returns:
        (H, W, 3) float array in [0, 1] for ``imshow``.
    """
    if not channels:
        raise ValueError("channels is empty")

    order = channel_order if channel_order is not None else list(channels.keys())
    order = [ch for ch in order if ch in channels]
    if not order:
        raise ValueError("no channels in `channel_order` matched `channels`")

    palette = colors if colors is not None else COLORS

    def _resolve(arg, ch, default):
        if isinstance(arg, dict):
            return float(arg.get(ch, default))
        return float(arg) if arg is not None else default

    first = channels[order[0]]
    h, w = first.shape

    stack = np.empty((len(order), h, w), dtype=np.float32)
    for i, ch in enumerate(order):
        lo = _resolve(vmin, ch, 0.0)
        hi = _resolve(vmax, ch, 1.0)
        denom = max(hi - lo, 1e-12)
        values = _finite_channel(channels[ch], nan=lo, posinf=hi, neginf=lo)
        stack[i] = np.clip((values - lo) / denom, 0.0, 1.0)

    max_idx = np.argmax(stack, axis=0)
    max_val = np.max(stack, axis=0)

    if overrides:
        for ch, thr in overrides.items():
            if ch not in order:
                continue
            ch_idx = order.index(ch)
            mask = stack[ch_idx] >= thr
            max_idx[mask] = ch_idx
            max_val[mask] = stack[ch_idx][mask]

    palette_rgb = np.array([_hex_to_rgb(palette[ch]) for ch in order], dtype=np.float32)
    target_color = palette_rgb[max_idx]                    # (H, W, 3)
    rgb = 1.0 - (1.0 - target_color) * max_val[..., np.newaxis]
    return rgb


def _to_numpy_array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _overlay_channel_name(name: str) -> str:
    aliases = {
        "basepairing": "base_pairing",
        "base_pair": "base_pairing",
        "bp": "base_pairing",
    }
    key = str(name).lower()
    return aliases.get(key, key)


def _default_fingerprint_channel_names(count: int) -> List[str]:
    if count == 3:
        return ["basepairing", "repeat", "other"]
    if count == 4:
        return ["basepairing", "repeat", "protein", "other"]
    if count == 5:
        return ["bp_forward", "bp_reverse", "repeat", "protein", "other"]
    return [f"channel_{i}" for i in range(count)]


def _fingerprint_channels(fingerprints, channel_names: Optional[List[str]] = None):
    if hasattr(fingerprints, "channels"):
        channels = dict(fingerprints.channels)
        names = channel_names or list(getattr(fingerprints, "channel_names", channels.keys()))
    elif isinstance(fingerprints, dict):
        channels = dict(fingerprints)
        names = channel_names or list(channels.keys())
    elif isinstance(fingerprints, tuple) and len(fingerprints) == 2:
        arr, names = fingerprints
        names = list(channel_names or names)
        arr = _to_numpy_array(arr)
        channels = {name: arr[i] for i, name in enumerate(names)}
    else:
        arr = _to_numpy_array(fingerprints)
        if arr.ndim != 3:
            raise ValueError(
                "fingerprints must be a FingerprintResult, dict, or (C, L, L) array"
            )
        names = list(channel_names or _default_fingerprint_channel_names(arr.shape[0]))
        if len(names) != arr.shape[0]:
            raise ValueError(
                "channel_names length must match fingerprints channel dimension: "
                f"{len(names)} != {arr.shape[0]}"
            )
        channels = {name: arr[i] for i, name in enumerate(names)}

    normalized = {}
    for name, values in channels.items():
        normalized[_overlay_channel_name(name)] = _to_numpy_array(values)

    if "base_pairing" not in normalized:
        bp_parts = [normalized[ch] for ch in ("bp_forward", "bp_reverse") if ch in normalized]
        if bp_parts:
            normalized["base_pairing"] = np.sum(np.stack(bp_parts, axis=0), axis=0)
    return normalized


def render_fingerprints(
    fingerprints,
    channel_names: Optional[List[str]] = None,
    *,
    style: str = "default",
    include_other: Optional[bool] = None,
    colors: Optional[Dict[str, object]] = None,
    vmin: Union[float, Dict[str, float]] = 0.0,
    vmax: Union[float, Dict[str, float]] = DEFAULT_FP_VMAX,
    overrides: Optional[Dict[str, float]] = DEFAULT_OVERLAY_OVERRIDES,
) -> np.ndarray:
    """Render named fingerprint channels as a winner-take-all RGB overlay.

    Args:
        fingerprints: ``FingerprintResult``, ``{channel: map}``, or a channel-first
            ``(C, L, L)`` tensor/array.
        channel_names: Required only when passing a raw array with non-default
            channel order.
        style: ``"default"`` or ``"publication"``. Publication style uses the
            Minerva figure palette and includes the ``other`` channel by default.
        include_other: Whether to include the unclassified channel in the overlay.
            Defaults to True only for ``style="publication"``.
        colors, vmin, vmax, overrides: Forwarded to :func:`contact_rgb_overlay`.

    Returns:
        ``(L, L, 3)`` float RGB image in ``[0, 1]``.
    """
    if style not in ("default", "publication"):
        raise ValueError("style must be 'default' or 'publication'")
    if include_other is None:
        include_other = style == "publication"
    if style == "publication" and colors is None:
        colors = PUBLICATION_OVERLAY_COLORS

    by_name = _fingerprint_channels(fingerprints, channel_names=channel_names)
    order = [ch for ch in OVERLAY_CHANNELS if ch in by_name]
    if include_other and "other" in by_name:
        order.append("other")
    if not order:
        raise ValueError(
            f"no renderable fingerprint channels; got {list(by_name)}, "
            f"expected some of {OVERLAY_CHANNELS}"
        )
    return contact_rgb_overlay(
        {ch: by_name[ch] for ch in order},
        channel_order=order,
        colors=colors,
        vmin=vmin,
        vmax=vmax,
        overrides=overrides,
    )


def jacobian_fingerprint_rgb(
    jac,
    tokens: List[str],
    *,
    bp_threshold: Optional[float] = None,
    repeat_threshold: Optional[float] = None,
    protein_threshold: Optional[float] = None,
    aa_start: int = 4,
    jac_aa_order: Optional[List[str]] = None,
    colors: Optional[Dict[str, object]] = None,
    vmin: Union[float, Dict[str, float]] = 0.0,
    vmax: Union[float, Dict[str, float]] = DEFAULT_FP_VMAX,
    overrides: Optional[Dict[str, float]] = DEFAULT_OVERLAY_OVERRIDES,
    include_other: bool = False,
):
    """One-shot Jacobian -> multimodal fingerprint -> RGB overlay.

    Wraps :func:`minerva.jacobian.fingerprint_jacobian_multimodality` (classifies
    each position-pair into base-pairing / repeat / protein / other via cosine
    similarity to the reference fingerprints) and :func:`contact_rgb_overlay`. The
    unclassified "other" channel is dropped from the overlay so only the three
    modality colors appear over white.

    Args:
        jac: Jacobian (L, A, L, A), numpy or torch, full alphabet channels
            (i.e. ``fast=False`` mode of ``get_categorical_jacobian``).
        tokens: L tokens aligned to Jacobian positions.
        bp_threshold, repeat_threshold, protein_threshold: optional similarity
            cutoffs; None -> Minerva defaults.
        aa_start, jac_aa_order: forwarded to the classifier.
        colors, vmin, vmax, overrides: forwarded to :func:`contact_rgb_overlay`.
        include_other: include the classifier's unassigned "other" channel.
            This is the gold Jacobian background used in the Minerva UG27
            publication fingerprint plots.

    Returns:
        (rgb, fingerprints, channel_names): rgb is (L, L, 3) in [0, 1];
        fingerprints is (C, L, L); channel_names lists the classifier channels.
    """
    try:
        from .jacobian import fingerprint_jacobian_multimodality
    except ImportError:  # HF snapshot imported as top-level visualization.py
        from jacobian import fingerprint_jacobian_multimodality

    fp_kwargs = {"split_bp": False, "aa_start": aa_start, "jac_aa_order": jac_aa_order}
    if bp_threshold is not None:
        fp_kwargs["bp_threshold"] = bp_threshold
    if repeat_threshold is not None:
        fp_kwargs["repeat_threshold"] = repeat_threshold
    if protein_threshold is not None:
        fp_kwargs["protein_threshold"] = protein_threshold

    _, fingerprints, channel_names = fingerprint_jacobian_multimodality(
        jac, tokens, **fp_kwargs,
    )
    # Classifier returns 'basepairing'; the palette key is 'base_pairing'.
    aliases = {"basepairing": "base_pairing"}
    norm = [aliases.get(n, n) for n in channel_names]
    by_name = {n: fingerprints[i] for i, n in enumerate(norm)}

    # Render only the three colored modality channels by default. Publication
    # fingerprint plots add "other" as a gold Jacobian-strength background.
    order = [ch for ch in OVERLAY_CHANNELS if ch in by_name]
    if include_other and "other" in by_name:
        order.append("other")
    rgb = contact_rgb_overlay(
        {ch: by_name[ch] for ch in order},
        channel_order=order,
        colors=colors,
        vmin=vmin,
        vmax=vmax,
        overrides=overrides,
    )
    return rgb, fingerprints, channel_names


def head_contacts_rgb(
    contacts: Dict[str, np.ndarray],
    tokens: Optional[List[str]] = None,
    colors: Optional[Dict[str, object]] = None,
    vmin: Union[float, Dict[str, float]] = 0.0,
    vmax: Union[float, Dict[str, float]] = 1.0,
    overrides: Optional[Dict[str, float]] = DEFAULT_OVERLAY_OVERRIDES,
) -> np.ndarray:
    """RGB overlay from regression-head contact maps.

    Args:
        contacts: dict channel name (``base_pairing`` / ``repeat`` / ``protein``)
            -> (L, L) contact map. Head-derived maps are sigmoid probabilities in
            [0, 1], so ``vmax=1.0`` is the natural scale.
        tokens: optional length-L token list. If given, applies token-aware
            masking (matching the Jacobian fingerprint convention): the protein
            channel is kept only on amino-acid/amino-acid pairs, and the
            base-pairing / repeat channels only on nucleotide/nucleotide pairs.
            This removes spurious protein signal on RNA/intergenic regions.
        colors, vmin, vmax, overrides: forwarded to :func:`contact_rgb_overlay`.

    Returns:
        (L, L, 3) float RGB array in [0, 1].
    """
    order = [ch for ch in OVERLAY_CHANNELS if ch in contacts]
    if not order:
        raise ValueError(
            f"no overlay channels in contacts; got keys {list(contacts)}, "
            f"expected some of {OVERLAY_CHANNELS}"
        )
    chan = {ch: _finite_channel(contacts[ch]) for ch in order}
    if tokens is not None and len(tokens) == chan[order[0]].shape[0]:
        tt = np.array(token_types(tokens))
        aa_ij = (tt == "protein")[:, None] & (tt == "protein")[None, :]
        nuc_ij = (tt == "nucleotide")[:, None] & (tt == "nucleotide")[None, :]
        for ch in order:
            mask = aa_ij if ch == "protein" else nuc_ij
            chan[ch] = np.where(mask, chan[ch], 0.0).astype(np.float32, copy=False)
    return contact_rgb_overlay(
        chan, channel_order=order, colors=colors, vmin=vmin, vmax=vmax,
        overrides=overrides,
    )


def _auto_contrast_contact_channels(
    channels: Dict[str, np.ndarray],
    *,
    percentile: float = 99.0,
    min_dynamic: float = 1e-8,
):
    """Scale structured low-probability contact maps without coloring constants."""
    scaled = {}
    vmax = {}
    stats = {}
    for ch, arr in channels.items():
        values = _finite_channel(arr)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            scaled[ch] = np.zeros_like(values, dtype=np.float32)
            vmax[ch] = 1.0
            stats[ch] = {"kept": False, "reason": "non-finite"}
            continue
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        dynamic = hi - lo
        if dynamic <= min_dynamic:
            scaled[ch] = np.zeros_like(values, dtype=np.float32)
            vmax[ch] = 1.0
            stats[ch] = {"kept": False, "reason": "constant", "min": lo, "max": hi}
            continue
        positive = finite[finite > 0]
        ref = positive if positive.size else finite
        p = float(np.percentile(ref, percentile))
        if not np.isfinite(p) or p <= 0:
            p = hi if hi > 0 else 1.0
        scaled[ch] = values
        vmax[ch] = max(p, min_dynamic)
        stats[ch] = {"kept": True, "min": lo, "max": hi, f"p{percentile:g}": p}
    return scaled, vmax, stats


def _mask_contact_channels_by_tokens(
    contacts: Dict[str, np.ndarray],
    tokens: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Apply the same token-aware channel mask used by head contact overlays."""
    order = [ch for ch in OVERLAY_CHANNELS if ch in contacts]
    chan = {ch: _finite_channel(contacts[ch]) for ch in order}
    if not order:
        return chan
    if tokens is not None and len(tokens) == chan[order[0]].shape[0]:
        tt = np.array(token_types(tokens))
        aa_ij = (tt == "protein")[:, None] & (tt == "protein")[None, :]
        nuc_ij = (tt == "nucleotide")[:, None] & (tt == "nucleotide")[None, :]
        for ch in order:
            mask = aa_ij if ch == "protein" else nuc_ij
            chan[ch] = np.where(mask, chan[ch], 0.0).astype(np.float32, copy=False)
    return chan


def publication_head_contacts_rgb(
    contacts: Dict[str, np.ndarray],
    tokens: Optional[List[str]] = None,
    auto_contrast: bool = False,
    contrast_percentile: float = 99.0,
    return_stats: bool = False,
    **kwargs,
) -> np.ndarray:
    """Regression-head overlay using the Minerva publication palette.

    By default this renders calibrated 0..1 contact probabilities, matching the
    Minerva publication contact-overlay convention. Set ``auto_contrast=True``
    only for exploratory viewing of very low-probability structure.
    """
    kwargs.setdefault("colors", PUBLICATION_OVERLAY_COLORS)
    rendered_contacts = _mask_contact_channels_by_tokens(contacts, tokens=tokens)
    stats = None
    if auto_contrast:
        rendered_contacts, auto_vmax, stats = _auto_contrast_contact_channels(
            rendered_contacts, percentile=contrast_percentile)
        kwargs.setdefault("vmax", auto_vmax)
    rgb = head_contacts_rgb(rendered_contacts, tokens=None, **kwargs)
    if return_stats:
        return rgb, stats
    return rgb


def publication_jacobian_fingerprint_rgb(jac, tokens: List[str], **kwargs):
    """Jacobian fingerprint overlay matching the Minerva UG27 figure style."""
    kwargs.setdefault("colors", PUBLICATION_OVERLAY_COLORS)
    kwargs.setdefault("include_other", True)
    return jacobian_fingerprint_rgb(jac, tokens, **kwargs)


def legend_handles(channels: Optional[List[str]] = None, colors: Optional[Dict] = None):
    """matplotlib Patch handles for a base-pairing / repeat / protein legend."""
    from matplotlib.patches import Patch

    channels = channels if channels is not None else OVERLAY_CHANNELS
    palette = colors if colors is not None else COLORS
    labels = {"base_pairing": "Base pairing (RNA)", "repeat": "Repeat (DNA)",
              "protein": "Protein"}
    return [Patch(facecolor=palette[ch], edgecolor="none", label=labels.get(ch, ch))
            for ch in channels]


# =============================================================================
# Locus annotation track (protein / nucleotide / special per token)
# =============================================================================
TRACK_COLORS = {
    "protein": COLORS["protein"],       # teal  -- CDS / gene (amino acids)
    "nucleotide": COLORS["repeat"],     # indigo -- intergenic / RNA (nucleotides)
    "special": "#C7CBD1",               # gray  -- orientation / special tokens
}


def token_types(tokens: List[str]) -> List[str]:
    """Classify each token as 'protein' | 'nucleotide' | 'special'.

    Amino acids are upper-case, nucleotides lower-case (a/c/g/t), and
    orientation/special tokens contain '<' or '>'.
    """
    out = []
    for t in tokens:
        if not t or "<" in t or ">" in t:
            out.append("special")
        elif t[:1].islower():
            out.append("nucleotide")
        elif t[:1].isupper():
            out.append("protein")
        else:
            out.append("special")
    return out


def _track_rgb(tokens: List[str]) -> np.ndarray:
    """(L, 3) RGB strip coloring each token by type."""
    return np.array([_hex_to_rgb(TRACK_COLORS[t]) for t in token_types(tokens)],
                    dtype=np.float32)


def plot_locus(
    rgb: np.ndarray,
    tokens: Optional[List[str]] = None,
    title: str = "Minerva locus",
    figsize=(9, 9),
    show_legend: bool = True,
    save: Optional[str] = None,
    dpi: int = 200,
):
    """Static contact overlay with an optional token-type track (top + left).

    ``tokens`` (len L, aligned to the map) draws a protein/nucleotide/special
    strip so gene vs intergenic structure is visible. ``save`` writes the figure
    (e.g. a ``.pdf``) at ``dpi``. Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    L = rgb.shape[0]
    has_track = tokens is not None and len(tokens) == L
    if not has_track:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(_finite_rgb(rgb), aspect="equal", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Position"); ax.set_ylabel("Position")
        if show_legend:
            ax.legend(handles=legend_handles(), loc="upper right", fontsize=9,
                      framealpha=0.9)
    else:
        strip = _track_rgb(tokens)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[0.035, 1], width_ratios=[0.035, 1],
                              hspace=0.015, wspace=0.015)
        ax_top = fig.add_subplot(gs[0, 1])
        ax_top.imshow(strip[None, :, :], aspect="auto", interpolation="nearest")
        ax_top.set_xticks([]); ax_top.set_yticks([])
        ax_top.set_title(title, fontsize=11)
        ax_left = fig.add_subplot(gs[1, 0])
        ax_left.imshow(strip[:, None, :], aspect="auto", interpolation="nearest")
        ax_left.set_xticks([]); ax_left.set_yticks([])
        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(_finite_rgb(rgb), aspect="equal", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if show_legend:
            ax.legend(handles=legend_handles(), loc="upper right", fontsize=8,
                      framealpha=0.9)
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def interactive_overlay(rgb: np.ndarray, title: str = "Minerva locus"):
    """Zoom/pan/hover view of the RGB overlay via Plotly. Returns a plotly Figure.

    Use in Colab/Jupyter: ``interactive_overlay(rgb).show()``, or
    ``fig.write_html("locus.html")`` for a standalone interactive file.
    """
    import plotly.express as px

    img = (_finite_rgb(rgb) * 255).astype(np.uint8)
    fig = px.imshow(img, title=title)
    fig.update_layout(dragmode="pan", margin=dict(l=10, r=10, t=40, b=10),
                      height=760, width=800)
    fig.update_xaxes(title="Position", constrain="domain")
    fig.update_yaxes(title="Position", scaleanchor="x")
    return fig


def plot_fingerprint_overlay(
    rgb: np.ndarray,
    title: str = "Minerva multimodal fingerprint",
    ax=None,
    extent=None,
    show_legend: bool = True,
    figsize=(9, 9),
    legend_channels: Optional[List[str]] = None,
    colors: Optional[Dict[str, object]] = None,
):
    """imshow an RGB overlay with a channel legend. Returns the Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(_finite_rgb(rgb), aspect="equal", interpolation="nearest",
              extent=extent, origin="upper")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    if show_legend:
        ax.legend(handles=legend_handles(legend_channels, colors=colors),
                  loc="upper right", framealpha=0.9, fontsize=9)
    return ax


def plot_fingerprints(
    fingerprints,
    channel_names: Optional[List[str]] = None,
    *,
    title: str = "Minerva multimodal fingerprint",
    style: str = "default",
    include_other: Optional[bool] = None,
    ax=None,
    extent=None,
    show_legend: bool = True,
    figsize=(9, 9),
    return_rgb: bool = False,
    **render_kwargs,
):
    """Plot a ``FingerprintResult`` or channel-first fingerprint tensor.

    Returns the matplotlib Axes by default. Set ``return_rgb=True`` to get
    ``(ax, rgb)`` for notebook panels that reuse the rendered image.
    """
    rgb = render_fingerprints(
        fingerprints,
        channel_names=channel_names,
        style=style,
        include_other=include_other,
        **render_kwargs,
    )
    colors = render_kwargs.get("colors")
    if style == "publication" and colors is None:
        colors = PUBLICATION_OVERLAY_COLORS
    legend_channels = [ch for ch in OVERLAY_CHANNELS if ch in _fingerprint_channels(fingerprints, channel_names)]
    ax = plot_fingerprint_overlay(
        rgb,
        title=title,
        ax=ax,
        extent=extent,
        show_legend=show_legend,
        figsize=figsize,
        legend_channels=legend_channels,
        colors=colors,
    )
    if return_rgb:
        return ax, rgb
    return ax


def _contact_map_2d(values, batch_index: int = 0) -> np.ndarray:
    arr = _to_numpy_array(values)
    if arr.ndim == 3:
        arr = arr[batch_index]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D contact map or batched 3D map, got shape {arr.shape}")
    return arr


def render_interactions(
    interactions,
    tokens: Optional[List[str]] = None,
    *,
    style: str = "default",
    batch_index: int = 0,
    colors: Optional[Dict[str, object]] = None,
    vmin: Union[float, Dict[str, float]] = 0.0,
    vmax: Union[float, Dict[str, float]] = 1.0,
    overrides: Optional[Dict[str, float]] = DEFAULT_OVERLAY_OVERRIDES,
) -> np.ndarray:
    """Render model interaction-head outputs as an RGB overlay."""
    if hasattr(interactions, "interactions") and interactions.interactions is not None:
        interactions = interactions.interactions
    if interactions is None:
        raise ValueError("interactions is None; call model(..., output_interactions=True)")
    if style not in ("default", "publication"):
        raise ValueError("style must be 'default' or 'publication'")

    contacts = {
        _overlay_channel_name(name): _contact_map_2d(values, batch_index=batch_index)
        for name, values in dict(interactions).items()
    }
    if style == "publication":
        return publication_head_contacts_rgb(
            contacts,
            tokens=tokens,
            colors=colors or PUBLICATION_OVERLAY_COLORS,
            vmin=vmin,
            vmax=vmax,
            overrides=overrides,
        )
    return head_contacts_rgb(
        contacts,
        tokens=tokens,
        colors=colors,
        vmin=vmin,
        vmax=vmax,
        overrides=overrides,
    )


def plot_interactions(
    interactions,
    tokens: Optional[List[str]] = None,
    *,
    title: str = "Minerva interactions",
    style: str = "default",
    batch_index: int = 0,
    ax=None,
    extent=None,
    show_legend: bool = True,
    figsize=(9, 9),
    return_rgb: bool = False,
    **render_kwargs,
):
    """Plot model interaction-head outputs from ``outputs.interactions``."""
    rgb = render_interactions(
        interactions,
        tokens=tokens,
        style=style,
        batch_index=batch_index,
        **render_kwargs,
    )
    colors = render_kwargs.get("colors")
    if style == "publication" and colors is None:
        colors = PUBLICATION_OVERLAY_COLORS
    ax = plot_fingerprint_overlay(
        rgb,
        title=title,
        ax=ax,
        extent=extent,
        show_legend=show_legend,
        figsize=figsize,
        colors=colors,
    )
    if return_rgb:
        return ax, rgb
    return ax

# =============================================================================
# Dot plots and publication-style single panels
# =============================================================================

def setup_publication_style():
    """Matplotlib rcParams matching the Minerva publication figures."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Helvetica Neue", "Arial", "Helvetica", "DejaVu Sans"
    ]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["hatch.linewidth"] = 0.5
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.size"] = 5
    plt.rcParams["ytick.major.size"] = 5
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True


def compute_dotplot_fwd_rc(seq, window: int = 6, threshold: Optional[int] = None):
    """Forward and reverse-complement self-comparison dot plot kernel.

    Both axes use forward-sequence coordinates. Forward exact k-mer matches mark
    repeat diagonals; reverse-complement matches mark potential stem/base-pairing
    anti-diagonals.
    """
    if threshold is None:
        threshold = window
    seq = seq.upper()
    n = len(seq)
    if n < window:
        empty = np.array([], dtype=int)
        return empty, empty, empty, empty, n
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)

    comp_table = np.zeros(256, dtype=np.uint8)
    for a, b in [(ord("A"), ord("T")), (ord("T"), ord("A")),
                 (ord("C"), ord("G")), (ord("G"), ord("C"))]:
        comp_table[a] = b
    arr_comp = comp_table[arr]

    fwd_r, fwd_c = [], []
    for offset in range(1, n - window + 1):
        a = arr[:n - offset]
        b = arr[offset:]
        if len(a) < window:
            continue
        matches = (a == b).astype(np.int32)
        cs = np.cumsum(matches)
        ws = np.empty(len(matches) - window + 1, dtype=np.int32)
        ws[0] = cs[window - 1]
        ws[1:] = cs[window:] - cs[:len(matches) - window]
        hits = np.where(ws >= threshold)[0]
        for h in hits:
            for k in range(window):
                fwd_r.append(h + k)
                fwd_c.append(h + offset + k)
                fwd_r.append(h + offset + k)
                fwd_c.append(h + k)

    arr_comp_rev = arr_comp[::-1].copy()
    rc_r, rc_c = [], []
    for offset in range(-(n - window), n - window + 1):
        if offset >= 0:
            a = arr[:n - offset]
            b = arr_comp_rev[offset:offset + len(a)]
        else:
            a = arr[-offset:]
            b = arr_comp_rev[:len(a)]
        if len(a) < window:
            continue
        matches = (a == b).astype(np.int32)
        cs = np.cumsum(matches)
        ws = np.empty(len(matches) - window + 1, dtype=np.int32)
        ws[0] = cs[window - 1]
        ws[1:] = cs[window:] - cs[:len(matches) - window]
        hits = np.where(ws >= threshold)[0]
        for h in hits:
            for k in range(window):
                if offset >= 0:
                    ri = h + k
                    ci = n - 1 - (h + offset + k)
                else:
                    ri = h - offset + k
                    ci = n - 1 - (h + k)
                if 0 <= ri < n and 0 <= ci < n:
                    rc_r.append(ri)
                    rc_c.append(ci)

    return (np.array(fwd_r, dtype=int), np.array(fwd_c, dtype=int),
            np.array(rc_r, dtype=int), np.array(rc_c, dtype=int), n)


def dotplot_rgb_from_tokens(
    tokens: List[str],
    word_size: int = 6,
    threshold: Optional[int] = None,
    colors: Optional[Dict[str, object]] = None,
):
    """Build a forward/RC dot plot RGB image for a mixed-token window.

    Non-nucleotide tokens remain white. Forward matches are repeat purple;
    reverse-complement matches are RNA red under the publication palette.
    """
    palette = colors if colors is not None else PUBLICATION_DATASET_COLORS
    dotplot_img = np.ones((len(tokens), len(tokens), 3), dtype=np.float32)
    nuc_subseq = ""
    nuc_sub_to_tok = {}
    for i, tok in enumerate(tokens):
        if len(tok) == 1 and tok.upper() in "ACGTN":
            nuc_sub_to_tok[len(nuc_subseq)] = i
            nuc_subseq += tok.upper()

    if len(nuc_subseq) < word_size:
        return dotplot_img, {"forward": 0, "revcomp": 0,
                             "nucleotide_tokens": len(nuc_subseq)}

    fwd_r, fwd_c, rc_r, rc_c, _ = compute_dotplot_fwd_rc(
        nuc_subseq, window=word_size, threshold=threshold)
    clr_fwd = np.array(palette["repeat"], dtype=np.float32)
    clr_rc = np.array(palette["rna"], dtype=np.float32)

    for r_nuc, c_nuc in zip(fwd_r, fwd_c):
        r_tok = nuc_sub_to_tok.get(int(r_nuc))
        c_tok = nuc_sub_to_tok.get(int(c_nuc))
        if r_tok is not None and c_tok is not None:
            dotplot_img[r_tok, c_tok] = clr_fwd

    for r_nuc, c_nuc in zip(rc_r, rc_c):
        r_tok = nuc_sub_to_tok.get(int(r_nuc))
        c_tok = nuc_sub_to_tok.get(int(c_nuc))
        if r_tok is not None and c_tok is not None:
            dotplot_img[r_tok, c_tok] = clr_rc

    return dotplot_img, {
        "forward": int(len(fwd_r)),
        "revcomp": int(len(rc_r)),
        "nucleotide_tokens": int(len(nuc_subseq)),
    }


def _publication_handles(kind: str = "heads"):
    from matplotlib.patches import Patch

    if kind == "fingerprint":
        items = [
            (PUBLICATION_JACOBIAN_COLOR, "Jacobian"),
            (PUBLICATION_DATASET_COLORS["rna"], "RNA"),
            (PUBLICATION_DATASET_COLORS["repeat"], "Repeats"),
        ]
    elif kind == "dotplot":
        items = [
            (PUBLICATION_DATASET_COLORS["repeat"], "Forward"),
            (PUBLICATION_DATASET_COLORS["rna"], "Rev. comp."),
        ]
    else:
        items = [
            (PUBLICATION_DATASET_COLORS["pdb"], "PDB"),
            (PUBLICATION_DATASET_COLORS["rna"], "RNA"),
            (PUBLICATION_DATASET_COLORS["repeat"], "Repeats"),
        ]
    return [Patch(facecolor=color, edgecolor="black", linewidth=0.4, label=label)
            for color, label in items]


def render_publication_panel(ax, rgb, label: str, show_yticks: bool = True,
                             genome_offset: int = 0):
    """Render one square publication-style image panel."""
    L = rgb.shape[0]
    ax.imshow(_finite_rgb(rgb), aspect="equal", interpolation="nearest")
    tick_step = max(1, L // 4)
    ticks = np.arange(0, L, tick_step)
    tick_labels = [f"{int(t + genome_offset):,}" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=6, rotation=0, ha="center")
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels if show_yticks else [], fontsize=6)
    ax.set_title(label, fontsize=11, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(PUBLICATION_LINE_WIDTH)
    ax.tick_params(axis="both", which="major", direction="out",
                   length=3, width=PUBLICATION_LINE_WIDTH, labelsize=6,
                   bottom=True, left=True, top=False, right=False)


def plot_dotplot(
    tokens: List[str],
    title: str = "Dot plot",
    *,
    word_size: int = 6,
    threshold: Optional[int] = None,
    ax=None,
    figsize=(5, 5),
    save: Optional[str] = None,
    dpi: int = 600,
):
    """Plot a reusable forward/RC nucleotide dot plot for any token window."""
    setup_publication_style()
    import matplotlib.pyplot as plt

    img, stats = dotplot_rgb_from_tokens(tokens, word_size=word_size,
                                         threshold=threshold)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    render_publication_panel(ax, img, title, show_yticks=True, genome_offset=0)
    leg = ax.legend(handles=_publication_handles("dotplot"), loc="upper right",
                    fontsize=8, frameon=True, framealpha=0.9,
                    edgecolor="black", fancybox=False)
    leg.get_frame().set_linewidth(0.6)
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, ax, stats


def plot_publication_locus(
    rgb: np.ndarray,
    title: str = "Minerva locus",
    *,
    genome_offset: int = 0,
    overlay_kind: str = "heads",
    show_legend: bool = True,
    ax=None,
    figsize=(5, 5),
    save: Optional[str] = None,
    dpi: int = 600,
):
    """Single-panel publication-style renderer for arbitrary loci.

    This intentionally does not add UG27-specific dot-plot, triptych, or inset
    panels. It just applies the Minerva figure styling to one RGB overlay.
    """
    setup_publication_style()
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    render_publication_panel(ax, rgb, title, show_yticks=True,
                             genome_offset=genome_offset)
    if show_legend:
        leg = ax.legend(handles=_publication_handles(overlay_kind), loc="upper right",
                        fontsize=8, frameon=True, framealpha=0.9,
                        edgecolor="black", fancybox=False)
        leg.get_frame().set_linewidth(0.6)
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, ax


# =============================================================================
# Interactive Bokeh contact viewer (zoom / pan / hover / per-channel sliders)
# =============================================================================
def _rgb01_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb[:3])


# Publication channel -> color, matching the static overlay / legend.
_BOKEH_CHANNEL_COLOR = {
    "base_pairing": PUBLICATION_DATASET_COLORS["rna"],
    "repeat":       PUBLICATION_DATASET_COLORS["repeat"],
    "protein":      PUBLICATION_DATASET_COLORS["pdb"],
}
_BOKEH_CHANNEL_LABEL = {
    "base_pairing": "RNA (base pairing)",
    "repeat":       "Repeats",
    "protein":      "PDB (protein)",
}


def bokeh_contact_viewer(
    channels: Dict[str, np.ndarray],
    tokens: Optional[List[str]] = None,
    *,
    title: str = "Minerva contacts",
    genome_offset: int = 0,
    prefilter: float = 0.05,
    init_threshold: float = 0.3,
    vmax: Union[float, Dict[str, float]] = 1.0,
    overrides: Optional[Dict[str, float]] = DEFAULT_OVERLAY_OVERRIDES,
    size: int = 760,
    track_px: int = 46,
):
    """Interactive Bokeh contact map: wheel-zoom/pan for per-pixel inspection,
    per-position token-type tracks (top + left), hover tooltips, and one
    real-time threshold slider per channel. Keeps the publication colors.

    Winner-take-all per pixel: each position-pair is assigned to a single
    channel via argmax of the vmax-normalized values (with the base-pairing
    ``overrides`` win), exactly like the static ``contact_rgb_overlay``. The
    per-channel sliders then threshold that channel's won pixels (on the
    normalized 0..1 scale).

    Args:
        channels: dict with any of ``base_pairing`` / ``repeat`` / ``protein``
            -> (L, L) contact-probability arrays.
        tokens: length-L token list; draws the protein/nucleotide/special
            per-position tracks and enables token hover. Optional.
        genome_offset: added to position labels in hover.
        prefilter: contacts below this are dropped before sending to the browser.
        init_threshold: initial slider value per channel.
        vmax: per-channel (or scalar) value mapped to full color saturation.
        size: main panel size in px. track_px: annotation track thickness.

    Returns:
        A Bokeh layout. In a notebook: ``from bokeh.io import output_notebook,
        show; output_notebook(); show(layout)``. To a file: ``save_bokeh_html``.
    """
    from bokeh.plotting import figure
    from bokeh.models import (ColumnDataSource, CustomJS, Slider, HoverTool,
                              Legend, LegendItem, Range1d, WheelZoomTool, Div)
    from bokeh.layouts import gridplot, column, row

    order = [c for c in OVERLAY_CHANNELS if c in channels]
    if not order:
        raise ValueError("channels must include one of base_pairing/repeat/protein")
    # Draw (z-order) bottom -> top: protein background, then repeat, then base
    # pairing on top -- matches the publication overlay's channel priority so
    # dense protein contacts don't bury the sparser repeat / base-pairing signal.
    draw_order = [c for c in ("protein", "repeat", "base_pairing") if c in channels]
    L = int(np.asarray(channels[order[0]]).shape[0])
    if isinstance(vmax, (int, float)):
        vmax = {c: float(vmax) for c in order}

    # Winner-take-all assignment, identical to contact_rgb_overlay: normalize
    # each channel by its vmax, argmax per pixel, then let `overrides` force a
    # channel to win where its normalized value clears the threshold.
    stack = np.stack([np.clip(_finite_channel(channels[c]) / max(vmax[c], 1e-12), 0.0, 1.0)
                      for c in order], axis=0)                     # (C, L, L)
    max_idx = np.argmax(stack, axis=0)
    max_val = np.max(stack, axis=0)
    for ch, thr in (overrides or {}).items():
        if ch in order:
            ci = order.index(ch)
            m = stack[ci] >= thr
            max_idx[m] = ci
            max_val[m] = stack[ci][m]

    def _points(channel, color, _vm=None):
        ci = order.index(channel)
        rows, cols = np.where((max_idx == ci) & (max_val > prefilter))
        vals = max_val[rows, cols]                                 # normalized 0..1
        base = np.asarray(color, dtype=float)
        blended = 1.0 - (1.0 - base)[None, :] * vals[:, None]      # subtract-from-white
        colors = [_rgb01_to_hex(c) for c in blended]
        return dict(i=cols.astype(int).tolist(), j=rows.astype(int).tolist(),
                    val=[round(float(v), 4) for v in vals], color=colors)

    x_range = Range1d(start=0, end=L, bounds=(0, L))
    y_range = Range1d(start=L, end=0, bounds=(0, L))
    p = figure(title=None, x_range=x_range, y_range=y_range,
               width=size, height=size, tools="pan,box_zoom,reset,save",
               toolbar_location="right", output_backend="webgl",
               x_axis_location="below", y_axis_location="left")
    wheel = WheelZoomTool(dimensions="both")
    p.add_tools(wheel); p.toolbar.active_scroll = wheel
    p.grid.visible = False
    p.line([0, L], [0, L], line_color="#BBBBBB", line_width=1, line_dash="dashed")
    p.xaxis.axis_label = "Position"; p.yaxis.axis_label = "Position"

    renderers, full_src, shown_src = {}, {}, {}
    for c in draw_order:   # draw bottom -> top (protein ... base_pairing)
        full = ColumnDataSource(_points(c, _BOKEH_CHANNEL_COLOR[c]))
        vals = np.asarray(full.data["val"])
        keep = vals >= init_threshold if len(vals) else np.zeros(0, bool)
        init = {k: list(np.asarray(v, dtype=object)[keep]) for k, v in full.data.items()}
        shown = ColumnDataSource(init)
        renderers[c] = p.rect(x="i", y="j", width=1.0, height=1.0, source=shown,
                              fill_color="color", line_color=None, fill_alpha=0.95)
        full_src[c], shown_src[c] = full, shown

    sliders = []
    for c in order:        # controls / legend in publication order
        slider = Slider(start=round(prefilter, 3), end=1.0, value=init_threshold, step=0.01,
                        title=_BOKEH_CHANNEL_LABEL[c], width=size // 2,
                        bar_color=_rgb01_to_hex(_BOKEH_CHANNEL_COLOR[c]))
        slider.js_on_change("value", CustomJS(args=dict(full=full_src[c], shown=shown_src[c]), code="""
            const t = cb_obj.value, f = full.data;
            const out = {i:[], j:[], val:[], color:[]};
            for (let k = 0; k < f.val.length; k++) {
                if (f.val[k] >= t) { out.i.push(f.i[k]); out.j.push(f.j[k]);
                    out.val.push(f.val[k]); out.color.push(f.color[k]); }
            }
            shown.data = out; shown.change.emit();
        """))
        sliders.append(slider)

    tips = [("position", "@i, @j"), ("value", "@val")]
    p.add_tools(HoverTool(renderers=list(renderers.values()), tooltips=tips, point_policy="follow_mouse"))

    # Legend OUTSIDE the plot box (to the right).
    legend = Legend(items=[LegendItem(label=_BOKEH_CHANNEL_LABEL[c], renderers=[renderers[c]]) for c in order],
                    location="top", border_line_color=None)
    p.add_layout(legend, "right")

    # Per-position token-type tracks (top + left), sharing the map's ranges.
    top_track = left_track = None
    if tokens is not None and len(tokens) == L:
        types = token_types(tokens)
        tcol = [_rgb01_to_hex(_hex_to_rgb(TRACK_COLORS[t])) for t in types]
        idx = list(range(L))
        top_src = ColumnDataSource(dict(left=idx, right=[i + 1 for i in idx],
                                        top=[1] * L, bottom=[0] * L, color=tcol,
                                        tok=list(tokens), typ=types))
        top_track = figure(x_range=x_range, y_range=Range1d(0, 1), width=size, height=track_px,
                           tools="", toolbar_location=None, output_backend="webgl")
        top_track.quad(left="left", right="right", top="top", bottom="bottom",
                       source=top_src, fill_color="color", line_color=None)
        top_track.add_tools(HoverTool(tooltips=[("pos", "@left"), ("token", "@tok"), ("type", "@typ")]))
        top_track.grid.visible = False; top_track.yaxis.visible = False
        top_track.xaxis.visible = False; top_track.title = title

        left_src = ColumnDataSource(dict(bottom=idx, top=[i + 1 for i in idx],
                                         left=[0] * L, right=[1] * L, color=tcol,
                                         tok=list(tokens), typ=types))
        left_track = figure(x_range=Range1d(0, 1), y_range=y_range, width=track_px, height=size,
                            tools="", toolbar_location=None, output_backend="webgl")
        left_track.quad(left="left", right="right", top="top", bottom="bottom",
                        source=left_src, fill_color="color", line_color=None)
        left_track.grid.visible = False; left_track.xaxis.visible = False; left_track.yaxis.visible = False

    if top_track is not None:
        grid = gridplot([[None, top_track], [left_track, p]], toolbar_location="right",
                        merge_tools=False)
    else:
        grid = gridplot([[p]], toolbar_location="right", merge_tools=False)
    header = Div(text=f"<b>{title}</b>", styles={"font-size": "14px", "margin": "2px 0"})
    return column(header, row(*sliders), grid)


def save_bokeh_html(layout, path: str, title: str = "Minerva contacts") -> str:
    """Save a Bokeh layout to a standalone interactive HTML file. Returns path."""
    from bokeh.io import output_file, save, reset_output
    output_file(path, title=title)
    save(layout)
    reset_output()
    return path
