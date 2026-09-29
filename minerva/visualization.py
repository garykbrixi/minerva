"""Contact-map visualization for Minerva.

Everything renders in one palette (the Minerva publication colors) through
three entry points that all accept the same ``source``:

    to_rgb(source, tokens)                    -> (L, L, 3) RGB overlay
    plot_contacts(source, tokens, track=True) -> static matplotlib panel
    plot_contacts_interactive(source, tokens) -> Bokeh viewer (``pip install minerva-dna[viz]``)

``source`` is any of:

* a ``{channel: (L, L)}`` dict of head contact maps, e.g. the ``predictions``
  from ``model.predict_contacts``;
* the outputs of ``model(..., output_interactions=True)``, or its
  ``.interactions`` dict;
* a ``FingerprintResult`` from ``model.get_fingerprints``;
* a channel-first ``(C, L, L)`` array, optionally as ``(array, channel_names)``;
* an already rendered ``(L, L, 3)`` RGB image (plotters only).

Head maps are sigmoid probabilities in ``[0, 1]``; Jacobian fingerprints are
Frobenius norms that span roughly ``0..10``. The source type picks the
matching ``vmax`` so both render at the same visual scale.

``plot_dotplot`` is a model-free forward / reverse-complement k-mer dot plot
kept for side-by-side comparison. The names from earlier versions of this
module still work but emit ``DeprecationWarning``; see the end of the file.
"""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np

# =============================================================================
# Palette
# =============================================================================
PALETTE = {
    "protein":      "#7DB4E0",  # PDB blue      - protein contacts
    "base_pairing": "#DA3832",  # RNA red       - base pairing
    "repeat":       "#5F308C",  # repeat purple - DNA repeats
    "other":        "#E09F0E",  # gold          - Jacobian signal the classifier left unassigned
}
LABELS = {
    "protein": "PDB",
    "base_pairing": "RNA",
    "repeat": "Repeats",
    "other": "Jacobian",
}
# Argmax stack order. ``other`` is only rendered for Jacobian fingerprints.
CHANNELS = ["base_pairing", "repeat", "protein", "other"]

# Per-token annotation track: CDS in the protein color so gene bodies and
# protein contacts share a hue; intergenic nucleotides in a light neutral;
# strand markers dark so gene boundaries read as ticks.
TRACK_COLORS = {
    "protein": PALETTE["protein"],
    "nucleotide": "#D9D6CF",
    "special": "#4A4A4A",
}

# Base pairing wins wherever its normalized value clears this, so faint RNA
# structure stays visible against strong protein backgrounds.
DEFAULT_OVERRIDES = {"base_pairing": 0.5}

HEAD_VMAX = 1.0          # sigmoid contact probabilities
FINGERPRINT_VMAX = 10.0  # Frobenius-norm Jacobian fingerprints after centering / APC

LINE_WIDTH = 0.8
PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "xtick.direction": "out",
    "ytick.direction": "out",
}

# Names kept for code written against earlier versions of this module.
COLORS = PALETTE
PUBLICATION_OVERLAY_COLORS = PALETTE
OVERLAY_CHANNELS = CHANNELS[:3]
DEFAULT_OVERLAY_OVERRIDES = DEFAULT_OVERRIDES
DEFAULT_FP_VMAX = FINGERPRINT_VMAX


# =============================================================================
# Small helpers
# =============================================================================
def _hex_to_rgb(c):
    if isinstance(c, str):
        c = c.lstrip("#")
        return np.array([int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.asarray(c, dtype=float)


def _rgb01_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb[:3])


def _finite(x, *, nan=0.0, posinf=1.0, neginf=0.0):
    """Float32 array with non-finite values replaced for stable rendering."""
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=nan, posinf=posinf, neginf=neginf)


def _finite_rgb(rgb):
    """RGB image in [0, 1], with invalid pixels rendered as white."""
    return np.clip(_finite(rgb, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _is_rgb(x) -> bool:
    return isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[-1] == 3


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
    return np.array([_hex_to_rgb(TRACK_COLORS[t]) for t in token_types(tokens)], dtype=np.float32)


# =============================================================================
# Source parsing: anything -> {channel: (L, L)} plus its value scale
# =============================================================================
_ALIASES = {"basepairing": "base_pairing", "base_pair": "base_pairing", "bp": "base_pairing"}


def _canonical(name: str) -> str:
    key = str(name).lower()
    for suffix in ("_l6", "_l2"):   # predict_contacts head names carry the depth
        if key.endswith(suffix) and key[:-len(suffix)] in CHANNELS:
            key = key[:-len(suffix)]
    return _ALIASES.get(key, key)


def _default_channel_names(count: int) -> List[str]:
    if count == 3:
        return ["basepairing", "repeat", "other"]
    if count == 4:
        return ["basepairing", "repeat", "protein", "other"]
    if count == 5:
        return ["bp_forward", "bp_reverse", "repeat", "protein", "other"]
    return [f"channel_{i}" for i in range(count)]


def _parse_source(source, channel_names=None, batch_index: int = 0):
    """Return ``(channels, kind)``: canonical ``{name: (L, L) float32}`` and
    ``kind`` in ``{"heads", "fingerprint"}`` (which sets the default vmax)."""
    if _is_rgb(source):
        raise TypeError("source is already an RGB image; pass it to plot_contacts instead")
    kind = "heads"
    if hasattr(source, "channels") and hasattr(source, "channel_names"):   # FingerprintResult
        raw, kind = dict(source.channels), "fingerprint"
    elif hasattr(source, "interactions"):                                    # model outputs
        if source.interactions is None:
            raise ValueError("outputs.interactions is None; call model(..., output_interactions=True)")
        raw = dict(source.interactions)
    elif isinstance(source, dict):
        raw = dict(source)
        if any(_canonical(k) == "other" or str(k).lower() in _ALIASES for k in raw):
            kind = "fingerprint"
    elif isinstance(source, tuple) and len(source) == 2:
        arr, names = source
        arr = _to_numpy(arr)
        names = list(channel_names or names)
        raw, kind = {n: arr[i] for i, n in enumerate(names)}, "fingerprint"
    else:
        arr = _to_numpy(source)
        if arr.ndim != 3:
            raise ValueError("source must be a channel dict, model outputs, a FingerprintResult, "
                             f"or a (C, L, L) array; got shape {arr.shape}")
        names = list(channel_names or _default_channel_names(arr.shape[0]))
        if len(names) != arr.shape[0]:
            raise ValueError(f"channel_names length {len(names)} != channel dimension {arr.shape[0]}")
        raw, kind = {n: arr[i] for i, n in enumerate(names)}, "fingerprint"

    channels = {}
    for name, values in raw.items():
        arr = _to_numpy(values)
        if arr.ndim == 3:
            arr = arr[batch_index]
        if arr.ndim != 2:
            raise ValueError(f"channel {name!r}: expected an (L, L) map, got shape {arr.shape}")
        channels[_canonical(name)] = _finite(arr)
    if "base_pairing" not in channels:
        parts = [channels[c] for c in ("bp_forward", "bp_reverse") if c in channels]
        if parts:
            channels["base_pairing"] = np.sum(np.stack(parts), axis=0)
    return channels, kind


def _select(channels: Dict[str, np.ndarray], include_other: bool) -> List[str]:
    order = [c for c in CHANNELS if c in channels and (include_other or c != "other")]
    if not order:
        raise ValueError("source must include one of base_pairing/repeat/protein; "
                         f"got {sorted(channels)}")
    return order


def _mask_by_tokens(channels: Dict[str, np.ndarray], tokens: Optional[List[str]]):
    """Protein only on amino-acid pairs, base pairing / repeats only on nucleotide
    pairs. Removes spurious protein signal on intergenic regions and vice versa."""
    if tokens is None:
        return channels
    L = next(iter(channels.values())).shape[0]
    if len(tokens) != L:
        raise ValueError(f"tokens has {len(tokens)} entries but the maps are {L} x {L}")
    tt = np.array(token_types(tokens))
    aa = (tt == "protein")[:, None] & (tt == "protein")[None, :]
    nuc = (tt == "nucleotide")[:, None] & (tt == "nucleotide")[None, :]
    masks = {"protein": aa, "base_pairing": nuc, "repeat": nuc}
    return {c: (np.where(masks[c], v, 0.0).astype(np.float32) if c in masks else v)
            for c, v in channels.items()}


# =============================================================================
# The overlay kernel (shared by the static and interactive renderers)
# =============================================================================
def _assign(channels, order, vmin, vmax, overrides):
    """Winner-take-all assignment.

    Each channel is rescaled to [0, 1] via ``clip((v - vmin) / (vmax - vmin))``,
    the per-pixel argmax picks the dominant channel, and ``overrides`` force a
    channel to win wherever its normalized value clears the threshold.
    Returns ``(max_idx, max_val)`` indexing into ``order``.
    """
    def _bound(arg, ch, default):
        if isinstance(arg, dict):
            return float(arg.get(_canonical(ch), default))
        return float(arg) if arg is not None else default

    first = channels[order[0]]
    stack = np.empty((len(order),) + first.shape, dtype=np.float32)
    for i, ch in enumerate(order):
        lo, hi = _bound(vmin, ch, 0.0), _bound(vmax, ch, 1.0)
        stack[i] = np.clip((channels[ch] - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    max_idx = np.argmax(stack, axis=0)
    max_val = np.max(stack, axis=0)
    for ch, thr in (overrides or {}).items():
        ch = _canonical(ch)
        if ch in order:
            i = order.index(ch)
            m = stack[i] >= thr
            max_idx[m] = i
            max_val[m] = stack[i][m]
    return max_idx, max_val


def _overlay(channels, order, vmin, vmax, overrides, colors=None) -> np.ndarray:
    """RGB from a winner-take-all assignment: ``1 - (1 - color) * value``, a
    subtract-from-white blend, so 0 is white and 1 is the saturated color."""
    palette = colors or PALETTE
    max_idx, max_val = _assign(channels, order, vmin, vmax, overrides)
    palette_rgb = np.array([_hex_to_rgb(palette[ch]) for ch in order], dtype=np.float32)
    return 1.0 - (1.0 - palette_rgb[max_idx]) * max_val[..., None]


def _render(source, tokens=None, *, vmin=0.0, vmax=None, include_other=None,
            overrides=DEFAULT_OVERRIDES, channel_names=None, batch_index=0, mask=True):
    channels, kind = _parse_source(source, channel_names=channel_names, batch_index=batch_index)
    if include_other is None:
        include_other = kind == "fingerprint"
    order = _select(channels, include_other)
    if mask:
        channels = _mask_by_tokens(channels, tokens)
    if vmax is None:
        vmax = FINGERPRINT_VMAX if kind == "fingerprint" else HEAD_VMAX
    return _overlay(channels, order, vmin, vmax, overrides), order


def to_rgb(
    source,
    tokens: Optional[List[str]] = None,
    *,
    vmax: Union[None, float, Dict[str, float]] = None,
    vmin: Union[float, Dict[str, float]] = 0.0,
    include_other: Optional[bool] = None,
    overrides: Optional[Dict[str, float]] = DEFAULT_OVERRIDES,
    channel_names: Optional[List[str]] = None,
    batch_index: int = 0,
    mask: bool = True,
) -> np.ndarray:
    """Render any contact ``source`` (see module docstring) as an RGB overlay.

    Args:
        source: head dict, model outputs, ``FingerprintResult``, or ``(C, L, L)`` array.
        tokens: length-L token list. When given, the protein channel is kept only
            on amino-acid pairs and the RNA / repeat channels only on nucleotide
            pairs (``mask=False`` disables this).
        vmax: value mapped to full saturation, scalar or ``{channel: value}``.
            Defaults to 1 for head probabilities and 10 for Jacobian fingerprints.
        include_other: render the gold unclassified channel. Defaults to True for
            Jacobian fingerprints, which are the only sources that have one.
        overrides: ``{channel: threshold}`` forced wins, or None for pure argmax.
        channel_names: names for a raw ``(C, L, L)`` array in non-default order.
        batch_index: which item to take from batched ``(B, L, L)`` maps.

    Returns:
        ``(L, L, 3)`` float array in ``[0, 1]``.
    """
    rgb, _ = _render(source, tokens, vmin=vmin, vmax=vmax, include_other=include_other,
                     overrides=overrides, channel_names=channel_names,
                     batch_index=batch_index, mask=mask)
    return rgb


# =============================================================================
# Static plotter
# =============================================================================
def _legend_handles(order: List[str]):
    from matplotlib.patches import Patch
    return [Patch(facecolor=PALETTE[ch], edgecolor="black", linewidth=0.4, label=LABELS[ch])
            for ch in order]


def _draw_panel(ax, rgb, title: Optional[str], genome_offset: int = 0, show_yticks: bool = True):
    """One square publication-style panel with genome-coordinate ticks."""
    L = rgb.shape[0]
    ax.imshow(_finite_rgb(rgb), aspect="equal", interpolation="nearest")
    ticks = np.arange(0, L, max(1, L // 4))
    labels = [f"{int(t + genome_offset):,}" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels if show_yticks else [], fontsize=6)
    if title:
        ax.set_title(title, fontsize=11, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LINE_WIDTH)
    ax.tick_params(axis="both", which="major", direction="out", length=3,
                   width=LINE_WIDTH, labelsize=6, bottom=True, left=True, top=False, right=False)


def _draw_track(ax, tokens: List[str], title: Optional[str], frac: float = 0.035, gap: float = 0.012):
    """Token-type strips above and left of ``ax`` (inset axes, so any ``ax`` works)."""
    strip = _track_rgb(tokens)
    top = ax.inset_axes([0.0, 1.0 + gap, 1.0, frac])
    top.imshow(strip[None, :, :], aspect="auto", interpolation="nearest")
    left = ax.inset_axes([-frac - gap, 0.0, frac, 1.0])
    left.imshow(strip[:, None, :], aspect="auto", interpolation="nearest")
    for a in (top, left):
        a.set_xticks([]); a.set_yticks([])
        for spine in a.spines.values():
            spine.set_linewidth(LINE_WIDTH * 0.5)
    # Push the y tick labels and the title out past the strips.
    fig = ax.figure
    fig.canvas.draw_idle()
    bbox = ax.get_position()
    strip_pts = (frac + gap) * bbox.width * fig.get_figwidth() * 72
    ax.tick_params(axis="y", pad=strip_pts + 2)
    if title:
        ax.set_title(title, fontsize=11, pad=(frac + gap) * bbox.height * fig.get_figheight() * 72 + 4)


def plot_contacts(
    source,
    tokens: Optional[List[str]] = None,
    *,
    track: bool = False,
    genome_offset: int = 0,
    title: Optional[str] = None,
    show_legend: bool = True,
    legend_channels: Optional[List[str]] = None,
    ax=None,
    figsize=(5, 5),
    save: Optional[str] = None,
    dpi: int = 600,
    **rgb_kwargs,
):
    """Static publication-style contact map. Returns ``(fig, ax)``.

    Args:
        source: anything :func:`to_rgb` accepts, or an already rendered
            ``(L, L, 3)`` RGB image.
        tokens: length-L token list; enables the token-aware channel mask and
            is required for ``track=True``.
        track: draw a per-token annotation strip (CDS / intergenic / strand
            marker) along the top and left edges.
        genome_offset: added to the tick labels, e.g. the window start.
        legend_channels: which swatches to show when ``source`` is a raw RGB
            image (otherwise the legend lists exactly the channels rendered).
        ax: draw into an existing Axes; otherwise a new figure of ``figsize``.
        save: write the figure to this path at ``dpi`` (PDF, PNG, SVG, ...).
        **rgb_kwargs: forwarded to :func:`to_rgb` (``vmax``, ``include_other``, ...).
    """
    import matplotlib.pyplot as plt

    if _is_rgb(source):
        rgb, order = source, list(legend_channels or CHANNELS[:3])
    else:
        rgb, order = _render(source, tokens, **rgb_kwargs)
        if legend_channels is not None:
            order = list(legend_channels)
    if track and (tokens is None or len(tokens) != rgb.shape[0]):
        raise ValueError("track=True needs `tokens` aligned to the map (one token per position)")

    with plt.rc_context(PUBLICATION_RC):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        _draw_panel(ax, rgb, title, genome_offset=genome_offset)
        if track:
            _draw_track(ax, tokens, title)
        if show_legend:
            leg = ax.legend(handles=_legend_handles([_canonical(c) for c in order]),
                            loc="upper right", fontsize=8, frameon=True, framealpha=0.9,
                            edgecolor="black", fancybox=False)
            leg.get_frame().set_linewidth(0.6)
        if save:
            fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, ax


# =============================================================================
# Interactive Bokeh viewer
# =============================================================================
def plot_contacts_interactive(
    source,
    tokens: Optional[List[str]] = None,
    *,
    track: bool = True,
    title: str = "Minerva contacts",
    genome_offset: int = 0,
    prefilter: float = 0.05,
    init_threshold: float = 0.3,
    vmax: Union[None, float, Dict[str, float]] = None,
    include_other: Optional[bool] = None,
    overrides: Optional[Dict[str, float]] = DEFAULT_OVERRIDES,
    size: int = 760,
    track_px: int = 46,
    mask: bool = True,
):
    """Interactive Bokeh contact map: wheel-zoom / pan, hover values, one
    threshold slider per channel, and optional token-type tracks. Needs
    ``pip install minerva-dna[viz]``.

    Pixels are assigned to channels exactly as in :func:`to_rgb`, so the
    interactive and static views agree. The sliders then threshold each
    channel's pixels on the normalized 0..1 scale.

    Args:
        source: anything :func:`to_rgb` accepts.
        tokens: length-L token list; enables the channel mask, token hover,
            and the tracks. ``track`` is ignored without tokens.
        genome_offset: added to hover positions.
        prefilter: contacts below this are dropped before sending to the browser.
        init_threshold: initial slider value per channel.
        vmax: value mapped to full color; defaults by source type as in :func:`to_rgb`.
        size: main panel size in px. track_px: track thickness in px.

    Returns:
        A Bokeh layout. In a notebook: ``from bokeh.io import output_notebook,
        show; output_notebook(); show(layout)``. To a file: :func:`save_bokeh_html`.
    """
    from bokeh.plotting import figure
    from bokeh.models import (ColumnDataSource, CustomJS, Slider, HoverTool,
                              Legend, LegendItem, Range1d, WheelZoomTool, Div)
    from bokeh.layouts import gridplot, column, row

    channels, kind = _parse_source(source)
    if include_other is None:
        include_other = kind == "fingerprint"
    order = _select(channels, include_other)
    if mask:
        channels = _mask_by_tokens(channels, tokens)
    if vmax is None:
        vmax = FINGERPRINT_VMAX if kind == "fingerprint" else HEAD_VMAX
    max_idx, max_val = _assign(channels, order, 0.0, vmax, overrides)
    L = int(max_idx.shape[0])
    # Draw bottom -> top so dense protein contacts don't bury sparser signal.
    draw_order = [c for c in ("other", "protein", "repeat", "base_pairing") if c in order]

    def _points(channel):
        ci = order.index(channel)
        rows, cols = np.where((max_idx == ci) & (max_val > prefilter))
        vals = max_val[rows, cols]
        base = _hex_to_rgb(PALETTE[channel])
        blended = 1.0 - (1.0 - base)[None, :] * vals[:, None]
        return dict(i=cols.astype(int).tolist(), j=rows.astype(int).tolist(),
                    val=[round(float(v), 4) for v in vals],
                    color=[_rgb01_to_hex(c) for c in blended])

    x_range = Range1d(start=0, end=L, bounds=(0, L))
    y_range = Range1d(start=L, end=0, bounds=(0, L))
    p = figure(title=None, x_range=x_range, y_range=y_range, width=size, height=size,
               tools="pan,box_zoom,reset,save", toolbar_location="right",
               output_backend="webgl", x_axis_location="below", y_axis_location="left")
    wheel = WheelZoomTool(dimensions="both")
    p.add_tools(wheel); p.toolbar.active_scroll = wheel
    p.grid.visible = False
    p.line([0, L], [0, L], line_color="#BBBBBB", line_width=1, line_dash="dashed")
    p.xaxis.axis_label = "Position"; p.yaxis.axis_label = "Position"

    renderers, full_src, shown_src = {}, {}, {}
    for c in draw_order:
        full = ColumnDataSource(_points(c))
        vals = np.asarray(full.data["val"])
        keep = vals >= init_threshold if len(vals) else np.zeros(0, bool)
        shown = ColumnDataSource({k: list(np.asarray(v, dtype=object)[keep]) for k, v in full.data.items()})
        renderers[c] = p.rect(x="i", y="j", width=1.0, height=1.0, source=shown,
                              fill_color="color", line_color=None, fill_alpha=0.95)
        full_src[c], shown_src[c] = full, shown

    sliders = []
    for c in order:
        slider = Slider(start=round(prefilter, 3), end=1.0, value=init_threshold, step=0.01,
                        title=LABELS[c], width=size // 2, bar_color=PALETTE[c])
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

    p.add_tools(HoverTool(renderers=list(renderers.values()),
                          tooltips=[("position", "@i, @j"), ("value", "@val")],
                          point_policy="follow_mouse"))
    legend = Legend(items=[LegendItem(label=LABELS[c], renderers=[renderers[c]]) for c in order],
                    location="top", border_line_color=None)
    p.add_layout(legend, "right")

    top_track = left_track = None
    if track and tokens is not None and len(tokens) == L:
        types = token_types(tokens)
        tcol = [TRACK_COLORS[t] for t in types]
        idx = list(range(L))
        top_src = ColumnDataSource(dict(left=idx, right=[i + 1 for i in idx], top=[1] * L,
                                        bottom=[0] * L, color=tcol, tok=list(tokens), typ=types,
                                        pos=[i + genome_offset for i in idx]))
        top_track = figure(x_range=x_range, y_range=Range1d(0, 1), width=size, height=track_px,
                           tools="", toolbar_location=None, output_backend="webgl")
        top_track.quad(left="left", right="right", top="top", bottom="bottom",
                       source=top_src, fill_color="color", line_color=None)
        top_track.add_tools(HoverTool(tooltips=[("pos", "@pos"), ("token", "@tok"), ("type", "@typ")]))
        top_track.grid.visible = False; top_track.yaxis.visible = False
        top_track.xaxis.visible = False; top_track.title = title

        left_src = ColumnDataSource(dict(bottom=idx, top=[i + 1 for i in idx], left=[0] * L,
                                         right=[1] * L, color=tcol, tok=list(tokens), typ=types))
        left_track = figure(x_range=Range1d(0, 1), y_range=y_range, width=track_px, height=size,
                            tools="", toolbar_location=None, output_backend="webgl")
        left_track.quad(left="left", right="right", top="top", bottom="bottom",
                        source=left_src, fill_color="color", line_color=None)
        left_track.grid.visible = False; left_track.xaxis.visible = False; left_track.yaxis.visible = False

    if top_track is not None:
        grid = gridplot([[None, top_track], [left_track, p]], toolbar_location="right", merge_tools=False)
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


# =============================================================================
# Model-free dot plot
# =============================================================================
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


def dotplot_rgb_from_tokens(tokens: List[str], word_size: int = 6, threshold: Optional[int] = None):
    """Forward / reverse-complement dot plot RGB image for a mixed-token window.

    Non-nucleotide tokens stay white. Forward matches use the repeat color;
    reverse-complement matches use the RNA color.
    """
    img = np.ones((len(tokens), len(tokens), 3), dtype=np.float32)
    nuc_subseq, nuc_sub_to_tok = "", {}
    for i, tok in enumerate(tokens):
        if len(tok) == 1 and tok.upper() in "ACGTN":
            nuc_sub_to_tok[len(nuc_subseq)] = i
            nuc_subseq += tok.upper()
    if len(nuc_subseq) < word_size:
        return img, {"forward": 0, "revcomp": 0, "nucleotide_tokens": len(nuc_subseq)}

    fwd_r, fwd_c, rc_r, rc_c, _ = compute_dotplot_fwd_rc(nuc_subseq, window=word_size, threshold=threshold)
    for rows, cols, color in ((fwd_r, fwd_c, PALETTE["repeat"]), (rc_r, rc_c, PALETTE["base_pairing"])):
        rgb = _hex_to_rgb(color).astype(np.float32)
        for r, c in zip(rows, cols):
            r_tok, c_tok = nuc_sub_to_tok.get(int(r)), nuc_sub_to_tok.get(int(c))
            if r_tok is not None and c_tok is not None:
                img[r_tok, c_tok] = rgb
    return img, {"forward": int(len(fwd_r)), "revcomp": int(len(rc_r)),
                 "nucleotide_tokens": int(len(nuc_subseq))}


def plot_dotplot(
    tokens: List[str],
    title: Optional[str] = "Dot plot",
    *,
    word_size: int = 6,
    threshold: Optional[int] = None,
    genome_offset: int = 0,
    ax=None,
    figsize=(5, 5),
    save: Optional[str] = None,
    dpi: int = 600,
):
    """Model-free forward / reverse-complement k-mer dot plot of a token window.
    Returns ``(fig, ax, stats)``."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    img, stats = dotplot_rgb_from_tokens(tokens, word_size=word_size, threshold=threshold)
    with plt.rc_context(PUBLICATION_RC):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        _draw_panel(ax, img, title, genome_offset=genome_offset)
        handles = [Patch(facecolor=PALETTE["repeat"], edgecolor="black", linewidth=0.4, label="Forward"),
                   Patch(facecolor=PALETTE["base_pairing"], edgecolor="black", linewidth=0.4, label="Rev. comp.")]
        leg = ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True,
                        framealpha=0.9, edgecolor="black", fancybox=False)
        leg.get_frame().set_linewidth(0.6)
        if save:
            fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, ax, stats


# =============================================================================
# Deprecated names. Each maps onto the functions above and will be removed in
# a later release.
# =============================================================================
def _deprecated(old: str, new: str):
    warnings.warn(f"minerva.visualization.{old} is deprecated; use {new} instead.",
                  DeprecationWarning, stacklevel=3)


def contact_rgb_overlay(channels, channel_order=None, colors=None, vmin=0.0, vmax=1.0,
                        overrides=DEFAULT_OVERRIDES):
    _deprecated("contact_rgb_overlay", "to_rgb")
    chans = {_canonical(k): _finite(v) for k, v in channels.items()}
    order = [_canonical(c) for c in (channel_order or list(channels))]
    order = [c for c in order if c in chans]
    if not order:
        raise ValueError("no channels in `channel_order` matched `channels`")
    return _overlay(chans, order, vmin, vmax, overrides, colors=colors)


def render_fingerprints(fingerprints, channel_names=None, *, style="default", include_other=None,
                        colors=None, vmin=0.0, vmax=FINGERPRINT_VMAX, overrides=DEFAULT_OVERRIDES):
    _deprecated("render_fingerprints", "to_rgb")
    if include_other is None:
        include_other = style == "publication"
    return to_rgb(fingerprints, vmin=vmin, vmax=vmax, include_other=include_other,
                  overrides=overrides, channel_names=channel_names)


def render_interactions(interactions, tokens=None, *, style="default", batch_index=0, colors=None,
                        vmin=0.0, vmax=HEAD_VMAX, overrides=DEFAULT_OVERRIDES):
    _deprecated("render_interactions", "to_rgb")
    return to_rgb(interactions, tokens, vmin=vmin, vmax=vmax, overrides=overrides,
                  batch_index=batch_index)


def head_contacts_rgb(contacts, tokens=None, colors=None, vmin=0.0, vmax=HEAD_VMAX,
                      overrides=DEFAULT_OVERRIDES):
    _deprecated("head_contacts_rgb", "to_rgb")
    return to_rgb(contacts, tokens, vmin=vmin, vmax=vmax, overrides=overrides, include_other=False)


def publication_head_contacts_rgb(contacts, tokens=None, auto_contrast=False,
                                  contrast_percentile=99.0, return_stats=False, **kwargs):
    _deprecated("publication_head_contacts_rgb", "to_rgb")
    if auto_contrast:
        warnings.warn("auto_contrast was removed (it rendered every pixel as base pairing); "
                      "pass an explicit vmax instead.", stacklevel=2)
    kwargs.pop("colors", None)
    rgb = to_rgb(contacts, tokens, include_other=False, **kwargs)
    return (rgb, None) if return_stats else rgb


def jacobian_fingerprint_rgb(jac, tokens, *, bp_threshold=None, repeat_threshold=None,
                             protein_threshold=None, aa_start=4, jac_aa_order=None, colors=None,
                             vmin=0.0, vmax=FINGERPRINT_VMAX, overrides=DEFAULT_OVERRIDES,
                             include_other=False):
    """Raw ``(L, A, L, A)`` Jacobian -> classifier -> RGB. Prefer
    ``model.get_fingerprints`` followed by :func:`to_rgb`."""
    _deprecated("jacobian_fingerprint_rgb", "model.get_fingerprints + to_rgb")
    try:
        from .jacobian import fingerprint_jacobian_multimodality
    except ImportError:  # HF snapshot imported as top-level visualization.py
        from jacobian import fingerprint_jacobian_multimodality
    kw = {"split_bp": False, "aa_start": aa_start, "jac_aa_order": jac_aa_order}
    for key, val in (("bp_threshold", bp_threshold), ("repeat_threshold", repeat_threshold),
                     ("protein_threshold", protein_threshold)):
        if val is not None:
            kw[key] = val
    _, fingerprints, channel_names = fingerprint_jacobian_multimodality(jac, tokens, **kw)
    rgb = to_rgb((fingerprints, channel_names), vmin=vmin, vmax=vmax, overrides=overrides,
                 include_other=include_other)
    return rgb, fingerprints, channel_names


def publication_jacobian_fingerprint_rgb(jac, tokens, **kwargs):
    kwargs.pop("colors", None)
    kwargs.setdefault("include_other", True)
    return jacobian_fingerprint_rgb(jac, tokens, **kwargs)


def legend_handles(channels=None, colors=None):
    _deprecated("legend_handles", "plot_contacts")
    return _legend_handles([_canonical(c) for c in (channels or CHANNELS[:3])])


def setup_publication_style():
    """Apply the publication rcParams globally. ``plot_contacts`` applies them
    per figure instead."""
    _deprecated("setup_publication_style", "plot_contacts")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PUBLICATION_RC)


def render_publication_panel(ax, rgb, label, show_yticks=True, genome_offset=0):
    _deprecated("render_publication_panel", "plot_contacts")
    _draw_panel(ax, rgb, label, genome_offset=genome_offset, show_yticks=show_yticks)


def plot_fingerprint_overlay(rgb, title="Minerva multimodal fingerprint", ax=None, extent=None,
                             show_legend=True, figsize=(9, 9), legend_channels=None, colors=None):
    _deprecated("plot_fingerprint_overlay", "plot_contacts")
    _, ax = plot_contacts(rgb, title=title, ax=ax, show_legend=show_legend, figsize=figsize,
                          legend_channels=legend_channels)
    return ax


def plot_fingerprints(fingerprints, channel_names=None, *, title="Minerva multimodal fingerprint",
                      style="default", include_other=None, ax=None, extent=None, show_legend=True,
                      figsize=(9, 9), return_rgb=False, **render_kwargs):
    _deprecated("plot_fingerprints", "plot_contacts")
    if include_other is None:
        include_other = style == "publication"
    render_kwargs.pop("colors", None)
    rgb = to_rgb(fingerprints, channel_names=channel_names, include_other=include_other, **render_kwargs)
    order = _select(_parse_source(fingerprints, channel_names=channel_names)[0], include_other)
    _, ax = plot_contacts(rgb, title=title, ax=ax, show_legend=show_legend, figsize=figsize,
                          legend_channels=order)
    return (ax, rgb) if return_rgb else ax


def plot_interactions(interactions, tokens=None, *, title="Minerva interactions", style="default",
                      batch_index=0, ax=None, extent=None, show_legend=True, figsize=(9, 9),
                      return_rgb=False, **render_kwargs):
    _deprecated("plot_interactions", "plot_contacts")
    render_kwargs.pop("colors", None)
    rgb = to_rgb(interactions, tokens, batch_index=batch_index, include_other=False, **render_kwargs)
    order = _select(_parse_source(interactions, batch_index=batch_index)[0], False)
    _, ax = plot_contacts(rgb, title=title, ax=ax, show_legend=show_legend, figsize=figsize,
                          legend_channels=order)
    return (ax, rgb) if return_rgb else ax


def plot_locus(rgb, tokens=None, title="Minerva locus", figsize=(9, 9), show_legend=True,
               save=None, dpi=200):
    _deprecated("plot_locus", "plot_contacts(..., track=True)")
    track = tokens is not None and len(tokens) == np.asarray(rgb).shape[0]
    fig, _ = plot_contacts(rgb, tokens if track else None, track=track, title=title,
                           figsize=figsize, show_legend=show_legend, save=save, dpi=dpi)
    return fig


def plot_publication_locus(rgb, title="Minerva locus", *, genome_offset=0, overlay_kind="heads",
                           show_legend=True, ax=None, figsize=(5, 5), save=None, dpi=600):
    _deprecated("plot_publication_locus", "plot_contacts")
    legend = ["other", "base_pairing", "repeat"] if overlay_kind == "fingerprint" else CHANNELS[:3]
    return plot_contacts(rgb, title=title, genome_offset=genome_offset, show_legend=show_legend,
                         legend_channels=legend, ax=ax, figsize=figsize, save=save, dpi=dpi)


def interactive_overlay(rgb, title="Minerva locus"):
    """Plotly zoom / pan of a rendered RGB image. Plotly is no longer a
    dependency; use :func:`plot_contacts_interactive`."""
    _deprecated("interactive_overlay", "plot_contacts_interactive")
    try:
        import plotly.express as px
    except ImportError as e:
        raise ImportError("interactive_overlay needs plotly, which is no longer installed with "
                          "minerva-dna. Use plot_contacts_interactive (pip install minerva-dna[viz]).") from e
    img = (_finite_rgb(rgb) * 255).astype(np.uint8)
    fig = px.imshow(img, title=title)
    fig.update_layout(dragmode="pan", margin=dict(l=10, r=10, t=40, b=10), height=760, width=800)
    fig.update_xaxes(title="Position", constrain="domain")
    fig.update_yaxes(title="Position", scaleanchor="x")
    return fig


def bokeh_contact_viewer(channels, tokens=None, **kwargs):
    _deprecated("bokeh_contact_viewer", "plot_contacts_interactive")
    return plot_contacts_interactive(channels, tokens, **kwargs)
