"""Tests for the pieces the example notebooks lean on.

The notebooks used to guard these at runtime (searching for example files,
string-matching the RMSNorm source, renaming fingerprint channels by hand).
Those guards now live here instead.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from minerva.data import EXAMPLES, example_path, extract_and_tokenize_gb
from minerva.modeling_minerva import rmsnorm_func


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_path_resolves_to_a_parseable_genbank(name):
    path = example_path(name)
    assert os.path.isfile(path)
    records = extract_and_tokenize_gb(path, use_existing_translations=True)
    assert records and records[0]["sequence"]


def test_example_path_is_case_insensitive_and_rejects_unknown():
    assert example_path("UG27") == example_path("ug27")
    with pytest.raises(ValueError, match="Unknown example"):
        example_path("nope")


def test_rmsnorm_is_fp16_safe():
    # 322**2 overflows fp16 (max 65504): squaring in half precision gives inf
    # and the layer collapses to zero. The norm must be computed in fp32.
    x = torch.full((1, 4), 322.0, dtype=torch.float16)
    y = rmsnorm_func(x, torch.ones(4, dtype=torch.float16), 1e-5)
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.ones_like(y), atol=1e-3, rtol=0)


def _toy_channels(L=12):
    contacts = np.zeros((L, L), dtype=np.float32)
    contacts[2, 9] = contacts[9, 2] = 0.9
    return contacts


def test_to_rgb_scales_heads_and_fingerprints_differently():
    from minerva.visualization import to_rgb, PALETTE, _hex_to_rgb

    c = _toy_channels()
    heads = to_rgb({"protein": c})                       # probabilities: vmax 1
    fp = to_rgb({"protein": c, "other": np.zeros_like(c)})  # fingerprint naming: vmax 10
    assert heads.shape == fp.shape == (12, 12, 3)
    np.testing.assert_allclose(heads[5, 5], [1, 1, 1])   # empty pixel is white
    blue = _hex_to_rgb(PALETTE["protein"])
    np.testing.assert_allclose(heads[2, 9], 1 - (1 - blue) * 0.9, atol=1e-6)
    np.testing.assert_allclose(fp[2, 9], 1 - (1 - blue) * 0.09, atol=1e-6)


def test_to_rgb_accepts_fingerprint_channel_names_and_l6_heads():
    from minerva.visualization import to_rgb

    c = _toy_channels()
    a = to_rgb({"basepairing": c}, vmax=1.0)
    b = to_rgb({"base_pairing_l6": c}, vmax=1.0)
    np.testing.assert_array_equal(a, b)


def test_to_rgb_masks_by_token_type():
    from minerva.visualization import to_rgb

    c = _toy_channels()
    tokens = ["a"] * 12                                  # all nucleotides: no protein pixels survive
    rgb = to_rgb({"protein": c}, tokens=tokens)
    np.testing.assert_allclose(rgb, 1.0)
    rgb = to_rgb({"repeat": c}, tokens=tokens)
    assert rgb[2, 9].min() < 1.0


def test_plot_contacts_legend_matches_rendered_channels():
    import matplotlib
    matplotlib.use("Agg")
    from minerva.visualization import plot_contacts, LABELS

    c = _toy_channels()
    fig, ax = plot_contacts({"repeat": c, "protein": c}, tokens=["a"] * 6 + ["A"] * 6, track=True)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == [LABELS["repeat"], LABELS["protein"]]
    with pytest.raises(ValueError, match="track=True"):
        plot_contacts({"repeat": c}, track=True)


def test_interactive_viewer_accepts_fingerprint_channel_names():
    pytest.importorskip("bokeh")
    from minerva.visualization import plot_contacts_interactive

    c = _toy_channels()
    # ``basepairing`` is how Jacobian fingerprints name the channel; an unknown
    # key must be ignored rather than break the viewer.
    layout = plot_contacts_interactive({"basepairing": c, "unknown": c}, vmax={"basepairing": 1.0})
    assert layout is not None


def test_interactive_viewer_rejects_empty_channels():
    pytest.importorskip("bokeh")
    from minerva.visualization import plot_contacts_interactive

    with pytest.raises(ValueError, match="base_pairing/repeat/protein"):
        plot_contacts_interactive({"unknown": np.zeros((4, 4))})


def test_deprecated_names_warn_and_match_new_api():
    import matplotlib
    matplotlib.use("Agg")
    from minerva import visualization as V

    c = _toy_channels()
    expected = V.to_rgb({"protein": c}, vmax=1.0)
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(V.head_contacts_rgb({"protein": c}), expected)
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(V.publication_head_contacts_rgb({"protein": c}), expected)
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(V.render_interactions({"protein": c}), expected)
    with pytest.warns(DeprecationWarning):
        fig, ax = V.plot_publication_locus(expected, title="t")
    assert ax.get_legend() is not None
