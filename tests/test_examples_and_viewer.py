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


def test_bokeh_viewer_accepts_fingerprint_channel_names():
    pytest.importorskip("bokeh")
    from minerva.visualization import bokeh_contact_viewer

    L = 12
    contacts = np.zeros((L, L), dtype=np.float32)
    contacts[2, 9] = contacts[9, 2] = 0.9
    # ``basepairing`` is how Jacobian fingerprints name the channel; an unknown
    # key must be ignored rather than break the viewer.
    layout = bokeh_contact_viewer({"basepairing": contacts, "other": contacts},
                                  vmax={"basepairing": 1.0})
    assert layout is not None


def test_bokeh_viewer_rejects_empty_channels():
    pytest.importorskip("bokeh")
    from minerva.visualization import bokeh_contact_viewer

    with pytest.raises(ValueError, match="base_pairing/repeat/protein"):
        bokeh_contact_viewer({"other": np.zeros((4, 4))})
