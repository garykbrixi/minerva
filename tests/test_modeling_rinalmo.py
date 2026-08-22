"""Tests for the RiNALMo backbone and its interaction heads.

All use a small randomly-initialised model, so none need the 2.6 GB checkpoint.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minerva.modeling_rinalmo import (  # noqa: E402
    PAD_TOKEN_ID,
    RiNALMoMinervaConfig,
    RiNALMoMinervaForMaskedLM,
)

CLS, EOS = 0, 2
NUM_BLOCKS, NUM_HEADS = 4, 4


def _model(**heads):
    cfg = RiNALMoMinervaConfig(
        embed_dim=64, num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
        linear_heads_config=heads or {
            task: {"type": "linear", "input_dim": 2 * NUM_HEADS, "layers": [2, 3],
                   "apply_symmetrize": True, "apply_apc": False}
            for task in RiNALMoMinervaForMaskedLM.interaction_tasks
        },
    )
    model = RiNALMoMinervaForMaskedLM(cfg).eval()
    model.head_depths = {2: ""}          # toy model has 4 layers, not 33
    return model


def _tokens(length=12, batch=1):
    ids = torch.randint(5, 22, (batch, length))
    ids[:, 0], ids[:, -1] = CLS, EOS
    return ids


def test_vendored_backbone_imports_without_flash_attn():
    from minerva.vendor_rinalmo.attention import _HAS_FLASH, MultiHeadSelfAttention

    assert MultiHeadSelfAttention is not None
    if not _HAS_FLASH:
        assert True   # the point: the import above did not raise


def test_forward_returns_logits_over_all_tokens():
    model, ids = _model(), _tokens(12)
    with torch.no_grad():
        out = model(ids)
    assert out.logits.shape == (1, 12, model.config.alphabet_size)


def test_interactions_are_cropped_to_the_sequence():
    """CLS/EOS must not appear as rows/columns; heads were trained without them."""
    model, ids = _model(), _tokens(12)
    with torch.no_grad():
        out = model(ids, output_interactions=True, interaction_layers=2)
    assert out.interactions["base_pairing"].shape == (1, 10, 10)


def test_both_head_types_run():
    model = _model(
        base_pairing={"type": "linear", "input_dim": 2 * NUM_HEADS, "layers": [2, 3],
                      "apply_symmetrize": True, "apply_apc": False},
        repeat={"type": "attention", "layer_idx": 3, "head_idx": 1},
    )
    with torch.no_grad():
        out = model(_tokens(12), output_interactions=True, interaction_layers=2)
    assert set(out.interactions) == {"base_pairing", "repeat"}


def test_interactions_cost_one_encoder_pass():
    """The headline property: heads reuse the hidden state feeding the LM head."""
    model, ids = _model(), _tokens(12)
    calls = []
    for block in model.rinalmo.transformer.blocks:
        original = block.forward
        block.forward = (lambda *a, _o=original, **k: (calls.append(1), _o(*a, **k))[1])

    with torch.no_grad():
        model(ids, output_interactions=True, interaction_layers=2)
    assert len(calls) == NUM_BLOCKS

    calls.clear()
    with torch.no_grad():
        model(ids)
    assert len(calls) == NUM_BLOCKS


def test_only_requested_layers_are_retained():
    """Keeping every layer's attention is what OOMs at long context."""
    model = _model()
    maps = model.get_attention_maps(_tokens(12), layers=[2, 3])
    assert sorted(maps) == [2, 3]
    assert maps[2].shape == (1, NUM_HEADS, 12, 12)


def test_padding_is_masked():
    model = _model()
    ids = _tokens(12, batch=1)
    ids[0, -3:] = PAD_TOKEN_ID
    maps = model.get_attention_maps(ids, layers=[3])
    # padded keys receive no attention mass
    assert torch.allclose(maps[3][0, :, :, -3:], torch.zeros(1), atol=1e-6)


def test_head_depth_must_be_declared():
    model = _model()
    with pytest.raises(ValueError, match="interaction_layers must be one of"):
        model(_tokens(12), output_interactions=True, interaction_layers=99)


def test_tokenizer_ids_match_upstream_alphabet():
    """Ids must match RiNALMo's ordering or the pretrained embeddings are wrong."""
    from minerva.modeling_rinalmo import RNA_TOKENS, SPECIAL_TOKENS
    from minerva.tokenization_rinalmo import build_rinalmo_tokenizer

    vocab = build_rinalmo_tokenizer().get_vocab()
    expected = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    expected.update({t: i + len(SPECIAL_TOKENS) for i, t in enumerate(RNA_TOKENS)})
    assert vocab == expected


def test_tokenizer_folds_u_onto_t_and_ignores_case():
    """Upstream encode() upper-cases and rewrites U to T; there is no U token."""
    from minerva.tokenization_rinalmo import build_rinalmo_tokenizer

    tok = build_rinalmo_tokenizer()
    assert tok("ACGU")["input_ids"] == tok("ACGT")["input_ids"] == tok("acgu")["input_ids"]
    assert "U" not in tok.get_vocab()


def test_tokenizer_wraps_with_cls_eos_one_token_per_base():
    from minerva.tokenization_rinalmo import build_rinalmo_tokenizer

    tok = build_rinalmo_tokenizer()
    ids = tok("ACGTACGT")["input_ids"]
    assert len(ids) == 8 + 2
    assert ids[0] == tok.cls_token_id and ids[-1] == tok.eos_token_id


def test_tokenizer_round_trips_through_the_model():
    from minerva.tokenization_rinalmo import build_rinalmo_tokenizer

    tok = build_rinalmo_tokenizer()
    model = _model()
    ids = torch.tensor([tok("ACGUACGUACGU")["input_ids"]])
    with torch.no_grad():
        out = model(ids, output_interactions=True, interaction_layers=2)
    assert out.logits.shape[1] == ids.shape[1]
    assert out.interactions["base_pairing"].shape[-1] == ids.shape[1] - 2
