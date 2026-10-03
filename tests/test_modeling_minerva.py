"""Tests for MinervaForMaskedLM's attention-map paths.

All use a small randomly-initialised model, so none need the checkpoint.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minerva import modeling_minerva  # noqa: E402
from minerva.modeling_minerva import MinervaConfig, MinervaForMaskedLM  # noqa: E402

DEPTH, HEADS = 4, 4

# CUDA runs in fp16: flash-attn needs half precision, and fp16 rounds finer than bf16.
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))]
TOLERANCE = {"cpu": {}, "cuda": {"atol": 1e-3, "rtol": 1e-3}}


def _model(device="cpu"):
    torch.manual_seed(0)
    cfg = MinervaConfig(
        dim=64, depth=DEPTH, heads=HEADS, vocab_size=37,
        linear_heads_config={
            task: {"type": "linear", "input_dim": 2 * HEADS, "layers": [2, 3],
                   "apply_symmetrize": True, "apply_apc": False}
            for task in MinervaForMaskedLM.interaction_tasks
        },
    )
    model = MinervaForMaskedLM(cfg).eval()
    model.head_depths = {2: ""}          # toy model has 4 layers, not 33
    for head in model.linear_heads.values():
        # with the near-zero default init, every head outputs ~0.5
        torch.nn.init.normal_(head.linear.weight)
    return model.to(device, torch.float32 if device == "cpu" else torch.float16)


def _padded_batch(short=10, length=14):
    """A short and a full-length sequence in one batch, plus the short one alone.

    The padding holds arbitrary tokens on purpose: masking must make it irrelevant.
    """
    torch.manual_seed(0)
    ids = torch.randint(4, 34, (2, length))
    mask = torch.ones(2, length, dtype=torch.bool)
    mask[0, short:] = False
    return ids, mask, ids[:1, :short]


@pytest.mark.parametrize("device", DEVICES)
def test_each_padded_row_matches_its_sequence_alone(device):
    """Padding is masked in every layer, and rows of a batch don't see each other."""
    model, tol = _model(device), TOLERANCE[device]
    ids, mask, _ = (t.to(device) for t in _padded_batch())
    with torch.no_grad():
        batch = model(ids, attention_mask=mask, output_interactions=True, output_attentions=True)
        for row in range(ids.shape[0]):
            n = int(mask[row].sum())
            alone = model(ids[row:row + 1, :n], output_interactions=True, output_attentions=True)

            torch.testing.assert_close(batch.logits[row, :n], alone.logits[0], **tol)
            for task, cmap in alone.interactions.items():
                torch.testing.assert_close(batch.interactions[task][row, :n, :n], cmap[0], **tol)
            for layer, attn in alone.attentions.items():
                torch.testing.assert_close(batch.attentions[layer][row, :, :n, :n], attn[0], **tol)


@pytest.mark.parametrize("device", DEVICES)
def test_padding_content_never_reaches_real_tokens(device):
    """Whatever the padded positions hold, real-token outputs are bit-identical."""
    model = _model(device)
    ids, mask, _ = (t.to(device) for t in _padded_batch())
    other = ids.clone()
    other[~mask] = (other[~mask] + 1) % 30 + 4
    with torch.no_grad():
        a = model(ids, attention_mask=mask, output_interactions=True, output_attentions=True)
        b = model(other, attention_mask=mask, output_interactions=True, output_attentions=True)

    n = int(mask[0].sum())
    assert torch.equal(a.logits[mask], b.logits[mask])
    for task in a.interactions:
        assert torch.equal(a.interactions[task][0, :n, :n], b.interactions[task][0, :n, :n])
    for layer in a.attentions:
        assert torch.equal(a.attentions[layer][0, :, :n, :n], b.attentions[layer][0, :, :n, :n])


def test_predict_contacts_batch_matches_single():
    model = _model()
    ids, mask, alone_ids = _padded_batch()
    n = alone_ids.shape[1]
    single = model.predict_contacts(input_ids=alone_ids[0], head_names=["base_pairing"])
    batch = model.predict_contacts(input_ids=ids, attention_mask=mask, head_names=["base_pairing"])
    torch.testing.assert_close(batch[0, :n, :n], single)


@pytest.mark.parametrize("request_maps", ["output_attentions", "output_interactions"])
def test_requesting_maps_does_not_change_training(request_maps):
    """Same loss and gradients as a plain forward, train mode kept, maps gradient-free."""
    model = _model().train()
    ids = torch.randint(4, 34, (2, 12))

    model(ids, labels=ids).loss.backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    out = model(ids, labels=ids, **{request_maps: True})
    out.loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}

    assert model.training
    assert grads.keys() == expected.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grad, expected[name])
    maps = out.attentions if request_maps == "output_attentions" else out.interactions
    assert maps and not any(m.requires_grad for m in maps.values())


@pytest.mark.skipif(not (torch.cuda.is_available() and modeling_minerva._HAS_FLASH),
                    reason="needs CUDA and flash-attn")
def test_flash_path_gradients_go_through_the_rotation(monkeypatch):
    """flash-attn's in-place rotary is invisible to autograd; compare q/k grads with SDPA."""
    model = _model("cuda").train()
    ids = torch.randint(4, 34, (2, 12), device="cuda")

    def qk_grads():
        model.zero_grad(set_to_none=True)
        model(ids, labels=ids).loss.backward()
        rows = 2 * model.config.dim  # wqkv's output rows are q, then k, then v
        return [layer.attention.wqkv.weight.grad[:rows].float() for layer in model.minerva.encoder.layers]

    flash = qk_grads()
    monkeypatch.setattr(modeling_minerva, "_HAS_FLASH", False)
    for got, ref in zip(flash, qk_grads()):
        torch.testing.assert_close(got, ref, rtol=0, atol=0.05 * ref.abs().max().item())


def test_inference_runs_the_encoder_once():
    """Under no_grad, logits and maps share one encoder pass."""
    model = _model()
    calls = []
    for layer in model.minerva.encoder.layers:
        original = layer.forward
        layer.forward = (lambda *a, _o=original, **k: (calls.append(1), _o(*a, **k))[1])

    with torch.no_grad():
        model(torch.randint(4, 34, (1, 12)), output_interactions=True)
    assert len(calls) == DEPTH
