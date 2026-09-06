"""Tests for minerva.backbones: per-backbone token groups and layer surgery."""

from __future__ import annotations

import pytest

from minerva.backbones import (
    TokenGroup,
    check_token_groups,
    get_backbone,
)

# RiNALMo is nucleotide-only and upper-case; it has no U (its alphabet maps U to T).
MINERVA_VOCAB = list("atgcn") + list("ACDEFGHIKLMNPQRSTVWY") + ["<+>", "<->", "<mask>"]
RINALMO_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"] + list("ACGTIRYKMSWBDHVN-")


class _Tokenizer:
    def __init__(self, chars):
        self._vocab = {c: i for i, c in enumerate(chars)}

    def get_vocab(self):
        return dict(self._vocab)


def test_minerva_groups_nucleotides_and_protein_separately():
    groups = get_backbone("minerva").token_groups(_Tokenizer(MINERVA_VOCAB))
    by_name = {g.name: g for g in groups}
    assert by_name["nucleotide"].alphabet_size == 4
    assert by_name["protein"].alphabet_size == 20
    check_token_groups(groups)


def test_rinalmo_has_no_protein_group():
    groups = get_backbone("rinalmo").token_groups(_Tokenizer(RINALMO_VOCAB))
    assert [g.name for g in groups] == ["nucleotide"]
    check_token_groups(groups)


def test_rna_bases_are_never_scaled_as_protein():
    """Regression test: hardcoded lower-case nucleotide / upper-case AA lists used
    to put upper-case RNA bases in the protein group, scaling them by log(20)."""
    tokenizer = _Tokenizer(RINALMO_VOCAB)
    groups = get_backbone("rinalmo").token_groups(tokenizer)
    vocab = tokenizer.get_vocab()

    scale_of = {tid: g.alphabet_size for g in groups for tid in g.token_ids}
    bases = [vocab[c] for c in "ACGTN"]
    assert all(scale_of.get(tid) == 4 for tid in bases)
    # every base shares one scale -- the specific failure was A/C/G/T/N and the
    # rest of the alphabet landing on different ones
    assert len({scale_of.get(vocab[c]) for c in "ACGTIRYKMSWBDHVN-"}) == 1


def test_check_token_groups_rejects_overlap():
    groups = [TokenGroup("nucleotide", (1, 2), 4), TokenGroup("protein", (2, 3), 20)]
    with pytest.raises(ValueError, match="both 'nucleotide' and 'protein'"):
        check_token_groups(groups)


def test_missing_vocab_entries_are_skipped():
    """A backbone spec should survive a tokenizer that lacks some characters."""
    groups = get_backbone("rinalmo").token_groups(_Tokenizer(list("ACGT")))
    assert len(groups) == 1 and len(groups[0].token_ids) == 4


def test_unknown_backbone_names_the_known_ones():
    with pytest.raises(ValueError, match="minerva"):
        get_backbone("nope")


def _logits_for(target_ids, vocab_size, confident_id=None):
    """Uniform logits, optionally peaked on `confident_id` for every position."""
    import torch

    logits = torch.zeros(1, len(target_ids), vocab_size)
    if confident_id is not None:
        logits[:, :, confident_id] = 10.0
    return logits, torch.tensor([target_ids])


def test_grouped_loss_scales_each_group_by_its_alphabet():
    import math

    import torch

    from minerva.losses import grouped_mlm_loss

    # one nucleotide token (id 0) and one protein token (id 1), both equally wrong
    groups = [TokenGroup("nucleotide", (0,), 4), TokenGroup("protein", (1,), 20)]
    logits, labels = _logits_for([0, 1], vocab_size=8)
    loss, metrics = grouped_mlm_loss(logits, labels, groups)

    # uniform logits => CE == log(vocab); after scaling, each group reports
    # log(8)/log(alphabet_size)
    assert metrics["nucleotide_loss"] == pytest.approx(math.log(8) / math.log(4), rel=1e-5)
    assert metrics["protein_loss"] == pytest.approx(math.log(8) / math.log(20), rel=1e-5)
    assert metrics["n_nucleotide_tokens"] == 1 and metrics["n_protein_tokens"] == 1
    assert torch.isfinite(loss)


def test_unclaimed_tokens_take_the_narrowest_scale_not_dropped():
    import torch

    from minerva.losses import grouped_mlm_loss

    groups = [TokenGroup("nucleotide", (0,), 4)]
    logits, labels = _logits_for([0, 5], vocab_size=8)   # id 5 belongs to no group
    loss, metrics = grouped_mlm_loss(logits, labels, groups)
    assert metrics["n_other_tokens"] == 1
    assert metrics["other_loss"] == pytest.approx(metrics["nucleotide_loss"], rel=1e-5)
    assert torch.isfinite(loss)


def test_ignore_index_is_excluded():
    from minerva.losses import grouped_mlm_loss

    groups = [TokenGroup("nucleotide", (0,), 4)]
    logits, labels = _logits_for([0, -100], vocab_size=8)
    _, metrics = grouped_mlm_loss(logits, labels, groups)
    assert metrics["n_nucleotide_tokens"] == 1
    assert metrics["n_other_tokens"] == 0


def test_no_groups_falls_back_to_plain_cross_entropy():
    import torch
    import torch.nn.functional as F

    from minerva.losses import grouped_mlm_loss

    logits, labels = _logits_for([0, 1, 2], vocab_size=8)
    loss, metrics = grouped_mlm_loss(logits, labels, [])
    expected = F.cross_entropy(logits.view(-1, 8), labels.view(-1), ignore_index=-100)
    assert torch.allclose(loss, expected)
    assert metrics == {}


def test_rna_bases_share_one_scale_end_to_end():
    """The bug, at the loss level: every base must contribute identically."""
    import torch

    from minerva.losses import grouped_mlm_loss

    tokenizer = _Tokenizer(RINALMO_VOCAB)
    groups = get_backbone("rinalmo").token_groups(tokenizer)
    vocab = tokenizer.get_vocab()

    per_base = []
    for base in "ACGTN":
        logits, labels = _logits_for([vocab[base]], vocab_size=len(RINALMO_VOCAB))
        _, metrics = grouped_mlm_loss(logits, labels, groups)
        per_base.append(metrics["nucleotide_loss"])
    assert len(set(round(v, 9) for v in per_base)) == 1


def test_lora_targets_exist_on_the_backbone():
    """Guessed target names only fail once PEFT runs, well into a training job."""
    pytest.importorskip("torch")
    import torch.nn as nn

    from minerva.modeling_rinalmo import RiNALMoMinervaConfig, RiNALMoMinervaForMaskedLM

    model = RiNALMoMinervaForMaskedLM(
        RiNALMoMinervaConfig(embed_dim=32, num_blocks=1, num_heads=2)
    )
    present = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    assert set(get_backbone("rinalmo").lora_targets) <= present
