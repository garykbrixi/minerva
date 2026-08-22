"""Per-backbone details the finetuning pipeline needs to know."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

# Amino acids in the mixed-modality vocab, upper-case.
_AA = list("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class TokenGroup:
    """Token ids sharing a loss scale of log(alphabet_size)."""
    name: str
    token_ids: tuple
    alphabet_size: int


@dataclass(frozen=True)
class Backbone:
    name: str
    modality: str          # "mixed" (protein + DNA) or "nucleotide"
    lora_targets: tuple
    _layers_path: tuple    # attribute path to the transformer layer list
    _embed_path: tuple
    _alphabet: tuple       # (chars, alphabet_size) per token group

    def token_groups(self, tokenizer) -> List[TokenGroup]:
        """Token groups for this vocab. Characters absent from it are skipped."""
        vocab = tokenizer.get_vocab()
        groups = []
        for group_name, chars, size in self._alphabet:
            ids = tuple(vocab[c] for c in chars if c in vocab)
            if ids:
                groups.append(TokenGroup(group_name, ids, size))
        return groups

    def resolve(self, model, path: tuple):
        for attr in path:
            model = getattr(model, attr)
        return model

    def freeze_layers(self, model, n_trainable: int) -> tuple:
        """Freeze embeddings and all but the last `n_trainable` layers.

        LM head stays trainable. Returns (n_frozen, n_layers).
        """
        layers = self.resolve(model, self._layers_path)
        n_frozen = max(len(layers) - n_trainable, 0)

        for param in self.resolve(model, self._embed_path).parameters():
            param.requires_grad = False
        for layer in layers[:n_frozen]:
            for param in layer.parameters():
                param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = True
        return n_frozen, len(layers)

    def enable_grad_checkpointing(self, model) -> bool:
        """Set checkpointing on the encoder directly.

        Trainer's own hook trips over custom architectures and PEFT wrappers, so
        callers should disable it when this returns True.
        """
        encoder = self.resolve(model, self._layers_path[:-1])
        if not hasattr(encoder, "gradient_checkpointing"):
            return False
        encoder.gradient_checkpointing = True
        return True


MINERVA = Backbone(
    name="minerva",
    modality="mixed",
    lora_targets=("wqkv", "wo", "w1", "w2", "w3"),
    _layers_path=("minerva", "encoder", "layers"),
    _embed_path=("minerva", "tok_embeddings"),
    _alphabet=(
        ("nucleotide", tuple("atgcn"), 4),
        ("protein", tuple(_AA), 20),
    ),
)

# RiNALMo is nucleotide-only and upper-case, and has no U token -- its alphabet
# maps U to T. IUPAC ambiguity codes share the nucleotide loss scale.
RINALMO = Backbone(
    name="rinalmo",
    modality="nucleotide",
    lora_targets=("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"),
    _layers_path=("rinalmo", "encoder", "layers"),
    _embed_path=("rinalmo", "tok_embeddings"),
    _alphabet=(
        ("nucleotide", tuple("ACGTIRYKMSWBDHVN-"), 4),
    ),
)

_REGISTRY = {b.name: b for b in (MINERVA, RINALMO)}


def get_backbone(name: str) -> Backbone:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown backbone {name!r}; known: {sorted(_REGISTRY)}"
        ) from None


def check_token_groups(groups: Sequence[TokenGroup]) -> None:
    """Raise if a token id lands in two groups; otherwise it is counted twice."""
    seen: Dict[int, str] = {}
    for group in groups:
        for tid in group.token_ids:
            if tid in seen:
                raise ValueError(
                    f"token id {tid} is in both {seen[tid]!r} and {group.name!r}"
                )
            seen[tid] = group.name
