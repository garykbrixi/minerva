"""RiNALMo with Minerva interaction heads.

The backbone is RiNALMo (vendored, Apache-2.0); the heads and their plumbing are
shared with Minerva via InteractionHeads.
"""

from typing import Dict, List, Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel

from .interaction_heads import (
    AttentionHeadExtractor,
    InteractionHeads,
    PyTorchLinearHead,
)
try:
    from .vendor_rinalmo import RiNALMo
except ImportError:                      # flattened layout inside an HF checkpoint
    from vendor_rinalmo import RiNALMo

# RiNALMo's alphabet: 5 special tokens then the RNA/IUPAC set. U has no token --
# upstream encode() maps U to T.
SPECIAL_TOKENS = ("<cls>", "<pad>", "<eos>", "<unk>", "<mask>")
RNA_TOKENS = tuple("ACGTIRYKMSWBDHVN-")
PAD_TOKEN_ID = SPECIAL_TOKENS.index("<pad>")
MASK_TOKEN_ID = SPECIAL_TOKENS.index("<mask>")

SIZES = {
    "giga": {"embed_dim": 1280, "num_blocks": 33, "num_heads": 20},
    "mega": {"embed_dim": 640, "num_blocks": 30, "num_heads": 20},
    "micro": {"embed_dim": 480, "num_blocks": 12, "num_heads": 20},
}


class _AttrDict(dict):
    """The vendored code reads config both as cfg.model.x and cfg.model["x"]."""
    __getattr__ = dict.__getitem__


class RiNALMoMinervaConfig(PretrainedConfig):
    model_type = "rinalmo_minerva"

    def __init__(
        self,
        embed_dim: int = 1280,
        num_blocks: int = 33,
        num_heads: int = 20,
        alphabet_size: int = len(SPECIAL_TOKENS) + len(RNA_TOKENS),
        use_rot_emb: bool = True,
        attn_qkv_bias: bool = False,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        transition_dropout: float = 0.0,
        transition_factor: int = 4,
        token_dropout: bool = True,
        mask_ratio: float = 0.15,
        mask_tkn_prob: float = 0.8,
        use_flash_attn: bool = False,
        linear_heads_config: Optional[Dict[str, dict]] = None,
        **kwargs,
    ):
        # RiNALMo's LM head is a 2-layer MLP, not tied to the embeddings. Left at
        # the default, HF aliases them and safetensors then refuses to save.
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.alphabet_size = alphabet_size
        self.use_rot_emb = use_rot_emb
        self.attn_qkv_bias = attn_qkv_bias
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.transition_dropout = transition_dropout
        self.transition_factor = transition_factor
        self.token_dropout = token_dropout
        self.mask_ratio = mask_ratio
        self.mask_tkn_prob = mask_tkn_prob
        self.use_flash_attn = use_flash_attn
        self.linear_heads_config = linear_heads_config
        self.auto_map = {
            "AutoConfig": "modeling_rinalmo.RiNALMoMinervaConfig",
            "AutoModelForMaskedLM": "modeling_rinalmo.RiNALMoMinervaForMaskedLM",
        }

    @classmethod
    def for_size(cls, name: str, **kwargs):
        return cls(**SIZES[name], **kwargs)

    def to_rinalmo_config(self):
        """The nested config the vendored RiNALMo expects, without ml_collections."""
        return _AttrDict(model=_AttrDict(
            embedding=_AttrDict(
                num_embeddings=self.alphabet_size,
                embedding_dim=self.embed_dim,
                padding_idx=PAD_TOKEN_ID,
            ),
            token_dropout=_AttrDict(
                active=self.token_dropout,
                mask_ratio=self.mask_ratio,
                mask_tkn_prob=self.mask_tkn_prob,
                mask_tkn_idx=MASK_TOKEN_ID,
                pad_tkn_idx=PAD_TOKEN_ID,
            ),
            transformer=_AttrDict(
                embed_dim=self.embed_dim,
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                use_rot_emb=self.use_rot_emb,
                attn_qkv_bias=self.attn_qkv_bias,
                attention_dropout=self.attention_dropout,
                transition_dropout=self.transition_dropout,
                residual_dropout=self.residual_dropout,
                transition_factor=self.transition_factor,
                use_flash_attn=self.use_flash_attn,
            ),
            lm_mask_head=_AttrDict(embed_dim=self.embed_dim, alphabet_size=self.alphabet_size),
        ))


class RiNALMoMinervaPreTrainedModel(PreTrainedModel):
    config_class = RiNALMoMinervaConfig
    base_model_prefix = "rinalmo"
    supports_gradient_checkpointing = True


class RiNALMoMinervaForMaskedLM(InteractionHeads, RiNALMoMinervaPreTrainedModel):
    # RiNALMo is RNA-only, so no protein head. Its heads are trained on layers
    # 27-32 (see the covarval regression grid), giving a single depth.
    interaction_tasks = ("base_pairing", "repeat")
    head_depths = {6: ""}

    def __init__(self, config: RiNALMoMinervaConfig):
        super().__init__(config)
        self.rinalmo = RiNALMo(config.to_rinalmo_config())
        self.num_layers = config.num_blocks

        self.linear_heads = nn.ModuleDict()
        for name, spec in (config.linear_heads_config or {}).items():
            if spec.get("type") == "attention":
                self.linear_heads[name] = AttentionHeadExtractor(
                    layer_idx=spec["layer_idx"], head_idx=spec["head_idx"],
                    apply_symmetrize=spec.get("apply_symmetrize", True),
                    apply_apc=spec.get("apply_apc", False),
                )
            else:
                self.linear_heads[name] = PyTorchLinearHead(
                    input_dim=spec["input_dim"],
                    apply_symmetrize=spec.get("apply_symmetrize", True),
                    apply_apc=spec.get("apply_apc", False),
                    layers=spec.get("layers"),
                )

    # -- HF plumbing --------------------------------------------------------
    @property
    def lm_head(self):
        return self.rinalmo.lm_mask_head

    def get_input_embeddings(self):
        return self.rinalmo.embedding

    def set_input_embeddings(self, value):
        self.rinalmo.embedding = value

    # -- attention capture --------------------------------------------------
    def get_attention_maps(
        self,
        input_ids: torch.Tensor,
        layers: List[int],
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_state: bool = False,
        output_hidden_states: bool = False,
    ):
        """Attention for `layers` only, as {layer: [B, heads, S, S]}.

        Runs the blocks directly rather than RiNALMo.forward(need_attn_weights=True),
        which stacks all 33 layers -- 41 GiB at L=4096, enough to OOM an 80 GB card.
        Values are identical; only the retained layers differ.
        """
        model = self.rinalmo
        transformer = model.transformer
        wanted = set(layers)

        pad_mask = input_ids.eq(PAD_TOKEN_ID)
        if attention_mask is not None:
            pad_mask = pad_mask | ~attention_mask.bool()
        # The flash path wants "True means keep"; the eager path wants the pad mask.
        key_padding_mask = (
            torch.logical_not(pad_mask) if transformer.use_flash_attn else pad_mask
        )

        maps, hidden = {}, []
        x = model.embedding(input_ids)
        x = model.token_dropout(x, input_ids)
        for layer_idx, block in enumerate(transformer.blocks):
            x, attn = block(
                x, key_padding_mask=key_padding_mask, need_attn_weights=layer_idx in wanted
            )
            if layer_idx in wanted:
                maps[layer_idx] = attn
            del attn
            if output_hidden_states:
                hidden.append(x)
        x = transformer.final_layer_norm(x)

        if not return_hidden_state:
            return maps
        return (maps, x, hidden) if output_hidden_states else (maps, x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        attention_layers: Optional[List[int]] = None,
        output_interactions: bool = False,
        interaction_layers: int = 6,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        head_to_task = (
            self._interaction_head_names(interaction_layers) if output_interactions else {}
        )
        needed = set(attention_layers or [])
        for name in head_to_task:
            head = self.linear_heads[name]
            needed.update(head.layers or self._default_contact_layers())

        if needed:
            # One pass: the same hidden state feeds the LM head, so requesting
            # interactions does not cost a second forward.
            maps, hidden = self.get_attention_maps(
                input_ids, sorted(needed), attention_mask, return_hidden_state=True
            )
            logits = self.lm_head(hidden)
        else:
            out = self.rinalmo(input_ids)
            logits, hidden, maps = out["logits"], out["representation"], {}

        interactions = None
        if output_interactions:
            # Heads were trained on maps with CLS/EOS stripped, so crop before
            # applying them; interaction maps are over the sequence, not tokens.
            cropped = {i: m[:, :, 1:-1, 1:-1] for i, m in maps.items()}
            interactions = self.predict_interactions(
                cropped, head_to_task, input_ids.shape[0], input_ids.shape[1] - 2
            )

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

        attentions = (
            {i: maps[i] for i in attention_layers if i in maps}
            if output_attentions and attention_layers
            else (maps if output_attentions else None)
        )
        out = MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
            attentions=attentions,
        )
        out.interactions = interactions
        return out


AutoConfig.register(RiNALMoMinervaConfig.model_type, RiNALMoMinervaConfig)
AutoModelForMaskedLM.register(RiNALMoMinervaConfig, RiNALMoMinervaForMaskedLM)
