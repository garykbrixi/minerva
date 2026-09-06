"""
Transformers-native Minerva masked LM with flash-attn path.

Key features:
- MinervaConfig / MinervaForMaskedLM usable via AutoConfig/AutoModelForMaskedLM
- Supports packed sequences via cu_seqlens + max_seq_len
- Supports regression heads for contact prediction (sklearn LogisticRegression)
- Uses flash-attn when available; falls back to torch SDPA otherwise
"""

import torch
from torch import nn
from typing import Optional, Tuple, Union, List, Dict
from dataclasses import dataclass
import contextlib
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, ModelOutput
from transformers import AutoConfig, AutoModelForMaskedLM
try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
    _HAS_FLASH_ROTARY = True
except ImportError:
    apply_rotary_emb_func = None
    _HAS_FLASH_ROTARY = False

# Relative sibling import — bundled with the HF checkpoint so trust_remote_code
# picks it up automatically. Works both pip-installed (resolves to
# minerva.jacobian) and HF-cached (resolves to transformers_modules.<repo>.jacobian).
from .jacobian import (
    FingerprintResult,
    classify_jacobian_contacts,
    compute_fingerprints,
    jac_to_contact,
)
# Direct import of constants (also a jacobian.py dependency). Re-exported here for
# convenience AND to guarantee constants.py is bundled: for local-directory
# trust_remote_code loads, HF copies only the DIRECT relative imports of this
# modeling file, not transitive ones, so constants.py must be imported here too.
from .interaction_heads import (  # noqa: F401  (re-exported)
    AttentionHeadExtractor,
    InteractionHeads,
    PyTorchLinearHead,
    apc,
    symmetrize,
)
from .constants import (  # noqa: F401
    MINERVA_AA_ORDER,
    MINERVA_NUCLEOTIDE_TOKENS,
    MINERVA_AMINO_ACID_TOKENS,
    get_minerva_token_ids,
    MINERVA_BP_CUTOFF,
    MINERVA_REPEAT_CUTOFF,
    MINERVA_PROTEIN_CUTOFF,
    BASE_PAIR_ARGMAX,
    REPEAT_ARGMAX,
    MINERVA_BP_FINGERPRINT,
    MINERVA_BP_FORWARD_FINGERPRINT,
    MINERVA_BP_REVERSE_FINGERPRINT,
    MINERVA_REPEAT_FINGERPRINT,
    MINERVA_PROTEIN_FINGERPRINT,
)


# ----------------
# Output Dataclass
# ----------------


@dataclass
class MinervaForMaskedLMOutput(ModelOutput):
    """
    Output type for MinervaForMaskedLM with extended features.
    
    Args:
        loss: Masked language modeling loss (if labels provided).
        logits: Prediction scores of the language modeling head [batch, seq_len, vocab_size].
        hidden_states: Hidden states from requested layers.
            - If output_hidden_states=True: tuple of all layers (HuggingFace compat)
            - If output_hidden_states=[layer_indices]: dict {layer_idx: tensor}
        attentions: Attention maps from requested layers {layer_idx: [batch, heads, seq, seq]}.
        contact_predictions: Backward-compatible contact predictions by head name.
        interactions: Interaction maps by task name: base_pairing, protein, repeat.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Union[Tuple[torch.FloatTensor, ...], Dict[int, torch.FloatTensor]]] = None
    attentions: Optional[Dict[int, torch.FloatTensor]] = None
    contact_predictions: Optional[Dict[str, torch.FloatTensor]] = None
    interactions: Optional[Dict[str, torch.FloatTensor]] = None


# ----------------
# Helper functions for attention processing
# ----------------

def collate_sequences(tokenizer, sequences, device=None):
    """Tokenize and pad variable-length sequences into a batch.

    Args:
        tokenizer: Tokenizer with encode() method.
        sequences: List of input sequence strings.
        device: Optional torch device to place tensors on.

    Returns:
        input_ids: [batch, max_len] padded tensor of token IDs.
        attention_mask: [batch, max_len] bool tensor (True = real token).
    """
    encoded = [tokenizer.encode(s) for s in sequences]
    max_len = max(len(e) for e in encoded)
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is None:
        pad_id = 0
    input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(encoded), max_len, dtype=torch.bool)
    for i, enc in enumerate(encoded):
        input_ids[i, :len(enc)] = torch.tensor(enc, dtype=torch.long)
        attention_mask[i, :len(enc)] = True
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


# ----------------
# PyTorch Linear Head for Regression
# ----------------
# Flash-attention is optional. Without it we fall back to torch SDPA
# (torch.nn.functional.scaled_dot_product_attention), which needs no external
# install and runs on CPU too. The SDPA path is numerically ~equivalent (bf16
# rounding) but only supports the unpacked path (single/uniform-length batches);
# packed varlen (cu_seqlens) still requires flash-attn.
try:
    from flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_varlen_func
    _FLASH_ATTENTION_VERSION = 3
    _HAS_FLASH = True
except Exception:
    try:
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
        _FLASH_ATTENTION_VERSION = 2
        _HAS_FLASH = True
    except Exception:
        _FLASH_ATTENTION_VERSION = None
        _HAS_FLASH = False
try:
    from flash_attn.bert_padding import pad_input, unpad_input
except Exception:
    pad_input = unpad_input = None

# ----------------
# Config
# ----------------
class MinervaConfig(PretrainedConfig):
    model_type = "minerva"

    def __init__(
        self,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        vocab_size: int = 37,
        norm_eps: float = 1e-5,
        swiglu_multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        qk_norm: bool = False,
        base: int = 10000,
        torch_dtype: str = "float32",
        # The input embeddings and output projection are trained independently
        # (not tied), so disable HF's default weight tying — otherwise it creates
        # an unused phantom `lm_head.weight` that complicates load/save.
        tie_word_embeddings: bool = False,
        # Persistent regression heads (saved with checkpoint)
        linear_heads_config: Optional[Dict[str, dict]] = None,
        **kwargs,
    ):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.vocab_size = vocab_size
        self.norm_eps = norm_eps
        self.swiglu_multiple_of = swiglu_multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.qk_norm = qk_norm
        self.base = base
        self.linear_heads_config = linear_heads_config
        super().__init__(torch_dtype=torch_dtype, tie_word_embeddings=tie_word_embeddings, **kwargs)

def rmsnorm_func(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    # Compute RMS variance in fp32. In fp16, late-layer activations can exceed
    # sqrt(65504), so squaring in fp16 overflows to inf and collapses the layer.
    hidden_states_f = hidden_states.float()
    variance = hidden_states_f.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states_f * torch.rsqrt(variance + float(variance_epsilon))
    return (weight.float() * hidden_states).to(input_dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        # Keep epsilon as a Python float rather than a non-persistent tensor
        # buffer. Transformers 5 can instantiate modules through a meta-device
        # path in from_pretrained; non-persistent buffers are not restored from
        # the checkpoint and can be left with uninitialized memory.
        self.variance_epsilon = float(eps)
    def forward(self, hidden_states):
        return rmsnorm_func(hidden_states, self.weight, self.variance_epsilon)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, interleaved=False, scale_base=None, pos_idx_in_fp32=True, device=None):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Keep rotary constants out of non-persistent buffers. Transformers 5 can
        # leave non-persistent buffers uninitialized during from_pretrained; these
        # small fp32 tensors are cheap to recompute when refreshing the cache.
        self.interleaved = interleaved
        self.scale_base = scale_base
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _compute_scale(self, device=None):
        if self.scale_base is None:
            return None
        return (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) + 0.4 * self.dim) / (1.4 * self.dim)

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            t_dtype = torch.float32
            t = torch.arange(seqlen, device=device, dtype=t_dtype)
            inv_freq = self._compute_inv_freq(device=device)
            freqs = torch.outer(t, inv_freq)
            scale_base = self._compute_scale(device=device)
            if scale_base is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (torch.arange(seqlen, dtype=scale_base.dtype, device=device) - seqlen // 2) / self.scale_base
                scale = scale_base ** power.unsqueeze(1)
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seqlen_offset: Union[int, torch.Tensor] = 0,
                cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None):
        if not _HAS_FLASH_ROTARY or q.device.type != "cuda":
            if cu_seqlens is not None or max_seqlen is not None or not (isinstance(seqlen_offset, int) and seqlen_offset == 0):
                raise RuntimeError(
                    "Packed or offset rotary embeddings require flash-attn's CUDA rotary kernel. "
                    "Use unpacked sequences with seqlen_offset=0, or run on CUDA with flash-attn installed."
                )
            return self.apply_torch(q, k)

        if cu_seqlens is not None:
            assert max_seqlen is not None
        seqlen = q.shape[1] if max_seqlen is None else max_seqlen
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=q.device, dtype=q.dtype)
        q = apply_rotary_emb_func(
            q,
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            inplace=True,
            seqlen_offsets=seqlen_offset,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if self.scale_base is None:
            k = apply_rotary_emb_func(
                k, self._cos_cached, self._sin_cached, interleaved=self.interleaved,
                inplace=True, seqlen_offsets=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
        else:
            k = apply_rotary_emb_func(
                k, self._cos_k_cached, self._sin_k_cached, interleaved=self.interleaved,
                inplace=True, seqlen_offsets=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
        return q, k

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_torch(self, q, k):
        """Pure-torch rotary for [B, S, H, D] q/k (no flash-attn).

        Matches flash_attn.apply_rotary_emb_func with interleaved=False
        (GPT-NeoX rotate-half): out = x*cos + rotate_half(x)*sin.
        """
        assert not self.interleaved, "torch rotary fallback assumes interleaved=False"
        S = q.shape[1]
        self._update_cos_sin_cache(S, device=q.device, dtype=q.dtype)

        def _cs(cos, sin):
            cos = torch.cat([cos[:S], cos[:S]], dim=-1)[None, :, None, :]  # [1,S,1,D]
            sin = torch.cat([sin[:S], sin[:S]], dim=-1)[None, :, None, :]
            return cos, sin

        cos_q, sin_q = _cs(self._cos_cached, self._sin_cached)
        q = q * cos_q + self._rotate_half(q) * sin_q
        if self.scale_base is None:
            k = k * cos_q + self._rotate_half(k) * sin_q
        else:
            cos_k, sin_k = _cs(self._cos_k_cached, self._sin_k_cached)
            k = k * cos_k + self._rotate_half(k) * sin_k
        return q, k

class Attention(nn.Module):
    def __init__(self, config: MinervaConfig):
        super().__init__()
        self.n_heads = config.heads
        self.head_dim = config.dim // config.heads
        self.qk_norm = getattr(config, "qk_norm", False)
        self.base = getattr(config, "base", 10000)
        self.wqkv = nn.Linear(config.dim, self.n_heads * self.head_dim * 3, bias=False)
        self.wo = nn.Linear(config.heads * self.head_dim, config.dim, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, self.base)
        if self.qk_norm:
            self.q_norm = RMSNorm(dim=self.head_dim)
            self.k_norm = RMSNorm(dim=self.head_dim)

    def _forward_varlen(self, x: torch.Tensor, cu_seqlens: Optional[torch.Tensor], max_seq_len: Optional[torch.Tensor]) -> torch.Tensor:
        total_seqlen, h_size = x.shape
        qkv = self.wqkv(x)
        q, k, v = torch.split(qkv, self.n_heads * self.head_dim, dim=-1)
        q = q.view(total_seqlen, self.n_heads, self.head_dim)
        k = k.view(total_seqlen, self.n_heads, self.head_dim)
        v = v.view(total_seqlen, self.n_heads, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(q, k, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)
        if _FLASH_ATTENTION_VERSION == 3:
            output = flash_attn_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens, max_seq_len, max_seq_len, dropout_p=0.0, causal=False
            )
        else:
            qkv = torch.stack([q, k, v], dim=1)
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len, dropout_p=0.0, causal=False
            )
        output = output.view(total_seqlen, h_size)
        return self.wo(output)

    def _forward_sdpa(self, x: torch.Tensor) -> torch.Tensor:
        """Torch-SDPA attention for unpacked [B, S, dim] input (no flash-attn)."""
        bsz, seqlen, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = torch.split(qkv, self.n_heads * self.head_dim, dim=-1)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb.apply_torch(q, k)
        # [B, S, H, D] -> [B, H, S, D] for SDPA
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.n_heads * self.head_dim)
        return self.wo(out)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        if not _HAS_FLASH or x.device.type != "cuda":
            if cu_seqlens is not None:
                raise RuntimeError(
                    "Packed (cu_seqlens) attention requires CUDA flash-attn. "
                    "Use unpadded/uniform-length batches for the SDPA fallback."
                )
            return self._forward_sdpa(x)
        reshape_back = False
        if cu_seqlens is None:
            bsz, seqlen, _ = x.shape
            cu_seqlens = torch.arange(0, (bsz + 1) * seqlen, step=seqlen, dtype=torch.int32, device=x.device)
            x = x.view(bsz * seqlen, -1)
            max_seq_len = seqlen if max_seq_len is None else max_seq_len
            reshape_back = True
        output = self._forward_varlen(x, cu_seqlens, max_seq_len)
        if reshape_back:
            output = output.view(bsz, seqlen, -1)
        return output

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: MinervaConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=4 * config.dim,
            multiple_of=config.swiglu_multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, cu_seqlens=None, max_seq_len=None):
        r = self.attention(self.attention_norm(x), cu_seqlens=cu_seqlens, max_seq_len=max_seq_len)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out

class TransformerLayers(nn.Module):
    def __init__(self, config: MinervaConfig):
        super().__init__()
        self.layers = torch.nn.ModuleList([TransformerBlock(config=config) for _ in range(config.depth)])
        self.gradient_checkpointing = False

    def _run_layers(self, x, cu_seqlens, max_seq_len, return_all_hiddens, hidden_layers_set, hiddens):
        """Run x through every transformer block, collecting requested hidden states.

        Mutates `hiddens` in place (append for all-hiddens, index-assign for the
        specific-layers dict) and returns the final hidden state.
        """
        for layer_idx, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cu_seqlens, max_seq_len, use_reentrant=False
                )
            else:
                x = layer(x, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len)
            if return_all_hiddens:
                hiddens.append(x)
            elif layer_idx in hidden_layers_set:
                hiddens[layer_idx] = x
        return x

    def forward(
        self,
        x: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        return_all_hiddens: bool = False,
        hidden_layers_to_return: Optional[List[int]] = None,
    ):
        # Use list for all hiddens (HF compat), dict for specific layers (efficient)
        hiddens = [] if return_all_hiddens else {}
        hidden_layers_set = set(hidden_layers_to_return) if hidden_layers_to_return else set()
        
        if cu_seqlens is None and attention_mask is not None:
            if x.dim() != 3:
                raise ValueError("Padded sequences must be 3D")
            batch_size, seq_len = x.shape[:2]
            should_unpad = not attention_mask.all()
            if should_unpad and (
                not _HAS_FLASH or x.device.type != "cuda" or unpad_input is None or pad_input is None
            ):
                raise RuntimeError(
                    "Padded variable-length batches require CUDA flash-attn (unpad/pack). "
                    "Without it, run one sequence at a time or use uniform-length batches."
                )
            if should_unpad:
                x, indices, cu_seqlens, max_seq_len, _ = unpad_input(x, attention_mask)
            else:
                indices = None
            x = self._run_layers(x, cu_seqlens, max_seq_len, return_all_hiddens, hidden_layers_set, hiddens)
            if should_unpad:
                x = pad_input(x, indices, batch_size, seq_len)
                if return_all_hiddens:
                    hiddens = [pad_input(h, indices, batch_size, seq_len) for h in hiddens]
                elif hidden_layers_set:
                    hiddens = {idx: pad_input(h, indices, batch_size, seq_len) for idx, h in hiddens.items()}
        elif cu_seqlens is not None:
            if x.dim() != 2:
                raise ValueError("Packed sequences must be 2D")
            x = self._run_layers(x, cu_seqlens, max_seq_len, return_all_hiddens, hidden_layers_set, hiddens)
        else:
            if x.dim() != 3:
                raise ValueError("Unpadded sequences must be 3D")
            x = self._run_layers(x, None, None, return_all_hiddens, hidden_layers_set, hiddens)
        if return_all_hiddens or hidden_layers_set:
            return x, hiddens
        return x

class MinervaPreTrainedModel(PreTrainedModel):
    config_class = MinervaConfig
    base_model_prefix = "minerva"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TransformerLayers):
            module.gradient_checkpointing = value

    def _init_weights(self, module, initializer_range=0.02):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

class MinervaModel(MinervaPreTrainedModel):
    def __init__(self, config: MinervaConfig):
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.encoder = TransformerLayers(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        output_hidden_states: Optional[Union[bool, List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # Determine if we need all hidden states or just specific layers
        need_all_hidden_states = output_hidden_states is True
        hidden_layers_to_return = output_hidden_states if isinstance(output_hidden_states, list) else None
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        h = self.tok_embeddings(input_ids)
        if cu_seqlens is not None:
            h = h.squeeze(0)

        encoder_outputs = self.encoder(
            h,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seq_len=max_seq_len,
            return_all_hiddens=need_all_hidden_states,
            hidden_layers_to_return=hidden_layers_to_return,
        )
        if need_all_hidden_states or hidden_layers_to_return:
            sequence_output, all_hidden_states = encoder_outputs
            # all_hidden_states is either a list (if need_all_hidden_states) or dict (if hidden_layers_to_return)
        else:
            sequence_output = encoder_outputs
            all_hidden_states = None

        if not return_dict:
            return (sequence_output, all_hidden_states)
        return BaseModelOutput(last_hidden_state=sequence_output, hidden_states=all_hidden_states, attentions=None)

class MinervaLMHead(nn.Module):
    def __init__(self, config: MinervaConfig):
        super().__init__()
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.proj_output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, features):
        return self.proj_output(self.norm(features))

class MinervaForMaskedLM(InteractionHeads, MinervaPreTrainedModel):
    def __init__(self, config: MinervaConfig):
        super().__init__(config)
        self.minerva = MinervaModel(config)
        self.lm_head = MinervaLMHead(config)
        # Initialize ModuleDict for multiple regression/attention heads
        self.linear_heads = nn.ModuleDict()

        # Reconstruct head modules from config (weights filled by load_state_dict)
        heads_cfg = getattr(config, "linear_heads_config", None) or {}
        for name, cfg in heads_cfg.items():
            head_type = cfg.get("type", "linear")
            if head_type == "linear":
                head = PyTorchLinearHead(
                    input_dim=cfg["input_dim"],
                    apply_symmetrize=cfg.get("apply_symmetrize", False),
                    apply_apc=cfg.get("apply_apc", False),
                    layers=cfg.get("layers"),
                )
            elif head_type == "attention":
                head = AttentionHeadExtractor(
                    layer_idx=cfg["layer_idx"],
                    head_idx=cfg["head_idx"],
                    apply_symmetrize=cfg.get("apply_symmetrize", True),
                    apply_apc=cfg.get("apply_apc", True),
                )
            else:
                raise ValueError(f"Unknown head type '{head_type}' for head '{name}'")
            self.linear_heads[name] = head

        # Fingerprint matrices baked into the checkpoint as persistent buffers,
        # seeded from constants.py defaults. Registered before post_init(); the
        # weight initializer (_init_weights) only touches nn.Linear/nn.Embedding,
        # so these buffers are left untouched (behavior is numerically inert).
        self.register_buffer("fp_bp_argmax", BASE_PAIR_ARGMAX.clone(), persistent=True)
        self.register_buffer("fp_rep_argmax", REPEAT_ARGMAX.clone(), persistent=True)
        self.register_buffer("fp_bp", MINERVA_BP_FINGERPRINT.clone(), persistent=True)
        self.register_buffer("fp_bp_fwd", MINERVA_BP_FORWARD_FINGERPRINT.clone(), persistent=True)
        self.register_buffer("fp_bp_rev", MINERVA_BP_REVERSE_FINGERPRINT.clone(), persistent=True)
        self.register_buffer("fp_repeat", MINERVA_REPEAT_FINGERPRINT.clone(), persistent=True)
        self.register_buffer("fp_protein", MINERVA_PROTEIN_FINGERPRINT.clone(), persistent=True)

        # Use post_init() (not init_weights()) to match HF convention and the
        # inner MinervaModel: initializes lm_head + linear_heads and runs the
        # gradient-checkpointing backward-compat setup. The inner model also
        # calls post_init(); this top-level re-init is the standard HF pattern.
        self.post_init()

    def get_input_embeddings(self):
        """Return the input embeddings layer (required by PEFT/transformers)."""
        return self.minerva.tok_embeddings
    
    def set_input_embeddings(self, value):
        """Set the input embeddings layer (required by PEFT/transformers)."""
        self.minerva.tok_embeddings = value
    
    @property
    def linear_head(self) -> Optional[PyTorchLinearHead]:
        """Backward compatibility: return default head."""
        return self.linear_heads.get("default", None)
    
    @property
    def num_layers(self) -> int:
        """Return number of transformer layers in the model."""
        return len(self.minerva.encoder.layers)

    def _fingerprints(self) -> dict:
        """Fingerprint matrices baked into the checkpoint (seeded from constants)."""
        return {
            "bp_argmax": self.fp_bp_argmax, "rep_argmax": self.fp_rep_argmax,
            "bp": self.fp_bp, "bp_fwd": self.fp_bp_fwd, "bp_rev": self.fp_bp_rev,
            "repeat": self.fp_repeat, "protein": self.fp_protein,
        }


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[Union[bool, List[int]]] = None,
        output_attentions: bool = False,
        attention_layers: Optional[List[int]] = None,
        output_interactions: bool = False,
        interaction_layers: int = 2,
        output_contacts: Optional[Union[bool, List[str]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MaskedLMOutput, MinervaForMaskedLMOutput]:
        """
        Forward pass with optional attention maps and interaction predictions.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            cu_seqlens: Cumulative sequence lengths for packed sequences.
            max_seq_len: Maximum sequence length for packed sequences.
            labels: Labels for masked language modeling loss.
            output_hidden_states: Return hidden states.
                - True: all layers (tuple, HuggingFace compat)
                - List[int]: specific layers (dict {layer_idx: tensor})
            output_attentions: Whether to return raw attention maps. This follows
                the Hugging Face convention used by BERT-style models.
            attention_layers: Which raw attention layers to return. If None and
                output_attentions=True, returns the layers used by the selected
                interaction/contact heads, or the last 2 layers when no heads are
                requested.
            output_interactions: Return the standard Minerva interaction maps
                (base_pairing, protein, repeat).
            interaction_layers: Which trained interaction head depth to use:
                2 for last-2-layer heads (default), or 6 for last-6-layer heads.
            output_contacts: Backward-compatible low-level contact head API.
                - True: all loaded heads
                - List[str]: specific internal head names
            return_dict: Whether to return a dataclass (default: True).
        
        Returns:
            MinervaForMaskedLMOutput with logits, hidden_states, attentions,
            contact_predictions, and interactions.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = input_ids.device
        
        # Determine if we need specific hidden state layers
        need_all_hidden_states = output_hidden_states is True
        hidden_state_layers = output_hidden_states if isinstance(output_hidden_states, list) else None
        
        # Determine which raw attention layers the caller explicitly wants returned.
        # If output_attentions=True and attention_layers=None, we choose a default
        # after contact/interaction layers are known.
        attention_layers_requested = set()
        if output_attentions and attention_layers is not None:
            attention_layers_requested.update(attention_layers)

        # Determine which interaction/contact heads to run and their required layers.
        if output_interactions and output_contacts:
            raise ValueError("Use output_interactions or output_contacts, not both.")

        interaction_head_to_task = None
        contact_head_names = None
        if output_interactions:
            interaction_head_to_task = self._interaction_head_names(interaction_layers)
            contact_head_names = list(interaction_head_to_task.keys())
        elif output_contacts:
            if output_contacts is True:
                contact_head_names = list(self.linear_heads.keys())
            else:
                contact_head_names = output_contacts

        attention_layers_needed = set(attention_layers_requested)
        if contact_head_names:
            # Auto-detect layers needed for contact/interaction heads.
            for head_name in contact_head_names:
                if head_name not in self.linear_heads:
                    raise ValueError(f"Contact head '{head_name}' not found. Available: {list(self.linear_heads.keys())}")
                head = self.linear_heads[head_name]
                if head.layers:
                    attention_layers_needed.update(head.layers)
            
            # If no layers specified and heads don't have layers, use the default.
            if not attention_layers_needed:
                attention_layers_needed.update(self._default_contact_layers())

        if output_attentions and attention_layers is None:
            if contact_head_names and attention_layers_needed:
                # For interactions/contacts, expose the layers used by the chosen
                # heads by default. Explicit attention_layers still overrides this.
                attention_layers_requested.update(attention_layers_needed)
            else:
                attention_layers_requested.update([self.num_layers - 2, self.num_layers - 1])
            attention_layers_needed.update(attention_layers_requested)
        
        attention_layers_needed = sorted(attention_layers_needed) if attention_layers_needed else None
        if attention_layers_needed and cu_seqlens is not None:
            raise NotImplementedError(
                "output_attentions, output_interactions, and output_contacts are not "
                "supported with packed cu_seqlens because attention-map extraction "
                "cannot be done in a single packed forward pass."
            )
        
        # Determine what to pass to the base model for hidden states
        if need_all_hidden_states:
            output_hidden_states_param = True
        elif hidden_state_layers is not None:
            output_hidden_states_param = hidden_state_layers
        else:
            output_hidden_states_param = None
        
        loss = None
        prediction_scores = None
        final_hidden_states = None
        extracted_attentions = None

        # Single-pass path: get_attention_maps already runs a full encoder
        # forward, so reuse its hidden state for logits/hidden states instead of
        # running the encoder twice. Packed attention-map extraction is rejected
        # above because it cannot satisfy the one-forward-pass contract.
        can_single_pass = attention_layers_needed

        if can_single_pass:
            attention_result = self.get_attention_maps(
                input_ids=input_ids,
                layers=attention_layers_needed,
                attention_mask=attention_mask,
                return_hidden_state=True,
                output_hidden_states=output_hidden_states_param,
            )
            if output_hidden_states_param is not None:
                extracted_attentions, sequence_output, final_hidden_states = attention_result
            else:
                extracted_attentions, sequence_output = attention_result
            prediction_scores = self.lm_head(sequence_output)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                labels = labels.to(prediction_scores.device)
                loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        else:
            outputs = self.minerva(
                input_ids,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                output_hidden_states=output_hidden_states_param,
                return_dict=True,
            )
            sequence_output = outputs.last_hidden_state
            prediction_scores = self.lm_head(sequence_output)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                labels = labels.to(prediction_scores.device)
                loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            if need_all_hidden_states:
                hs = outputs.hidden_states
                final_hidden_states = tuple(hs) if hs is not None and not isinstance(hs, tuple) else hs
            elif hidden_state_layers is not None and outputs.hidden_states is not None:
                final_hidden_states = outputs.hidden_states

            if attention_layers_needed:
                extracted_attentions = self.get_attention_maps(
                    input_ids=input_ids,
                    layers=attention_layers_needed,
                    attention_mask=attention_mask,
                )

        # Compute contact predictions if requested
        contact_predictions = None
        if contact_head_names and extracted_attentions:
            contact_predictions = self.predict_interactions(
                extracted_attentions,
                {n: (interaction_head_to_task.get(n, n) if interaction_head_to_task else n)
                 for n in contact_head_names},
                input_ids.shape[0], input_ids.shape[1],
                default_layers=attention_layers_needed,
            )

        # Only expose raw attention maps explicitly requested by the caller.
        # Internal layers needed solely for interactions stay internal.
        if output_attentions and extracted_attentions is not None:
            final_attentions = {
                layer_idx: extracted_attentions[layer_idx]
                for layer_idx in sorted(attention_layers_requested)
                if layer_idx in extracted_attentions
            }
        else:
            final_attentions = None
        
        if not return_dict:
            output = (prediction_scores,)
            if final_hidden_states is not None:
                output = output + (final_hidden_states,)
            if final_attentions is not None:
                output = output + (final_attentions,)
            if contact_predictions is not None:
                output = output + (contact_predictions,)
            return ((loss,) + output) if loss is not None else output
        
        interactions = contact_predictions if output_interactions else None
        return MinervaForMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=final_hidden_states,
            attentions=final_attentions,
            contact_predictions=contact_predictions,
            interactions=interactions,
        )

    @torch.no_grad()
    def get_categorical_jacobian(
        self,
        sequence: str,
        tokenizer,
        nuc_token_ids: Optional[List[int]] = None,
        aa_token_ids: Optional[List[int]] = None,
        fast: bool = True,
        max_batch_size: int = 256,
        pos_chunk_size: int = 256,
        show_progress: bool = True,
        autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
        return_fingerprints: Union[bool, dict] = False,
        position_range: Optional[Tuple[int, int]] = None,
    ) -> Union[Tuple[torch.Tensor, List[str]], Tuple[torch.Tensor, List[str], torch.Tensor]]:
        """
        Compute the categorical Jacobian for a sequence.

        In fast mode, masks each position with [MASK] token.
        In full mode, substitutes each position with all valid tokens.

        Args:
            sequence: Input sequence string to analyze.
            tokenizer: Tokenizer with encode() method and mask_token_id attribute.
            nuc_token_ids: Optional nucleotide token IDs. Defaults to the Minerva
                nucleotide order derived from the tokenizer: a, t, g, c.
            aa_token_ids: Optional amino-acid token IDs. Defaults to the Minerva
                protein-fingerprint order derived from the tokenizer.
            fast: If True, only mask positions (faster, less memory).
                If False, compute all valid substitutions (slower, more comprehensive).
            max_batch_size: Maximum batch size for forward passes.
            pos_chunk_size: Chunk size for masking operations (memory optimization).
            show_progress: Whether to display a tqdm progress bar.
            autocast_dtype: Dtype for autocast. Set to None to disable autocast.
            return_fingerprints: Whether to compute contact fingerprints.
                Fingerprinting requires fast=False because classifiers need full
                substitution-channel Jacobian blocks.
                - If False (default): returns (jacobian, tokens)
                - If True: uses argmax method for classification
                - If dict: can specify classification options:
                    - method: "argmax" (default), "similarity", or
                      "similarity_multimodality"
                    - threshold options for similarity methods
            position_range: Optional (start, end) tuple to restrict Jacobian computation
                to a subsequence. Only positions in [start, end) are mutated, and the
                output tensor has L_seed = end - start instead of L. Useful when the
                sequence has flanking context that will be cropped away.

        Returns:
            If return_fingerprints=False:
                Tuple of (jacobian, tokens) where:
                - jacobian: Tensor of shape (L, 1, L, num_tokens) if fast=True,
                    or (L, num_tokens, L, num_tokens) if fast=False. On CPU.
                - tokens: List of token strings for each position.

            If return_fingerprints=True or dict:
                Tuple of (jacobian, tokens, fingerprints) where:
                - jacobian, tokens: as above
                - fingerprints: Tensor of shape (3, L, L) with channels:
                    [0] base_pair: Watson-Crick pairing contacts
                    [1] repeat: identity/self-similarity contacts
                    [2] other: contacts not matching either pattern

        Example:
            >>> tokenizer = AutoTokenizer.from_pretrained("path/to/model")
            >>> model = MinervaForMaskedLM.from_pretrained("path/to/model")
            >>> # Default: no fingerprints
            >>> jac, tokens = model.get_categorical_jacobian("acgt", tokenizer)
            >>> # With fingerprints (argmax method)
            >>> jac, tokens, fingerprints = model.get_categorical_jacobian(
            ...     "acgt", tokenizer, fast=False, return_fingerprints=True
            ... )
            >>> # With fingerprints (multimodal similarity method)
            >>> jac, tokens, fingerprints = model.get_categorical_jacobian(
            ...     "acgt", tokenizer, fast=False,
            ...     return_fingerprints={'method': 'similarity_multimodality'}
            ... )
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
            if show_progress:
                import warnings
                warnings.warn("tqdm not installed, progress bar disabled")
                show_progress = False

        if nuc_token_ids is None or aa_token_ids is None:
            default_nuc_token_ids, default_aa_token_ids = get_minerva_token_ids(tokenizer)
            if nuc_token_ids is None:
                nuc_token_ids = default_nuc_token_ids
            if aa_token_ids is None:
                aa_token_ids = default_aa_token_ids
        if return_fingerprints and fast:
            raise ValueError(
                "Fingerprinting requires full substitution-channel Jacobian blocks. "
                "Call get_categorical_jacobian(..., fast=False, return_fingerprints=...)."
            )

        device = next(self.parameters()).device

        # Token setup - keep on CPU initially, move to GPU only when needed
        all_tokens = nuc_token_ids + aa_token_ids
        num_tokens = len(all_tokens)
        nuc_count = len(nuc_token_ids)
        NUC_T_cpu = torch.tensor(nuc_token_ids)
        AA_T_cpu = torch.tensor(aa_token_ids)
        ALL_T_cpu = torch.tensor(all_tokens)
        NUC_T = NUC_T_cpu.to(device)
        AA_T = AA_T_cpu.to(device)
        ALL_T = ALL_T_cpu.to(device)
        NUC_IDX = {t: i for i, t in enumerate(nuc_token_ids)}
        AA_IDX = {t: i for i, t in enumerate(aa_token_ids)}

        # Encode sequence
        encoded = tokenizer.encode(sequence)
        input_ids = torch.tensor(encoded, dtype=torch.long, device=device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().tolist())
        L = input_ids.size(0)
        x0 = input_ids.unsqueeze(0)  # [1, L]
        input_ids_cpu = input_ids.cpu()

        # Get mask token
        mask_token_id = getattr(tokenizer, 'mask_token_id', None)
        if mask_token_id is None:
            raise ValueError("Tokenizer does not have a mask_token_id")

        # Identify special token positions to skip
        special_ids = set(getattr(tokenizer, 'all_special_ids', []))
        non_special_mask = torch.tensor(
            [encoded[i] not in special_ids for i in range(L)], dtype=torch.bool
        )

        # Position range: restrict mutations to [pr_start, pr_end)
        if position_range is not None:
            pr_start, pr_end = position_range
            L_out = pr_end - pr_start
        else:
            pr_start, pr_end = 0, L
            L_out = L

        # Build mutation list
        if fast:
            # Fast mode: just mask each non-special position
            non_special_positions = non_special_mask.nonzero(as_tuple=True)[0]
            if position_range is not None:
                non_special_positions = non_special_positions[
                    (non_special_positions >= pr_start) & (non_special_positions < pr_end)
                ]
            M = len(non_special_positions)
            positions = non_special_positions.to(device)
            new_tokens = torch.full((M,), mask_token_id, device=device, dtype=torch.long)
            channels = torch.zeros((M,), device=device, dtype=torch.long)
            channel_count = 1
        else:
            # Full mode: all valid substitutions for non-special positions
            meta = []
            for i, orig in enumerate(input_ids_cpu.tolist()):
                if orig in special_ids:
                    continue  # Skip special tokens
                if position_range is not None and (i < pr_start or i >= pr_end):
                    continue  # Skip positions outside range
                if orig in NUC_IDX:
                    for j, t in enumerate(nuc_token_ids):
                        if t != orig:
                            meta.append((i, t, j))
                elif orig in AA_IDX:
                    for j, t in enumerate(aa_token_ids):
                        if t != orig:
                            meta.append((i, t, nuc_count + j))
            M = len(meta)
            positions = torch.tensor([m[0] for m in meta], dtype=torch.long, device=device)
            new_tokens = torch.tensor([m[1] for m in meta], dtype=torch.long, device=device)
            channels = torch.tensor([m[2] for m in meta], dtype=torch.long, device=device)
            channel_count = num_tokens

        # Prepare result tensor on CPU for memory efficiency
        if fast:
            result_shape = (L_out, 1, L_out, num_tokens)
        else:
            result_shape = (L_out, num_tokens, L_out, num_tokens)
        fx_h = torch.zeros(result_shape, dtype=torch.float32)

        # Context manager for autocast
        if autocast_dtype is not None and device.type == 'cuda':
            autocast_ctx = torch.amp.autocast('cuda', dtype=autocast_dtype)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            # Run baseline
            base_output = self(x0)
            base = base_output.logits  # [1, L, V]
            fx = base[0, :, ALL_T].float()  # [L, num_tokens]
            fx_cpu = fx[pr_start:pr_end].cpu()  # [L_out, num_tokens]
            del base, fx
            torch.cuda.empty_cache()

            # Process in batches
            if show_progress and tqdm is not None:
                pbar = tqdm(total=M, desc="Computing jacobian")
            else:
                pbar = None

            try:
                for chunk_start in range(0, M, max_batch_size):
                    chunk_end = min(chunk_start + max_batch_size, M)
                    chunk_size = chunk_end - chunk_start

                    # Create batch for this chunk
                    batch = x0.expand(chunk_size, L).clone()
                    pos_chunk = positions[chunk_start:chunk_end]
                    tok_chunk = new_tokens[chunk_start:chunk_end]
                    ch_chunk = channels[chunk_start:chunk_end]
                    batch_idx = torch.arange(chunk_size, device=device)
                    batch[batch_idx, pos_chunk] = tok_chunk

                    # Run forward pass and move to CPU immediately
                    batch_output = self(batch)
                    out = batch_output.logits[:, :, ALL_T].float()
                    out_cpu = out.cpu()

                    for i in range(chunk_size):
                        p = pos_chunk[i].item()
                        c = ch_chunk[i].item()
                        fx_h[p - pr_start, c] = out_cpu[i, pr_start:pr_end]

                    del batch, out, out_cpu, batch_idx
                    torch.cuda.empty_cache()

                    if pbar is not None:
                        pbar.update(chunk_size)
            finally:
                if pbar is not None:
                    pbar.close()

        # Subtract baseline
        if fast:
            jac = fx_h - fx_cpu
        else:
            fx_exp = fx_cpu.unsqueeze(0).unsqueeze(1)
            jac = fx_h - fx_exp
        del fx_h

        # Apply validity masking (use seed-region tokens for position_range)
        ids_for_mask = input_ids_cpu[pr_start:pr_end]
        if fast:
            # Fast mode: mask invalid nuc/aa combinations
            is_nuc_pos = torch.isin(ids_for_mask, NUC_T_cpu).view(L_out, 1, 1, 1).expand(L_out, 1, L_out, 1)
            is_aa_pos = torch.isin(ids_for_mask, AA_T_cpu).view(L_out, 1, 1, 1).expand(L_out, 1, L_out, 1)
            is_nuc_tok = torch.isin(ALL_T_cpu, NUC_T_cpu).view(1, 1, 1, num_tokens).expand(1, 1, 1, num_tokens)
            is_aa_tok = torch.isin(ALL_T_cpu, AA_T_cpu).view(1, 1, 1, num_tokens).expand(1, 1, 1, num_tokens)
            valid = (is_nuc_pos & is_nuc_tok) | (is_aa_pos & is_aa_tok)
            jac = torch.where(valid, jac, torch.zeros_like(jac))
        else:
            # Full mode: chunked masking for memory efficiency
            is_nuc_token_lookup = torch.isin(ids_for_mask, NUC_T_cpu)
            is_aa_token_lookup = torch.isin(ids_for_mask, AA_T_cpu)
            nuc_tok_mask = torch.isin(ALL_T_cpu, NUC_T_cpu)
            aa_tok_mask = torch.isin(ALL_T_cpu, AA_T_cpu)

            for pos_start in range(0, L_out, pos_chunk_size):
                pos_end = min(pos_start + pos_chunk_size, L_out)
                pos_slice = slice(pos_start, pos_end)

                is_nuc_pos_chunk = is_nuc_token_lookup[pos_slice].view(-1, 1, 1, 1)
                is_aa_pos_chunk = is_aa_token_lookup[pos_slice].view(-1, 1, 1, 1)

                for tok_start in range(0, num_tokens, pos_chunk_size):
                    tok_end = min(tok_start + pos_chunk_size, num_tokens)
                    tok_slice = slice(tok_start, tok_end)

                    is_nuc_tok_chunk = nuc_tok_mask[tok_slice].view(1, -1, 1, 1)
                    is_aa_tok_chunk = aa_tok_mask[tok_slice].view(1, -1, 1, 1)

                    valid_chunk = (is_nuc_pos_chunk & is_nuc_tok_chunk) | (is_aa_pos_chunk & is_aa_tok_chunk)
                    jac[pos_slice, tok_slice] = torch.where(
                        valid_chunk, jac[pos_slice, tok_slice], torch.zeros_like(jac[pos_slice, tok_slice])
                    )

            # Zero out self-substitutions
            for n, tok in enumerate(ids_for_mask.tolist()):
                if tok in NUC_IDX:
                    jac[n, NUC_IDX[tok]] = 0.0
                elif tok in AA_IDX:
                    jac[n, nuc_count + AA_IDX[tok]] = 0.0

        # Slice tokens to seed region if position_range is set
        tokens_out = tokens[pr_start:pr_end]

        # Compute fingerprints if requested
        if return_fingerprints:
            # Compute contact map from Jacobian (torch tensor on jac's device)
            contact = jac_to_contact(jac, symm=True, center=True, diag="remove", apc=True).to(jac.dtype)

            # Parse fingerprint options
            if isinstance(return_fingerprints, dict):
                method = return_fingerprints.get('method', 'argmax')
                kwargs = {k: v for k, v in return_fingerprints.items() if k != 'method'}
            else:
                method = 'argmax'
                kwargs = {}

            # Classify contacts into fingerprint channels
            fingerprints = classify_jacobian_contacts(jac, contact, tokens_out, method=method, fingerprints=self._fingerprints(), **kwargs)

            return jac, tokens_out, fingerprints

        return jac, tokens_out


    @torch.no_grad()
    def get_fingerprints(
        self,
        sequence: str,
        tokenizer,
        nuc_token_ids: Optional[List[int]] = None,
        aa_token_ids: Optional[List[int]] = None,
        max_batch_size: int = 256,
        pos_chunk_size: int = 256,
        show_progress: bool = True,
        autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
        position_range: Optional[Tuple[int, int]] = None,
        method: str = "similarity_multimodality",
        return_jacobian: bool = False,
        fingerprint_refs: Optional[dict] = None,
        **fingerprint_kwargs,
    ) -> FingerprintResult:
        """Compute named interaction fingerprints for a sequence.

        This is the user-facing fingerprinting API. It computes the required
        full categorical Jacobian internally, classifies it into named channels,
        and returns a :class:`FingerprintResult`. The large raw Jacobian is not
        stored on the result unless ``return_jacobian=True``.

        Args:
            sequence: Input sequence string to analyze.
            tokenizer: Tokenizer used by the Minerva checkpoint.
            nuc_token_ids, aa_token_ids: Optional token IDs for the categorical
                Jacobian alphabet. Defaults are derived from the tokenizer.
            max_batch_size, pos_chunk_size, show_progress, autocast_dtype,
                position_range: Forwarded to :meth:`get_categorical_jacobian`.
            method: Fingerprint classifier: ``"similarity_multimodality"``
                (default), ``"argmax"``, or ``"similarity"``.
            return_jacobian: If True, attach the raw full Jacobian to
                ``result.jacobian`` for advanced analysis.
            fingerprint_refs: Optional reference fingerprint matrices. Defaults
                to the checkpoint-baked Minerva references.
            **fingerprint_kwargs: Method-specific classifier options such as
                ``bp_threshold``, ``repeat_threshold``, ``protein_threshold``,
                ``aa_start``, ``jac_aa_order``, and ``split_bp``.

        Returns:
            FingerprintResult with ``tokens``, named ``fingerprints``,
            ``channel_names``, and the derived ``contacts`` map.

        Example:
            >>> fp = model.get_fingerprints(sequence, tokenizer)
            >>> fp["basepairing"]
            >>> fp["protein"]
        """
        jac, tokens = self.get_categorical_jacobian(
            sequence,
            tokenizer,
            nuc_token_ids=nuc_token_ids,
            aa_token_ids=aa_token_ids,
            fast=False,
            max_batch_size=max_batch_size,
            pos_chunk_size=pos_chunk_size,
            show_progress=show_progress,
            autocast_dtype=autocast_dtype,
            return_fingerprints=False,
            position_range=position_range,
        )
        refs = self._fingerprints() if fingerprint_refs is None else fingerprint_refs
        return compute_fingerprints(
            jac,
            tokens,
            method=method,
            fingerprints=refs,
            include_jacobian=return_jacobian,
            **fingerprint_kwargs,
        )


    @torch.no_grad()
    def get_attention_maps(
        self,
        input_ids: torch.Tensor,
        layers: Optional[List[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_state: bool = False,
        output_hidden_states: Optional[Union[bool, List[int]]] = None,
    ) -> Union[dict, Tuple[dict, torch.Tensor]]:
        """
        Extract attention maps from specified transformer layers.

        Uses monkey-patching to temporarily replace attention forward methods
        with versions that compute and store attention scores.

        Args:
            input_ids: Input token IDs of shape [batch_size, sequence_length].
            layers: List of 0-indexed layer numbers to extract from.
                   If None, extracts from last 2 layers.
            attention_mask: Optional attention mask for the input.
            return_hidden_state: If True, also return the final hidden state
                from the encoder pass (avoids needing a second forward pass).
            output_hidden_states: Optional hidden-state collection request using
                the same convention as forward/MinervaModel: True for all layer
                outputs, or a list of layer indices for a sparse dict.

        Returns:
            Dict mapping layer indices to attention tensors of shape
            [batch_size, num_heads, seq_len, seq_len].
            If return_hidden_state=True, returns (attention_dict, hidden_state)
            or (attention_dict, hidden_state, hidden_states) when
            output_hidden_states is requested.
        """
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        batch_size, seq_len = input_ids.shape
        
        # Get all layers
        all_layers = self.minerva.encoder.layers
        num_layers = len(all_layers)
        
        # Default to last 2 layers
        if layers is None:
            layers = [num_layers - 2, num_layers - 1]
        
        layers_to_patch = [l for l in layers if 0 <= l < num_layers]
        if not layers_to_patch:
            raise RuntimeError(
                f"No valid layers to patch. Requested: {layers}, "
                f"Model has {num_layers} layers (valid range: 0 to {num_layers - 1})"
            )
        
        extracted_maps = {}
        original_forwards = {}
        
        def make_attn_forward_with_scores(attn_module, layer_idx, storage, batch_attention_mask=None):
            """Create a forward function that captures attention scores to external storage."""

            def forward_with_scores(x, cu_seqlens=None, max_seq_len=None, attention_mask=None):
                # For attention extraction, we need batch format (not packed)
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                    was_flat = True
                else:
                    was_flat = False

                bsz, seqlen, hidden = x.shape

                # Compute Q, K, V
                qkv = attn_module.wqkv(x)
                q, k, v = torch.split(qkv, attn_module.n_heads * attn_module.head_dim, dim=-1)

                # Shape: (batch, seqlen, n_heads, head_dim)
                q = q.view(bsz, seqlen, attn_module.n_heads, attn_module.head_dim)
                k = k.view(bsz, seqlen, attn_module.n_heads, attn_module.head_dim)
                v = v.view(bsz, seqlen, attn_module.n_heads, attn_module.head_dim)

                # Apply QK norm if present
                if attn_module.qk_norm:
                    q = attn_module.q_norm(q)
                    k = attn_module.k_norm(k)

                q, k = attn_module.rotary_emb(q, k)

                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                scale = 1.0 / (attn_module.head_dim ** 0.5)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale

                # Mask padding positions in attention scores
                if batch_attention_mask is not None:
                    # Mask keys: real tokens don't attend to padding
                    key_mask = batch_attention_mask[:, None, None, :]  # [B, 1, 1, S]
                    scores = scores.masked_fill(~key_mask.bool(), float('-inf'))

                attn_weights = torch.nn.functional.softmax(scores, dim=-1)

                if batch_attention_mask is not None:
                    # Zero out padding query rows (they shouldn't attend to anything)
                    query_mask = batch_attention_mask[:, None, :, None]  # [B, 1, S, 1]
                    attn_weights = attn_weights * query_mask.to(attn_weights.dtype)

                # Store weights to external dict
                storage[layer_idx] = attn_weights.detach()

                # Compute output
                output = torch.matmul(attn_weights, v)
                output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
                output = attn_module.wo(output)

                if was_flat:
                    output = output.squeeze(0)

                return output

            return forward_with_scores
        
        # Patch the layers
        for layer_idx in layers_to_patch:
            attn_module = all_layers[layer_idx].attention
            original_forwards[layer_idx] = attn_module.forward
            attn_module.forward = make_attn_forward_with_scores(attn_module, layer_idx, extracted_maps, batch_attention_mask=attention_mask)
        
        need_all_hidden_states = output_hidden_states is True
        hidden_layers_set = set(output_hidden_states) if isinstance(output_hidden_states, list) else set()
        hidden_states = [] if need_all_hidden_states else ({} if hidden_layers_set else None)

        try:
            # Forward pass
            h = self.minerva.tok_embeddings(input_ids)
            for layer_idx, layer in enumerate(self.minerva.encoder.layers):
                h = layer(h)
                if need_all_hidden_states:
                    hidden_states.append(h)
                elif layer_idx in hidden_layers_set:
                    hidden_states[layer_idx] = h
        finally:
            # Restore original forward methods
            for layer_idx, original_fwd in original_forwards.items():
                all_layers[layer_idx].attention.forward = original_fwd

        # Verify we got all requested layers
        missing_layers = set(layers_to_patch) - set(extracted_maps.keys())
        if missing_layers:
            raise RuntimeError(
                f"Failed to extract attention from layers {sorted(missing_layers)}. "
            )

        if return_hidden_state:
            if hidden_states is not None:
                return extracted_maps, h, hidden_states
            return extracted_maps, h
        if hidden_states is not None:
            return extracted_maps, hidden_states
        return extracted_maps

    @torch.no_grad()
    def predict_contacts(
        self,
        sequence: str = None,
        tokenizer = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        seed_start: int = 0,
        seed_end: int = None,
        contact_size: int = None,
        layers_to_extract: List[int] = None,
        head_names: List[str] = None,
        return_attention_maps: bool = False,
        return_dict: bool = None,
    ) -> Union[torch.Tensor, Dict]:
        """
        End-to-end contact prediction from sequence to contact matrix.
        Supports batched input for multiple sequences.

        Args:
            sequence: Input sequence string to analyze. Either sequence+tokenizer or input_ids must be provided.
            tokenizer: Tokenizer with encode() method. Required if sequence is provided.
            input_ids: Pre-tokenized input IDs [seq_len] or [batch, seq_len].
            attention_mask: Boolean mask [batch, seq_len] where True = real token.
                Required when batching variable-length sequences.
            seed_start: Start position of region to predict contacts for (default: 0)
            seed_end: End position of region to predict contacts for (default: full sequence length)
            contact_size: Size of the contact region (alternative to seed_end)
            layers_to_extract: List of layer indices to extract attention from.
                              If None, uses union of all heads' required layers.
            head_names: List of regression head names to use.
                       If None, uses "default" head if available, else all heads.
            return_attention_maps: Whether to return raw and processed attention maps (default: False)
            return_dict: Force return type - True=dict, False=tensor, None=auto (default: None)
                - Auto behavior: returns tensor if single head and no attention maps, else dict

        Returns:
            For single sequence (backward compatible):
                torch.Tensor [region_size, region_size] or dict with same shapes.
            For batch:
                torch.Tensor [batch, region_size, region_size] or dict with batched tensors.

        Example:
            >>> # Single sequence
            >>> contacts = model.predict_contacts(sequence="ACGTACGT...", tokenizer=tokenizer)
            >>> # Batch
            >>> ids, mask = collate_sequences(tokenizer, ["ACGT...", "TGCA..."])
            >>> contacts = model.predict_contacts(input_ids=ids, attention_mask=mask)
        """
        if len(self.linear_heads) == 0:
            raise ValueError("No contact heads in this checkpoint.")

        # Handle input - either sequence+tokenizer or input_ids
        device = next(self.parameters()).device
        is_single = False
        if input_ids is not None:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                is_single = True
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        elif sequence is not None:
            if tokenizer is None:
                raise ValueError("tokenizer is required when providing sequence")
            encoded = tokenizer.encode(sequence)
            input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
            is_single = True
        else:
            raise ValueError("Either sequence+tokenizer or input_ids must be provided")

        batch_size = input_ids.shape[0]
        if batch_size == 1:
            is_single = True

        # Determine which heads to use
        if head_names is None:
            if "default" in self.linear_heads:
                head_names = ["default"]
            else:
                head_names = list(self.linear_heads.keys())
        else:
            for name in head_names:
                if name not in self.linear_heads:
                    raise ValueError(
                        f"Regression head '{name}' not found. "
                        f"Available heads: {list(self.linear_heads.keys())}"
                    )

        # Determine which layers to extract: union of all heads' required layers
        if layers_to_extract is None:
            all_layers = set()
            for head_name in head_names:
                head = self.linear_heads[head_name]
                if head.layers is not None:
                    all_layers.update(head.layers)
            if all_layers:
                layers_to_extract = sorted(list(all_layers))
            else:
                layers_to_extract = self._default_contact_layers()

        seq_len = input_ids.shape[1]

        # Determine the region to predict contacts for
        if seed_start is None:
            seed_start = 0                      # None means "from the start"
        if contact_size is not None:
            seed_end = seed_start + contact_size
        elif seed_end is None:
            seed_end = seq_len
            contact_size = seed_end - seed_start
        else:
            contact_size = seed_end - seed_start

        if contact_size <= 0:
            raise ValueError(
                f"Invalid region: seed_start={seed_start}, seed_end={seed_end}, "
                f"contact_size={contact_size}"
            )

        # Extract attention maps once (union of all heads' required layers)
        attention_maps_raw = self.get_attention_maps(
            input_ids=input_ids,
            layers=layers_to_extract,
            attention_mask=attention_mask,
        )

        # Debug: verify attention extraction worked
        if not attention_maps_raw:
            raise RuntimeError(
                f"Failed to extract attention maps. Requested layers: {layers_to_extract}, "
                f"Model has {self.num_layers} layers. Check that layers are 0-indexed and within range [0, {self.num_layers - 1}]."
            )

        # Prepare results dictionary
        results = {
            'predictions': {},
        }

        if return_attention_maps:
            results['attention_maps_raw'] = attention_maps_raw
            results['attention_maps_processed'] = {}

        # Process each regression head
        raw_feature_cache = {}  # tuple(head_layers) -> raw (un-symmetrized) features, reused across heads
        for head_name in head_names:
            linear_head = self.linear_heads[head_name]

            # Handle AttentionHeadExtractor separately
            if isinstance(linear_head, AttentionHeadExtractor):
                # Crop attention maps to seed region before extraction
                cropped_maps = {}
                for layer_idx, attn in attention_maps_raw.items():
                    # attn: [B, heads, S, S]
                    cropped_maps[layer_idx] = attn[:, :, seed_start:seed_end, seed_start:seed_end]

                pred = linear_head.predict_from_attention(cropped_maps)
                # pred: [B, R, R] or [R, R]
                if is_single and pred.ndim == 3:
                    pred = pred.squeeze(0)
                results['predictions'][head_name] = pred
                continue

            # Handle PyTorchLinearHead (sklearn regressor)
            head_layers = linear_head.layers if linear_head.layers is not None else layers_to_extract
            store = results['attention_maps_processed'] if return_attention_maps else None

            if linear_head.apply_apc or store is not None:
                # APC is nonlinear (can't defer past matmul); and when caller wants the
                # per-head processed maps stashed, use the original symmetrize-then-matmul route.
                concat_features = self._build_head_features(
                    linear_head, attention_maps_raw, head_layers,
                    seed_start=seed_start, seed_end=seed_end, device=device,
                    store=store, head_name=head_name, warn=True,
                )
                if concat_features is None:
                    print(f"Warning: No valid layers found for head '{head_name}'. Skipping.")
                    continue
                concat_features = concat_features.float()
                probs = linear_head.predict_proba(concat_features)[:, 1]  # Positive class probabilities
                pred_matrix = probs.reshape(batch_size, contact_size, contact_size)
            else:
                # Fast path: build raw features once per layer-set, symmetrize after matmul.
                key = tuple(head_layers)
                raw = raw_feature_cache.get(key)
                if raw is None:
                    raw = self._build_head_features(
                        linear_head, attention_maps_raw, head_layers,
                        seed_start=seed_start, seed_end=seed_end, device=device,
                        store=None, head_name=head_name, warn=True,
                        apply_transforms=False,
                    )
                    raw_feature_cache[key] = raw
                if raw is None:
                    print(f"Warning: No valid layers found for head '{head_name}'. Skipping.")
                    continue
                pred_matrix = linear_head.predict_map(raw, batch_size, contact_size)

            # Squeeze batch dim for single-sequence backward compat
            if is_single:
                pred_matrix = pred_matrix.squeeze(0)

            results['predictions'][head_name] = pred_matrix

        # Determine return type
        if return_dict is None:
            # Auto-determine return type
            if len(head_names) == 1 and not return_attention_maps:
                # Return just the tensor for backward compatibility
                return results['predictions'][head_names[0]]
            else:
                return results
        elif return_dict:
            return results
        else:
            # Explicitly requested tensor - return first head's prediction
            return results['predictions'][head_names[0]]


AutoConfig.register("minerva", MinervaConfig)
AutoModelForMaskedLM.register(MinervaConfig, MinervaForMaskedLM)
