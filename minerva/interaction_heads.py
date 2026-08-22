"""Interaction heads over a backbone's attention maps.

Mixed into a model that provides ``linear_heads``, ``num_layers`` and the
attention maps themselves; nothing here is backbone-specific.
"""

from typing import Dict, List, Optional

import torch

from torch import nn


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    """Make tensor symmetric in final two dimensions, used for contact prediction."""
    return x + x.transpose(-1, -2)


def apc(x: torch.Tensor) -> torch.Tensor:
    """Perform Average Product Correction, used for contact prediction."""
    input_dtype = x.dtype
    # Do reductions in fp32: in fp16 the 1e-10 guard underflows to 0, so an
    # all-zero or fully masked map can produce 0/0 -> NaN.
    x = x.float()
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12 + 1e-10)  # in-place to reduce memory, avoid div by zero
    normalized = x - avg
    return normalized.to(input_dtype)


class PyTorchLinearHead(nn.Module):
    """
    Logistic regression contact head.
    """
    
    def __init__(
        self,
        input_dim: int = None,
        apply_symmetrize: bool = False,
        apply_apc: bool = False,
        layers: List[int] = None
    ):
        """
        Initialize the PyTorch linear head.
        
        Args:
            input_dim: Input feature dimension (set from linear_heads_config when loading a checkpoint)
            apply_symmetrize: Whether to apply symmetrization to attention features
            apply_apc: Whether to apply APC correction to attention features
            layers: List of layer indices this head was trained on (for validation)
        """
        super().__init__()
        self.linear = None
        self.apply_symmetrize = apply_symmetrize
        self.apply_apc = apply_apc
        self.layers = layers  # Store which layers this head expects
        if input_dim is not None:
            self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear head.
        
        Args:
            x: Input features [batch_size, num_features] or [num_contact_pairs, num_features]
            
        Returns:
            Logits [batch_size, 1] or [num_contact_pairs, 1]
        """
        if self.linear is None:
            raise ValueError("Linear head not initialized.")
        # Cast both sides to match — avoids dtype mismatch when model is bf16
        return torch.nn.functional.linear(
            x.to(self.linear.weight.dtype),
            self.linear.weight,
            self.linear.bias,
        )
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities (equivalent to sklearn's predict_proba).
        
        Args:
            x: Input features [batch_size, num_features]
            
        Returns:
            Probabilities [batch_size, 2] where [:, 1] is positive class
        """
        logits = self.forward(x)
        probs_pos = torch.sigmoid(logits).squeeze(-1)  # [batch_size]
        probs_neg = 1 - probs_pos
        return torch.stack([probs_neg, probs_pos], dim=1)  # [batch_size, 2]

    def predict_map(self, raw_features: torch.Tensor, batch_size: int, R: int) -> torch.Tensor:
        """Contact-probability map from UN-symmetrized features (symmetrize-after-matmul).

        ``raw_features``: ``[B*R*R, F]`` built WITHOUT symmetrize/apc. Returns sigmoid
        probabilities ``[B, R, R]``. Mathematically identical (to floating-point rounding)
        to ``sigmoid(linear(symmetrize(features)))`` when ``apply_symmetrize`` and NOT
        ``apply_apc``: the linear head contracts the channel axis while ``symmetrize``
        (``X + X^T``) acts on the (i, j) axes, so the two commute. The un-normalised
        symmetrize would double the bias, so the matmul is applied WITHOUT bias, the
        1-channel map is symmetrized, then the bias is added ONCE.
        """
        if self.apply_apc:
            raise ValueError("predict_map is invalid with apply_apc=True (APC is nonlinear).")
        if self.linear is None:
            raise ValueError("Linear head not initialized.")
        w = self.linear.weight
        b = self.linear.bias
        # Matmul in the weight dtype (tensor cores accumulate in fp32), collapsing F
        # channels -> 1. Then upcast only the small [R, R] map for the symmetrize/bias/
        # sigmoid so those steps are numerically safe without materializing an fp32 copy
        # of the large [B*R*R, F] feature tensor.
        logits = torch.nn.functional.linear(raw_features.to(w.dtype), w, None)  # matmul, NO bias
        logits = logits.reshape(batch_size, R, R).float()
        if self.apply_symmetrize:
            logits = logits + logits.transpose(-1, -2)  # same X + X^T as symmetrize()
        if b is not None:
            logits = logits + b.float()  # bias added exactly ONCE
        return torch.sigmoid(logits)


class AttentionHeadExtractor(nn.Module):
    """
    Simple contact predictor that extracts a specific attention head.
    Used when a single attention head is a good proxy for contacts.
    
    All indices are 0-indexed:
    - For a 33-layer model: layer_idx in [0, 32], last layer is 32
    - For a 20-head model: head_idx in [0, 19], last head is 19
    """
    
    def __init__(
        self,
        layer_idx: int,
        head_idx: int,
        apply_symmetrize: bool = True,
        apply_apc: bool = True,
    ):
        """
        Initialize attention head extractor.
        
        Args:
            layer_idx: Transformer layer index (0-indexed). For depth=33, valid range is [0, 32].
            head_idx: Attention head index (0-indexed). For heads=20, valid range is [0, 19].
            apply_symmetrize: Whether to symmetrize the attention matrix.
            apply_apc: Whether to apply APC correction.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.apply_symmetrize = apply_symmetrize
        self.apply_apc = apply_apc
        self.layers = [layer_idx]  # For compatibility with layer detection
        self.linear = None  # Not used, but keeps interface consistent
    
    def predict_from_attention(self, attention_maps: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Extract contact prediction from attention maps.

        Args:
            attention_maps: Dict {layer_idx: [batch, heads, seq, seq]}

        Returns:
            Contact matrix [batch, seq, seq] if batched, [seq, seq] if unbatched.
        """
        if self.layer_idx not in attention_maps:
            raise ValueError(f"Layer {self.layer_idx} not in attention maps")

        attn = attention_maps[self.layer_idx]

        # Extract specific head, preserving batch dimension
        if attn.ndim == 4:
            head_attn = attn[:, self.head_idx]  # [batch, seq, seq]
        else:
            head_attn = attn[self.head_idx]  # [seq, seq]

        if self.apply_symmetrize:
            head_attn = head_attn + head_attn.transpose(-2, -1)

        if self.apply_apc:
            if head_attn.ndim == 2:
                head_attn = apc(head_attn.unsqueeze(0).unsqueeze(0)).squeeze()
            else:
                # [batch, seq, seq] → [batch, 1, seq, seq] for apc → squeeze back
                head_attn = apc(head_attn.unsqueeze(1)).squeeze(1)

        return head_attn


class InteractionHeads:
    # Public task keys, and the head-name suffix each trained depth uses.
    interaction_tasks = ("base_pairing", "protein", "repeat")
    head_depths = {2: "", 6: "_l6"}

    def _default_contact_layers(self) -> List[int]:
        """Default layers for contact prediction: the last 2 transformer layers.

        Matches the shipped regression heads, which are trained on the last 2
        layers. Heads that specify their own ``layers`` override this default.
        """
        return list(range(max(0, self.num_layers - 2), self.num_layers))

    def _interaction_head_names(self, interaction_layers: int) -> Dict[str, str]:
        """Map public interaction depth to internal head names.

        Returns a mapping from internal head name to public task name. Public
        interaction outputs always use the stable task keys base_pairing,
        protein, and repeat; interaction_layers selects the trained head depth.
        """
        if interaction_layers not in self.head_depths:
            raise ValueError(
                f"interaction_layers must be one of {sorted(self.head_depths)}"
            )

        tasks = self.interaction_tasks
        suffix = self.head_depths[interaction_layers]
        mapping = {f"{task}{suffix}": task for task in tasks}
        missing = [name for name in mapping if name not in self.linear_heads]
        if missing:
            raise ValueError(
                f"Interaction head(s) not found: {missing}. "
                f"Available heads: {list(self.linear_heads.keys())}"
            )
        return mapping

    def _contacts_from_attention(
        self,
        heads,
        attention_maps,
        head_layers,
    ):
        """Contact maps for several linear heads straight from the attention maps.

        Equivalent (to floating-point rounding) to ``predict_map`` over
        ``_build_head_features(..., apply_transforms=False)``, but contracts the
        channel axis in place instead of materializing it. ``_build_head_features``
        clones each layer's attention, forces a transposing ``reshape``, and
        ``cat``s the layers -- three passes over a ``[B, F, R, R]`` tensor (1.34 GB
        at R=4096, F=40) to set up ~4 GFLOP of matmul, so the head is entirely
        memory-bound. Here every head that shares ``head_layers`` is contracted in
        one pass, and the symmetrize is folded in afterwards (it acts on the (i, j)
        axes, the head on the channel axis, so the two commute).

        Parameters
        ----------
        heads : sequence of PyTorchLinearHead
            All must share ``head_layers`` and have ``apply_apc=False``.
        attention_maps : dict
            ``{layer_idx: [B, heads, R, R]}``.
        head_layers : sequence of int
            Layers whose attention forms the feature axis, in feature order.

        Returns
        -------
        torch.Tensor
            ``[len(heads), B, R, R]`` sigmoid probabilities.
        """
        missing = [l for l in head_layers if l not in attention_maps]
        if missing:
            raise ValueError(
                f"layers {missing} needed by the contact head were not extracted; "
                f"have {sorted(attention_maps)}. Feature columns are positional, so "
                "silently dropping a layer would misalign every weight."
            )
        for head in heads:
            if head.apply_apc:
                raise ValueError("_contacts_from_attention is invalid with apply_apc=True")
            if head.linear is None:
                raise ValueError("Linear head not initialized.")

        weight = torch.stack([h.linear.weight[0] for h in heads])          # [K, F]
        bias = torch.stack([h.linear.bias[0] for h in heads]).float()      # [K]

        logits = None
        offset = 0
        for layer_idx in head_layers:
            attn = attention_maps[layer_idx]
            if attn.ndim == 3:
                attn = attn.unsqueeze(0)
            n_heads = attn.shape[1]
            w = weight[:, offset:offset + n_heads].to(attn.dtype)          # [K, heads]
            offset += n_heads
            part = torch.einsum("bhij,kh->kbij", attn, w).float()
            logits = part if logits is None else logits + part
        if offset != weight.shape[1]:
            raise ValueError(
                f"feature width mismatch: attention supplied {offset} channels, "
                f"head expects {weight.shape[1]}"
            )

        if heads[0].apply_symmetrize:
            logits = logits + logits.transpose(-1, -2)
        return torch.sigmoid(logits + bias[:, None, None, None])

    def _build_head_features(
        self,
        head,
        attention_maps,
        head_layers,
        seed_start: Optional[int] = None,
        seed_end: Optional[int] = None,
        device=None,
        store=None,
        head_name: Optional[str] = None,
        warn: bool = False,
        apply_transforms: bool = True,
    ) -> Optional[torch.Tensor]:
        """Build concatenated per-head attention features for a linear regression head.

        Shared by ``forward(output_contacts=...)`` and ``predict_contacts()``. For each
        layer in ``head_layers``: clone the attention map, apply the head's
        symmetrize/apc transforms, ensure a 4D ``[B, heads, S, S]`` shape, optionally
        crop to ``[seed_start:seed_end]``, optionally move to ``device`` and stash the
        processed map into ``store[(head_name, layer_idx)]``, then reshape to
        ``[B*R*R, heads]``. Returns the concatenated ``[N, total_heads]`` feature
        tensor (uncast), or ``None`` if no usable layers were found.

        ``warn=True`` emits the skip warnings used by ``predict_contacts``; ``forward``
        passes ``warn=False`` to stay silent.
        """
        layer_features = []
        for layer_idx in head_layers:
            if layer_idx not in attention_maps:
                if warn:
                    print(f"Warning: Layer {layer_idx} needed by head '{head_name}' was not extracted. Skipping.")
                continue

            # Start with raw attention tensor (copy to avoid modifying original)
            attn_tensor = attention_maps[layer_idx].clone()

            # Apply head-specific transformations (skipped when building raw features
            # for the symmetrize-after-matmul fast path).
            if apply_transforms and head.apply_symmetrize:
                attn_tensor = symmetrize(attn_tensor)
            if apply_transforms and head.apply_apc:
                attn_tensor = apc(attn_tensor)

            # Ensure 4D: [B, heads, S, S]
            if attn_tensor.ndim == 3:
                attn_tensor = attn_tensor.unsqueeze(0)

            # Optionally crop to the seed region: [B, heads, R, R]
            if seed_start is not None:
                if seed_end > attn_tensor.shape[-1]:
                    if warn:
                        print(
                            f"Warning: Sequence region [{seed_start}:{seed_end}] exceeds "
                            f"attention tensor length {attn_tensor.shape[-1]} for layer {layer_idx}"
                        )
                    continue
                attn_tensor = attn_tensor[:, :, seed_start:seed_end, seed_start:seed_end]
                if attn_tensor.numel() == 0:
                    if warn:
                        print(f"Warning: Cropped tensor is empty for layer {layer_idx}")
                    continue

            if device is not None:
                attn_tensor = attn_tensor.to(device)

            # Store processed attention map if requested
            if store is not None:
                store[(head_name, layer_idx)] = attn_tensor.clone()

            # [B, heads, R, R] → [B*R*R, heads]
            num_heads = attn_tensor.shape[1]
            reshaped = attn_tensor.permute(0, 2, 3, 1).reshape(-1, num_heads)
            layer_features.append(reshaped)

        if not layer_features:
            return None
        return torch.cat(layer_features, dim=1)
