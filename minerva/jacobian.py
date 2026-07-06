"""
Jacobian utilities for contact classification and fingerprinting.

This module provides functions to convert Jacobian tensors to contact maps
and classify contacts into categories (base pair, repeat, other) based on
the coupling patterns in the Jacobian.

Key functions:
- jac_to_contact(): Convert Jacobian to contact map
- classify_contacts_by_argmax(): Fast argmax-based classification
- classify_contacts_by_similarity(): Cosine similarity-based classification (3 channels)
- classify_contacts_by_similarity_multimodality(): 5-channel multimodal classification
- fingerprint_jacobian_multimodality(): End-to-end multimodal fingerprint helper
- classify_jacobian_contacts(): Dispatcher for classification methods
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from .constants import (
    BASE_PAIR_ARGMAX,
    REPEAT_ARGMAX,
    MINERVA_BP_FINGERPRINT,
    MINERVA_BP_FORWARD_FINGERPRINT,
    MINERVA_BP_REVERSE_FINGERPRINT,
    MINERVA_REPEAT_FINGERPRINT,
    MINERVA_BP_CUTOFF,
    MINERVA_REPEAT_CUTOFF,
    MINERVA_PROTEIN_CUTOFF,
    MINERVA_AA_ORDER,
    MINERVA_PROTEIN_FINGERPRINT,
)


_CHANNEL_ALIASES = {
    "base_pair": "basepairing",
    "base_pairing": "basepairing",
    "basepair": "basepairing",
    "bp": "basepairing",
}


def _canonical_channel_name(name: str) -> str:
    key = str(name).lower()
    return _CHANNEL_ALIASES.get(key, key)


@dataclass
class FingerprintResult:
    """Named fingerprint channels produced from a categorical Jacobian.

    ``fingerprints`` is a channel-first tensor/array with shape ``(C, L, L)``.
    Channels can be accessed either by name or by index:

    >>> result["basepairing"]
    >>> result["base_pairing"]  # accepted alias
    >>> result[0]
    """

    tokens: List[str]
    fingerprints: Union[torch.Tensor, np.ndarray]
    channel_names: List[str]
    contacts: Optional[Union[torch.Tensor, np.ndarray]] = None
    jacobian: Optional[Union[torch.Tensor, np.ndarray]] = None
    method: str = "similarity_multimodality"

    def __post_init__(self):
        if len(self.channel_names) != int(self.fingerprints.shape[0]):
            raise ValueError(
                "channel_names length must match fingerprints channel dimension: "
                f"{len(self.channel_names)} != {int(self.fingerprints.shape[0])}"
            )

    @property
    def channels(self) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Mapping from channel name to the corresponding ``(L, L)`` map."""
        return {
            name: self.fingerprints[i]
            for i, name in enumerate(self.channel_names)
        }

    def _resolve_channel(self, name: str) -> str:
        target = _canonical_channel_name(name)
        for channel in self.channel_names:
            if _canonical_channel_name(channel) == target:
                return channel
        raise KeyError(
            f"Unknown fingerprint channel {name!r}; available channels: "
            f"{self.channel_names}"
        )

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.channels[self._resolve_channel(key)]
        return self.fingerprints[key]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self.channels.keys()

    def items(self):
        return self.channels.items()


def _fp(fingerprints, key, default):
    """Return fingerprints[key] if provided (dict), else the module-constant default."""
    if fingerprints is not None and key in fingerprints:
        return fingerprints[key]
    return default


# =============================================================================
# Jacobian to Contact Map Conversion
# =============================================================================

def jac_to_contact(
    jac: Union[np.ndarray, torch.Tensor],
    symm: bool = True,
    center: bool = True,
    diag: str = "remove",
    apc: bool = True,
) -> torch.Tensor:
    """
    Convert a Jacobian tensor to a contact map.

    Computes the Frobenius norm over the alphabet dimensions to get coupling
    strength between positions, then applies standard contact map corrections.

    This is the single canonical contact-map implementation. It runs entirely
    in torch and keeps the result on the input tensor's device, avoiding a
    GPU->CPU copy of the (potentially multi-GB) Jacobian.

    Args:
        jac: Jacobian tensor of shape (Lx, Ax, Ly, Ay) where:
            - Lx, Ly: sequence positions
            - Ax, Ay: alphabet dimensions (e.g., 4 for nucleotides, 20 for amino acids)
            Can be numpy array or torch tensor.
        symm: Whether to symmetrize the contact map (average with transpose).
        center: Whether to center the Jacobian before computing norms
            (subtract mean along each dimension, applied sequentially).
        diag: How to handle the diagonal:
            - "remove": Zero out the diagonal
            - "normalize": Normalize by diagonal values
            - None or other: Leave diagonal unchanged
        apc: Whether to apply Average Product Correction.

    Returns:
        Contact map as a torch tensor of shape (Lx, Ly), on the same device as
        the input (CPU for numpy input). Call ``.cpu().numpy()`` if a numpy
        array is required.

    Example:
        >>> jac = model.get_categorical_jacobian(sequence, tokenizer, ...)[0]
        >>> contact_map = jac_to_contact(jac, symm=True, apc=True)
    """
    # Accept numpy or torch; compute in float32 on the input's device.
    if isinstance(jac, np.ndarray):
        X = torch.from_numpy(jac).to(torch.float32)
    else:
        X = jac.detach().to(torch.float32)

    # Center the Jacobian along each dimension. Applied sequentially: each step
    # operates on the already-centered tensor (this yields the cross-term
    # corrections of multi-way centering). None of these ops mutate the input.
    if center:
        for i in range(4):
            if X.shape[i] > 1:
                X = X - X.mean(dim=i, keepdim=True)

    # Frobenius norm over the alphabet dimensions (1, 3)
    contacts = torch.sqrt((X ** 2).sum(dim=(1, 3)))

    # Symmetrize
    if symm:
        contacts = (contacts + contacts.T) / 2

    # Handle diagonal
    if diag == "remove":
        contacts.fill_diagonal_(0)
    elif diag == "normalize":
        contacts_diag = torch.diag(contacts)
        eps = 1e-10
        norm_factor = torch.sqrt(
            contacts_diag.unsqueeze(1) * contacts_diag.unsqueeze(0) + eps
        )
        contacts = contacts / norm_factor

    # Apply Average Product Correction
    if apc:
        row_sum = contacts.sum(dim=0, keepdim=True)
        col_sum = contacts.sum(dim=1, keepdim=True)
        total_sum = contacts.sum() + 1e-10
        ap = row_sum * col_sum / total_sum
        contacts = contacts - ap

    # Zero diagonal again after APC if needed
    if diag == "remove":
        contacts.fill_diagonal_(0)

    return contacts


# =============================================================================
# Contact Classification Functions
# =============================================================================

def _get_nuc_indices(tokens: List[str]) -> torch.Tensor:
    """
    Get indices of nucleotide positions in token list.

    Returns a boolean mask indicating which positions are nucleotides.
    Uses lowercase convention: nucleotides are 'a', 'c', 'g', 't'.
    Uppercase letters (e.g. 'A', 'C', 'G', 'T') are amino acids and excluded.
    """
    nuc_set = {'a', 'c', 'g', 't'}
    return torch.tensor([t in nuc_set for t in tokens], dtype=torch.bool)


def _get_nuc_token_map(tokens: List[str]) -> torch.Tensor:
    """
    Map token strings to nucleotide channel indices (0=a, 1=t, 2=g, 3=c).

    Returns tensor of indices, with -1 for non-nucleotide tokens.
    Uses lowercase convention: only 'a', 'c', 'g', 't' are mapped.
    Channel ordering: ['a', 't', 'g', 'c'].
    """
    nuc_to_idx = {'a': 0, 't': 1, 'g': 2, 'c': 3}
    return torch.tensor([nuc_to_idx.get(t, -1) for t in tokens], dtype=torch.long)


def classify_contacts_by_argmax(
    J: torch.Tensor,
    contact: torch.Tensor,
    tokens: List[str],
    fingerprints: Optional[dict] = None,
) -> torch.Tensor:
    """
    Classify contacts using argmax pattern matching.

    Centers the Jacobian along the input-alphabet dimension (dim 1), which
    recovers a meaningful pattern in the zeroed self-substitution row via
    negation of the mean. Then checks whether ALL 4 rows of each 4x4 block
    match the base-pair or repeat argmax pattern.

    Args:
        J: Jacobian tensor of shape (L, A, L, A) where A >= 4 for nucleotides.
        contact: Contact map tensor of shape (L, L).
        tokens: List of token strings for each position.

    Returns:
        Fingerprint tensor of shape (3, L, L) where:
            - Channel 0: Base pair contacts (Watson-Crick pairing pattern)
            - Channel 1: Repeat contacts (identity pattern)
            - Channel 2: Other contacts (neither pattern)

        Contact values are preserved and distributed to appropriate channels.
    """
    if J.ndim != 4:
        raise ValueError(f"Expected Jacobian with 4 dims (L, A, L, A), got shape {tuple(J.shape)}")
    if J.shape[1] == 1:
        raise ValueError(
            "Received fast-mode Jacobian (shape Lx1xLxA). "
            "Argmax fingerprinting requires full substitution-channel Jacobian blocks. "
            "Re-run get_categorical_jacobian(..., fast=False)."
        )

    device = J.device
    L = J.shape[0]

    # Center along dim 1 — recovers self-sub row pattern via -mean
    jac = J - J.mean(1, keepdim=True)

    # Subset to nuc channels, permute to (L, L, 4_in, 4_out)
    slices = jac[:, :4, :, :4].permute(0, 2, 1, 3)  # (L, L, 4, 4)

    # Argmax per input channel row
    argmaxs = slices.argmax(dim=-1)  # (L, L, 4)

    # Pattern matching — require ALL 4 rows to match
    bp_pattern = _fp(fingerprints, 'bp_argmax', BASE_PAIR_ARGMAX).to(device)
    rep_pattern = _fp(fingerprints, 'rep_argmax', REPEAT_ARGMAX).to(device)

    is_bp = (argmaxs == bp_pattern).all(dim=-1)   # (L, L)
    is_rep = (argmaxs == rep_pattern).all(dim=-1)  # (L, L)

    # Nucleotide mask — only nuc-nuc pairs can be bp/repeat
    nuc_set = {'a', 'c', 'g', 't'}
    is_nuc = torch.tensor([t in nuc_set for t in tokens], device=device)
    is_nuc_ij = is_nuc[:, None] & is_nuc[None, :]

    # Classify: bp priority on ties, mask by nuc identity
    fingerprints_idx = torch.full((L, L), 2, dtype=torch.long, device=device)
    fingerprints_idx[is_nuc_ij & is_bp] = 0
    fingerprints_idx[is_nuc_ij & is_rep & ~is_bp] = 1

    # One-hot → 3-channel mask
    one_hot = F.one_hot(fingerprints_idx, num_classes=3)  # (L, L, 3)
    mask = one_hot.permute(2, 0, 1).to(contact.dtype)     # (3, L, L)

    return mask * contact


def classify_contacts_by_similarity(
    J: torch.Tensor,
    contact: torch.Tensor,
    tokens: List[str],
    bp_threshold: float = 0.65,
    rep_threshold: float = 0.80,
    bp_pattern: Optional[torch.Tensor] = None,
    rep_pattern: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Classify contacts using cosine similarity to reference patterns.

    Subsets the Jacobian to nucleotide channels, symmetrizes, demeans along
    all 4 dimensions, then computes cosine similarity of each flattened 4x4
    block against the base-pair and repeat reference patterns.

    Args:
        J: Jacobian tensor of shape (L, A, L, A) where A >= 4 for nucleotides.
        contact: Contact map tensor of shape (L, L).
        tokens: List of token strings for each position.
        bp_threshold: Minimum cosine similarity to classify as base pair
            (default 0.65). Contacts below threshold go to "other".
        rep_threshold: Minimum cosine similarity to classify as repeat
            (default 0.80). Contacts below threshold go to "other".
        bp_pattern: Reference 4x4 pattern for base pairs. Defaults to
            MINERVA_BP_FINGERPRINT if None.
        rep_pattern: Reference 4x4 pattern for repeats. Defaults to
            MINERVA_REPEAT_FINGERPRINT if None.

    Returns:
        Fingerprint tensor of shape (3, L, L) where:
            - Channel 0: Base pair contacts
            - Channel 1: Repeat contacts
            - Channel 2: Other contacts

        Contact values are preserved and distributed to appropriate channels.
    """
    if J.ndim != 4:
        raise ValueError(f"Expected Jacobian with 4 dims (L, A, L, A), got shape {tuple(J.shape)}")
    if J.shape[1] == 1:
        raise ValueError(
            "Received fast-mode Jacobian (shape Lx1xLxA). "
            "Similarity fingerprinting requires full substitution-channel Jacobian blocks. "
            "Re-run get_categorical_jacobian(..., fast=False)."
        )

    device = J.device
    L = J.shape[0]

    # Subset to nuc channels
    jac_nuc = J[:, :4, :, :4].clone()

    # Symmetrize: make (i,j) and (j,i) blocks consistent
    jac_nuc = (jac_nuc + jac_nuc.permute(2, 3, 0, 1)) / 2

    # Demean along all 4 dimensions
    for dim in range(4):
        if jac_nuc.shape[dim] > 1:
            jac_nuc = jac_nuc - jac_nuc.mean(dim=dim, keepdim=True)

    # Reshape to (L, L, 16) for vectorized cosine similarity
    blocks = jac_nuc.permute(0, 2, 1, 3).reshape(L, L, 16)

    # Normalize blocks (F.normalize handles zero vectors via eps)
    blocks_norm = F.normalize(blocks, dim=-1)

    # Normalized reference patterns
    _bp = bp_pattern if bp_pattern is not None else MINERVA_BP_FINGERPRINT
    _rep = rep_pattern if rep_pattern is not None else MINERVA_REPEAT_FINGERPRINT
    bp_ref = F.normalize(_bp.to(device).flatten(), dim=0)
    rep_ref = F.normalize(_rep.to(device).flatten(), dim=0)

    # Cosine similarity against both patterns
    cos_bp = (blocks_norm * bp_ref).sum(dim=-1)   # (L, L)
    cos_rep = (blocks_norm * rep_ref).sum(dim=-1)  # (L, L)

    # Zero out similarities for zero-norm blocks (no signal)
    block_norms = blocks.norm(dim=-1)
    zero_mask = block_norms < 1e-8
    cos_bp[zero_mask] = 0
    cos_rep[zero_mask] = 0

    # Classification: bp wins ties when both above threshold
    is_bp = (cos_bp > bp_threshold) & (cos_bp >= cos_rep)
    is_rep = (cos_rep > rep_threshold) & ~is_bp

    # Nucleotide mask — only nuc-nuc pairs can be bp/repeat
    nuc_set = {'a', 'c', 'g', 't'}
    is_nuc = torch.tensor([t in nuc_set for t in tokens], device=device)
    is_nuc_ij = is_nuc[:, None] & is_nuc[None, :]

    # Classify
    fingerprints_idx = torch.full((L, L), 2, dtype=torch.long, device=device)
    fingerprints_idx[is_nuc_ij & is_bp] = 0
    fingerprints_idx[is_nuc_ij & is_rep] = 1

    # One-hot → 3-channel mask
    one_hot = F.one_hot(fingerprints_idx, num_classes=3)  # (L, L, 3)
    mask = one_hot.permute(2, 0, 1).to(contact.dtype)     # (3, L, L)

    return mask * contact


def classify_contacts_by_similarity_multimodality(
    J: torch.Tensor,
    contact: torch.Tensor,
    tokens: List[str],
    bp_threshold: float = MINERVA_BP_CUTOFF,
    repeat_threshold: float = MINERVA_REPEAT_CUTOFF,
    protein_threshold: float = MINERVA_PROTEIN_CUTOFF,
    aa_start: int = 4,
    jac_aa_order: Optional[List[str]] = None,
    split_bp: bool = False,
    fingerprints: Optional[dict] = None,
) -> torch.Tensor:
    """
    Classify contacts into modality channels using cosine similarity
    to Minerva reference fingerprints.

    For nucleotide-nucleotide pairs: subsets the Jacobian to the first 4
    channels (a, t, g, c), symmetrizes and centers, then computes cosine
    similarity against Minerva base-pairing and repeat fingerprints.
    Optionally splits base-pairing into forward and reverse by taking the
    argmax over the forward and reverse fingerprint similarities.

    For amino acid-amino acid pairs: subsets the Jacobian to the 20 amino
    acid channels, symmetrizes and centers, then computes cosine similarity
    against the Minerva protein interaction fingerprint.

    Args:
        J: Jacobian tensor of shape (L, A, L, A) where A >= 24
            (4 nucleotide + 20 amino acid channels).
        contact: Contact map tensor of shape (L, L).
        tokens: List of token strings for each position.
            Lowercase (a, t, g, c) = nucleotides.
            Uppercase (A, C, D, ..., Y) = amino acids.
        bp_threshold: Cosine similarity cutoff for base-pairing (default 0.35).
        repeat_threshold: Cosine similarity cutoff for repeat (default 0.2).
        protein_threshold: Cosine similarity cutoff for protein interaction
            (default 0.07).
        aa_start: Start index of amino acid channels in the Jacobian
            (default 4, i.e. channels 4:24 are the 20 amino acids).
        jac_aa_order: List of 20 amino acid single-letter codes in the order
            they appear in the Jacobian channels (starting at aa_start).
            If None, assumes minerva ordering (ACFILMVWYPHKRDENQSTG).
            Pass e.g. list('ACDEFGHIKLMNPQRSTVWY') for alphabetical order.
        split_bp: Whether to split base-pairing into forward and reverse
            channels (default False). When False, returns 4 channels:
            [basepairing, repeat, protein, other]. When True, returns 5
            channels: [bp_forward, bp_reverse, repeat, protein, other].

    Returns:
        Fingerprint tensor of shape (C, L, L) where C=4 (split_bp=False)
        or C=5 (split_bp=True):
            split_bp=False: [basepairing, repeat, protein, other]
            split_bp=True:  [bp_forward, bp_reverse, repeat, protein, other]

        Contact values are preserved and distributed to appropriate channels.
    """
    if J.ndim != 4:
        raise ValueError(
            f"Expected Jacobian with 4 dims (L, A, L, A), got shape {tuple(J.shape)}"
        )
    if J.shape[1] == 1:
        raise ValueError(
            "Received fast-mode Jacobian (shape Lx1xLxA). "
            "Multimodal fingerprinting requires full Jacobian channels. "
            "Re-run get_categorical_jacobian(..., fast=False)."
        )
    if J.shape[1] < 4 or J.shape[3] < 4:
        raise ValueError(
            f"Jacobian has too few channels for nucleotide fingerprinting: {tuple(J.shape)}"
        )

    device = J.device
    L = J.shape[0]
    aa_end = aa_start + 20

    # Build protein fingerprint in the Jacobian's AA channel ordering
    _protein_fp = _fp(fingerprints, 'protein', MINERVA_PROTEIN_FINGERPRINT)
    if jac_aa_order is not None and list(jac_aa_order) != MINERVA_AA_ORDER:
        perm = [MINERVA_AA_ORDER.index(aa) for aa in jac_aa_order]
        prot_fingerprint = _protein_fp[perm][:, perm]
    else:
        prot_fingerprint = _protein_fp

    # --- Nucleotide analysis ---
    # Subset to nucleotide channels (first 4: a, t, g, c)
    jac_nuc = J[:, :4, :, :4].clone()

    # Symmetrize
    jac_nuc = (jac_nuc + jac_nuc.permute(2, 3, 0, 1)) / 2

    # Demean along all 4 dimensions
    for dim in range(4):
        if jac_nuc.shape[dim] > 1:
            jac_nuc = jac_nuc - jac_nuc.mean(dim=dim, keepdim=True)

    # Reshape to (L, L, 16) for vectorized cosine similarity
    nuc_blocks = jac_nuc.permute(0, 2, 1, 3).reshape(L, L, 16)
    nuc_blocks_norm = F.normalize(nuc_blocks, dim=-1)

    # Cosine similarity against nucleotide fingerprints
    bp_ref = F.normalize(_fp(fingerprints, 'bp', MINERVA_BP_FINGERPRINT).to(device).flatten(), dim=0)
    rep_ref = F.normalize(_fp(fingerprints, 'repeat', MINERVA_REPEAT_FINGERPRINT).to(device).flatten(), dim=0)

    cos_bp = (nuc_blocks_norm * bp_ref).sum(dim=-1)       # (L, L)
    cos_rep = (nuc_blocks_norm * rep_ref).sum(dim=-1)      # (L, L)

    if split_bp:
        bp_fwd_ref = F.normalize(_fp(fingerprints, 'bp_fwd', MINERVA_BP_FORWARD_FINGERPRINT).to(device).flatten(), dim=0)
        bp_rev_ref = F.normalize(_fp(fingerprints, 'bp_rev', MINERVA_BP_REVERSE_FINGERPRINT).to(device).flatten(), dim=0)
        cos_bp_fwd = (nuc_blocks_norm * bp_fwd_ref).sum(dim=-1)  # (L, L)
        cos_bp_rev = (nuc_blocks_norm * bp_rev_ref).sum(dim=-1)  # (L, L)

    # Zero out similarities for zero-norm blocks
    nuc_norms = nuc_blocks.norm(dim=-1)
    zero_nuc = nuc_norms < 1e-8
    cos_bp[zero_nuc] = 0
    cos_rep[zero_nuc] = 0
    if split_bp:
        cos_bp_fwd[zero_nuc] = 0
        cos_bp_rev[zero_nuc] = 0

    # --- Protein analysis ---
    # Subset to amino acid channels (20 channels starting at aa_start)
    # Guard: skip protein analysis if the Jacobian doesn't have enough channels
    has_protein_channels = J.shape[1] >= aa_end
    if has_protein_channels:
        jac_aa = J[:, aa_start:aa_end, :, aa_start:aa_end].clone()

        # Symmetrize
        jac_aa = (jac_aa + jac_aa.permute(2, 3, 0, 1)) / 2

        # Demean along all 4 dimensions
        for dim in range(4):
            if jac_aa.shape[dim] > 1:
                jac_aa = jac_aa - jac_aa.mean(dim=dim, keepdim=True)

        # Reshape to (L, L, 400) for vectorized cosine similarity
        aa_blocks = jac_aa.permute(0, 2, 1, 3).reshape(L, L, 400)
        aa_blocks_norm = F.normalize(aa_blocks, dim=-1)

        # Cosine similarity against protein fingerprint
        prot_ref = F.normalize(prot_fingerprint.to(device).flatten(), dim=0)
        cos_prot = (aa_blocks_norm * prot_ref).sum(dim=-1)  # (L, L)

        # Zero out for zero-norm blocks
        aa_norms = aa_blocks.norm(dim=-1)
        zero_aa = aa_norms < 1e-8
        cos_prot[zero_aa] = 0
    else:
        cos_prot = torch.zeros(L, L, device=device)

    # --- Position masks ---
    nuc_set = {'a', 'c', 'g', 't'}
    aa_set = set('ACDEFGHIKLMNPQRSTVWY')
    is_nuc = torch.tensor([t in nuc_set for t in tokens], device=device)
    is_aa = torch.tensor([t in aa_set for t in tokens], device=device)
    is_nuc_ij = is_nuc[:, None] & is_nuc[None, :]
    is_aa_ij = is_aa[:, None] & is_aa[None, :]

    # --- Classification ---
    is_bp = is_nuc_ij & (cos_bp > bp_threshold)

    # Repeat: cosine sim > repeat_threshold, bp takes priority
    is_rep = is_nuc_ij & (cos_rep > repeat_threshold) & ~is_bp

    # Protein interaction: cosine sim > protein_threshold
    is_prot = is_aa_ij & (cos_prot > protein_threshold)

    if split_bp:
        # 5 channels: [bp_forward, bp_reverse, repeat, protein, other]
        is_bp_fwd = is_bp & (cos_bp_fwd >= cos_bp_rev)
        is_bp_rev = is_bp & (cos_bp_rev > cos_bp_fwd)

        num_classes = 5
        fingerprints_idx = torch.full((L, L), 4, dtype=torch.long, device=device)
        fingerprints_idx[is_prot] = 3
        fingerprints_idx[is_rep] = 2
        fingerprints_idx[is_bp_rev] = 1
        fingerprints_idx[is_bp_fwd] = 0
    else:
        # 4 channels: [basepairing, repeat, protein, other]
        num_classes = 4
        fingerprints_idx = torch.full((L, L), 3, dtype=torch.long, device=device)
        fingerprints_idx[is_prot] = 2
        fingerprints_idx[is_rep] = 1
        fingerprints_idx[is_bp] = 0

    # One-hot → C-channel mask
    one_hot = F.one_hot(fingerprints_idx, num_classes=num_classes)
    mask = one_hot.permute(2, 0, 1).to(contact.dtype)

    return mask * contact


def multimodality_channel_names(split_bp: bool = False) -> List[str]:
    """Return ordered channel names for multimodal fingerprint outputs."""
    if split_bp:
        return ["bp_forward", "bp_reverse", "repeat", "protein", "other"]
    return ["basepairing", "repeat", "protein", "other"]


def fingerprint_jacobian_multimodality(
    jac: Union[np.ndarray, torch.Tensor],
    tokens: List[str],
    symm: bool = True,
    center: bool = True,
    diag: str = "remove",
    apc: bool = True,
    bp_threshold: float = MINERVA_BP_CUTOFF,
    repeat_threshold: float = MINERVA_REPEAT_CUTOFF,
    protein_threshold: float = MINERVA_PROTEIN_CUTOFF,
    aa_start: int = 4,
    jac_aa_order: Optional[List[str]] = None,
    split_bp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    End-to-end multimodal Jacobian fingerprinting helper.

    Computes a contact map from the Jacobian and then classifies each contact
    into multimodal channels using Minerva reference fingerprints.

    Args:
        jac: Jacobian of shape (L, A, L, A), numpy array or torch tensor.
        tokens: Token list aligned to Jacobian positions.
        symm, center, diag, apc: Contact-map conversion options.
        bp_threshold, repeat_threshold, protein_threshold: Similarity cutoffs.
        aa_start: AA-channel start index in Jacobian (default 4).
        jac_aa_order: Optional AA channel order for Jacobian channels.
        split_bp: If True, split basepairing into forward/reverse channels.

    Returns:
        Tuple ``(contacts, fingerprints, channel_names)`` where:
            - contacts: numpy array, shape (L, L)
            - fingerprints: numpy array, shape (C, L, L), C=4 or 5
            - channel_names: ordered list of channel names
    """
    if isinstance(jac, torch.Tensor):
        jac_t = jac.detach()
    else:
        jac_t = torch.as_tensor(jac)
    jac_t = jac_t.to(dtype=torch.float32)

    if jac_t.ndim != 4:
        raise ValueError(
            f"Expected Jacobian with 4 dims (L, A, L, A), got shape {tuple(jac_t.shape)}"
        )
    if jac_t.shape[1] == 1:
        raise ValueError(
            "Received fast-mode Jacobian (shape Lx1xLxA). "
            "Set fast=False when calling get_categorical_jacobian for multimodal fingerprinting."
        )

    contacts_t = jac_to_contact(jac_t, symm=symm, center=center, diag=diag, apc=apc).to(
        device=jac_t.device,
        dtype=jac_t.dtype,
    )

    fingerprints_t = classify_contacts_by_similarity_multimodality(
        jac_t,
        contacts_t,
        tokens,
        bp_threshold=bp_threshold,
        repeat_threshold=repeat_threshold,
        protein_threshold=protein_threshold,
        aa_start=aa_start,
        jac_aa_order=jac_aa_order,
        split_bp=split_bp,
    )

    return (
        contacts_t.detach().cpu().numpy(),
        fingerprints_t.detach().cpu().numpy(),
        multimodality_channel_names(split_bp=split_bp),
    )


def classify_jacobian_contacts(
    J: torch.Tensor,
    contact: torch.Tensor,
    tokens: List[str],
    method: str = "argmax",
    fingerprints: Optional[dict] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Classify contacts in a Jacobian-derived contact map.

    This is the main dispatcher function for contact classification. It routes
    to the appropriate classification method based on the `method` parameter.

    Args:
        J: Jacobian tensor of shape (L, A, L, A).
        contact: Contact map tensor of shape (L, L).
        tokens: List of token strings for each position.
        method: Classification method:
            - "argmax": Argmax-based classification (default)
            - "similarity": Cosine similarity-based classification
            - "similarity_multimodality": 5-channel multimodal classification
        **kwargs: Additional arguments passed to the classification method:
            - For "similarity": threshold (float, default 0.7)
            - For "similarity_multimodality": bp_threshold, repeat_threshold,
              protein_threshold, aa_start, split_bp

    Returns:
        Fingerprint tensor of shape (C, L, L) where C=3 for argmax/similarity
        or C=4/5 for similarity_multimodality:
            argmax/similarity: [base_pair, repeat, other]
            similarity_multimodality (split_bp=False): [basepairing, repeat,
                                       protein, other]
            similarity_multimodality (split_bp=True): [bp_forward, bp_reverse,
                                       repeat, protein, other]

    Example:
        >>> jac, tokens = model.get_categorical_jacobian(sequence, tokenizer, ...)
        >>> contact = jac_to_contact(jac)  # torch tensor on jac's device
        >>> fingerprints = classify_jacobian_contacts(jac, contact, tokens)
        >>> base_pairs = fingerprints[0]
        >>> repeats = fingerprints[1]
        >>> other = fingerprints[2]
    """
    if method == "argmax":
        return classify_contacts_by_argmax(J, contact, tokens, fingerprints=fingerprints)
    elif method == "similarity":
        bp_threshold = kwargs.get("bp_threshold", 0.65)
        rep_threshold = kwargs.get("rep_threshold", 0.80)
        bp_pattern = kwargs.get("bp_pattern", None)
        rep_pattern = kwargs.get("rep_pattern", None)
        return classify_contacts_by_similarity(
            J, contact, tokens,
            bp_threshold=bp_threshold, rep_threshold=rep_threshold,
            bp_pattern=bp_pattern, rep_pattern=rep_pattern,
        )
    elif method == "similarity_multimodality":
        return classify_contacts_by_similarity_multimodality(J, contact, tokens, fingerprints=fingerprints, **kwargs)
    else:
        raise ValueError(
            f"Unknown classification method: {method}. "
            "Use 'argmax', 'similarity', or 'similarity_multimodality'."
        )

def fingerprint_channel_names(method: str = "similarity_multimodality", split_bp: bool = False) -> List[str]:
    """Return ordered channel names for a fingerprint classification method."""
    if method in ("argmax", "similarity"):
        return ["basepairing", "repeat", "other"]
    if method == "similarity_multimodality":
        return multimodality_channel_names(split_bp=split_bp)
    raise ValueError(
        f"Unknown classification method: {method}. "
        "Use 'argmax', 'similarity', or 'similarity_multimodality'."
    )


def compute_fingerprints(
    jac: Union[np.ndarray, torch.Tensor],
    tokens: List[str],
    *,
    method: str = "similarity_multimodality",
    contact: Optional[Union[np.ndarray, torch.Tensor]] = None,
    symm: bool = True,
    center: bool = True,
    diag: str = "remove",
    apc: bool = True,
    fingerprints: Optional[dict] = None,
    include_jacobian: bool = False,
    **kwargs,
) -> FingerprintResult:
    """Compute named fingerprint channels from a full categorical Jacobian.

    Args:
        jac: Full categorical Jacobian with shape ``(L, A, L, A)``. Fast-mode
            Jacobians with shape ``(L, 1, L, A)`` are not sufficient.
        tokens: Token strings aligned to the Jacobian positions.
        method: ``"similarity_multimodality"`` (default), ``"argmax"``, or
            ``"similarity"``.
        contact: Optional precomputed contact map. If omitted, it is computed
            from ``jac`` with :func:`jac_to_contact`.
        symm, center, diag, apc: Contact-map conversion options used only when
            ``contact`` is omitted.
        fingerprints: Optional reference fingerprint matrices. The model-level
            ``get_fingerprints`` method passes the checkpoint-baked references.
        include_jacobian: Store the input Jacobian on the returned result.
        **kwargs: Method-specific classifier options, such as
            ``bp_threshold``, ``repeat_threshold``, ``protein_threshold``,
            ``aa_start``, ``jac_aa_order``, and ``split_bp``.

    Returns:
        :class:`FingerprintResult` with named channels and the contact map.
    """
    if isinstance(jac, torch.Tensor):
        jac_t = jac.detach()
    else:
        jac_t = torch.as_tensor(jac)
    jac_t = jac_t.to(dtype=torch.float32)

    if jac_t.ndim != 4:
        raise ValueError(
            f"Expected Jacobian with 4 dims (L, A, L, A), got shape {tuple(jac_t.shape)}"
        )
    if jac_t.shape[1] == 1:
        raise ValueError(
            "Received fast-mode Jacobian (shape Lx1xLxA). "
            "Fingerprinting requires full substitution-channel Jacobian blocks. "
            "Re-run get_categorical_jacobian(..., fast=False)."
        )

    if contact is None:
        contact_t = jac_to_contact(jac_t, symm=symm, center=center, diag=diag, apc=apc).to(
            device=jac_t.device,
            dtype=jac_t.dtype,
        )
    elif isinstance(contact, torch.Tensor):
        contact_t = contact.detach().to(device=jac_t.device, dtype=jac_t.dtype)
    else:
        contact_t = torch.as_tensor(contact, device=jac_t.device, dtype=jac_t.dtype)

    fingerprint_t = classify_jacobian_contacts(
        jac_t,
        contact_t,
        tokens,
        method=method,
        fingerprints=fingerprints,
        **kwargs,
    )
    channel_names = fingerprint_channel_names(
        method=method,
        split_bp=bool(kwargs.get("split_bp", False)),
    )

    return FingerprintResult(
        tokens=list(tokens),
        fingerprints=fingerprint_t,
        channel_names=channel_names,
        contacts=contact_t,
        jacobian=jac if include_jacobian else None,
        method=method,
    )

