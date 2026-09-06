import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


def grouped_mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    groups: Sequence,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CE with each group's loss divided by log(alphabet_size).

    Keeps a 20-letter protein alphabet from dominating a 4-letter nucleotide one.
    Tokens no group claims take the narrowest scale. No groups -> plain CE.
    """
    target = labels.long().view(-1)
    flat_logits = logits.view(-1, logits.size(-1))
    valid = target != ignore_index

    if not groups:
        return F.cross_entropy(flat_logits, target, ignore_index=ignore_index), {}

    ce = F.cross_entropy(flat_logits, target, ignore_index=ignore_index, reduction="none")
    device = logits.device
    total = torch.zeros((), device=device)
    matched = torch.zeros_like(valid)
    metrics: Dict[str, float] = {}

    for group in groups:
        ids = torch.tensor(group.token_ids, dtype=target.dtype, device=device)
        mask = torch.isin(target, ids) & valid
        matched |= mask
        n = mask.sum()
        raw = ce[mask].sum() if n > 0 else torch.zeros((), device=device)
        scaled = raw / math.log(group.alphabet_size)
        total = total + scaled
        metrics[f"{group.name}_loss"] = (scaled / n.clamp(min=1)).item()
        metrics[f"n_{group.name}_tokens"] = int(n)

    other = valid & ~matched
    n_other = other.sum()
    if n_other > 0:
        scaled = ce[other].sum() / math.log(min(g.alphabet_size for g in groups))
        total = total + scaled
        metrics["other_loss"] = (scaled / n_other).item()
    metrics["n_other_tokens"] = int(n_other)

    return total / valid.sum().clamp(min=1), metrics
