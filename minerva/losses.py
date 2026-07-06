import torch
import torch.nn.functional as F

def _create_token_masks(labels: torch.Tensor, nuc_tokens: torch.Tensor, aa_tokens: torch.Tensor, ignore_token_id: int):
    labels_exp = labels.unsqueeze(-1)
    dna_mask = (labels_exp == nuc_tokens.view(1, -1)).any(-1)
    aa_mask = (labels_exp == aa_tokens.view(1, -1)).any(-1)
    other_mask = ~dna_mask & ~aa_mask & (labels != ignore_token_id)
    return dna_mask, aa_mask, other_mask

def _masked_ce(logits, labels, mask, ignore_index):
    if mask.any():
        return F.cross_entropy(
            logits[mask].view(-1, logits.size(-1)),
            labels[mask],
            ignore_index=ignore_index,
        )
    return torch.tensor(0.0, device=logits.device)

def minerva_mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    nuc_tokens: torch.Tensor,
    aa_tokens: torch.Tensor,
    ignore_token_id: int,
    dna_aa_reweighting: bool = False,
    dna_upweight_factor: float = 1.0,
    fixed_factor_upweighting: bool = False,
):
    # logits: (B, L, V); labels: (B, L)
    if logits.dim() == 3:
        loss_input = logits.transpose(1, 2)  # (B, V, L)
        target = labels.long()
    else:
        loss_input = logits
        target = labels.long().view(-1)

    dna_mask, aa_mask, other_mask = _create_token_masks(target, nuc_tokens, aa_tokens, ignore_token_id)
    dna_loss = _masked_ce(logits, target, dna_mask, ignore_token_id)
    aa_loss = _masked_ce(logits, target, aa_mask | other_mask, ignore_token_id)

    if fixed_factor_upweighting:
        dna_loss = dna_loss / torch.log(torch.tensor(4.0, device=logits.device))
        aa_loss = aa_loss / torch.log(torch.tensor(20.0, device=logits.device))
        return dna_loss * dna_upweight_factor + aa_loss

    if dna_aa_reweighting:
        n_dna = dna_mask.sum().float()
        n_aa = (aa_mask | other_mask).sum().float()
        dna_w = n_dna * dna_upweight_factor
        aa_w = n_aa
        total = dna_w + aa_w
        if total > 0:
            dna_w = dna_w / total
            aa_w = aa_w / total
        return dna_loss * dna_w + aa_loss * aa_w

    return dna_loss + aa_loss