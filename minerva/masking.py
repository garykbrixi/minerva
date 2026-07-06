import torch
from dataclasses import dataclass
from typing import Optional, List
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

def mask_with_tokens(tokens: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return (tokens.unsqueeze(-1) == token_ids.view(1, 1, -1)).any(-1)

def get_mask_subset_with_prob(mask: torch.Tensor, prob: float) -> torch.Tensor:
    rand = torch.rand_like(mask, dtype=torch.float32)
    return (rand < prob) & mask

@dataclass
class DataCollatorForMinervaMLM(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    mask_prob: float = 0.30
    random_token_prob: float = 0.0
    replace_prob: float = 0.9
    mask_ignore_token_ids: Optional[List[int]] = None

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        input_ids = batch["input_ids"]
        device = input_ids.device

        ignore_ids = torch.tensor(
            [self.tokenizer.pad_token_id] if self.mask_ignore_token_ids is None else self.mask_ignore_token_ids,
            device=device,
        )

        no_mask = mask_with_tokens(input_ids, ignore_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        labels = input_ids.masked_fill(~mask, self.tokenizer.pad_token_id)
        masked = input_ids.clone()

        if self.random_token_prob > 0:
            rand_mask = get_mask_subset_with_prob(mask, self.random_token_prob)
            vocab_size = self.tokenizer.vocab_size
            random_tokens = torch.randint(0, vocab_size, input_ids.shape, device=device)
            masked = torch.where(rand_mask, random_tokens, masked)
            mask = mask & ~rand_mask
            replace_mask = get_mask_subset_with_prob(mask, self.replace_prob)
            masked = masked.masked_fill(replace_mask, self.tokenizer.mask_token_id)
        else:
            masked = masked.masked_fill(mask, self.tokenizer.mask_token_id)

        batch["input_ids"] = masked
        batch["labels"] = labels
        return batch