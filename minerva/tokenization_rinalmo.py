"""HF tokenizer matching RiNALMo's Alphabet.

Upstream ships a custom Alphabet, but AutoTokenizer, the MLM collators and the
block-splitting all need a real HF tokenizer.
"""

from typing import Optional

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from .modeling_rinalmo import RNA_TOKENS, SPECIAL_TOKENS


def build_rinalmo_tokenizer(save_to: Optional[str] = None) -> PreTrainedTokenizerFast:
    """One token per character, `<cls> ... <eos>`, U folded onto T.

    Upstream's encode() upper-cases and rewrites U to T -- there is no U token --
    so both are done here or ids would not match the pretrained embeddings.
    """
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for token in RNA_TOKENS:
        vocab[token] = len(vocab)

    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    # `tokenizers` has no upper-casing normalizer, and aliasing lower case into
    # the vocab would overwrite the upper-case entries -- WordLevel is a bijection.
    backend.normalizer = normalizers.Sequence(
        [normalizers.Replace(t.lower(), t) for t in RNA_TOKENS if t.isalpha()]
        + [normalizers.Replace("u", "T"), normalizers.Replace("U", "T")]
    )
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.post_processor = processors.TemplateProcessing(
        single="<cls> $A <eos>",
        special_tokens=[("<cls>", vocab["<cls>"]), ("<eos>", vocab["<eos>"])],
    )
    backend.decoder = decoders.Fuse()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        cls_token="<cls>",
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        mask_token="<mask>",
    )
    if save_to:
        tokenizer.save_pretrained(save_to)
    return tokenizer
