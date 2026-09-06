"""
Reusable MLM-finetuning logic for the Minerva model.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset, DatasetDict, load_dataset
from transformers import Trainer

from .backbones import check_token_groups
from .losses import grouped_mlm_loss
from .data import extract_and_tokenize_gb
from .modeling_minerva import MinervaForMaskedLM
from .sequence_utils import chunk_sequence_with_stride


def load_model_from_lightning_ckpt(
    ckpt_path: str,
    config,
    device: str = "cpu",
    legacy_module_path: Optional[str] = None,
):
    """
    Load MinervaForMaskedLM from a PyTorch Lightning .ckpt file.

    Handles the key mapping from Lightning's state_dict format to Minerva's format.
    """
    import sys

    # Add legacy module path if needed (for unpickling old checkpoints)
    if legacy_module_path and legacy_module_path not in sys.path:
        sys.path.insert(0, legacy_module_path)
        print(f"Added legacy module path: {legacy_module_path}")

    print(f"Loading model from Lightning checkpoint: {ckpt_path}")
    model = MinervaForMaskedLM(config)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get("state_dict", state)

    # Map keys from Lightning format to Minerva format
    mapped = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("model."):  # drop Lightning prefix
            nk = nk[len("model."):]
        nk = nk.replace("glm2.", "minerva.")  # align module name
        mapped[nk] = v

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    return model


@dataclass
class ModelArguments:
    """Arguments pertaining to model configuration."""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow custom model code on HuggingFace Hub"}
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to data inputs."""
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a text file)."}
    )
    genbank_file: Optional[str] = field(
        default=None,
        metadata={"help": "Input GenBank file (.gb, .gbk, .genbank) to train on."}
    )
    validation_genbank_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional validation GenBank file."}
    )
    use_existing_translations: bool = field(
        default=False,
        metadata={"help": "Use existing protein translations from GenBank if available."}
    )
    translation_table: int = field(
        default=11,
        metadata={"help": "Default NCBI genetic-code table for CDS features "
                          "lacking a /transl_table qualifier (11 = bacterial/"
                          "archaeal/plant plastid). Per-feature /transl_table "
                          "always takes precedence."}
    )
    max_seq_length: Optional[int] = field(
        default=8192,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )


class MinervaTrainer(Trainer):
    """Custom Trainer that uses Minerva MLM loss with optional token-type upweighting."""

    def __init__(self, *args, token_groups=None, ignore_token_id=-100, token_type_upweighting=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_groups = list(token_groups or [])
        if self.token_groups:
            check_token_groups(self.token_groups)
        self.ignore_token_id = ignore_token_id
        self.token_type_upweighting = token_type_upweighting
        # Store custom metrics to be logged
        self._custom_metrics = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss using minerva_mlm_loss instead of default loss.

        Expected inputs:
        - input_ids: tokenized input sequences
        - labels: masked labels (same as input_ids, with -100 for non-masked tokens)
        - attention_mask: optional attention mask
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        target = labels.long()

        if self.token_type_upweighting and self.token_groups:
            loss, self._custom_metrics = grouped_mlm_loss(
                logits, target, self.token_groups, self.ignore_token_id)
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=self.ignore_token_id,
            )
            self._custom_metrics = {}

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, start_time: float = None) -> None:
        """Override log to inject custom metrics."""
        # Add our custom metrics to whatever is being logged
        if self._custom_metrics:
            logs = {**logs, **self._custom_metrics}
        super().log(logs, start_time)


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the examples."""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
    )
    return result


def _split_into_blocks(dataset: Dataset, block_size: int) -> Dataset:
    """Split each example into consecutive **non-overlapping** blocks.

    A LOCUS longer than ``block_size`` tokens is cut into back-to-back windows
    of ``block_size`` tokens; the final short remainder is kept as its own
    example. Every token appears in exactly one block — no overlap (so no
    duplication / interior oversampling, unlike inference-time scanning) and no
    dropped tail (unlike plain truncation). Applied per split so blocks from one
    LOCUS never straddle the train/validation boundary.
    """
    def _expand(examples):
        out: list[str] = []
        for text in examples["text"]:
            # stride == block_size => non-overlapping, lossless tiling
            out.extend(chunk_sequence_with_stride(text, block_size, block_size))
        return {"text": out}

    return dataset.map(
        _expand, batched=True, remove_columns=dataset.column_names,
        desc=f"Splitting LOCUS into {block_size}-token blocks",
    )


def _build_splits(texts, val_texts=None, validation_split_percentage=5, block_size=None):
    """Train/validation split, then blocking per split.

    Blocking happens after the split so blocks from one record cannot leak
    across the boundary.
    """
    dataset = Dataset.from_dict({"text": texts})
    if val_texts:
        result = DatasetDict({"train": dataset,
                              "validation": Dataset.from_dict({"text": val_texts})})
    else:
        min_for_split = max(2, int(100 / validation_split_percentage) + 1)
        if len(texts) < min_for_split:
            print(f"Dataset too small to split ({len(texts)} samples); training without validation.")
            result = DatasetDict({"train": dataset})
        else:
            split = dataset.train_test_split(test_size=validation_split_percentage / 100, seed=42)
            result = DatasetDict({"train": split["train"], "validation": split["test"]})

    if block_size:
        before = {k: len(v) for k, v in result.items()}
        result = DatasetDict({k: _split_into_blocks(v, block_size) for k, v in result.items()})
        print(f"Split into {block_size}-token blocks: {before} -> "
              f"{ {k: len(v) for k, v in result.items()} }")
    return result


def read_sequences(path: str):
    """Plain nucleotide sequences from FASTA or GenBank, ignoring annotations."""
    from Bio import SeqIO

    fmt = "genbank" if path.lower().endswith((".gb", ".gbk", ".genbank")) else "fasta"
    return [str(record.seq) for record in SeqIO.parse(path, fmt) if len(record.seq)]


def load_sequence_dataset(
    path: str,
    validation_file: Optional[str] = None,
    validation_split_percentage: int = 5,
    block_size: Optional[int] = None,
) -> DatasetDict:
    """Nucleotide-only dataset for single-modality backbones.

    GenBank is fine as a source here -- only the sequence is taken, never the
    CDS translations that the mixed-modality path emits.
    """
    texts = read_sequences(path)
    print(f"Read {len(texts)} sequences from {path}")
    val_texts = read_sequences(validation_file) if validation_file else None
    return _build_splits(texts, val_texts, validation_split_percentage, block_size)


def load_genbank_dataset(
    genbank_file: str,
    validation_genbank_file: Optional[str] = None,
    use_existing_translations: bool = False,
    validation_split_percentage: int = 5,
    translation_table: int = 11,
    block_size: Optional[int] = None,
) -> DatasetDict:
    """
    Load GenBank file(s) and convert to HuggingFace Dataset.

    Each LOCUS becomes one example with a 'text' field containing
    the tokenized sequence (with orientation tokens like <+>, <->).

    If ``block_size`` is given, a LOCUS longer than ``block_size`` tokens is
    split into consecutive **non-overlapping** blocks (each an independent
    training example) rather than truncated — so the whole genome is used, the
    tail is not dropped, and no region is duplicated. LOCUS boundaries are
    preserved (a block never spans two LOCUS records), and splitting happens
    after the train/validation split so blocks from one LOCUS cannot leak
    across it.
    """
    print(f"Loading GenBank file: {genbank_file}")
    tokenized_records = extract_and_tokenize_gb(
        genbank_file,
        use_existing_translations=use_existing_translations,
        translation_table=translation_table,
    )

    # Convert to list of texts
    texts = [record["sequence"] for record in tokenized_records if record["sequence"].strip()]
    print(f"Extracted {len(texts)} sequences from {len(tokenized_records)} LOCUS records")

    if validation_genbank_file:
        print(f"Loading validation GenBank file: {validation_genbank_file}")
        val_records = extract_and_tokenize_gb(
            validation_genbank_file,
            use_existing_translations=use_existing_translations,
            translation_table=translation_table,
        )
        val_texts = [r["sequence"] for r in val_records if r["sequence"].strip()]
    else:
        val_texts = None

    result = _build_splits(texts, val_texts, validation_split_percentage, block_size)
    return result


def load_dataset_from_args(args: DataTrainingArguments, tokenizer, backbone=None):
    """Load and preprocess a dataset from GenBank, FASTA, text or the HF hub.

    A nucleotide-only backbone reads sequences directly, skipping the mixed
    protein+DNA tokenization that GenBank would otherwise produce.
    """
    nucleotide_only = backbone is not None and backbone.modality == "nucleotide"
    source = args.genbank_file or args.train_file

    if nucleotide_only and source is not None:
        raw_datasets = load_sequence_dataset(
            source,
            validation_file=args.validation_genbank_file or args.validation_file,
            validation_split_percentage=args.validation_split_percentage,
            block_size=args.max_seq_length,
        )
    # Option 1: GenBank file
    elif args.genbank_file is not None:
        # A long LOCUS is split into consecutive non-overlapping blocks (each an
        # independent training example), preserving LOCUS boundaries. Loci are
        # never concatenated together, which would fabricate cross-contig
        # junctions the model would learn as real context.
        raw_datasets = load_genbank_dataset(
            genbank_file=args.genbank_file,
            validation_genbank_file=args.validation_genbank_file,
            use_existing_translations=args.use_existing_translations,
            validation_split_percentage=args.validation_split_percentage,
            translation_table=getattr(args, "translation_table", 11),
            block_size=args.max_seq_length,
        )
    # Option 2: HuggingFace dataset
    elif args.dataset_name is not None:
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=getattr(args, 'cache_dir', None),
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=getattr(args, 'cache_dir', None),
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=getattr(args, 'cache_dir', None),
            )
    # Option 3: Text files
    elif args.train_file is not None:
        data_files = {}
        data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file

        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=getattr(args, 'cache_dir', None))
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=getattr(args, 'cache_dir', None),
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=getattr(args, 'cache_dir', None),
            )
    else:
        raise ValueError("Must specify one of: --genbank_file, --dataset_name, or --train_file")

    # Tokenize datasets. GenBank input is already split into <= max_seq_length
    # blocks per LOCUS (see load_genbank_dataset); truncation here is only a
    # safety net. Sequences are never concatenated across records.
    if args.max_seq_length and "text" in raw_datasets["train"].column_names:
        raw_datasets = DatasetDict(
            {k: _split_into_blocks(v, args.max_seq_length) for k, v in raw_datasets.items()}
        )

    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_seq_length),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    return tokenized_datasets


def build_block_dataset(
    genbank_file,
    tokenizer,
    block_size=1024,
    validation_split_percentage=5,
    use_existing_translations=True,
    num_proc=1,
    translation_table=11,
):
    """
    Convenience helper for notebook use.

    Load a GenBank file and produce a tokenized ``DatasetDict`` of
    ``<= block_size`` blocks (train and, if the dataset is large enough,
    validation), ready to be passed straight to a ``MinervaTrainer``.

    Each LOCUS is split into consecutive non-overlapping ``block_size``-token
    blocks (the final short remainder kept), preserving LOCUS boundaries — loci
    are never concatenated together. This mirrors the pipeline used by
    ``load_dataset_from_args`` / ``main()``.
    """
    raw_datasets = load_genbank_dataset(
        genbank_file=genbank_file,
        use_existing_translations=use_existing_translations,
        validation_split_percentage=validation_split_percentage,
        translation_table=translation_table,
        block_size=block_size,
    )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        fn_kwargs={"tokenizer": tokenizer, "max_length": block_size},
        desc="Running tokenizer on dataset",
    )

    return tokenized_datasets


def apply_lora(model, r=1, alpha=2, dropout=0.05, target_modules=None):
    """
    Wrap a Minerva model with a PEFT LoRA adapter and return the PEFT model.

    Mirrors the LoRA configuration used by the CLI finetuning script. If
    ``target_modules`` is None, defaults to the Minerva attention/FFN linear
    layer names (``wqkv``, ``wo`` for attention; ``w1``, ``w2``, ``w3`` for the
    SwiGLU FFN).
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise RuntimeError("pip install peft") from e

    if target_modules is None:
        target_modules = ["wqkv", "wo", "w1", "w2", "w3"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.TOKEN_CLS,  # Closest to MLM
    )
    return get_peft_model(model, lora_config)
