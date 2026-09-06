"""
Finetuning script for Minerva model using HuggingFace Trainer and Accelerate.

Supports multi-GPU (single node) training with custom Minerva MLM loss.

Usage:
    # From GenBank file (recommended for genomic data)
    python scripts/finetune.py --output_dir ./output --genbank_file genome.gb --tokenizer_name tattabio/gLM2_150M

    # Multi-GPU with GenBank
    accelerate launch --num_processes=8 scripts/finetune.py \\
        --output_dir ./output \\
        --genbank_file genome.gb \\
        --tokenizer_name tattabio/gLM2_150M \\
        --per_device_train_batch_size 4 \\
        --bf16

    # From text file
    python scripts/finetune.py --output_dir ./output --train_file data.txt --tokenizer_name tattabio/gLM2_150M

    # Multi-GPU (single node) - use accelerate launch
    accelerate launch --num_processes=8 scripts/finetune.py --output_dir ./output --train_file data.txt --tokenizer_name tattabio/gLM2_150M

    # Or use torchrun
    torchrun --nproc_per_node=8 scripts/finetune.py --output_dir ./output --train_file data.txt --tokenizer_name tattabio/gLM2_150M

    # With HuggingFace dataset
    accelerate launch --num_processes=8 scripts/finetune.py \\
        --output_dir ./output \\
        --dataset_name your_dataset \\
        --tokenizer_name tattabio/gLM2_150M \\
        --model_name_or_path /path/to/checkpoint \\
        --per_device_train_batch_size 4 \\
        --gradient_accumulation_steps 4 \\
        --learning_rate 5e-5 \\
        --num_train_epochs 3 \\
        --bf16

    # LoRA finetuning (parameter-efficient, trains ~1% of params)
    accelerate launch --num_processes=8 scripts/finetune.py \\
        --output_dir ./output \\
        --genbank_file genome.gb \\
        --tokenizer_name tattabio/gLM2_150M \\
        --model_name_or_path /path/to/checkpoint \\
        --use_lora \\
        --lora_r 1 \\
        --lora_alpha 2 \\
        --learning_rate 1e-4 \\
        --bf16

Note: When validation data is provided, load_best_model_at_end=True is enabled by default.
If save_steps and eval_steps don't align, eval_steps will automatically be set to match save_steps.
"""

import argparse
import os
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
from minerva.backbones import check_token_groups, get_backbone
from minerva.modeling_minerva import MinervaForMaskedLM, MinervaConfig
from minerva.finetuning import (
    load_model_from_lightning_ckpt,
    ModelArguments,
    DataTrainingArguments,
    MinervaTrainer,
    tokenize_function,
    load_genbank_dataset,
    load_dataset_from_args,
)


# Optional: PEFT/LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Finetune Minerva model")
    
    # Parse arguments
    parser.add_argument("--model_args", type=str, help="JSON string of model arguments")
    parser.add_argument("--data_args", type=str, help="JSON string of data arguments")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=True)
    parser.add_argument("--model_revision", type=str, default="main")
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--genbank_file", type=str, default=None, help="Input GenBank file (.gb, .gbk)")
    parser.add_argument("--validation_genbank_file", type=str, default=None, help="Validation GenBank file")
    parser.add_argument("--use_existing_translations", action="store_true", help="Use existing protein translations from GenBank")
    parser.add_argument("--translation_table", type=int, default=11, help="Default NCBI genetic-code table for CDS lacking a /transl_table qualifier (11 = bacterial/archaeal/plant plastid). Per-feature /transl_table always wins.")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--overwrite_cache", action="store_true", default=False)
    parser.add_argument("--validation_split_percentage", type=int, default=5)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    
    # Training arguments (will be passed to TrainingArguments)
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides num_train_epochs if > 0)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=None, help="Keep all checkpoints by default")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--report_to", type=str, default="none", help="Reporting integration (wandb, tensorboard, etc.)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory")
    parser.add_argument("--run_name", type=str, default=None, help="Name for wandb run")
    
    # LoRA arguments
    parser.add_argument("--backbone", type=str, default="minerva",
                        help="Which backbone spec to use (see minerva.backbones)")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient finetuning")
    parser.add_argument("--lora_r", type=int, default=1, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=2, help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default=None,
                        help="Comma-separated modules for LoRA; defaults to the backbone's")
    
    # Loss arguments
    parser.add_argument("--token_type_upweighting", action="store_true", help="Normalize losses by log(vocab_size) for DNA/AA")
    
    # Layer freezing
    parser.add_argument("--freeze_layers_except_last", type=int, default=None, help="Freeze all layers except the last N transformer layers")
    
    # Legacy checkpoint loading
    parser.add_argument("--legacy_module_path", type=str, default=None,
                        help="Path to legacy module for loading old .ckpt files")
    parser.add_argument("--model_size", type=str, default=None, choices=["650m", "3b"],
                        help="Model size for architecture config (auto-detected from path if not specified)")
    
    args = parser.parse_args()
    
    # Setup model arguments
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        config_name=args.config_name,
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.cache_dir,
        use_fast_tokenizer=args.use_fast_tokenizer,
        model_revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
    )
    
    # Setup data arguments
    data_args = DataTrainingArguments(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        train_file=args.train_file,
        validation_file=args.validation_file,
        genbank_file=args.genbank_file,
        validation_genbank_file=args.validation_genbank_file,
        use_existing_translations=args.use_existing_translations,
        translation_table=args.translation_table,
        max_seq_length=args.max_seq_length,
        overwrite_cache=args.overwrite_cache,
        validation_split_percentage=args.validation_split_percentage,
        preprocessing_num_workers=args.preprocessing_num_workers,
        mlm_probability=args.mlm_probability,
    )
    
    # Load tokenizer first (needed for dataset loading)
    tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if tokenizer_name is None:
        raise ValueError("Must specify either tokenizer_name or model_name_or_path")
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    backbone = get_backbone(args.backbone)

    # Load dataset BEFORE creating TrainingArguments (so we know if validation exists)
    tokenized_datasets = load_dataset_from_args(data_args, tokenizer, backbone)
    has_validation = "validation" in tokenized_datasets
    
    # Setup training arguments (now we can check for validation)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
        overwrite_output_dir=args.overwrite_output_dir,
        eval_strategy="steps" if has_validation else "no",
        load_best_model_at_end=has_validation,
        save_strategy="steps",
        metric_for_best_model="loss",
        greater_is_better=False,
    )
    
    # Auto-align eval_steps with save_steps when load_best_model_at_end is enabled
    # This ensures checkpoints exist at evaluation points for selecting the best model
    if training_args.load_best_model_at_end:
        if training_args.eval_steps % training_args.save_steps != 0 and training_args.save_steps % training_args.eval_steps != 0:
            print(f"Warning: load_best_model_at_end=True requires save_steps and eval_steps to align.")
            print(f"  Automatically setting eval_steps={training_args.save_steps} to match save_steps.")
            training_args.eval_steps = training_args.save_steps
    
    # Load model
    if model_args.model_name_or_path:
        ckpt_path = model_args.model_name_or_path
        
        # Check if it's a Lightning .ckpt file
        if ckpt_path.endswith(".ckpt"):
            # Load from PyTorch Lightning checkpoint - need config for architecture
            # Detect model size from path or use specified size
            model_size = getattr(args, 'model_size', None)
            if model_size == "650m" or "650" in ckpt_path.lower():
                print("Using 650M model config")
                config = MinervaConfig(
                    dim=1280,
                    depth=24,
                    heads=20,
                    vocab_size=37,
                    norm_eps=1e-5,
                    swiglu_multiple_of=256,
                    ffn_dim_multiplier=None,
                    qk_norm=False,
                    base=10000,
                )
            else:
                print("Using 3B model config")
                config = MinervaConfig(
                    dim=2560,
                    depth=36,
                    heads=40,
                    vocab_size=37,
                    norm_eps=1e-5,
                    swiglu_multiple_of=256,
                    ffn_dim_multiplier=None,
                    qk_norm=True,
                    base=20000,
                )
            model = load_model_from_lightning_ckpt(ckpt_path, config, legacy_module_path=args.legacy_module_path)
        else:
            # Load from HuggingFace format (directory or Hub)
            model = backbone.load(
                ckpt_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        # Create new model from config
        if model_args.config_name:
            config = MinervaConfig.from_pretrained(
                model_args.config_name,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                trust_remote_code=model_args.trust_remote_code,
            )
        else:
            config = MinervaConfig()
        model = MinervaForMaskedLM(config)
    
    # Freeze layers if requested (for partial finetuning)
    if args.freeze_layers_except_last is not None:
        n_frozen, n_layers = backbone.freeze_layers(model, args.freeze_layers_except_last)
        print(f"Froze first {n_frozen} of {n_layers} layers and the embeddings; LM head left trainable")

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Enable gradient checkpointing manually on the base model
    # (Trainer's automatic method can fail with custom models or PEFT wrappers)
    if training_args.gradient_checkpointing:
        print("Enabling gradient checkpointing on model")
        if backbone.enable_grad_checkpointing(model):
            print("  -> Set gradient_checkpointing=True on encoder")
            training_args.gradient_checkpointing = False
    
    # Apply LoRA if requested
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required for LoRA. Install with: pip install peft")
        
        target_modules = (
            [m.strip() for m in args.lora_target_modules.split(",")]
            if args.lora_target_modules else list(backbone.lora_targets)
        )
        print(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}, target_modules={target_modules}")
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.TOKEN_CLS,  # Closest to MLM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=data_args.mlm_probability,
    )
    
    token_groups = backbone.token_groups(tokenizer)
    check_token_groups(token_groups)
    for group in token_groups:
        print(f"{group.name}: {len(group.token_ids)} tokens, /log({group.alphabet_size})")

    trainer = MinervaTrainer(
        model=model,
        token_groups=token_groups,
        ignore_token_id=-100,
        token_type_upweighting=args.token_type_upweighting,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        data_collator=data_collator,
    )
    
    # Resume from checkpoint if specified
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)
    
    # Train
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate
    if tokenized_datasets.get("validation") is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()