
<h1><img src="assets/minerva_owl.png" alt="" height="46" valign="middle"> Minerva</h1>

**Coevolutionary discovery using genome language models**

Minerva predicts coevolution using genome language models. Powered by Minerva-1, it delivers database-scale, alignment-free, interaction-specific predictions across prokaryotic genomes. Through adaptation on homologous loci, Minerva can discover additional interactions.

## Install

To install Minerva, use:

```bash
pip install minerva-dna
```

Minerva uses `flash-attn` automatically when it is installed. Otherwise it falls
back to PyTorch SDPA. Please install flash attention first for faster inference.

## Pretrained Checkpoints

Minerva-1 is a 650M parameter transformer trained for over 1.3 trillion tokens (~3.4 Terabases). Minerva-1 is initialized from [gLM2 650M](https://github.com/TattaBio/gLM2) and adopts the mixed-modality tokenization, and was trained at 4096 and 8192 context lengths.

Checkpoints are hosted on Hugging Face:

| Model       | Context | Hugging Face repo                                   |
| ----------- | ------- | --------------------------------------------------- |
| Minerva-1   | 4096    | [`gbrixi/minerva`](https://huggingface.co/gbrixi/minerva)         |
| Minerva-1-8k   | 8192    | [`gbrixi/minerva-8k`](https://huggingface.co/gbrixi/minerva-8k)   |

All checkpoints include three interaction heads and Jacobian fingerprint types:

- **base_pairing** — RNA base-pairing contacts
- **protein** — protein contact prediction
- **repeat** — repeat element detection

### Quick start

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model = AutoModelForMaskedLM.from_pretrained(
    "gbrixi/minerva", trust_remote_code=True, torch_dtype=torch.bfloat16,
).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained("gbrixi/minerva")

tokens = tokenizer(
    "<+>cgcggggtggagcagcctggtagctcgtcgggctcataacccgaagatcgtcggttcaaatccggcccccgcaacca",
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    outputs = model(**tokens, output_interactions=True)

base_pairing = outputs.interactions["base_pairing"]  # [batch, L, L]
protein = outputs.interactions["protein"]            # [batch, L, L]
repeat = outputs.interactions["repeat"]              # [batch, L, L]
```

> `trust_remote_code=True` is required for the **model** because Minerva uses a custom architecture.

### Forward pass with interactions

```python
tokens = tokenizer(
    "<+>cgcggggtggagcagcctggtagctcgtcgggctcataacccgaagatcgtcggttcaaatccggcccccgcaacca",
    return_tensors="pt",
).to(model.device)
outputs = model(**tokens, output_interactions=True)

outputs.interactions["base_pairing"]  # [batch, L, L]
outputs.interactions["protein"]       # [batch, L, L]
outputs.interactions["repeat"]        # [batch, L, L]
```

To plot the interaction-head outputs:

```python
from minerva.visualization import plot_interactions

token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].tolist())
plot_interactions(outputs.interactions, tokens=token_list)
```

Set `interaction_layers=6` to use the six-layer interaction heads:

```python
outputs = model(**tokens, output_interactions=True, interaction_layers=6)
```

Raw attention tensors follow the Hugging Face convention:

```python
outputs = model(
    **tokens,
    output_interactions=True,
    output_attentions=True,
    attention_layers=[31, 32],
)
```

### Jacobian fingerprinting

Use `get_fingerprints` when you want named interaction-pattern channels from a
sequence. It computes the required full categorical Jacobian internally and returns
a `FingerprintResult`; the large raw Jacobian is not kept unless requested.

```python
sequence = "<+>cgcggggtggagcagcctggtagctcgtcgggctcataacccgaagatcgtcggttcaaatccggcccccgcaacca"
window = (0, min(len(sequence), 96))

fp = model.get_fingerprints(
    sequence,
    tokenizer,
    position_range=window,
    max_batch_size=64,
)

fp.channel_names                     # ["basepairing", "repeat", "protein", "other"]
basepairing = fp["basepairing"]      # [L, L]
protein = fp["protein"]              # [L, L]
repeat = fp["repeat"]                # [L, L]
```

To plot the result:

```python
from minerva.visualization import plot_fingerprints

plot_fingerprints(fp, title="Minerva multimodal fingerprint")
```

## Preparing inputs

Minerva reads a **mixed protein + DNA** sequence: coding regions are upper-case
amino acids, intergenic regions are lower-case nucleotides, and `<+>` / `<->`
markers denote strand.

```
<+>MALTKVEKRNRIKRRVRGK<+>aatttaaggaa<->MLGIDNIERVKPGGLELVDRLV
   └── CDS (protein) ──┘└ intergenic ┘└──── CDS on - strand ────┘
```

There are **three ways** to produce this format depending on what you start
with:

| You have | Use | What happens |
| --- | --- | --- |
| **Annotated GenBank** (CDS features) | `minerva.data.extract_and_tokenize_gb(path)` | CDS features are translated to amino acids, intergenic DNA is kept lower-case, strand markers inserted. One sequence per LOCUS. |
| **Unannotated sequence** (FASTA / raw DNA) | `minerva.gene_calling.build_minerva_input(seq)` | Genes are called with **Pyrodigal**, then packaged into the mixed-token format. |
| **Raw genome + external CDS calls** | `minerva.sequence_utils.build_prodigal_mixed_sequence(seq, cds)` | Your own `[{start, end, strand}]` calls (from Prodigal, MGnify, IMG, …) are packaged, and genome↔token coordinate maps are returned. |

### From unannotated sequence (gene calling)

If you only have a FASTA file or a raw nucleotide string, let Minerva call the
genes for you (Pyrodigal is a core dependency, so nothing extra to install):

```python
from minerva.gene_calling import build_minerva_input, fasta_to_minerva_inputs

# From a single nucleotide string
out = build_minerva_input(sequence)          # dict: token_string + coord maps
token_string = out["token_string"]

# From a FASTA file (one result per record)
inputs = fasta_to_minerva_inputs("contigs.fasta")
```

Single-genome training is used for sequences ≥ 20 kb; shorter contigs fall back
to Pyrodigal's metagenomic mode automatically. Pass `meta=True` for
metagenomic assemblies. The returned dict also carries `token_to_genome` /
`genome_to_token` maps for projecting model outputs back to genome coordinates.

### Context length & capping

Minerva's context is **4096 tokens** (`gbrixi/minerva`) or **8192**
(`gbrixi/minerva-8k`). Because the tokenizer is character-level, one token is
one amino acid, one nucleotide, or one strand marker — so coding regions are
~3× denser than raw DNA. As a rule of thumb, a typical (~88 % coding)
bacterial genome packs to **~10 kb per 4096 tokens** (~20 kb for the 8k model),
i.e. roughly 10–12 genes (20–24 for the 8k model). A whole chromosome LOCUS is
far larger than the context and will not fit in a single forward pass.

To cap a sequence at the model context, pass `max_tokens` to the builders. This
truncates **at a gene boundary** (never mid-marker), keeps the 5′/left end, and
keeps the returned coordinate maps consistent — unlike the tokenizer's
`truncation=True`, which would slice mid-protein:

```python
out = build_minerva_input(sequence, max_tokens=4096)   # <= 4096 tokens, gene-aligned
assert len(tokenizer(out["token_string"])["input_ids"]) <= 4096
```

`max_tokens` is available on `build_minerva_input`, `fasta_to_minerva_inputs`,
and `build_prodigal_mixed_sequence`. It *truncates* (keeps the left window and
drops the rest). To run a model over an **entire** long genome, tile it instead
with `minerva.sequence_utils.chunk_sequence_with_stride(token_string, chunk_size,
stride)` — the overlapping-window primitive used for genome scanning at
inference — and use the returned `token_to_genome` map to place each window's
outputs back on the genome. (Overlap is for scanning, not training.)

### Translation tables

CDS are translated with **NCBI genetic-code table 11** (bacterial / archaeal /
plant plastid) by default. A CDS feature's own `/transl_table` qualifier in a
GenBank file always takes precedence, so files mixing genetic codes (e.g. a
table-4 *Mycoplasma* gene) translate correctly. Override the default with the
`translation_table=` argument on `extract_and_tokenize_gb`,
`build_minerva_input`, and `build_prodigal_mixed_sequence`, or `--translation_table`
on `scripts/finetune.py`.

See [`examples/`](examples/) for runnable, end-to-end walkthroughs.

## Finetuning

`scripts/finetune.py` wraps HF `Trainer` + `accelerate` with Minerva's MLM loss.
It supports full finetuning and LoRA, and ingests GenBank files directly.

```bash
# LoRA finetune
accelerate launch --num_processes=8 scripts/finetune.py \
    --output_dir ./output \
    --genbank_file genome.gb \
    --tokenizer_name gbrixi/minerva \
    --model_name_or_path gbrixi/minerva \
    --use_lora --lora_r 1 --lora_alpha 2 \
    --learning_rate 1e-4 --bf16

# Full finetune on a GenBank file across 8 GPUs
accelerate launch --num_processes=8 scripts/finetune.py \
    --output_dir ./output \
    --genbank_file genome.gb \
    --tokenizer_name gbrixi/minerva \
    --model_name_or_path gbrixi/minerva \
    --per_device_train_batch_size 4 \
    --bf16
```

LoRA checkpoints load with PEFT:

```python
from peft import PeftModel
from transformers import AutoModelForMaskedLM

base = AutoModelForMaskedLM.from_pretrained("gbrixi/minerva", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "path/to/lora_ckpt")
```

A LOCUS longer than `--max_seq_length` is split into consecutive
**non-overlapping** blocks of `--max_seq_length` tokens, each kept as its own
training example (the final short remainder included). Every token is seen
exactly once: the tail is not dropped (unlike plain truncation) and no region is
duplicated (unlike the overlapping windows used only for inference-time
scanning). LOCUS boundaries are always preserved — a block never spans two LOCUS
records, so the model is never trained on a fabricated junction between
unrelated loci.

## Repo layout

```
minerva/
  modeling_minerva.py   # MinervaConfig / MinervaForMaskedLM (custom transformer + heads)
  data.py               # GenBank parsing + tokenization
  gene_calling.py       # Pyrodigal gene calling: FASTA/raw DNA -> mixed tokens
  sequence_utils.py     # reverse-complement + external-CDS -> mixed tokens
  masking.py            # DataCollatorForMinervaMLM
  losses.py             # minerva_mlm_loss
scripts/
  finetune.py                  # HF Trainer / accelerate wrapper
examples/                       # end-to-end tutorials & notebooks
  call_genes_from_fasta.py      # FASTA/raw DNA -> Minerva input walkthrough
  notebooks/                    # interactive Colab-ready notebooks
  data/                         # sample GenBank genomes
tests/                          # package unit + smoke tests
```

## Citation

If you use Minerva in your work, please cite the paper.

If you use the Jacobian fingerprints, please cite the categorical Jacobian (Zhang et al., PNAS 2024)