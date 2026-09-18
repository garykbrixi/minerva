
<h1><img src="assets/minerva_owl.png" alt="" height="46" valign="middle"> Minerva</h1>

[![tests](https://github.com/garykbrixi/minerva/actions/workflows/tests.yml/badge.svg)](https://github.com/garykbrixi/minerva/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/garykbrixi/minerva)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Minerva--MLM-yellow)](https://huggingface.co/gbrixi/minerva-mlm)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Coevolutionary discovery using genome language models**

Minerva predicts coevolution using genome language models. Powered by Minerva-MLM, it delivers database-scale, alignment-free, interaction-specific predictions across prokaryotic genomes. Through adaptation on homologous loci, Minerva can discover additional interactions.

[Install](#install) · [Checkpoints](#pretrained-checkpoints) · [Quick start](#quick-start) · [Preparing inputs](#preparing-inputs) · [Interaction heads](#interaction-heads) · [Jacobian fingerprinting](#jacobian-fingerprinting) · [RNA structure](#rna-secondary-structure) · [Eukaryotic RNA](#eukaryotic-rna) · [Finetuning](#finetuning) · [Examples](examples/) · [Citation](#citation)

## Install

To install Minerva, use:

```bash
pip install minerva-dna
```

Minerva uses `flash-attn` automatically when it is installed. Otherwise it falls
back to PyTorch SDPA. Please install flash attention first for faster inference.

## Pretrained Checkpoints

Minerva-MLM is a 650M parameter transformer trained for over 1.3 trillion tokens (~3.4 Terabases). Minerva-MLM is initialized from [gLM2 650M](https://github.com/TattaBio/gLM2) and adopts the mixed-modality tokenization, and was trained at 4096 and 8192 context lengths.

Checkpoints are hosted on Hugging Face:

| Model       | Context | Hugging Face repo                                   |
| ----------- | ------- | --------------------------------------------------- |
| Minerva-MLM   | 4096    | [`gbrixi/minerva-mlm`](https://huggingface.co/gbrixi/minerva-mlm)         |
| Minerva-MLM-8k   | 8192    | [`gbrixi/minerva-mlm-8k`](https://huggingface.co/gbrixi/minerva-mlm-8k)   |

Minerva-MLM checkpoints include three interaction heads and Jacobian fingerprint types:

- **base_pairing** — RNA base-pairing contacts
- **protein** — protein contact prediction
- **repeat** — repeat element detection

## Quick start

```python
from transformers import AutoTokenizer
from minerva import MinervaForMaskedLM
import torch

model = MinervaForMaskedLM.from_pretrained(
    "gbrixi/minerva-mlm", torch_dtype=torch.bfloat16,
).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained("gbrixi/minerva-mlm")

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

> Importing the class directly needs no `trust_remote_code`. Without the package
> installed, use `AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True)`.

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
| **Annotated GenBank** (CDS features) | `minerva.data.extract_and_tokenize_gb(path)` | CDS translated, intergenic kept as DNA, strand markers inserted. One sequence per LOCUS. |
| **Unannotated sequence** (FASTA / raw DNA) | `minerva.gene_calling.build_minerva_input(seq)` | Genes called with **Pyrodigal**, then packaged. |
| **Raw genome + external CDS calls** | `minerva.sequence_utils.build_prodigal_mixed_sequence(seq, cds)` | Your own `[{start, end, strand}]` calls packaged, with genome↔token maps. |

### From unannotated sequence (gene calling)

If you only have a FASTA file or a raw nucleotide string, we use [Pyrodigal](https://github.com/althonos/pyrodigal) to automatically call genes:

```python
from minerva.gene_calling import build_minerva_input, fasta_to_minerva_inputs

# From a single nucleotide string
out = build_minerva_input(sequence)          # dict: token_string + coord maps
token_string = out["token_string"]

# From a FASTA file (one result per record)
inputs = fasta_to_minerva_inputs("contigs.fasta")
```

Pyrodigal needs ≥ 20 kb to estimate gene-scoring statistics from a sequence;
shorter contigs use its pre-trained profiles, which `meta=True` forces
for metagenomic assemblies. A CDS token is one amino acid and an intergenic
token one base, so `token_to_genome` / `genome_to_token` map token index to
genome position.

### Context length & capping

Minerva's context is 4096 (`gbrixi/minerva-mlm`) or 8192 tokens
(`gbrixi/minerva-mlm-8k`). One token is one amino acid, one nucleotide, or one
strand marker, so a typical (~88 % coding) bacterial genome packs to ~10 kb per
4096 tokens (~20 kb for the 8k model).

Pass `max_tokens` to the builders to cap a sequence. It truncates at a gene
boundary, keeps the 5′ end, and keeps the coordinate maps consistent.

```python
out = build_minerva_input(sequence, max_tokens=4096)   # <= 4096 tokens
```

To cover a whole genome, tile it instead with
`minerva.sequence_utils.chunk_sequence_with_stride`.

### Translation tables

CDS translate with NCBI table 11 by default, but a GenBank feature's own
`/transl_table` takes precedence. Override with `translation_table=` on the
builders or `--translation_table` on `scripts/finetune.py`.

See [`examples/`](examples/) for runnable, end-to-end walkthroughs.

## Using the model

These continue from the [quick start](#quick-start), with `model`, `tokenizer`, `tokens` and
`outputs` already defined.

### Interaction heads

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

### RNA secondary structure

The `base_pairing` head returns a dense contact map which can be converted to an RNA structure using `minerva.rna_structure`:

```python
from minerva.rna_structure import call_structure, call_structures

token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].tolist())
s = call_structure(outputs.interactions["base_pairing"], tokens=token_list)

s.dot_bracket        # '(((((((..((((........)))).(((((.......)))))...'
s.to_vienna("t.fa")  # read by RNAfold, forna, VARNA, R2R
s.to_ct("t.ct")      # connect table, keeps pseudoknots
s.plot()             # matplotlib Figure
```

Each intergenic region of a mixed locus is a separate molecule, so
`call_structures` returns one structure per region:

```python
structures = call_structures(outputs.interactions["base_pairing"], token_list)
```

An interactive viewer is in
[`examples/notebooks/rna_structure.ipynb`](examples/notebooks/rna_structure.ipynb).

## Eukaryotic RNA

For researchers studying eukaryotic RNAs, Minerva provides a RiNALMo-based checkpoint with base-pairing and repeat interaction heads. See [eukaryotic RNA support](examples/eukaryotic_rna/) for usage and finetuning.

## Finetuning

`scripts/finetune.py` wraps HF `Trainer` + `accelerate` with Minerva's MLM loss.
It supports full finetuning and LoRA, and ingests GenBank files directly.

```bash
# LoRA finetune
accelerate launch --num_processes=8 scripts/finetune.py \
    --output_dir ./output \
    --genbank_file genome.gb \
    --tokenizer_name gbrixi/minerva-mlm \
    --model_name_or_path gbrixi/minerva-mlm \
    --use_lora --lora_r 1 --lora_alpha 2 \
    --learning_rate 1e-4 --bf16

# Full finetune on a GenBank file across 8 GPUs
accelerate launch --num_processes=8 scripts/finetune.py \
    --output_dir ./output \
    --genbank_file genome.gb \
    --tokenizer_name gbrixi/minerva-mlm \
    --model_name_or_path gbrixi/minerva-mlm \
    --per_device_train_batch_size 4 \
    --bf16
```

Minerva-MLM LoRA checkpoints loaded with PEFT:

```python
from peft import PeftModel
from minerva import MinervaForMaskedLM

base = MinervaForMaskedLM.from_pretrained("gbrixi/minerva-mlm")
model = PeftModel.from_pretrained(base, "path/to/lora_ckpt")
```

A LOCUS longer than `--max_seq_length` is split into non-overlapping blocks,
each its own training example, see
`minerva.finetuning.load_genbank_dataset`.

## Repo layout

```
minerva/
  modeling_minerva.py   # MinervaConfig / MinervaForMaskedLM (custom transformer + heads)
  modeling_rinalmo.py   # RiNALMoMinervaForMaskedLM (RNA backbone + heads)
  tokenization_rinalmo.py # RiNALMo nucleotide tokenizer
  backbones.py          # Backbone-specific training configuration
  interaction_heads.py # Shared interaction heads
  data.py               # GenBank parsing + tokenization
  gene_calling.py       # Pyrodigal gene calling: FASTA/raw DNA -> mixed tokens
  sequence_utils.py     # reverse-complement + external-CDS -> mixed tokens
  masking.py            # DataCollatorForMinervaMLM
  losses.py             # grouped_mlm_loss
  example_data/         # sample GenBank loci (minerva.data.example_path)
scripts/
  finetune.py                  # HF Trainer / accelerate wrapper
examples/                       # end-to-end tutorials & notebooks
  call_genes_from_fasta.py      # FASTA/raw DNA -> Minerva input walkthrough
  notebooks/                    # interactive Colab-ready notebooks
tests/                          # package unit + smoke tests
```

## Citation

If you use Minerva in your work, please cite the paper.

If you use the Jacobian fingerprints, please also cite the categorical Jacobian:

> Zhang, Z., Wayment-Steele, H.K., Brixi, G., Wang, H., Kern, D. & Ovchinnikov, S. Protein language
> models learn evolutionary statistics of interacting sequence motifs. *Proc. Natl. Acad. Sci. U.S.A.*
> **121** (45), e2406285121 (2024). https://doi.org/10.1073/pnas.2406285121

```bibtex
@article{zhang2024categoricaljacobian,
  title   = {Protein language models learn evolutionary statistics of interacting sequence motifs},
  author  = {Zhang, Z. and Wayment-Steele, H. K. and Brixi, G. and Wang, H. and Kern, D. and Ovchinnikov, S.},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {121},
  number  = {45},
  pages   = {e2406285121},
  year    = {2024},
  doi     = {10.1073/pnas.2406285121}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE). Minerva-MLM is initialized from
[gLM2 650M](https://github.com/TattaBio/gLM2) (Tatta Bio, Apache 2.0).
