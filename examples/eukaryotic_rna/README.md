# Eukaryotic RNA Support

Minerva provides a RiNALMo-based checkpoint for researchers studying eukaryotic
RNAs, extending its interaction predictions beyond Minerva-MLM's prokaryotic
genome setting. The checkpoint
[`gbrixi/minerva-rinalmo-giga`](https://huggingface.co/gbrixi/minerva-rinalmo-giga)
includes base-pairing and repeat interaction heads.

## Loading

Install `minerva-dna` following the [installation instructions](../../README.md#install).
Use plain nucleotide sequences without Minerva's strand markers or protein
tokens. DNA-alphabet sequences (A/C/G/T) are accepted directly. The tokenizer
normalizes case and maps U to T, so equivalent RNA- and DNA-alphabet sequences
produce identical token IDs.

```python
import torch
from transformers import AutoTokenizer
from minerva.modeling_rinalmo import RiNALMoMinervaForMaskedLM

repo = "gbrixi/minerva-rinalmo-giga"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RiNALMoMinervaForMaskedLM.from_pretrained(repo).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(repo)
sequence = "GGGGCUUUAGCUCAGCUGGGAGAGCGCCUGCCUUGCACGCAGGAGGUCAGCGGUUCGAUCCGCUAAGCUCCA"
tokens = tokenizer(sequence, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**tokens, output_interactions=True)
base_pairing = outputs.interactions["base_pairing"]
repeat = outputs.interactions["repeat"]
```

For a single unpadded sequence, interaction maps have shape `[1, L, L]`,
excluding the tokenizer's boundary tokens. RiNALMo uses six-layer heads and
has no protein interaction head.

## Input Windows and Limitations

Prefer the RNA sequence itself or a short window around it. Base-pairing
accuracy drops with long genomic flanks; Minerva-MLM remains the model for
prokaryotic genomic context. Interaction heads are specific to this backbone.

## Finetuning

Run the shared [finetuning script](../../scripts/finetune.py) from the repository
root with `--backbone rinalmo` and RNA FASTA or GenBank input. GenBank records
are read as nucleotides, without translating CDS features. LoRA targets and
token loss groups are selected for the chosen backbone.

```bash
accelerate launch --num_processes=1 scripts/finetune.py \
    --backbone rinalmo \
    --model_name_or_path gbrixi/minerva-rinalmo-giga \
    --train_file rna.fasta --output_dir ./output-rinalmo \
    --max_seq_length 512 --per_device_train_batch_size 1 \
    --use_lora --lora_r 1 --lora_alpha 2 --learning_rate 1e-4
```

Omit `--use_lora` for full finetuning. To load a saved LoRA adapter:

```python
from peft import PeftModel
from minerva.modeling_rinalmo import RiNALMoMinervaForMaskedLM

base = RiNALMoMinervaForMaskedLM.from_pretrained("gbrixi/minerva-rinalmo-giga")
model = PeftModel.from_pretrained(base, "path/to/lora_ckpt")
```

## License

The RiNALMo backbone code is vendored under the
[Apache 2.0 license](../../minerva/vendor_rinalmo/LICENSE).

## Citation

If you use Minerva in your work, please cite the paper.

If you use the RiNALMo-based checkpoint, please also cite RiNALMo:

> Penić, R.J., Vlašić, T., Huber, R.G. *et al.* RiNALMo: general-purpose RNA language models can
> generalize well on structure prediction tasks. *Nat Commun* **16**, 5671 (2025).
> https://doi.org/10.1038/s41467-025-60872-5

```bibtex
@article{penic2025rinalmo,
  title   = {{RiNALMo}: general-purpose {RNA} language models can generalize well on structure prediction tasks},
  author  = {Peni{\'c}, Rafael Josip and Vla{\v{s}}i{\'c}, Tin and Huber, Roland G. and others},
  journal = {Nature Communications},
  volume  = {16},
  pages   = {5671},
  year    = {2025},
  doi     = {10.1038/s41467-025-60872-5}
}
```
