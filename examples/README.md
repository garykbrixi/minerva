# Minerva examples

Walkthroughs for using Minerva.

| File | Colab | Content |
| --- | --- | --- |
| [`notebooks/loci_viewer.ipynb`](notebooks/loci_viewer.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garykbrixi/minerva/blob/main/examples/notebooks/loci_viewer.ipynb) | View contact maps on a locus, from the interaction heads or Jacobian fingerprints, as an interactive viewer or a publication PDF. |
| [`notebooks/finetune.ipynb`](notebooks/finetune.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garykbrixi/minerva/blob/main/examples/notebooks/finetune.ipynb) | Finetune Minerva with LoRA on your own GenBank genome and compare contacts before and after. |
| [`notebooks/rna_structure.ipynb`](notebooks/rna_structure.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garykbrixi/minerva/blob/main/examples/notebooks/rna_structure.ipynb) | Call, draw and export an RNA secondary structure from the base-pairing head. |
| [`call_genes_from_fasta.py`](call_genes_from_fasta.py) | | Go from an unannotated FASTA or raw DNA string to Minerva input with Pyrodigal gene calling, and map outputs back to genome coordinates. |
| [`eukaryotic_rna/`](eukaryotic_rna/) | | Use the RiNALMo-based checkpoint for eukaryotic RNA: loading, interaction heads and finetuning. |

Minerva reads a mixed protein + DNA sequence. [Preparing inputs](../README.md#preparing-inputs)
in the main README explains the format and which builder to use for GenBank, FASTA or your own
CDS calls.
