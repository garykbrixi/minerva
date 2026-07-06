# Minerva examples

End-to-end, runnable walkthroughs. Each one is a single task — start here rather
than reading the source. If you're new, read
[**Preparing inputs**](../README.md#preparing-inputs) in the main README first:
it explains Minerva's mixed protein + DNA token format and the three ways to
produce it.

## Tutorials

| Example | What you'll do |
| --- | --- |
| [`call_genes_from_fasta.py`](call_genes_from_fasta.py) | Go from an unannotated FASTA / raw DNA string to Minerva mixed-token input using Pyrodigal gene calling, and map model outputs back to genome coordinates. |

## Notebooks

These live under [`notebooks/`](notebooks/) and are the interactive
counterparts to the tutorials above:

| Notebook | What it covers |
| --- | --- |
| [`finetune_colab.ipynb`](notebooks/finetune_colab.ipynb) | Finetune Minerva on your own GenBank genome (Colab-ready). |
| [`loci_viewer.ipynb`](notebooks/loci_viewer.ipynb) | Visualize interaction-head outputs across real loci. |
| [`loci_viewer_hf.ipynb`](notebooks/loci_viewer_hf.ipynb) | Same, loading the model straight from the Hugging Face Hub. |
| [`loci_viewer_bokeh.ipynb`](notebooks/loci_viewer_bokeh.ipynb) | Interactive Bokeh version of the loci viewer. |

## Choosing an input path

| You have | Use |
| --- | --- |
| Annotated GenBank (CDS features) | `minerva.data.extract_and_tokenize_gb` |
| Unannotated FASTA / raw DNA | `minerva.gene_calling.build_minerva_input` (this folder's tutorial) |
| Raw genome + external CDS calls | `minerva.sequence_utils.build_prodigal_mixed_sequence` |
