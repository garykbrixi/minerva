"""Call genes on unannotated DNA and run Minerva on the result.

This walkthrough covers the "I only have a FASTA / raw sequence" path:

    raw DNA  ->  Pyrodigal gene calling  ->  Minerva mixed-token input
             ->  forward pass  ->  interactions mapped to genome coordinates

Pyrodigal is a core Minerva dependency, so there is nothing extra to install.

Run it standalone (no model download; just the input-prep steps):

    python examples/call_genes_from_fasta.py

Pass a FASTA to use your own sequence:

    python examples/call_genes_from_fasta.py path/to/contigs.fasta
"""

import sys

from minerva.gene_calling import build_minerva_input, call_genes, read_fasta

# A short demo contig (two toy ORFs on opposite strands separated by an
# intergenic spacer). Replace by passing a FASTA path on the command line.
DEMO_SEQUENCE = (
    "ATG" + "GCTGCAAAACGTGAAGCACTGAGCGATCTGGAA" * 6 + "TAA"
    + "acgtacgtacgtacgtacgt"
    + "ATG" + "AGCACCGTTAAAGGCGATCTGCTGGCAAACCGT" * 6 + "TGA"
)


def main(fasta_path: str | None = None) -> None:
    if fasta_path:
        records = read_fasta(fasta_path)
    else:
        records = [("demo_contig", DEMO_SEQUENCE)]

    for rec_id, sequence in records:
        print(f"\n=== {rec_id} ({len(sequence)} bp) ===")

        # 1. Call genes. Sequences >= 20 kb train a single-genome model;
        #    shorter ones fall back to metagenomic mode automatically.
        #    Pass meta=True explicitly for metagenomic assemblies.
        cds = call_genes(sequence)
        print(f"Pyrodigal called {len(cds)} CDS:")
        for c in cds:
            print(f"  start={c['start']:>6}  end={c['end']:>6}  strand={c['strand']:+d}")

        # 2. Build Minerva's mixed protein + DNA token string.
        #    Pass max_tokens=<context> (4096 for gbrixi/minerva-1, 8192 for the
        #    8k model) to cap the output at a gene boundary so it fits the model
        #    context. Omit it to get the whole sequence.
        out = build_minerva_input(sequence, max_tokens=4096)
        token_string = out["token_string"]
        print(f"\ntoken_string ({len(token_string)} chars):")
        print(f"  {token_string[:120]}{'...' if len(token_string) > 120 else ''}")

        # 3. `out` also carries coordinate maps so you can project model
        #    outputs back onto the genome:
        #      out['token_to_genome'][i]  -> (genome_start, genome_end) for token char i
        #      out['genome_to_token'][pos] -> token-char index for genome position `pos`
        print(f"\ngenome_to_token map: shape {out['genome_to_token'].shape}")

        # 4. Feed `token_string` to the tokenizer + model, e.g.:
        #
        #    from transformers import AutoModelForMaskedLM, AutoTokenizer
        #    import torch
        #    model = AutoModelForMaskedLM.from_pretrained(
        #        "gbrixi/minerva-1", trust_remote_code=True, torch_dtype=torch.bfloat16,
        #    ).cuda().eval()
        #    tokenizer = AutoTokenizer.from_pretrained("gbrixi/minerva-1")
        #    tokens = tokenizer(token_string, return_tensors="pt").to(model.device)
        #    with torch.no_grad():
        #        outputs = model(**tokens, output_interactions=True)
        #    # outputs.interactions["protein"] is [batch, L, L] over token positions;
        #    # use out['token_to_genome'] to relate positions back to the genome.


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
