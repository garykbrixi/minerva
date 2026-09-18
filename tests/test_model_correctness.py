"""Golden-value correctness tests for the Minerva model.

Runs the tRNA sequence from the README through ``MinervaForMaskedLM`` and checks
the outputs against reference values captured from a trusted checkpoint. This
guards against silent regressions in the architecture / forward pass: any change
that alters the model's numerical output will shift the logits and fail here.

Two paths are tested, each with its own golden reference:

  * **fp32 / CPU / SDPA** — the portable, fully deterministic path (no GPU or
    flash-attn required). Reference: ``tests/data/trna_reference.npz``,
    tolerance ``atol=1e-4``.
  * **bf16 / CUDA / flash-attn** — the path users actually run (see README
    "Quick start"). bf16 is deterministic run-to-run but differs from fp32 by
    up to ~0.8 on raw logits, so it has its own golden captured in bf16.
    Reference: ``tests/data/trna_reference_bf16.npz``, tolerance ``atol=0.1``
    (comfortably absorbs cross-GPU bf16 rounding). Skipped when CUDA/flash-attn
    are unavailable.

Model resolution order:
  1. ``$MINERVA_MODEL_PATH`` (local dir or hub id), if set
  2. ``sharing_minerva_gdrive/`` (local dev checkpoint), if present
  3. ``gbrixi/minerva-mlm`` on the Hugging Face Hub

If none can be loaded (offline, no local weights) the tests are skipped.

Regenerate both goldens after an *intentional* model change (needs a CUDA GPU
with flash-attn for the bf16 reference):
    python tests/test_model_correctness.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

# tRNA sequence from README.md "Quick start".
SEQUENCE = "<+>cgcggggtggagcagcctggtagctcgtcgggctcataacccgaagatcgtcggttcaaatccggcccccgcaacca"
DATA_DIR = Path(__file__).parent / "data"
REPO_ROOT = Path(__file__).resolve().parents[1]

# One config per numerical path. Each pairs a (device, dtype) with its own golden
# reference and a tolerance appropriate to that dtype.
CONFIGS = {
    # float32 SDPA is deterministic to well under this across machines.
    "fp32_cpu": dict(
        device="cpu", dtype=torch.float32,
        reference="trna_reference.npz", atol=1e-4, rtol=1e-4,
    ),
    # bf16 flash-attn is deterministic run-to-run; 0.1 covers cross-GPU rounding.
    # A real regression moves logits by O(1), so this still fails loudly.
    "bf16_cuda": dict(
        device="cuda", dtype=torch.bfloat16,
        reference="trna_reference_bf16.npz", atol=0.1, rtol=0.0,
    ),
}


def _candidate_model_paths() -> list[str]:
    paths: list[str] = []
    env = os.environ.get("MINERVA_MODEL_PATH")
    if env:
        paths.append(env)
    local = REPO_ROOT / "sharing_minerva_gdrive"
    if local.is_dir():
        paths.append(str(local))
    paths.append("gbrixi/minerva-mlm")  # Hugging Face Hub fallback
    return paths


def _load_model_and_tokenizer(device: str, dtype: torch.dtype):
    from transformers import AutoTokenizer

    from minerva.modeling_minerva import MinervaForMaskedLM

    last_err: Exception | None = None
    for path in _candidate_model_paths():
        try:
            tok = AutoTokenizer.from_pretrained(path)
            model = MinervaForMaskedLM.from_pretrained(path, dtype=dtype)
            return model.to(device).eval(), tok, path
        except Exception as err:  # network down, missing weights, OOM, ...
            last_err = err
            continue
    pytest.skip(f"no Minerva checkpoint available to load ({last_err})")


@pytest.fixture(scope="module", params=list(CONFIGS), ids=list(CONFIGS))
def path_case(request):
    """Yield (name, config, reference-array, model-outputs) for one numerical path."""
    name = request.param
    cfg = CONFIGS[name]

    if cfg["device"] == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for the bf16/flash-attn path")
        # The bf16 golden was captured with flash-attn; the SDPA fallback
        # differs from it by more than the tolerance.
        from minerva import modeling_minerva
        if not modeling_minerva._HAS_FLASH:
            pytest.skip("flash-attn not installed for the bf16/flash-attn path")

    reference_path = DATA_DIR / cfg["reference"]
    if not reference_path.exists():
        pytest.skip(f"reference file missing: {reference_path}")
    reference = np.load(reference_path, allow_pickle=True)

    model, tok, _ = _load_model_and_tokenizer(cfg["device"], cfg["dtype"])
    enc = tok(SEQUENCE, return_tensors="pt").to(cfg["device"])
    with torch.no_grad():
        out = model(**enc, output_interactions=True)
    outputs = {
        "input_ids": enc["input_ids"][0].cpu().numpy(),
        "logits": out.logits[0].float().cpu().numpy(),
        "base_pairing": out.interactions["base_pairing"][0].float().cpu().numpy(),
    }
    return cfg, reference, outputs


def test_tokenization_matches_reference(path_case):
    _, reference, outputs = path_case
    np.testing.assert_array_equal(
        outputs["input_ids"], reference["input_ids"],
        err_msg="Tokenization of the tRNA sequence changed.",
    )


def test_logits_argmax_matches_reference(path_case):
    # Robust, dtype-insensitive correctness signal: the predicted token at every
    # position must be unchanged.
    _, reference, outputs = path_case
    np.testing.assert_array_equal(
        outputs["logits"].argmax(-1), reference["logits"].argmax(-1),
        err_msg="Per-position argmax predictions changed.",
    )


def test_logits_close_to_reference(path_case):
    cfg, reference, outputs = path_case
    np.testing.assert_allclose(
        outputs["logits"], reference["logits"],
        atol=cfg["atol"], rtol=cfg["rtol"],
        err_msg="MLM logits drifted beyond tolerance from the golden reference.",
    )


def test_base_pairing_close_to_reference(path_case):
    cfg, reference, outputs = path_case
    np.testing.assert_allclose(
        outputs["base_pairing"], reference["base_pairing"],
        atol=cfg["atol"], rtol=cfg["rtol"],
        err_msg="base_pairing interaction head drifted beyond tolerance.",
    )


def regenerate() -> None:
    """Recompute and overwrite both golden references. Run only after an
    intentional, verified model change. The bf16 reference needs a CUDA GPU with
    flash-attn; it is skipped with a warning if CUDA is unavailable."""
    from transformers import AutoTokenizer

    from minerva.modeling_minerva import MinervaForMaskedLM

    path = _candidate_model_paths()[0]
    tok = AutoTokenizer.from_pretrained(path)

    for name, cfg in CONFIGS.items():
        if cfg["device"] == "cuda" and not torch.cuda.is_available():
            print(f"skip {name}: CUDA unavailable")
            continue
        model = MinervaForMaskedLM.from_pretrained(path, dtype=cfg["dtype"])
        model = model.to(cfg["device"]).eval()
        enc = tok(SEQUENCE, return_tensors="pt").to(cfg["device"])
        with torch.no_grad():
            out = model(**enc, output_interactions=True)
        dest = DATA_DIR / cfg["reference"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest,
            sequence=np.array(SEQUENCE),
            input_ids=enc["input_ids"][0].cpu().numpy().astype(np.int64),
            logits=out.logits[0].float().cpu().numpy().astype(np.float32),
            base_pairing=out.interactions["base_pairing"][0].float().cpu().numpy().astype(np.float32),
        )
        print(f"wrote {dest} ({name}, model: {path})")


if __name__ == "__main__":
    regenerate()
