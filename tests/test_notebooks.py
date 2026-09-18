"""Smoke tests for ``examples/notebooks``.

The static checks always run. Executing the notebooks needs a CUDA GPU, access
to the checkpoint and a few minutes, so it is opt-in:

    pip install -e ".[viz,dev]"
    MINERVA_RUN_NOTEBOOKS=1 pytest tests/test_notebooks.py

Form values are overridden to keep runs short. Set ``$MINERVA_MODEL_PATH`` to
load a local checkpoint instead of ``gbrixi/minerva-mlm``.
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "examples" / "notebooks"
HUB_MODEL = "gbrixi/minerva-mlm"

# (notebook, form overrides, files the run must produce)
CASES = [
    ("loci_viewer", {"run_jacobian": True, "jac_max_tokens": 48}, ["heads.html", "jacobian.html"]),
    ("loci_viewer", {"renderer": "publication (PDF)", "example": "twoayggay", "head_set": "l6 (last-6)"},
     ["heads.pdf"]),
    ("finetune", {"max_steps": 4}, ["minerva_lora_adapter/adapter_config.json", "finetune_before_after.pdf"]),
    ("rna_structure", {}, ["structure.fa", "structure.ct"]),
]


def _load(name):
    return json.loads((NOTEBOOK_DIR / f"{name}.ipynb").read_text())


def _code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _override(source, name, value):
    """Rewrite ``name = ...`` (keeping any ``#@param`` annotation); returns (source, n_replaced)."""
    pattern = re.compile(rf"^({re.escape(name)}\s*=\s*)(.*?)(\s*#@param.*)?$", re.M)
    return pattern.subn(lambda m: f"{m.group(1)}{value!r}{m.group(3) or ''}", source, count=1)


def test_every_notebook_has_a_case():
    assert {p.stem for p in NOTEBOOK_DIR.glob("*.ipynb")} == {name for name, _, _ in CASES}


@pytest.mark.parametrize("name", sorted({name for name, _, _ in CASES}))
def test_notebook_is_clean_and_parses(name):
    nb = _load(name)
    assert f"/examples/notebooks/{name}.ipynb" in "".join(nb["cells"][0]["source"]), "Colab badge points elsewhere"
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None, "commit notebooks without outputs"
    for source in _code_cells(nb):
        # IPython shell escapes are not Python; everything else must compile.
        ast.parse(re.sub(r"^(\s*)!.*$", r"\1pass", source, flags=re.M))


@pytest.mark.parametrize("name,overrides", [(n, o) for n, o, _ in CASES])
def test_overrides_target_real_form_fields(name, overrides):
    sources = _code_cells(_load(name))
    for key, value in overrides.items():
        assert sum(_override(s, key, value)[1] for s in sources) == 1, f"{name}: no single `{key} = ...` line"


@pytest.mark.skipif(not os.environ.get("MINERVA_RUN_NOTEBOOKS"), reason="set MINERVA_RUN_NOTEBOOKS=1 to execute")
@pytest.mark.parametrize("name,overrides,expected", CASES)
def test_notebook_executes(name, overrides, expected, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("notebooks need a CUDA GPU")

    nb = nbformat.read(NOTEBOOK_DIR / f"{name}.ipynb", as_version=4)
    model = os.environ.get("MINERVA_MODEL_PATH", HUB_MODEL)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell.source = cell.source.replace(f'"{HUB_MODEL}"', repr(model))
        for key, value in overrides.items():
            cell.source, _ = _override(cell.source, key, value)

    # Run from an empty directory so ``minerva`` comes from the installed
    # package and outputs land in tmp_path.
    nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}}).execute()
    for rel in expected:
        assert (tmp_path / rel).exists(), f"{name} did not write {rel}"
