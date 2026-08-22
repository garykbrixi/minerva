"""RiNALMo model code, vendored from github.com/lbcb-sci/RiNALMo.

Copyright the RiNALMo authors, Apache-2.0 (see LICENSE in this directory).

Changes from upstream:
  - flash-attn and einops imports guarded, so the module loads without them and
    falls back to RiNALMo's own dot_product_attention path
  - intra-package imports rewritten to this directory
"""

from .model import RiNALMo  # noqa: F401
