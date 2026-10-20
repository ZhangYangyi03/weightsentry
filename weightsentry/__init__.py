"""weightsentry - static supply-chain audit for local model artifacts.

Pure standard library on purpose for the static layer: an audit tool must not
import the very framework (torch/pickle) whose loaders are the attack surface
it inspects. The numeric layer needs numpy and nothing else.
"""
import os as _os

# OpenBLAS reserves a private buffer per thread when it loads. On a machine with
# a small page file that reservation fails and every matmul dies with a
# MemoryError that looks like a bug in the caller -- including 16x16 products.
# Pin the pools before numpy is ever imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

__version__ = "0.2.0"
