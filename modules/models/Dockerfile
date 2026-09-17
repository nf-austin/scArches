# Single image for every process in this pipeline: the scArches stack is a
# superset of what the QC, concat and compress steps need.
#
# A custom image rather than a public biocontainer because scArches is not on
# conda-forge or Bioconda -- it installs from a git ref, which is exactly why
# this needs pinning rather than tracking a branch.
#
# BASE_IMAGE lets the CI build a CPU and a CUDA variant from one Dockerfile.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        tar \
        gzip \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Pinned to a commit, not @master: an unpinned git ref makes every environment
# build a different one. Refresh deliberately by bumping the SHA below.
# (PyPI also publishes scarches releases, e.g. scarches==0.6.1, if you would
# rather track releases than this branch.)
# Let scArches drive its own dependency versions.
#
# The repo previously pinned anndata==0.9.2 and pandas<2.2 alongside an
# unpinned `scanpy`. Those pins are incompatible with this scArches commit:
# it imports `SysVI` from scvi.external, which needs a scvi-tools recent enough
# to require anndata >= 0.10, while scanpy >= 1.10 needs `anndata.io` (also
# 0.10+). Holding anndata at 0.9.2 makes the set unsolvable, and the failure
# only shows up at import time, not during install.
#
# The scArches commit stays pinned -- that is the reproducibility anchor -- and
# scanpy / anndata / scvi-tools resolve to whatever it actually requires.
#
# numpy<2 is required, not cosmetic: scArches holds pandas at 1.5.3, whose C
# extension is built against the numpy 1.x ABI. Left free, pip installs numpy 2
# and `import pandas` dies with "numpy.dtype size changed ... Expected 96 from C
# header, got 88". Verified by resolving the tree in this base image.
RUN pip install --no-cache-dir \
        "numpy<2" \
        "git+https://github.com/theislab/scarches.git@d35cde61fff4dec6b86660817f4750fa35ff836d" \
        matplotlib \
        scikit-learn

RUN python3 -c 'import scanpy, anndata, pandas, scarches; print("scarches", scarches.__version__); print("All packages OK")'
