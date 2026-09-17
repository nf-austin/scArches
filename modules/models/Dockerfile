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
RUN pip install --no-cache-dir \
        'scanpy' \
        'anndata==0.9.2' \
        'pandas<2.2' \
        'setuptools<82' \
        matplotlib \
        scikit-learn \
        "git+https://github.com/theislab/scarches.git@d35cde61fff4dec6b86660817f4750fa35ff836d"

# Fail the build rather than a run if an import is broken.
RUN python3 -c 'import scanpy, anndata, pandas, scarches; print("scarches", scarches.__version__); print("All packages OK")'
