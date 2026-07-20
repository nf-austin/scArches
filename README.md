# nf-austin/scArches

A Nextflow DSL2 pipeline for integration/annotation using [scArches](https://docs.scarches.org/en/latest/index.html).

The workflow trains a reference integration model first (or reuses a previously trained one), then
uses scArches to map each query dataset onto that reference, transferring embeddings and (for
SCANVI/scPoli) cell type labels.

## Pipeline steps

1. **TRAIN_MODEL** (`scvi-tools`/`scArches`) — Trains a reference model (SCVI, SCANVI, or scPoli) on
   `--train_h5ad`. Skipped when `--train_model false`.
2. **COMPRESS** / **DECOMPRESS** (`tar`) — Packages the trained model directory into a single
   `<model_name>.tar.gz` artifact published to `results/`, and unpacks it again on later runs that
   reuse the model instead of retraining.
3. **APPLY_MODEL** (`scvi-tools`/`scArches`) — For each file in `--h5ad_dir`, maps the query dataset
   onto the reference model via `load_query_data`, writing the integrated latent embedding
   (`X_scVI`/`X_scANVI`/`X_scPoli`) and, for SCANVI/scPoli, predicted cell types back into the h5ad.
4. **CONCAT_H5ADS** (`anndata`) — Concatenates all integrated per-sample h5ads into a single
   `combined_annotated.h5ad`, deduplicating barcodes across samples.
5. **MAKE_REPORT** (`scanpy`/`matplotlib`) — Computes a UMAP over the merged, integrated latent space
   and renders cell-type/prediction overlays into `qc_report.pdf`.

## Requirements

- Nextflow >= 24.04.0
- Docker, Singularity, or Conda
- For `--use_gpu true`: a CUDA-capable GPU and, for the `docker`/`singularity` profiles, the
  NVIDIA Container Toolkit (or equivalent) so the container runtime can pass the GPU through

## Model reuse

`--model_name` must match between the run that trains the model and any later run reusing it:
training publishes `results/<model_name>.tar.gz`, and a `--train_model false` run reads that same
path back in.

## Usage

Train a reference model and integrate query datasets against it:

```bash
nextflow run nf-austin/scArches \
    -profile docker \
    --train_model true \
    --train_h5ad "data/reference.h5ad" \
    --celltype_obs cell_type \
    --h5ad_dir "data/query_*.h5ad"
```

Reuse a previously trained model (skips `TRAIN_MODEL`, requires `results/<model_name>.tar.gz` to
already exist):

```bash
nextflow run nf-austin/scArches \
    -profile docker \
    --train_model false \
    --h5ad_dir "data/query_*.h5ad"
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--h5ad_dir` | `data/*.h5ad` | Glob pattern for query h5ad files to integrate. |
| `--outdir` | `results` | Output directory. |
| `--train_model` | `true` | Train a new reference model; set to `false` to reuse an existing one. |
| `--model_type` | `SCANVI` | Reference model backend: `SCVI`, `SCANVI`, or `SCPOLI`. |
| `--model_name` | `model` | Name of the model artifact (`<model_name>.tar.gz` under `--outdir`). |
| `--train_h5ad` | _(none)_ | Reference dataset to train on. Required when `--train_model true`. |
| `--dataset_obs` | _(empty)_ | obs column identifying batches/datasets. Blank treats all cells as one batch. |
| `--celltype_obs` | `cell_type` | obs column with reference cell type labels. |
| `--n_hvgs` | `3000` | Number of highly variable genes used for training. |
| `--train_max_epochs` | `400` | Max training epochs for `TRAIN_MODEL`. |
| `--finetune_epochs` | `20` | SCANVI fine-tuning epochs after the SCVI pretraining stage. |
| `--integrate_max_epochs` | `200` | Max epochs for `APPLY_MODEL`'s query mapping. |
| `--use_gpu` | `false` | Train/apply on GPU instead of CPU. Adds `--gpus all`/`--nv` to the `docker`/`singularity` profiles and requests an `accelerator` on cluster/cloud executors. |
| `--max_memory` | `128.GB` | Memory cap applied to all processes. |
| `--max_cpus` | `32` | CPU cap applied to all processes. |
| `--max_time` | `72.h` | Runtime cap applied to all processes. |

## Output structure

```text
results/
├── <model_name>.tar.gz          # Trained reference model artifact
├── combined_annotated.h5ad      # All query samples integrated and merged; obsm/obs columns added:
│                                 #   X_scVI/X_scANVI/X_scPoli, predicted_cell_type (SCANVI/scPoli)
└── qc_report.pdf                # UMAP QC report over the merged, integrated dataset
```
