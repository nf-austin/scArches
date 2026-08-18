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
3. **APPLY_MODEL** (`scvi-tools`/`scArches`) — For each file in `--h5ad_dir`, checks the query's gene
   panel against the reference's (see [Query gene panel](#query-gene-panel)), then maps the query
   dataset onto the reference model via `load_query_data`, writing the integrated latent embedding
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

## Query gene panel

scArches' surgery silently zero-fills any of the reference model's genes that a query file is
missing. That padding is not neutral — it hits low-UMI cell types (T/NK cells especially) hardest,
because once their few informative genes are zeroed there is little left to place them and they
collapse onto whichever dense reference neighbourhood survives. The symptom is confident-but-wrong
label transfer rather than an error.

`APPLY_MODEL` therefore reports the overlap per query file and fails below `--min_gene_overlap`
(90% by default). If a run stops here, check the reported missing genes first: a near-total miss is
usually an identifier mismatch (Ensembl gene IDs vs gene symbols) rather than a genuinely different
panel, and the log calls that case out explicitly. Re-key one side and rerun, or pass
`--min_gene_overlap 0` to map anyway.

## Label balancing

Both label-transfer paths favour a reference's largest classes: SCANVI's classification head trains
on the raw label distribution, and the weighted-KNN scores each label by its share of a query cell's
neighbours. On a reference spanning orders of magnitude between its largest and smallest cell type,
that prior pulls ambiguous query cells onto the abundant classes.

`--n_samples_per_label` (SCANVI fine-tuning) and `--max_cells_per_label` (weighted-KNN) counteract
this by sampling a fixed number of cells per label instead. `--max_cells_per_label` also bounds the
KNN's neighbour index, which is brute-force and scans every reference cell it was fit on for each
query cell — so raising the cap on a large reference costs query time as well as accuracy.

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

| Parameter                | Default       | Description                                                                                                                                                      |
|--------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--h5ad_dir`             | `data/*.h5ad` | Glob pattern for query h5ad files to integrate.                                                                                                                  |
| `--outdir`               | `results`     | Output directory.                                                                                                                                                |
| `--train_model`          | `true`        | Train a new reference model; set to `false` to reuse an existing one.                                                                                            |
| `--model_type`           | `SCANVI`      | Reference model backend: `SCVI`, `SCANVI`, or `SCPOLI`.                                                                                                          |
| `--model_name`           | `model`       | Name of the model artifact (`<model_name>.tar.gz` under `--outdir`).                                                                                             |
| `--train_h5ad`           | _(none)_      | Reference dataset to train on. Required when `--train_model true`.                                                                                               |
| `--dataset_obs`          | _(empty)_     | obs column identifying batches/datasets. Blank treats all cells as one batch.                                                                                    |
| `--celltype_obs`         | `cell_type`   | obs column with reference cell type labels.                                                                                                                      |
| `--n_hvgs`               | `6000`        | Number of highly variable genes used for training.                                                                                                               |
| `--train_max_epochs`     | `200`         | Max training epochs for `TRAIN_MODEL`.                                                                                                                           |
| `--finetune_epochs`      | `20`          | SCANVI fine-tuning epochs after the SCVI pretraining stage.                                                                                                      |
| `--integrate_max_epochs` | `100`         | Max epochs for `APPLY_MODEL`'s query mapping.                                                                                                                    |
| `--min_gene_overlap`     | `0.9`         | Fraction of the reference model's genes that must be present in a query file before `APPLY_MODEL` will map it; the rest are silently zero-filled by the surgery. `0` disables the check. See [Query gene panel](#query-gene-panel). |
| `--n_layers`             | `3`           | Number of hidden layers in the encoder/decoder (SCVI/SCANVI: `n_layers`; scPoli: repeats its own default layer width `n_layers` times via `hidden_layer_sizes`). |
| `--dropout_rate`         | `0.2`         | Dropout rate applied in the encoder/decoder (SCVI/SCANVI: `dropout_rate`; scPoli: `dr_rate`, whose own default is `0.05`).                                       |
| `--learning_rate`        | `0.001`       | Optimizer learning rate for all `TRAIN_MODEL` stages (SCVI, SCANVI's pretraining and fine-tuning, and scPoli).                                                   |
| `--batch_size`           | `1024`        | Minibatch size passed to `TRAIN_MODEL`'s `model.train()`, over the whole reference.                                                                               |
| `--integrate_batch_size` | `128`         | Minibatch size passed to `APPLY_MODEL`'s `model.train()`. Kept separate from, and much smaller than, `--batch_size`: a query file holds a single sample, so a reference-scale batch size leaves the surgery only a handful of steps per epoch to fit the query's new batch embedding from scratch. |
| `--use_knn`              | `false`       | Transfer labels with the weighted-KNN classifier (fit on the reference latents at training time) instead of the model's native classifier. SCVI has no native classifier and always uses the KNN, regardless of this flag. |
| `--knn_neighbors`        | `50`          | Number of neighbors for the weighted-KNN classifier `TRAIN_MODEL` fits on the reference latent space.                                                            |
| `--max_cells_per_label`  | `50000`       | Cap on reference cells per label when fitting the weighted-KNN classifier, which equalizes its abundance-driven label prior and bounds its brute-force neighbour index. `0` uses every cell. See [Label balancing](#label-balancing). |
| `--n_samples_per_label`  | `100`         | SCANVI only: cells sampled per label per epoch during the fine-tuning stage, so an unbalanced reference doesn't dominate the classification head. `0` uses scvi-tools' unbalanced default. |
| `--use_gpu`              | `false`       | Train/apply on GPU instead of CPU. Adds `--gpus all`/`--nv` to the `docker`/`singularity` profiles and requests an `accelerator` on cluster/cloud executors.     |
| `--max_memory`           | `128.GB`      | Memory cap applied to all processes.                                                                                                                             |
| `--max_cpus`             | `32`          | CPU cap applied to all processes.                                                                                                                                |
| `--max_time`             | `72.h`        | Runtime cap applied to all processes.                                                                                                                            |

## Output structure

```text
results/
├── <model_name>.tar.gz          # Trained reference model artifact
├── combined_annotated.h5ad      # All query samples integrated and merged; obsm/obs columns added:
│                                 #   X_scVI/X_scANVI/X_scPoli, predicted_cell_type
└── qc_report.pdf                # UMAP QC report over the merged, integrated dataset
```
