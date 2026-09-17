# nf-austin/scArches

A Nextflow DSL2 pipeline for integration/annotation using [scArches](https://docs.scarches.org/en/latest/index.html).

The workflow trains a reference integration model first (or reuses a previously trained one), then
uses scArches to map each query dataset onto that reference, transferring embeddings and (for
SCANVI/scPoli) cell type labels.

## Pipeline steps

1. **TRAIN_MODEL** (`scvi-tools`/`scArches`) — Drops genes that are structurally absent from whole
   datasets (see [Reference gene panel](#reference-gene-panel)), calls HVGs, then trains a reference
   model (SCVI, SCANVI, or scPoli) on `--train_h5ad`. Skipped when `--train_model false`.
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
- Docker (local) or Singularity/Apptainer (HPC); Conda works as a fallback
- Optional: a SLURM cluster, and a GPU for training — see [HPC / SLURM](#hpc--slurm)
- For `--use_gpu true`: a CUDA-capable GPU and, for the `docker`/`singularity` profiles, the
  NVIDIA Container Toolkit (or equivalent) so the container runtime can pass the GPU through

## Model reuse

`--model_name` must match between the run that trains the model and any later run reusing it:
training publishes `results/<model_name>.tar.gz`, and a `--train_model false` run reads that same
path back in.

## Reference gene panel

A reference stitched together from several sub-atlases by an outer join keeps genes present in only
some sources and zero-fills them in the rest. Those structural zeros are not biology — the model
learns them as near-perfect sub-atlas discriminators, so a query with a complete panel carries real
signal in them and maps to whichever sub-atlas "has" them rather than to its own tissue. In a
unified lung+brain reference this surfaces as lung myeloid cells labelled Microglia, because MRC1
and FCN1 are nonzero only on the brain side.

`TRAIN_MODEL` drops any gene detected in fewer than `--min_dataset_detection` of the
`--dataset_obs` levels, before HVG selection — a gene that is zero across half the datasets and
expressed across the other half looks enormously variable, so leaving the filter until later lets
the artifacts get picked *as* HVGs. Progress lines are prefixed `[panel]` and report the absence
pattern per dataset, which is worth reading: a single dataset responsible for most of the dropped
genes is usually cheaper to exclude from the reference than to make every other dataset pay for.

The threshold is a fraction rather than a count so that it scales with however many datasets the
reference has — the requirement can never exceed the levels that exist, and a single-dataset
reference resolves to `1`. The default of `1.0` requires every dataset, which is the exact
intersection of their panels. Lower it only when you know a particular source is missing genes you
want to keep, and bear in mind that a partially-present gene is exactly the kind that acts as a
sub-atlas discriminator.

Genuinely tissue-restricted genes survive the filter, because ambient RNA puts them at some low
level in every dataset — SFTPC is detected atlas-wide even though only lung has AT2 cells. Only
never-measured genes are *exactly* zero. Gene-symbol synonyms split across annotation releases
(`MARCH8` vs `MARCHF8`) are dropped rather than merged, since each half is absent from the datasets
that used the other name.

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
| `--input`                | _(one of these two)_ | Samplesheet CSV of query datasets, columns `sample,h5ad`. |
| `--h5ad_dir`             | _(one of these two)_ | Glob pattern for query h5ad files to integrate. |
| `--outdir`               | `results`     | Output directory.                                                                                                                                                |
| `--train_model`          | `true`        | Train a new reference model; set to `false` to reuse an existing one.                                                                                            |
| `--model_type`           | `SCANVI`      | Reference model backend: `SCVI`, `SCANVI`, or `SCPOLI`.                                                                                                          |
| `--model_name`           | `model`       | Name of the model artifact (`<model_name>.tar.gz` under `--outdir`).                                                                                             |
| `--train_h5ad`           | _(none)_      | Reference dataset to train on. Required when `--train_model true`. |
| `--model_tarball`        | _(none)_      | Trained model archive to reuse. Required when `--train_model false` on Platform, where `<outdir>/<model_name>.tar.gz` will not exist. |
| `--dataset_obs`          | _(empty)_     | obs column identifying batches/datasets. Blank treats all cells as one batch.                                                                                    |
| `--celltype_obs`         | `cell_type`   | obs column with reference cell type labels.                                                                                                                      |
| `--n_hvgs`               | `6000`        | Number of highly variable genes used for training.                                                                                                               |
| `--train_max_epochs`     | `200`         | Max training epochs for `TRAIN_MODEL`.                                                                                                                           |
| `--finetune_epochs`      | `20`          | SCANVI fine-tuning epochs after the SCVI pretraining stage.                                                                                                      |
| `--integrate_max_epochs` | `100`         | Max epochs for `APPLY_MODEL`'s query mapping.                                                                                                                    |
| `--min_dataset_detection`| `1.0`         | Fraction of the `--dataset_obs` levels a gene must be detected in to survive into training (`1.0` = all of them); removes genes an outer-join reference merge left structurally zero in whole sub-atlases. `0` disables, as does a blank `--dataset_obs`. See [Reference gene panel](#reference-gene-panel). |
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

| `--scarches_container`     | `ghcr.io/nf-austin/scarches:1.0.0`      | CPU image (multi-arch). |
| `--scarches_container_gpu` | `ghcr.io/nf-austin/scarches:1.0.0-cuda` | CUDA image, used with `--use_gpu true`. |
| `--scarches_container_any` | _(auto)_      | Pin a specific image, overriding the CPU/GPU selection. |
| `--gpu_cluster_options`  | `--gres=gpu:1` | sbatch options used to request a GPU on SLURM, which ignores `accelerator`. Combined with `--cluster_options`. |
| `--slurm_queue`          | _(cluster default)_ | SLURM partition (`sbatch --partition`). Used by `-profile slurm`. |
| `--slurm_account`        | _(none)_      | SLURM account to charge (`sbatch --account`). |
| `--cluster_options`      | _(none)_      | Raw sbatch options added to every job. Use the `=` form for values starting with `--`. |
| `--singularity_cache_dir` | `$NXF_SINGULARITY_CACHEDIR` | Shared directory for pulled images. |
| `--conda_cache_dir`      | `$NXF_CONDA_CACHEDIR` | Shared directory for conda environments. |
| `--singularity_bind`     | _(none)_      | Extra bind mounts, comma-separated, e.g. `/mnt/gpfs,/scratch`. |

## Output structure

```text
results/
├── <model_name>.tar.gz          # Trained reference model artifact
├── combined_annotated.h5ad      # All query samples integrated and merged; obsm/obs columns added:
│                                 #   X_scVI/X_scANVI/X_scPoli, predicted_cell_type
└── qc_report.pdf                # UMAP QC report over the merged, integrated dataset
```

## Seqera Platform (Nextflow Tower)

The repo ships everything Platform needs:

- **`nextflow_schema.json`** — renders the launch form. `--input` appears as a file picker wired to
  Data Explorer, options are grouped by stage, and tuning knobs are marked hidden.
- **`assets/schema_input.json`** — the samplesheet contract (`sample`, `h5ad`) for the **query**
  datasets. The reference atlas is separate: `--train_h5ad`.
- **`tower.yml`** — puts the QC report, the merged h5ad and the trained model archive in the run's
  **Reports** tab.

To add it: **Pipelines → Add pipeline**, point at this repository, and pick a compute environment.
Use **absolute paths** for `--input`, the h5ads it references, `--train_h5ad` and `--outdir`.

**Reusing a model on Platform needs `--model_tarball`.** With `--train_model false` the pipeline
otherwise looks for `<outdir>/<model_name>.tar.gz`, which only resolves when a previous run wrote
there — not true on a fresh scratch directory.

## HPC / SLURM

The `slurm` profile sets only the executor and queue, so it composes with an engine profile in
either order:

```bash
nextflow run nf-austin/scArches \
    -profile slurm,singularity \
    --slurm_queue gpu \
    --use_gpu true \
    --input /mnt/gpfs/project/queries.csv \
    --train_h5ad /mnt/gpfs/project/reference.h5ad \
    --outdir /mnt/gpfs/project/results \
    --singularity_cache_dir /mnt/gpfs/shared/singularity
```

- **GPU jobs need `--gres`, not `accelerator`.** The SLURM executor silently drops Nextflow's
  `accelerator` directive — verified on 26.04: `cpus` becomes `-c` and `memory` becomes `--mem`, but
  no `--gres` line is emitted. `--use_gpu true` therefore adds `--gpu_cluster_options` (default
  `--gres=gpu:1`) to the sbatch options for `TRAIN_MODEL` and `APPLY_MODEL`. Adjust for your site,
  e.g. `--gpu_cluster_options='--gres=gpu:a100:1'`. Your own `--cluster_options` is *combined* with
  it, not replaced.
- **Quote option values that start with `--` using the `=` form.** `--cluster_options '--qos=long'`
  is parsed by Nextflow as a bare flag followed by a separate `--qos` parameter, and the job ends up
  with a literal `true` in its sbatch header. Write `--cluster_options='--qos=long'`.
- **Under Singularity, `--nv` and `-B` binds are combined**, so requesting a GPU does not drop your
  bind mounts and vice versa.
- **Use absolute paths** for every input and for `--outdir`.
- **Put `--singularity_cache_dir` on shared storage.** `$HOME` is usually quota-limited and is not
  always mounted on compute nodes.
- **`--singularity_bind` is the escape hatch for symlinked filesystems.** `autoMounts` binds only
  the paths Nextflow resolved itself; if `/data` is a symlink to `/mnt/gpfs/...`, the container sees
  a dangling link and reports a missing file. Bind the real parent.
- **Seqera Platform already sets the executor** when you launch against a SLURM compute environment,
  so `-profile slurm` is mainly for launching by hand from a login node.

## Container images

Built from `modules/models/Dockerfile` and published to GHCR by `.github/workflows/docker.yml`:

| Tag | Base | Platforms | Used when |
| --- | --- | --- | --- |
| `ghcr.io/nf-austin/scarches:<ver>` | `python:3.11-slim` | amd64, arm64 | default (CPU) |
| `ghcr.io/nf-austin/scarches:<ver>-cuda` | PyTorch CUDA runtime | amd64 | `--use_gpu true` |

One image serves every process — the scArches stack is a superset of what the QC, concat and
compress steps need. A custom image rather than a public biocontainer because **scArches is not
packaged on conda-forge or Bioconda**; it installs from a git ref.

**The git ref is pinned to a commit, not `@master`.** An unpinned ref makes every environment build a
different one. Bump the SHA in `modules/models/Dockerfile` and `modules/models/environment.yml`
together when you want a newer scArches. The GHCR packages must be **public** for `nextflow run` to
pull them without credentials.

Every module also ships an `environment.yml`, so `-profile conda` remains a working fallback.

## Notes

- `nextflow run . -stub-run --input assets/samplesheet_example.csv --train_h5ad ref.h5ad` exercises
  the real channel wiring and publishing with no containers and no data.
- `nextflow lint main.nf nextflow.config modules/*/main.nf` catches config errors that `-preview`
  accepts.
