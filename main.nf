#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { COMPRESS ; DECOMPRESS }    from "./modules/compress/main.nf"
include { TRAIN_MODEL; APPLY_MODEL } from "./modules/models/main.nf"
include { MAKE_REPORT }              from "./modules/qc/main.nf"
include { CONCAT_H5ADS }             from "./modules/concat_h5ads/main.nf"

def helpMessage() {
    log.info """
    nf-austin/scArches -- reference-based integration and label transfer

    Usage, samplesheet (recommended; this is what Seqera Platform launches with):
      nextflow run main.nf -profile docker --input samplesheet.csv \\
          --train_h5ad reference.h5ad --outdir results

      samplesheet.csv columns: sample, h5ad  (the QUERY datasets to map)

    Usage, ad-hoc glob:
      nextflow run main.nf -profile docker --h5ad_dir "data/*.h5ad" --train_h5ad reference.h5ad

    Required (one of):
      --input         Samplesheet CSV of query datasets.
      --h5ad_dir      Glob matching query .h5ad files.

    Model:
      --train_model   Train a reference model (default: ${params.train_model}).
      --train_h5ad    Reference atlas to train on; required when --train_model true.
      --model_tarball Reuse a previously trained model; required when --train_model false.
      --model_type    SCVI, SCANVI or SCPOLI (default: ${params.model_type}).

    Common options:
      --outdir        Output directory (default: ${params.outdir}).
      --celltype_obs  obs column with reference cell type labels (default: ${params.celltype_obs}).
      --dataset_obs   obs column identifying batches/datasets.
      --use_gpu       Train and apply on GPU (default: ${params.use_gpu}).

    On a cluster, add `-profile slurm,singularity --slurm_queue <partition>`, and for
    GPU work `--use_gpu true` (which adds --gres via --gpu_cluster_options). See the
    README for the HPC notes.
    """.stripIndent()
}

/**
 * Resolve one samplesheet entry to a file.
 *
 * A relative entry is resolved against the samplesheet's OWN directory first,
 * which is what someone editing that sheet expects. Nextflow's default is the
 * launch directory, and on Seqera Platform the launch directory is the work
 * directory -- so a relative path there silently resolves somewhere unrelated.
 * Falls back to launch-dir resolution, and only then reports the entry missing.
 */
def resolveInput(path, sheet_dir, row_num) {
    if (path.startsWith('/') || path ==~ /^[a-zA-Z][a-zA-Z0-9+.-]*:\/\/.*/) {
        return file(path, checkIfExists: true)
    }
    def beside_sheet = sheet_dir.resolve(path)
    if (beside_sheet.exists()) {
        return beside_sheet
    }
    def from_launch = file(path)
    if (from_launch.exists()) {
        return from_launch
    }
    error "Samplesheet row ${row_num}: h5ad not found as '${beside_sheet}' (relative to the samplesheet) nor as '${from_launch}' (relative to the launch directory). Use an absolute path."
}

/**
 * Turn samplesheet rows into (sample_id, h5ad) tuples.
 *
 * Validated eagerly over the fully-read row list rather than inside a channel
 * closure: errors raised in a closure are lazy -- they never fire under
 * -preview, and in a real run they surface only once the channel is consumed.
 * A bad samplesheet must fail at launch, before Platform provisions compute.
 */
def buildSamples(rows, sheet_dir) {
    if (!rows) {
        error "Samplesheet is empty: ${params.input}"
    }
    def required = ['sample', 'h5ad']
    def missing = required.findAll { c -> !rows[0].containsKey(c) }
    if (missing) {
        error "Samplesheet is missing column(s): ${missing.join(', ')}. Found: ${rows[0].keySet().join(', ')}"
    }

    def seen = [] as Set
    return rows.withIndex().collect { row, idx ->
        def sample_id = row.sample?.trim()
        if (!sample_id) {
            error "Samplesheet row ${idx + 1} has an empty 'sample' value"
        }
        if (!seen.add(sample_id)) {
            error "Samplesheet has a duplicate sample id: '${sample_id}'. Sample ids become output paths and must be unique."
        }
        def h5ad = row.h5ad?.trim()
        if (!h5ad) {
            error "Samplesheet row ${idx + 1} ('${sample_id}') has an empty 'h5ad' value"
        }
        tuple(sample_id, resolveInput(h5ad, sheet_dir, idx + 1))
    }
}


workflow {
    if (params.help) {
        helpMessage()
        return
    }

    def model_type = params.model_type.toLowerCase()
    if (!(model_type in ['scvi', 'scanvi', 'scpoli'])) {
        error "Unknown --model_type '${params.model_type}'. Must be one of scvi, scanvi, scpoli."
    }
    def latent_key = [scvi: 'X_scVI', scanvi: 'X_scANVI', scpoli: 'X_scPoli'][model_type]

    if (params.input && params.h5ad_dir) {
        error "Use either --input (samplesheet) or --h5ad_dir (glob), not both."
    }
    if (!params.input && !params.h5ad_dir) {
        error "No query input given. Provide --input samplesheet.csv or --h5ad_dir \"data/*.h5ad\". Run with --help for details."
    }

    if (params.input) {
        def sheet = file(params.input, checkIfExists: true)
        def rows = sheet.splitCsv(header: true, strip: true)
        ch_samples = channel.fromList(buildSamples(rows, sheet.parent))
    }
    else {
        ch_samples = channel.fromPath(params.h5ad_dir, checkIfExists: true)
            .map { f -> tuple(f.baseName.replaceFirst(/_annotated$/, ''), f) }
    }

    log.info """
    P I P E L I N E   nf-austin/scArches
    ====================================
    query      : ${params.input ?: params.h5ad_dir}
    model      : ${params.train_model ? "train from ${params.train_h5ad}" : "reuse ${params.model_tarball ?: "${params.outdir}/${params.model_name}.tar.gz"}"}
    model type : ${params.model_type}
    gpu        : ${params.use_gpu}
    outdir     : ${params.outdir}
    """.stripIndent()

    if (params.train_model) {
        if (!params.train_h5ad) {
            error "--train_model true requires --train_h5ad (the reference atlas to train on)."
        }
        def train_file = file(params.train_h5ad, checkIfExists: true)

        TRAIN_MODEL(
            train_file,
            params.model_name,
            model_type,
            params.dataset_obs,
            params.celltype_obs,
            params.n_hvgs,
            params.train_max_epochs,
            params.finetune_epochs,
            params.batch_size,
            params.n_layers,
            params.dropout_rate,
            params.learning_rate,
            params.knn_neighbors,
            params.n_samples_per_label,
            params.max_cells_per_label,
            params.min_dataset_detection,
            params.use_gpu
        )
        COMPRESS(TRAIN_MODEL.out.model_dir)
        ch_model_dir = TRAIN_MODEL.out.model_dir.first()
    } else {
        // Prefer an explicit --model_tarball. Falling back to a path inside
        // --outdir only works when a previous run wrote there and outdir is
        // still readable, which is not true on a fresh Platform launch.
        def model_file = params.model_tarball ?: "${params.outdir}/${params.model_name}.tar.gz"
        def model_path = file(model_file)
        if (!model_path.exists()) {
            error "--train_model false needs an existing model archive, but '${model_file}' does not exist. Point at one with --model_tarball, or run with --train_model true first."
        }

        DECOMPRESS(channel.value(model_path))
        ch_model_dir = DECOMPRESS.out.decompressed_file.first()
    }

    APPLY_MODEL(
        ch_samples,
        ch_model_dir,
        model_type,
        params.celltype_obs,
        params.dataset_obs,
        params.integrate_max_epochs,
        params.integrate_batch_size,
        params.min_gene_overlap,
        params.use_gpu,
        params.use_knn
    )

    APPLY_MODEL.out.h5ad
        | map { _sample_id, h5ad -> h5ad }
        | collect
        | set { ch_all_h5ads }

    CONCAT_H5ADS(ch_all_h5ads)

    // Integration quality is a cross-sample property, so QC runs once on the merged
    // dataset rather than per-sample (unlike per-sample modules such as CONCAT_H5ADS' inputs).
    MAKE_REPORT(CONCAT_H5ADS.out.combined_h5ad, latent_key, params.celltype_obs)
}
