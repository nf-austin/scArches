#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { COMPRESS ; DECOMPRESS }    from "./modules/compress/main.nf"
include { TRAIN_MODEL; APPLY_MODEL } from "./modules/models/main.nf"
include { MAKE_REPORT }              from "./modules/qc/main.nf"
include { CONCAT_H5ADS }             from "./modules/concat_h5ads/main.nf"

workflow {
    def model_type = params.model_type.toLowerCase()
    if (!(model_type in ['scvi', 'scanvi', 'scpoli'])) {
        exit 1, "Execution halted: Unknown --model_type '${params.model_type}'. Must be one of scvi, scanvi, scpoli."
    }
    def latent_key = [scvi: 'X_scVI', scanvi: 'X_scANVI', scpoli: 'X_scPoli'][model_type]
    def model_file = "${params.outdir}/${params.model_name}.tar.gz"

    channel.fromPath(params.h5ad_dir)
        | map { f -> tuple(f.baseName.replaceFirst(/_annotated$/, ''), f) }
        | set { ch_samples }

    if (params.train_model) {
        if (!params.train_h5ad || !file(params.train_h5ad).exists()) {
            exit 1, "Execution halted: --train_h5ad ('${params.train_h5ad}') does not exist or was not provided."
        }

        TRAIN_MODEL(
            file(params.train_h5ad),
            params.model_name,
            model_type,
            params.dataset_obs,
            params.celltype_obs,
            params.n_hvgs,
            params.train_max_epochs,
            params.finetune_epochs
        )
        COMPRESS(TRAIN_MODEL.out.model_dir)
        ch_model_dir = TRAIN_MODEL.out.model_dir.first()
    } else {
        if (!file(model_file).exists()) {
            exit 1, "Execution halted: The required model file (${model_file}) does not exist."
        }

        DECOMPRESS(Channel.value(file(model_file)))
        ch_model_dir = DECOMPRESS.out.decompressed_file.first()
    }

    APPLY_MODEL(
        ch_samples,
        ch_model_dir,
        model_type,
        params.celltype_obs,
        params.integrate_max_epochs
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
