process CONCAT_H5ADS {
    // Spans every sample, so no `tag`.
    publishDir "${params.outdir}", mode: 'copy'

    conda "${moduleDir}/environment.yml"
    // Container comes from the process-level default in nextflow.config.

    input:
    path h5ads

    output:
    path "combined_annotated.h5ad", emit: combined_h5ad

    script:
    """
    concat_h5ads.py \\
        --inputs ${h5ads} \\
        --out_h5ad combined_annotated.h5ad
    """

    stub:
    """
    touch combined_annotated.h5ad
    """
}
