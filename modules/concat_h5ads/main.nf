process CONCAT_H5ADS {
    publishDir "${params.outdir}", mode: 'copy'

    conda "${moduleDir}/environment.yml"

    input:
    path h5ads

    output:
    path "combined_annotated.h5ad", emit: combined_h5ad

    script:
    """
    python3 ${moduleDir}/concat_h5ads.py \\
        --inputs ${h5ads} \\
        --out_h5ad combined_annotated.h5ad
    """
}