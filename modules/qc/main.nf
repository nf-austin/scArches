process MAKE_REPORT {
    publishDir "${params.outdir}", mode: 'copy'

    conda "${moduleDir}/environment.yml"

    input:
    path combined_h5ad
    val latent_key
    val celltype_obs

    output:
    path "qc_report.pdf", emit: report

    script:
    """
    python3 ${moduleDir}/report.py \\
        --input_h5ad ${combined_h5ad} \\
        --output_pdf qc_report.pdf \\
        --latent_key ${latent_key} \\
        --celltype_obs "${celltype_obs}"
    """
}
