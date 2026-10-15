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
    // numba (pulled in by scanpy) JIT-caches next to its package by default,
    // and matplotlib wants ~/.config. Neither is writable inside a read-only
    // Singularity image or with a read-only $HOME, which makes the import die
    // with "cannot cache function ... no locator available". Point both at the
    // task work dir, which is always writable.
    """
    export NUMBA_CACHE_DIR="\$PWD/.numba_cache"
    export MPLCONFIGDIR="\$PWD/.mplconfig"

    report.py \\
        --input_h5ad ${combined_h5ad} \\
        --output_pdf qc_report.pdf \\
        --latent_key ${latent_key} \\
        --celltype_obs "${celltype_obs}"
    """

    stub:
    """
    touch qc_report.pdf
    """
}
