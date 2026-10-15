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
    // numba (pulled in by scanpy) JIT-caches next to its package by default,
    // and matplotlib wants ~/.config. Neither is writable inside a read-only
    // Singularity image or with a read-only $HOME, which makes the import die
    // with "cannot cache function ... no locator available". Point both at the
    // task work dir, which is always writable.
    """
    export NUMBA_CACHE_DIR="\$PWD/.numba_cache"
    export MPLCONFIGDIR="\$PWD/.mplconfig"

    concat_h5ads.py \\
        --inputs ${h5ads} \\
        --out_h5ad combined_annotated.h5ad
    """

    stub:
    """
    touch combined_annotated.h5ad
    """
}
