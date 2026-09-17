process COMPRESS {
    publishDir "${params.outdir}", mode: 'copy'

    // Container comes from the process-level default in nextflow.config.
    // Without one, this process would have no runtime at all under
    // -profile docker/singularity, which disable conda.

    input:
        path in_file

    output:
        path "${in_file}.tar.gz", emit: compressed_file

    script:
        """
        tar -czf ${in_file}.tar.gz ${in_file}
        """

    stub:
        """
        mkdir -p stub_model && touch stub_model/placeholder
        tar -czf ${in_file}.tar.gz stub_model
        """
}

process DECOMPRESS {
    input:
        path in_file

    output:
        path "${in_file.getSimpleName()}", emit: decompressed_file

    script:
        """
        tar -xzf ${in_file}
        """

    stub:
        """
        mkdir -p ${in_file.getSimpleName()}
        """
}
