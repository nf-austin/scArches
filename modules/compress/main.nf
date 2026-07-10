process COMPRESS {
    publishDir "${params.outdir}", mode: 'copy'

    input:
        path in_file

    output:
        path "${in_file}.tar.gz", emit: compressed_file

    script:
        """
        tar -czf ${in_file}.tar.gz ${in_file}
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
}