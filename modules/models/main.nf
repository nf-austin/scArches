process TRAIN_MODEL {
    conda "${moduleDir}/environment.yml"

    input:
        path train_h5ad
        val model_name
        val model_type
        val dataset_obs
        val celltype_obs
        val n_hvgs
        val max_epochs
        val finetune_epochs
        val use_gpu

    output:
    path "${model_name}", emit: model_dir

    script:
    def gpu_flag = use_gpu ? '--use_gpu' : ''
    """
    python3 ${moduleDir}/train.py \\
        --train_h5ad ${train_h5ad} \\
        --out_model ${model_name} \\
        --model_type ${model_type} \\
        --dataset_obs "${dataset_obs}" \\
        --celltype_obs "${celltype_obs}" \\
        --n_hvgs ${n_hvgs} \\
        --max_epochs ${max_epochs} \\
        --finetune_epochs ${finetune_epochs} \\
        ${gpu_flag}
    """
}

process APPLY_MODEL {
    tag { sample_id }

    conda "${moduleDir}/environment.yml"

    input:
    tuple val(sample_id), path(h5ad)
    path model_dir
    val model_type
    val celltype_obs
    val max_epochs
    val use_gpu

    output:
    tuple val(sample_id), path("${sample_id}_integrated.h5ad"), emit: h5ad

    script:
    def gpu_flag = use_gpu ? '--use_gpu' : ''
    """
    python3 ${moduleDir}/integrate.py \\
        --input_h5ad ${h5ad} \\
        --output_h5ad ${sample_id}_integrated.h5ad \\
        --in_model ${model_dir} \\
        --model_type ${model_type} \\
        --celltype_obs "${celltype_obs}" \\
        --max_epochs ${max_epochs} \\
        ${gpu_flag}
    """
}
