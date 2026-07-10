# <pipeline-name>

A Nextflow DSL2 pipeline for [pipeline purpose].

## Pipeline steps

1. **Step 1** (`tool`) — Description.
2. **Step 2** (`tool`) — Description.

## Requirements

- Nextflow >= 24.04.0
- Docker or Singularity

## Usage

```bash
nextflow run nf-austin/<pipeline-name> \
    -profile docker \
    --input "data/*"
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--outdir` | `results` | Output directory. |

## Output structure

```text
results/
├── step1_output/
└── step2_output/
```
