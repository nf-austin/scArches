#!/usr/bin/env python3
import argparse

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scanpy as sc


def report(adata: ad.AnnData, pdf_file: str, latent_key: str, celltype_obs: str):
    sc.pp.neighbors(adata, use_rep=latent_key)
    sc.tl.umap(adata)

    color_keys = [
        key for key in (celltype_obs, "predicted_cell_type", "prediction_probability", "prediction_uncertainty")
        if key in adata.obs.columns
    ]
    if not color_keys:
        color_keys = [None]

    with PdfPages(pdf_file) as pdf:
        for key in color_keys:
            sc.pl.umap(adata, color=key, show=False)
            pdf.savefig(plt.gcf())
            plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Generate a QC report of UMAP + cell type assignments.")
    parser.add_argument("--input_h5ad", required=True)
    parser.add_argument("--output_pdf", required=True)
    parser.add_argument("--latent_key", required=True)
    parser.add_argument("--celltype_obs", required=True)
    args = parser.parse_args()

    report(sc.read_h5ad(args.input_h5ad), args.output_pdf, args.latent_key, args.celltype_obs)


if __name__ == "__main__":
    main()
