#!/usr/bin/env python3
import argparse
import anndata as ad
import scanpy as sc


def main():
    parser = argparse.ArgumentParser(description="Concatenate an array of h5ad files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_h5ad", required=True)
    args = parser.parse_args()

    adatas = [sc.read_h5ad(f) for f in args.inputs]

    # index_unique avoids barcode collisions across samples (10x barcodes can
    # repeat across libraries).
    combined = ad.concat(adatas, merge="same", index_unique="-")
    combined.write_h5ad(args.out_h5ad)


if __name__ == "__main__":
    main()