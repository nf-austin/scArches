import argparse
import argparse

import anndata as ad
import scanpy as sc
import scarches as sca
import numpy as np


def integrate(
    adata: ad.AnnData,
    output_h5ad: str,
    in_model: str,
    model_type: str,
    celltype_obs: str,
    max_epochs: int
):
    orig_adata = adata.copy()

    # Ensure raw counts are stored in .layers["counts"] as required by scvi-tools
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    if model_type == "scvi":
        sca.models.SCVI.prepare_query_anndata(adata, in_model)
        query_model = sca.models.SCVI.load_query_data(adata, in_model, freeze_dropout=True)
        query_model.train(max_epochs=max_epochs, plan_kwargs={"weight_decay": 0.0})  # Heavy regularization
        latent_key = "X_scVI"
        orig_adata.obsm[latent_key] = query_model.get_latent_representation()
        # TODO: scarches weighted knn model to transfer from reference to query
    elif model_type == "scanvi":
        sca.models.SCANVI.prepare_query_anndata(adata, in_model)
        query_model = sca.models.SCANVI.load_query_data(adata, in_model, freeze_dropout=True)
        query_model._unlabeled_indices = np.arange(adata.n_obs)
        query_model._labeled_indices = []
        query_model.train(max_epochs=max_epochs, plan_kwargs={"weight_decay": 0.0}, check_val_every_n_epoch=10)  # Heavy regularization
        latent_key = "X_scANVI"
        orig_adata.obsm[latent_key] = query_model.get_latent_representation()
        # soft=True forces the network to return a DataFrame of softmax class probabilities
        predictions = query_model.predict(soft=True)
        orig_adata.obs["predicted_cell_type"] = predictions.idxmax(axis=1)
        orig_adata.obs["prediction_probability"] = predictions.max(axis=1)
    elif model_type == "scpoli":
        # labeled_indices=[] signifies the entire query is unannotated
        query_model = sca.model.scPoli.load_query_data(adata, in_model, labeled_indices=[])
        query_model.train(max_epochs=max_epochs, pretraining_epochs=max_epochs - max_epochs//5, eta=10)
        results = query_model.classify(
            adata,
            scale_uncertainties=True
        )
        preds = results[celltype_obs]["preds"]
        uncert = results[celltype_obs]["uncert"]
        orig_adata.obs["predicted_cell_type"] = preds
        orig_adata.obs["prediction_uncertainty"] = uncert
        orig_adata.obsm["X_scPoli"] = query_model.get_latent(adata, mean=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    orig_adata.write_h5ad(output_h5ad)


def main():
    parser = argparse.ArgumentParser(description="Integrate an array of h5ad files using scArches.")
    parser.add_argument("--input_h5ad", required=True)
    parser.add_argument("--output_h5ad", required=True)
    parser.add_argument("--in_model", required=True)
    parser.add_argument("--model_type", default="scvi", choices=["scvi", "scanvi", "scpoli"])
    parser.add_argument("--celltype_obs", required=True)
    parser.add_argument("--max_epochs", type=int, default=200)
    args = parser.parse_args()

    integrate(
        sc.read_h5ad(args.input_h5ad),
        args.output_h5ad,
        args.in_model,
        args.model_type,
        args.celltype_obs,
        args.max_epochs
    )


if __name__ == "__main__":
    main()
