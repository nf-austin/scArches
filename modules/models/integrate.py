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
    dataset_obs: str,
    sample_id: str,
    max_epochs: int,
    batch_size: int = 128,
    use_gpu: bool = False
):
    accelerator = "gpu" if use_gpu else "cpu"

    orig_adata = adata.copy()

    # The reference model expects the same batch/dataset obs column train.py registered it with
    # (params.dataset_obs; "batch" when that was left blank, see train.py). A per-sample query file
    # generally won't already carry it, so treat the whole file as one new, unseen batch.
    batch_key = dataset_obs if dataset_obs else "batch"
    if batch_key not in adata.obs:
        adata.obs[batch_key] = sample_id

    # Ensure raw counts are stored in .layers["counts"] as required by scvi-tools
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    if model_type == "scvi":
        sca.models.SCVI.prepare_query_anndata(adata, in_model)
        query_model = sca.models.SCVI.load_query_data(adata, in_model, freeze_dropout=True)
        query_model.train(max_epochs=max_epochs, plan_kwargs={"weight_decay": 0.0}, batch_size=batch_size, accelerator=accelerator)  # Heavy regularization
        latent_key = "X_scVI"
        orig_adata.obsm[latent_key] = query_model.get_latent_representation()
        # TODO: scarches weighted knn model to transfer from reference to query
    elif model_type == "scanvi":
        sca.models.SCANVI.prepare_query_anndata(adata, in_model)
        query_model = sca.models.SCANVI.load_query_data(adata, in_model, freeze_dropout=True)
        query_model._unlabeled_indices = np.arange(adata.n_obs)
        query_model._labeled_indices = []
        query_model.train(max_epochs=max_epochs, plan_kwargs={"weight_decay": 0.0}, check_val_every_n_epoch=10, batch_size=batch_size, accelerator=accelerator)  # Heavy regularization
        latent_key = "X_scANVI"
        orig_adata.obsm[latent_key] = query_model.get_latent_representation()
        # soft=True forces the network to return a DataFrame of softmax class probabilities
        predictions = query_model.predict(soft=True)
        orig_adata.obs["predicted_cell_type"] = predictions.idxmax(axis=1)
        orig_adata.obs["prediction_probability"] = predictions.max(axis=1)
    elif model_type == "scpoli":
        # labeled_indices=[] signifies the entire query is unannotated
        query_model = sca.model.scPoli.load_query_data(adata, in_model, labeled_indices=[])
        query_model.train(max_epochs=max_epochs, pretraining_epochs=max_epochs - max_epochs//5, eta=10, batch_size=batch_size, use_gpu=use_gpu)
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
    parser.add_argument("--dataset_obs", default="", help="obs column identifying batches/datasets; must match the value train.py was run with.")
    parser.add_argument("--sample_id", required=True, help="Label used for this file's batch/dataset if --dataset_obs isn't already present in the input.")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128, help="Minibatch size for model.train().")
    parser.add_argument("--use_gpu", action="store_true", help="Integrate on GPU instead of CPU.")
    args = parser.parse_args()

    integrate(
        sc.read_h5ad(args.input_h5ad),
        args.output_h5ad,
        args.in_model,
        args.model_type,
        args.celltype_obs,
        args.dataset_obs,
        args.sample_id,
        args.max_epochs,
        args.batch_size,
        args.use_gpu
    )


if __name__ == "__main__":
    main()
