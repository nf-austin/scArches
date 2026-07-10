import argparse

import anndata as ad
import scanpy as sc
import scarches as sca


def train(
    adata: ad.AnnData,
    out_model: str,
    model_type: str,
    dataset_obs: str,
    celltype_obs: str,
    n_hvgs: int,
    max_epochs: int,
    finetune_epochs: int = 20
):
    # Ensure raw counts are stored in .layers["counts"] as required by scvi-tools
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    has_batch_obs = dataset_obs != ""
    if not has_batch_obs:
        adata.obs["batch"] = "batch_0"
        dataset_obs = "batch"

    # Call HVGs
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvgs,
        batch_key=dataset_obs,
        flavor="seurat_v3",
        layer="counts",
        subset=True
    )

    if model_type == "scvi":
        sca.models.SCVI.setup_anndata(
            adata,
            batch_key=dataset_obs,
            labels_key=celltype_obs,
            layer="counts",
        )
        model = sca.models.SCVI(
            adata,
            n_layers=2,
            dropout_rate=0.2,
            encode_covariates=True,
            deeply_inject_covariates=False,
            use_layer_norm="both",
            use_batch_norm="none",
        )
        model.train(max_epochs=max_epochs, early_stopping=True)
    elif model_type == "scanvi":
        sca.models.SCVI.setup_anndata(
            adata,
            batch_key=dataset_obs,
            labels_key=celltype_obs,
            layer="counts",
        )
        model = sca.models.SCVI(
            adata,
            n_layers=2,
            encode_covariates=True,
            deeply_inject_covariates=False,
            use_layer_norm="both",
            use_batch_norm="none",
        )
        model.train(max_epochs=max_epochs, early_stopping=True)
        model = sca.models.SCANVI.from_scvi_model(
            model,
            unlabeled_category="Unknown",
        )
        print("Labelled Indices: ", len(model._labeled_indices))
        print("Unlabelled Indices: ", len(model._unlabeled_indices))
        model.train(max_epochs=finetune_epochs)
    elif model_type == "scpoli":
        model = sca.models.scPoli(
            adata,
            condition_keys=dataset_obs,
            cell_type_keys=celltype_obs,
            embedding_dims=5,
            recon_loss="nb"
        )
        model.train(
            n_epochs=max_epochs,  # Should be 100 for scPoli
            pretraining_epochs=max_epochs - max_epochs // 5,
            eta=5,
            early_stopping_kwargs={
                "early_stopping_metric": "val_prototype_loss",
                "mode": "min",
                "threshold": 0,
                "patience": 20,
                "reduce_lr": True,
                "lr_patience": 13,
                "lr_factor": 0.1,
            }
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.save(out_model, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")
    parser.add_argument("--train_h5ad", required=True)
    parser.add_argument("--out_model", required=True)
    parser.add_argument("--model_type", default="scvi", choices=["scvi", "scanvi", "scpoli"])
    parser.add_argument("--dataset_obs", default="")
    parser.add_argument("--celltype_obs", required=True)
    parser.add_argument("--n_hvgs", type=int, default=3000)
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    args = parser.parse_args()

    train(
        sc.read_h5ad(args.train_h5ad),
        args.out_model,
        args.model_type,
        args.dataset_obs,
        args.celltype_obs,
        args.n_hvgs,
        args.max_epochs,
        args.finetune_epochs
    )


if __name__ == "__main__":
    main()
