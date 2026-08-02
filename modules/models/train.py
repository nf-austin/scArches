import argparse
import math

import anndata as ad
import pandas as pd
import scanpy as sc
import scarches as sca
import numpy as np
import torch


def _patch_scpoli_get_latent_train():
    # See modules/models/integrate.py's _patch_scpoli_get_latent_train for the full
    # explanation: scPoliTrainer.get_latent_train misuses batch_size as a section count
    # instead of a chunk size, which crashes when there are fewer cells than batch_size.
    from scarches.trainers.scpoli.trainer import scPoliTrainer

    def get_latent_train(self):
        latents = []
        indices = np.arange(len(self.train_data))
        n_chunks = max(1, math.ceil(len(indices) / self.batch_size))
        subsampled_indices = np.array_split(indices, n_chunks)
        for batch in subsampled_indices:
            batch_data = self.train_data[batch]
            latent = self.model.get_latent(
                batch_data["x"].to(self.device),
                batch_data["batch"].to(self.device),
            )
            latents += [latent.cpu().detach()]
        latent = torch.cat(latents)
        return latent.to(self.device)

    scPoliTrainer.get_latent_train = get_latent_train


def train(
    adata: ad.AnnData,
    out_model: str,
    model_type: str,
    dataset_obs: str,
    celltype_obs: str,
    n_hvgs: int,
    max_epochs: int,
    finetune_epochs: int = 20,
    batch_size: int = 128,
    n_layers: int = 3,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    use_gpu: bool = False
):
    accelerator = "gpu" if use_gpu else "cpu"

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
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            encode_covariates=True,
            deeply_inject_covariates=False,
            use_layer_norm="both",
            use_batch_norm="none",
        )
        model.train(max_epochs=max_epochs, early_stopping=True, batch_size=batch_size, accelerator=accelerator, plan_kwargs={"lr": learning_rate})
    elif model_type == "scanvi":
        sca.models.SCVI.setup_anndata(
            adata,
            batch_key=dataset_obs,
            labels_key=celltype_obs,
            layer="counts",
        )
        model = sca.models.SCVI(
            adata,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            encode_covariates=True,
            deeply_inject_covariates=False,
            use_layer_norm="both",
            use_batch_norm="none",
        )
        model.train(max_epochs=max_epochs, early_stopping=True, batch_size=batch_size, accelerator=accelerator, plan_kwargs={"lr": learning_rate})
        model = sca.models.SCANVI.from_scvi_model(
            model,
            unlabeled_category="Unknown",
        )
        print("Labelled Indices: ", len(model._labeled_indices))
        print("Unlabelled Indices: ", len(model._unlabeled_indices))
        model.train(max_epochs=finetune_epochs, batch_size=batch_size, accelerator=accelerator, plan_kwargs={"lr": learning_rate})
    elif model_type == "scpoli":
        # scPoli's constructor defaults labeled_indices to range(len(adata)) and looks up
        # cell types via adata.obs[celltype_obs][range(len(adata))]. AnnData always keeps
        # obs_names as strings, and pandas >=2.0 dropped the positional fallback for integer
        # keys against a non-integer index, so that lookup now raises KeyError for any real
        # dataset. Swapping in a real integer index (bypassing AnnData's obs_names setter, so
        # it isn't coerced back to strings) makes the lookup a genuine label match. This is
        # never persisted: model.save() below only writes adata when save_anndata=True.
        adata.obs.index = pd.RangeIndex(adata.n_obs)
        # scPoli has no n_layers/dropout_rate args (unlike SCVI/SCANVI above); it takes
        # hidden_layer_sizes (a list, one entry per layer, decoder mirrors it in reverse) and
        # dr_rate instead. When hidden_layer_sizes is left None, scPoli itself defaults to a
        # single layer sized ceil(sqrt(n_genes)) - reuse that width, repeated n_layers times,
        # so --n_layers has a comparable effect across all three model types.
        layer_width = int(np.ceil(np.sqrt(adata.n_vars)))
        model = sca.models.scPoli(
            adata,
            condition_keys=dataset_obs,
            cell_type_keys=celltype_obs,
            embedding_dims=5,
            recon_loss="nb",
            hidden_layer_sizes=[layer_width] * n_layers,
            dr_rate=dropout_rate,
        )
        _patch_scpoli_get_latent_train()
        model.train(
            n_epochs=max_epochs,  # Should be 100 for scPoli
            pretraining_epochs=max_epochs - max_epochs // 5,
            eta=5,
            lr=learning_rate,
            batch_size=batch_size,
            use_gpu=use_gpu,
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
    parser.add_argument("--n_hvgs", type=int, default=6000)
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128, help="Minibatch size for model.train().")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of hidden layers in the encoder/decoder (SCVI/SCANVI: n_layers; scPoli: hidden_layer_sizes length).")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for the encoder/decoder (SCVI/SCANVI: dropout_rate; scPoli: dr_rate).")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Optimizer learning rate (SCVI/SCANVI: plan_kwargs['lr']; scPoli: lr). Applied to all training stages, including SCANVI fine-tuning.")
    parser.add_argument("--use_gpu", action="store_true", help="Train on GPU instead of CPU.")
    args = parser.parse_args()

    train(
        sc.read_h5ad(args.train_h5ad),
        args.out_model,
        args.model_type,
        args.dataset_obs,
        args.celltype_obs,
        args.n_hvgs,
        args.max_epochs,
        args.finetune_epochs,
        args.batch_size,
        args.n_layers,
        args.dropout_rate,
        args.learning_rate,
        args.use_gpu
    )


if __name__ == "__main__":
    main()
