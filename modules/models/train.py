import argparse
import math
import os
import pickle

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


def _dataset_detection(adata: ad.AnnData, dataset_obs: str, chunk_size: int = 100_000) -> pd.DataFrame:
    """genes x datasets matrix of "cells with > 0 counts", accumulated in row chunks.

    Chunked because `adata.layers["counts"] > 0` over a multi-million-cell reference would
    materialise a second copy of the whole matrix; this bounds peak memory at chunk_size rows.
    """
    codes = adata.obs[dataset_obs].astype("category")
    levels = list(codes.cat.categories)
    code_values = codes.cat.codes.to_numpy()
    counts = np.zeros((adata.n_vars, len(levels)), dtype=np.int64)

    for start in range(0, adata.n_obs, chunk_size):
        stop = min(start + chunk_size, adata.n_obs)
        detected = adata.layers["counts"][start:stop] > 0
        block_codes = code_values[start:stop]
        for j in range(len(levels)):
            mask = block_codes == j
            if mask.any():
                counts[:, j] += np.asarray(detected[mask].sum(axis=0)).ravel()

    return pd.DataFrame(counts, index=adata.var_names, columns=levels)


def _drop_dataset_absent_genes(adata: ad.AnnData, dataset_obs: str, min_fraction: float):
    """Drop genes detected in fewer than `min_fraction` of the dataset levels, in place.

    A unified reference assembled from several sub-atlases by an outer join keeps genes present
    in only some sources and zero-fills them in the rest. Those structural zeros are not biology:
    the model learns them as near-perfect sub-atlas discriminators, so a query with a complete
    panel carries real signal in them and maps to whichever sub-atlas "has" them rather than to
    its own tissue. In a lung+brain reference this surfaces as lung myeloid cells labelled
    Microglia, because MRC1/FCN1 are nonzero only on the brain side.

    Genuinely tissue-restricted genes survive this filter in practice: ambient RNA puts them at
    some low level in every dataset, so SFTPC is detected atlas-wide even though only lung has
    AT2 cells. Only never-measured genes are *exactly* zero.

    Gene-symbol synonyms split across annotation releases (MARCH8 vs MARCHF8) are dropped rather
    than merged - each half is absent from the datasets that used the other name.
    """
    detection = _dataset_detection(adata, dataset_obs)
    n_datasets = detection.shape[1]
    absent = detection == 0
    n_present = (~absent).sum(axis=1)
    # Expressing the requirement as a fraction makes it self-scaling: it can never exceed the
    # number of levels that exist, so a single-dataset reference simply requires 1. The min() only
    # guards against a value above 1.0 being passed.
    required = max(0, min(int(np.ceil(min_fraction * n_datasets)), n_datasets))
    keep = n_present >= required

    print(f"[panel] {n_datasets} datasets, {adata.n_vars} genes")
    print(
        "[panel]   genes by number of datasets absent from: "
        + ", ".join(f"{k}:{v}" for k, v in absent.sum(axis=1).value_counts().sort_index().items())
    )
    per_dataset = absent.sum(axis=0)
    print(
        "[panel]   genes absent per dataset: "
        + ", ".join(f"{d}={int(n)}" for d, n in per_dataset.sort_values(ascending=False).items())
    )
    if not keep.all():
        dropped = list(keep.index[~keep])
        print(
            f"[panel] dropping {len(dropped)} genes detected in fewer than {required}/"
            f"{n_datasets} datasets; {int(keep.sum())} retained"
        )
        print(f"[panel]   first dropped: {', '.join(map(str, dropped[:20]))}")
        adata._inplace_subset_var(keep.to_numpy())
    else:
        print(f"[panel] every gene is detected in at least {required}/{n_datasets} datasets")


def _balanced_label_indices(labels: pd.Series, max_per_label: int, seed: int = 0) -> np.ndarray:
    """Positional indices that cap each label at `max_per_label` cells (<=0 keeps everything)."""
    if max_per_label <= 0:
        return np.arange(len(labels))
    values = labels.to_numpy()
    rng = np.random.default_rng(seed)
    kept = []
    for label in pd.unique(values):
        positions = np.flatnonzero(values == label)
        if len(positions) > max_per_label:
            positions = rng.choice(positions, size=max_per_label, replace=False)
        kept.append(positions)
    return np.sort(np.concatenate(kept))


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
    knn_neighbors: int = 50,
    n_samples_per_label: int = 0,
    max_cells_per_label: int = 0,
    min_dataset_detection: float = 1.0,
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

    # Runs before HVG selection so the panel artifacts can't be picked as HVGs - a gene that is
    # zero across half the datasets and expressed across the other half looks enormously variable.
    # With no --dataset_obs there is nothing to compare across, so the fraction collapses to 0
    # (off); a single-level column resolves to a requirement of 1, which is a no-op beyond
    # dropping genes that are zero everywhere.
    detection_fraction = min_dataset_detection if has_batch_obs else 0.0
    if detection_fraction > 0:
        _drop_dataset_absent_genes(adata, dataset_obs, detection_fraction)

    # Call HVGs
    if adata.shape[1] > n_hvgs:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvgs,
            batch_key=dataset_obs,
            flavor="seurat_v3",
            layer="counts",
            subset=True
        )
    else:
        # Worth saying out loud: a reference that arrives pre-subset to <= n_hvgs genes skips this
        # step entirely, so --n_hvgs silently has no effect and the model trains on the full panel.
        print(
            f"Skipping HVG selection: the reference has {adata.shape[1]} genes, "
            f"already at or below --n_hvgs ({n_hvgs}). Training on all of them."
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
        # Without n_samples_per_label the classification head sees the reference's raw label
        # distribution every epoch, so on a reference spanning three orders of magnitude between
        # its largest and smallest label it spends its capacity separating the abundant classes.
        # Passing it resamples a fixed number of cells per label instead. None = scvi-tools'
        # unbalanced default.
        model.train(
            max_epochs=finetune_epochs,
            batch_size=batch_size,
            accelerator=accelerator,
            plan_kwargs={"lr": learning_rate},
            n_samples_per_label=n_samples_per_label if n_samples_per_label > 0 else None,
        )
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

    # Fit a weighted-KNN classifier on the reference latent space and persist it inside the
    # model directory (so COMPRESS/DECOMPRESS carry it with the model). integrate.py then
    # transfers labels without ever loading the (potentially huge) reference dataset.
    # weighted_knn_transfer indexes ref_adata_obs positionally, so the saved labels must stay
    # in the same row order as the latents the trainer is fit on — both come from `adata`.
    latent_key = {"scvi": "X_scVI", "scanvi": "X_scANVI", "scpoli": "X_scPoli"}[model_type]
    if model_type == "scpoli":
        adata.obsm[latent_key] = model.get_latent(adata, mean=True)
    else:
        adata.obsm[latent_key] = model.get_latent_representation()
    # weighted_knn_transfer scores a query cell by summing its neighbours' distance weights per
    # label, so each label's vote is proportional to its abundance among those neighbours - an
    # unbalanced reference systematically pulls boundary cells onto its largest classes. Capping
    # cells per label equalises that prior; it also bounds weighted_knn_trainer's index, which is
    # a brute-force KNeighborsTransformer and therefore scans every reference cell per query.
    # Both the index and the label CSV must come from the same subset in the same order, since
    # weighted_knn_transfer indexes ref_adata_obs positionally against the index's rows.
    knn_idx = _balanced_label_indices(adata.obs[celltype_obs], max_cells_per_label)
    knn_adata = adata[knn_idx]
    if len(knn_idx) < adata.n_obs:
        print(
            f"Capping the KNN reference at {max_cells_per_label} cells per label: "
            f"{len(knn_idx)}/{adata.n_obs} cells retained."
        )
    knn_model = sca.utils.knn.weighted_knn_trainer(
        train_adata=knn_adata,
        train_adata_emb=latent_key,
        n_neighbors=knn_neighbors,
    )
    with open(os.path.join(out_model, "knn_classifier.pkl"), "wb") as f:
        pickle.dump(knn_model, f)
    knn_adata.obs[[celltype_obs]].to_csv(os.path.join(out_model, "knn_ref_labels.csv"), index=False)


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
    parser.add_argument("--knn_neighbors", type=int, default=50, help="Number of neighbors for the weighted-KNN label-transfer classifier fit on the reference latent space.")
    parser.add_argument("--n_samples_per_label", type=int, default=0, help="SCANVI only: cells sampled per label per epoch during the fine-tuning stage, so an unbalanced reference doesn't dominate the classification head. 0 uses scvi-tools' unbalanced default.")
    parser.add_argument("--max_cells_per_label", type=int, default=0, help="Cap on reference cells per label when fitting the weighted-KNN classifier, which equalizes its abundance-driven label prior (and bounds its brute-force neighbor index). 0 uses every cell.")
    parser.add_argument("--min_dataset_detection", type=float, default=1.0, help="Fraction of the --dataset_obs levels a gene must be detected in to be kept (1.0 = all of them). Removes genes that an outer-join reference merge left structurally zero in whole sub-atlases, which the model would otherwise learn as sub-atlas discriminators. 0 disables; no --dataset_obs also disables.")
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
        args.knn_neighbors,
        args.n_samples_per_label,
        args.max_cells_per_label,
        args.min_dataset_detection,
        args.use_gpu
    )


if __name__ == "__main__":
    main()
