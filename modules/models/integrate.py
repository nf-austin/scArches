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
    # scPoliTrainer.get_latent_train splits the training data via
    # np.array_split(indices, self.batch_size), treating batch_size as a *section count*
    # rather than a chunk size (every other DataLoader in that trainer uses batch_size
    # correctly as the chunk size). When there are fewer cells than batch_size, most
    # sections come back empty, and indexing the dataset with an empty index array
    # collapses a tensor from 2-D to 1-D, crashing torch.cat in the encoder. Patch in a
    # version that derives the section count from the chunk size instead.
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


def _reference_var_names(adata: ad.AnnData, in_model: str, model_type: str) -> pd.Index:
    """The gene panel the reference model was trained on, read without altering `adata`."""
    if model_type == "scpoli":
        # scPoli isn't an scvi-tools ArchesMixin so it has no prepare_query_anndata() to ask.
        # scArches' BaseMixin.save() writes the reference panel next to the weights as a plain
        # newline-delimited list, which is what scPoli.load_query_data itself reads back.
        var_names = np.genfromtxt(
            os.path.join(in_model, "var_names.csv"), delimiter=",", dtype=str
        )
        return pd.Index(np.atleast_1d(var_names).astype(str))
    # return_reference_var_names=True returns before any of the padding/reordering work, so
    # passing our real adata here leaves it untouched.
    model_cls = sca.models.SCANVI if model_type == "scanvi" else sca.models.SCVI
    return pd.Index(
        model_cls.prepare_query_anndata(adata, in_model, return_reference_var_names=True)
    ).astype(str)


def _check_gene_overlap(adata: ad.AnnData, ref_var_names: pd.Index, min_overlap: float):
    """Report (and optionally reject) how much of the reference panel the query is missing.

    Both prepare_query_anndata() and scPoli.load_query_data() silently zero-fill any reference
    gene absent from the query. That padding is not neutral: it hits low-UMI cell types (T/NK
    cells especially) hardest, because once their few informative genes are zeroed they have
    almost nothing left to place them and they collapse onto whichever dense reference
    neighbourhood survives. Surfacing the fraction turns a silent distortion into a hard signal.
    """
    ref = pd.Index(ref_var_names.astype(str)).unique()
    query_vars = pd.Index(adata.var_names.astype(str))
    present = ref.isin(query_vars)
    n_shared, n_ref = int(present.sum()), len(ref)
    frac = n_shared / n_ref if n_ref else 0.0

    print(
        f"[gene-overlap] {n_shared}/{n_ref} ({frac:.1%}) reference genes present in the query; "
        f"{n_ref - n_shared} will be zero-filled."
    )
    if n_shared < n_ref:
        missing = ref[~present]
        print(f"[gene-overlap] first missing: {', '.join(missing[:10])}")
        # A large miss is far more often an identifier-namespace mismatch than a genuinely
        # different panel, so name that likely culprit rather than just reporting the count.
        ref_ens = float(ref.str.startswith("ENSG").mean())
        query_ens = float(query_vars.str.startswith("ENSG").mean())
        if abs(ref_ens - query_ens) > 0.5:
            print(
                f"[gene-overlap] reference var_names are {ref_ens:.0%} Ensembl-style vs "
                f"{query_ens:.0%} in the query - the two panels are probably keyed on different "
                "identifiers (Ensembl gene IDs vs gene symbols). Re-key one side before mapping."
            )

    if frac < min_overlap:
        raise ValueError(
            f"Only {frac:.1%} of the {n_ref} reference genes are present in the query, below the "
            f"--min_gene_overlap threshold of {min_overlap:.1%}. Mapping through a mostly "
            "zero-filled panel yields a distorted latent space and confident-but-wrong label "
            "transfer. Fix the gene identifiers/panel, or pass --min_gene_overlap 0 to map anyway."
        )


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
    min_gene_overlap: float = 0.9,
    use_gpu: bool = False,
    use_knn: bool = False
):
    accelerator = "gpu" if use_gpu else "cpu"

    # SCVI has no native classifier, so it always transfers labels via the weighted-KNN
    # classifier that train.py fit on the reference latent space.
    use_knn = use_knn or model_type == "scvi"

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

    # Checked before the surgery calls below, since those are what pad the missing genes away.
    _check_gene_overlap(
        adata, _reference_var_names(adata, in_model, model_type), min_gene_overlap
    )

    if model_type == "scvi":
        sca.models.SCVI.prepare_query_anndata(adata, in_model)
        query_model = sca.models.SCVI.load_query_data(adata, in_model, freeze_dropout=True)
        query_model.train(max_epochs=max_epochs, plan_kwargs={"weight_decay": 0.0}, batch_size=batch_size, accelerator=accelerator)  # Heavy regularization
        latent_key = "X_scVI"
        orig_adata.obsm[latent_key] = query_model.get_latent_representation()
    elif model_type == "scanvi":
        sca.models.SCANVI.prepare_query_anndata(adata, in_model)
        query_model = sca.models.SCANVI.load_query_data(adata, in_model, freeze_dropout=True)
        query_model._unlabeled_indices = np.arange(adata.n_obs)
        query_model._labeled_indices = []
        query_model.train(max_epochs=max_epochs, plan_kwargs={"weight_decay": 0.0}, check_val_every_n_epoch=10, batch_size=batch_size, accelerator=accelerator)  # Heavy regularization
        latent_key = "X_scANVI"
        orig_adata.obsm[latent_key] = query_model.get_latent_representation()
        if not use_knn:
            # soft=True forces the network to return a DataFrame of softmax class probabilities
            predictions = query_model.predict(soft=True)
            orig_adata.obs["predicted_cell_type"] = predictions.idxmax(axis=1)
            orig_adata.obs["prediction_probability"] = predictions.max(axis=1)
    elif model_type == "scpoli":
        # scPoli.load_query_data looks up adata.obs[celltype_obs][labeled_indices] before
        # applying labeled_indices, so the column must exist even though labeled_indices=[]
        # means the whole query is unannotated and no lookup values are actually used.
        if celltype_obs not in adata.obs:
            adata.obs[celltype_obs] = "Unknown"
        _patch_scpoli_get_latent_train()
        query_model = sca.models.scPoli.load_query_data(adata, in_model, labeled_indices=[])
        query_model.train(max_epochs=max_epochs, pretraining_epochs=max_epochs - max_epochs//5, eta=10, batch_size=batch_size, use_gpu=use_gpu)
        # load_query_data() gene-aligns (reorders/pads to the reference var_names) the adata it's
        # given and keeps that aligned copy as query_model.adata, but never mutates our adata in
        # place. classify()/get_latent() do no alignment of their own, so passing our still-raw,
        # full-gene-panel adata here shapes-mismatches against the model's HVG-subset input layer.
        # query_model.adata has the same cells in the same order (only columns are touched), so
        # it's safe to index positionally back onto orig_adata below.
        if not use_knn:
            results = query_model.classify(
                query_model.adata,
                scale_uncertainties=True
            )
            preds = results[celltype_obs]["preds"]
            uncert = results[celltype_obs]["uncert"]
            orig_adata.obs["predicted_cell_type"] = preds
            orig_adata.obs["prediction_uncertainty"] = uncert
        latent_key = "X_scPoli"
        orig_adata.obsm[latent_key] = query_model.get_latent(query_model.adata, mean=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if use_knn:
        # Transfer labels with the weighted-KNN classifier train.py fit on the reference
        # latent space (persisted inside the model directory), so the reference dataset
        # itself is never needed here. weighted_knn_transfer indexes ref_adata_obs
        # positionally, and knn_ref_labels.csv preserves the trainer's row order.
        knn_pkl = os.path.join(in_model, "knn_classifier.pkl")
        ref_labels_csv = os.path.join(in_model, "knn_ref_labels.csv")
        if not (os.path.exists(knn_pkl) and os.path.exists(ref_labels_csv)):
            raise FileNotFoundError(
                f"KNN classifier artifacts ({knn_pkl}, {ref_labels_csv}) not found in the model "
                "directory. The reference model predates KNN support - retrain it with the "
                "current train.py to use the weighted-KNN classifier."
            )
        with open(knn_pkl, "rb") as f:
            knn_model = pickle.load(f)
        ref_labels = pd.read_csv(ref_labels_csv)
        labels, uncert = sca.utils.knn.weighted_knn_transfer(
            query_adata=orig_adata,
            query_adata_emb=latent_key,
            ref_adata_obs=ref_labels,
            label_keys=celltype_obs,
            knn_model=knn_model,
        )
        orig_adata.obs["predicted_cell_type"] = labels[celltype_obs].values
        orig_adata.obs["prediction_uncertainty"] = uncert[celltype_obs].values.astype(float)

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
    parser.add_argument("--min_gene_overlap", type=float, default=0.9, help="Fail if fewer than this fraction of the reference model's genes are present in the query (the rest get silently zero-filled by the scArches surgery). 0 disables the check.")
    parser.add_argument("--use_gpu", action="store_true", help="Integrate on GPU instead of CPU.")
    parser.add_argument("--use_knn", action="store_true", help="Use a weighted knn classifier instead of directly querying the trained model. Always on for scvi, which has no native classifier.")
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
        args.min_gene_overlap,
        args.use_gpu,
        args.use_knn
    )


if __name__ == "__main__":
    main()
