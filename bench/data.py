"""Dataset download, loading, ground truth and preprocessing."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import requests
import scanpy as sc

URLS = {
    # Hao et al. 2021 (Cell), PBMC CITE-seq, Azimuth reference.
    "pbmc_seurat_v4.h5ad": "https://ndownloader.figshare.com/files/27458840",
    # Siletti et al. 2023 (Science), dissection "Middle frontal gyrus (MFG) - A46",
    # CELLxGENE dataset db1d2c1b-2ee1-45d5-9d53-69a327cd77e6.
    "siletti_MFG_A46.h5ad": (
        "https://datasets.cellxgene.cziscience.com/17e6555e-4a78-4196-81a3-d2698c37109c.h5ad"
    ),
    "siletti_cluster_annotation.xlsx": (
        "https://raw.githubusercontent.com/linnarsson-lab/adult-human-brain/main/"
        "tables/cluster_annotation.xlsx"
    ),
}

CELLTYPIST_REGISTRY = "https://celltypist.cog.sanger.ac.uk/models/models.json"


def download(data_dir: str | Path) -> None:
    """Download the input files that are not already in ``data_dir``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        path = data_dir / name
        if path.exists():
            continue
        tmp = path.with_suffix(path.suffix + ".part")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 << 20):
                    f.write(chunk)
        tmp.replace(path)


def download_celltypist_models(model_dir: str | Path, names) -> None:
    """Download CellTypist models (``<name>.pkl``) that are not already in ``model_dir``."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    registry = requests.get(CELLTYPIST_REGISTRY, timeout=60).json()["models"]
    urls = {m["filename"]: m["url"] for m in registry}
    for name in names:
        path = model_dir / f"{name}.pkl"
        if not path.exists():
            response = requests.get(urls[f"{name}.pkl"], timeout=300)
            response.raise_for_status()
            path.write_bytes(response.content)


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
_PBMC_T_L2 = {
    "CD4 Naive": "CD4 Naive/CM",
    "CD4 TCM": "CD4 Naive/CM",
    "CD4 TEM": "CD4 EM",
    "CD4 CTL": "CD4 CTL",
    "Treg": "Treg",
    "CD8 Naive": "CD8 Naive/CM",
    "CD8 TCM": "CD8 Naive/CM",
    "CD8 TEM": "CD8 EM",
    "MAIT": "MAIT",
    "gdT": "gdT",
    "CD4 Proliferating": "T Prolif",
    "CD8 Proliferating": "T Prolif",
    "dnT": None,  # no consistent marker set; kept in the data, not scored
}
_PBMC_MYELOID_L2 = {k: k for k in ["CD14 Mono", "CD16 Mono", "cDC1", "cDC2", "pDC", "ASDC"]}

_SUPERCLUSTER_CLASS = {
    "Upper-layer intratelencephalic": "Excitatory",
    "Deep-layer intratelencephalic": "Excitatory",
    "Deep-layer corticothalamic and 6b": "Excitatory",
    "Deep-layer near-projecting": "Excitatory",
    "Hippocampal CA1-3": "Excitatory",
    "MGE interneuron": "Inhibitory",
    "CGE interneuron": "Inhibitory",
    "LAMP5-LHX6 and Chandelier": "Inhibitory",
    "Eccentric medium spiny neuron": "Inhibitory",
    "Astrocyte": "Astrocyte",
    "Oligodendrocyte": "Oligodendrocyte",
    "Oligodendrocyte precursor": "OPC",
    "Committed oligodendrocyte precursor": "OPC",
    "Microglia": "Microglia",
    "Vascular": "Vascular",
    "Fibroblast": "Vascular",
    # "Miscellaneous" (immune cells and mixed neurons) and "Splatter" are not scored.
}
_INTERNEURON_SUPERCLUSTERS = ["MGE interneuron", "CGE interneuron", "LAMP5-LHX6 and Chandelier"]


def _pbmc_l1(row_l1: str, row_l2: str):
    if row_l2 == "Doublet":
        return None
    if row_l1 in ("CD4 T", "CD8 T", "other T"):
        return "T"
    if row_l2 == "ILC":
        return "NK"
    if row_l1 == "other":
        return row_l2  # Platelet, Eryth, HSPC
    return row_l1  # B, NK, Mono, DC


def _symbols_as_var_names(adata, column: str):
    adata.var["gene_id"] = adata.var_names.astype(str)
    adata.var_names = adata.var[column].astype(str).to_numpy()
    adata.var_names_make_unique()


def load_pbmc(data_dir, timepoint: int = 0) -> ad.AnnData:
    """Hao et al. 2021 PBMC, one time point (default: day 0, before vaccination)."""
    adata = ad.read_h5ad(Path(data_dir) / "pbmc_seurat_v4.h5ad", backed="r")
    keep = (adata.obs["time"].astype(int) == timepoint).to_numpy()
    adata = adata[keep].to_memory()
    del adata.obsm["protein_counts"]
    obs = adata.obs
    obs["truth_all"] = [
        _pbmc_l1(a, b) for a, b in zip(obs["celltype.l1"], obs["celltype.l2"], strict=True)
    ]
    obs["truth_T"] = obs["celltype.l2"].map(_PBMC_T_L2)
    obs["truth_myeloid"] = obs["celltype.l2"].map(_PBMC_MYELOID_L2)
    obs["batch"] = obs["donor"].astype(str)
    adata = adata[obs["celltype.l2"] != "Doublet"].copy()
    return adata


def load_siletti(data_dir) -> ad.AnnData:
    """Siletti et al. 2023, MFG (A46) dissection, with class and MTG subclass truth."""
    data_dir = Path(data_dir)
    adata = ad.read_h5ad(data_dir / "siletti_MFG_A46.h5ad")
    _symbols_as_var_names(adata, "feature_name")

    annotation = pd.read_excel(data_dir / "siletti_cluster_annotation.xlsx")
    annotation["cluster_id"] = annotation["Cluster ID"].astype("Int64").astype(str)
    mtg = annotation.set_index("cluster_id")["Transferred MTG Label"]

    obs = adata.obs
    obs["supercluster"] = obs["supercluster_term"].astype(str)
    obs["mtg_label"] = obs["cluster_id"].astype(str).map(mtg)
    obs["truth_all"] = obs["supercluster"].map(_SUPERCLUSTER_CLASS)
    is_interneuron = obs["supercluster"].isin(_INTERNEURON_SUPERCLUSTERS)
    obs["truth_inh"] = obs["mtg_label"].where(is_interneuron)
    obs["batch"] = obs["donor_id"].astype(str)
    return adata


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def preprocess(
    adata: ad.AnnData,
    *,
    n_hvg: int = 2000,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    resolution: float = 2.0,
    min_cells: int = 10,
    seed: int = 0,
) -> ad.AnnData:
    """Standard Scanpy workflow. ``adata.X`` must contain raw counts.

    Output: ``layers['counts']``; ``X`` = log1p(CP10k) for all genes (what the
    annotators use); PCA on scaled HVGs; kNN graph; Leiden clusters (``leiden``).
    """
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, batch_key="batch")
    hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(hvg, max_value=10)
    sc.pp.pca(hvg, n_comps=n_pcs, random_state=seed)
    adata.obsm["X_pca"] = hvg.obsm["X_pca"]
    del hvg

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca", random_state=seed)
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=seed,
    )
    return adata


def subset_and_reprocess(adata: ad.AnnData, mask, **kwargs) -> ad.AnnData:
    """Subset cells and redo the whole workflow from counts (HVG, PCA, graph, Leiden)."""
    sub = adata[np.asarray(mask)].copy()
    sub.X = sub.layers["counts"].copy()
    del sub.layers["counts"]
    sub.obsm.pop("X_pca", None)
    for key in list(sub.obsp.keys()):
        del sub.obsp[key]
    for key in ("neighbors", "leiden", "hvg", "log1p", "pca"):
        sub.uns.pop(key, None)
    return preprocess(sub, **kwargs)
