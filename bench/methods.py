"""Annotation methods compared in the benchmark.

Every method returns a ``pandas.Series`` of predicted labels per cell, already
mapped to the evaluation vocabulary of the scale. Cells a method leaves
unassigned get ``"unassigned"``; CellTypist labels without an equivalent get
``"other"``. Both count as errors.

All cluster-based methods use the same Leiden clusters (``adata.obs['leiden']``).
"""

from __future__ import annotations

import time
import warnings

import cellassign
import pandas as pd
import scanpy as sc

from . import cellassign_v1

UNASSIGNED = "unassigned"
OTHER = "other"


def _to_eval(labels: pd.Series, to_eval: dict) -> pd.Series:
    return labels.astype(str).map(lambda x: to_eval.get(x, x))


def run_cellassign_v1(adata, markers, to_eval, *, groupby="leiden", **kwargs):
    """cellassign 1.0.1 with its default parameters (min_score=0.6)."""
    work = adata.copy()
    cellassign_v1.assign_cats(
        work, markers, column_groupby=groupby, key_added="v1", verbose=False, **kwargs
    )
    return _to_eval(work.obs["v1"], to_eval), None


def run_cellassign_v2(adata, markers, to_eval, *, groupby="leiden", **kwargs):
    """cellassign 2.0 (this branch) with default parameters unless overridden.

    Also returns the cluster summary (label, confidence ``score``, raw ``best``
    score, ``margin``, ``null_pvalue``, ``loo_stability``).
    """
    work = adata.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cellassign.assign_cats(
            work, markers, column_groupby=groupby, key_added="v2", verbose=False, **kwargs
        )
    meta = work.uns["cell_assign"]["v2"]
    summary = meta["cluster_summary"].copy()
    summary["label_eval"] = _to_eval(summary["label"], to_eval)
    scores = meta["cluster_scores"]
    # Best category regardless of min_score (alphabetical tie rule, as in cellassign).
    summary["best_label_eval"] = _to_eval(scores[sorted(scores.columns)].idxmax(axis=1), to_eval)
    summary["n_cells"] = work.obs[groupby].astype(str).value_counts().reindex(summary.index)
    return _to_eval(work.obs["v2"], to_eval), summary


def run_score_genes(adata, markers, to_eval, *, groupby="leiden", random_state=0):
    """Baseline: ``sc.tl.score_genes`` per category, cluster mean, arg-max (no threshold)."""
    work = adata.copy()
    columns = []
    for i, genes in enumerate(markers.values()):
        genes = [g for g in genes if g in work.var_names]
        name = f"_sg_{i}"
        sc.tl.score_genes(work, genes, score_name=name, random_state=random_state)
        columns.append(name)
    per_cluster = work.obs[[groupby] + columns].groupby(groupby, observed=True).mean()
    per_cluster.columns = list(markers)
    best = per_cluster.idxmax(axis=1).to_dict()
    labels = work.obs[groupby].map(best).astype(str)
    return _to_eval(labels, to_eval), None


def run_celltypist(adata, model_path, label_map, *, groupby="leiden"):
    """CellTypist with majority voting over the same Leiden clusters."""
    import celltypist

    pred = celltypist.annotate(
        adata, model=str(model_path), majority_voting=True, over_clustering=groupby
    )
    raw = pred.predicted_labels["majority_voting"].astype(str)
    mapped = raw.map(lambda x: label_map(x) or OTHER)
    return mapped.reindex(adata.obs_names), None


def run_all(adata, scale: dict, model_dir, *, groupby="leiden"):
    """Run every method on one scale.

    Returns ``(predictions, v2_summaries, runtimes)``; ``v2_summaries`` maps
    ``"cellassign v2"`` and ``"cellassign v2 + negatives"`` to their cluster summaries.
    """
    markers, to_eval = scale["markers"], scale["to_eval"]
    markers_neg = scale.get("markers_neg", markers)
    markers_curated = scale.get("markers_neg_curated", markers_neg)
    jobs = {
        "cellassign v1": lambda: run_cellassign_v1(adata, markers, to_eval, groupby=groupby),
        "cellassign v2": lambda: run_cellassign_v2(adata, markers, to_eval, groupby=groupby),
        "cellassign v2 + negatives": lambda: run_cellassign_v2(
            adata, markers_neg, to_eval, groupby=groupby
        ),
        "cellassign v2 + negatives (curated)": lambda: run_cellassign_v2(
            adata, markers_curated, to_eval, groupby=groupby
        ),
        "score_genes": lambda: run_score_genes(adata, markers, to_eval, groupby=groupby),
    }
    for model, label_map in scale["celltypist"].items():
        path = f"{model_dir}/{model}.pkl"
        jobs[f"CellTypist {model}"] = lambda path=path, label_map=label_map: run_celltypist(
            adata, path, label_map, groupby=groupby
        )

    predictions, runtimes, summaries = {}, {}, {}
    for name, job in jobs.items():
        start = time.perf_counter()
        labels, extra = job()
        runtimes[name] = time.perf_counter() - start
        predictions[name] = labels
        if extra is not None:
            summaries[name] = extra
    return pd.DataFrame(predictions, index=adata.obs_names), summaries, pd.Series(runtimes)
