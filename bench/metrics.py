"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from .methods import UNASSIGNED


def cluster_oracle(truth: pd.Series, clusters: pd.Series) -> pd.Series:
    """Majority truth label of each cluster mapped to its cells.

    Upper bound for any method that gives one label per cluster.
    """
    scored = truth.notna()
    majority = (
        truth[scored]
        .groupby(clusters[scored], observed=True)
        .agg(lambda x: x.value_counts().index[0])
    )
    return clusters.map(majority).astype(object)


def summarise(predictions: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:
    """Accuracy, macro-F1 and unassigned fraction for every method (scored cells only)."""
    scored = truth.notna()
    y = truth[scored].astype(str)
    classes = sorted(y.unique())
    rows = []
    for method in predictions.columns:
        pred = predictions.loc[scored, method].astype(str)
        assigned = pred != UNASSIGNED
        rows.append(
            {
                "method": method,
                "accuracy": float((pred == y).mean()),
                "macro_F1": f1_score(y, pred, labels=classes, average="macro", zero_division=0),
                "unassigned": float((~assigned).mean()),
                "accuracy_assigned": float((pred[assigned] == y[assigned]).mean())
                if assigned.any()
                else np.nan,
                "n_cells": int(scored.sum()),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def per_class_f1(predictions: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:
    scored = truth.notna()
    y = truth[scored].astype(str)
    classes = sorted(y.unique())
    table = {
        method: f1_score(
            y,
            predictions.loc[scored, method].astype(str),
            labels=classes,
            average=None,
            zero_division=0,
        )
        for method in predictions.columns
    }
    out = pd.DataFrame(table, index=classes).T
    out.loc["n_cells"] = y.value_counts().reindex(classes).to_numpy()
    return out


def confidence_auroc(summary: pd.DataFrame, truth: pd.Series, clusters: pd.Series):
    """Do v2 confidence metrics separate correct from wrong clusters?

    A cluster is correct when its assigned label equals the majority truth of the
    cluster. Unassigned clusters are excluded. Returns AUROC per metric and the
    per-cluster table.
    """
    majority = cluster_oracle(truth, clusters).groupby(clusters.astype(str)).first()
    table = summary.copy()
    table["majority_truth"] = majority.reindex(table.index)
    table = table[table["majority_truth"].notna() & (table["label"] != UNASSIGNED)]
    table["correct"] = table["label_eval"] == table["majority_truth"]
    auroc = {}
    for metric in ("score", "margin", "loo_stability"):
        values = table[metric]
        ok = values.notna()
        if table.loc[ok, "correct"].nunique() == 2:
            auroc[metric] = roc_auc_score(table.loc[ok, "correct"], values[ok])
        else:
            auroc[metric] = np.nan
    return pd.Series(auroc), table


def min_score_sweep(summary, truth, clusters, thresholds):
    """Cell-level accuracy and assigned fraction of v2 for several ``min_score``."""
    clusters = clusters.astype(str)
    scored = truth.notna()
    rows = []
    for threshold in thresholds:
        label = summary["best_label_eval"].where(summary["score"] >= threshold, UNASSIGNED)
        pred = clusters.map(label)
        rows.append(
            {
                "min_score": threshold,
                "accuracy": float((pred[scored] == truth[scored]).mean()),
                "assigned": float((pred[scored] != UNASSIGNED).mean()),
            }
        )
    return pd.DataFrame(rows)
