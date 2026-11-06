"""Score variants of cellassign v2 and the novel-type (dropped category) experiment.

Everything is computed from the per-cluster tables cellassign stores in
``adata.uns['cell_assign'][key]``:

- ``raw``: score used to choose the category (quantile of balanced marker
  percentiles, ``cluster_scores``),
- ``null`` / ``null_sd``: mean / SD of the same statistic over control gene sets,
- ``ceiling``: score a perfectly specific marker would reach in the cluster.
"""

from __future__ import annotations

import warnings

import cellassign
import numpy as np
import pandas as pd

from .methods import UNASSIGNED

#: Candidate rules to choose the category of each cluster.
ARGMAX = {
    "raw": lambda t: t["raw"],
    "raw - null": lambda t: t["raw"] - t["null"],
    "normalised excess": lambda t: (t["raw"] - t["null"]) / (t["ceiling"] - t["null"]),
    "z": lambda t: (t["raw"] - t["null"]) / t["null_sd"],
}


def cluster_tables(adata, markers, *, groupby="leiden", key="v2", **kwargs):
    """Run cellassign once (no leave-one-out) and return its per-cluster tables."""
    work = adata.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cellassign.assign_cats(
            work,
            markers,
            column_groupby=groupby,
            key_added=key,
            verbose=False,
            leave_one_out=False,
            **kwargs,
        )
    meta = work.uns["cell_assign"][key]
    raw = meta["cluster_scores"]
    ceiling = pd.Series(meta["group_ceiling"]).reindex(raw.index).to_numpy()[:, None]
    return {
        "raw": raw,
        "null": meta["null_scores"],
        "null_sd": meta["null_sd"],
        "ceiling": pd.DataFrame(
            np.repeat(ceiling, raw.shape[1], 1), index=raw.index, columns=raw.columns
        ),
        "confidence": meta["cluster_confidence"],
    }


def _best(scores: pd.DataFrame) -> pd.Series:
    return scores[sorted(scores.columns)].idxmax(axis=1)


def predict(
    tables, to_eval, clusters, *, argmax="raw", min_confidence=None, rule="null-normalised"
):
    """Cell-level predictions for an argmax rule and an optional acceptance cut.

    ``rule="null-normalised"`` thresholds the confidence (cellassign default);
    ``rule="raw"`` thresholds the raw score (cellassign with ``null_correction=False``).
    """
    best = _best(ARGMAX[argmax](tables))
    label = best.map(lambda x: to_eval.get(x, x))
    if min_confidence is not None:
        conf = tables["confidence" if rule == "null-normalised" else "raw"]
        chosen = pd.Series([conf.loc[i, c] for i, c in best.items()], index=best.index)
        label = label.where(chosen >= min_confidence, UNASSIGNED)
    return clusters.astype(str).map(label)


def argmax_accuracy(tables, to_eval, truth, clusters) -> pd.Series:
    """Accuracy of each argmax rule without any threshold."""
    scored = truth.notna()
    out = {}
    for name in ARGMAX:
        pred = predict(tables, to_eval, clusters, argmax=name)
        out[name] = float((pred[scored] == truth[scored]).mean())
    return pd.Series(out)


#: Thresholds compared in the calibration experiments, per acceptance rule.
RULES = {
    "null-normalised": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "raw": [0.0, 0.5, 0.6, 0.7, 0.8, 0.9],
}
DEFAULTS = {"null-normalised": 0.3, "raw": 0.6}


def novel_type_experiment(
    adata, scale, truth_col, rules=RULES, *, groupby="leiden", min_cells=50, **kwargs
):
    """Drop each evaluation class from the marker dictionary in turn.

    For every acceptance rule and threshold, report the fraction of the dropped
    class left unassigned (``novel_unassigned``, higher is better) and the
    accuracy on all other cells (``known_accuracy``).
    """
    markers, to_eval = scale["markers"], scale["to_eval"]
    truth = adata.obs[truth_col]
    clusters = adata.obs[groupby]
    counts = truth.value_counts()
    rows = []
    for dropped in counts.index[counts >= min_cells]:
        reduced = {c: g for c, g in markers.items() if to_eval.get(c, c) != dropped}
        if len(reduced) == len(markers) or len(reduced) < 2:
            continue
        tables = cluster_tables(adata, reduced, groupby=groupby, **kwargs)
        novel = truth == dropped
        known = truth.notna() & ~novel
        for rule, thresholds in rules.items():
            for threshold in thresholds:
                pred = predict(tables, to_eval, clusters, min_confidence=threshold, rule=rule)
                rows.append(
                    {
                        "dropped": dropped,
                        "rule": rule,
                        "min_score": threshold,
                        "novel_unassigned": float((pred[novel] == UNASSIGNED).mean()),
                        "known_accuracy": float((pred[known] == truth[known]).mean()),
                    }
                )
    return pd.DataFrame(rows)


#: Ubiquitously expressed genes used as a non-specific "category".
HOUSEKEEPING = ["ACTB", "GAPDH", "MALAT1", "TPT1", "EEF1A1", "B2M"]


def nonspecific_category(adata, rules=RULES, *, genes=HOUSEKEEPING, groupby="leiden"):
    """Fraction of cells labelled with a category made of housekeeping genes only.

    Any accepted label is a false positive, so lower is better.
    """
    genes = [g for g in genes if g in adata.var_names]
    tables = cluster_tables(adata, {"Housekeeping": genes}, groupby=groupby)
    clusters = adata.obs[groupby]
    rows = []
    for rule, thresholds in rules.items():
        for threshold in thresholds:
            pred = predict(tables, {}, clusters, min_confidence=threshold, rule=rule)
            rows.append(
                {
                    "rule": rule,
                    "min_score": threshold,
                    "false_positive_cells": float((pred != UNASSIGNED).mean()),
                }
            )
    return pd.DataFrame(rows)


#: v2 components switched off one at a time (and all together).
ABLATIONS = {
    "v2 default": {},
    "- null correction": {"null_correction": False},
    "- balanced percentiles": {"balance_groups": False},
    "- smoothing": {"smooth": False},
    "+ marker weights": {"weight_shared_markers": True},
    "v2 all off": {"null_correction": False, "balance_groups": False},
    "+ negative markers": {"_markers": "markers_neg"},
}


def ablation(adata, scale, truth_col, *, groupby="leiden", configs=ABLATIONS):
    """Accuracy of v2 variants: with the default threshold and choosing the best
    category without threshold (``argmax``)."""
    from .methods import run_cellassign_v2

    truth = adata.obs[truth_col]
    scored = truth.notna()
    clusters = adata.obs[groupby].astype(str)
    rows = []
    for name, config in configs.items():
        kwargs = dict(config)
        markers_key = kwargs.pop("_markers", "markers")
        labels, summary = run_cellassign_v2(
            adata,
            scale[markers_key],
            scale["to_eval"],
            groupby=groupby,
            leave_one_out=False,
            **kwargs,
        )
        argmax = clusters.map(summary["best_label_eval"])
        rows.append(
            {
                "config": name,
                "accuracy": float((labels[scored] == truth[scored]).mean()),
                "accuracy_argmax": float((argmax[scored] == truth[scored]).mean()),
                "unassigned": float((labels[scored] == UNASSIGNED).mean()),
            }
        )
    return pd.DataFrame(rows)


def negative_marker_diagnostic(adata, scale, truth_col, *, groupby="leiden"):
    """Accuracy change when the negative markers of one category are removed.

    ``delta = accuracy(all negatives) - accuracy(without this category's negatives)``:
    positive values mean that category's negative markers help.
    """
    from .methods import run_cellassign_v2

    truth = adata.obs[truth_col]
    scored = truth.notna()
    full = scale["markers_neg"]

    def accuracy(markers):
        labels, _ = run_cellassign_v2(
            adata, markers, scale["to_eval"], groupby=groupby, leave_one_out=False
        )
        return float((labels[scored] == truth[scored]).mean())

    reference = accuracy(full)
    rows = []
    for category, spec in full.items():
        if not isinstance(spec, dict):
            continue
        reduced = dict(full)
        reduced[category] = list(spec["pos"])
        rows.append(
            {
                "category": category,
                "negatives": ", ".join(spec["neg"]),
                "accuracy_all_negatives": reference,
                "accuracy_without": accuracy(reduced),
            }
        )
    out = pd.DataFrame(rows)
    out["delta"] = out["accuracy_all_negatives"] - out["accuracy_without"]
    return out
