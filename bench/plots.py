"""Figures of the benchmark (matplotlib only)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOCAL = "#C2410C"  # cellassign v2
COLORS = {
    "cellassign v2 + negatives (curated)": "#F59E0B",
    "cellassign v2 + negatives": "#7F1D1D",
    "cellassign v2": FOCAL,
    "cellassign v1": "#1F77B4",
    "score_genes": "#7F7F7F",
    "CellTypist": "#2CA02C",
    "oracle (cluster majority)": "#000000",
}
MARKERS = {
    "cellassign v2 + negatives (curated)": "X",
    "cellassign v2 + negatives": "P",
    "cellassign v2": "o",
    "cellassign v1": "s",
    "score_genes": "D",
    "CellTypist": "^",
}

SCALE_TITLES = {
    "pbmc_all": "PBMC, all cells (8 lineages)",
    "pbmc_T": "PBMC, T cells re-clustered (9 subsets)",
    "pbmc_myeloid": "PBMC, myeloid re-clustered (6 subsets)",
    "brain_all": "Cortex MFG, all cells (7 classes)",
    "brain_inh": "Cortex MFG, interneurons re-clustered (8 subclasses)",
}


def style():
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _family(method: str) -> str:
    return "CellTypist" if method.startswith("CellTypist") else method


def accuracy_dotplot(summary: pd.DataFrame, metric: str = "accuracy"):
    """One panel per scale: methods on y, metric on x, oracle as a dashed line.

    ``summary``: long table with columns ``scale``, ``method`` and ``metric``.
    Values are printed in a column to the right of the axis.
    """
    style()
    scales = list(dict.fromkeys(summary["scale"]))
    fig, axes = plt.subplots(len(scales), 1, figsize=(5.0, 1.15 * len(scales) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, scale in zip(axes, scales, strict=True):
        sub = summary[summary["scale"] == scale].set_index("method")
        oracle = sub.loc["oracle (cluster majority)", metric]
        sub = sub.drop(index="oracle (cluster majority)")
        order = sorted(sub.index, key=lambda m: (not m.startswith("cellassign v2"), m))
        y = np.arange(len(order))[::-1]
        ax.axvline(oracle, color="black", ls="--", lw=0.8, zorder=0)
        for yi, method in zip(y, order, strict=True):
            fam = _family(method)
            value = sub.loc[method, metric]
            ax.plot([0, value], [yi, yi], color="#D9D9D9", lw=1, zorder=1)
            ax.scatter(
                value,
                yi,
                s=28 if fam.startswith("cellassign v2") else 18,
                color=COLORS[fam],
                marker=MARKERS[fam],
                zorder=3,
                clip_on=False,
            )
            ax.text(
                1.03,
                yi,
                f"{value:.2f}",
                va="center",
                fontsize=6,
                color=COLORS[fam],
                transform=ax.get_yaxis_transform(),
            )
        ax.text(
            1.03,
            len(order) - 0.4,
            f"oracle {oracle:.2f}",
            va="center",
            fontsize=6,
            transform=ax.get_yaxis_transform(),
        )
        ax.set_yticks(y, order)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.7, len(order) - 0.1)
        ax.set_title(SCALE_TITLES.get(scale, scale))
    axes[-1].set_xlabel(
        f"{metric.replace('_', ' ')} (cells; unassigned = error); dashed line = cluster oracle"
    )
    fig.tight_layout(h_pad=0.6)
    return fig


def f1_heatmap(per_class: pd.DataFrame, title: str):
    """Methods x classes F1 heatmap with values printed in every cell."""
    style()
    table = per_class.drop(index="n_cells")
    n = per_class.loc["n_cells"].astype(int)
    fig, ax = plt.subplots(figsize=(0.62 * table.shape[1] + 2.6, 0.32 * table.shape[0] + 1.1))
    im = ax.imshow(table.to_numpy(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if v < 0.6 else "black",
            )
    ax.set_xticks(
        range(table.shape[1]), [f"{c}\n(n={n[c]})" for c in table.columns], rotation=45, ha="right"
    )
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="F1 per class")
    fig.tight_layout()
    return fig


def novelty_tradeoff(novel: pd.DataFrame, defaults: dict):
    """Per scale: unassigned dropped type vs accuracy on other cells, one line per rule.

    Points are thresholds (mean over dropped classes); open circles mark the defaults.
    """
    style()
    mean = novel.groupby(["scale", "rule", "min_score"])[
        ["novel_unassigned", "known_accuracy"]
    ].mean()
    scales = list(dict.fromkeys(novel["scale"]))
    rule_style = {
        "null-normalised": (FOCAL, "-", "null-normalised confidence (v2 default)"),
        "raw": ("#4C72B0", "--", "raw score (null_correction=False)"),
    }
    fig, axes = plt.subplots(
        1, len(scales), figsize=(1.75 * len(scales) + 0.4, 2.5), sharex=True, sharey=True
    )
    for ax, scale in zip(np.atleast_1d(axes), scales, strict=True):
        for rule, (color, ls, label) in rule_style.items():
            if (scale, rule) not in mean.index.droplevel(2):
                continue
            d = mean.loc[(scale, rule)].reset_index()
            ax.plot(
                d["novel_unassigned"],
                d["known_accuracy"],
                ls=ls,
                marker="o",
                ms=2.5,
                lw=1,
                color=color,
                label=label,
            )
            at = d[np.isclose(d["min_score"], defaults[rule])]
            ax.scatter(
                at["novel_unassigned"],
                at["known_accuracy"],
                s=55,
                facecolor="none",
                edgecolor=color,
                lw=1.2,
                zorder=4,
            )
        ax.set_title(scale)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.4, 1.02)
    axes[0].set_ylabel("accuracy on the other cells")
    fig.supxlabel(
        "fraction of the dropped type left unassigned (both axes: higher = better)", fontsize=8
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.07)
    )
    fig.tight_layout()
    return fig


def confidence_strip(tables: dict[str, pd.DataFrame], metric: str):
    """Per-cluster ``metric`` for correctly and wrongly labelled clusters, per scale."""
    style()
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    rng = np.random.default_rng(0)
    for i, t in enumerate(tables.values()):
        for offset, correct, color in ((-0.17, True, "#4C72B0"), (0.17, False, FOCAL)):
            v = t.loc[t["correct"] == correct, metric].dropna()
            x = i + offset + rng.uniform(-0.06, 0.06, len(v))
            ax.scatter(x, v, s=8, color=color, alpha=0.8, lw=0)
            if len(v):
                x0, x1 = i + offset - 0.1, i + offset + 0.1
                ax.plot([x0, x1], [v.median()] * 2, color="black", lw=1)
    ax.set_xticks(range(len(tables)), list(tables), rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.scatter([], [], s=8, color="#4C72B0", label="cluster label correct")
    ax.scatter([], [], s=8, color=FOCAL, label="cluster label wrong")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"v2 {metric.replace('_', ' ')} per cluster (black tick = median)")
    fig.tight_layout()
    return fig
