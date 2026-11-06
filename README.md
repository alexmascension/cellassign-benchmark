# cellassign-benchmark

Benchmark of [`cellassign`](https://github.com/alexmascension/cell_asign) against its previous
version and other annotators, on two well-annotated datasets and at two annotation scales.

| Dataset | Coarse scale (all cells) | Fine scale (compartment re-processed from counts) | Ground truth |
|---|---|---|---|
| PBMC CITE-seq, Hao et al. 2021, *Cell* (day 0, 8 donors) | 8 lineages | T cells (9 subsets), myeloid (6 subsets) | Azimuth `celltype.l1/l2` (RNA + protein) |
| Human cortex, Siletti et al. 2023, *Science* (MFG / A46 dissection, 3 donors) | 7 classes | Interneurons (8 subclasses) | superclusters; Allen MTG subclass transferred by the authors |

Methods: `cellassign` 1.0.1 (vendored in `bench/cellassign_v1.py`), `cellassign` 2.0 with positive
markers only and with positive + negative markers, `scanpy.tl.score_genes` + cluster arg-max,
CellTypist with majority voting, and a cluster-majority oracle (the ceiling for one-label-per-cluster
methods). All cluster-level methods use the same Leiden clusters.

## Layout

```
bench/
  data.py          download, loading, ground truth, preprocessing
  markers.py       marker dictionaries and label mappings for every scale
  methods.py       wrappers returning per-cell labels in the evaluation vocabulary
  metrics.py       accuracy, macro-F1, per-class F1, oracle, confidence AUROC
  calibration.py   v2 ablation, arg-max rules, novel-type (dropped category) experiment
  plots.py         figures
  cellassign_v1.py vendored baseline
notebooks/benchmark.ipynb
results/           CSV tables and figures written by the notebook
```

## Running

```bash
pip install -e .                      # installs cellassign>=2.0 and the benchmark stack
cd notebooks
jupyter nbconvert --to notebook --execute --inplace benchmark.ipynb
```

The notebook downloads about 1.4 GB of data into `../data` (override with `BENCH_DATA`) and four
CellTypist models. It runs in about 25 minutes on 8 CPU cores and needs about 12 GB of RAM.

## Data sources

- Hao et al. 2021: `pbmc_seurat_v4.h5ad`, figshare file 27458840.
- Siletti et al. 2023: CELLxGENE dataset `db1d2c1b-2ee1-45d5-9d53-69a327cd77e6`
  (Human Brain Cell Atlas v1.0, dissection "Middle frontal gyrus (MFG) - A46") and
  `tables/cluster_annotation.xlsx` from `linnarsson-lab/adult-human-brain`.

## Caveats

- Ground truth is computational in both datasets. The CellTypist `Adult_Human_MTG` model is trained
  on the same Allen taxonomy used to label the Siletti interneurons, which favours it.
- Fine scales are subset with the ground truth, so first-level errors do not propagate.
- Marker dictionaries are literature-based starting points fixed before running; they were not tuned.
- One clustering resolution (2.0), one seed, no batch correction beyond batch-aware HVG selection.
