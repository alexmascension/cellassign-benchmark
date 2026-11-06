"""Marker dictionaries and label mappings for every benchmark scale.

Markers were fixed a priori from the literature before looking at the results
(PBMC: Azimuth / Hao et al. 2021 marker lists and standard immunology markers;
cortex: Hodge et al. 2019, Bakken et al. 2021, Siletti et al. 2023). They are a
starting point, not an optimised set. Entries marked ``# low confidence`` are
the weakest.

Each scale defines:

- ``markers``: categories used by the marker-based methods. Categories can be
  finer than the evaluation vocabulary (e.g. ``pDC`` and ``cDC`` for ``DC``).
- ``to_eval``: maps marker categories to the evaluation vocabulary.
- ``celltypist``: CellTypist model(s) and a function mapping model labels to
  the evaluation vocabulary (``None`` = no equivalent, counted as an error).
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# PBMC (Hao et al. 2021, CITE-seq; ground truth = Azimuth celltype.l1/l2)
# --------------------------------------------------------------------------- #
PBMC_ALL = {
    "T": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC"],
    "NK": ["KLRF1", "NCAM1", "NCR1", "GNLY", "NKG7", "KLRD1"],
    "B": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "PAX5"],
    "Plasmablast": ["MZB1", "JCHAIN", "TNFRSF17"],
    "CD14 Mono": ["CD14", "LYZ", "S100A8", "S100A9", "VCAN", "FCN1"],
    "CD16 Mono": ["FCGR3A", "MS4A7", "CDKN1C", "CX3CR1"],
    "cDC": ["FCER1A", "CD1C", "CLEC10A", "CLEC9A"],
    "pDC": ["LILRA4", "CLEC4C", "IL3RA", "PLD4"],
    "Platelet": ["PPBP", "PF4", "GP9", "ITGA2B"],
    "Eryth": ["HBB", "HBA1", "HBA2", "ALAS2"],
    "HSPC": ["CD34", "PRSS57", "SPINK2", "SOX4"],
}
PBMC_ALL_TO_EVAL = {
    "Plasmablast": "B",
    "CD14 Mono": "Mono",
    "CD16 Mono": "Mono",
    "cDC": "DC",
    "pDC": "DC",
}

PBMC_T = {
    "CD4 Naive/CM": ["CD4", "CCR7", "SELL", "LEF1", "TCF7"],
    "CD4 EM": ["CD4", "IL7R", "GZMK", "AQP3", "KLRB1"],
    "CD4 CTL": ["CD4", "GZMH", "FGFBP2", "GNLY", "GZMB"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "RTKN2", "TIGIT"],
    "CD8 Naive/CM": ["CD8A", "CD8B", "CCR7", "SELL", "LEF1"],
    "CD8 EM": ["CD8A", "CD8B", "GZMK", "CCL5", "GZMH"],
    "MAIT": ["SLC4A10", "TRAV1-2", "KLRB1", "ZBTB16", "NCR3"],
    "gdT": ["TRDC", "TRGC1", "TRGC2", "TRDV2", "TRGV9"],
    "T Prolif": ["MKI67", "TOP2A", "TYMS", "NUSAP1"],
}

PBMC_MYELOID = {
    "CD14 Mono": ["CD14", "LYZ", "S100A8", "S100A9", "VCAN", "FCN1"],
    "CD16 Mono": ["FCGR3A", "MS4A7", "CDKN1C", "CX3CR1"],
    "cDC1": ["CLEC9A", "XCR1", "CADM1", "IDO1"],
    "cDC2": ["CD1C", "FCER1A", "CLEC10A", "CD1E"],
    "pDC": ["LILRA4", "CLEC4C", "IL3RA", "PLD4"],
    "ASDC": ["AXL", "SIGLEC6", "PPP1R14A", "SCT"],
}


def _immune_high(label: str):
    table = {
        "T cells": "T",
        "Double-negative thymocytes": "T",
        "Double-positive thymocytes": "T",
        "ILC": "NK",
        "ILC precursor": "NK",
        "B cells": "B",
        "B-cell lineage": "B",
        "Plasma cells": "B",
        "Monocytes": "Mono",
        "Mono-mac": "Mono",
        "Monocyte precursor": "Mono",
        "DC": "DC",
        "pDC": "DC",
        "DC precursor": "DC",
        "pDC precursor": "DC",
        "Megakaryocytes/platelets": "Platelet",
        "Early MK": "Platelet",
        "Megakaryocyte precursor": "Platelet",
        "Erythrocytes": "Eryth",
        "Erythroid": "Eryth",
        "HSC/MPP": "HSPC",
    }
    return table.get(label)


def _immune_low_t(label: str):
    table = {
        "Tcm/Naive helper T cells": "CD4 Naive/CM",
        "Tem/Effector helper T cells": "CD4 EM",
        "Tem/Effector helper T cells PD1+": "CD4 EM",
        "Type 1 helper T cells": "CD4 EM",
        "Type 17 helper T cells": "CD4 EM",
        "Follicular helper T cells": "CD4 EM",
        "Memory CD4+ cytotoxic T cells": "CD4 CTL",
        "Regulatory T cells": "Treg",
        "Treg(diff)": "Treg",
        "Tcm/Naive cytotoxic T cells": "CD8 Naive/CM",
        "Tem/Temra cytotoxic T cells": "CD8 EM",
        "Tem/Trm cytotoxic T cells": "CD8 EM",
        "Trm cytotoxic T cells": "CD8 EM",
        "MAIT cells": "MAIT",
        "gamma-delta T cells": "gdT",
        "CRTAM+ gamma-delta T cells": "gdT",
        "Cycling T cells": "T Prolif",
        "Cycling gamma-delta T cells": "T Prolif",
    }
    return table.get(label)


def _immune_low_myeloid(label: str):
    table = {
        "Classical monocytes": "CD14 Mono",
        "Monocytes": "CD14 Mono",
        "Non-classical monocytes": "CD16 Mono",
        "DC1": "cDC1",
        "DC2": "cDC2",
        "DC3": "cDC2",
        "pDC": "pDC",
        "Transitional DC": "ASDC",
    }
    return table.get(label)


# --------------------------------------------------------------------------- #
# Human cortex (Siletti et al. 2023, middle frontal gyrus A46 dissection)
# Ground truth: supercluster -> class; interneurons: label transferred from the
# Allen MTG taxonomy (Siletti et al. cluster annotation table).
# --------------------------------------------------------------------------- #
BRAIN_ALL = {
    "Excitatory": ["SLC17A7", "SATB2", "NEUROD6", "TBR1"],
    "Inhibitory": ["GAD1", "GAD2", "SLC32A1", "DLX1"],
    "Astrocyte": ["AQP4", "GFAP", "ALDH1L1", "GJA1", "FGFR3"],
    "Oligodendrocyte": ["MOG", "MOBP", "PLP1", "OPALIN", "MAG"],
    "OPC": ["PDGFRA", "CSPG4", "VCAN", "PCDH15"],
    "Microglia": ["P2RY12", "CX3CR1", "CSF1R", "C1QB", "PTPRC"],
    "Endothelial": ["CLDN5", "FLT1", "VWF", "PECAM1"],
    "Mural/VLMC": ["PDGFRB", "DCN", "COL1A2", "RGS5"],
}
BRAIN_ALL_TO_EVAL = {"Endothelial": "Vascular", "Mural/VLMC": "Vascular"}

BRAIN_INH = {
    "Pvalb": ["PVALB", "KCNS3", "LHX6"],
    "Sst": ["SST", "LHX6"],
    "Chandelier": ["SCUBE3", "UNC5B", "PVALB", "LHX6"],  # low confidence
    "Lamp5 Lhx6": ["LAMP5", "LHX6", "NXPH1"],  # low confidence
    "Lamp5": ["LAMP5", "NDNF", "RELN", "ADARB2"],
    "Vip": ["VIP", "CALB2", "ADARB2"],
    "Sncg": ["CCK", "CNR1", "SNCG", "ADARB2"],  # low confidence
    "Pax6": ["PAX6", "ADARB2"],
}

# --------------------------------------------------------------------------- #
# Negative markers, fixed a priori from the definition of each population
# (lineage exclusion: CD4 vs CD8, CD3- NK cells, CCR7- effector memory, VIP- Sncg/Pax6,
# LHX6- Lamp5 ...). Applied on top of the positive dictionaries above.
# --------------------------------------------------------------------------- #
NEGATIVES = {
    "pbmc_all": {
        "NK": ["CD3D", "CD3E"],
        "Plasmablast": ["MS4A1"],
    },
    "pbmc_T": {
        "CD4 Naive/CM": ["CD8A", "CD8B"],
        "CD4 EM": ["CD8A", "CD8B", "CCR7"],
        "CD4 CTL": ["CD8A", "CD8B"],
        "Treg": ["CD8A", "CD8B"],
        "CD8 Naive/CM": ["CD4"],
        "CD8 EM": ["CD4", "CCR7"],
        "MAIT": ["TRDC"],
    },
    "pbmc_myeloid": {
        "CD14 Mono": ["FCGR3A"],
        "CD16 Mono": ["CD14"],
    },
    "brain_all": {
        "Excitatory": ["GAD1", "GAD2"],
        "Inhibitory": ["SLC17A7"],
    },
    "brain_inh": {
        "Lamp5": ["LHX6"],
        "Lamp5 Lhx6": ["SST", "PVALB"],
        "Sst": ["PVALB"],
        "Sncg": ["VIP"],
        "Pax6": ["VIP"],  # low confidence
    },
}


#: Post hoc variant: drops the two negative sets that the first run showed to be transcribed
#: in the target population (CD3D/CD3E in NK cells, FCGR3A in CD14 monocytes). Chosen after
#: seeing the benchmark, so its results are optimistic.
_DROPPED_NEGATIVES = {("pbmc_all", "NK"), ("pbmc_myeloid", "CD14 Mono")}
NEGATIVES_CURATED = {
    scale: {c: g for c, g in negs.items() if (scale, c) not in _DROPPED_NEGATIVES}
    for scale, negs in NEGATIVES.items()
}


def with_negatives(markers: dict, negatives: dict) -> dict:
    """Positive marker dictionary -> cellassign dictionary with negative markers."""
    out = {}
    for category, genes in markers.items():
        neg = negatives.get(category)
        out[category] = {"pos": list(genes), "neg": list(neg)} if neg else list(genes)
    return out


_MTG_EXC = {"L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L6 IT Car3", "L5 ET", "L5/6 NP", "L6 CT", "L6b"}
_MTG_INH = {
    "Pvalb",
    "Sst",
    "Sst Chodl",
    "Vip",
    "Lamp5",
    "Lamp5 Lhx6",
    "Sncg",
    "Pax6",
    "Chandelier",
}
_MTG_GLIA = {
    "Astro": "Astrocyte",
    "Oligo": "Oligodendrocyte",
    "OPC": "OPC",
    "Micro-PVM": "Microglia",
    "Endo": "Vascular",
    "VLMC": "Vascular",
}


def _mtg_subclass(label: str) -> str:
    """'Pvalb_12' -> 'Pvalb'; 'Lamp5_Lhx6_1' -> 'Lamp5 Lhx6'."""
    return re.sub(r"_\d+$", "", label).replace("_", " ")


def _mtg_class(label: str):
    sub = _mtg_subclass(label)
    if sub in _MTG_EXC:
        return "Excitatory"
    if sub in _MTG_INH:
        return "Inhibitory"
    return _MTG_GLIA.get(sub)


def _mtg_inh(label: str):
    sub = _mtg_subclass(label)
    if sub == "Sst Chodl":
        return "Sst"
    return sub if sub in _MTG_INH else None


def _pfc_class(label: str):
    first = label.split(" ")[0]
    if re.match(r"^L\d", first):
        return "Excitatory"
    table = {
        "InN": "Inhibitory",
        "Astro": "Astrocyte",
        "Oligo": "Oligodendrocyte",
        "OPC": "OPC",
        "COP": "OPC",
        "Micro": "Microglia",
        "Macro": "Microglia",
        "Myeloid": "Microglia",
        "Endo": "Vascular",
        "PC": "Vascular",
        "SMC": "Vascular",
        "VLMC": "Vascular",
    }
    return table.get(first)


def _pfc_inh(label: str):
    """Ma et al. 2022 DLPFC labels. ADARB2 types cannot be split into Vip/Sncg/Pax6."""
    if not label.startswith("InN"):
        return None
    if label.startswith("InN LAMP5 LHX6"):
        return "Lamp5 Lhx6"
    for prefix, sub in (("InN PVALB", "Pvalb"), ("InN SST", "Sst"), ("InN VIP", "Vip")):
        if label.startswith(prefix):
            return sub
    if label.startswith("InN LAMP5"):
        return "Lamp5"
    return None


SCALES = {
    "pbmc_all": {
        "markers": PBMC_ALL,
        "to_eval": PBMC_ALL_TO_EVAL,
        "celltypist": {"Immune_All_High": _immune_high},
    },
    "pbmc_T": {
        "markers": PBMC_T,
        "to_eval": {},
        "celltypist": {"Immune_All_Low": _immune_low_t},
    },
    "pbmc_myeloid": {
        "markers": PBMC_MYELOID,
        "to_eval": {},
        "celltypist": {"Immune_All_Low": _immune_low_myeloid},
    },
    "brain_all": {
        "markers": BRAIN_ALL,
        "to_eval": BRAIN_ALL_TO_EVAL,
        "celltypist": {
            "Adult_Human_MTG": _mtg_class,
            "Adult_Human_PrefrontalCortex": _pfc_class,
        },
    },
    "brain_inh": {
        "markers": BRAIN_INH,
        "to_eval": {},
        "celltypist": {
            "Adult_Human_MTG": _mtg_inh,
            "Adult_Human_PrefrontalCortex": _pfc_inh,
        },
    },
}


for _name, _scale in SCALES.items():
    _scale["markers_neg"] = with_negatives(_scale["markers"], NEGATIVES.get(_name, {}))
    _scale["markers_neg_curated"] = with_negatives(
        _scale["markers"], NEGATIVES_CURATED.get(_name, {})
    )
