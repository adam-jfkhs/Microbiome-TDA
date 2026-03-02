"""Mycobiome analysis split into two orthogonal axes.

Axis 1: Environmental mold exposure proxy
    - Based on Shoemaker panel taxa (Aspergillus, Penicillium, Cladosporium,
      Fusarium, Stachybotrys, Wallemia)
    - These are environmental fungi whose presence in gut samples may indicate
      inhalation/ingestion exposure
    - Does NOT require ITS sequencing — can be approximated from metadata,
      environmental surveys, or if present in shotgun metagenomics

Axis 2: Gut mycobiome composition
    - Endogenous gut fungi (Candida, Malassezia, Saccharomyces)
    - Reflects gut ecosystem state independent of environmental exposure
    - Requires ITS or shotgun metagenomics data

Splitting these prevents the project from collapsing if ITS data is missing,
and allows independent analysis of:
    - Exposure proxy → bacterial topology shift
    - Gut mycobiome → bacterial topology shift
    - Exposure × gut mycobiome interaction (only if both axes have data)
"""

import numpy as np
import pandas as pd


# --- Axis 1: Environmental mold exposure proxy ---

ENVIRONMENTAL_MOLD_TAXA = {
    "Aspergillus": {
        "source": "indoor_mold",
        "mycotoxins": ["aflatoxin", "ochratoxin", "gliotoxin"],
        "shoemaker_panel": True,
        "reference": "Shoemaker & House 2006 Neurotoxicol Teratol",
    },
    "Penicillium": {
        "source": "indoor_mold",
        "mycotoxins": ["patulin", "citrinin", "ochratoxin"],
        "shoemaker_panel": True,
        "reference": "Liew & Mohd-Redzwan 2018 J Toxicol",
    },
    "Cladosporium": {
        "source": "indoor_mold",
        "mycotoxins": ["cladosporin"],
        "shoemaker_panel": True,
        "reference": "Guerra et al. 2022 Toxins",
    },
    "Fusarium": {
        "source": "food_contaminant",
        "mycotoxins": ["deoxynivalenol", "zearalenone", "fumonisin"],
        "shoemaker_panel": True,
        "reference": "Robert et al. 2017 Toxicol Lett",
    },
    "Stachybotrys": {
        "source": "indoor_mold",
        "mycotoxins": ["satratoxin", "trichothecenes"],
        "shoemaker_panel": True,
        "reference": "Pestka et al. 2008 Toxicol Sci",
    },
    "Wallemia": {
        "source": "indoor_mold",
        "mycotoxins": ["walleminol"],
        "shoemaker_panel": True,
        "reference": "Zalar et al. 2005 Stud Mycol",
    },
}


# --- Axis 2: Gut mycobiome composition ---

GUT_MYCOBIOME_TAXA = {
    "Candida": {
        "role": "commensal_opportunist",
        "disruption_mechanism": "competitive exclusion, candidalysin",
        "bacterial_targets": ["Lactobacillus", "Bifidobacterium"],
        "reference": "Sokol et al. 2017 Gut",
    },
    "Malassezia": {
        "role": "skin_gut_colonizer",
        "disruption_mechanism": "Th17 immune activation",
        "bacterial_targets": ["Bacteroides", "Parabacteroides"],
        "reference": "Limon et al. 2019 Nature",
    },
    "Saccharomyces": {
        "role": "probiotic_transient",
        "disruption_mechanism": "competitive_exclusion (mild, transient)",
        "bacterial_targets": [],
        "reference": "McFarland 2010 World J Gastroenterol",
    },
}


def compute_exposure_proxy(otu_table, taxonomy, method="abundance"):
    """Compute environmental mold exposure proxy score per sample.

    This can work even without ITS data if environmental mold taxa
    are detectable in shotgun metagenomics or if metadata contains
    environmental mold survey results.

    Args:
        otu_table: Samples x taxa relative abundance DataFrame.
        taxonomy: Taxonomy DataFrame with 'Genus' and 'Kingdom' columns.
        method: 'abundance' (sum of environmental mold relative abundance)
                or 'binary' (presence/absence above threshold).

    Returns:
        Series of exposure proxy scores per sample.
    """
    genus_col = "Genus" if "Genus" in taxonomy.columns else None
    if genus_col is None:
        return pd.Series(0.0, index=otu_table.index, name="exposure_proxy")

    env_mold_otus = []
    for otu_id in otu_table.columns:
        if otu_id in taxonomy.index:
            genus = taxonomy.loc[otu_id, genus_col]
            if genus in ENVIRONMENTAL_MOLD_TAXA:
                env_mold_otus.append(otu_id)

    if not env_mold_otus:
        return pd.Series(0.0, index=otu_table.index, name="exposure_proxy")

    if method == "abundance":
        scores = otu_table[env_mold_otus].sum(axis=1)
    elif method == "binary":
        scores = (otu_table[env_mold_otus].sum(axis=1) > 0.001).astype(float)
    else:
        raise ValueError(f"Unknown method: {method}")

    return scores.rename("exposure_proxy")


def compute_exposure_proxy_from_metadata(metadata, column="mold_exposure"):
    """Extract exposure proxy directly from sample metadata.

    Use this when environmental survey data is available but mycobiome
    sequencing is not.

    Args:
        metadata: Sample metadata DataFrame.
        column: Column name containing exposure information.

    Returns:
        Series of exposure proxy values.
    """
    if column not in metadata.columns:
        raise KeyError(
            f"Column '{column}' not found in metadata. "
            f"Available: {list(metadata.columns)}"
        )
    return metadata[column].rename("exposure_proxy")


def compute_gut_mycobiome_burden(otu_table, taxonomy):
    """Compute per-sample gut mycobiome burden (endogenous gut fungi).

    Only counts gut-resident fungi (Candida, Malassezia, Saccharomyces),
    NOT environmental mold taxa.

    Args:
        otu_table: Samples x taxa relative abundance DataFrame.
        taxonomy: Taxonomy DataFrame with 'Genus' column.

    Returns:
        DataFrame with per-genus burden and total.
    """
    genus_col = "Genus" if "Genus" in taxonomy.columns else None
    if genus_col is None:
        return pd.DataFrame(index=otu_table.index)

    burdens = {}
    for fungus in GUT_MYCOBIOME_TAXA:
        matching_otus = [
            otu_id for otu_id in otu_table.columns
            if otu_id in taxonomy.index and taxonomy.loc[otu_id, genus_col] == fungus
        ]
        if matching_otus:
            burdens[f"burden_{fungus}"] = otu_table[matching_otus].sum(axis=1)

    if not burdens:
        return pd.DataFrame({"gut_mycobiome_total": 0.0}, index=otu_table.index)

    result = pd.DataFrame(burdens, index=otu_table.index)
    result["gut_mycobiome_total"] = result.sum(axis=1)
    return result


def partition_by_exposure(otu_table, taxonomy, threshold=None, quantile=0.75):
    """Partition samples by environmental mold exposure proxy.

    Args:
        otu_table: Samples x taxa relative abundance DataFrame.
        taxonomy: Taxonomy DataFrame.
        threshold: Absolute threshold. If None, uses quantile.
        quantile: Quantile for threshold (default: top 25% = high exposure).

    Returns:
        Tuple of (high_exposure_idx, low_exposure_idx).
    """
    proxy = compute_exposure_proxy(otu_table, taxonomy)

    if threshold is None:
        threshold = proxy.quantile(quantile)

    high = proxy.index[proxy >= threshold]
    low = proxy.index[proxy < threshold]
    return high, low


def partition_by_gut_mycobiome(otu_table, taxonomy, threshold=None, quantile=0.75):
    """Partition samples by gut mycobiome burden.

    Args:
        otu_table: Samples x taxa relative abundance DataFrame.
        taxonomy: Taxonomy DataFrame.
        threshold: Absolute threshold. If None, uses quantile.
        quantile: Quantile for threshold.

    Returns:
        Tuple of (high_burden_idx, low_burden_idx).
    """
    burdens = compute_gut_mycobiome_burden(otu_table, taxonomy)
    total = burdens.get("gut_mycobiome_total", pd.Series(0.0, index=otu_table.index))

    if threshold is None:
        threshold = total.quantile(quantile)

    high = total.index[total >= threshold]
    low = total.index[total < threshold]
    return high, low


def interaction_groups(otu_table, taxonomy, exposure_quantile=0.5, burden_quantile=0.5):
    """Create 2x2 interaction groups: exposure × gut mycobiome.

    Only meaningful if both axes have data. Returns four groups:
        - low_exposure_low_burden
        - low_exposure_high_burden
        - high_exposure_low_burden
        - high_exposure_high_burden

    Args:
        otu_table: Samples x taxa relative abundance DataFrame.
        taxonomy: Taxonomy DataFrame.
        exposure_quantile: Median split for exposure.
        burden_quantile: Median split for gut mycobiome.

    Returns:
        Series mapping sample IDs to group labels.
    """
    proxy = compute_exposure_proxy(otu_table, taxonomy)
    burdens = compute_gut_mycobiome_burden(otu_table, taxonomy)
    total_burden = burdens.get("gut_mycobiome_total",
                               pd.Series(0.0, index=otu_table.index))

    exp_thresh = proxy.quantile(exposure_quantile)
    bur_thresh = total_burden.quantile(burden_quantile)

    groups = pd.Series("", index=otu_table.index, name="interaction_group")

    high_exp = proxy >= exp_thresh
    high_bur = total_burden >= bur_thresh

    groups[~high_exp & ~high_bur] = "low_exposure_low_burden"
    groups[~high_exp & high_bur] = "low_exposure_high_burden"
    groups[high_exp & ~high_bur] = "high_exposure_low_burden"
    groups[high_exp & high_bur] = "high_exposure_high_burden"

    return groups
