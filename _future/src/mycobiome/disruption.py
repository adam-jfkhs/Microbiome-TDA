"""
Mycobiome disruption topology analysis.

Quantifies how fungal taxa disrupt the topological structure of bacterial
co-occurrence networks. Fungi (especially Candida, Aspergillus, Malassezia)
can fragment bacterial community structure through competition, mycotoxin
production, and immune modulation.

The key idea: compare persistence diagrams of the bacterial co-occurrence
network WITH vs WITHOUT fungal-associated perturbation to measure the
topological impact of mycobiome presence.
"""

import numpy as np
import pandas as pd

# Known gut-relevant fungi and their disruption mechanisms
DISRUPTIVE_FUNGI = {
    "Candida": {
        "mechanism": "competitive_exclusion",
        "targets": ["Lactobacillus", "Bifidobacterium"],
        "effect": "Displaces GABA and serotonin precursor producers",
        "reference": "Sokol et al. 2017 Gut",
    },
    "Aspergillus": {
        "mechanism": "mycotoxin_production",
        "targets": ["Faecalibacterium", "Roseburia"],
        "effect": "Aflatoxin/ochratoxin disrupts butyrate producers and barrier",
        "reference": "Maresca & Fantini 2010 Toxins",
    },
    "Malassezia": {
        "mechanism": "immune_modulation",
        "targets": ["Bacteroides", "Parabacteroides"],
        "effect": "Triggers Th17 response, shifts bacterial community",
        "reference": "Limon et al. 2019 Nature",
    },
    "Cladosporium": {
        "mechanism": "mycotoxin_production",
        "targets": ["Faecalibacterium", "Roseburia"],
        "effect": "Environmental mold; cladosporin disrupts mitochondria",
        "reference": "Guerra et al. 2022 Toxins",
    },
    "Penicillium": {
        "mechanism": "mycotoxin_production",
        "targets": ["Bifidobacterium", "Lactobacillus"],
        "effect": "Patulin disrupts epithelial tight junctions",
        "reference": "Liew & Mohd-Redzwan 2018 J Toxicol",
    },
    "Fusarium": {
        "mechanism": "mycotoxin_production",
        "targets": ["Faecalibacterium", "Coprococcus"],
        "effect": "Deoxynivalenol impairs barrier function and SCFA production",
        "reference": "Robert et al. 2017 Toxicol Lett",
    },
}


def identify_fungal_taxa(taxonomy, level="Genus"):
    """
    Identify fungal taxa in a combined bacterial-fungal taxonomy table.

    Parameters
    ----------
    taxonomy : pd.DataFrame
        Taxonomy table with at least 'Kingdom' and the specified level column.
    level : str
        Taxonomic level for identification (default 'Genus').

    Returns
    -------
    list
        OTU/ASV IDs identified as fungal.
    dict
        Mapping of fungal OTU IDs to their genus names.
    """
    if "Kingdom" not in taxonomy.columns:
        raise ValueError("Taxonomy table must have a 'Kingdom' column")

    fungal_mask = taxonomy["Kingdom"].isin(["Fungi", "k__Fungi", "Eukaryota"])
    fungal_ids = taxonomy.index[fungal_mask].tolist()

    genus_map = {}
    if level in taxonomy.columns:
        for otu_id in fungal_ids:
            genus_map[otu_id] = taxonomy.loc[otu_id, level]

    return fungal_ids, genus_map


def compute_fungal_burden(otu_table, fungal_ids):
    """
    Compute per-sample fungal burden as total relative abundance of
    fungal taxa.

    Parameters
    ----------
    otu_table : pd.DataFrame
        Samples x OTUs relative abundance table.
    fungal_ids : list
        OTU IDs identified as fungal.

    Returns
    -------
    pd.Series
        Per-sample fungal burden (0-1 scale).
    """
    present = [f for f in fungal_ids if f in otu_table.columns]
    if not present:
        return pd.Series(0.0, index=otu_table.index, name="fungal_burden")
    return otu_table[present].sum(axis=1).rename("fungal_burden")


def compute_disruption_score(otu_table, taxonomy):
    """
    Compute a disruption score quantifying how much disruptive fungi
    co-occur with their known bacterial targets.

    For each disruptive fungus present, the score increases proportionally
    to both the fungal abundance AND the loss of its target bacterial genera.

    Parameters
    ----------
    otu_table : pd.DataFrame
        Samples x OTUs relative abundance table (bacteria + fungi).
    taxonomy : pd.DataFrame
        Combined taxonomy table with 'Kingdom' and 'Genus' columns.

    Returns
    -------
    pd.DataFrame
        Per-sample disruption scores by fungal genus.
    """
    genus_map = taxonomy["Genus"].to_dict() if "Genus" in taxonomy.columns else {}

    # Map OTUs to genera
    otu_to_genus = {}
    for otu_id in otu_table.columns:
        g = genus_map.get(otu_id, "")
        if g:
            otu_to_genus[otu_id] = g

    scores = {}
    for fungus, info in DISRUPTIVE_FUNGI.items():
        # Find OTUs matching this fungus
        fungal_otus = [
            otu for otu, genus in otu_to_genus.items() if genus == fungus
        ]
        if not fungal_otus:
            continue

        fungal_abundance = otu_table[fungal_otus].sum(axis=1)

        # Find OTUs matching target bacteria
        target_otus = [
            otu
            for otu, genus in otu_to_genus.items()
            if genus in info["targets"]
        ]
        if not target_otus:
            target_depletion = pd.Series(1.0, index=otu_table.index)
        else:
            target_abundance = otu_table[target_otus].sum(axis=1)
            # Depletion = 1 - normalized target abundance
            max_target = target_abundance.max()
            if max_target > 0:
                target_depletion = 1.0 - (target_abundance / max_target)
            else:
                target_depletion = pd.Series(1.0, index=otu_table.index)

        # Disruption = fungal presence * target depletion
        scores[f"disruption_{fungus}"] = fungal_abundance * target_depletion

    if not scores:
        return pd.DataFrame(index=otu_table.index)

    result = pd.DataFrame(scores, index=otu_table.index)
    result["disruption_total"] = result.sum(axis=1)
    return result


def partition_samples_by_mycobiome(otu_table, taxonomy, threshold=0.01):
    """
    Partition samples into high-fungal and low-fungal groups for
    comparative topology analysis.

    Parameters
    ----------
    otu_table : pd.DataFrame
        Samples x OTUs relative abundance table.
    taxonomy : pd.DataFrame
        Taxonomy table with 'Kingdom' column.
    threshold : float
        Fungal burden threshold for partitioning (default 1%).

    Returns
    -------
    tuple of (pd.Index, pd.Index)
        (high_fungal_samples, low_fungal_samples)
    """
    fungal_ids, _ = identify_fungal_taxa(taxonomy)
    burden = compute_fungal_burden(otu_table, fungal_ids)

    high = burden.index[burden >= threshold]
    low = burden.index[burden < threshold]

    return high, low


def bacterial_network_without_fungi(correlation_matrix, taxonomy):
    """
    Extract the bacteria-only subnetwork from a mixed correlation matrix.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Full OTU x OTU correlation matrix (bacteria + fungi).
    taxonomy : pd.DataFrame
        Taxonomy table with 'Kingdom' column.

    Returns
    -------
    pd.DataFrame
        Bacteria-only correlation submatrix.
    """
    fungal_ids, _ = identify_fungal_taxa(taxonomy)
    bacterial_ids = [
        otu for otu in correlation_matrix.columns if otu not in fungal_ids
    ]
    return correlation_matrix.loc[bacterial_ids, bacterial_ids]
