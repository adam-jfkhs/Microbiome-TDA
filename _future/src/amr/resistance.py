"""
Antimicrobial resistance (AMR) gene carrier analysis for topological disruption.

AMR gene carriers — taxa harboring antibiotic resistance genes — can act as
topological disruptors in microbial co-occurrence networks by:
1. Persisting through antibiotic perturbations while neighbors die off
2. Acquiring competitive advantages that reshape community structure
3. Horizontal gene transfer creating new edges in the network
4. Disrupting cross-feeding networks that sustain NT production

This module identifies AMR-associated taxa and quantifies their topological
impact on the co-occurrence network.
"""

import numpy as np
import pandas as pd

# Known AMR-associated genera in the human gut
# Based on CARD database prevalence and gut microbiome literature
AMR_CARRIER_TAXA = {
    "Escherichia": {
        "resistance_classes": [
            "beta-lactam",
            "fluoroquinolone",
            "aminoglycoside",
            "tetracycline",
        ],
        "hgt_potential": "high",
        "reference": "Salyers et al. 2004 Clin Infect Dis",
    },
    "Klebsiella": {
        "resistance_classes": ["beta-lactam", "carbapenem", "aminoglycoside"],
        "hgt_potential": "high",
        "reference": "Wyres & Holt 2018 Microbiol Genomics",
    },
    "Enterococcus": {
        "resistance_classes": ["vancomycin", "aminoglycoside", "macrolide"],
        "hgt_potential": "high",
        "reference": "Arias & Murray 2012 Nat Rev Microbiol",
    },
    "Bacteroides": {
        "resistance_classes": ["beta-lactam", "tetracycline", "macrolide"],
        "hgt_potential": "medium",
        "reference": "Wexler 2007 Clin Microbiol Rev",
    },
    "Clostridium": {
        "resistance_classes": ["fluoroquinolone", "beta-lactam"],
        "hgt_potential": "medium",
        "reference": "Spigaglia 2016 J Med Microbiol",
    },
    "Staphylococcus": {
        "resistance_classes": ["methicillin", "vancomycin", "macrolide"],
        "hgt_potential": "medium",
        "reference": "Chambers & DeLeo 2009 Nat Rev Microbiol",
    },
    "Pseudomonas": {
        "resistance_classes": [
            "carbapenem",
            "fluoroquinolone",
            "aminoglycoside",
        ],
        "hgt_potential": "high",
        "reference": "Breidenstein et al. 2011 Trends Microbiol",
    },
    "Acinetobacter": {
        "resistance_classes": ["carbapenem", "aminoglycoside", "tetracycline"],
        "hgt_potential": "high",
        "reference": "Peleg et al. 2008 Clin Microbiol Rev",
    },
}


def identify_amr_carriers(taxonomy, level="Genus"):
    """
    Identify OTUs/ASVs belonging to known AMR carrier genera.

    Parameters
    ----------
    taxonomy : pd.DataFrame
        Taxonomy table with at least the specified level column.
    level : str
        Taxonomic level for matching (default 'Genus').

    Returns
    -------
    list
        OTU IDs matching AMR carrier genera.
    dict
        Mapping of OTU ID -> AMR carrier info dict.
    """
    if level not in taxonomy.columns:
        raise ValueError(f"Taxonomy table must have a '{level}' column")

    genus_map = taxonomy[level].to_dict()
    amr_otus = []
    amr_info = {}

    for otu_id, genus in genus_map.items():
        if genus in AMR_CARRIER_TAXA:
            amr_otus.append(otu_id)
            amr_info[otu_id] = AMR_CARRIER_TAXA[genus]

    return amr_otus, amr_info


def compute_amr_burden(otu_table, amr_otus):
    """
    Compute per-sample AMR carrier burden.

    Parameters
    ----------
    otu_table : pd.DataFrame
        Samples x OTUs relative abundance table.
    amr_otus : list
        OTU IDs identified as AMR carriers.

    Returns
    -------
    pd.Series
        Per-sample AMR carrier relative abundance.
    """
    present = [a for a in amr_otus if a in otu_table.columns]
    if not present:
        return pd.Series(0.0, index=otu_table.index, name="amr_burden")
    return otu_table[present].sum(axis=1).rename("amr_burden")


def compute_amr_diversity(otu_table, taxonomy):
    """
    Compute AMR-related diversity metrics per sample.

    Returns the number of distinct AMR carrier genera present and
    the number of distinct resistance classes represented.

    Parameters
    ----------
    otu_table : pd.DataFrame
        Samples x OTUs relative abundance table.
    taxonomy : pd.DataFrame
        Taxonomy table with 'Genus' column.

    Returns
    -------
    pd.DataFrame
        Per-sample AMR diversity metrics.
    """
    genus_map = taxonomy["Genus"].to_dict() if "Genus" in taxonomy.columns else {}

    results = []
    for sample in otu_table.index:
        present_genera = set()
        present_classes = set()

        for otu_id in otu_table.columns:
            if otu_table.loc[sample, otu_id] > 0:
                genus = genus_map.get(otu_id, "")
                if genus in AMR_CARRIER_TAXA:
                    present_genera.add(genus)
                    for rc in AMR_CARRIER_TAXA[genus]["resistance_classes"]:
                        present_classes.add(rc)

        results.append(
            {
                "sample": sample,
                "amr_genera_count": len(present_genera),
                "resistance_class_count": len(present_classes),
            }
        )

    return pd.DataFrame(results).set_index("sample")


def network_resilience_to_amr(correlation_matrix, taxonomy, removal_fraction=0.5):
    """
    Simulate the topological impact of AMR carrier dominance by
    progressively removing non-AMR taxa from the network.

    This models antibiotic perturbation: AMR carriers survive while
    susceptible taxa are eliminated, and we track how the network
    topology degrades.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        OTU x OTU correlation matrix.
    taxonomy : pd.DataFrame
        Taxonomy table with 'Genus' column.
    removal_fraction : float
        Fraction of non-AMR taxa to remove (0-1).

    Returns
    -------
    list of pd.DataFrame
        Sequence of progressively degraded correlation submatrices.
    list of float
        Fraction of non-AMR taxa removed at each step.
    """
    amr_otus, _ = identify_amr_carriers(taxonomy)
    amr_set = set(amr_otus) & set(correlation_matrix.columns)
    non_amr = [c for c in correlation_matrix.columns if c not in amr_set]

    n_remove_total = int(len(non_amr) * removal_fraction)
    n_steps = min(10, n_remove_total)
    if n_steps == 0:
        return [correlation_matrix], [0.0]

    step_size = n_remove_total // n_steps

    # Randomly order non-AMR taxa for removal
    rng = np.random.default_rng(42)
    removal_order = rng.permutation(non_amr).tolist()

    matrices = [correlation_matrix]
    fractions = [0.0]

    remaining = list(correlation_matrix.columns)
    for i in range(1, n_steps + 1):
        n_remove = i * step_size
        to_remove = set(removal_order[:n_remove])
        kept = [c for c in remaining if c not in to_remove]

        if len(kept) < 2:
            break

        sub = correlation_matrix.loc[kept, kept]
        matrices.append(sub)
        fractions.append(n_remove / len(non_amr))

    return matrices, fractions


def hgt_edge_potential(correlation_matrix, taxonomy):
    """
    Identify edges in the co-occurrence network that connect high-HGT-potential
    AMR carriers, representing potential horizontal gene transfer routes.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        OTU x OTU correlation matrix.
    taxonomy : pd.DataFrame
        Taxonomy table with 'Genus' column.

    Returns
    -------
    pd.DataFrame
        Edges between high-HGT AMR carriers with their correlation strength.
    """
    genus_map = taxonomy["Genus"].to_dict() if "Genus" in taxonomy.columns else {}

    high_hgt = [
        otu
        for otu in correlation_matrix.columns
        if AMR_CARRIER_TAXA.get(genus_map.get(otu, ""), {}).get("hgt_potential")
        == "high"
    ]

    if len(high_hgt) < 2:
        return pd.DataFrame(columns=["source", "target", "correlation", "source_genus", "target_genus"])

    edges = []
    for i, otu_a in enumerate(high_hgt):
        for otu_b in high_hgt[i + 1 :]:
            corr = correlation_matrix.loc[otu_a, otu_b]
            if abs(corr) > 0.3:  # Only meaningful correlations
                edges.append(
                    {
                        "source": otu_a,
                        "target": otu_b,
                        "correlation": corr,
                        "source_genus": genus_map.get(otu_a, ""),
                        "target_genus": genus_map.get(otu_b, ""),
                    }
                )

    return pd.DataFrame(edges)
