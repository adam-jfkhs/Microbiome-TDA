"""Synthetic microbiome cohort generator for pipeline development and testing.

Generates realistic OTU tables mimicking gut microbiome composition with:
- Dirichlet-multinomial sampling for compositional structure
- Realistic phylum-level proportions (Firmicutes/Bacteroidetes dominated)
- Two-group design (e.g., high vs low exposure proxy) with planted
  topological differences in co-occurrence structure
- Matching taxonomy and metadata tables
"""

import numpy as np
import pandas as pd


# Realistic genus-level composition for healthy adult gut microbiome.
# Proportions are approximate means from HMP stool samples.
GENERA = {
    # Firmicutes
    "Faecalibacterium": {"phylum": "Firmicutes", "base_prop": 0.12},
    "Roseburia": {"phylum": "Firmicutes", "base_prop": 0.06},
    "Eubacterium": {"phylum": "Firmicutes", "base_prop": 0.05},
    "Coprococcus": {"phylum": "Firmicutes", "base_prop": 0.04},
    "Ruminococcus": {"phylum": "Firmicutes", "base_prop": 0.04},
    "Blautia": {"phylum": "Firmicutes", "base_prop": 0.03},
    "Clostridium": {"phylum": "Firmicutes", "base_prop": 0.03},
    "Lactobacillus": {"phylum": "Firmicutes", "base_prop": 0.02},
    "Enterococcus": {"phylum": "Firmicutes", "base_prop": 0.01},
    "Streptococcus": {"phylum": "Firmicutes", "base_prop": 0.01},
    "Turicibacter": {"phylum": "Firmicutes", "base_prop": 0.005},
    # Bacteroidetes
    "Bacteroides": {"phylum": "Bacteroidetes", "base_prop": 0.15},
    "Prevotella": {"phylum": "Bacteroidetes", "base_prop": 0.08},
    "Parabacteroides": {"phylum": "Bacteroidetes", "base_prop": 0.03},
    "Alistipes": {"phylum": "Bacteroidetes", "base_prop": 0.02},
    # Actinobacteria
    "Bifidobacterium": {"phylum": "Actinobacteria", "base_prop": 0.04},
    "Collinsella": {"phylum": "Actinobacteria", "base_prop": 0.01},
    "Eggerthella": {"phylum": "Actinobacteria", "base_prop": 0.005},
    # Proteobacteria
    "Escherichia": {"phylum": "Proteobacteria", "base_prop": 0.02},
    "Desulfovibrio": {"phylum": "Proteobacteria", "base_prop": 0.01},
    "Hafnia": {"phylum": "Proteobacteria", "base_prop": 0.005},
    # Verrucomicrobia
    "Akkermansia": {"phylum": "Verrucomicrobia", "base_prop": 0.03},
    # Fusobacteria
    "Fusobacterium": {"phylum": "Fusobacteria", "base_prop": 0.005},
    # Fungi (for mycobiome axis)
    "Candida": {"phylum": "Ascomycota", "base_prop": 0.005, "kingdom": "Fungi"},
    "Aspergillus": {"phylum": "Ascomycota", "base_prop": 0.002, "kingdom": "Fungi"},
    "Malassezia": {"phylum": "Basidiomycota", "base_prop": 0.001, "kingdom": "Fungi"},
}

# NT-producing taxa (for planted signal)
SEROTONIN_TAXA = {"Enterococcus", "Streptococcus", "Escherichia", "Turicibacter"}
SCFA_TAXA = {"Faecalibacterium", "Roseburia", "Eubacterium", "Coprococcus",
             "Akkermansia", "Bacteroides", "Bifidobacterium"}


def generate_synthetic_cohort(
    n_samples=200,
    n_per_group=None,
    seed=42,
    effect_size=0.4,
    sequencing_depth_mean=50000,
    sequencing_depth_sd=15000,
    overdispersion=50.0,
):
    """Generate a synthetic microbiome cohort with two groups.

    Group A ("low_exposure"): baseline community structure.
    Group B ("high_exposure"): shifted composition — depleted SCFA producers,
        elevated Proteobacteria, slightly elevated fungi. This mimics the
        dysbiosis pattern associated with inflammatory exposure.

    The shift is designed to produce detectable topological differences
    in co-occurrence networks (altered connectivity among SCFA/NT producers).

    Args:
        n_samples: Total number of samples.
        n_per_group: Samples per group. If None, split evenly.
        seed: Random seed for reproducibility.
        effect_size: Magnitude of compositional shift (0-1 scale).
        sequencing_depth_mean: Mean reads per sample.
        sequencing_depth_sd: Std dev of reads per sample.
        overdispersion: Dirichlet concentration parameter. Lower = more
            variable communities. 50 is realistic for human gut.

    Returns:
        Tuple of (otu_df, metadata_df, taxonomy_df):
            - otu_df: samples × taxa count matrix
            - metadata_df: sample metadata with group labels
            - taxonomy_df: taxonomy table with Kingdom through Genus
    """
    rng = np.random.default_rng(seed)

    if n_per_group is None:
        n_per_group = n_samples // 2

    n_a = n_per_group
    n_b = n_samples - n_a

    genera = list(GENERA.keys())
    n_taxa = len(genera)

    # Base proportions
    base_props = np.array([GENERA[g]["base_prop"] for g in genera])
    base_props = base_props / base_props.sum()  # normalize

    # Group B shift: deplete SCFA producers, elevate Proteobacteria + fungi
    shifted_props = base_props.copy()
    for i, g in enumerate(genera):
        if g in SCFA_TAXA:
            shifted_props[i] *= (1 - effect_size * 0.6)
        elif GENERA[g]["phylum"] == "Proteobacteria":
            shifted_props[i] *= (1 + effect_size * 1.5)
        elif GENERA[g].get("kingdom") == "Fungi":
            shifted_props[i] *= (1 + effect_size * 2.0)
        elif g in SEROTONIN_TAXA:
            shifted_props[i] *= (1 - effect_size * 0.3)
    shifted_props = shifted_props / shifted_props.sum()

    # Sample counts using Dirichlet-multinomial
    counts = np.zeros((n_samples, n_taxa), dtype=int)
    depths = rng.normal(sequencing_depth_mean, sequencing_depth_sd, n_samples)
    depths = np.clip(depths, 5000, 200000).astype(int)

    for i in range(n_samples):
        if i < n_a:
            alpha = base_props * overdispersion
        else:
            alpha = shifted_props * overdispersion
        proportions = rng.dirichlet(alpha)
        counts[i] = rng.multinomial(depths[i], proportions)

    # Build DataFrames
    sample_ids = [f"S{i:04d}" for i in range(n_samples)]
    otu_df = pd.DataFrame(counts, index=sample_ids, columns=genera)

    # Metadata
    groups = ["low_exposure"] * n_a + ["high_exposure"] * n_b
    metadata = pd.DataFrame({
        "group": groups,
        "exposure_proxy": [0] * n_a + [1] * n_b,
        "sequencing_depth": depths,
        "body_site": "stool",
        "subject_id": [f"SUBJ{i:04d}" for i in range(n_samples)],
    }, index=sample_ids)

    # Add simulated continuous biomarker proxies (for later correlation)
    # These are noise + planted correlation with exposure group
    metadata["shannon_diversity"] = _simulate_diversity(
        counts, rng, group_effect=-0.3 * effect_size, n_a=n_a
    )

    # Taxonomy table
    taxonomy = pd.DataFrame([
        {
            "Kingdom": GENERA[g].get("kingdom", "Bacteria"),
            "Phylum": GENERA[g]["phylum"],
            "Class": "",
            "Order": "",
            "Family": "",
            "Genus": g,
            "Species": "",
        }
        for g in genera
    ], index=genera)

    return otu_df, metadata, taxonomy


def _simulate_diversity(counts, rng, group_effect, n_a):
    """Compute Shannon diversity + planted group effect."""
    from scipy.stats import entropy as shannon_entropy
    diversities = []
    for i, row in enumerate(counts):
        props = row / row.sum()
        props = props[props > 0]
        h = shannon_entropy(props)
        # Add noise + group effect
        noise = rng.normal(0, 0.15)
        shift = group_effect if i >= n_a else 0
        diversities.append(h + noise + shift)
    return diversities
