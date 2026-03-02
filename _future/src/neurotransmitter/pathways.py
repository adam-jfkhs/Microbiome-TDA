"""
Neurotransmitter pathway mapping for gut microbiome taxa.

Maps bacterial genera/species to their known roles in neurotransmitter
biosynthesis, precursor production, and modulation. Based on published
literature linking specific taxa to serotonin, GABA, dopamine, and
their precursor pathways.
"""

import pandas as pd
import numpy as np

# --- Curated taxa-neurotransmitter mappings ---
# Each entry: genus -> {pathway, role, metabolite, evidence_level}
# evidence_level: "direct" (produces NT), "precursor" (produces precursor),
#                 "modulator" (modulates host production)

SEROTONIN_TAXA = {
    "Enterococcus": {
        "pathway": "tryptophan_hydroxylase",
        "role": "direct",
        "metabolite": "5-HT",
        "reference": "Yano et al. 2015 Cell",
    },
    "Streptococcus": {
        "pathway": "tryptophan_hydroxylase",
        "role": "direct",
        "metabolite": "5-HT",
        "reference": "Yano et al. 2015 Cell",
    },
    "Escherichia": {
        "pathway": "tryptophan_hydroxylase",
        "role": "direct",
        "metabolite": "5-HT",
        "reference": "Yano et al. 2015 Cell",
    },
    "Clostridium": {
        "pathway": "spore_forming_5ht_induction",
        "role": "modulator",
        "metabolite": "5-HT",
        "reference": "Yano et al. 2015 Cell",
    },
    "Turicibacter": {
        "pathway": "serotonin_transporter_expression",
        "role": "modulator",
        "metabolite": "5-HT",
        "reference": "Fung et al. 2019 Nat Microbiol",
    },
    "Ruminococcus": {
        "pathway": "tryptophan_metabolism",
        "role": "precursor",
        "metabolite": "tryptophan",
        "reference": "Valles-Colomer et al. 2019 Nat Microbiol",
    },
    "Lactobacillus": {
        "pathway": "tryptophan_indole",
        "role": "precursor",
        "metabolite": "indole-3-aldehyde",
        "reference": "Zelante et al. 2013 Immunity",
    },
}

GABA_TAXA = {
    "Lactobacillus": {
        "pathway": "glutamate_decarboxylase",
        "role": "direct",
        "metabolite": "GABA",
        "reference": "Barrett et al. 2012 J Appl Microbiol",
    },
    "Bifidobacterium": {
        "pathway": "glutamate_decarboxylase",
        "role": "direct",
        "metabolite": "GABA",
        "reference": "Barrett et al. 2012 J Appl Microbiol",
    },
    "Bacteroides": {
        "pathway": "glutamate_decarboxylase",
        "role": "direct",
        "metabolite": "GABA",
        "reference": "Strandwitz et al. 2019 Nat Microbiol",
    },
    "Parabacteroides": {
        "pathway": "glutamate_decarboxylase",
        "role": "direct",
        "metabolite": "GABA",
        "reference": "Strandwitz et al. 2019 Nat Microbiol",
    },
    "Eggerthella": {
        "pathway": "gaba_consumption",
        "role": "modulator",
        "metabolite": "GABA",
        "reference": "Valles-Colomer et al. 2019 Nat Microbiol",
    },
}

DOPAMINE_TAXA = {
    "Enterococcus": {
        "pathway": "tyrosine_decarboxylase",
        "role": "precursor",
        "metabolite": "tyramine",
        "reference": "van Kessel et al. 2019 Nat Med",
    },
    "Escherichia": {
        "pathway": "tyrosine_hydroxylase",
        "role": "direct",
        "metabolite": "dopamine",
        "reference": "Asano et al. 2012 Am J Physiol",
    },
    "Bacillus": {
        "pathway": "tyrosine_hydroxylase",
        "role": "direct",
        "metabolite": "dopamine",
        "reference": "Tsavkelova et al. 2000 Process Biochem",
    },
    "Staphylococcus": {
        "pathway": "tyrosine_decarboxylase",
        "role": "precursor",
        "metabolite": "tyramine",
        "reference": "Kuley et al. 2011 Food Chem",
    },
    "Hafnia": {
        "pathway": "tyrosine_decarboxylase",
        "role": "precursor",
        "metabolite": "tyramine",
        "reference": "Ku et al. 2016 Appl Microbiol",
    },
}

SCFA_TAXA = {
    "Faecalibacterium": {
        "pathway": "butyrate_synthesis",
        "role": "direct",
        "metabolite": "butyrate",
        "reference": "Louis & Flint 2017 Environ Microbiol",
    },
    "Roseburia": {
        "pathway": "butyrate_synthesis",
        "role": "direct",
        "metabolite": "butyrate",
        "reference": "Louis & Flint 2017 Environ Microbiol",
    },
    "Eubacterium": {
        "pathway": "butyrate_synthesis",
        "role": "direct",
        "metabolite": "butyrate",
        "reference": "Louis & Flint 2017 Environ Microbiol",
    },
    "Coprococcus": {
        "pathway": "butyrate_synthesis",
        "role": "direct",
        "metabolite": "butyrate",
        "reference": "Vital et al. 2014 mBio",
    },
    "Akkermansia": {
        "pathway": "propionate_synthesis",
        "role": "direct",
        "metabolite": "propionate",
        "reference": "Derrien et al. 2004 Int J Syst Evol Microbiol",
    },
    "Bacteroides": {
        "pathway": "propionate_synthesis",
        "role": "direct",
        "metabolite": "propionate",
        "reference": "Reichardt et al. 2014 ISME J",
    },
    "Bifidobacterium": {
        "pathway": "acetate_synthesis",
        "role": "direct",
        "metabolite": "acetate",
        "reference": "Fukuda et al. 2011 Nature",
    },
}


def get_pathway_database():
    """Return the full taxa-neurotransmitter mapping as a DataFrame."""
    records = []
    for db_name, db in [
        ("serotonin", SEROTONIN_TAXA),
        ("GABA", GABA_TAXA),
        ("dopamine", DOPAMINE_TAXA),
        ("SCFA", SCFA_TAXA),
    ]:
        for genus, info in db.items():
            records.append(
                {
                    "genus": genus,
                    "neurotransmitter": db_name,
                    "pathway": info["pathway"],
                    "role": info["role"],
                    "metabolite": info["metabolite"],
                    "reference": info["reference"],
                }
            )
    return pd.DataFrame(records)


def score_neurotransmitter_potential(otu_table, taxonomy, target="all"):
    """
    Score each sample's neurotransmitter production potential based on
    the relative abundance of taxa known to participate in NT pathways.

    Parameters
    ----------
    otu_table : pd.DataFrame
        Samples x OTUs relative abundance table.
    taxonomy : pd.DataFrame
        OTU taxonomy table with at least a 'Genus' column.
    target : str
        One of 'serotonin', 'GABA', 'dopamine', 'SCFA', or 'all'.

    Returns
    -------
    pd.DataFrame
        Samples x neurotransmitter scores, with columns for each pathway.
    """
    target_dbs = {
        "serotonin": SEROTONIN_TAXA,
        "GABA": GABA_TAXA,
        "dopamine": DOPAMINE_TAXA,
        "SCFA": SCFA_TAXA,
    }
    if target != "all":
        target_dbs = {target: target_dbs[target]}

    # Map OTU IDs to genus level
    if "Genus" not in taxonomy.columns:
        raise ValueError("Taxonomy table must have a 'Genus' column")

    genus_map = taxonomy["Genus"].to_dict()

    scores = {}
    for nt_name, taxa_db in target_dbs.items():
        role_weights = {"direct": 1.0, "precursor": 0.5, "modulator": 0.3}

        nt_score = np.zeros(len(otu_table))
        for otu_id in otu_table.columns:
            genus = genus_map.get(otu_id, "")
            if genus in taxa_db:
                weight = role_weights.get(taxa_db[genus]["role"], 0.1)
                nt_score += otu_table[otu_id].values * weight

        scores[f"{nt_name}_score"] = nt_score

    return pd.DataFrame(scores, index=otu_table.index)


def identify_nt_subnetwork(correlation_matrix, taxonomy, target="serotonin"):
    """
    Extract the subnetwork of OTUs involved in a specific neurotransmitter
    pathway from a full correlation matrix.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        OTU x OTU correlation matrix.
    taxonomy : pd.DataFrame
        OTU taxonomy table with 'Genus' column.
    target : str
        Neurotransmitter pathway to extract.

    Returns
    -------
    pd.DataFrame
        Submatrix of correlation_matrix restricted to NT-associated OTUs.
    list
        List of OTU IDs in the subnetwork.
    """
    taxa_db = {
        "serotonin": SEROTONIN_TAXA,
        "GABA": GABA_TAXA,
        "dopamine": DOPAMINE_TAXA,
        "SCFA": SCFA_TAXA,
    }[target]

    genus_map = taxonomy["Genus"].to_dict()

    nt_otus = [
        otu_id
        for otu_id in correlation_matrix.columns
        if genus_map.get(otu_id, "") in taxa_db
    ]

    if len(nt_otus) < 2:
        return pd.DataFrame(), nt_otus

    return correlation_matrix.loc[nt_otus, nt_otus], nt_otus


def compute_crossfeeding_potential(otu_table, taxonomy):
    """
    Compute cross-feeding potential between neurotransmitter pathways.

    Cross-feeding occurs when SCFA producers (especially butyrate) support
    the growth of serotonin/GABA producers, or when tryptophan metabolizers
    provide precursors. This function quantifies co-occurrence of taxa
    across different NT pathways within each sample.

    Parameters
    ----------
    otu_table : pd.DataFrame
        Samples x OTUs relative abundance table.
    taxonomy : pd.DataFrame
        OTU taxonomy table with 'Genus' column.

    Returns
    -------
    pd.DataFrame
        Samples x pathway-pair cross-feeding scores.
    """
    all_scores = score_neurotransmitter_potential(otu_table, taxonomy, target="all")

    pairs = [
        ("SCFA_score", "serotonin_score"),
        ("SCFA_score", "GABA_score"),
        ("SCFA_score", "dopamine_score"),
        ("serotonin_score", "GABA_score"),
    ]

    crossfeed = {}
    for col_a, col_b in pairs:
        # Geometric mean of pathway scores — high only when both present
        crossfeed[f"crossfeed_{col_a}_{col_b}"] = np.sqrt(
            all_scores[col_a] * all_scores[col_b]
        )

    return pd.DataFrame(crossfeed, index=otu_table.index)
