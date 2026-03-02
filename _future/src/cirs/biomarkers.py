"""Evidence-weighted biomarker priors for CIRS-associated inflammatory signaling.

This module implements a knowledge graph of taxa-to-biomarker relationships,
where each edge carries:
    - direction (+/-)
    - evidence grade (A/B/C)
    - study types supporting the link
    - citations

This is NOT a deterministic mapping. It provides evidence-weighted priors
linking taxa/modules to inflammatory signaling, used to interpret topological
shifts and generate testable hypotheses.

Biomarker hierarchy:
    Primary:     TGF-beta1
    Secondary:   VIP (vasoactive intestinal peptide)
    Exploratory: MMP-9, C4a
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class EvidenceGrade(Enum):
    """Strength of evidence for a taxon-biomarker link.

    A: Human interventional or large prospective cohort (n > 100)
    B: Human association study, multiple animal studies, or meta-analysis of smaller studies
    C: Single animal study, in vitro, or computational prediction
    """
    A = "A"
    B = "B"
    C = "C"


class StudyType(Enum):
    RCT = "RCT"
    PROSPECTIVE_COHORT = "prospective_cohort"
    CROSS_SECTIONAL = "cross_sectional"
    ANIMAL_MODEL = "animal_model"
    IN_VITRO = "in_vitro"
    META_ANALYSIS = "meta_analysis"
    COMPUTATIONAL = "computational"


# Evidence grade to numeric weight for scoring
EVIDENCE_WEIGHTS = {
    EvidenceGrade.A: 1.0,
    EvidenceGrade.B: 0.6,
    EvidenceGrade.C: 0.3,
}


@dataclass
class Biomarker:
    """A CIRS-associated inflammatory biomarker."""
    name: str
    full_name: str
    role: str                     # biological role in CIRS cascade
    tier: str                     # 'primary', 'secondary', 'exploratory'
    normal_range: Optional[str] = None
    cirs_direction: Optional[str] = None  # 'elevated' or 'reduced' in CIRS


@dataclass
class TaxonBiomarkerEdge:
    """An evidence-weighted link between a taxon and a biomarker.

    This is a prior, not a claim. Each edge represents published evidence
    that a taxon influences a biomarker, weighted by evidence quality.
    """
    taxon: str                    # genus name
    biomarker: str                # biomarker name
    direction: str                # '+' (increases) or '-' (decreases)
    evidence_grade: EvidenceGrade
    study_types: List[StudyType]
    mechanism: str                # brief mechanism description
    citations: List[str] = field(default_factory=list)
    notes: str = ""


# --- Biomarker definitions ---

BIOMARKERS = {
    "TGF-beta1": Biomarker(
        name="TGF-beta1",
        full_name="Transforming Growth Factor Beta 1",
        role="Master regulator of tissue remodeling and immune suppression; "
             "drives fibrosis and Treg differentiation. Chronically elevated "
             "in CIRS, contributing to immune dysregulation.",
        tier="primary",
        normal_range="<2380 pg/mL",
        cirs_direction="elevated",
    ),
    "VIP": Biomarker(
        name="VIP",
        full_name="Vasoactive Intestinal Peptide",
        role="Neuropeptide regulating intestinal motility, epithelial barrier "
             "integrity, and anti-inflammatory signaling. Depleted in CIRS, "
             "leading to barrier dysfunction and dysregulated peristalsis.",
        tier="secondary",
        normal_range="23-63 pg/mL",
        cirs_direction="reduced",
    ),
    "MMP-9": Biomarker(
        name="MMP-9",
        full_name="Matrix Metalloproteinase 9",
        role="Extracellular matrix degradation enzyme. Elevated in systemic "
             "inflammation, contributes to tissue destruction and barrier "
             "breakdown. Marker of innate immune activation.",
        tier="exploratory",
        normal_range="85-332 ng/mL",
        cirs_direction="elevated",
    ),
    "C4a": Biomarker(
        name="C4a",
        full_name="Complement Component 4a",
        role="Complement split product indicating lectin pathway activation. "
             "Elevated by mold exposure and biotoxins. Early marker of innate "
             "immune response to environmental triggers.",
        tier="exploratory",
        normal_range="0-2830 ng/mL",
        cirs_direction="elevated",
    ),
}


# --- Evidence-weighted priors: taxon → biomarker edges ---

TAXON_BIOMARKER_PRIORS = [
    # === TGF-beta1 edges ===
    TaxonBiomarkerEdge(
        taxon="Faecalibacterium",
        biomarker="TGF-beta1",
        direction="-",
        evidence_grade=EvidenceGrade.A,
        study_types=[StudyType.PROSPECTIVE_COHORT, StudyType.ANIMAL_MODEL],
        mechanism="F. prausnitzii produces butyrate, which suppresses NF-kB "
                  "and downstream TGF-beta1 overexpression. Butyrate also "
                  "promotes Treg differentiation via HDAC inhibition.",
        citations=[
            "Sokol et al. 2008 PNAS",
            "Furusawa et al. 2013 Nature",
        ],
    ),
    TaxonBiomarkerEdge(
        taxon="Roseburia",
        biomarker="TGF-beta1",
        direction="-",
        evidence_grade=EvidenceGrade.B,
        study_types=[StudyType.CROSS_SECTIONAL, StudyType.ANIMAL_MODEL],
        mechanism="Major butyrate producer; depletion associated with elevated "
                  "inflammatory markers including TGF-beta1 in IBD cohorts.",
        citations=[
            "Machiels et al. 2014 Gut",
            "Tamanai-Shacoori et al. 2017 Front Microbiol",
        ],
    ),
    TaxonBiomarkerEdge(
        taxon="Bacteroides",
        biomarker="TGF-beta1",
        direction="-",
        evidence_grade=EvidenceGrade.B,
        study_types=[StudyType.ANIMAL_MODEL, StudyType.IN_VITRO],
        mechanism="B. fragilis PSA induces IL-10 and Tregs via TLR2, "
                  "counterbalancing TGF-beta1-driven inflammation.",
        citations=[
            "Round & Mazmanian 2010 PNAS",
            "Mazmanian et al. 2008 Nature",
        ],
    ),
    TaxonBiomarkerEdge(
        taxon="Escherichia",
        biomarker="TGF-beta1",
        direction="+",
        evidence_grade=EvidenceGrade.B,
        study_types=[StudyType.ANIMAL_MODEL, StudyType.CROSS_SECTIONAL],
        mechanism="LPS from E. coli activates TLR4/NF-kB, upregulating "
                  "TGF-beta1 and pro-inflammatory cytokines. Bloom of "
                  "Proteobacteria is a hallmark of dysbiosis.",
        citations=[
            "Shin et al. 2015 Cell Host Microbe",
            "Litvak et al. 2017 Trends Microbiol",
        ],
    ),
    TaxonBiomarkerEdge(
        taxon="Desulfovibrio",
        biomarker="TGF-beta1",
        direction="+",
        evidence_grade=EvidenceGrade.C,
        study_types=[StudyType.ANIMAL_MODEL],
        mechanism="H2S production damages epithelium and activates "
                  "inflammatory cascades including TGF-beta1 signaling.",
        citations=[
            "Rowan et al. 2010 Inflamm Bowel Dis",
        ],
    ),

    # === VIP edges ===
    TaxonBiomarkerEdge(
        taxon="Lactobacillus",
        biomarker="VIP",
        direction="+",
        evidence_grade=EvidenceGrade.B,
        study_types=[StudyType.ANIMAL_MODEL, StudyType.IN_VITRO],
        mechanism="Lactobacillus spp. stimulate VIP release from "
                  "enteroendocrine cells and enhance epithelial barrier "
                  "integrity via tight junction protein upregulation.",
        citations=[
            "Laval et al. 2015 PLoS ONE",
            "Bednorz et al. 2013 PLoS ONE",
        ],
    ),
    TaxonBiomarkerEdge(
        taxon="Bifidobacterium",
        biomarker="VIP",
        direction="+",
        evidence_grade=EvidenceGrade.B,
        study_types=[StudyType.ANIMAL_MODEL, StudyType.IN_VITRO],
        mechanism="Bifidobacterium spp. produce acetate and lactate that "
                  "cross-feed butyrate producers and support enteric "
                  "nervous system signaling including VIP pathways.",
        citations=[
            "Riviere et al. 2016 Front Microbiol",
        ],
    ),
    TaxonBiomarkerEdge(
        taxon="Akkermansia",
        biomarker="VIP",
        direction="+",
        evidence_grade=EvidenceGrade.C,
        study_types=[StudyType.ANIMAL_MODEL],
        mechanism="A. muciniphila strengthens barrier function via mucin "
                  "turnover; intact barriers support normal VIP signaling.",
        citations=[
            "Plovier et al. 2017 Nat Med",
        ],
    ),
    TaxonBiomarkerEdge(
        taxon="Candida",
        biomarker="VIP",
        direction="-",
        evidence_grade=EvidenceGrade.C,
        study_types=[StudyType.ANIMAL_MODEL, StudyType.IN_VITRO],
        mechanism="Candida overgrowth disrupts epithelial barrier and "
                  "may impair VIP-producing enteroendocrine cells via "
                  "candidalysin-mediated damage.",
        citations=[
            "Moyes et al. 2016 Nature",
        ],
    ),

    # === MMP-9 edges ===
    TaxonBiomarkerEdge(
        taxon="Escherichia",
        biomarker="MMP-9",
        direction="+",
        evidence_grade=EvidenceGrade.B,
        study_types=[StudyType.CROSS_SECTIONAL, StudyType.ANIMAL_MODEL],
        mechanism="LPS-driven inflammation upregulates MMP-9 via NF-kB "
                  "and MAPK pathways. Proteobacteria bloom correlates with "
                  "elevated serum MMP-9.",
        citations=[
            "Gan et al. 2004 Am J Physiol",
        ],
    ),
    TaxonBiomarkerEdge(
        taxon="Faecalibacterium",
        biomarker="MMP-9",
        direction="-",
        evidence_grade=EvidenceGrade.B,
        study_types=[StudyType.CROSS_SECTIONAL, StudyType.ANIMAL_MODEL],
        mechanism="Butyrate inhibits MMP-9 expression by suppressing "
                  "NF-kB activation in intestinal epithelial cells.",
        citations=[
            "Sokol et al. 2008 PNAS",
        ],
    ),

    # === C4a edges ===
    TaxonBiomarkerEdge(
        taxon="Aspergillus",
        biomarker="C4a",
        direction="+",
        evidence_grade=EvidenceGrade.B,
        study_types=[StudyType.CROSS_SECTIONAL, StudyType.ANIMAL_MODEL],
        mechanism="Aspergillus mycotoxins and beta-glucans activate the "
                  "lectin complement pathway, elevating C4a.",
        citations=[
            "Shoemaker & House 2006 Neurotoxicol Teratol",
        ],
    ),
    TaxonBiomarkerEdge(
        taxon="Candida",
        biomarker="C4a",
        direction="+",
        evidence_grade=EvidenceGrade.C,
        study_types=[StudyType.IN_VITRO, StudyType.ANIMAL_MODEL],
        mechanism="Candida cell wall mannans activate lectin pathway "
                  "complement via MBL binding.",
        citations=[
            "Ip & Bhatt 2018 J Fungi",
        ],
    ),
]


def get_priors():
    """Return the evidence-weighted prior table as a DataFrame.

    Each row is a taxon-biomarker edge with direction, evidence weight,
    mechanism, and citations.

    Returns:
        DataFrame with columns: taxon, biomarker, direction, evidence_grade,
            evidence_weight, mechanism, study_types, citations.
    """
    records = []
    for edge in TAXON_BIOMARKER_PRIORS:
        records.append({
            "taxon": edge.taxon,
            "biomarker": edge.biomarker,
            "direction": edge.direction,
            "evidence_grade": edge.evidence_grade.value,
            "evidence_weight": EVIDENCE_WEIGHTS[edge.evidence_grade],
            "mechanism": edge.mechanism,
            "study_types": ", ".join(st.value for st in edge.study_types),
            "citations": "; ".join(edge.citations),
            "notes": edge.notes,
        })
    return pd.DataFrame(records)


def get_biomarkers():
    """Return biomarker definitions as a DataFrame.

    Returns:
        DataFrame with biomarker metadata (name, role, tier, CIRS direction).
    """
    records = []
    for bm in BIOMARKERS.values():
        records.append({
            "name": bm.name,
            "full_name": bm.full_name,
            "tier": bm.tier,
            "role": bm.role,
            "normal_range": bm.normal_range,
            "cirs_direction": bm.cirs_direction,
        })
    return pd.DataFrame(records)


def score_biomarker_pressure(abundances, taxonomy_df, biomarker_name="TGF-beta1"):
    """Compute a probabilistic biomarker pressure score from abundances.

    NOT a prediction of biomarker level. This score represents the
    evidence-weighted alignment between observed taxonomic composition
    and priors suggesting pressure on a specific biomarker.

    Higher positive scores = composition aligns with priors predicting
    biomarker elevation. Lower/negative = composition aligns with priors
    predicting biomarker reduction.

    Args:
        abundances: Series or 1D array of relative abundances indexed by taxon.
        taxonomy_df: DataFrame with 'Genus' column, indexed by taxon name.
        biomarker_name: Which biomarker to score.

    Returns:
        Float: evidence-weighted biomarker pressure score.
    """
    priors = get_priors()
    bm_priors = priors[priors["biomarker"] == biomarker_name]

    if len(bm_priors) == 0:
        return 0.0

    # Map taxa to genera
    if isinstance(abundances, pd.Series):
        genus_map = {}
        for taxon in abundances.index:
            if taxon in taxonomy_df.index:
                genus = taxonomy_df.loc[taxon, "Genus"]
            else:
                genus = taxon  # assume index is already genus names
            genus_map[taxon] = genus

        genus_abundances = {}
        for taxon, genus in genus_map.items():
            genus_abundances[genus] = genus_abundances.get(genus, 0) + abundances[taxon]
    else:
        genus_abundances = dict(zip(taxonomy_df["Genus"], abundances))

    score = 0.0
    for _, edge in bm_priors.iterrows():
        abundance = genus_abundances.get(edge["taxon"], 0.0)
        sign = 1.0 if edge["direction"] == "+" else -1.0
        weight = edge["evidence_weight"]
        score += sign * weight * abundance

    return float(score)


def score_signature(otu_df, taxonomy_df, biomarker_name="TGF-beta1"):
    """Score biomarker pressure for all samples in a cohort.

    Args:
        otu_df: DataFrame of relative abundances (samples x taxa).
        taxonomy_df: Taxonomy DataFrame.
        biomarker_name: Which biomarker to score.

    Returns:
        Series of biomarker pressure scores indexed by sample ID.
    """
    scores = {}
    for sample_id in otu_df.index:
        scores[sample_id] = score_biomarker_pressure(
            otu_df.loc[sample_id], taxonomy_df, biomarker_name
        )
    return pd.Series(scores, name=f"{biomarker_name}_pressure")
