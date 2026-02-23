# Microbiome-TDA

**Persistent Homology Reveals Microbial Community Topologies Linked to Serotonin and Neuropeptide Production**

*A Topological Biomarker Layer for Evaluating CIRS-Associated Inflammatory Signatures in Mold-Exposed Cohorts*

## Overview

This project applies persistent homology to bacterial co-occurrence networks to detect topological signatures of microbial community configurations, and interprets these shifts through evidence-weighted priors linking taxa to inflammatory signaling.

Key distinctions:

- **Evidence-weighted priors, not deterministic mappings** — each taxon-biomarker edge carries direction, evidence grade (A/B/C), study types, and citations. We generate testable hypotheses, not predictions.
- **Two orthogonal mycobiome axes** — environmental mold exposure proxy (Aspergillus, Penicillium, etc.) vs endogenous gut mycobiome (Candida, Malassezia). Each axis is analyzed independently.
- **CIRS evaluated, not endorsed** — the Shoemaker/CIRS framework is treated as a case study for topology-informed inflammatory signature evaluation.

## What works right now

The pipeline runs end-to-end:

```
load_cohort() → preprocess (filter + CLR) → Spearman co-occurrence →
correlation distance → Vietoris-Rips → persistent homology (Betti-0/1/2) →
persistence entropy + landscapes → permutation test on diagram distances
```

**Notebook 01** (`notebooks/01_data_exploration.ipynb`) is the minimum publishable unit — loads a cohort, preprocesses, computes per-group persistent homology, and produces publication-quality figures with statistical tests.

## Repository Structure

```
src/
  data/             - Loaders (HMP, AGP, curatedMGD, synthetic), preprocessing
  networks/         - Co-occurrence (Spearman), distance matrices (Aitchison)
  tda/              - Filtration, persistent homology (ripser), features, regimes
  analysis/         - Statistics (permutation, FDR, effect sizes), correlation, ML
  visualization/    - Persistence diagrams, Betti curves, networks
  cirs/             - Evidence-weighted biomarker priors, mycobiome decomposition
  neurotransmitter/ - NT pathway knowledge base (serotonin, GABA, dopamine, SCFA)
  mycobiome/        - Legacy fungal disruption module
  amr/              - AMR carrier disruption and perturbation simulation
notebooks/          - Jupyter analysis notebooks
paper/              - LaTeX manuscript
tests/              - Unit tests
configs/            - Pipeline parameters
scripts/            - Data download scripts
```

## Setup

```bash
pip install numpy pandas scipy networkx matplotlib seaborn scikit-learn ripser persim
```

For real data (optional — synthetic cohort works out of the box):
```bash
bash scripts/download_hmp.sh
bash scripts/download_agp.sh
```

## Biomarker Hierarchy

| Tier | Biomarker | CIRS Direction |
|------|-----------|----------------|
| Primary | TGF-β1 | Elevated |
| Secondary | VIP | Reduced |
| Exploratory | MMP-9 | Elevated |
| Exploratory | C4a | Elevated |

## License

MIT — see [LICENSE](LICENSE).
