# Microbiome-TDA

**Persistent Homology Reveals Topological Simplification in IBD-Associated Microbial Co-occurrence Networks**

*A Paired Bootstrap Analysis of the American Gut Project with Covariate Matching, IBDMDB Validation, and Taxa-Set Sensitivity Testing*

Adam Levine — Independent Researcher · adalevine@bmchsd.org · February 2026 Preprint

---

## Overview

This project applies persistent homology (topological data analysis, TDA) to Spearman co-occurrence networks derived from 3,409 American Gut Project stool samples. We test whether inflammatory bowel disease (IBD) is associated with measurable differences in the loop-level topology of microbial co-occurrence networks — a structural property that conventional diversity metrics cannot capture.

**Key findings:**
- IBD-associated microbiomes show systematically lower topological complexity across all six H₁ scalar features (Cohen's *d* = 0.75–2.38; *p*_FDR < 0.005), fully preserved after age/sex/BMI covariate matching.
- Results replicate in the independent IBDMDB cohort (shotgun metagenomics; *d* = −0.52 to −1.46; all 6/6 features FDR-significant).
- High fecal calprotectin (≥250 µg/g) produces fewer but longer-lived loops — a distinct topological signature of active mucosal inflammation.

---

## Pipeline

```
AGP stool samples (n = 3,409)
  → CLR transformation (top-80 prevalent taxa)
  → Spearman co-occurrence matrix
  → Correlation distance → Vietoris–Rips filtration
  → Persistent homology (Ripser v0.6.14, H₁)
  → Six scalar features per diagram
  → Paired bootstrap (200 iterations, n = 100/group)
  → Sign-flip permutation test (500 shuffles) + Wilcoxon confirmatory
  → Benjamini–Hochberg FDR correction (18 tests/subset)
  → Independent replication: IBDMDB (n = 1,338, shotgun)
  → Classification benchmark: per-sample k-NN topology vs Shannon
```

---

## Repository Structure

```
src/
  data/              - AGP/IBDMDB loaders, CLR preprocessing, cohort filtering
  networks/          - Spearman co-occurrence, correlation distance matrices
  tda/               - Vietoris–Rips filtration, persistent homology, H₁ features
  analysis/          - Bootstrap, permutation tests, FDR, Cohen's d, Wilcoxon, ML
  visualization/     - Persistence diagrams, Betti curves, barcode plots

  [future modules — not used in current paper:]
  amr/               - AMR-associated taxa and resilience simulation
  cirs/              - Evidence-weighted inflammatory biomarker priors
  mycobiome/         - Fungal disruption indices
  neurotransmitter/  - Neurotransmitter pathway scoring

scripts/
  run_agp_bootstrap_v2.py       - PRIMARY: paired bootstrap on AGP (paper §2–3)
  run_agp_analysis.py           - Single-pass AGP comparison (exploratory)
  run_agp_bootstrap.py          - Bootstrap v1 (superseded by v2)
  run_ibdmdb_bootstrap.py       - IBDMDB replication analysis (paper §3.6)
  run_taxa_sensitivity.py       - Taxa-set sensitivity at N∈{50,80,120} (paper §3.4)
  run_classification_benchmark.py - Per-sample topology vs Shannon AUC comparison
  compute_shannon_comparison.py - Group-level Shannon diversity benchmark
  download_agp.sh / download_ibdmdb.sh / download_hmp.sh - Data download helpers
  fetch_agp_redbiom.py          - Alternative AGP fetch via redbiom

paper/
  main.tex + sections/          - LaTeX manuscript (compiled to main.pdf)
  references.bib                - Bibliography

notebooks/
  01_data_exploration.ipynb     - Data loading, QC, prevalence distributions

figures/                        - PNG outputs from analysis scripts
results/                        - CSV outputs (bootstrap results, sensitivity, AUC)
r_scripts/                      - R helpers (HMP download via BiocManager)
tests/                          - pytest unit tests for all src modules
configs/                        - Pipeline parameters (YAML)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate microbiome-tda
```

For AGP data (required for full analysis):
```bash
bash scripts/download_agp.sh
```

For IBDMDB data:
```bash
bash scripts/download_ibdmdb.sh
```

---

## Reproducing the Paper

All analyses use a fixed random seed (42). The scripts correspond to paper sections as follows:

| Script | Paper section |
|--------|--------------|
| `scripts/run_agp_bootstrap_v2.py` | §2 Methods, §3.1–3.3 Results |
| `scripts/run_taxa_sensitivity.py` | §3.4 Sensitivity analysis |
| `scripts/run_ibdmdb_bootstrap.py` | §3.6 IBDMDB replication |
| `scripts/run_classification_benchmark.py` | §3.5 Classification benchmark |

Run in order after downloading data:
```bash
python scripts/run_agp_bootstrap_v2.py
python scripts/run_taxa_sensitivity.py
python scripts/run_ibdmdb_bootstrap.py
python scripts/run_classification_benchmark.py
```

Results are written to `results/` and figures to `figures/`.

---

## Tests

```bash
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).
