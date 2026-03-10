# Microbiome-TDA

**Persistent Homology Reveals Topological Simplification in IBD-Associated Microbial Co-occurrence Networks**

*A Paired Bootstrap Analysis of the American Gut Project with Covariate Matching, IBDMDB Validation, and Taxa-Set Sensitivity Testing*

Adam Levine — Independent Researcher · March 2026

---

## Overview

This project applies persistent homology (topological data analysis, TDA) to Spearman co-occurrence networks derived from 3,409 American Gut Project (AGP) stool samples. We test whether inflammatory bowel disease (IBD) is associated with measurable differences in the loop-level topology of microbial co-occurrence networks — a structural property that conventional diversity metrics cannot capture.

**Key findings:**
- IBD-associated microbiomes show systematically lower topological complexity across all six H₁ scalar features (Cohen's |*d*| = 0.75–2.38; *p*_FDR < 0.005), fully preserved after age/sex/BMI covariate matching and across taxon-set sizes N ∈ {50, 80, 120}.
- Aitchison-PCoA(10) is the strongest single IBD classifier (LR AUC = 0.734); topology scalars (LR AUC = 0.668) and Shannon diversity (LR AUC = 0.688) perform comparably. Combined Aitchison + Topology + Shannon reaches LR AUC = 0.735.
- In IBDMDB (shotgun metagenomics, n = 106 subjects, 1,338 samples), the full-dataset IBD vs. non-IBD comparison is driven by overrepresentation of active-disease timepoints. A one-sample-per-subject sensitivity check reverses the direction for 5/6 features (|*d*| = 0.61–1.22, IBD > non-IBD). Within-IBD comparisons (calprotectin, HBI, SCCAI) are unaffected by this bias.
- High fecal calprotectin (≥250 µg/g) produces fewer but longer-lived loops — a distinct topological signature of active mucosal inflammation.

---

## Repository Structure

```
src/
  data/           - AGP/IBDMDB loaders, CLR preprocessing, cohort filtering
  networks/       - Spearman co-occurrence, correlation distance matrices
  tda/            - Vietoris–Rips filtration, persistent homology, H₁ features
  analysis/       - Bootstrap, permutation tests, FDR, Cohen's d, Wilcoxon, ML
  visualization/  - Persistence diagrams, Betti curves, barcode plots

scripts/
  run_agp_bootstrap_v2.py           - PRIMARY: paired bootstrap on AGP (§2–3)
  run_taxa_sensitivity.py           - Taxa-set sensitivity at N∈{50,80,120} (§3.4)
  run_ibdmdb_bootstrap.py           - IBDMDB validation + 1-sample-per-subject check (§3.5)
  run_classification_benchmark_v2.py - 9-feature-set AUC benchmark (§3.7)
  run_precision_recall_analysis.py  - PR curves at two operating points (§3.7)
  compute_shannon_comparison.py     - Group-level Shannon diversity benchmark
  download_agp.sh                   - AGP data download helper
  download_ibdmdb.sh                - IBDMDB data download helper
  fetch_agp_redbiom.py              - Alternative AGP fetch via redbiom

paper/
  main.tex + sections/  - LaTeX manuscript (compiled to main.pdf)
  references.bib        - Bibliography

figures/          - PNG outputs from analysis scripts
results/          - CSV outputs (bootstrap results, sensitivity, AUC, PR metrics)
tests/            - pytest unit tests for all src modules
configs/          - Pipeline parameters (YAML)
archive/          - Superseded v1 scripts and figures (not used in paper)
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

All analyses use a fixed random seed (42). Run scripts in order after downloading data:

| Script | Paper section | Output |
|--------|--------------|--------|
| `scripts/run_agp_bootstrap_v2.py` | §2 Methods, §3.1–3.3 | `results/agp_bootstrap_v2.csv` |
| `scripts/run_taxa_sensitivity.py` | §3.4 Sensitivity | `results/taxa_sensitivity.csv` |
| `scripts/run_ibdmdb_bootstrap.py` | §3.5 IBDMDB validation | `results/ibdmdb_bootstrap.csv`, `results/ibdmdb_bootstrap_1ps.csv` |
| `scripts/run_classification_benchmark_v2.py` | §3.7 Classification | `results/classification_benchmark_v2.csv` |
| `scripts/run_precision_recall_analysis.py` | §3.7 PR analysis | `results/precision_recall_metrics.csv` |

```bash
python scripts/run_agp_bootstrap_v2.py
python scripts/run_taxa_sensitivity.py
python scripts/run_ibdmdb_bootstrap.py
python scripts/run_classification_benchmark_v2.py
python scripts/run_precision_recall_analysis.py
```

Results are written to `results/` and figures to `figures/`.

---

## Tests

```bash
pytest tests/ -v
```

---

## Data Availability

- AGP: European Nucleotide Archive (accession ERP012803) and via [redbiom](https://redbiom.readthedocs.io)
- IBDMDB: [ibdmdb.org](https://ibdmdb.org)

---

## License

MIT — see [LICENSE](LICENSE).
