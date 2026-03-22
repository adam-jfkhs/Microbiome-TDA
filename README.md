# Microbiome-TDA

**Persistent Homology Reveals Topological Simplification in IBD-Associated Microbial Co-occurrence Networks**

*A Paired Bootstrap Analysis of the American Gut Project with Covariate Matching, IBDMDB Validation, and Taxa-Set Sensitivity Testing*

Adam Levine — Independent Researcher · March 2026

---

## Overview

This project applies **persistent homology** (topological data analysis) to Spearman co-occurrence networks built from 16S rRNA and shotgun metagenomics microbiome data. We test whether inflammatory bowel disease (IBD) is associated with measurable differences in the **loop-level topology** (H1 homology) of microbial co-occurrence networks — a structural property that conventional alpha/beta diversity metrics cannot capture.

### Key findings

- IBD-associated microbiomes show systematically lower topological complexity across all six H1 scalar features (Cohen's |*d*| = 0.75–2.38; *p*_FDR < 0.005), fully preserved after age/sex/BMI covariate matching and across taxon-set sizes N in {50, 80, 120}.
- Aitchison-PCoA(10) is the strongest single IBD classifier (LR AUC = 0.734); topology scalars (LR AUC = 0.668) and Shannon diversity (LR AUC = 0.688) perform comparably. Combined Aitchison + Topology + Shannon reaches LR AUC = 0.735.
- In IBDMDB (shotgun metagenomics, n = 106 subjects, 1,338 samples), the full-dataset IBD vs. non-IBD comparison is driven by overrepresentation of active-disease timepoints. A one-sample-per-subject sensitivity check reverses the direction for 5/6 features (|*d*| = 0.61–1.22, IBD > non-IBD). Within-IBD comparisons (calprotectin, HBI, SCCAI) are unaffected by this bias.
- High fecal calprotectin (>=250 ug/g) produces fewer but longer-lived loops — a distinct topological signature of active mucosal inflammation.

---

## System Requirements

- **Python:** 3.10 or higher (tested on 3.11)
- **OS:** Linux or macOS (Windows may work but is untested)
- **RAM:** 8 GB minimum, 16 GB recommended for the classification benchmark
- **Disk:** ~15 GB for AGP data download, ~500 MB for IBDMDB data, ~5 MB for outputs

---

## Installation

### Option A: pip (recommended)

```bash
git clone https://github.com/adam-jfkhs/Microbiome-TDA.git
cd Microbiome-TDA
pip install -r requirements.txt
pip install -e .
```

### Option B: conda

```bash
git clone https://github.com/adam-jfkhs/Microbiome-TDA.git
cd Microbiome-TDA
conda env create -f environment.yml
conda activate microbiome-tda
pip install -e .
```

### Verify installation

```bash
pytest tests/ -v
```

All 6 test files should pass. Tests use synthetic data and do not require downloading any datasets.

---

## Data Download

You need to download two datasets before running the analysis. Both are publicly available with no login required.

### 1. American Gut Project (AGP)

The AGP data (~14 GB compressed) is downloaded from Qiita (study 10317):

```bash
bash scripts/download_agp.sh
```

This creates:
- `data/raw/agp/agp_otu_table.biom` — 16S Deblur-150nt OTU table
- `data/raw/agp/agp_metadata.tsv` — sample metadata (demographics, health, diet)

**Alternative:** If the Qiita download fails, use `redbiom`:
```bash
pip install redbiom
python scripts/fetch_agp_redbiom.py
```

### 2. IBDMDB (HMP2)

The IBDMDB data (~400 MB) is downloaded from the public Globus endpoint:

```bash
bash scripts/download_ibdmdb.sh
```

This creates:
- `data/raw/ibdmdb/hmp2_metadata.csv` — clinical metadata (diagnosis, biomarkers)
- `data/raw/ibdmdb/taxonomic_profiles.tsv.gz` — MetaPhlAn2 species-level profiles

**Source:** [ibdmdb.org](https://ibdmdb.org)

---

## Reproducing the Paper

All analyses use a fixed random seed (42) for full reproducibility. Run scripts **in order** — later scripts depend on outputs from earlier ones.

### Step-by-step

```bash
# Step 1: Primary analysis — paired bootstrap on 3,409 AGP stool samples
#   Compares IBD vs healthy, antibiotics vs none, omnivore vs vegetarian
#   Runtime: ~5-10 minutes
python scripts/run_agp_bootstrap_v2.py

# Step 2: Sensitivity analysis — repeat bootstrap at N = 50, 80, 120 taxa
#   Runtime: ~30 minutes
python scripts/run_taxa_sensitivity.py

# Step 3: Independent validation on IBDMDB (106 subjects, 1,338 samples)
#   Includes 1-sample-per-subject sensitivity check
#   Runtime: ~3 minutes
python scripts/run_ibdmdb_bootstrap.py

# Step 4: Classification benchmark — 9 feature sets, LR + RF, 5-fold CV
#   Caches topological features to results/topo_sample_features.npz
#   Runtime: ~10-15 minutes (first run; <2 min with cache)
python scripts/run_classification_benchmark_v2.py

# Step 5: Precision-recall analysis at two operating points
#   Runtime: ~2 minutes (reads cached features)
python scripts/run_precision_recall_analysis.py
```

### Optional supplementary analyses

```bash
# Loop attribution — which taxa anchor H1 loops?
python scripts/run_loop_attribution.py
python scripts/run_ibdmdb_loop_attribution.py

# Shannon diversity reference
python scripts/compute_shannon_comparison.py
```

### What each script produces

| Step | Script | Paper Section | Output files |
|------|--------|---------------|--------------|
| 1 | `run_agp_bootstrap_v2.py` | Sections 2-3.3 | `results/agp_bootstrap_v2.csv`, `figures/agp_ibd_v2.png`, `figures/agp_antibiotics_v2.png`, `figures/agp_diet_v2.png` |
| 2 | `run_taxa_sensitivity.py` | Section 3.4 | `results/taxa_sensitivity.csv`, `figures/taxa_sensitivity_full.png`, `figures/taxa_sensitivity_matched.png` |
| 3 | `run_ibdmdb_bootstrap.py` | Section 3.5 | `results/ibdmdb_bootstrap.csv`, `results/ibdmdb_bootstrap_1ps.csv`, 6 figures in `figures/` |
| 4 | `run_classification_benchmark_v2.py` | Section 3.7 | `results/classification_benchmark_v2.csv`, `results/topo_sample_features.npz`, `figures/classification_roc_v2.png` |
| 5 | `run_precision_recall_analysis.py` | Section 3.7 | `results/precision_recall_metrics.csv`, `figures/precision_recall_curves.png` |

---

## Repository Structure

```
Microbiome-TDA/
|
|-- src/                          Core library (installed as microbiome_tda package)
|   |-- data/                     AGP/IBDMDB loaders, CLR preprocessing, cohort filtering
|   |-- networks/                 Spearman co-occurrence, correlation distance matrices
|   |-- tda/                      Vietoris-Rips filtration, persistent homology, H1 features
|   |-- analysis/                 Bootstrap, permutation tests, FDR, Cohen's d, ML classifiers
|   |-- visualization/            Persistence diagrams, Betti curves, paper-ready figures
|
|-- scripts/                      Executable analysis pipelines (run these to reproduce)
|   |-- run_agp_bootstrap_v2.py   PRIMARY: paired bootstrap on AGP
|   |-- run_taxa_sensitivity.py   Taxa-set sensitivity at N in {50, 80, 120}
|   |-- run_ibdmdb_bootstrap.py   IBDMDB validation + 1-per-subject check
|   |-- run_classification_benchmark_v2.py   9-feature-set AUC benchmark
|   |-- run_precision_recall_analysis.py     PR curves at two operating points
|   |-- run_loop_attribution.py   Leave-one-out taxa contribution (AGP)
|   |-- run_ibdmdb_loop_attribution.py       Loop attribution (IBDMDB)
|   |-- compute_shannon_comparison.py        Shannon diversity benchmark
|   |-- download_agp.sh           AGP data download
|   |-- download_ibdmdb.sh        IBDMDB data download
|   |-- fetch_agp_redbiom.py      Alternative AGP fetch via redbiom
|
|-- paper/                        LaTeX manuscript
|   |-- main.tex                  Main document
|   |-- sections/                 Section files
|   |-- figures/                  Paper figures
|   |-- references.bib            Bibliography
|
|-- results/                      CSV and NPZ outputs from analysis scripts
|-- figures/                      PNG figure outputs
|-- tests/                        pytest unit tests for all src modules
|-- notebooks/                    Exploratory Jupyter notebooks
|-- archive/                      Superseded v1 scripts (not used in paper)
|-- _future/                      Planned extensions (AMR, mycobiome, etc.)
```

---

## Methodology Summary

### Six H1 topological features extracted per cohort subsample

| Feature | Description |
|---------|-------------|
| `h1_count` | Number of 1-dimensional loops (cycles) in the co-occurrence network |
| `h1_entropy` | Persistence entropy — diversity of loop lifetimes |
| `h1_total_persistence` | Sum of all loop lifetimes |
| `h1_mean_lifetime` | Average loop lifetime |
| `h1_max_lifetime` | Longest-lived loop |
| `max_betti1` | Maximum Betti-1 number (peak simultaneous loop count) |

### Statistical framework

- **Primary p-value:** Sign-flip label-permutation test (500 permutations)
- **Effect size:** Cohen's d on per-iteration bootstrap distributions (200 iterations)
- **Multiple testing correction:** Benjamini-Hochberg FDR
- **Confounding control:** Age/sex/BMI covariate-matched subsets run in parallel
- **Validation:** Independent cohort (IBDMDB), taxa-set sensitivity, 1-per-subject check

### Pipeline flow

```
Raw abundance table
    --> filter low-abundance taxa (top N by prevalence)
    --> CLR transform
    --> Spearman correlation matrix (co-occurrence network)
    --> correlation distance matrix
    --> Vietoris-Rips filtration
    --> persistent homology via Ripser (H0, H1)
    --> extract 6 H1 scalar features
    --> paired bootstrap resampling (200 iterations)
    --> permutation test + Cohen's d + FDR correction
```

---

## Tests

```bash
pytest tests/ -v
```

Tests cover: bootstrap resampling, co-occurrence networks, TDA feature extraction, persistent homology, statistical tests, and synthetic data generation. All tests use generated synthetic data and require no external downloads.

---

## Data Availability

- **AGP:** European Nucleotide Archive (accession ERP012803) and via [Qiita study 10317](https://qiita.ucsd.edu/study/description/10317)
- **IBDMDB:** [ibdmdb.org](https://ibdmdb.org) (HMP2, public Globus endpoint)

---

## Citation

If you use this code or methodology, please cite:

> Levine, A. (2026). Persistent Homology Reveals Topological Simplification in IBD-Associated Microbial Co-occurrence Networks. Zenodo. [DOI pending]

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run `pip install -e .` from the repository root to install the package in development mode.

**AGP download fails or times out**
The AGP BIOM table is ~14 GB. If `download_agp.sh` fails, try the `redbiom` alternative (see Data Download above) or download manually from [Qiita study 10317](https://qiita.ucsd.edu/study/description/10317).

**IBDMDB download fails**
The Globus endpoint occasionally goes down. Wait and retry, or download manually from [ibdmdb.org](https://ibdmdb.org).

**`ripser` or `giotto-tda` installation fails**
These packages require a C++ compiler. On Ubuntu/Debian: `sudo apt install build-essential`. On macOS: `xcode-select --install`.

**Out of memory during classification benchmark**
The benchmark computes per-sample topological features for ~3,400 samples. Reduce memory usage by running with fewer cross-validation folds or reducing the subsample size in the script constants.

---

## License

MIT — see [LICENSE](LICENSE).
