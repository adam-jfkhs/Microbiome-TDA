# Microbiome-TDA: Persistent Homology Reveals Microbial Community Topologies Linked to Serotonin and Neuropeptide Production

Applying persistent homology to bacterial co-occurrence networks to identify topological signatures of neurotransmitter-producing microbial community configurations, and characterizing how mycobiome disruption and antimicrobial resistance fragment these structures.

## Overview

This project constructs co-occurrence networks from 16S rRNA and metagenomic microbiome data (Human Microbiome Project, American Gut Project, curatedMetagenomicData), applies persistent homology to detect topological features (connected components, loops, voids), and links these topological signatures to:

- **Neurotransmitter production capacity** — serotonin (5-HT), GABA, dopamine precursors
- **Mycobiome disruption** — how fungal co-colonization fragments bacterial network topology
- **AMR gene carriers** — antibiotic-resistant taxa as topological disruptors
- **Gut permeability** — intestinal barrier function linked to community shape
- **Gut-brain axis outcomes** — mood, cognition, and neurological self-reports

## Repository Structure

```
src/                - Python source modules
  data/             - Download, loading, preprocessing
  networks/         - Co-occurrence network construction, distance matrices
  tda/              - Filtration, persistent homology, feature extraction, regime detection
  analysis/         - Correlation, statistics, ML classification
  visualization/    - Persistence diagrams, Betti curves, networks, paper figures
  neurotransmitter/ - Taxa-to-neurotransmitter pathway mapping
  mycobiome/        - Fungal disruption topology analysis
  amr/              - AMR gene carrier disruption analysis
notebooks/          - Jupyter notebooks for each analysis phase
scripts/            - Data download and setup scripts
r_scripts/          - R-based analyses (phyloseq, DESeq2, curatedMetagenomicData)
configs/            - Pipeline parameters and dataset configuration
paper/              - LaTeX manuscript and figures
tests/              - Unit tests
```

## Setup

### Python environment

```bash
# Option 1: venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option 2: conda
conda env create -f environment.yml
conda activate microbiome-tda
```

### R environment

```bash
Rscript scripts/setup_r_env.R
```

### Download data

```bash
make data
# or manually:
bash scripts/download_hmp.sh
bash scripts/download_agp.sh
Rscript scripts/download_curatedmgd.R
```

## Analysis Pipeline

1. **Phase 1** — Data acquisition and exploration (`notebooks/01_data_exploration.ipynb`)
2. **Phase 2** — Co-occurrence network construction (`notebooks/02_network_construction.ipynb`)
3. **Phase 3** — Persistent homology pipeline (`notebooks/03_tda_pipeline.ipynb`)
4. **Phase 4** — Neurotransmitter pathway topology (`notebooks/04_neurotransmitter_topology.ipynb`)
5. **Phase 5** — Mycobiome and AMR disruption analysis (`notebooks/05_disruption_analysis.ipynb`)
6. **Phase 6** — Regime detection via sliding windows (`notebooks/06_regime_detection.ipynb`)
7. **Phase 7** — Biomarker correlation and gut-brain axis (`notebooks/07_gutbrain_correlation.ipynb`)

## Key Dependencies

- **TDA**: giotto-tda, ripser, persim, gudhi
- **Microbiome**: biom-format, scikit-bio
- **Networks**: networkx
- **Statistics**: statsmodels, pingouin, scipy
- **ML**: scikit-learn

## License

See [LICENSE](LICENSE) for details.
