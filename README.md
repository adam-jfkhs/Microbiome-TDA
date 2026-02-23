# Microbiome-TDA: Topological Signatures of Microbiome Dysbiosis

Applying Topological Data Analysis (TDA) — specifically persistent homology — to characterize structural transitions in gut microbiome co-occurrence networks associated with dysbiosis.

## Overview

This project constructs co-occurrence networks from 16S rRNA microbiome data (Human Microbiome Project + American Gut Project), applies persistent homology to detect topological features (connected components, loops, voids), and uses these topological signatures to identify regime shifts associated with disease states and lifestyle factors.

## Repository Structure

```
src/           - Python source modules (data, networks, TDA, analysis, visualization)
notebooks/     - Jupyter notebooks for each analysis phase
scripts/       - Data download and setup scripts
r_scripts/     - R-based analyses (phyloseq, DESeq2, vegan)
configs/       - Pipeline parameters and dataset configuration
paper/         - LaTeX manuscript and figures
tests/         - Unit tests
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
```

## Analysis Pipeline

1. **Phase 1** - Data acquisition and exploration (`notebooks/01_data_exploration.ipynb`)
2. **Phase 2** - Co-occurrence network construction (`notebooks/02_network_construction.ipynb`)
3. **Phase 3** - Persistent homology pipeline (`notebooks/03_tda_pipeline.ipynb`)
4. **Phase 4** - Regime detection via sliding windows (`notebooks/04_regime_detection.ipynb`)
5. **Phase 5** - Biomarker correlation analysis (`notebooks/05_biomarker_correlation.ipynb`)

## Key Dependencies

- **TDA**: giotto-tda, ripser, persim, gudhi
- **Microbiome**: biom-format, scikit-bio
- **Networks**: networkx
- **Statistics**: statsmodels, pingouin, scipy

## License

See [LICENSE](LICENSE) for details.
