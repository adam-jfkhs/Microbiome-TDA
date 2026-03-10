# Install R packages for microbiome analysis

# Bioconductor packages
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c(
    "phyloseq",                  # Microbiome data handling
    "DESeq2",                    # Differential abundance
    "microbiome",                # Microbiome analysis utilities
    "curatedMetagenomicData"     # Curated datasets
))

# CRAN packages
install.packages(c(
    "vegan",             # Community ecology (diversity, ordination)
    "ape",               # Phylogenetic analysis
    "ggplot2",           # Visualization
    "TDAstats",          # TDA in R (for cross-validation with Python)
    "reticulate"         # Python-R bridge
))
