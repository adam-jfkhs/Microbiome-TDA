# Phyloseq-based diversity analysis for microbiome data
#
# This script loads OTU tables and metadata, creates phyloseq objects,
# and computes alpha and beta diversity metrics.

library(phyloseq)
library(ggplot2)
library(vegan)

# --- Load data ---
# Adjust paths as needed
# otu_table <- read.csv("../data/processed/otu_table.csv", row.names = 1)
# metadata <- read.csv("../data/processed/metadata.csv", row.names = 1)

# --- Create phyloseq object ---
# ps <- phyloseq(
#   otu_table(as.matrix(otu_table), taxa_are_rows = TRUE),
#   sample_data(metadata)
# )

# --- Alpha diversity ---
# alpha_div <- estimate_richness(ps, measures = c("Shannon", "Simpson", "Observed"))
# plot_richness(ps, x = "group", measures = c("Shannon", "Simpson"))

# --- Beta diversity ---
# ord <- ordinate(ps, method = "PCoA", distance = "bray")
# plot_ordination(ps, ord, color = "group") +
#   theme_minimal() +
#   ggtitle("PCoA of Bray-Curtis Distances")
