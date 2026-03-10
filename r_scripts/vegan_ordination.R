# Ordination and beta-diversity analysis using vegan
#
# Computes distance matrices, performs PCoA/NMDS,
# and runs PERMANOVA tests.

library(vegan)
library(ape)
library(ggplot2)

# --- Load data ---
# otu_table <- read.csv("../data/processed/otu_table.csv", row.names = 1)
# metadata <- read.csv("../data/processed/metadata.csv", row.names = 1)

# --- Bray-Curtis distance ---
# bc_dist <- vegdist(t(otu_table), method = "bray")

# --- PCoA ---
# pcoa_res <- pcoa(bc_dist)
# biplot(pcoa_res)

# --- NMDS ---
# nmds_res <- metaMDS(t(otu_table), distance = "bray", k = 2)
# plot(nmds_res, type = "t")

# --- PERMANOVA ---
# adonis_res <- adonis2(bc_dist ~ group, data = metadata, permutations = 999)
# print(adonis_res)

# --- Beta dispersion test ---
# betadisp_res <- betadisper(bc_dist, metadata$group)
# permutest(betadisp_res)
