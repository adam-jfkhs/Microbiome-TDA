# Differential abundance testing using DESeq2
#
# Identifies OTUs with significantly different abundances
# between sample groups (e.g., healthy vs. dysbiotic).

library(DESeq2)
library(phyloseq)

# --- Convert phyloseq to DESeq2 ---
# ps <- readRDS("../data/processed/phyloseq_object.rds")
# dds <- phyloseq_to_deseq2(ps, ~ group)

# --- Run DESeq2 ---
# dds <- DESeq(dds, test = "Wald", fitType = "parametric")
# res <- results(dds, contrast = c("group", "dysbiotic", "healthy"))
# res <- res[order(res$padj), ]

# --- Filter significant results ---
# sig_res <- subset(res, padj < 0.05 & abs(log2FoldChange) > 1)
# print(paste("Significant OTUs:", nrow(sig_res)))
