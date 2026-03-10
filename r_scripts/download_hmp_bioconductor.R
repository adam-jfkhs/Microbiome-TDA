#!/usr/bin/env Rscript
# Download HMP 16S V3-V5 stool samples using the HMP16SData Bioconductor package
#
# This is the cleanest path: one function call gets you the full
# SummarizedExperiment with OTU table, taxonomy, phylogenetic tree, and metadata.
#
# Reference: https://waldronlab.io/HMP16SData/
# Install: BiocManager::install("HMP16SData")

suppressPackageStartupMessages({
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

  if (!requireNamespace("HMP16SData", quietly = TRUE)) {
    cat("Installing HMP16SData from Bioconductor...\n")
    BiocManager::install("HMP16SData", ask = FALSE)
  }

  library(HMP16SData)
  library(SummarizedExperiment)
})

output_dir <- "data/raw/hmp"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== Downloading HMP 16S V3-V5 data via HMP16SData package ===\n")
cat("Source: https://waldronlab.io/HMP16SData/\n\n")

# V35() returns ALL body sites. We'll filter to stool below.
cat("Fetching V3-V5 SummarizedExperiment (this downloads ~200MB on first run, cached after)...\n")
hmp_se <- V35()

cat(sprintf("Full dataset: %d samples, %d OTUs\n", ncol(hmp_se), nrow(hmp_se)))

# Filter to stool samples only
meta <- as.data.frame(colData(hmp_se))
cat("Body sites available:\n")
print(table(meta$HMP_BODY_SUBSITE))

stool_idx <- meta$HMP_BODY_SUBSITE == "Stool"
hmp_stool <- hmp_se[, stool_idx]
cat(sprintf("\nStool samples: %d\n", ncol(hmp_stool)))

# --- Export OTU counts ---
cat("\nExporting OTU table (samples x OTUs)...\n")
otu_matrix <- t(assay(hmp_stool))  # samples as rows
otu_df <- as.data.frame(otu_matrix)
otu_path <- file.path(output_dir, "hmp1_otu_table.tsv")
write.table(otu_df, file = otu_path, sep = "\t", quote = FALSE,
            row.names = TRUE, col.names = TRUE)
cat(sprintf("  OTU table: %s (%d samples x %d OTUs)\n",
            otu_path, nrow(otu_df), ncol(otu_df)))

# --- Export metadata ---
cat("Exporting metadata...\n")
meta_stool <- as.data.frame(colData(hmp_stool))
# Normalize key column names to match the pipeline's expected format
names(meta_stool)[names(meta_stool) == "HMP_BODY_SUBSITE"] <- "body_site"
names(meta_stool)[names(meta_stool) == "SEX"] <- "sex"
names(meta_stool)[names(meta_stool) == "RUN_CENTER"] <- "run_center"
meta_stool$body_site <- tolower(meta_stool$body_site)  # "Stool" -> "stool"

meta_path <- file.path(output_dir, "hmp1_metadata.tsv")
write.table(meta_stool, file = meta_path, sep = "\t", quote = FALSE,
            row.names = TRUE, col.names = TRUE)
cat(sprintf("  Metadata: %s (%d samples, %d columns)\n",
            meta_path, nrow(meta_stool), ncol(meta_stool)))

# --- Export taxonomy ---
cat("Exporting taxonomy...\n")
# OTU IDs are of the form: k__Bacteria;p__Firmicutes;c__...;g__...;s__...
# Parse into standard columns
otu_ids <- rownames(hmp_stool)
parse_otu_taxonomy <- function(otu_id) {
  parts <- strsplit(otu_id, ";")[[1]]
  get_level <- function(prefix) {
    match <- parts[startsWith(trimws(parts), prefix)]
    if (length(match) == 0) return("")
    gsub(prefix, "", trimws(match[1]))
  }
  list(
    Kingdom = get_level("k__"),
    Phylum  = get_level("p__"),
    Class   = get_level("c__"),
    Order   = get_level("o__"),
    Family  = get_level("f__"),
    Genus   = get_level("g__"),
    Species = get_level("s__")
  )
}

tax_rows <- lapply(otu_ids, parse_otu_taxonomy)
tax_df <- do.call(rbind, lapply(tax_rows, as.data.frame))
rownames(tax_df) <- otu_ids

tax_path <- file.path(output_dir, "taxonomy.tsv")
write.table(tax_df, file = tax_path, sep = "\t", quote = FALSE,
            row.names = TRUE, col.names = TRUE)
cat(sprintf("  Taxonomy: %s (%d OTUs)\n", tax_path, nrow(tax_df)))

# Quick summary
cat("\n=== HMP download complete ===\n")
cat(sprintf("  Stool samples:  %d\n", ncol(hmp_stool)))
cat(sprintf("  OTUs:           %d\n", nrow(hmp_stool)))
cat(sprintf("  Output dir:     %s\n", output_dir))
cat(sprintf("  Files:          hmp1_otu_table.tsv, hmp1_metadata.tsv, taxonomy.tsv\n"))
cat("\nNote: the pipeline loaders.py expects TSV format when loading real HMP data.\n")
cat("Update load_hmp() to use pd.read_csv(..., sep='\\t') instead of load_table().\n")
