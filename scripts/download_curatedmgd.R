#!/usr/bin/env Rscript
# Download and export curatedMetagenomicData for Microbiome-TDA pipeline
#
# Extracts stool samples from multiple studies with relevant metadata
# (mental health, diet, antibiotics, disease state) and exports as TSV
# for the Python pipeline.

suppressPackageStartupMessages({
  library(curatedMetagenomicData)
  library(SummarizedExperiment)
  library(TreeSummarizedExperiment)
})

output_dir <- "data/raw/curatedmgd"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== Downloading curatedMetagenomicData ===\n")

# Target studies with gut-brain relevant metadata
# These contain stool samples with mental health, diet, or lifestyle data
target_studies <- c(
  "HMP_2012",                    # HMP reference (cross-validation)
  "LiJ_2017",                    # Chinese gut metagenomes
  "QinJ_2012",                   # T2D gut metagenomes
  "ZellerG_2014",                # CRC gut metagenomes
  "FengQ_2015",                  # CRC gut metagenomes
  "NielsenHB_2014",              # MetaHIT Danish gut
  "KarlssonFH_2013",             # T2D Swedish gut
  "VogtmannE_2016",              # CRC stool metagenomes
  "ThomasAM_2019_a",             # CRC multi-cohort
  "PasolliBE_2019"               # Global gut metagenomes
)

# Download species-level relative abundance
cat("Fetching species-level abundance data...\n")
all_abundance <- list()
all_metadata <- list()

for (study in target_studies) {
  pattern <- paste0(study, ".relative_abundance")
  cat(sprintf("  Downloading %s...\n", study))

  tryCatch({
    se <- curatedMetagenomicData(pattern, dryrun = FALSE, counts = FALSE)

    if (length(se) > 0) {
      se <- se[[1]]

      # Filter to stool samples only
      if ("body_site" %in% colnames(colData(se))) {
        stool_idx <- colData(se)$body_site == "stool"
        se <- se[, stool_idx]
      }

      if (ncol(se) > 0) {
        # Extract abundance matrix
        abund <- as.data.frame(assay(se))

        # Extract metadata
        meta <- as.data.frame(colData(se))
        meta$study_name <- study

        all_abundance[[study]] <- abund
        all_metadata[[study]] <- meta

        cat(sprintf("    Got %d samples, %d features\n", ncol(se), nrow(se)))
      }
    }
  }, error = function(e) {
    cat(sprintf("    Skipping %s: %s\n", study, e$message))
  })
}

if (length(all_abundance) == 0) {
  stop("No data downloaded. Check network connection and package installation.")
}

# Combine across studies (union of features, fill missing with 0)
cat("\nCombining datasets...\n")
all_features <- unique(unlist(lapply(all_abundance, rownames)))

combined_abundance <- matrix(0,
  nrow = length(all_features),
  ncol = sum(sapply(all_abundance, ncol)),
  dimnames = list(all_features, NULL)
)

col_offset <- 0
sample_names <- character(0)
for (study in names(all_abundance)) {
  abund <- all_abundance[[study]]
  n <- ncol(abund)
  idx <- match(rownames(abund), all_features)
  combined_abundance[idx, (col_offset + 1):(col_offset + n)] <- as.matrix(abund)
  sample_names <- c(sample_names, colnames(abund))
  col_offset <- col_offset + n
}
colnames(combined_abundance) <- sample_names

# Combine metadata
combined_metadata <- do.call(rbind, lapply(all_metadata, function(m) {
  # Keep only common/useful columns
  keep_cols <- intersect(colnames(m), c(
    "sample_id", "study_name", "body_site", "age", "gender",
    "BMI", "country", "disease", "antibiotics_current_use",
    "number_reads", "median_read_length"
  ))
  m[, keep_cols, drop = FALSE]
}))

cat(sprintf("Combined: %d samples, %d features\n",
  ncol(combined_abundance), nrow(combined_abundance)))

# Export
cat("\nExporting to TSV...\n")

abundance_path <- file.path(output_dir, "curated_abundance.tsv")
write.table(combined_abundance, file = abundance_path,
  sep = "\t", quote = FALSE, row.names = TRUE, col.names = TRUE)
cat(sprintf("  Abundance: %s\n", abundance_path))

metadata_path <- file.path(output_dir, "curated_metadata.tsv")
write.table(combined_metadata, file = metadata_path,
  sep = "\t", quote = FALSE, row.names = TRUE, col.names = TRUE)
cat(sprintf("  Metadata: %s\n", metadata_path))

cat("\n=== Done ===\n")
cat(sprintf("Total: %d samples from %d studies\n",
  ncol(combined_abundance), length(all_abundance)))
