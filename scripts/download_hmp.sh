#!/bin/bash
set -e

DATA_DIR="data/raw/hmp"
mkdir -p $DATA_DIR

echo "=== Downloading HMP 16S OTU tables ==="

# HMP1 — 16S v3-v5 region OTU table (stool samples)
curl -L -o $DATA_DIR/hmp1_otu_table.biom \
  "https://downloads.hmpdacc.org/dacc/hhs/genome/microbiome/wgs/analysis/hmqcp/v35/otu_table_psn_v35.biom"

# HMP1 metadata — subject demographics + sample info
curl -L -o $DATA_DIR/hmp1_metadata.tsv \
  "https://downloads.hmpdacc.org/dacc/hhs/genome/microbiome/wgs/analysis/hmqcp/v35/map_v35.txt"

echo "=== HMP Phase 1 data downloaded ==="
