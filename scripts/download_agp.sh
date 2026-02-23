#!/bin/bash
set -e

DATA_DIR="data/raw/agp"
mkdir -p $DATA_DIR

echo "=== Downloading American Gut Project data ==="

# Deblurred 150nt BIOM table from Qiita study 10317
curl -L -o $DATA_DIR/agp_otu_table.biom \
  "https://qiita.ucsd.edu/public_artifact_download/?artifact_id=77316"

# Sample metadata
curl -L -o $DATA_DIR/agp_metadata.tsv \
  "https://qiita.ucsd.edu/public_download/?data=sample_information&study_id=10317"

echo "=== AGP data downloaded ==="
