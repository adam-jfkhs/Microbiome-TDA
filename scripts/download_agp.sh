#!/bin/bash
# Download American Gut Project data from Qiita (study 10317).
#
# The original FTP endpoint (ftp://ftp.microbio.me/AmericanGut/latest) is
# no longer reliably available.  This script uses Qiita's public download
# API instead, which serves the same Deblur-processed BIOM tables and
# sample metadata as zip archives.
#
# No login required — Qiita study 10317 is fully public.
#
# Reference:
#   McDonald et al. 2018 mSystems — https://doi.org/10.1128/mSystems.00031-18
#   Qiita study page: https://qiita.ucsd.edu/study/description/10317

set -e

DATA_DIR="data/raw/agp"
mkdir -p "$DATA_DIR"

QIITA_BASE="https://qiita.ucsd.edu/public_download"

echo "=== Downloading American Gut Project data from Qiita ==="
echo ""

# ── BIOM table ───────────────────────────────────────────────────────────────
echo "Downloading 16S BIOM table (this may take a few minutes)..."
curl -L --retry 4 --retry-delay 5 --progress-bar \
  -o "$DATA_DIR/agp_biom.zip" \
  "${QIITA_BASE}/?data=biom&study_id=10317&data_type=16S"

if [ -f "$DATA_DIR/agp_biom.zip" ] && [ -s "$DATA_DIR/agp_biom.zip" ]; then
  echo "Extracting BIOM archive..."
  unzip -o -d "$DATA_DIR/biom_extract" "$DATA_DIR/agp_biom.zip"

  # Find the Deblur 150nt BIOM — prefer exact match, fall back to first .biom
  BIOM_FILE=$(find "$DATA_DIR/biom_extract" -name "*.biom" | grep -i -E "deblur|150nt" | head -1)
  if [ -z "$BIOM_FILE" ]; then
    BIOM_FILE=$(find "$DATA_DIR/biom_extract" -name "*.biom" | head -1)
  fi
  if [ -n "$BIOM_FILE" ]; then
    cp "$BIOM_FILE" "$DATA_DIR/agp_otu_table.biom"
    echo "OTU table: $(du -h "$DATA_DIR/agp_otu_table.biom" | cut -f1)"
  else
    echo "WARNING: No .biom file found in archive. Listing contents:"
    find "$DATA_DIR/biom_extract" -type f
  fi
else
  echo "WARNING: BIOM download may have failed."
fi

# ── Sample metadata ──────────────────────────────────────────────────────────
echo ""
echo "Downloading sample metadata..."
curl -L --retry 4 --retry-delay 5 --progress-bar \
  -o "$DATA_DIR/agp_metadata.zip" \
  "${QIITA_BASE}/?data=sample_information&study_id=10317"

if [ -f "$DATA_DIR/agp_metadata.zip" ] && [ -s "$DATA_DIR/agp_metadata.zip" ]; then
  echo "Extracting metadata archive..."
  unzip -o -d "$DATA_DIR/meta_extract" "$DATA_DIR/agp_metadata.zip"

  META_FILE=$(find "$DATA_DIR/meta_extract" \( -name "*.tsv" -o -name "*.txt" \) | head -1)
  if [ -n "$META_FILE" ]; then
    cp "$META_FILE" "$DATA_DIR/agp_metadata.tsv"
    echo "Metadata: $(du -h "$DATA_DIR/agp_metadata.tsv" | cut -f1)"
    echo "Sample count: $(tail -n +2 "$DATA_DIR/agp_metadata.tsv" | wc -l)"
  else
    echo "WARNING: No .tsv/.txt file found in metadata archive. Listing contents:"
    find "$DATA_DIR/meta_extract" -type f
  fi
else
  echo "WARNING: Metadata download may have failed."
fi

# ── Cleanup temp files ───────────────────────────────────────────────────────
rm -rf "$DATA_DIR/biom_extract" "$DATA_DIR/meta_extract"
rm -f "$DATA_DIR/agp_biom.zip" "$DATA_DIR/agp_metadata.zip"

echo ""
echo "=== AGP download complete ==="
echo "Output directory: $DATA_DIR"
echo "Expected files:"
echo "  $DATA_DIR/agp_otu_table.biom"
echo "  $DATA_DIR/agp_metadata.tsv"
echo ""

# Verify both files exist
if [ ! -s "$DATA_DIR/agp_otu_table.biom" ] || [ ! -s "$DATA_DIR/agp_metadata.tsv" ]; then
  echo "ERROR: One or both files are missing or empty." >&2
  echo "" >&2
  echo "Alternative: use redbiom (pip install redbiom):" >&2
  echo "  redbiom fetch samples \\" >&2
  echo "    --context Deblur-Illumina-16S-V4-150nt-780653 \\" >&2
  echo "    --output $DATA_DIR/agp_otu_table.biom" >&2
  echo "" >&2
  echo "Or download manually from:" >&2
  echo "  https://qiita.ucsd.edu/study/description/10317" >&2
  exit 1
fi
