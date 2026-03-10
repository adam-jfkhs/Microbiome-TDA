#!/bin/bash
# Download American Gut Project data via public FTP
#
# Source: ftp://ftp.microbio.me/AmericanGut/latest
# No login required. This is the standard public release.
#
# What we download:
#   - Deblur 150nt BIOM table (the one used in the AGP publication)
#   - Sample metadata (full mapping file)
#
# Reference:
#   McDonald et al. 2018 mSystems — https://doi.org/10.1128/mSystems.00031-18
#   Qiita study page: https://qiita.ucsd.edu/study/description/10317

set -e

DATA_DIR="data/raw/agp"
mkdir -p "$DATA_DIR"

FTP_BASE="ftp://ftp.microbio.me/AmericanGut/latest"

echo "=== Downloading American Gut Project data via FTP ==="
echo "Source: $FTP_BASE"
echo ""

# Deblur-processed 150nt BIOM table (this is the standard cross-study version)
echo "Downloading OTU table (Deblur 150nt)..."
curl -L --retry 4 --retry-delay 5 --progress-bar \
  -o "$DATA_DIR/agp_otu_table.biom" \
  "$FTP_BASE/ag-cleaned.biom"

echo "Downloading metadata..."
curl -L --retry 4 --retry-delay 5 --progress-bar \
  -o "$DATA_DIR/agp_metadata.tsv" \
  "$FTP_BASE/ag-cleaned_md.txt"

# Verify downloads
if [ -f "$DATA_DIR/agp_otu_table.biom" ] && [ -s "$DATA_DIR/agp_otu_table.biom" ]; then
  echo "OTU table: $(du -h "$DATA_DIR/agp_otu_table.biom" | cut -f1)"
else
  echo "WARNING: OTU table download may have failed. Check $DATA_DIR/agp_otu_table.biom"
fi

if [ -f "$DATA_DIR/agp_metadata.tsv" ] && [ -s "$DATA_DIR/agp_metadata.tsv" ]; then
  echo "Metadata: $(du -h "$DATA_DIR/agp_metadata.tsv" | cut -f1)"
  echo "Sample count: $(tail -n +2 "$DATA_DIR/agp_metadata.tsv" | wc -l)"
else
  echo "WARNING: Metadata download may have failed."
fi

echo ""
echo "=== AGP download complete ==="
echo "Output: $DATA_DIR"
echo ""
echo "If the FTP path above fails (FTP paths shift occasionally), use:"
echo "  redbiom fetch samples --context Deblur_2021.09-Illumina-16S-V4-150nt-780f26b \\"
echo "    --output $DATA_DIR/agp_otu_table.biom"
echo "Install redbiom with: pip install redbiom"
