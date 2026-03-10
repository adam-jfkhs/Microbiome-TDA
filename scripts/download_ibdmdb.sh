#!/usr/bin/env bash
# Download IBDMDB (HMP2) metagenomics data from the public Globus endpoint.
# No authentication required — the endpoint is open-access.
#
# Usage: bash scripts/download_ibdmdb.sh

set -euo pipefail

DEST="data/raw/ibdmdb"
mkdir -p "$DEST"

BASE="https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb"

echo "Downloading IBDMDB metadata..."
curl -L --retry 4 --retry-delay 4 \
  "$BASE/metadata/hmp2_metadata_2018-08-20.csv" \
  -o "$DEST/hmp2_metadata.csv"

echo "Downloading MetaPhlAn2 taxonomic profiles..."
curl -L --retry 4 --retry-delay 4 \
  "$BASE/products/HMP2/MGX/2017-08-12/taxonomic_profiles.tsv.gz" \
  -o "$DEST/taxonomic_profiles.tsv.gz"

echo "Done. Files written to $DEST/"
ls -lh "$DEST/"
