#!/usr/bin/env python3
"""
Fetch American Gut Project data using redbiom.

redbiom provides programmatic access to Qiita's public microbiome data
without needing to log in through the web interface.

Install: pip install redbiom
Docs:    https://github.com/biocore/redbiom

This script fetches stool samples from AGP study 10317 using the
Deblur 150nt context (the standard AGP processing).
"""

import subprocess
import sys
import os

OUTPUT_DIR = "data/raw/agp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The Deblur 150nt context is the standard for cross-study comparison.
# Context name from: redbiom search contexts "American Gut"
CONTEXT = "Deblur_2021.09-Illumina-16S-V4-150nt-780f26b"


def run(cmd, desc):
    print(f"{desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: {result.stderr}")
        sys.exit(1)
    print(f"  OK: {result.stdout.strip()[:200]}")


def main():
    # Check redbiom is installed
    result = subprocess.run("python3 -m redbiom --version",
                            shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("redbiom not found. Install with: pip install redbiom")
        sys.exit(1)

    print("=== Fetching AGP data via redbiom ===")
    print(f"Context: {CONTEXT}")
    print(f"Output:  {OUTPUT_DIR}")
    print()

    biom_path = os.path.join(OUTPUT_DIR, "agp_otu_table.biom")
    meta_path = os.path.join(OUTPUT_DIR, "agp_metadata.tsv")

    # Step 1: Find all stool sample IDs from study 10317
    print("Finding stool sample IDs from study 10317...")
    ids_path = os.path.join(OUTPUT_DIR, "agp_stool_ids.txt")
    run(
        f'redbiom search metadata "where sample_type == \'feces\' and qiita_study_id == 10317" '
        f'> {ids_path}',
        "Querying stool sample IDs"
    )

    with open(ids_path) as f:
        ids = [line.strip() for line in f if line.strip()]
    print(f"Found {len(ids)} stool samples")

    if not ids:
        print("No stool samples found. The context or query may have changed.")
        print("Try: redbiom search contexts to list available contexts")
        sys.exit(1)

    # Step 2: Fetch BIOM table
    run(
        f'redbiom fetch samples '
        f'--context {CONTEXT} '
        f'--from {ids_path} '
        f'--output {biom_path}',
        f"Fetching BIOM table to {biom_path}"
    )

    # Step 3: Fetch metadata
    run(
        f'redbiom fetch sample-metadata '
        f'--from {ids_path} '
        f'--context {CONTEXT} '
        f'--output {meta_path}',
        f"Fetching metadata to {meta_path}"
    )

    print()
    print("=== Done ===")
    print(f"BIOM table: {biom_path}")
    print(f"Metadata:   {meta_path}")
    print()
    print("Key metadata columns for the pipeline:")
    print("  sample_type, age, sex, bmi_corrected, diet_type,")
    print("  antibiotic_history, ibd, mental_illness_type")


if __name__ == "__main__":
    main()
