#!/bin/bash
# Re-run the entire analysis pipeline after code fixes.
#
# Prerequisites:
#   1. Raw data downloaded (bash scripts/download_agp.sh && bash scripts/download_ibdmdb.sh)
#   2. Package installed (pip install -e ".[dev]")
#
# This script runs every analysis in the correct order,
# then generates comparison tables and LaTeX macros.
#
# Estimated total time: 4-8 hours depending on hardware.
# The bootstrap scripts (AGP, IBDMDB, taxa sensitivity) are the slowest.

set -e

echo "============================================================"
echo "FULL PIPELINE RE-RUN"
echo "============================================================"
echo ""

# Verify raw data exists
if [ ! -f "data/raw/agp/agp_otu_table.biom" ]; then
    echo "ERROR: AGP data not found. Run first:"
    echo "  bash scripts/download_agp.sh"
    exit 1
fi

if [ ! -f "data/raw/ibdmdb/hmp2_metadata.csv" ]; then
    echo "ERROR: IBDMDB data not found. Run first:"
    echo "  bash scripts/download_ibdmdb.sh"
    exit 1
fi

echo "[1/7] AGP bootstrap (main analysis)..."
python scripts/run_agp_bootstrap_v2.py 2>&1 | tee results/log_agp_bootstrap.txt
echo ""

echo "[2/7] Taxa sensitivity analysis..."
python scripts/run_taxa_sensitivity.py 2>&1 | tee results/log_taxa_sensitivity.txt
echo ""

echo "[3/7] IBDMDB bootstrap..."
python scripts/run_ibdmdb_bootstrap.py 2>&1 | tee results/log_ibdmdb_bootstrap.txt
echo ""

echo "[4/7] Classification benchmark..."
python scripts/run_classification_benchmark_v2.py 2>&1 | tee results/log_classification.txt
echo ""

echo "[5/7] Precision-recall analysis..."
python scripts/run_precision_recall_analysis.py 2>&1 | tee results/log_precision_recall.txt
echo ""

echo "[6/7] Loop attribution (AGP + IBDMDB)..."
python scripts/run_loop_attribution.py 2>&1 | tee results/log_loop_attribution.txt
python scripts/run_ibdmdb_loop_attribution.py 2>&1 | tee results/log_ibdmdb_loop_attribution.txt
echo ""

echo "[7/7] Comparing old vs new results & generating LaTeX macros..."
python scripts/compare_results.py
python scripts/generate_latex_macros.py
echo ""

echo "============================================================"
echo "DONE — All analyses complete."
echo ""
echo "Next steps:"
echo "  1. Review scripts/compare_results.py output above"
echo "  2. Update paper tables with new numbers (see paper/generated_macros.tex)"
echo "  3. Recompile LaTeX"
echo "============================================================"
