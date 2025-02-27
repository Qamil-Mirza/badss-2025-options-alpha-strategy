#!/bin/bash
set -e  # Exit on error

# Ensure Conda is loaded in the shell
source ~/anaconda3/etc/profile.d/conda.sh
conda activate badss

# Navigate and run MILP model
echo "Running pnlmin model..."
cd models && python pnlmin.py && cd ..

# Extract results
echo "Extracting results..."
python extract_results.py

echo "Pipeline completed successfully."
