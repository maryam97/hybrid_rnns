#!/bin/bash
# Run from anywhere: bash cluster/submit.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs trained_models hybrid_rnns_pytorch/data
DATA=hybrid_rnns_pytorch/data/openSourceRawDataset.csv     # OSF dw7f6 (rewards 0-1)
[ -f "$DATA" ] || { curl -fL --retry 3 -o "$DATA.part" https://osf.io/download/dw7f6/ && mv "$DATA.part" "$DATA"; }
sbatch cluster/paper_runs.sbatch
sbatch cluster/laplace_runs.sbatch
