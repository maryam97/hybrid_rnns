#!/bin/bash
# Whole-data Laplace: marglik training on all 4,134 blocks, then the exact-GGN evidence.
# -> results/<model>_marglik_..._fit=all_seed=<seed>_v2.json and results/<NAME>_posthoc_fit=all_full.json
#SBATCH --job-name=hybrnn-laplace
#SBATCH --array=0-1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
# --time: ~15-25 h with --compile; outputs only at the end
# --partition / --qos: check with `sinfo -s` and `sacctmgr show qos format=Name,MaxWall`
set -euo pipefail
SEED=42
MODELS=(birnn rnn)
BATCHES=(128 64)          # paper batch sizes (Memory-ANN, vanilla RNN)
EPOCHS=(30303 15385)      # ~1M SGD steps on 4,134 blocks: 33 and 65 batches per epoch
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
NAME="${M}_marglik_all_seed${SEED}_v2"
source cluster/env.sh

# 1. Marglik training (Kron prior updates every 50 epochs after a 20% burn-in).
python -u run_laplace_training.py --model "$M" --no-debug --compile --seed "$SEED" \
    --batch-size "${BATCHES[$SLURM_ARRAY_TASK_ID]}" --epochs "${EPOCHS[$SLURM_ARRAY_TASK_ID]}" \
    --marglik-freq 50 --eval-freq 50 --save "trained_models/$NAME.pt"

# 2. Evidence from the exact GGN (FuncGGN) on the same blocks; test metrics are in-sample.
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -u run_laplace.py --model "$M" --checkpoint "trained_models/$NAME.pt" --fit-on all
