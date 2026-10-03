#!/bin/bash
# Paper-comparable runs: Memory-ANN and vanilla RNN, 1M steps, paper split + metric, 3 seeds each.
# Test accuracy on the 413 held-out blocks -> results/<model>_hs=..._steps=1000000_seed=..._v2.json
#SBATCH --job-name=hybrnn-paper
#SBATCH --array=0-5
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1-12:00:00
#SBATCH --output=results/logs/%x_%A_%a.out
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
# --time: outputs are written only at the end
# --partition / --qos: check with `sinfo -s` and `sacctmgr show qos format=Name,MaxWall`
set -euo pipefail
MODELS=(birnn birnn birnn rnn rnn rnn)
SEEDS=(42 1 2 42 1 2)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
source cluster/env.sh
python -u run_training.py --model "$M" --no-debug --compile --save --seed "$SEED"
