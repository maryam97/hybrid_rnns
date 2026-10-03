# Sourced by the job scripts. Edit for your cluster: modules, conda path, env name.
set +u                                   # module/conda scripts can read unset variables
# module load gcc                        # torch.compile (--compile) needs g++ >= 9 on the node
source "$HOME/miniconda3/etc/profile.d/conda.sh"   # or: source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hybrnn
set -u
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1                 # the models are tiny; one thread is fastest
# torch.compile: one worker per allocated core, and a per-job cache (not shared /tmp)
export TORCHINDUCTOR_COMPILE_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.inductor_cache/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
