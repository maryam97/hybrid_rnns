# Cluster runs (SLURM)

**1. Copy the working tree** (the fixes are not committed yet, so a git clone would run the old code):
```bash
rsync -av --exclude .git --exclude __pycache__ --exclude results --exclude trained_models --exclude comparison \
      ./ <cluster>:hybrid_rnns/
```

**2. Environment (once)**, CPU-only torch (the default Linux wheel pulls ~3.5 GB of CUDA):
```bash
conda create -n hybrnn -c conda-forge --override-channels python=3.12 -y && conda activate hybrnn
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r cluster/requirements.txt
```
Edit `cluster/env.sh` (conda path, `module load gcc` for `--compile`) and the `--partition` / `--qos` lines in
both `.sbatch` files; check the QOS allows the requested `--time`.

**3. Test speed and `--compile` on a compute node** (~40 min; writes `..._steps=2001_...`, no clash):
```bash
srun -c2 --mem=4G -t 0:45:00 bash -c 'source cluster/env.sh && python -u run_training.py --model birnn --no-debug --compile --steps 2001'
```
`secs_per_500_steps` × 2000 ≈ the full run. If compile fails, drop `--compile` from the `.sbatch` files and
raise `--time` (about 2.6× slower).

**4. Submit:** `bash cluster/submit.sh` (downloads the data if missing).

| Job | What | Output | Time per task (estimate) |
|---|---|---|---|
| `paper_runs.sbatch` (6 tasks) | Memory-ANN + vanilla RNN, seeds 42/1/2, 1M steps | held-out test accuracy, paper metric | 6–12 h (BiRNN), less for RNN |
| `laplace_runs.sbatch` (2 tasks) | marglik on all 4,134 blocks (~1M steps), then exact-GGN evidence | log evidence; test metrics in-sample | 15–25 h |

Paper targets (413 test blocks): Memory-ANN NLL/block 61.3, acc 68.3%; vanilla RNN 61.7, 68.1%.
Use `test_acc_paper` (best checkpoint on validation NLL) or `last_step_test_acc_paper`.

Copy results back into a separate folder (`rsync -av <cluster>:hybrid_rnns/results/ results_cluster/`):
seed-42 file names are the same as local runs.
