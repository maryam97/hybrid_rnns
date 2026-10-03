PyTorch code for [hybrid_rnn](https://github.com/google-deepmind/hybrid_rnns_reward_learning/)

## Notes (2026-10-02)

- **Accuracy** is the paper's: mean over test blocks of `exp(-NLL_block / 150)` (`acc_paper`). The older pooled
  `exp(-total NLL / valid trials)` (`acc_pooled`) reads about 2 points lower and is not comparable to the paper.
- **Split** is the author's (`hybrid_rnns_pytorch/splits/open_source_mapping.csv`, 3302/419/413 blocks). The old
  seed-1356 split only matched the block counts, and `--seed` used to change it.
- **Rewards** must be 0–1: `openSourceRawDataset.csv` (OSF dw7f6) already is; the v2 file (OSF sa62j) is 0–100 and
  `load_osf_dataframe` divides it.
- Results written before this change (`results/*.json` without `_v2`) use the old split and the pooled metric.
- **Laplace** (`run_laplace_training.py`, `run_laplace.py`) fits the whole data by default: all 4,134 kept
  blocks, so its test numbers are in-sample and its output is the log evidence. Paper-comparable held-out
  accuracies come from `run_training.py`. Report the evidence from the exact GGN (`run_laplace.py`, default
  `--hessian full`); Kron is fine for training but biased for the recurrent layers. Last-layer Laplace
  (`kron_ll`, `--subset last_layer`) is refused for the BiRNN: laplace-torch drops its value stream there.
- Background and the JAX comparison: `comparison/ablation_results/README.md`.
