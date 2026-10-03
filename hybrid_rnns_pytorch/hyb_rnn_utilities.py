"""Utilities — PyTorch port of hybrid_rnns_reward_learning/hyb_rnn_utilities.py.

Tensor layout used everywhere: (n_blocks, n_trials, n_actions + 2) float32,
columns [one-hot action (n_actions) | reward (1) | valid mask (1)].
Missed trials have an all-zero action vector, reward 0 and valid == 0.
"""

import os

import numpy as np
import pandas as pd
import torch

# The author's split (osf_s_id, osf_block, split); kept outside data/, which is gitignored.
from .rnn_config import DEFAULT_SPLIT_FILE  # noqa: F401  (re-exported)

MAX_MISSED = 15  # paper exclusion: drop blocks with >15 of 150 trials missed


def load_osf_dataframe(source, reward_scale=None) -> pd.DataFrame:
    """Read either OSF file and return one schema with rewards on a 0-1 scale.

    dw7f6 (openSourceRawDataset.csv):    reward 0-1,   missed action = -1,  payout_1..4
    sa62j (openSourceRawDataset_v2.csv): reward 0-100, missed action = NaN, payout_0..3

    reward_scale: divisor for reward/payout columns; None = auto (100 if max > 1).
    """
    df = pd.read_csv(source, low_memory=False) if isinstance(source, str) else source.copy()
    if 'payout_0' in df.columns:  # sa62j names payouts 0..3
        df = df.rename(columns={f'payout_{i}': f'payout_{i + 1}' for i in range(4)})
    df['action'] = df['action'].fillna(-1)
    df = df.astype({'s_id': int, 'block': int, 'trial_id': int})
    payout_cols = [c for c in (f'payout_{i + 1}' for i in range(4)) if c in df.columns]
    if reward_scale is None:
        reward_scale = 100.0 if df['reward'].max() > 1.0 else 1.0
    if reward_scale != 1.0:
        df['reward'] = df['reward'] / reward_scale
        df[payout_cols] = df[payout_cols] / reward_scale
    _check_rewards(df)
    return df


def _check_rewards(hum_dat) -> None:
    lo, hi = hum_dat['reward'].min(), hum_dat['reward'].max()
    if lo < 0.0 or hi > 1.0:
        raise ValueError(f'rewards must be in [0, 1], got [{lo}, {hi}]. '
                         'Load the CSV with load_osf_dataframe() (OSF sa62j is 0-100).')


def _block_tensor(hum_dat: pd.DataFrame, n_actions: int, n_trials: int):
    """Return (tensor (n_blocks, n_trials, n_actions + 2), ids DataFrame[s_id, block])."""
    hum_dat = hum_dat.sort_values(['s_id', 'block', 'trial_id']).reset_index(drop=True)
    sizes = hum_dat.groupby(['s_id', 'block'], sort=True).size()
    if not (sizes == n_trials).all():
        raise ValueError(f'{int((sizes != n_trials).sum())} blocks do not have {n_trials} trials')
    action = hum_dat['action'].fillna(-1).to_numpy()
    valid = action >= 0                                               # -1 / NaN -> missed
    onehot = (action[:, None] == np.arange(n_actions)) & valid[:, None]
    flat = np.column_stack([onehot, hum_dat['reward'].to_numpy(), valid]).astype(np.float32)
    tensor = torch.from_numpy(flat).view(-1, n_trials, n_actions + 2)
    return tensor, sizes.index.to_frame(index=False)


def _kept_blocks(tensor: torch.Tensor, n_actions: int, n_trials: int) -> np.ndarray:
    return (tensor[:, :, n_actions + 1].sum(dim=1) >= n_trials - MAX_MISSED).numpy()


def format_data_for_model_training(
    hum_dat: pd.DataFrame,
    n_actions: int = 4,
    split_file: str | None = DEFAULT_SPLIT_FILE,
    return_ids: bool = False,
) -> dict:
    """Train/valid/test tensors on the paper's split (3302/419/413 blocks).

    With split_file (the author's open_source_mapping.csv), each block gets the
    author's label; blocks not listed are the 24 with >15 missed trials and are
    dropped. Without it, the rule that file encodes is used: participants in CSV
    ROW ORDER, first 10% test, next 10% valid, rest train, then the >15-missed
    exclusion. This reproduces the file block for block, so do not sort or
    shuffle `hum_dat` before calling.

    `hum_dat` must have rewards on a 0-1 scale (use load_osf_dataframe).
    """
    _check_rewards(hum_dat)
    n_trials = hum_dat['trial_id'].nunique()
    subs_in_file_order = pd.unique(hum_dat['s_id'])                # before any sorting
    tensor, ids = _block_tensor(hum_dat, n_actions, n_trials)
    kept = _kept_blocks(tensor, n_actions, n_trials)

    if split_file is not None and os.path.exists(split_file):
        mapping = pd.read_csv(split_file).rename(columns={'osf_s_id': 's_id', 'osf_block': 'block'})
        mapping = mapping.astype({'s_id': int, 'block': int})
        labels = ids.merge(mapping[['s_id', 'block', 'split']], on=['s_id', 'block'], how='left')['split']
        # Every unlisted block must be an excluded one; otherwise the ids are not OSF ids.
        if (labels.isna().to_numpy() & kept).any():
            raise ValueError(f'dataset s_id/block do not match {split_file}; '
                             'pass split_file=None to use the file-order split')
        source = f'author split file ({os.path.basename(split_file)})'
    else:
        n_held = len(subs_in_file_order) // 10
        sub_split = {**{s: 'test' for s in subs_in_file_order[:n_held]},
                     **{s: 'valid' for s in subs_in_file_order[n_held:2 * n_held]},
                     **{s: 'train' for s in subs_in_file_order[2 * n_held:]}}
        labels = ids['s_id'].map(sub_split).where(kept)                # NaN = excluded
        source = 'file-order participant split' + ('' if split_file is None else ' (split file not found)')

    out = {}
    for split in ('train', 'valid', 'test'):
        m = (labels == split).to_numpy()
        out[f'{split}_dat'] = tensor[torch.from_numpy(m)]
        if return_ids:
            out[f'{split}_ids'] = ids[m].reset_index(drop=True)
    n_subs = {s: ids.loc[(labels == s).to_numpy(), 's_id'].nunique() for s in ('train', 'valid', 'test')}
    print(f'Split        — {source}; excluded blocks: {int(labels.isna().sum())}')
    print(f"Participants — train: {n_subs['train']}, valid: {n_subs['valid']}, test: {n_subs['test']}")
    print(f"Blocks       — train: {len(out['train_dat'])}, valid: {len(out['valid_dat'])}, "
          f"test: {len(out['test_dat'])}")
    return out


def format_all_data(
    hum_dat: pd.DataFrame,
    n_actions: int = 4,
    split_file: str | None = DEFAULT_SPLIT_FILE,
    splits: tuple = ('train', 'valid', 'test'),
) -> torch.Tensor:
    """All kept blocks (the paper's 4,134), optionally only some splits.

    splits=('train', 'valid') keeps the test set held out for evaluation.
    """
    t = format_data_for_model_training(hum_dat, n_actions, split_file=split_file)
    data_all = torch.cat([t[f'{s}_dat'] for s in splits], dim=0)
    print(f'All data — blocks: {len(data_all)} ({"+".join(splits)})')
    return data_all


def block_nll(pred_probs: torch.Tensor, batch_dat: torch.Tensor, n_actions: int = 4) -> torch.Tensor:
    """NLL of each block, as in the paper's code.

    pred_probs: (batch, time - 1, n_actions); pred_probs[:, t] predicts the action
    at t + 1 (pass `probs[:, :-1]` from model.unroll). Probabilities are smoothed
    with (1 - 1e-5) p + 5e-4; missed trials have all-zero targets and add 0.
    """
    p = (1 - 1e-5) * pred_probs + 5e-4
    return -(torch.log(p) * batch_dat[:, 1:, :n_actions]).sum(dim=(1, 2))


def accuracy_metrics(nll_per_block: torch.Tensor, batch_dat: torch.Tensor,
                     n_actions: int = 4) -> dict:
    """Paper accuracy = mean over blocks of exp(-NLL_block / n_trials) (Methods; Supp. Table 6).

    'acc_pooled' = exp(-total NLL / valid trials) is ~2 points lower for the same
    model and is NOT comparable to the paper's numbers.
    """
    nll = nll_per_block.double()
    n_trials = batch_dat.shape[1]
    n_valid = batch_dat[:, 1:, :n_actions].sum().double()
    return {'nll_per_block': nll.mean().item(),
            'acc_paper': torch.exp(-nll / n_trials).mean().item(),
            'acc_pooled': torch.exp(-nll.sum() / n_valid).item()}


def get_batch(
    tensor_dat: torch.Tensor,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a random batch of blocks without replacement.

    Args:
        tensor_dat: (n_blocks, n_trials, features)
        batch_size: number of blocks to sample
        generator:  optional torch.Generator for reproducibility

    Returns:
        (batch_size, n_trials, features)
    """
    n_blocks = tensor_dat.shape[0]
    idx = torch.randperm(n_blocks, generator=generator)[:batch_size]
    return tensor_dat[idx]
