"""Utilities — PyTorch port of hybrid_rnns_reward_learning/hyb_rnn_utilities.py."""

import numpy as np
import torch
import torch.nn.functional as F


def format_data_for_model_training(
    hum_dat,
    n_actions: int = 4,
    random_seed: int = 1356,
) -> dict[str, torch.Tensor]:
    """Format a pandas DataFrame for model training.

    Splits 80/10/10 by participant (paper methodology): all blocks from a
    participant land entirely in one split, so there is no leakage between
    train and test.

    Returns a dict with keys 'train_dat', 'valid_dat', 'test_dat', each a
    float32 tensor of shape (n_blocks, n_trials, n_actions + 2).
    Columns: [one-hot action (n_actions) | reward (1) | valid mask (1)].
    Missed trials (action == -1) have an all-zeros action vector and valid == 0.
    """
    n_trials = len(np.unique(hum_dat['trial_id']))

    # Sort so each consecutive group of n_trials rows is one complete block
    hum_dat = hum_dat.sort_values(['s_id', 'block', 'trial_id']).reset_index(drop=True)

    action_arr = np.array(hum_dat['action'])
    valid      = torch.from_numpy((action_arr >= 0).astype(np.float32)).unsqueeze(-1)

    # Clamp -1 → 0 before one-hot; multiply by valid mask so missed trials are all-zeros
    actions_clipped = torch.from_numpy(np.clip(action_arr, 0, n_actions - 1)).long()
    actions_onehot  = F.one_hot(actions_clipped, num_classes=n_actions).float() * valid

    rewards = torch.from_numpy(
        np.array(hum_dat['reward'])).float().unsqueeze(-1)

    # Layout: [one-hot action (n_actions) | reward (1) | valid mask (1)]
    data_flat = torch.cat([actions_onehot, rewards, valid], dim=-1)   # (N, n_actions+2)

    # ---- split 80/10/10 by participant ----
    unique_subs = np.unique(hum_dat['s_id'].values)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_subs)

    n_subs = len(unique_subs)
    n_held = n_subs // 10                        # 10% each for valid & test
    # seed=1356 with this ordering reproduces the paper's reported split sizes
    # exactly: 690/86/86 participants, 3,302/419/413 blocks (train/valid/test)
    # after the >15-missed-trial exclusion below.
    valid_subs = set(unique_subs[:n_held])
    test_subs  = set(unique_subs[n_held: 2 * n_held])
    train_subs = set(unique_subs[2 * n_held:])

    def _blocks_for(subs):
        mask     = np.isin(hum_dat['s_id'].values, list(subs))
        sub_flat = data_flat[mask]
        n_blocks = sub_flat.shape[0] // n_trials
        blocks   = sub_flat[: n_blocks * n_trials].view(n_blocks, n_trials, -1)
        # Paper exclusion criterion: drop blocks with >15 missed trials (>10%)
        valid_counts = blocks[:, :, n_actions + 1].sum(dim=1)  # valid mask is last col
        return blocks[valid_counts >= (n_trials - 15)]

    train_dat = _blocks_for(train_subs)
    valid_dat = _blocks_for(valid_subs)
    test_dat  = _blocks_for(test_subs)

    print(f'Participants — train: {len(train_subs)}, valid: {len(valid_subs)}, test: {len(test_subs)}')
    print(f'Blocks       — train: {len(train_dat)}, valid: {len(valid_dat)}, test: {len(test_dat)}')

    return {'train_dat': train_dat, 'valid_dat': valid_dat, 'test_dat': test_dat}


def format_all_data(
    hum_dat,
    n_actions: int = 4,
) -> torch.Tensor:
    """Format all data as a single block tensor — no train/test split.

    Use this for Laplace marginal-likelihood training where the goal is
    Bayesian inference over all participants, not cross-validated evaluation.

    Returns a float32 tensor of shape (n_blocks, n_trials, n_actions + 2).
    Columns: [one-hot action (n_actions) | reward (1) | valid mask (1)].
    """
    n_trials = len(np.unique(hum_dat['trial_id']))
    hum_dat  = hum_dat.sort_values(['s_id', 'block', 'trial_id']).reset_index(drop=True)

    action_arr      = np.array(hum_dat['action'])
    valid           = torch.from_numpy((action_arr >= 0).astype(np.float32)).unsqueeze(-1)
    actions_clipped = torch.from_numpy(np.clip(action_arr, 0, n_actions - 1)).long()
    actions_onehot  = F.one_hot(actions_clipped, num_classes=n_actions).float() * valid
    rewards         = torch.from_numpy(np.array(hum_dat['reward'])).float().unsqueeze(-1)

    data_flat = torch.cat([actions_onehot, rewards, valid], dim=-1)   # (N, n_actions+2)
    n_blocks  = data_flat.shape[0] // n_trials
    data_all  = data_flat[: n_blocks * n_trials].view(n_blocks, n_trials, -1)

    print(f'All data — blocks: {n_blocks}, trials per block: {n_trials}')
    return data_all


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
