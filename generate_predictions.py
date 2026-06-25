"""generate_predictions.py — Generate synthetic action sequences from a trained RNN/BiRNN.

For each random seed, runs the model step-by-step through the task and samples
discrete action choices from the predicted probabilities.  Saves one CSV per seed
that mirrors the original dataset structure with only 'action' and 'reward' replaced.

Why both action AND reward are replaced
---------------------------------------
Both RNN and BiRNN receive (action, reward) as sequential input and update their
internal state from it.  If the model chose action 2 but we fed it the participant's
reward for action 0, the model's value estimates would be corrupted for every
subsequent step.  The payout_1..4 columns give the correct reward for each possible
action at each trial, so we use payout_{chosen_action+1} as the reward that feeds
back into the model — and store it in the output CSV.  All other columns
(s_id, block, trial_id, rt, payout_1..4) are copied unchanged from the original data.

Usage
-----
    python generate_predictions.py --checkpoint birnn_marglik.pt --n-seeds 10
    python generate_predictions.py --checkpoint birnn_marglik.pt --seeds 0 1 2 42 99
    python generate_predictions.py --checkpoint rnn_trained.pt --model rnn --n-seeds 5
    python generate_predictions.py --checkpoint birnn_marglik.pt --no-debug --n-seeds 20 --save-dir results/synthetic
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from hybrid_rnns_pytorch.rnn_config import get_config
from hybrid_rnns_pytorch.bi_rnn import BiRNN
from hybrid_rnns_pytorch.rnn import RNN


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate synthetic action sequences from a trained RNN/BiRNN.')
    p.add_argument('--checkpoint',  type=str, required=True,
                   help='Path to saved model weights (.pt).')
    p.add_argument('--model',       choices=['rnn', 'birnn'], default='birnn',
                   help='Model architecture (must match checkpoint).')
    p.add_argument('--dataset',     type=str,
                   default='hybrid_rnns_pytorch/data/openSourceRawDataset.csv')
    p.add_argument('--n-seeds',     type=int, default=10,
                   help='Number of synthetic datasets to generate.')
    p.add_argument('--first-seed',  type=int, default=0,
                   help='First seed; seeds run first-seed .. first-seed+n-seeds-1.')
    p.add_argument('--seeds',       type=int, nargs='+', default=None,
                   help='Explicit seed list (overrides --n-seeds / --first-seed).')
    p.add_argument('--no-debug',    action='store_true',
                   help='Use all participants (default: first 20 for a quick check).')
    p.add_argument('--save-dir',    type=str, default='synthetic_data',
                   help='Output directory (created if needed).')
    p.add_argument('--prefix',      type=str, default=None,
                   help='Filename prefix.  Default: checkpoint name without extension.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_choices(
    model: torch.nn.Module,
    payouts: torch.Tensor,
    seed: int,
    n_actions: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model generatively through all blocks and sample actions.

    Starts from the model's initial state with a zero input (no prior action
    or reward), then feeds each sampled action together with its correct payout
    back as input for the next trial.

    Parameters
    ----------
    model    : trained RNN or BiRNN (use_rnn_cell=False)
    payouts  : (n_blocks, n_trials, n_actions) — payout for each action at each trial
    seed     : random seed for Categorical sampling
    n_actions: number of discrete actions

    Returns
    -------
    gen_actions : (n_blocks, n_trials) int array — sampled action indices 0..n_actions-1
    gen_rewards : (n_blocks, n_trials) float array — payout for each sampled action
    """
    torch.manual_seed(seed)

    n_blocks, n_trials, _ = payouts.shape
    device = payouts.device

    gen_action_list = []

    model.eval()
    with torch.no_grad():
        state       = model.initial_state(n_blocks, device)
        prev_onehot = torch.zeros(n_blocks, n_actions, device=device)
        prev_reward = torch.zeros(n_blocks, device=device)

        for t in range(n_trials):
            inp          = torch.cat([prev_onehot, prev_reward.unsqueeze(-1)], dim=-1)
            probs, state = model.forward(inp, state)

            sampled      = torch.distributions.Categorical(probs=probs).sample()
            gen_action_list.append(sampled)

            prev_onehot = F.one_hot(sampled, num_classes=n_actions).float()
            prev_reward = payouts[torch.arange(n_blocks), t, sampled]

    gen_actions_t = torch.stack(gen_action_list, dim=1)              # (n_blocks, n_trials)
    bix = torch.arange(n_blocks).unsqueeze(1).expand_as(gen_actions_t)
    tix = torch.arange(n_trials).unsqueeze(0).expand_as(gen_actions_t)
    gen_rewards_t = payouts[bix, tix, gen_actions_t]                 # (n_blocks, n_trials)

    return gen_actions_t.cpu().numpy(), gen_rewards_t.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    seeds = args.seeds if args.seeds is not None else \
            list(range(args.first_seed, args.first_seed + args.n_seeds))

    # ---- config & model ----
    config = get_config()
    config.model_name = args.model
    config.network_params.use_rnn_cell = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = (BiRNN if args.model == 'birnn' else RNN)(
        config.rnn_rl_params, config.network_params).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print(f'Model      : {args.model}  ({sum(p.numel() for p in model.parameters())} params)')
    print(f'Checkpoint : {args.checkpoint}')
    print(f'Seeds      : {seeds}')

    # ---- load data ----
    n_actions   = config.network_params.n_actions
    payout_cols = [f'payout_{i + 1}' for i in range(n_actions)]

    hum_dat = pd.read_csv(args.dataset)
    if not args.no_debug:
        hum_dat = hum_dat[hum_dat['s_id'].isin(hum_dat['s_id'].unique()[:20])]
        print(f'Debug mode : {hum_dat["s_id"].nunique()} participants')

    hum_dat  = hum_dat.sort_values(['s_id', 'block', 'trial_id']).reset_index(drop=True)
    n_trials = hum_dat['trial_id'].nunique()
    n_blocks = len(hum_dat) // n_trials

    payouts = torch.tensor(
        hum_dat[payout_cols].values.reshape(n_blocks, n_trials, n_actions),
        dtype=torch.float32, device=device)

    print(f'Data       : {n_blocks} blocks × {n_trials} trials')

    # ---- generate & save ----
    os.makedirs(args.save_dir, exist_ok=True)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.checkpoint))[0]

    for seed in seeds:
        gen_actions, gen_rewards = generate_choices(model, payouts, seed, n_actions)

        # Build output DataFrame — copy everything, replace action + reward
        out = hum_dat.copy()
        out['action'] = gen_actions.reshape(-1).astype(float)
        out['reward'] = gen_rewards.reshape(-1)

        csv_path = os.path.join(args.save_dir, f'{prefix}_seed{seed}.csv')
        out.to_csv(csv_path, index=False)
        print(f'seed={seed}  →  {csv_path}  '
              f'(action freqs: {np.round(np.bincount(gen_actions.reshape(-1), minlength=n_actions) / gen_actions.size, 3)})')


if __name__ == '__main__':
    main()
