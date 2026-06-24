"""evaluate_predictions.py — Compare generated synthetic data against human data.

Metrics
-------
1. Action frequency — do generated agents favour the same actions as humans?
2. Win-stay / Lose-shift — classic behavioural signatures of RL.
   "Win"  = reward above the dataset mean (continuous rewards).
   "Lose" = reward at or below the dataset mean.
3. Learning curve — P(chose highest-payout option) over trials within a block.
   Shows whether agents exploit the best arm increasingly over time.
4. Mean reward per trial — average earned reward trajectory over the block.

Multiple generated files (one per seed) are shown as mean ± shaded std band.

Usage
-----
    # Evaluate one generated file against human data
    python evaluate_predictions.py --generated generated/birnn_marglik_seed0.csv

    # Evaluate multiple seeds (shaded band = std across seeds)
    python evaluate_predictions.py --generated generated/birnn_marglik_seed*.csv

    # Different human data path
    python evaluate_predictions.py \\
        --original hybrid_rnns_pytorch/data/openSourceRawDataset.csv \\
        --generated generated/birnn_marglik_seed*.csv \\
        --save comparison.png
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare generated synthetic data to human behavioural data.')
    p.add_argument('--original',   type=str,
                   default='hybrid_rnns_pytorch/data/openSourceRawDataset.csv',
                   help='Path to the original human dataset CSV.')
    p.add_argument('--generated',  type=str, nargs='+', required=True,
                   help='Path(s) to generated CSV files (glob patterns accepted).')
    p.add_argument('--label',      type=str, default=None,
                   help='Label for the model in the plot (default: filename stem).')
    p.add_argument('--save',       type=str, default=None,
                   help='Save figure to this path (e.g. comparison.png). '
                        'If omitted, figure is shown interactively.')
    p.add_argument('--no-debug',   action='store_true',
                   help='Use all participants. Default: first 20 (matching generate_predictions.py).')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_df(path: str, n_actions: int = 4, debug_subs: int | None = None) -> pd.DataFrame:
    """Load and clean a dataset CSV. Drops missed trials (action < 0)."""
    df = pd.read_csv(path)
    if debug_subs is not None:
        keep = df['s_id'].unique()[:debug_subs]
        df   = df[df['s_id'].isin(keep)]
    df = df[df['action'] >= 0].copy()
    df['action'] = df['action'].astype(int)
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def action_frequencies(df: pd.DataFrame, n_actions: int = 4) -> np.ndarray:
    counts = df['action'].value_counts(normalize=True)
    return np.array([counts.get(i, 0.0) for i in range(n_actions)])


def win_stay_lose_shift(df: pd.DataFrame) -> tuple[float, float]:
    """Win-stay and lose-shift rates.

    Win  = reward > mean reward across the whole dataset.
    Stay = same action as the previous trial.
    """
    threshold = df['reward'].mean()
    ws, ls    = [], []

    for (_, _), grp in df.groupby(['s_id', 'block']):
        grp = grp.sort_values('trial_id')
        acts = grp['action'].values
        rews = grp['reward'].values
        for t in range(1, len(grp)):
            if acts[t - 1] < 0 or acts[t] < 0:
                continue
            stayed = int(acts[t] == acts[t - 1])
            if rews[t - 1] > threshold:
                ws.append(stayed)
            else:
                ls.append(1 - stayed)

    return (np.mean(ws) if ws else np.nan,
            np.mean(ls) if ls else np.nan)


def learning_curve(df: pd.DataFrame, n_actions: int = 4) -> np.ndarray:
    """P(chose highest-payout action) per trial index, averaged over blocks."""
    payout_cols = [f'payout_{i + 1}' for i in range(n_actions)]
    df = df.copy()
    df['best_action']  = df[payout_cols].values.argmax(axis=1)
    df['chose_best']   = (df['action'] == df['best_action']).astype(float)
    curve = df.groupby('trial_id')['chose_best'].mean().sort_index()
    return curve.index.values, curve.values


def reward_curve(df: pd.DataFrame) -> np.ndarray:
    """Mean reward per trial index, averaged over blocks."""
    curve = df.groupby('trial_id')['reward'].mean().sort_index()
    return curve.index.values, curve.values


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

HUMAN_COLOR = '#2c7bb6'
MODEL_COLOR = '#d7191c'
ALPHA_BAND  = 0.25


def _smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
    """Simple moving average (valid padding)."""
    if w <= 1 or len(y) < w:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode='same')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- expand glob patterns ----
    paths = []
    for pattern in args.generated:
        expanded = glob.glob(pattern)
        paths.extend(expanded if expanded else [pattern])
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f'No files matched: {args.generated}')

    debug_subs = None if args.no_debug else 20
    n_actions  = 4

    # ---- load human data ----
    df_hum = load_df(args.original, n_actions, debug_subs)
    print(f'Human data  : {df_hum["s_id"].nunique()} participants, '
          f'{len(df_hum)} valid trials')

    # ---- load generated data ----
    gen_dfs = []
    for p in paths:
        try:
            gen_dfs.append(load_df(p, n_actions, debug_subs))
            print(f'Generated   : {p}  ({len(gen_dfs[-1])} rows)')
        except Exception as e:
            print(f'WARNING: could not load {p}: {e}')
    if not gen_dfs:
        raise RuntimeError('No generated files loaded successfully.')

    label = args.label or os.path.splitext(os.path.basename(paths[0]))[0]
    if len(paths) > 1:
        # strip trailing _seed* for a cleaner label
        stem = os.path.splitext(os.path.basename(paths[0]))[0]
        label = stem.rsplit('_seed', 1)[0] if '_seed' in stem else stem
        label += f' (n={len(paths)} seeds)'

    # ---- compute metrics ----
    # Human
    hum_freqs         = action_frequencies(df_hum, n_actions)
    hum_ws, hum_ls    = win_stay_lose_shift(df_hum)
    hum_lc_x, hum_lc  = learning_curve(df_hum, n_actions)
    hum_rc_x, hum_rc  = reward_curve(df_hum)

    # Generated (per seed)
    gen_freqs_list = []
    gen_ws_list, gen_ls_list = [], []
    gen_lc_list, gen_rc_list = [], []

    for df_g in gen_dfs:
        gen_freqs_list.append(action_frequencies(df_g, n_actions))
        ws, ls = win_stay_lose_shift(df_g)
        gen_ws_list.append(ws); gen_ls_list.append(ls)
        _, lc = learning_curve(df_g, n_actions)
        _, rc = reward_curve(df_g)
        gen_lc_list.append(lc)
        gen_rc_list.append(rc)

    gen_freqs_mean = np.mean(gen_freqs_list, axis=0)
    gen_freqs_std  = np.std(gen_freqs_list,  axis=0)
    gen_ws_mean, gen_ws_std = np.nanmean(gen_ws_list), np.nanstd(gen_ws_list)
    gen_ls_mean, gen_ls_std = np.nanmean(gen_ls_list), np.nanstd(gen_ls_list)
    gen_lc_mean = np.mean(gen_lc_list, axis=0)
    gen_lc_std  = np.std(gen_lc_list,  axis=0)
    gen_rc_mean = np.mean(gen_rc_list, axis=0)
    gen_rc_std  = np.std(gen_rc_list,  axis=0)

    # ---- print summary ----
    print(f'\n{"="*55}')
    print(f'{"Metric":<28}  {"Human":>8}  {"Model":>16}')
    print(f'{"-"*55}')
    for a in range(n_actions):
        print(f'  P(action={a})               {hum_freqs[a]:>8.3f}  '
              f'{gen_freqs_mean[a]:>7.3f} ±{gen_freqs_std[a]:.3f}')
    print(f'  Win-stay rate             {hum_ws:>8.3f}  '
          f'{gen_ws_mean:>7.3f} ±{gen_ws_std:.3f}')
    print(f'  Lose-shift rate           {hum_ls:>8.3f}  '
          f'{gen_ls_mean:>7.3f} ±{gen_ls_std:.3f}')
    print(f'  Mean reward               {df_hum["reward"].mean():>8.3f}  '
          f'{np.mean([df["reward"].mean() for df in gen_dfs]):>7.3f} '
          f'±{np.std([df["reward"].mean() for df in gen_dfs]):.3f}')
    print(f'  P(best @ trial 0)         {hum_lc[0]:>8.3f}  '
          f'{gen_lc_mean[0]:>7.3f} ±{gen_lc_std[0]:.3f}')
    print(f'  P(best @ last trial)      {hum_lc[-1]:>8.3f}  '
          f'{gen_lc_mean[-1]:>7.3f} ±{gen_lc_std[-1]:.3f}')
    print(f'{"="*55}')

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Human vs {label}', fontsize=13, fontweight='bold')

    # — Panel 1: Action frequencies —
    ax = axes[0, 0]
    x   = np.arange(n_actions)
    w   = 0.35
    ax.bar(x - w/2, hum_freqs,       width=w, color=HUMAN_COLOR, label='Human',  alpha=0.85)
    ax.bar(x + w/2, gen_freqs_mean,  width=w, color=MODEL_COLOR, label=label,    alpha=0.85,
           yerr=gen_freqs_std, capsize=4, error_kw=dict(elinewidth=1))
    ax.set_xlabel('Action')
    ax.set_ylabel('Proportion')
    ax.set_title('Action frequency')
    ax.set_xticks(x);  ax.set_xticklabels([f'Action {i}' for i in x])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)

    # — Panel 2: Win-stay / Lose-shift —
    ax = axes[0, 1]
    cats   = ['Win-stay', 'Lose-shift']
    hum_v  = [hum_ws, hum_ls]
    gen_v  = [gen_ws_mean, gen_ls_mean]
    gen_e  = [gen_ws_std,  gen_ls_std]
    x2     = np.arange(len(cats))
    ax.bar(x2 - w/2, hum_v, width=w, color=HUMAN_COLOR, alpha=0.85, label='Human')
    ax.bar(x2 + w/2, gen_v, width=w, color=MODEL_COLOR, alpha=0.85, label=label,
           yerr=gen_e, capsize=4, error_kw=dict(elinewidth=1))
    ax.set_xticks(x2); ax.set_xticklabels(cats)
    ax.set_ylabel('Rate')
    ax.set_title('Win-stay / Lose-shift')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)

    # — Panel 3: Learning curve —
    ax = axes[1, 0]
    sm = 7  # smoothing window
    ax.plot(hum_lc_x, _smooth(hum_lc, sm), color=HUMAN_COLOR, lw=2, label='Human')
    ax.plot(hum_lc_x, _smooth(gen_lc_mean, sm), color=MODEL_COLOR, lw=2, label=label)
    ax.fill_between(hum_lc_x,
                    _smooth(gen_lc_mean - gen_lc_std, sm),
                    _smooth(gen_lc_mean + gen_lc_std, sm),
                    color=MODEL_COLOR, alpha=ALPHA_BAND)
    ax.axhline(1 / n_actions, color='gray', lw=1, ls='--', label='Chance')
    ax.set_xlabel('Trial')
    ax.set_ylabel('P(best action)')
    ax.set_title('Learning curve  (smoothed)')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)

    # — Panel 4: Mean reward per trial —
    ax = axes[1, 1]
    ax.plot(hum_rc_x, _smooth(hum_rc, sm), color=HUMAN_COLOR, lw=2, label='Human')
    ax.plot(hum_rc_x, _smooth(gen_rc_mean, sm), color=MODEL_COLOR, lw=2, label=label)
    ax.fill_between(hum_rc_x,
                    _smooth(gen_rc_mean - gen_rc_std, sm),
                    _smooth(gen_rc_mean + gen_rc_std, sm),
                    color=MODEL_COLOR, alpha=ALPHA_BAND)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Mean reward')
    ax.set_title('Reward trajectory  (smoothed)')
    ax.legend(fontsize=9)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f'\nFigure saved to {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
