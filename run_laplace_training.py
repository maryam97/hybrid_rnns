"""run_laplace_training.py — Train from scratch with Laplace marginal likelihood.

Jointly optimises model weights and prior precision over ALL data (no split).

Usage
-----
    python run_laplace_training.py                               # birnn, scalar, debug
    python run_laplace_training.py --model rnn
    python run_laplace_training.py --no-debug --epochs 500
    python run_laplace_training.py --prior-structure layerwise  # one precision per param tensor
    python run_laplace_training.py --backend full               # FullLaplace+CurvlinopsGGN
                                                                # required for layerwise on BiRNN

Backend / prior-structure combinations
---------------------------------------
  scalar  + kron_ll  (default) : one scalar prior for last-layer Kron
  scalar  + kron_ll            : same
  layerwise + full             : one prior per param tensor (10 for BiRNN), full Hessian
                                 Recommended for best calibration on our small models.
  scalar  + full               : one scalar prior, full Hessian (slower, rarely better)

The kron/kron_full backends require use_rnn_cell=False AND all parameters inside
nn.Linear (ASDL limitation). For BiRNN the two scalar nn.Parameters
(_raw_init_value_v/h) cause KronLaplace to crash, so use 'kron_ll' or 'full'.
"""

import argparse
import json
import os
import time
import torch
import pandas as pd

from laplace          import KronLLLaplace, FullLaplace
from laplace.curvature import AsdlGGN, CurvlinopsGGN

from hybrid_rnns_pytorch.rnn_config import get_config
from hybrid_rnns_pytorch import hyb_rnn_utilities
from hybrid_rnns_pytorch.laplace_compat import laplace_ready, make_dataloader
from hybrid_rnns_pytorch.marglik_training import marglik_optimization
from hybrid_rnns_pytorch.bi_rnn import BiRNN
from hybrid_rnns_pytorch.rnn    import RNN


def parse_args():
    p = argparse.ArgumentParser(
        description='Train RNN/BiRNN from scratch with Laplace marginal likelihood.')
    p.add_argument('--model',        choices=['rnn', 'birnn'], default='birnn')
    p.add_argument('--no-debug',     action='store_true')
    p.add_argument('--epochs',       type=int, default=None,
                   help='Training epochs (default: 10 debug / 500 full).')
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--lr-hyp',       type=float, default=1e-1,
                   help='Learning rate for prior precision hyperparameter.')
    p.add_argument('--batch-size',   type=int,   default=32)
    p.add_argument('--burnin',       type=int,   default=None,
                   help='Epochs before marglik updates start '
                        '(default: 20%% of total epochs).')
    p.add_argument('--marglik-freq', type=int,   default=1)
    p.add_argument('--n-hypersteps', type=int,   default=100,
                   help='Inner-loop steps on prior precision per marglik update.')
    p.add_argument('--dataset',      type=str,
                   default='hybrid_rnns_pytorch/data/openSourceRawDataset.csv')
    p.add_argument('--save',         type=str,   default=None,
                   help='Path to save final model weights.')
    p.add_argument('--seed',         type=int,   default=0,
                   help='Random seed for model init (default: 0). '
                        'Seed 42 is the rnn_config default but produces extreme '
                        'initial BiRNN logits — seed 0 is more stable.')
    p.add_argument('--backend',      choices=['kron_ll', 'full'], default='kron_ll',
                   help='Laplace backend. '
                        '"kron_ll" = KronLLLaplace+AsdlGGN (default, last-layer Kron). '
                        '"full"    = FullLaplace+CurvlinopsGGN (full Hessian, '
                        'supports layerwise prior on all param types).')
    p.add_argument('--prior-structure', choices=['scalar', 'layerwise', 'diagonal'],
                   default=None,
                   help='Prior structure. Default: "scalar" for kron_ll, '
                        '"layerwise" for full. '
                        '"layerwise" gives one prior precision per parameter tensor '
                        '(10 for BiRNN) and is the recommended setting with --backend full.')
    p.add_argument('--hidden-size',   type=int, default=64,
                   help='Hidden units per RNN layer (default: 64, paper-optimal). '
                        'Must match the checkpoint if loading one.')
    return p.parse_args()


def _build_model(config):
    if config.model_name == 'birnn':
        return BiRNN(config.rnn_rl_params, config.network_params)
    return RNN(config.rnn_rl_params, config.network_params)


def main():
    args = parse_args()

    config             = get_config()
    config.model_name  = args.model
    config.dataset_path = args.dataset
    config.network_params.hidden_size = args.hidden_size
    config.network_params.use_rnn_cell = False
    # Weight decay is NOT applied here: the Laplace prior precision already acts
    # as L2 regularization and is optimised via marginal likelihood. Adding
    # AdamW weight decay on top would double-regularise.
    config.weight_decay = 0.0

    debug    = not args.no_debug
    n_epochs = args.epochs or (10 if debug else 500)
    burnin   = args.burnin or max(1, n_epochs // 5)

    # ---- resolve backend / prior-structure -----------------------------------
    if args.backend == 'kron_ll':
        laplace_cls = KronLLLaplace
        backend_cls = AsdlGGN
        prior_struct = args.prior_structure or 'scalar'
        if prior_struct != 'scalar':
            raise ValueError('--backend kron_ll only supports --prior-structure scalar')
    else:  # 'full'
        laplace_cls = FullLaplace
        backend_cls = CurvlinopsGGN
        prior_struct = args.prior_structure or 'layerwise'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device          : {device}')
    print(f'Model           : {config.model_name}')
    print(f'Hidden size     : {config.network_params.hidden_size}')
    print(f'Epochs          : {n_epochs}  (burnin: {burnin})')
    print(f'LR              : {args.lr}  lr_hyp: {args.lr_hyp}')
    print(f'Backend         : {args.backend}  ({laplace_cls.__name__}+{backend_cls.__name__})')
    print(f'Prior structure : {prior_struct}')

    # ---- load ALL data — no train/test split ----
    print(f'\nLoading {args.dataset}')
    hum_dat = pd.read_csv(args.dataset)

    if debug:
        unique_subs = hum_dat['s_id'].unique()[:20]
        hum_dat = hum_dat[hum_dat['s_id'].isin(unique_subs)]
        print(f'Debug mode: {len(unique_subs)} participants')

    all_dat = hyb_rnn_utilities.format_all_data(
        hum_dat, n_actions=config.network_params.n_actions)
    all_dat = all_dat.to(device)

    train_loader = make_dataloader(
        all_dat,
        n_actions  = config.network_params.n_actions,
        batch_size = args.batch_size,
        shuffle    = True,
    )
    print(f'DataLoader: {len(train_loader)} batches/epoch')

    # ---- build model and wrap ----
    torch.manual_seed(args.seed)
    model   = _build_model(config).to(device)
    wrapped = laplace_ready(model, n_actions=config.network_params.n_actions)

    # Materialise any LazyLinear layers before Laplace sees the model
    with torch.no_grad():
        dummy_x = next(iter(train_loader))[0].to(device)
        wrapped(dummy_x)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    # ---- marginal-likelihood training ----
    t0 = time.time()
    best_model_dict, best_precision, best_marglik = marglik_optimization(
        model             = wrapped,
        train_loader      = train_loader,
        prior_structure   = prior_struct,
        n_epochs          = n_epochs,
        lr                = args.lr,
        lr_hyp            = args.lr_hyp,
        n_epochs_burnin   = burnin,
        n_hypersteps      = args.n_hypersteps,
        marglik_frequency = args.marglik_freq,
        laplace           = laplace_cls,
        backend           = backend_cls,
    )

    elapsed = time.time() - t0
    print(f'\n=== Done ===')
    print(f'Training time  : {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'Best marglik : {best_marglik:.4f}')
    print(f'Prior precision: {best_precision}')

    results = {
        'model':           args.model,
        'hidden_size':     args.hidden_size,
        'backend':         args.backend,
        'prior_structure': prior_struct,
        'epochs':          n_epochs,
        'lr':              args.lr,
        'lr_hyp':          args.lr_hyp,
        'seed':            args.seed,
        'best_marglik':    round(best_marglik, 4),
        'best_precision':  best_precision.tolist() if best_precision is not None else None,
        'training_time_s': round(elapsed, 1),
    }
    os.makedirs('results', exist_ok=True)
    results_path = f'results/{args.model}_marglik_be={args.backend}_e={n_epochs}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {results_path}')

    # Restore best weights
    if best_model_dict is not None:
        wrapped.load_state_dict(best_model_dict)

    save_path = args.save or f'trained_models/{args.model}_marglik_be={args.backend}_e={n_epochs}.pt'
    torch.save(model.state_dict(), save_path)
    print(f'Weights saved to {save_path}')

    return wrapped, best_precision, best_marglik


if __name__ == '__main__':
    main()
