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
  scalar  + kron_ll  (default) : last-layer Kron (habit_out_linear only for BiRNN)
  scalar  + kron                : all-Linear-layer Kron (habit AND value streams for
                                 BiRNN), backed by AsdlGGN -- see note below.
  layerwise + full              : one prior per param tensor, full Hessian, every
                                 parameter including BiRNN's bare scalars.
                                 Recommended for best calibration on our small models.
  scalar  + full                : one scalar prior, full Hessian (slower, rarely better)

All three require use_rnn_cell=False so the recurrence is a manual per-timestep
loop over nn.Linear layers (no nn.RNN).

'kron' uses AsdlGGN, not CurvlinopsGGN: AsdlGGN is ~2.5x faster for our models
(measured 6.7s vs 16.2s to fit BiRNN's Kron over the full training set) and,
once the two fixes below are applied, works fine despite habit_rnn_linear/
value_rnn_linear being called once per timestep (149x per forward pass):
  1. The wrapper's value stream must NOT be detached (detach_value=False,
     wired below) -- ASDL's hooks only populate module.fisher for modules
     that actually receive a backward gradient; if the value stream is
     detached (as it is for 'kron_ll'), value_rnn_linear/value_out_linear
     never get a `.fisher` stat, and multiplying Kron by a curvature factor
     that's None for those blocks raises `TypeError: 'float' * NoneType`.
     This was previously (mis)diagnosed as a fundamental ASDL weight-sharing
     limitation -- it was actually this same detach bug.
  2. Bare nn.Parameters (BiRNN's scalar init/forget values) can't be
     Kron-factored by ASDL either (it only walks nn.Linear/nn.Conv modules),
     and BaseLaplace snapshots which parameters it covers at construction
     time -- so they must be frozen (requires_grad=False) BEFORE constructing
     the Laplace object, not just around .fit(), or self.H's block count
     silently drifts out of sync with what the backend actually returns.
     freeze_non_linear_parameters() in laplace_compat.py handles this; those
     parameters fall back to the scalar/layerwise prior, same as before.
"""

import argparse
import json
import os
import time
import torch
import pandas as pd

from laplace          import KronLaplace, KronLLLaplace, FullLaplace
from laplace.curvature import AsdlGGN, CurvlinopsGGN

from hybrid_rnns_pytorch.rnn_config import get_config, get_rnn_config, get_birnn_config
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
    p.add_argument('--lr',           type=float, default=1e-4)
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
    p.add_argument('--backend',      choices=['kron_ll', 'kron', 'full'], default='kron_ll',
                   help='Laplace backend. '
                        '"kron_ll" = KronLLLaplace+AsdlGGN (default, last-layer Kron '
                        '            -- habit stream only for BiRNN). '
                        '"kron"    = KronLaplace+AsdlGGN (all Linear layers -- habit '
                        '            AND value streams for BiRNN, ~2.5x faster than '
                        '            CurvlinopsGGN. Bare nn.Parameters (BiRNN scalar '
                        '            init/forget values) are excluded from curvature '
                        '            and covered by the prior only. '
                        '"full"    = FullLaplace+CurvlinopsGGN (full Hessian over every '
                        '            parameter including bare ones; slowest).')
    p.add_argument('--prior-structure', choices=['scalar', 'layerwise', 'diagonal'],
                   default=None,
                   help='Prior structure. Default: "scalar" for kron_ll and kron, '
                        '"layerwise" for full. '
                        '"layerwise" gives one prior precision per parameter tensor.')
    p.add_argument('--hidden-size',   type=int, default=None,
                   help='Hidden units per RNN layer. Default: paper-optimal for '
                        '--model (64 for rnn, 32 for birnn) -- only pass this to '
                        'override, e.g. to match a checkpoint\'s hidden_size.')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile() the plain SGD training/eval forward pass '
                        '(not the Laplace/ASDL curvature fit -- see '
                        'marglik_training.py\'s compile_model docstring).')
    return p.parse_args()


def _build_model(config):
    if config.model_name == 'birnn':
        return BiRNN(config.rnn_rl_params, config.network_params)
    return RNN(config.rnn_rl_params, config.network_params)


def main():
    args = parse_args()

    # Model-specific paper-verified configs (s=True hidden-state feedback for
    # RNN; w_v=1/w_h=1/fit_forget=True/zero_values=True for BiRNN) -- NOT
    # get_config(), which is a generic placeholder with these flags off and
    # caps achievable accuracy well below what run_training.py reaches
    # regardless of how much marglik training is run.
    config = get_birnn_config() if args.model == 'birnn' else get_rnn_config()
    config.dataset_path = args.dataset
    if args.hidden_size is not None:
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
        laplace_cls  = KronLLLaplace
        backend_cls  = AsdlGGN
        prior_struct = args.prior_structure or 'scalar'
        if prior_struct != 'scalar':
            raise ValueError('--backend kron_ll only supports --prior-structure scalar')
    elif args.backend == 'kron':
        laplace_cls  = KronLaplace
        backend_cls  = AsdlGGN
        prior_struct = args.prior_structure or 'scalar'
    else:  # 'full'
        laplace_cls  = FullLaplace
        backend_cls  = CurvlinopsGGN
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
    # detach_value=False for 'kron'/'full': both are meant to cover the value
    # stream too (see laplace_compat.py docstring); 'kron_ll' keeps the
    # last-layer approximation, so the value stream stays detached.
    torch.manual_seed(args.seed)
    model   = _build_model(config).to(device)
    wrapped = laplace_ready(model, n_actions=config.network_params.n_actions,
                             detach_value=(args.backend == 'kron_ll'))

    # Materialise any LazyLinear layers before Laplace sees the model
    with torch.no_grad():
        dummy_x = next(iter(train_loader))[0].to(device)
        wrapped(dummy_x)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    # ---- held-out test set (same 80/10/10 participant split as normal training) ----
    tensors  = hyb_rnn_utilities.format_data_for_model_training(hum_dat)
    test_dat = tensors['test_dat'].to(device)

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
        compile_model     = args.compile,
    )

    elapsed = time.time() - t0

    # ---- restore best weights then evaluate on full held-out test set ----
    if best_model_dict is not None:
        wrapped.load_state_dict(best_model_dict)

    n_actions = config.network_params.n_actions
    inner_model = wrapped.model
    inner_model.eval()
    with torch.no_grad():
        test_input = test_dat[:, :, :n_actions + 1]
        action_probs_seq, _ = inner_model.unroll(test_input)
        action_probs_seq = (1 - 1e-5) * action_probs_seq + 5e-4
        targets  = test_dat[:, 1:, :n_actions]
        mask     = test_dat[:, 1:, n_actions + 1]
        preds    = action_probs_seq[:, :-1]
        step_nll = -(torch.log(preds) * targets).sum(dim=-1)
        n_valid  = mask.sum()
        test_acc = torch.exp(-(step_nll * mask).sum() / n_valid).item()
        test_argmax_acc = ((preds.argmax(-1) == targets.argmax(-1)) * mask).sum().item() / n_valid.item()

    print(f'\n=== Done ===')
    print(f'Training time  : {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'Best marglik   : {best_marglik:.4f}')
    print(f'Test acc       : {test_acc*100:.2f}%  (paper target: ~68.3%)')
    print(f'Prior precision: {best_precision}')

    results = {
        'model':            args.model,
        'hidden_size':      config.network_params.hidden_size,
        'backend':          args.backend,
        'prior_structure':  prior_struct,
        'epochs':           n_epochs,
        'lr':               args.lr,
        'lr_hyp':           args.lr_hyp,
        'seed':             args.seed,
        'best_marglik':     round(best_marglik, 4),
        'best_precision':   best_precision.tolist() if best_precision is not None else None,
        'test_acc':         round(test_acc, 4),
        'test_argmax_acc':  round(test_argmax_acc, 4),
        'training_time_s':  round(elapsed, 1),
    }
    os.makedirs('results', exist_ok=True)
    results_path = f'results/{args.model}_marglik_be={args.backend}_e={n_epochs}_hs={config.network_params.hidden_size}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {results_path}')

    save_path = args.save or f'trained_models/{args.model}_marglik_be={args.backend}_e={n_epochs}_hs={config.network_params.hidden_size}.pt'
    torch.save(model.state_dict(), save_path)
    print(f'Weights saved to {save_path}')

    return wrapped, best_precision, best_marglik


if __name__ == '__main__':
    main()
