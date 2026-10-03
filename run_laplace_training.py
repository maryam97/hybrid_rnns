"""run_laplace_training.py — Train from scratch with Laplace marginal likelihood.

Jointly optimises model weights and prior precision. By default the model is
fitted on the WHOLE data: all 4,134 kept blocks of the paper's dataset (the 24
blocks with >15 missed trials are excluded, as in the paper). The result is the
log marginal likelihood (evidence) for model comparison; test-set accuracies are
then in-sample. Paper-comparable held-out accuracies come from run_training.py
(trained on the paper's train split, scored on its 413 test blocks).

Usage
-----
    python run_laplace_training.py --no-debug                    # birnn, all data, kron
    python run_laplace_training.py --no-debug --model rnn
    python run_laplace_training.py --no-debug --fit-on train     # keep test held out
    python run_laplace.py --checkpoint trained_models/<this run>.pt   # exact-GGN evidence afterwards

Backend / prior-structure combinations
---------------------------------------
  scalar + kron (default) : all-Linear-layer Kron (habit AND value streams for
                            BiRNN), AsdlGGN -- see notes below. Good for choosing
                            the prior during training; report the evidence from
                            run_laplace.py --hessian full (exact GGN).
  scalar + kron_ll        : last-layer Kron. RNN only: for the BiRNN it drops the
                            value offset, so it is refused.
  layerwise + full        : one prior per param tensor, exact GGN over every
                            parameter (FuncGGN). Exact but ~1 s per block per
                            marglik update.
  layerwise + kron        : RNN only (the BiRNN's frozen scalars misalign the prior).


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
     parameters are not in the Kron posterior (kept at their MAP values).
"""

import argparse
import json
import os
import time
import torch

from laplace          import KronLaplace, KronLLLaplace, FullLaplace
from laplace.curvature import AsdlGGN

from hybrid_rnns_pytorch.rnn_config import get_rnn_config, get_birnn_config
from hybrid_rnns_pytorch import hyb_rnn_utilities
from hybrid_rnns_pytorch.laplace_compat import (
    laplace_ready, make_dataloader, count_valid_samples, FuncGGN)
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
    p.add_argument('--marglik-freq', type=int,   default=1,
                   help='Epochs between marglik (prior precision) updates.')
    p.add_argument('--eval-freq',    type=int,   default=1,
                   help='Epochs between training-set metric evaluations (each costs ~30%% of an epoch).')
    p.add_argument('--n-hypersteps', type=int,   default=100,
                   help='Inner-loop steps on prior precision per marglik update.')
    p.add_argument('--dataset',      type=str,
                   default='hybrid_rnns_pytorch/data/openSourceRawDataset.csv')
    p.add_argument('--fit-on',       choices=['train', 'trainvalid', 'all'], default='all',
                   help='Blocks to fit on (paper split). "all" (default) = the whole '
                        'data, 4,134 blocks: test metrics are in-sample. "train" / '
                        '"trainvalid" keep the 413 test blocks held out.')
    p.add_argument('--save',         type=str,   default=None,
                   help='Path to save final model weights.')
    p.add_argument('--seed',         type=int,   default=42,
                   help='Random seed for model init (default: 42, as in rnn_config).')
    p.add_argument('--backend',      choices=['kron_ll', 'kron', 'full'], default='kron',
                   help='Laplace backend. '
                        '"kron" (default) = KronLaplace+AsdlGGN over all Linear layers '
                        '(BiRNN bare scalars excluded, kept at MAP). '
                        '"kron_ll" = last-layer Kron, RNN only. '
                        '"full" = FullLaplace + exact GGN (FuncGGN) over every parameter; slow.')
    p.add_argument('--prior-structure', choices=['scalar', 'layerwise', 'diagonal'],
                   default=None,
                   help='Prior structure. Default: "scalar" for kron_ll and kron, '
                        '"layerwise" for full. "layerwise" gives one prior precision '
                        'per parameter tensor (with kron: RNN only).')
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
    if min(args.eval_freq, args.marglik_freq) < 1:
        raise SystemExit('--eval-freq and --marglik-freq must be >= 1')

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
        if args.model == 'birnn':
            raise SystemExit('--backend kron_ll is wrong for birnn: KronLLLaplace fits '
                             'habit_out_linear(features) without the value offset. Use --backend kron.')
    elif args.backend == 'kron':
        laplace_cls  = KronLaplace
        backend_cls  = AsdlGGN
        prior_struct = args.prior_structure or 'scalar'
        # log_prior_prec is sized over ALL parameter tensors (BiRNN: 11), the Kron
        # posterior only over the unfrozen Linear ones (8); Kron has no diagonal prior.
        if prior_struct == 'diagonal' or (prior_struct == 'layerwise' and args.model == 'birnn'):
            raise SystemExit(f'--backend kron does not support --prior-structure {prior_struct} for {args.model}')
    else:  # 'full'
        laplace_cls  = FullLaplace
        backend_cls  = FuncGGN          # exact GGN; CurvlinopsGGN is the same matrix, ~8 min/batch
        prior_struct = args.prior_structure or 'layerwise'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device          : {device}')
    print(f'Model           : {config.model_name}')
    print(f'Hidden size     : {config.network_params.hidden_size}')
    print(f'Epochs          : {n_epochs}  (burnin: {burnin})')
    print(f'LR              : {args.lr}  lr_hyp: {args.lr_hyp}')
    print(f'Backend         : {args.backend}  ({laplace_cls.__name__}+{backend_cls.__name__})')
    print(f'Prior structure : {prior_struct}')

    # ---- load data and split FIRST (paper split), then pick the fitted blocks ----
    n_actions = config.network_params.n_actions
    print(f'\nLoading {args.dataset}')
    hum_dat = hyb_rnn_utilities.load_osf_dataframe(args.dataset)   # rewards 0-1
    tensors = hyb_rnn_utilities.format_data_for_model_training(
        hum_dat, n_actions=n_actions, split_file=config.split_path)
    fit_splits = {'train': ('train',), 'trainvalid': ('train', 'valid'),
                  'all': ('train', 'valid', 'test')}[args.fit_on]
    fit_dat  = torch.cat([tensors[f'{s}_dat'] for s in fit_splits])
    test_dat = tensors['test_dat'].to(device)
    test_held_out = args.fit_on != 'all'
    if args.fit_on == 'all' and not args.no_debug:
        print('Note: debug fits a random subset, so "in-sample" test metrics are only partly in-sample.')

    if debug:
        # Random blocks of the fitted splits (fixed generator, so every debug
        # run fits the same blocks); the test set stays complete.
        g = torch.Generator().manual_seed(0)
        fit_dat = fit_dat[torch.randperm(len(fit_dat), generator=g)[:100]]
        print(f'Debug mode: {len(fit_dat)} random blocks')
    print(f'Fitting on   : {"+".join(fit_splits)} ({len(fit_dat)} blocks)')
    fit_dat = fit_dat.to(device)

    train_loader = make_dataloader(
        fit_dat,
        n_actions  = config.network_params.n_actions,
        batch_size = args.batch_size,
        shuffle    = True,
    )
    print(f'DataLoader: {len(train_loader)} batches/epoch')

    # ---- build model and wrap ----
    torch.manual_seed(args.seed)
    model   = _build_model(config).to(device)
    # Never detach: this wrapper is also the model trained by SGD below.
    wrapped = laplace_ready(model, n_actions=config.network_params.n_actions,
                             detach_value=False)

    # Dry-run forward pass (kept: creating the shuffled iterator advances the global RNG,
    # so removing it would change the batch order)
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
        compile_model     = args.compile,
        eval_frequency    = args.eval_freq,
    )

    elapsed = time.time() - t0

    # ---- restore best weights then evaluate on the full test set (paper metric) ----
    if best_model_dict is not None:
        wrapped.load_state_dict(best_model_dict)

    inner_model = wrapped.model
    inner_model.eval()
    with torch.no_grad():
        probs, _ = inner_model.unroll(test_dat[:, :, :n_actions + 1])
        preds    = probs[:, :-1]                          # output t predicts action t+1
        test_m   = hyb_rnn_utilities.accuracy_metrics(
            hyb_rnn_utilities.block_nll(preds, test_dat, n_actions), test_dat, n_actions)
        targets  = test_dat[:, 1:, :n_actions]
        mask     = test_dat[:, 1:, n_actions + 1]
        test_argmax_acc = ((preds.argmax(-1) == targets.argmax(-1)) * mask).sum().item() / mask.sum().item()

    test_label = 'held-out' if test_held_out else 'IN-SAMPLE: test blocks were fitted'
    print(f'\n=== Done ===')
    print(f'Training time  : {elapsed:.1f}s ({elapsed/60:.1f} min)')
    n_fit = count_valid_samples(train_loader)         # fitted choices (missed trials excluded)
    print(f'Log marginal likelihood ({args.backend}, best epoch): {-best_marglik:.2f}  '
          f'({-best_marglik / len(fit_dat):.3f}/block, {-best_marglik / n_fit:.5f}/choice)')
    print(f'Test set ({len(test_dat)} blocks, {test_label}):')
    print(f"  NLL/block {test_m['nll_per_block']:.2f} | acc_paper {test_m['acc_paper'] * 100:.2f}% "
          f"| acc_pooled {test_m['acc_pooled'] * 100:.2f}% | argmax {test_argmax_acc * 100:.2f}%")
    if test_held_out and config.model_name == 'birnn' and config.network_params.hidden_size == 32:
        print('  Paper Memory-ANN: NLL/block 61.3, acc 68.3%')
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
        'debug':            debug,
        'fit_on':           args.fit_on,
        'n_fit_blocks':     len(fit_dat),
        'test_held_out':    test_held_out,
        # -log evidence (backend approximation); None if no marglik epoch ran
        'neg_log_marglik':  round(best_marglik, 4) if best_model_dict is not None else None,
        'n_fit_choices':    n_fit,
        'best_precision':   best_precision.tolist() if best_precision is not None else None,
        # v2: acc_paper = mean over blocks of exp(-NLL/150); acc_pooled = old formula
        'metric_version':     2,
        'test_nll_per_block': round(test_m['nll_per_block'], 4),
        'test_acc_paper':     round(test_m['acc_paper'], 4),
        'test_acc_pooled':    round(test_m['acc_pooled'], 4),
        'test_argmax_acc':    round(test_argmax_acc, 4),
        'training_time_s':  round(elapsed, 1),
    }
    run_name = (f'{args.model}_marglik_be={args.backend}_e={n_epochs}'
                f'_hs={config.network_params.hidden_size}_fit={args.fit_on}_seed={args.seed}_v2')
    os.makedirs('results', exist_ok=True)
    results_path = f'results/{run_name}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {results_path}')

    save_path = args.save or f'trained_models/{run_name}.pt'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'Weights saved to {save_path}')

    return wrapped, best_precision, best_marglik


if __name__ == '__main__':
    main()
