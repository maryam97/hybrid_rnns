"""run_laplace.py — Post-hoc Laplace approximation for a trained RNN or BiRNN.

Requires:
    pip install laplace-torch

Workflow
--------
1. Load data on the paper split.
2. Load a model checkpoint (or train one on the train split).
3. Wrap with laplace_ready() so Laplace sees a standard (x -> logits) API.
4. Fit Laplace on --fit-on: the blocks the checkpoint was trained on (default:
   all 4,134 kept blocks, as for run_laplace_training.py checkpoints).
5. Optimise the prior precision by marginal likelihood and report the evidence
   (total, per block, per choice) -- the model-comparison quantity.
6. Evaluate MAP and Laplace (GLM) predictive on the 413 test blocks with the
   paper metric (in-sample when --fit-on all).

Default: exact full GGN over all weights (subset=all, hessian=full, FuncGGN).
Use this for the reported evidence: Kron is biased for the weight-shared
recurrent layers (~2k nats off for the BiRNN). It costs roughly 1 s per block.
Bare nn.Parameters (BiRNN's init/forget scalars) are in the full posterior but
excluded from Kron (kept at their MAP values).

Usage
-----
    python run_laplace.py --checkpoint trained_models/<marglik run>.pt                     # birnn, all data
    python run_laplace.py --checkpoint trained_models/<marglik run>.pt --model rnn
    python run_laplace.py --checkpoint trained_models/birnn_hs=32_steps=1000000_seed=42_v2.pt --fit-on train
    python run_laplace.py --checkpoint ... --hessian kron                                  # fast, approximate
"""

import argparse
import json
import os
from contextlib import nullcontext
import torch

from hybrid_rnns_pytorch.rnn_config   import get_rnn_config, get_birnn_config
from hybrid_rnns_pytorch.fit_hyb_rnn  import train
from hybrid_rnns_pytorch import hyb_rnn_utilities
from hybrid_rnns_pytorch.laplace_compat import (
    laplace_ready, make_dataloader, freeze_non_linear_parameters,
    rows_to_blocks, count_valid_samples, fit_kron_with_correct_N, FuncGGN,
)
from hybrid_rnns_pytorch.bi_rnn import BiRNN
from hybrid_rnns_pytorch.rnn    import RNN
from laplace import Laplace
from laplace.curvature import AsdlGGN


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Post-hoc Laplace approximation on a trained hybrid RNN.')
    p.add_argument('--model',       choices=['rnn', 'birnn'], default='birnn')
    p.add_argument('--hidden-size', type=int, default=None,
                   help='Hidden units per RNN layer. Default: paper config '
                        '(32 for birnn, 64 for rnn); set it to match a checkpoint.')
    p.add_argument('--no-debug',    action='store_true',
                   help='Run full training instead of quick debug training.')
    p.add_argument('--checkpoint',  type=str, default=None,
                   help='Path to a saved model state_dict (.pt). '
                        'If given, skip training and load weights directly.')
    p.add_argument('--dataset',     type=str,
                   default='hybrid_rnns_pytorch/data/openSourceRawDataset.csv')
    p.add_argument('--fit-on',      choices=['train', 'trainvalid', 'all'], default='all',
                   help='Blocks the posterior is fitted on (default: all 4,134). MUST be the '
                        'blocks the checkpoint was trained on: run_laplace_training.py -> its '
                        '--fit-on (default all); run_training.py -> train.')
    p.add_argument('--batch-size',  type=int, default=32,
                   help='Batch size for Laplace fit DataLoader.')
    p.add_argument('--n-samples',   type=int, default=200,
                   help='Posterior samples for predictive accuracy estimate.')
    p.add_argument('--subset',      choices=['last_layer', 'all'], default='all',
                   help='Which weights to put under the Laplace posterior. '
                        '"all" (default) covers every parameter — best for '
                        'BiRNN where uncertainty lives in recurrent layers. '
                        '"last_layer" is fast but a poor approximation.')
    p.add_argument('--hessian',     choices=['kron', 'full', 'diag'], default=None,
                   help='Hessian structure. Default: "kron" for last_layer, '
                        '"full" (exact GGN) for all. "kron" (all Linear layers, '
                        'AsdlGGN) is much faster but its evidence is biased for the '
                        'recurrent layers and it leaves out BiRNN\'s bare scalars.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(config):
    if config.model_name == 'birnn':
        return BiRNN(config.rnn_rl_params, config.network_params)
    return RNN(config.rnn_rl_params, config.network_params)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---------------------------------------------------------------- config
    # Paper configs (Memory-ANN / vanilla RNN), so run_training.py checkpoints
    # load. debug=False -> 1M steps and the paper batch size (Table 1).
    get_cfg = get_birnn_config if args.model == 'birnn' else get_rnn_config
    config  = get_cfg(debug=not args.no_debug)
    config.dataset_path = args.dataset
    if args.hidden_size is not None:
        config.network_params.hidden_size = args.hidden_size
    n_actions = config.network_params.n_actions

    # resolve subset / hessian structure / backend
    subset = args.subset
    if subset == 'last_layer' and args.model == 'birnn':
        # laplace-torch's last-layer predictive calls last_layer(feats) directly, which drops
        # the BiRNN value offset (reference weights: Laplace NLL 79.5 vs MAP 61.6 per block).
        raise SystemExit('--subset last_layer is wrong for birnn (drops the value stream); use --subset all')
    if args.checkpoint is None and args.fit_on != 'train':
        # Without a checkpoint the MAP is trained by fit_hyb_rnn.train on the train split;
        # a posterior is only valid around the MAP of the same data.
        raise SystemExit('Without --checkpoint the model is trained on the train split: '
                         'pass --fit-on train (or a checkpoint trained on the other blocks).')
    hessian = args.hessian or ('kron' if subset == 'last_layer' else 'full')
    # Kron: AsdlGGN (the value stream must stay in the graph and bare scalars are
    # frozen out, see laplace_compat.py). Full: exact GGN via torch.func.
    backend_cls = FuncGGN if hessian == 'full' else AsdlGGN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device          : {device}')
    print(f'Subset          : {subset}  |  Hessian: {hessian}  |  Backend: {backend_cls.__name__}')

    # ------------------------------------------------------------ data split
    print(f'Loading data from {config.dataset_path}')
    hum_dat  = hyb_rnn_utilities.load_osf_dataframe(config.dataset_path)   # rewards 0-1
    tensors  = hyb_rnn_utilities.format_data_for_model_training(
        hum_dat, n_actions=n_actions, split_file=config.split_path)
    train_dat = tensors['train_dat'].to(device)
    valid_dat = tensors['valid_dat'].to(device)
    test_dat  = tensors['test_dat'].to(device)

    # ------------------------------------------------------- train or load
    if args.checkpoint is not None:
        print(f'Loading checkpoint from {args.checkpoint}')
        model = _build_model(config).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print('Training model first...')
        _, model = train(config)
        model.to(device)

    model.eval()
    print('Model ready.')

    # ----------------------------------------------------------- wrap model
    # The value stream must stay in the gradient graph (see laplace_compat.py).
    wrapped = laplace_ready(model, n_actions=n_actions, detach_value=False)

    # --------------------------------------------------- build data loaders
    fit_dat = {'train': train_dat, 'trainvalid': torch.cat([train_dat, valid_dat]),
               'all': torch.cat([train_dat, valid_dat, test_dat])}[args.fit_on]
    print(f'Fitting posterior on {args.fit_on} ({len(fit_dat)} blocks); test metrics are '
          f'{"IN-SAMPLE" if args.fit_on == "all" else "held-out"}')
    train_loader = make_dataloader(
        fit_dat,
        n_actions  = n_actions,
        batch_size = 4 if hessian == 'full' else args.batch_size,
        shuffle    = True,
    )
    test_loader = make_dataloader(
        test_dat,
        n_actions  = n_actions,
        batch_size = 4,           # GLM predictive: Jacobians per batch, keep batches small
        shuffle    = False,   # keep test_dat block order for the per-block metrics
    )

    # --------------------------------------------------------------- Laplace
    print('\nFitting Laplace...')
    # Kron over subset='all' needs bare nn.Parameters hidden from the KFAC
    # backend (it only supports nn.Linear/Conv). This must wrap the
    # constructor too, not just .fit(): BaseLaplace.__init__ snapshots
    # self.params from model.parameters() (by requires_grad) immediately,
    # so freezing only around .fit() leaves a stale, larger self.params that
    # no longer matches the Kron blocks the (correctly-filtered) backend
    # produces -- causing a shape mismatch when accumulating batches.
    fit_scope = freeze_non_linear_parameters(wrapped) if (subset == 'all' and hessian == 'kron') \
        else nullcontext()
    with fit_scope:
        la = Laplace(
            wrapped,
            likelihood        = 'classification',
            subset_of_weights = subset,
            hessian_structure = hessian,
            backend           = backend_cls,
        )
        if hessian == 'kron':
            fit_kron_with_correct_N(la, train_loader)   # N = choices, not blocks
        else:
            la.fit(train_loader)
        print('Laplace fit complete.')

        # ------------------------------------------ optimise prior precision
        print('Optimising prior precision...')
        la.optimize_prior_precision(method='marglik')
        marglik = la.log_marginal_likelihood(la.prior_precision).item()
    print(f'Optimised prior precision: {la.prior_precision.item():.4f}')
    n_fit = count_valid_samples(train_loader)
    print(f'Log marginal likelihood  : {marglik:.4f}  ({len(fit_dat)} blocks, {n_fit} choices: '
          f'{marglik / len(fit_dat):.3f}/block, {marglik / n_fit:.5f}/choice)')

    # ---------------------------------------------------------- evaluate
    print('\nEvaluating on test set...')
    nll_blocks = {'map': [], 'laplace': []}
    n_correct = {'map': 0, 'laplace': 0}

    # GLM predictive on 4-block batches (~0.7 s/block). Wrapper outputs are rows for non-missed targets only (y_batch is filtered
    # the same way); rows_to_blocks puts them back on the (B, T-1, A) grid.
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            batch_probs = {
                # MAP model (plain softmax); computed first, before la() samples weights
                'map':     wrapped.predict_proba(x_batch),
                # Posterior predictive (mean over Laplace posterior)
                # GLM (linearised) predictive: the one consistent with a GGN posterior.
                # pred_type='nn' samples weights through the 149-step recurrence and is far
                # worse than MAP; link_approx='probit' ignores the softmax-shift correlation.
                'laplace': la(x_batch, pred_type='glm', link_approx='mc',
                              n_samples=args.n_samples),
            }
            for k, probs in batch_probs.items():
                block_probs = rows_to_blocks(probs, x_batch, n_actions)
                nll_blocks[k].append(hyb_rnn_utilities.block_nll(block_probs, x_batch, n_actions).cpu())
                n_correct[k] += (probs.argmax(dim=-1) == y_batch).sum().item()

    n_valid = count_valid_samples(test_loader)
    print(f'\n=== Test-set results ({len(test_dat)} blocks, '
          f'{"IN-SAMPLE" if args.fit_on == "all" else "held-out"}) ===')
    print('acc_paper = mean over blocks of exp(-NLL_block/150) (paper metric); '
          'acc_pooled = exp(-total NLL / valid trials)')
    test_metrics = {}
    for k, name in (('map', 'MAP'), ('laplace', f'Laplace ({args.n_samples} MC samples)')):
        m = hyb_rnn_utilities.accuracy_metrics(torch.cat(nll_blocks[k]), test_dat.cpu(), n_actions)
        test_metrics[k] = {**m, 'argmax_acc': n_correct[k] / n_valid}
        print(f"{name:24s}: NLL/block {m['nll_per_block']:.2f} | acc_paper {m['acc_paper'] * 100:.2f}% "
              f"| acc_pooled {m['acc_pooled'] * 100:.2f}% | argmax {n_correct[k] / n_valid * 100:.2f}%")
    if args.fit_on != 'all' and config.model_name == 'birnn' and config.network_params.hidden_size == 32:
        print('Paper Memory-ANN: NLL/block 61.3, acc 68.3%')
    print(f'Prior precision   : {la.prior_precision.item():.4f}')

    stem = (os.path.splitext(os.path.basename(args.checkpoint))[0] if args.checkpoint
            else f'{args.model}_hs={config.network_params.hidden_size}')
    tag = f'{stem}_posthoc_fit={args.fit_on}_{hessian}'
    results = {
        'checkpoint': args.checkpoint, 'model': args.model,
        'hidden_size': config.network_params.hidden_size,
        'subset': subset, 'hessian': hessian, 'fit_on': args.fit_on,
        'n_fit_blocks': len(fit_dat), 'n_fit_choices': n_fit,
        'log_marglik': marglik, 'log_marglik_per_block': marglik / len(fit_dat),
        'log_marglik_per_choice': marglik / n_fit,
        'prior_precision': la.prior_precision.tolist(),
        'test_held_out': args.fit_on != 'all',
        # v2: acc_paper = mean over blocks of exp(-NLL/150); acc_pooled = old formula
        'metric_version': 2,
        'test_map': test_metrics['map'], 'test_laplace': test_metrics['laplace'],
    }
    os.makedirs('results', exist_ok=True)
    with open(f'results/{tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to results/{tag}.json')

    return la, model


if __name__ == '__main__':
    main()
