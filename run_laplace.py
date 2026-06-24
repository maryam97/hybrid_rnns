"""run_laplace.py — Fit last-layer Laplace to a trained RNN or BiRNN.

Requires:
    pip install laplace-torch

Workflow
--------
1. Load data and split by participant (same as training).
2. Train (or load) a model checkpoint.
3. Wrap with laplace_ready() so Laplace sees a standard (x -> logits) API.
4. Fit Laplace on the training set.
5. Optimise prior precision via marginal likelihood.
6. Evaluate calibration on the test set.

Usage
-----
    python run_laplace.py                           # debug mode, birnn
    python run_laplace.py --model rnn --no-debug    # full run, rnn
    python run_laplace.py --checkpoint model.pt     # load saved weights
"""

import argparse
import torch
import pandas as pd

from hybrid_rnns_pytorch.rnn_config   import get_config
from hybrid_rnns_pytorch.fit_hyb_rnn  import train
from hybrid_rnns_pytorch import hyb_rnn_utilities
from hybrid_rnns_pytorch.laplace_compat import laplace_ready, make_dataloader
from hybrid_rnns_pytorch.bi_rnn import BiRNN
from hybrid_rnns_pytorch.rnn    import RNN


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Last-layer Laplace on a trained hybrid RNN.')
    p.add_argument('--model',       choices=['rnn', 'birnn'], default='birnn')
    p.add_argument('--no-debug',    action='store_true',
                   help='Run full training instead of quick debug training.')
    p.add_argument('--checkpoint',  type=str, default=None,
                   help='Path to a saved model state_dict (.pt). '
                        'If given, skip training and load weights directly.')
    p.add_argument('--dataset',     type=str,
                   default='hybrid_rnns_pytorch/data/openSourceRawDataset_v2.csv')
    p.add_argument('--batch-size',  type=int, default=32,
                   help='Batch size for Laplace fit DataLoader.')
    p.add_argument('--n-samples',   type=int, default=50,
                   help='Posterior samples for predictive accuracy estimate.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(config):
    if config.model_name == 'birnn':
        return BiRNN(config.rnn_rl_params, config.network_params)
    return RNN(config.rnn_rl_params, config.network_params)


def accuracy_from_probs(probs: torch.Tensor, y: torch.Tensor) -> float:
    """Geometric-mean probability assigned to each correct class.

    Matches the paper formula: acc = exp(-mean NLL per valid trial).
    """
    correct_probs = probs[torch.arange(len(y)), y].clamp(min=1e-8)
    return correct_probs.log().mean().exp().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---------------------------------------------------------------- config
    config             = get_config()
    config.model_name  = args.model
    config.dataset_path = args.dataset
    if args.no_debug:
        config.debug             = False
        config.n_training_steps  = int(1e6)
        config.batch_size        = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ------------------------------------------------------------ data split
    print(f'Loading data from {config.dataset_path}')
    hum_dat  = pd.read_csv(config.dataset_path)
    tensors  = hyb_rnn_utilities.format_data_for_model_training(hum_dat)
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
    try:
        from laplace import Laplace
    except ImportError:
        raise ImportError(
            "laplace-torch is not installed. Run:\n"
            "    pip install laplace-torch"
        )

    wrapped = laplace_ready(model, n_actions=config.network_params.n_actions)
    print(f'Wrapper last linear: {wrapped.model.output_linear if isinstance(model, RNN) else wrapped.model.habit_out_linear}')

    # --------------------------------------------------- build data loaders
    train_loader = make_dataloader(
        train_dat,
        n_actions  = config.network_params.n_actions,
        batch_size = args.batch_size,
        shuffle    = True,
    )
    test_loader = make_dataloader(
        test_dat,
        n_actions  = config.network_params.n_actions,
        batch_size = args.batch_size,
        shuffle    = False,
    )

    # --------------------------------------------------------------- Laplace
    print('\nFitting Laplace...')
    la = Laplace(
        wrapped,
        likelihood          = 'classification',
        subset_of_weights   = 'last_layer',
        hessian_structure   = 'kron',
    )
    la.fit(train_loader)
    print('Laplace fit complete.')

    # ------------------------------------------ optimise prior precision
    print('Optimising prior precision...')
    la.optimize_prior_precision(
        method       = 'marglik',
        pred_type    = 'glm',
        link_approx  = 'probit',
    )
    print(f'Optimised prior precision: {la.prior_precision.item():.4f}')

    # ---------------------------------------------------------- evaluate
    print('\nEvaluating on test set...')
    all_probs = []
    all_y     = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            # Posterior predictive (mean over Laplace posterior)
            probs = la(x_batch, pred_type='glm', link_approx='probit')
            all_probs.append(probs.cpu())
            all_y.append(y_batch.cpu())

    all_probs = torch.cat(all_probs, dim=0)   # (N_total, n_actions)
    all_y     = torch.cat(all_y,     dim=0)   # (N_total,)

    # MAP accuracy (argmax)
    map_acc = (all_probs.argmax(dim=-1) == all_y).float().mean().item()
    # Paper-style accuracy (geometric mean probability assigned to correct action)
    paper_acc = accuracy_from_probs(all_probs, all_y)

    print(f'\n=== Laplace test-set results ===')
    print(f'Argmax accuracy   : {map_acc * 100:.2f}%')
    print(f'Paper-style acc   : {paper_acc * 100:.2f}%  '
          f'(paper target ~68.3% for BiRNN)')
    print(f'Prior precision   : {la.prior_precision.item():.4f}')

    # ------------------------------------------ optional: save checkpoint
    save_path = f'{args.model}_trained.pt'
    torch.save(model.state_dict(), save_path)
    print(f'\nModel weights saved to {save_path}')

    return la, model


if __name__ == '__main__':
    main()
