"""run_laplace.py — Post-hoc Laplace approximation for a trained RNN or BiRNN.

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

Default: full-parameter Laplace (subset=all, hessian=full, CurvlinopsGGN).
This covers ALL weights including BiRNN's bare nn.Parameters — a much better
approximation than last-layer only. Post-hoc fitting is one shot (not per epoch),
so even full Hessian is fast (seconds to minutes).

subset='all' requires the model wrapper's value stream to be un-detached
(detach_value=False, wired below) -- otherwise value_rnn_linear/value_out_linear
and BiRNN's bare scalar nn.Parameters get silently zeroed-out curvature, since
Curvlinops' Jacobian-based backends don't error on unused parameters the way
a plain autograd.grad() call would. See laplace_compat.py for details.

Fitting Kron/full over the whole model is NOT the slow part (~15s, <1GB even
over the full 3,300-block training set, either backend). The evaluation step
uses pred_type='nn' (MC weight samples + plain forward passes), not 'glm':
'glm' computes an exact per-sample Jacobian-based predictive covariance,
and our wrapper flattens every (block, timestep) pair into one giant sample
dimension (batch_size=32 blocks * ~149 timesteps =~ 4,768 samples/batch), so
'glm' measured ~5-6 minutes and several GB of RAM for a SINGLE test batch.
'nn' measured 0.3s for the same batch -- use --n-samples to trade accuracy
of the MC estimate for speed.

Usage
-----
    python run_laplace.py --checkpoint model.pt --no-debug        # full, birnn
    python run_laplace.py --checkpoint model.pt --model rnn       # full, rnn
    python run_laplace.py --checkpoint model.pt --hessian kron    # cheaper, all Linear layers
    python run_laplace.py --checkpoint model.pt --subset last_layer  # fast, poor approx
"""

import argparse
from contextlib import nullcontext
import torch
import pandas as pd

from hybrid_rnns_pytorch.rnn_config   import get_config
from hybrid_rnns_pytorch.fit_hyb_rnn  import train
from hybrid_rnns_pytorch import hyb_rnn_utilities
from hybrid_rnns_pytorch.laplace_compat import (
    laplace_ready, make_dataloader, freeze_non_linear_parameters,
)
from hybrid_rnns_pytorch.bi_rnn import BiRNN
from hybrid_rnns_pytorch.rnn    import RNN
from laplace.curvature import AsdlGGN, CurvlinopsGGN


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
                   default='hybrid_rnns_pytorch/data/openSourceRawDataset.csv')
    p.add_argument('--batch-size',  type=int, default=32,
                   help='Batch size for Laplace fit DataLoader.')
    p.add_argument('--n-samples',   type=int, default=50,
                   help='Posterior samples for predictive accuracy estimate.')
    p.add_argument('--subset',      choices=['last_layer', 'all'], default='all',
                   help='Which weights to put under the Laplace posterior. '
                        '"all" (default) covers every parameter — best for '
                        'BiRNN where uncertainty lives in recurrent layers. '
                        '"last_layer" is fast but a poor approximation.')
    p.add_argument('--hessian',     choices=['kron', 'full', 'diag'], default=None,
                   help='Hessian structure. Default: "kron" for last_layer, '
                        '"full" for all. With subset=all, "kron" (all Linear '
                        'layers, AsdlGGN) is cheaper than "full" but '
                        'leaves BiRNN\'s bare nn.Parameters to the prior only; '
                        '"full" is the only structure that gives them real '
                        'curvature.')
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

    # resolve subset / hessian structure / backend
    subset = args.subset
    hessian = args.hessian or ('kron' if subset == 'last_layer' else 'full')
    # AsdlGGN works fine for subset='all' + hessian='kron' -- including when
    # habit_rnn_linear/value_rnn_linear are called repeatedly per forward pass
    # (149x, once per timestep) -- PROVIDED the wrapper's value stream is not
    # detached (detach_value=False, wired below): ASDL's hooks only populate
    # a module's `.fisher` stat if it actually receives a backward gradient,
    # so a detached value stream previously looked like a weight-sharing
    # crash (`TypeError: 'float' * NoneType`) but was really this. AsdlGGN is
    # ~2.5x faster than CurvlinopsGGN here. Bare nn.Parameters still can't be
    # Kron-factored by either backend, so they're frozen out of the fit below
    # and left to the prior (see freeze_non_linear_parameters). Only 'full'
    # (no Kron structure) needs CurvlinopsGGN.
    backend_cls = CurvlinopsGGN if hessian == 'full' else AsdlGGN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device          : {device}')
    print(f'Subset          : {subset}  |  Hessian: {hessian}  |  Backend: {backend_cls.__name__}')

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

    # detach_value=False for subset='all': the value stream (value_rnn_linear,
    # value_out_linear, and any bare init/forget scalars) must stay in the
    # gradient graph or it silently gets zero curvature (see laplace_compat.py).
    wrapped = laplace_ready(model, n_actions=config.network_params.n_actions,
                             detach_value=(subset != 'all'))

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
        la.fit(train_loader)
        print('Laplace fit complete.')

        # ------------------------------------------ optimise prior precision
        print('Optimising prior precision...')
        la.optimize_prior_precision(
            method       = 'marglik',
            pred_type    = 'glm',
            link_approx  = 'probit',
        )
        marglik = la.log_marginal_likelihood(la.prior_precision).item()
    print(f'Optimised prior precision: {la.prior_precision.item():.4f}')
    print(f'Log marginal likelihood  : {marglik:.4f}')

    # ---------------------------------------------------------- evaluate
    print('\nEvaluating on test set...')
    all_probs = []
    all_y     = []

    # pred_type='nn' (MC-sample weights from the Laplace posterior, then a
    # plain forward pass per sample) instead of 'glm'. 'glm' computes an
    # exact per-sample Jacobian-based predictive covariance, which is
    # O(n_samples_in_batch * n_params) -- and our wrapper flattens every
    # (block, timestep) pair into one giant "sample" dimension (batch_size=32
    # blocks * ~149 timesteps =~ 4,768 samples/batch), so 'glm' took ~5-6
    # minutes and several GB of RAM for a SINGLE test batch. 'nn' only
    # samples args.n_samples weight vectors and does ordinary forward
    # passes: ~1000x cheaper (measured 0.3s vs 336s per batch) and this cost
    # is independent of which curvature backend (ASDL/Curvlinops) fit the
    # Kron structure -- fitting itself was never the bottleneck.
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            # Posterior predictive (mean over Laplace posterior)
            probs = la(x_batch, pred_type='nn', link_approx='mc',
                       n_samples=args.n_samples)
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
