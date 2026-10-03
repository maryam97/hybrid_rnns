"""marglik_training.py — Marginal-likelihood optimisation for hybrid RNNs.

Adapted from model_recovery/marglik.py (Immer et al. style marglik training).
Fits a Laplace posterior (last-layer, all-Linear-Kron, or full) while jointly
optimising the prior precision via marginal likelihood.

The default is all-Linear Kron (KronLaplace + AsdlGGN). Kron is a cheap
approximation that is fine for choosing the prior precision during training,
but its evidence is biased for the weight-shared recurrent layers (off by ~2k
nats for the BiRNN), so report the evidence from an exact GGN post hoc
(run_laplace.py --hessian full). Last-layer Laplace (KronLLLaplace) is only
valid for the RNN: for the BiRNN it drops the value offset (see laplace_compat).

`laplace=KronLaplace` (all Linear layers, both habit and value streams) works
fine with `backend=AsdlGGN` -- and is ~2.5x faster than CurvlinopsGGN for
these models (6.7s vs 16.2s to fit BiRNN's Kron over the full training set)
-- PROVIDED two things hold:
  1. The wrapper's value stream must not be detached (`detach_value=False`
     when the model was wrapped via laplace_ready/SequenceModelWrapper).
     ASDL's hooks only populate a module's `.fisher` stat if that module
     actually receives a backward gradient; if the value stream is detached
     (as it is for last-layer-only fits), value_rnn_linear/value_out_linear
     never get one, and multiplying Kron by a None curvature factor raises
     `TypeError: 'float' * NoneType`. This was initially (mis)diagnosed as
     ASDL being fundamentally unable to Kron-factor a Linear layer called
     repeatedly per forward pass (our manual per-timestep recurrence reuses
     habit_rnn_linear/value_rnn_linear 149 times) -- it isn't; that pattern
     works fine once the value stream is left in the gradient graph.
  2. Bare nn.Parameters (BiRNN's scalar init/forget values) still can't be
     Kron-factored by ASDL (it only walks nn.Linear/nn.Conv modules), so
     they're excluded from curvature via freeze_non_linear_parameters()
     below: they are not part of the Kron posterior (kept at their MAP
     values) and only the SGD L2 term sees them -- see laplace_compat.py.

CurvlinopsGGN is NOT a usable fallback for Kron: with missed-trial rows dropped,
the number of rows per batch is not a multiple of the number of blocks, which
curvlinops' KFAC requires.

N (dataset size) bug: laplace-torch computes `N = len(train_loader.dataset)`
and uses it to normalise both the Kron-fit's implicit evidence-vs-complexity
tradeoff and (here) the explicit (delta*theta)@theta/N prior term below --
expecting N to count the same thing `M = len(y)` counts per batch, i.e.
(block, timestep) samples (see Curvlinops' `_rescale_kron_factors`).
`SequenceDataset.__len__()` must return the number of BLOCKS for DataLoader
indexing to work, so `len(train_loader.dataset)` under-counts the true
sample size by ~n_trials (~149x here). Left uncorrected, this makes the
explicit prior term below ~149x too strong, which visibly destabilises
training once burn-in ends. Both this file's prior term and the periodic Kron
fit use the corrected N (count_valid_samples; fit_kron_with_correct_N rescales
only the Kron input factors, not the bias blocks).
"""

from contextlib import nullcontext
from copy import deepcopy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils import parameters_to_vector
from laplace import KronLaplace, KronLLLaplace, FullLaplace
from laplace.curvature import AsdlGGN, CurvlinopsGGN

from . import hyb_rnn_utilities
from .laplace_compat import (
    freeze_non_linear_parameters, count_valid_samples, fit_kron_with_correct_N,
    rows_to_blocks,
)

try:
    import schedulefree
    _HAS_SCHEDULEFREE = True
except ImportError:
    _HAS_SCHEDULEFREE = False


def expand_prior_precision(prior_prec, model):
    """Expand scalar / layerwise / diagonal prior precision to per-parameter vector."""
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten()
                          for delta, m in zip(prior_prec, model.parameters())])


def marglik_optimization(
    model,
    train_loader,
    prior_structure='scalar',
    prior_prec_init=1.,
    n_epochs=100,
    lr=1e-3,
    n_epochs_burnin=0,
    n_hypersteps=100,
    marglik_frequency=1,
    lr_hyp=1e-1,
    laplace=KronLaplace,
    backend=AsdlGGN,
    compile_model=False,
    eval_frequency=1,
):
    """Joint optimisation of model weights and prior precision via marginal likelihood.

    Parameters
    ----------
    model : nn.Module
        The wrapped sequence model (SequenceModelWrapper). Its forward(x)
        must return raw logits of shape (N, n_classes).
    train_loader : DataLoader
        From make_dataloader(): yields (x, y) batches over the data the
        posterior is fitted on (run_laplace_training.py: train+valid blocks by
        default, --fit-on all adds test). No validation set is needed.
    prior_structure : str
        'scalar' (default), 'layerwise', or 'diagonal'.
        'scalar' is recommended: KronLLLaplace has one Kron block (the last
        linear), so a scalar prior_precision is the only unambiguous choice.
    prior_prec_init : float
        Initial prior precision (before optimisation).
    n_epochs : int
        Total training epochs.
    lr : float
        Learning rate for the model weights optimiser.
    n_epochs_burnin : int
        Epochs to train without marglik updates (let weights stabilise first).
    n_hypersteps : int
        Inner-loop steps on the prior precision per marglik update.
    marglik_frequency : int
        How often (in epochs) to run a marglik update.
    lr_hyp : float
        Learning rate for the prior precision hyperparameter (0.01–0.1 typical).
    laplace : Laplace class
        Default: KronLLLaplace (last-layer Kron — hooks only the final nn.Linear).
        Pass KronLaplace + backend=AsdlGGN for a full-Linear-layer Kron fit
        covering both the habit and value streams (see module docstring;
        requires the model to be wrapped with detach_value=False). Bare
        nn.Parameters are still excluded from curvature and covered only by
        the scalar/layerwise prior via the training loop's L2 term.
        Pass FullLaplace + backend=laplace_compat.FuncGGN for the full (non-Kron)
        Hessian over every parameter, bare ones included.
    backend : curvature backend
        Default: AsdlGGN. Use laplace_compat.FuncGGN with `laplace=FullLaplace`
        (exact GGN; CurvlinopsGGN gives the same matrix far more slowly).
    compile_model : bool
        torch.compile() the plain SGD training/eval forward pass only. Does
        NOT touch the forward calls Laplace/ASDL/Curvlinops make internally
        during `.fit()` / `log_marginal_likelihood()` -- those rely on
        forward/backward hooks to capture per-layer curvature stats (see
        laplace_compat.py and the module docstring), and compiling the graph
        those hooks fire inside risks silently breaking that capture. The
        recurrence's manual per-timestep Python loop (see laplace_compat.py's
        SequenceModelWrapper) is exactly the kind of many-tiny-ops pattern
        torch.compile targets, so this is worth trying for the SGD passes,
        which dominate wall time when marglik_frequency > 1.
    eval_frequency : int
        Evaluate the training-set metrics every this many epochs (and on every
        marglik epoch and the last one). Default 1 = every epoch.

    Returns
    -------
    best_model_dict : dict
        State dict of the model at the epoch with the best (lowest) marglik.
    best_precision : Tensor
        Prior precision at the best epoch.
    best_marglik : float
        Best (lowest) NEGATIVE log marginal likelihood seen during training.
    """
    device = parameters_to_vector(model.parameters()).device
    if device.type == 'cpu':
        # See fit_hyb_rnn.py: the manual per-timestep recurrence does ~149
        # tiny ops per forward call, and PyTorch's default intra-op thread
        # pool adds per-op sync overhead that dwarfs the op cost at this
        # size, so single-threaded outperforms the multi-threaded default.
        torch.set_num_threads(1)
    # NOT len(train_loader.dataset) (block count) -- see module docstring.
    N = count_valid_samples(train_loader)
    H = len(list(model.parameters()))

    # Compiled only for the plain SGD forward pass below -- NOT used for the
    # `laplace(model, ...)` construction/fit further down, which needs the
    # uncompiled graph for its forward/backward hooks to reliably capture
    # curvature (see compile_model in the docstring above).
    forward_fn = torch.compile(model) if compile_model else model

    # ---- differentiable hyperparameter: log prior precision ----
    log_prior_prec_init = np.log(prior_prec_init)
    if prior_structure == 'scalar':
        log_prior_prec = log_prior_prec_init * torch.ones(1, device=device)
    elif prior_structure == 'layerwise':
        log_prior_prec = log_prior_prec_init * torch.ones(H, device=device)
    elif prior_structure == 'diagonal':
        P = len(parameters_to_vector(model.parameters()))
        log_prior_prec = log_prior_prec_init * torch.ones(P, device=device)
    else:
        raise ValueError(f'Invalid prior_structure: {prior_structure!r}')
    log_prior_prec.requires_grad_(True)

    criterion = CrossEntropyLoss(reduction='mean')

    # ---- model optimiser ----
    if _HAS_SCHEDULEFREE:
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr)
        optimizer.train()
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # ---- hyperparameter optimiser ----
    hyper_optimizer = torch.optim.Adam([log_prior_prec], lr=lr_hyp)

    best_marglik    = np.inf
    best_model_dict = None
    best_precision  = None
    losses   = []
    margliks = []

    t_total = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        t_epoch = time.perf_counter()

        # ---- training pass ----
        model.train()
        if _HAS_SCHEDULEFREE:
            optimizer.train()
        epoch_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            prior_prec = torch.exp(log_prior_prec).detach()
            theta      = parameters_to_vector(model.parameters())
            delta      = expand_prior_precision(prior_prec, model)

            # Missed targets are already dropped from both f and y (see
            # SequenceModelWrapper / make_dataloader in laplace_compat.py).
            f    = forward_fn(X)
            loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().item() / len(train_loader)

        t_train = time.perf_counter()

        # Only every eval_frequency epochs (and on marglik epochs): a full pass
        # costs ~30% of an epoch. The next epoch starts in train mode again.
        do_marglik = epoch >= n_epochs_burnin and epoch % marglik_frequency == 0
        if not (do_marglik or epoch % eval_frequency == 0 or epoch == n_epochs):
            losses.append(epoch_loss)
            continue

        # ---- eval-mode metrics -------------------------------------------------
        # For schedulefree Adam, training-mode params are the momentum interpolant
        # ("x"), not the true iterate ("z"). optimizer.eval() switches to "z" so
        # metrics reflect the actual solution, not the interpolant.
        model.eval()
        if _HAS_SCHEDULEFREE:
            optimizer.eval()

        # Monitoring on the TRAINING data (not held out): paper formula
        # (smoothed NLL per block, mean_b exp(-NLL_b/150)) plus argmax accuracy.
        nll_blocks, n_correct, n_valid = [], 0, 0
        with torch.no_grad():
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                f    = forward_fn(X)
                n_actions = f.shape[-1]
                probs = rows_to_blocks(F.softmax(f, dim=-1), X, n_actions)
                nll_blocks.append(hyb_rnn_utilities.block_nll(probs, X, n_actions))
                n_correct += (torch.argmax(f, dim=-1) == y).sum().item()
                n_valid   += len(y)

        t_eval = time.perf_counter()

        train_m = hyb_rnn_utilities.accuracy_metrics(
            torch.cat(nll_blocks), train_loader.dataset.tensor, n_actions)
        train_nll_per_block = train_m['nll_per_block']
        train_acc           = train_m['acc_paper']
        train_argmax_acc    = n_correct / n_valid
        losses.append(epoch_loss)
        epoch_secs   = t_eval - t_epoch
        elapsed_secs = t_eval - t_total
        print(f'MARGLIK[epoch={epoch}/{n_epochs}]: '
              f'loss={losses[-1]:.3f}  train_nll_per_block={train_nll_per_block:.2f}  '
              f'train_acc={train_acc:.4f}  train_argmax_acc={train_argmax_acc:.4f}  '
              f'| train={t_train-t_epoch:.1f}s  eval={t_eval-t_train:.1f}s  epoch={epoch_secs:.1f}s  '
              f'elapsed={elapsed_secs:.0f}s ({elapsed_secs/60:.1f}min)',
              flush=True)

        # ---- marglik hyperparameter update ----
        # model is already in eval mode (and optimizer in eval mode if schedulefree)
        if not do_marglik:
            # restore training mode for next epoch
            model.train()
            if _HAS_SCHEDULEFREE:
                optimizer.train()
            continue

        t_marglik_start = time.perf_counter()

        # Do NOT pass prior_precision to the constructor: KronLLLaplace.fit()
        # reasserts self.prior_precision before kron._H is built, so any tensor
        # triggers a length-validation error.  We pass it only to
        # log_marginal_likelihood below, after the Kron structure exists.
        #
        # Kron-structured backends (KFAC) only support nn.Linear/nn.Conv, so
        # any bare nn.Parameter still requiring grad must be hidden from them
        # or Curvlinops raises "Found parameters in un-supported layers".
        # FullLaplace has no such restriction, so it's left untouched.
        needs_linear_only = laplace in (KronLaplace, KronLLLaplace)
        with freeze_non_linear_parameters(model) if needs_linear_only else nullcontext():
            lap = laplace(model, 'classification', backend=backend)
            if needs_linear_only:
                fit_kron_with_correct_N(lap, train_loader)
            else:
                # Full GGN (FuncGGN) holds per-block Jacobians: ~0.5 GB per block in a
                # batch, so fit on 4-block batches, not the SGD batch size.
                lap.fit(torch.utils.data.DataLoader(
                    train_loader.dataset, batch_size=4, collate_fn=train_loader.collate_fn))

            for _ in range(n_hypersteps):
                hyper_optimizer.zero_grad()
                prior_prec = torch.exp(log_prior_prec)
                marglik    = -lap.log_marginal_likelihood(prior_prec)
                marglik.backward()
                hyper_optimizer.step()
                margliks.append(marglik.item())

        t_marglik_end = time.perf_counter()

        if margliks[-1] < best_marglik:
            best_model_dict = deepcopy(model.state_dict())
            best_precision  = deepcopy(prior_prec.detach())
            best_marglik    = margliks[-1]
            print(f'MARGLIK[epoch={epoch}/{n_epochs}]: marglik={best_marglik:.2f}  '
                  f'[new best — saving]  hessian+hyp={t_marglik_end-t_marglik_start:.1f}s',
                  flush=True)
        else:
            print(f'MARGLIK[epoch={epoch}/{n_epochs}]: marglik={margliks[-1]:.2f}  '
                  f'[no improvement over {best_marglik:.2f}]  '
                  f'hessian+hyp={t_marglik_end-t_marglik_start:.1f}s',
                  flush=True)

        # restore training mode for next epoch
        model.train()
        if _HAS_SCHEDULEFREE:
            optimizer.train()

    if _HAS_SCHEDULEFREE:
        optimizer.eval()   # leave the weights at the schedule-free iterate, not the interpolant
    print('MARGLIK: training complete.')
    return best_model_dict, best_precision, best_marglik
