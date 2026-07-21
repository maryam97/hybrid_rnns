"""marglik_training.py — Marginal-likelihood optimisation for hybrid RNNs.

Adapted from model_recovery/marglik.py (Immer et al. style marglik training).
Fits a Laplace posterior (last-layer, all-Linear-Kron, or full) while jointly
optimising the prior precision via marginal likelihood.

This is appropriate here because our RNN/BiRNN models are small (~2,500
params for the paper-optimal BiRNN), so even the full Hessian is cheap.

Note: requires use_rnn_cell=False in NetworkParams so that all layers are
nn.Linear -- Kron-structured backends only Kronecker-factor nn.Linear/Conv,
not nn.RNN.

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
     below and left to the scalar/layerwise prior -- see laplace_compat.py.

CurvlinopsGGN remains available as a slower fallback (e.g. if a future
PyTorch release removes the non-full backward hook ASDL currently warns
about deprecating).

N (dataset size) bug: laplace-torch computes `N = len(train_loader.dataset)`
and uses it to normalise both the Kron-fit's implicit evidence-vs-complexity
tradeoff and (here) the explicit (delta*theta)@theta/N prior term below --
expecting N to count the same thing `M = len(y)` counts per batch, i.e.
(block, timestep) samples (see Curvlinops' `_rescale_kron_factors`).
`SequenceDataset.__len__()` must return the number of BLOCKS for DataLoader
indexing to work, so `len(train_loader.dataset)` under-counts the true
sample size by ~n_trials (~149x here). Left uncorrected, this makes the
marglik-optimal prior precision come out ~149x too concentrated on
regularisation relative to fit, which visibly destabilises training once
burn-in ends (loss jumps up and never fully recovers even after thousands of
further epochs) rather than settling into a value that actually improves
held-out accuracy the way marginal-likelihood optimisation is supposed to.
Both this file's explicit prior term and the periodic `lap.fit(train_loader)`
call use the corrected N (see count_valid_samples/fit_kron_with_correct_N in
laplace_compat.py) so the whole pipeline is internally consistent.
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

from .laplace_compat import (
    freeze_non_linear_parameters, count_valid_samples, fit_kron_with_correct_N,
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
    laplace=KronLLLaplace,
    backend=AsdlGGN,
):
    """Joint optimisation of model weights and prior precision via marginal likelihood.

    Parameters
    ----------
    model : nn.Module
        The wrapped sequence model (SequenceModelWrapper). Its forward(x)
        must return raw logits of shape (N, n_classes).
    train_loader : DataLoader
        Yields (x, y) batches over ALL available data (no train/test split).
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
        Pass FullLaplace + backend=CurvlinopsGGN for the full (non-Kron)
        Hessian over every parameter, bare ones included.
    backend : curvature backend
        Default: AsdlGGN. Use CurvlinopsGGN with `laplace=FullLaplace`, or as
        a slower fallback for `laplace=KronLaplace` (see module docstring).

    Returns
    -------
    best_model_dict : dict
        State dict of the model at the epoch with the best (lowest) marglik.
    best_precision : Tensor
        Prior precision at the best epoch.
    best_marglik : float
        Best (lowest) marginal likelihood value seen during training.
    """
    device = parameters_to_vector(model.parameters()).device
    # NOT len(train_loader.dataset) (block count) -- see module docstring.
    N = count_valid_samples(train_loader)
    H = len(list(model.parameters()))

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

            f    = model(X)
            # Missed trials get an arbitrary y=argmax(all-zero)=0 label from
            # SequenceDataset -- must exclude them from the likelihood term or
            # the model is trained to predict class 0 on every missed trial.
            # Matches fit_hyb_rnn.py's loss_fn, which masks the same way.
            mask = X[:, 1:, -1].reshape(-1).bool()
            loss = criterion(f[mask], y[mask]) + (0.5 * (delta * theta) @ theta) / N
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().item() / len(train_loader)

        t_train = time.perf_counter()

        # ---- eval-mode metrics -------------------------------------------------
        # For schedulefree Adam, training-mode params are the momentum interpolant
        # ("x"), not the true iterate ("z"). optimizer.eval() switches to "z" so
        # metrics reflect the actual solution, not the interpolant.
        model.eval()
        if _HAS_SCHEDULEFREE:
            optimizer.eval()

        nll_sum, n_correct, n_valid = 0.0, 0, 0
        with torch.no_grad():
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                f    = model(X)
                mask = X[:, 1:, -1].reshape(-1).bool()
                f_v, y_v = f[mask], y[mask]
                nll_sum  += F.cross_entropy(f_v, y_v, reduction='sum').item()
                n_correct += (torch.argmax(f_v, dim=-1) == y_v).sum().item()
                n_valid   += mask.sum().item()

        t_eval = time.perf_counter()

        paper_acc  = np.exp(-nll_sum / n_valid)
        argmax_acc = n_correct / n_valid
        losses.append(epoch_loss)
        epoch_secs   = t_eval - t_epoch
        elapsed_secs = t_eval - t_total
        print(f'MARGLIK[epoch={epoch}/{n_epochs}]: '
              f'loss={losses[-1]:.3f}  acc={paper_acc:.4f}  argmax_acc={argmax_acc:.4f}  '
              f'| train={t_train-t_epoch:.1f}s  eval={t_eval-t_train:.1f}s  epoch={epoch_secs:.1f}s  '
              f'elapsed={elapsed_secs:.0f}s ({elapsed_secs/60:.1f}min)',
              flush=True)

        # ---- marglik hyperparameter update ----
        # model is already in eval mode (and optimizer in eval mode if schedulefree)
        if epoch < n_epochs_burnin or (epoch % marglik_frequency) != 0:
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
                lap.fit(train_loader)

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

    print('MARGLIK: training complete.')
    return best_model_dict, best_precision, best_marglik
