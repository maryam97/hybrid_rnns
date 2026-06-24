"""Training script — PyTorch port of hybrid_rnns_reward_learning/fit_hyb_rnn.py.

Fits a CogMod, RNN, or BiRNN to human bandit-task behaviour using
cross-entropy loss and AdamW.

Usage
-----
    python fit_hyb_rnn.py
or import and call train(config) directly.
"""

import time
import torch
import pandas as pd

from . import hyb_rnn_utilities
from .rnn_config import get_config
from .bi_rnn import BiRNN
from .cogmod import CogMod
from .rnn import RNN


def train(config=None):
    """Fit one model (cogmod / RNN / biRNN) to human bandit task behaviour."""

    if config is None:
        config = get_config()

    # ------------------------------------------------------------------ device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    torch.manual_seed(config.random_seed)

    # ----------------------------------------------------------- build model
    if config.model_name == 'cogmod':
        print('Using CogMod to fit data.')
        model = CogMod(config.rnn_rl_params, config.network_params)
    elif config.model_name == 'birnn':
        print('Using BiRNN to fit data.')
        model = BiRNN(config.rnn_rl_params, config.network_params)
    elif config.model_name == 'rnn':
        print('Using RNN to fit data.')
        model = RNN(config.rnn_rl_params, config.network_params)
    else:
        raise ValueError(f'Unknown model_name: {config.model_name!r}')

    model.to(device)

    # ------------------------------------------------------- load & split data
    print(f'Loading data from {config.dataset_path}')
    hum_dat = pd.read_csv(config.dataset_path)
    tensors = hyb_rnn_utilities.format_data_for_model_training(hum_dat)

    train_dat = tensors['train_dat'].to(device)
    valid_dat = tensors['valid_dat'].to(device)
    test_dat  = tensors['test_dat'].to(device)
    print(f'Training blocks: {len(train_dat)}')

    # ---------------------------------------------------------------- optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ---------------------------------------------------------------- loss fn
    # Data layout: [one-hot action (n_actions) | reward (1) | valid mask (1)]
    # The model only sees the first n_actions+1 columns (action + reward).
    n_actions = config.network_params.n_actions

    def loss_fn(batch_dat: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss between model-predicted and observed behaviour.

        batch_dat: (batch, time, n_actions + 2)  — last column is valid mask
        """
        model_input = batch_dat[:, :, :n_actions + 1]          # strip mask column
        action_probs_seq, _ = model.unroll(model_input)         # (batch, time, n_actions)

        # Smooth to avoid log(0) — matches the original JAX code exactly
        action_probs_seq = (1 - 1e-5) * action_probs_seq + 5e-4

        # Targets and validity mask at t+1 (we predict next action from current input)
        targets = batch_dat[:, 1:, :n_actions]          # (batch, time-1, n_actions)
        mask    = batch_dat[:, 1:, n_actions + 1]        # (batch, time-1)  1=valid, 0=missed
        preds   = action_probs_seq[:, :-1]               # (batch, time-1, n_actions)

        loss = -(torch.log(preds) * targets * mask.unsqueeze(-1)).sum() / batch_dat.shape[0]
        return loss

    def accuracy_fn(batch_dat: torch.Tensor) -> float:
        """Paper accuracy: exp(-mean NLL per trial) — equation from Methods p.13.

        This is the geometric mean probability assigned to the correct action,
        NOT the fraction of argmax-correct predictions.

        batch_dat: (batch, time, n_actions + 2)  — last column is valid mask
        """
        model_input = batch_dat[:, :, :n_actions + 1]
        action_probs_seq, _ = model.unroll(model_input)          # (batch, time, n_actions)
        action_probs_seq = (1 - 1e-5) * action_probs_seq + 5e-4

        targets = batch_dat[:, 1:, :n_actions]                   # (batch, time-1, n_actions)
        mask    = batch_dat[:, 1:, n_actions + 1]                # (batch, time-1)
        preds   = action_probs_seq[:, :-1]                       # (batch, time-1, n_actions)

        # NLL per step: -log p(chosen action)
        step_nll = -(torch.log(preds) * targets).sum(dim=-1)     # (batch, time-1)

        # Mean NLL per trial averaged over valid steps only
        n_valid       = mask.sum()
        mean_nll      = (step_nll * mask).sum() / n_valid

        # Paper formula: acc = exp(-L / (bs * ntrials)), ntrials = 150
        # Equivalent here: exp(-mean NLL per valid trial)
        return torch.exp(-mean_nll).item()

    # ---------------------------------------------------------------- training
    rng = torch.Generator()
    rng.manual_seed(config.random_seed)

    scalars = {}
    print('Start fitting the model')
    t_last = time.perf_counter()

    for step in range(config.n_training_steps):
        model.train()
        batch = hyb_rnn_utilities.get_batch(train_dat, config.batch_size, rng)

        optimizer.zero_grad()
        loss = loss_fn(batch)
        loss.backward()
        optimizer.step()

        scalars['train_loss'] = loss.item()

        if step % 500 == 0:
            t_now     = time.perf_counter()
            elapsed   = t_now - t_last
            t_last    = t_now

            model.eval()
            with torch.no_grad():
                test_batch  = hyb_rnn_utilities.get_batch(
                    test_dat,  config.batch_size, rng)
                valid_batch = hyb_rnn_utilities.get_batch(
                    valid_dat, config.batch_size, rng)

                test_loss  = loss_fn(test_batch).item()
                valid_loss = loss_fn(valid_batch).item()
                test_acc   = accuracy_fn(test_batch)
                valid_acc  = accuracy_fn(valid_batch)

            scalars.update({
                'step':       step,
                'test_loss':  test_loss,
                'valid_loss': valid_loss,
                'test_acc':   test_acc,
                'valid_acc':  valid_acc,
                'secs_per_500_steps': round(elapsed, 2),
            })
            print(f'Step: {step}\nScalars: {scalars}')

    # ------------------------------------------------ final eval on full test set
    model.eval()
    with torch.no_grad():
        final_test_loss = loss_fn(test_dat).item()
        final_test_acc  = accuracy_fn(test_dat)
        final_valid_acc = accuracy_fn(valid_dat)

    scalars.update({
        'final_test_loss': final_test_loss,
        'final_test_acc':  final_test_acc,
        'final_valid_acc': final_valid_acc,
    })
    print(f'\n=== Final evaluation (all test blocks) ===')
    print(f'Test  accuracy : {final_test_acc * 100:.2f}%  (paper target: ~68.3% for BiRNN)')
    print(f'Valid accuracy : {final_valid_acc * 100:.2f}%')
    print(f'Test  loss     : {final_test_loss:.4f}')

    return scalars, model


def main():
    train(get_config())


if __name__ == '__main__':
    main()
