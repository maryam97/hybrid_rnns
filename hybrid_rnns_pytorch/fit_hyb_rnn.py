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

from . import hyb_rnn_utilities
from .rnn_config import get_config
from .bi_rnn import BiRNN
from .cogmod import CogMod
from .rnn import RNN


def train(config=None, compile_model=False):
    """Fit one model (cogmod / RNN / biRNN) to human bandit task behaviour."""

    if config is None:
        config = get_config()

    # ------------------------------------------------------------------ device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if device.type == 'cpu':
        # The manual per-timestep unroll (see rnn.py) does ~150 tiny ops per
        # training step. PyTorch's default intra-op thread pool adds sync
        # overhead per op that dwarfs the op cost at this size, so a single
        # thread outperforms the default multi-threaded backend here.
        torch.set_num_threads(1)

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
    if compile_model:
        # loss_fn/eval_metrics call model.unroll(...) directly, not model(...),
        # so compile that method rather than wrapping the module itself.
        model.unroll = torch.compile(model.unroll)

    # ------------------------------------------------------- load & split data
    print(f'Loading data from {config.dataset_path}')
    hum_dat = hyb_rnn_utilities.load_osf_dataframe(config.dataset_path)   # rewards 0-1
    tensors = hyb_rnn_utilities.format_data_for_model_training(
        hum_dat, n_actions=config.network_params.n_actions, split_file=config.split_path)

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

    def eval_metrics(batch_dat: torch.Tensor) -> dict:
        """NLL per block and accuracy, computed as in the paper.

        acc_paper = mean over blocks of exp(-NLL_block / 150), as in the paper's
        Methods and Supp. Table 6. acc_pooled = exp(-total NLL / valid trials),
        which is ~2 points lower and not comparable to the paper.
        """
        probs, _ = model.unroll(batch_dat[:, :, :n_actions + 1])
        nll = hyb_rnn_utilities.block_nll(probs[:, :-1], batch_dat, n_actions)
        return hyb_rnn_utilities.accuracy_metrics(nll, batch_dat, n_actions)

    # ---------------------------------------------------------------- training
    rng = torch.Generator()
    rng.manual_seed(config.random_seed)

    scalars = {}
    best_valid_nll  = float('inf')
    best_model_dict = None
    best_step       = 0

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

        if step % 500 == 0 or step == config.n_training_steps - 1:   # last step is a selection candidate too
            t_now     = time.perf_counter()
            elapsed   = t_now - t_last
            t_last    = t_now

            model.eval()
            with torch.no_grad():
                test_batch = hyb_rnn_utilities.get_batch(
                    test_dat, config.batch_size, rng)

                test_m  = eval_metrics(test_batch)   # random batch: noisy, for monitoring only
                # Full validation set for checkpoint selection (paper: best fit on validation)
                valid_m = eval_metrics(valid_dat)

            if valid_m['nll_per_block'] < best_valid_nll:
                best_valid_nll  = valid_m['nll_per_block']
                best_model_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_step       = step

            scalars.update({
                'monitor_step':                     step,
                'monitor_test_batch_nll_per_block': test_m['nll_per_block'],
                'monitor_valid_nll_per_block':      valid_m['nll_per_block'],
                'monitor_valid_acc_paper':          valid_m['acc_paper'],
                'secs_per_500_steps':         round(elapsed, 2),
            })
            print(f'Step: {step}\nScalars: {scalars}')

    # ------------------------------------------------ final eval on full sets
    # Last-step weights (what the author's notebook and upstream report), then
    # the best checkpoint on validation NLL (the paper's selection rule).
    model.eval()
    with torch.no_grad():
        last_test, last_valid = eval_metrics(test_dat), eval_metrics(valid_dat)
    if best_model_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_dict.items()})
    with torch.no_grad():
        best_test, best_valid = eval_metrics(test_dat), eval_metrics(valid_dat)

    scalars['best_step'] = best_step
    for tag, m in [('test', best_test), ('valid', best_valid),
                   ('last_step_test', last_test), ('last_step_valid', last_valid)]:
        scalars.update({f'{tag}_{k}': v for k, v in m.items()})

    print(f'\n=== Final evaluation on the full test set ({len(test_dat)} blocks) ===')
    for name, m in [(f'best checkpoint (step {best_step})', best_test), ('last step', last_test)]:
        print(f"{name:28s}: NLL/block {m['nll_per_block']:.2f} | acc {m['acc_paper'] * 100:.2f}% "
              f"(pooled {m['acc_pooled'] * 100:.2f}%)")
    hidden = config.network_params.hidden_size
    if config.model_name == 'birnn' and hidden == 32:
        print('Paper Memory-ANN: NLL/block 61.3, acc 68.3%')
    elif config.model_name == 'rnn' and hidden == 64:
        print('Paper vanilla RNN: NLL/block 61.7, acc 68.1%')

    return scalars, model


def main():
    train(get_config())


if __name__ == '__main__':
    main()
