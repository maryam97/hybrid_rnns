"""laplace_compat.py — laplace-torch compatibility hooks for hybrid_rnns_pytorch.

laplace-torch (https://github.com/aleximmer/Laplace) expects a model with:
  - forward(x) -> raw logits (pre-softmax), shape (N, n_classes)
  - the last nn.Linear in the forward graph is the "last layer" for
    subset_of_weights='last_layer' — Laplace auto-detects it via a hook

Because our models are sequence models, the wrapper:
  1. Accepts x of shape (batch, n_trials, n_actions+2) — includes valid-mask column
  2. Strips the mask column before passing to the model backbone
  3. Unrolls the backbone to collect hidden-state features for t=0..T-2
  4. Calls the last linear ONCE with a 2D tensor (batch*(T-1), hidden) so
     Laplace's feature hook captures the full set of predictions in one shot
  5. Returns raw logits (batch*(T-1), n_actions) — no softmax

BiRNN note: last-layer Laplace is applied to habit_out_linear only.  The
value stream is treated as a fixed offset (detached from the gradient graph).
This is an approximation; for full uncertainty use subset_of_weights='all'.

Usage
-----
    from hybrid_rnns_pytorch.laplace_compat import laplace_ready, make_dataloader
    from laplace import Laplace

    wrapped     = laplace_ready(trained_birnn)
    train_loader = make_dataloader(train_dat)

    la = Laplace(wrapped, likelihood='classification',
                 subset_of_weights='last_layer',
                 hessian_structure='kron')   # 'kron', not 'kfac'
    la.fit(train_loader)
    la.optimize_prior_precision(method='marglik', pred_type='glm',
                                link_approx='probit')

    # Predictive (returns class probabilities)
    probs = la(x_test, pred_type='glm', link_approx='probit')
    # Predictive samples
    samples = la.predictive_samples(x_test, pred_type='glm', n_samples=100)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn    import RNN
from .bi_rnn import BiRNN
from .cogmod import CogMod


# ---------------------------------------------------------------------------
# Sequence DataLoader helper
# ---------------------------------------------------------------------------

class SequenceDataset(torch.utils.data.Dataset):
    """Wraps a (n_blocks, n_trials, n_actions+2) tensor for use with DataLoader.

    Each item is (x, y) where:
        x : (n_trials, n_actions+2)   — full sequence including valid-mask column
        y : (n_trials-1,)             — action class indices at t+1

    Missed trials (valid == 0) are kept in y as class 0 (the clipped index).
    This is a minor approximation; they make up a small fraction of trials.
    """

    def __init__(self, tensor: torch.Tensor, n_actions: int = 4):
        self.tensor   = tensor
        self.n_actions = n_actions

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx: int):
        seq = self.tensor[idx]                          # (n_trials, n_actions+2)
        y   = seq[1:, :self.n_actions].argmax(dim=-1)  # (n_trials-1,) class indices
        return seq, y


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class SequenceModelWrapper(nn.Module):
    """Wraps RNN/BiRNN so Laplace sees a standard forward(x) -> logits API.

    forward(x) where x is (batch, n_trials, n_actions+2) returns raw logits
    of shape (batch*(n_trials-1), n_actions) — one prediction per timestep.

    The last linear layer (output_linear for RNN, habit_out_linear for BiRNN)
    is called exactly ONCE per forward pass with a 2D input tensor so that
    Laplace's feature hook captures the complete (batch*(T-1), hidden) matrix.
    """

    def __init__(self, model: RNN | BiRNN, n_actions: int, n_trials: int):
        super().__init__()

        if isinstance(model, CogMod):
            raise TypeError(
                "CogMod has no trainable linear head. "
                "Use subset_of_weights='all', or use RNN/BiRNN instead."
            )
        if not isinstance(model, (RNN, BiRNN)):
            raise TypeError(f"Unsupported model type: {type(model)}")

        self.model     = model
        self.n_actions = n_actions
        self.n_trials  = n_trials

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_trials, n_actions+2)  — includes valid-mask column

        Returns:
            logits: (batch*(n_trials-1), n_actions)  — raw pre-softmax logits
        """
        # Strip the valid-mask column; models expect (n_actions+1) features
        x_model = x[:, :, :self.n_actions + 1]   # (batch, n_trials, n_actions+1)

        if isinstance(self.model, RNN):
            return self._forward_rnn(x_model)
        else:
            return self._forward_birnn(x_model)

    # ------------------------------------------------------------------

    def _forward_rnn(self, x: torch.Tensor) -> torch.Tensor:
        """RNN: collect hidden states for t=0..T-2, then call output_linear once."""
        batch, time, _ = x.shape
        model = self.model

        if hasattr(model, 'rnn_cell'):
            # Fast path: single nn.RNN call gives all hidden states at once.
            h0 = torch.zeros(1, batch, model._hidden_size, device=x.device)
            hidden_seq, _ = model.rnn_cell(x, h0)          # (batch, time, hidden)
            # Hidden at t predicts action at t+1, so take t=0..T-2
            features = hidden_seq[:, :-1].contiguous().view(-1, model._hidden_size)
        else:
            # Slow path: manual loop to collect hidden states.
            state   = model.initial_state(batch, x.device)
            hiddens = []
            for t in range(time - 1):
                _, state = model.forward(x[:, t], state)
                hiddens.append(state[1])               # state = (gist, hidden)
            features = torch.stack(hiddens, dim=1).contiguous().view(-1, model._hidden_size)

        # output_linear called ONCE with 2D input — Laplace hooks here
        logits = model.output_linear(features)         # (batch*(time-1), n_actions)
        return logits

    # ------------------------------------------------------------------

    def _forward_birnn(self, x: torch.Tensor) -> torch.Tensor:
        """BiRNN: collect habit hidden states + value offsets, call habit_out_linear once."""
        batch, time, _ = x.shape
        model = self.model

        if hasattr(model, 'habit_rnn_cell') and not model._ho and not model._hs:
            # Fast path: habit stream uses nn.RNN.
            h_state, v_state, habit, value = model.initial_state(batch, x.device)
            actions_seq = x[:, :, :model._n_actions]
            h0          = h_state.unsqueeze(0)
            habit_hiddens, _ = model.habit_rnn_cell(actions_seq, h0)  # (batch, time, hidden)

            # Value stream still needs a step-by-step loop (custom Q-update).
            v_outputs = []
            for t in range(time - 1):
                action = x[:, t, :model._n_actions]
                reward = x[:, t, -1]
                next_value, v_state = model._value_rnn(v_state, value, action, reward)
                v_outputs.append(next_value)
                value = next_value

            features = habit_hiddens[:, :-1].contiguous().view(-1, model._hidden_size)

        else:
            # Slow path: manual loop for both streams.
            # habit_out_linear is NOT called inside the loop so the hook fires
            # only once (the bulk call below).
            state = model.initial_state(batch, x.device)
            h_hiddens = []
            v_outputs = []

            for t in range(time - 1):
                h_state, v_state, habit, value = state
                action = x[:, t, :model._n_actions]
                reward = x[:, t, -1]

                next_value, next_v_state = model._value_rnn(v_state, value, action, reward)

                # Habit hidden state — replicate _habit_rnn but skip the output linear.
                h_in = action
                if model._hs: h_in = torch.cat([h_in, h_state], dim=-1)
                if model._ho: h_in = torch.cat([h_in, habit],   dim=-1)
                next_h_state = torch.tanh(model.habit_rnn_linear(h_in))

                h_hiddens.append(next_h_state)
                v_outputs.append(next_value)

                # habit placeholder for _ho feedback; habit_out_linear NOT called here
                if model._ho:
                    # Must compute habit for the next step's feedback input
                    next_habit = model.habit_out_linear(next_h_state)
                else:
                    next_habit = habit  # not used as input — safe placeholder

                state = (next_h_state, next_v_state, next_habit, next_value)

            features = torch.stack(h_hiddens, dim=1).contiguous().view(-1, model._hidden_size)

        # habit_out_linear called ONCE with 2D input — Laplace hooks here last
        habit_logits  = model.habit_out_linear(features)                        # (N, n_actions)
        value_offsets = torch.stack(v_outputs, dim=1).contiguous().view(-1, model._n_actions)

        # Pre-softmax combination; detach value so Laplace gradient only covers habit head
        logits = model.beta * (model._w_v * value_offsets.detach() + model._w_h * habit_logits)
        return logits

    # ------------------------------------------------------------------

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: softmax probabilities for evaluation (not for Laplace fit)."""
        return F.softmax(self.forward(x), dim=-1)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def laplace_ready(
    model: RNN | BiRNN,
    n_actions: int = 4,
    n_trials:  int = 150,
) -> SequenceModelWrapper:
    """Return a laplace-torch-compatible wrapper around a trained model.

    Laplace auto-detects the last nn.Linear by walking the module graph.
    For RNN that will be output_linear; for BiRNN it will be habit_out_linear
    (the last linear called in the forward pass).  You can override detection
    by passing last_layer_name= to the Laplace constructor:

        la = Laplace(wrapped, 'classification',
                     subset_of_weights='last_layer',
                     hessian_structure='kron',
                     last_layer_name='model.output_linear')   # RNN example
    """
    model.eval()
    return SequenceModelWrapper(model, n_actions=n_actions, n_trials=n_trials)


def make_dataloader(
    tensor: torch.Tensor,
    n_actions:  int  = 4,
    batch_size: int  = 32,
    shuffle:    bool = True,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader compatible with laplace-torch from a block tensor.

    Args:
        tensor:     (n_blocks, n_trials, n_actions+2) float tensor
                    — the same tensors produced by format_data_for_model_training
        n_actions:  number of actions (first n_actions columns are one-hot)
        batch_size: mini-batch size
        shuffle:    whether to shuffle blocks each epoch

    Returns:
        DataLoader yielding (x, y) where:
            x : (batch, n_trials, n_actions+2)
            y : (batch*(n_trials-1),)  — flattened target class indices
    """
    dataset = SequenceDataset(tensor, n_actions=n_actions)

    def collate_fn(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs)              # (batch, n_trials, n_actions+2)
        y = torch.stack(ys).reshape(-1)  # (batch*(n_trials-1),)
        return x, y

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
