"""RNN — PyTorch port of hybrid_rnns_reward_learning/rnn.py."""

import math
from typing import Optional
import torch
import torch.nn as nn

# State is a tuple of (gist, hidden_state), both (batch, dim)
RNNState = tuple[torch.Tensor, torch.Tensor]


class RNN(nn.Module):
    """RNN that predicts action logits based on all inputs (action, reward).

    Mirrors the Haiku RNN. Accepts options `s` (feed previous hidden state
    back as input) and `o` (feed previous output / 'gist' back as input).

    When network_params.use_rnn_cell is True and neither s nor o is set, the
    model has no recurrence, so unroll() applies the hidden layer to all time
    steps at once instead of looping (same result).
    """

    def __init__(self, rl_params, network_params):
        super().__init__()

        self._s = rl_params.s
        self._o = rl_params.o
        self._use_rnn_cell = network_params.use_rnn_cell

        self._n_actions   = network_params.n_actions
        self._hidden_size = network_params.hidden_size
        self._final_activation_fn = network_params.final_activation_fn

        in_dim = self._n_actions + 1                 # action one-hot + reward
        if self._o:
            in_dim += self._n_actions
        if self._s:
            in_dim += self._hidden_size
        self.rnn_linear = nn.Linear(in_dim, self._hidden_size)
        self.output_linear = nn.Linear(self._hidden_size, self._n_actions)

        # hk.Linear default init: truncated normal (+-2 std), std = 1/sqrt(fan_in); zero bias
        for lin in (self.rnn_linear, self.output_linear):
            std = 1.0 / math.sqrt(lin.in_features)
            nn.init.trunc_normal_(lin.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(lin.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        inputs: torch.Tensor,       # (batch, input_dim)
        prev_state: RNNState,
    ) -> tuple[torch.Tensor, RNNState]:
        """Single-step forward pass (used by the manual loop path)."""
        gist, state = prev_state

        if self._o:
            inputs = torch.cat([inputs, gist], dim=-1)
        if self._s:
            inputs = torch.cat([inputs, state], dim=-1)

        next_state   = torch.tanh(self.rnn_linear(inputs))
        next_gist    = self.output_linear(next_state)
        action_probs = self._final_activation_fn(next_gist)

        return action_probs, (next_gist, next_state)

    # ------------------------------------------------------------------
    def initial_state(self, batch_size: int, device=None) -> RNNState:
        return (
            torch.zeros(batch_size, self._n_actions,   device=device),
            torch.zeros(batch_size, self._hidden_size, device=device),
        )

    # ------------------------------------------------------------------
    def unroll(
        self,
        input_seq: torch.Tensor,
        initial_state: Optional[RNNState] = None,
    ) -> tuple[torch.Tensor, RNNState]:
        """Unroll over (batch, time, input_dim); return (outputs, final_state)."""
        batch_size = input_seq.size(0)
        device     = input_seq.device

        if self._use_rnn_cell and not self._o and not self._s:
            # ---- fast path (s=o=False): no recurrence in the original model,
            # so the hidden layer is applied to every time step at once.
            hidden_seq   = torch.tanh(self.rnn_linear(input_seq))   # (batch, time, hidden)
            gist_seq     = self.output_linear(hidden_seq)           # (batch, time, n_actions)
            action_probs = self._final_activation_fn(gist_seq)
            return action_probs, (gist_seq[:, -1], hidden_seq[:, -1])

        else:
            # ---- original path: manual loop ----
            state = initial_state or self.initial_state(batch_size, device)
            outputs = []
            for t in range(input_seq.size(1)):
                out, state = self.forward(input_seq[:, t], state)
                outputs.append(out)
            return torch.stack(outputs, dim=1), state
