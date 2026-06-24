"""BiRNN — PyTorch port of hybrid_rnns_reward_learning/bi_rnn.py.

A bifurcating RNN with separate 'habit' and 'value' modules.
"""

from typing import Optional
import torch
import torch.nn as nn

# State = (h_state, v_state, habit, value)  each (batch, dim)
BiRNNState = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class BiRNN(nn.Module):
    """Bifurcating RNN: 'habit' processes action sequences; 'value' processes rewards.

    The two modules are combined via a weighted sum:
        hv_combo = w_v * value + w_h * habit
        action_probs = final_activation_fn(beta * hv_combo)

    Learnable scalar parameters (init_value_v/h, init_state_v/h, forget)
    are stored in raw (unconstrained) form and passed through activations
    matching the original Haiku code.
    """

    def __init__(self, rl_params, network_params, init_value: float = 0.5):
        super().__init__()

        self._hs = rl_params.s   # feed hidden state back into habit RNN
        self._vs = rl_params.s   # feed hidden state back into value RNN
        self._ho = rl_params.o   # feed previous habit output back in
        self._vo = rl_params.o   # feed previous value output back in

        self._zero_values     = rl_params.zero_values
        self._n_actions       = network_params.n_actions
        self._hidden_size     = network_params.hidden_size
        self._final_activation_fn = network_params.final_activation_fn
        self._use_rnn_cell    = network_params.use_rnn_cell

        self._w_v = rl_params.w_v
        self._w_h = rl_params.w_h
        self.beta = rl_params.beta

        # --- Learnable init values / states ----------------------------

        if rl_params['fit_init_v']:
            self._raw_init_value_v = nn.Parameter(torch.empty(1).normal_(1, 1))
            self._fit_init_v = True
        else:
            self.register_buffer('_fixed_init_value_v',
                                 torch.tensor([init_value]))
            self._fit_init_v = False

        if rl_params['fit_init_h']:
            self._raw_init_value_h = nn.Parameter(torch.empty(1).normal_(1, 1))
            self._fit_init_h = True
        else:
            self.register_buffer('_fixed_init_value_h',
                                 torch.tensor([init_value]))
            self._fit_init_h = False

        if rl_params['fit_init_v_state']:
            self._raw_init_v_state = nn.Parameter(
                torch.empty(1, self._hidden_size).normal_(1, 1))
            self._fit_init_v_state = True
        else:
            self._fit_init_v_state = False

        if rl_params['fit_init_h_state']:
            self._raw_init_h_state = nn.Parameter(
                torch.empty(1, self._hidden_size).normal_(1, 1))
            self._fit_init_h_state = True
        else:
            self._fit_init_h_state = False

        if rl_params['fit_forget']:
            self._raw_forget = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5
            self._fit_forget = True
        else:
            self.register_buffer('_fixed_forget',
                                 torch.tensor([float(rl_params['forget'])]))
            self._fit_forget = False

        # --- Value RNN linear layers -----------------------------------
        # Input: [pre_act_val (1), reward (1)]
        #   + optional value output  (+n_actions if _vo)
        #   + optional hidden state  (+hidden_size if _vs)
        v_in = 2
        if self._vo:
            v_in += self._n_actions
        if self._vs:
            v_in += self._hidden_size
        self.value_rnn_linear = nn.Linear(v_in, self._hidden_size)
        self.value_out_linear = nn.Linear(self._hidden_size, 1)

        # --- Habit RNN linear layers -----------------------------------
        # Input: action (n_actions)
        #   + optional habit output  (+n_actions if _ho)
        #   + optional hidden state  (+hidden_size if _hs)
        h_in = self._n_actions
        if self._ho:
            h_in += self._n_actions
        if self._hs:
            h_in += self._hidden_size

        if self._use_rnn_cell and not self._ho and not self._hs:
            # Fast path: nn.RNN processes the full action sequence in one call.
            self.habit_rnn_cell = nn.RNN(
                self._n_actions, self._hidden_size,
                batch_first=True, nonlinearity='tanh',
            )
        else:
            self.habit_rnn_linear = nn.Linear(h_in, self._hidden_size)
        self.habit_out_linear = nn.Linear(self._hidden_size, self._n_actions)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def init_value_v(self) -> torch.Tensor:
        return self._raw_init_value_v if self._fit_init_v else self._fixed_init_value_v

    @property
    def init_value_h(self) -> torch.Tensor:
        return self._raw_init_value_h if self._fit_init_h else self._fixed_init_value_h

    @property
    def forget(self) -> torch.Tensor:
        return torch.sigmoid(self._raw_forget) if self._fit_forget else self._fixed_forget

    # ------------------------------------------------------------------
    # Sub-modules
    # ------------------------------------------------------------------

    def _value_rnn(
        self,
        state: torch.Tensor,   # (batch, hidden_size)
        value: torch.Tensor,   # (batch, n_actions)
        action: torch.Tensor,  # (batch, n_actions)  one-hot
        reward: torch.Tensor,  # (batch,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre_act_val = (value * action).sum(dim=-1, keepdim=True)  # (batch, 1)
        if self._zero_values:
            pre_act_val = torch.zeros_like(pre_act_val)

        inputs = torch.cat([pre_act_val, reward.unsqueeze(-1)], dim=-1)
        if self._vo:
            inputs = torch.cat([inputs, value], dim=-1)
        if self._vs:
            inputs = torch.cat([inputs, state], dim=-1)

        next_state = torch.tanh(self.value_rnn_linear(inputs))  # (batch, hidden)
        update     = self.value_out_linear(next_state)           # (batch, 1)

        # Forget non-chosen values: decay toward init_value_v
        value = (1 - self.forget) * value + self.forget * self.init_value_v
        if self._zero_values:
            next_value = (1 - action) * value + action * update
        else:
            next_value = value + action * update

        return next_value, next_state

    def _habit_rnn(
        self,
        state: torch.Tensor,   # (batch, hidden_size)
        habit: torch.Tensor,   # (batch, n_actions)
        action: torch.Tensor,  # (batch, n_actions)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = action
        if self._ho:
            inputs = torch.cat([inputs, habit], dim=-1)
        if self._hs:
            inputs = torch.cat([inputs, state], dim=-1)

        next_state = torch.tanh(self.habit_rnn_linear(inputs))   # (batch, hidden)
        next_habit = self.habit_out_linear(next_state)            # (batch, n_actions)

        return next_habit, next_state

    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: torch.Tensor,      # (batch, n_actions + 1)
        prev_state: BiRNNState,
    ) -> tuple[torch.Tensor, BiRNNState]:
        h_state, v_state, habit, value = prev_state
        action = inputs[:, :self._n_actions]
        reward = inputs[:, -1]

        next_value, next_v_state = self._value_rnn(v_state, value, action, reward)
        next_habit, next_h_state = self._habit_rnn(h_state, habit, action)

        hv_combo     = self._w_v * next_value + self._w_h * next_habit
        action_probs = self._final_activation_fn(self.beta * hv_combo)

        return action_probs, (next_h_state, next_v_state, next_habit, next_value)

    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int, device=None) -> BiRNNState:
        zeros_h = torch.zeros(batch_size, self._hidden_size, device=device)
        zeros_v = torch.zeros(batch_size, self._hidden_size, device=device)

        h_state = self._raw_init_h_state.expand(batch_size, -1).clone() \
            if self._fit_init_h_state else zeros_h
        v_state = self._raw_init_v_state.expand(batch_size, -1).clone() \
            if self._fit_init_v_state else zeros_v

        habit = self.init_value_h * torch.ones(
            batch_size, self._n_actions, device=device)
        value = self.init_value_v * torch.ones(
            batch_size, self._n_actions, device=device)

        return (h_state, v_state, habit, value)

    def unroll(
        self,
        input_seq: torch.Tensor,
        initial_state: Optional[BiRNNState] = None,
    ) -> tuple[torch.Tensor, BiRNNState]:
        """Unroll over (batch, time, features); return (outputs, final_state)."""
        batch_size = input_seq.size(0)
        device     = input_seq.device
        state = initial_state if initial_state is not None \
            else self.initial_state(batch_size, device)

        if self._use_rnn_cell and not self._ho and not self._hs:
            # ---- fast path: habit runs as a single nn.RNN call ----
            # Value still needs a loop (custom forgetting + Q-update each step).
            h_state, v_state, habit, value = state

            actions = input_seq[:, :, :self._n_actions]   # (batch, time, n_actions)
            h0 = h_state.unsqueeze(0)                      # (1, batch, hidden)
            habit_hiddens, final_h = self.habit_rnn_cell(actions, h0)
            # habit_hiddens: (batch, time, hidden)
            all_habits = self.habit_out_linear(habit_hiddens)  # (batch, time, n_actions)

            outputs = []
            for t in range(input_seq.size(1)):
                action = input_seq[:, t, :self._n_actions]
                reward = input_seq[:, t, -1]

                next_value, v_state = self._value_rnn(v_state, value, action, reward)
                next_habit = all_habits[:, t]

                hv_combo     = self._w_v * next_value + self._w_h * next_habit
                action_probs = self._final_activation_fn(self.beta * hv_combo)
                outputs.append(action_probs)
                value = next_value
                habit = next_habit

            final_state = (final_h.squeeze(0), v_state, habit, value)
            return torch.stack(outputs, dim=1), final_state

        else:
            # ---- original path: manual loop ----
            outputs = []
            for t in range(input_seq.size(1)):
                out, state = self.forward(input_seq[:, t], state)
                outputs.append(out)
            return torch.stack(outputs, dim=1), state
