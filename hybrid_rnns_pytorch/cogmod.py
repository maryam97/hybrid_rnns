"""CogMod — PyTorch port of hybrid_rnns_reward_learning/cogmod.py.

Classic alpha-beta RL model formulated as an RNN.
"""

from typing import Optional
import torch
import torch.nn as nn


class CogMod(nn.Module):
    """Classic cognitive model (Rescorla-Wagner) formulated as an RNN.

    Learnable scalar parameters (alpha, beta, forget, persev_p, persev_t,
    init_value) are registered as nn.Parameter and stored in unconstrained
    ('raw') form, then passed through the same activation functions used in
    the original Haiku code to enforce their ranges.

    Parameter ranges
    ----------------
    alpha    : sigmoid  -> (0, 1)
    beta     : relu     -> (0, ∞)
    forget   : sigmoid  -> (0, 1)
    persev_p : tanh     -> (-1, 1)
    persev_t : tanh     -> (-1, 1)
    init_value: unconstrained scalar (fitted directly)
    """

    def __init__(self, rl_params, network_params, init_value: float = 0.5):
        super().__init__()

        self._n_actions = network_params.n_actions
        self._final_activation_fn = network_params.final_activation_fn

        # --- init_value ------------------------------------------------
        if rl_params['fit_init_v']:
            self._raw_init_value = nn.Parameter(torch.empty(1).normal_(1, 1))
            self._fit_init_v = True
        else:
            self.register_buffer('_fixed_init_value',
                                 torch.tensor([init_value]))
            self._fit_init_v = False

        # --- alpha (learning rate) -------------------------------------
        if rl_params['fit_alpha']:
            self._raw_alpha = nn.Parameter(torch.ones(1))   # sigmoid(1) ≈ 0.73
            self._fit_alpha = True
        else:
            self.register_buffer('_fixed_alpha',
                                 torch.tensor([float(rl_params['alpha'])]))
            self._fit_alpha = False

        # --- beta (inverse temperature) --------------------------------
        if rl_params['fit_beta']:
            self._raw_beta = nn.Parameter(torch.ones(1))    # relu(1) = 1
            self._fit_beta = True
        else:
            self.register_buffer('_fixed_beta',
                                 torch.tensor([float(rl_params['beta'])]))
            self._fit_beta = False

        # --- forget ----------------------------------------------------
        if rl_params['fit_forget']:
            self._raw_forget = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5
            self._fit_forget = True
        else:
            self.register_buffer('_fixed_forget',
                                 torch.tensor([float(rl_params['forget'])]))
            self._fit_forget = False

        # --- persev_p (perseverance on previous action) ----------------
        if rl_params['fit_persev_p']:
            self._raw_persev_p = nn.Parameter(torch.zeros(1))
            self._fit_persev_p = True
        else:
            self.register_buffer('_fixed_persev_p',
                                 torch.tensor([float(rl_params['persev_p'])]))
            self._fit_persev_p = False

        # --- persev_t (perseverance on current action) -----------------
        if rl_params['fit_persev_t']:
            self._raw_persev_t = nn.Parameter(torch.zeros(1))
            self._fit_persev_t = True
        else:
            self.register_buffer('_fixed_persev_t',
                                 torch.tensor([float(rl_params['persev_t'])]))
            self._fit_persev_t = False

    # ------------------------------------------------------------------
    # Properties apply the activation functions to the raw parameters.
    # ------------------------------------------------------------------

    @property
    def init_value(self) -> torch.Tensor:
        return self._raw_init_value if self._fit_init_v else self._fixed_init_value

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._raw_alpha) if self._fit_alpha else self._fixed_alpha

    @property
    def beta(self) -> torch.Tensor:
        return torch.relu(self._raw_beta) if self._fit_beta else self._fixed_beta

    @property
    def forget(self) -> torch.Tensor:
        return torch.sigmoid(self._raw_forget) if self._fit_forget else self._fixed_forget

    @property
    def persev_p(self) -> torch.Tensor:
        return torch.tanh(self._raw_persev_p) if self._fit_persev_p else self._fixed_persev_p

    @property
    def persev_t(self) -> torch.Tensor:
        return torch.tanh(self._raw_persev_t) if self._fit_persev_t else self._fixed_persev_t

    # ------------------------------------------------------------------

    def _rl_value_fn(
        self,
        prev_value: torch.Tensor,   # (batch, n_actions)
        action: torch.Tensor,       # (batch, n_actions)  one-hot
        reward: torch.Tensor,       # (batch,)
    ) -> torch.Tensor:
        rpe = reward - (action * prev_value).sum(dim=-1)  # (batch,)
        new_value = prev_value + action * self.alpha * rpe.unsqueeze(-1)
        return new_value

    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: torch.Tensor,        # (batch, n_actions + 1)
        prev_state: torch.Tensor,    # (batch, n_actions)  = prev Q-values
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev_value = prev_state
        action = inputs[:, :self._n_actions]
        reward = inputs[:, -1]

        # Value update (Rescorla-Wagner)
        new_value = self._rl_value_fn(prev_value, action, reward)

        # Forgetting: decay toward init_value
        new_value = (1 - self.forget) * new_value + self.forget * self.init_value

        # Perseverance
        new_value  = new_value + self.persev_p * action
        pers_value = new_value + self.persev_t * action

        action_probs = self._final_activation_fn(self.beta * pers_value)

        return action_probs, new_value

    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        return self.init_value * torch.ones(
            batch_size, self._n_actions, device=device
        )

    def unroll(
        self,
        input_seq: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unroll over (batch, time, features); return (outputs, final_state)."""
        batch_size = input_seq.size(0)
        device     = input_seq.device
        state = initial_state if initial_state is not None \
            else self.initial_state(batch_size, device)

        outputs = []
        for t in range(input_seq.size(1)):
            out, state = self.forward(input_seq[:, t], state)
            outputs.append(out)

        return torch.stack(outputs, dim=1), state
