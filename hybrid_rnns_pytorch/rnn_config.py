"""Config — PyTorch port of hybrid_rnns_reward_learning/rnn_config.py.

Uses a plain dataclass instead of ml_collections.ConfigDict.
"""

from dataclasses import dataclass, field
import torch.nn.functional as F


@dataclass
class NetworkParams:
    n_actions:            int    = 4
    hidden_size:          int    = 64
    final_activation_fn:  object = field(
        default_factory=lambda: (lambda x: F.softmax(x, dim=-1))
    )
    use_rnn_cell:         bool   = False #True  # use nn.RNN instead of manual loop


@dataclass
class RLParams:
    # Flags: which parameters are fitted?
    fit_alpha:        bool = True
    fit_beta:         bool = False
    fit_bias:         bool = False
    fit_forget:       bool = False
    fit_persev_p:     bool = False
    fit_persev_t:     bool = False
    fit_init_v:       bool = True
    fit_init_h:       bool = True
    fit_init_v_state: bool = False
    fit_init_h_state: bool = False
    fit_w:            bool = False

    # Fixed values for parameters that are not fitted
    alpha:    float = 0.0   # needs to be 0 for zero value update
    beta:     float = 1.0
    bias:     float = 0.0
    forget:   float = 0.0
    persev_p: float = 0.0
    persev_t: float = 0.0

    # Combination weights (value vs. habit)
    w_v: float = 0.5
    w_h: float = 0.5

    # Recurrent feedback flags
    o:  bool = False   # feed previous output back in (RNN)
    s:  bool = False   # feed previous hidden state back in (RNN)
    vo: bool = False   # feed previous value output back in (BiRNN)
    vs: bool = False   # feed previous value hidden state back in (BiRNN)
    ho: bool = False   # feed previous habit output back in (BiRNN)
    hs: bool = False   # feed previous habit hidden state back in (BiRNN)

    zero_values: bool = False  # zero out previous action value as input

    # Allow dict-style access so rl_params['fit_alpha'] works throughout
    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class Config:
    debug: bool = True

    random_seed:  int = 1356  # gives exact paper block counts: train=3302, valid=413, test=419
    dataset_path: str = 'hybrid_rnns_pytorch/data/openSourceRawDataset.csv' #smallExampleDataset

    n_trials:   int = 150
    n_datasets: int = 3520 + 388

    model_name: str = 'birnn'   # 'rnn' | 'birnn' | 'cogmod'

    # Set by __post_init__ based on debug flag
    n_training_steps: int = field(init=False)
    batch_size:       int = field(init=False)

    learning_rate: float = 1e-4
    weight_decay:  float = 1e-5

    network_params: NetworkParams = field(default_factory=NetworkParams)
    rnn_rl_params:  RLParams      = field(default_factory=RLParams)

    def __post_init__(self):
        if self.debug:
            self.n_training_steps = 100
            self.batch_size       = 2
        else:
            self.n_training_steps = int(1e6)
            self.batch_size       = 32


def get_config() -> Config:
    return Config()


def get_rnn_config() -> Config:
    """Vanilla RNN config from the paper (Table 1).

    Confirmed by parameter count: hidden_size=64 + s=True → 4,740 params exactly.
    Without s=True the model has only 644 params and performs much worse.
    Paper also uses batch_size=64 (not 32) for the RNN.
    """
    config = Config()
    config.model_name = 'rnn'
    config.network_params.hidden_size = 64
    config.rnn_rl_params.s = True   # hidden-state feedback — REQUIRED for 4,740 params
    config.rnn_rl_params.o = False
    return config


def get_birnn_config() -> Config:
    """Winning hybRNN config — "Memory-ANN" from paper Table 1.

    Paper Table 1 (Memory-ANN): hidden_size=32, batch_size=128, n_params=2,472,
    accuracy=68.3%.  The notebook Example 2 uses hidden_size=16/batch_size=32
    as a fast demo, NOT the paper-optimal config.

    Parameter count verification (s=True, zero_values=True, hidden_size=32):
      value_rnn  Linear(34→32): 1120  value_out  Linear(32→1): 33
      habit_rnn  Linear(36→32): 1184  habit_out  Linear(32→4): 132
      fit_init_v, fit_init_h, fit_forget: 3
      Total: 2,472  ✓
    """
    config = Config()
    config.model_name = 'birnn'
    config.network_params.hidden_size = 32
    config.rnn_rl_params.w_v          = 1.0
    config.rnn_rl_params.w_h          = 1.0
    config.rnn_rl_params.fit_forget   = True
    config.rnn_rl_params.o            = False
    config.rnn_rl_params.s            = True
    config.rnn_rl_params.zero_values  = True
    config.rnn_rl_params.fit_init_v   = True
    config.rnn_rl_params.fit_init_h   = True
    return config
