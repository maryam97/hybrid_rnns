"""Config — PyTorch port of hybrid_rnns_reward_learning/rnn_config.py.

Uses a plain dataclass instead of ml_collections.ConfigDict.
"""

from dataclasses import dataclass, field
import torch.nn.functional as F


@dataclass
class NetworkParams:
    n_actions:            int    = 4
    hidden_size:          int    = 16
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

    random_seed:  int = 42
    dataset_path: str = 'hybrid_rnns_pytorch/data/openSourceRawDataset_v2.csv' #smallExampleDataset

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
