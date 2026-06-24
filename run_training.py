"""run_training.py — top-level entry point.

Place this file next to (not inside) the hybrid_rnns_pytorch/ package folder:

    project/
    ├── run_training.py          ← this file
    └── hybrid_rnns_pytorch/
        ├── __init__.py
        ├── rnn_config.py
        ├── cogmod.py
        ├── rnn.py
        ├── bi_rnn.py
        ├── hyb_rnn_utilities.py
        ├── fit_hyb_rnn.py
        └── laplace_compat.py

Usage
-----
    python run_training.py                         # defaults from rnn_config.py
    python run_training.py --model birnn           # choose model
    python run_training.py --model cogmod --no-debug
"""

import argparse
from hybrid_rnns_pytorch.fit_hyb_rnn import train
from hybrid_rnns_pytorch.rnn_config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a hybrid RNN reward-learning model.')
    parser.add_argument('--model', choices=['cogmod', 'rnn', 'birnn'], default=None,
                        help='Model to train (overrides config default).')
    parser.add_argument('--no-debug', action='store_true',
                        help='Run full training (1M steps, batch=32) instead of debug mode.')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset CSV (overrides config default).')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config default).')
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()

    if args.model is not None:
        config.model_name = args.model
    if args.no_debug:
        config.debug = False
        config.n_training_steps = int(1e6)
        config.batch_size = 32
    if args.dataset is not None:
        config.dataset_path = args.dataset
    if args.lr is not None:
        config.learning_rate = args.lr

    print(f'Config: model={config.model_name}, debug={config.debug}, '
          f'steps={config.n_training_steps}, lr={config.learning_rate}')

    scalars, model = train(config)
    print(f'\nTraining complete. Final scalars: {scalars}')


if __name__ == '__main__':
    main()
