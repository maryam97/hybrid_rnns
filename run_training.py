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
import json
import os
import time
import torch
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
                        help='Learning rate (default: 1e-3, paper-optimal).')
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='Hidden units per RNN layer (default: 64, paper-optimal).')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='AdamW weight decay (default: 1e-4, paper-optimal).')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size (default: 32).')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of training steps (default: 1M).')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save the trained model weights.')
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
    if args.hidden_size is not None:
        config.network_params.hidden_size = args.hidden_size
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.steps is not None:
        config.n_training_steps = args.steps

    print(f'Device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print(f'Model       : {config.model_name}')
    print(f'Hidden size : {config.network_params.hidden_size}')
    print(f'LR          : {config.learning_rate}  weight_decay: {config.weight_decay}')
    print(f'Batch size  : {config.batch_size}  steps: {config.n_training_steps}')

    t0 = time.time()
    scalars, model = train(config)
    elapsed = time.time() - t0
    print(f'\nTraining complete. Final scalars: {scalars}')
    print(f'Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)')

    results = {
        'model':           config.model_name,
        'hidden_size':     config.network_params.hidden_size,
        'steps':           config.n_training_steps,
        'lr':              config.learning_rate,
        'weight_decay':    config.weight_decay,
        'batch_size':      config.batch_size,
        'debug':           config.debug,
        'training_time_s': round(elapsed, 1),
        **scalars,
    }
    os.makedirs('results', exist_ok=True)
    results_path = f'results/{config.model_name}_hs={config.network_params.hidden_size}_s={scalars["step"]}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {results_path}')

    if args.save:
        os.makedirs('trained_models', exist_ok=True)
        save_path = f'trained_models/{args.model}_e={scalars["step"]}_pred.pt'
        torch.save(model.state_dict(), save_path)
        print(f'Weights saved to {save_path}')


if __name__ == '__main__':
    main()
