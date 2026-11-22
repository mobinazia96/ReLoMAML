"""Command-line entry point for ReLoMAML training and evaluation."""

import argparse
import os
import random

import numpy as np
import torch

from train import train
from test import test


def parse_args():
    parser = argparse.ArgumentParser(description='ReLoMAML: Shared–Rotating Low-Rank MAML')

    # Experiment and dataset settings.
    parser.add_argument('--dataset', type=str, default='miniImageNet', help='Training dataset.')
    parser.add_argument('--test', action='store_true', help='Run evaluation instead of training.')
    parser.add_argument('--dataset_for_source', type=str, default='miniImageNet', help='Dataset used for training the checkpoint.')
    parser.add_argument('--dataset_for_target', type=str, default='miniImageNet', help='Dataset used for evaluation.')
    parser.add_argument('--flags', type=str, default='last', choices=['best', 'last'], help='Checkpoint to evaluate when using the built-in checkpoint selection.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU index; used only when CUDA is enabled.')
    parser.add_argument('--folder', type=str, default='./data', help='Local dataset directory.')
    parser.add_argument('--K_shots_for_support', type=int, default=1, help='Support examples per class.')
    parser.add_argument('--K_shots_for_query', type=int, default=15, help='Query examples per class.')
    parser.add_argument('--N_ways', type=int, default=5, help='Number of classes per task.')
    parser.add_argument('--train_num_update_step', type=int, default=2, help='Number of inner-loop steps in training.')
    parser.add_argument('--test_num_update_step', type=int, default=10, help='Number of inner-loop steps in validation and testing.')
    parser.add_argument('--first-order', action='store_true', help='Use first-order approximation (not used for the second-order MAML results in the paper).')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner-loop learning rate (where used by the training implementation).')
    parser.add_argument('--hidden_size', type=int, default=128, help='Channels per convolutional layer.')
    parser.add_argument('--batch_size', type=int, default=4, help='Tasks per meta-batch.')
    parser.add_argument('--iter', type=int, default=60000, help='Number of meta-training iterations.')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data-loader workers.')
    parser.add_argument('--download', action='store_true', help='Download the dataset if supported by the dataset loader.')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available.')
    parser.add_argument('--outer_lr1', type=float, default=1e-4, help='Outer-loop learning rate for MAML parameters (where used).')

    # ReLoMAML low-rank parameterization.
    parser.add_argument('--shared_rank', type=int, default=16, help='Shared rank for convolutional layers except the first.')
    parser.add_argument('--rotating_rank', type=int, default=4, help='Rotating rank for convolutional layers except the first.')
    parser.add_argument('--first_layer_shared_rank', type=int, default=10, help='Shared rank for the first convolutional layer.')
    parser.add_argument('--first_layer_rotating_rank', type=int, default=6, help='Rotating rank for the first convolutional layer.')
    parser.add_argument('--lora_alpha', type=float, default=1.0, help='Fixed LoRA scaling numerator (alpha).')
    parser.add_argument('--first_layer_lora_alpha', type=float, default=1.0, help='Fixed LoRA scaling numerator for the first layer.')
    parser.add_argument('--warmup_iterations', type=int, default=10000, help='Meta-iterations before warm-up finalization.')
    parser.add_argument('--rotation_interval', type=int, default=3500, help='Meta-iterations between subspace rotations.')
    rotation = parser.add_mutually_exclusive_group()
    rotation.add_argument('--enable_rotation', dest='enable_rotation', action='store_true', help='Enable rotating-subspace replacement (default).')
    rotation.add_argument('--disable_rotation', dest='enable_rotation', action='store_false', help='Disable rotating-subspace replacement for ablations.')
    parser.set_defaults(enable_rotation=True)
    parser.add_argument('--lora_outer_lr', type=float, default=0.005, help='Outer-loop learning rate for low-rank factors.')
    parser.add_argument('--classifier_inner_lr', type=float, default=0.001, help='Inner-loop learning rate for the classifier.')

    # Outputs and diagnostics.
    parser.add_argument('--output_root', type=str, default='./results', help='Directory for experiment logs and checkpoints.')
    parser.add_argument('--exp_name', type=str, default='rank20', help='Experiment subdirectory name.')
    parser.add_argument('--load_path', type=str, default=None, help='Checkpoint path for testing.')
    parser.add_argument('--debug_lora', action='store_true', help='Print low-rank diagnostics.')
    parser.add_argument('--debug_batches', type=int, default=2, help='Number of initial meta-iterations for diagnostics.')

    return parser.parse_args()


def main():
    args = parse_args()

    # Device selection. For reliable GPU selection, CUDA must not have been initialized yet.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.shared_rank < 0 or args.rotating_rank < 0:
        raise ValueError('Shared and rotating ranks must be non-negative.')
    if args.shared_rank + args.rotating_rank <= 0:
        raise ValueError('Total ReLoMAML rank must be positive.')
    if args.first_layer_shared_rank < 0 or args.first_layer_rotating_rank < 0:
        raise ValueError('First-layer ranks must be non-negative.')
    if args.first_layer_shared_rank + args.first_layer_rotating_rank <= 0:
        raise ValueError('Total first-layer rank must be positive.')
    if args.lora_alpha <= 0 or args.first_layer_lora_alpha <= 0:
        raise ValueError('LoRA alpha must be positive for all layers.')
    if args.warmup_iterations < 0:
        raise ValueError('Warm-up iterations must be non-negative.')
    if args.rotation_interval <= 0:
        raise ValueError('Rotation interval must be positive.')

    args.total_rank = args.shared_rank + args.rotating_rank
    args.device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    if args.test:
        test(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
