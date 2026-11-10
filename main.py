import os
import argparse
import random
import numpy as np
import torch

from train import train
from test import test


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

    # Argument for train
    parser.add_argument(
        '--dataset',
        type=str,
        default='miniImageNet',
        help='The dataset to train the model.'
    )

    # Argument for test
    parser.add_argument(
        '--test',
        action='store_true'
    )

    parser.add_argument(
        '--dataset_for_source',
        type=str,
        default='miniImageNet',
        help='The dataset used to train the model.'
    )

    parser.add_argument(
        '--dataset_for_target',
        type=str,
        default='miniImageNet',
        help='The dataset to test the trained model.'
    )

    parser.add_argument(
        '--flags',
        type=str,
        default='last',
        help='best or last'
    )

    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed.'
    )

    # Argument for both train and test
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU ID'
    )

    parser.add_argument(
        '--folder',
        type=str,
        default='/home/math/zi/dr_rezghi_5090/',
        help='Path to the folder the data is downloaded to.'
    )

    parser.add_argument(
        '--K_shots_for_support',
        type=int,
        default=1,
        help='The number of support set examples per class (i.e., K in "K-shot").'
    )

    parser.add_argument(
        '--K_shots_for_query',
        type=int,
        default=15,
        help='The number of query set examples per class.'
    )

    parser.add_argument(
        '--N_ways',
        type=int,
        default=5,
        help='The number of classes per task (i.e., N in "N-way").'
    )

    parser.add_argument(
        '--train_num_update_step',
        type=int,
        default=2,
        help='The number of update-step in the inner loop process during training.'
    )

    parser.add_argument(
        '--test_num_update_step',
        type=int,
        default=10,
        help='The number of update-step in the inner loop process during validating/testing.'
    )

    parser.add_argument(
        '--first-order',
        action='store_true',
        help='Use the first-order approximation of MAML.'
    )

    parser.add_argument(
        '--inner_lr',
        type=float,
        default=0.01,
        help='The step-size for the gradient step for adaptation.'
    )

    parser.add_argument(
        '--hidden_size',
        type=int,
        default=128,
        help='The number of channels for each convolutional layer.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='The number of tasks in a mini-batch of tasks.'
    )

    parser.add_argument(
        '--iter',
        type=int,
        default=60000,
        help='The number of iteration for training'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='The number of workers for data loading (default: 1).'
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Download the dataset in the data folder.'
    )

    parser.add_argument(
        '--use-cuda',
        action='store_true',
        help='Use CUDA if available.'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='The number of models used in the ensemble (if you want).'
    )

    parser.add_argument(
        '--outer_lr1',
        type=float,
        default=1e-4,
        help='The learning rate for initialization in MAML.'
    )

    parser.add_argument(
        '--outer_lr2',
        type=float,
        default=1e-4,
        help='The learning rate for meta-parameters (i.e., for preconditioner).'
    )

    parser.add_argument(
        '--GAP',
        action='store_true',
        help='Use the GAP in MAML.'
    )

    parser.add_argument(
        '--approx',
        action='store_true',
        help='Use the approximation of GAP'
    )

    # SR-LoRA arguments
    parser.add_argument(
        '--shared_rank',
        type=int,
        default=10,
        help='Rank of the persistent shared SRLoRA branch.'
    )

    parser.add_argument(
        '--rotating_rank',
        type=int,
        default=6,
        help='Rank of the rotating SRLoRA branch.'
    )

    parser.add_argument(
        '--first_layer_shared_rank',
        type=int,
        default=None
    )

    parser.add_argument(
        '--first_layer_rotating_rank',
        type=int,
        default=None
    )

    parser.add_argument(
        '--first_layer_lora_alpha',
        type=float,
        default=None
    )

    parser.add_argument(
        '--lora_alpha',
        type=float,
        default=16.0,
        help='SRLoRA scaling parameter.'
    )

    parser.add_argument(
        '--warmup_iterations',
        type=int,
        default=10000,
        help='Number of meta-iterations before SRLoRA rotation begins.'
    )

    parser.add_argument(
        '--rotation_interval',
        type=int,
        default=3500,
        help='Number of meta-iterations between SRLoRA rotations.'
    )

    parser.add_argument(
        '--enable_rotation',
        action='store_true',
        help='Enable SRLoRA rotating-subspace replacement.'
    )

    parser.add_argument(
        '--output_root',
        type=str,
        default='./results',
        help='Root directory for logs and checkpoints.'
    )

    parser.add_argument(
        '--exp_name',
        type=str,
        default='default',
        help='Name of the experiment subdirectory.'
    )

    parser.add_argument(
        '--classifier_inner_lr',
        type=float,
        default=0.001,
        help='Inner-loop learning rate for the classifier.'
    )

    parser.add_argument(
        '--debug_lora',
        action='store_true',
        help='Print LoRA diagnostics during the first meta-iterations.'
    )

    parser.add_argument(
        '--debug_batches',
        type=int,
        default=2,
        help='Number of initial meta-iterations to debug.'
    )

    parser.add_argument(
        '--load_path',
        type=str,
        default=None,
        help='Path to the trained checkpoint for testing.'
    )

    parser.add_argument(
        '--lora_outer_lr',
        type=float,
        default=0.005
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # GPU selection
    # ---------------------------------------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # ---------------------------------------------------------
    # Set random seed
    # ---------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------------
    # Check SR-LoRA arguments
    # ---------------------------------------------------------
    if args.shared_rank < 0:
        raise ValueError('shared_rank must be non-negative.')

    if args.rotating_rank < 0:
        raise ValueError('rotating_rank must be non-negative.')

    if args.shared_rank + args.rotating_rank <= 0:
        raise ValueError('Total SRLoRA rank must be positive.')

    if args.lora_alpha <= 0:
        raise ValueError('lora_alpha must be positive.')

    if args.warmup_iterations < 0:
        raise ValueError('warmup_iterations must be non-negative.')

    if args.rotation_interval <= 0:
        raise ValueError('rotation_interval must be positive.')

    args.total_rank = args.shared_rank + args.rotating_rank

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    args.device = torch.device(
        'cuda'
        if args.use_cuda and torch.cuda.is_available()
        else 'cpu'
    )

    # ---------------------------------------------------------
    # Train / Test
    # ---------------------------------------------------------
    if args.test:
        test(args)
    else:
        train(args)