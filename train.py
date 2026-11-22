"""Meta-training and evaluation checkpoints for ReLoMAML.

The inner-loop adaptation is delegated to gradient_update_parameters().
The outer loop updates the shared/rotating low-rank factors, BatchNorm affine
parameters, and classifier; full-rank convolutional base weights stay frozen.
"""

import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchmeta.utils.data import BatchMetaDataLoader

from backbone import ConvolutionalNeuralNetwork, print_initial_spectral_energy
from dataset import get_dataset
from inner_loop_optimizer import gradient_update_parameters
from transform import get_transform
from utils import evaluate, get_accuracy, get_log_path, get_logger, get_save_path

@torch.no_grad()
def print_srlora_state(
    params,
    lora_alpha,
    first_layer_lora_alpha,
    tag,
    print_values=True
    ):

    print("\n========== SRLORA STATE: {} ==========".format(tag), flush=True)

    for name, shared_A in params.items():
        if not name.endswith("lora_shared_A"):
            continue

        prefix = name[:-len("lora_shared_A")]
        shared_B_name = prefix + "lora_shared_B"
        rotating_A_name = prefix + "lora_rotating_A"
        rotating_B_name = prefix + "lora_rotating_B"

        required_names = [shared_B_name, rotating_A_name, rotating_B_name]
        missing_names = [parameter_name for parameter_name in required_names if parameter_name not in params]

        if len(missing_names) > 0:
            print("Missing SRLoRA parameters for {}: {}".format(prefix, missing_names), flush=True)
            continue

        shared_B = params[shared_B_name]
        rotating_A = params[rotating_A_name]
        rotating_B = params[rotating_B_name]

        shared_rank = int(shared_A.shape[0])
        rotating_rank = int(rotating_A.shape[0])
        total_rank = shared_rank + rotating_rank


        current_lora_alpha = first_layer_lora_alpha if prefix.startswith("features.block1.") else lora_alpha
        scaling = current_lora_alpha / float(total_rank)

        shared_delta = scaling * (shared_B @ shared_A)
        rotating_delta = scaling * (rotating_B @ rotating_A)
        total_delta = shared_delta + rotating_delta

        shared_actual_rank = int(torch.linalg.matrix_rank(shared_delta.float()).item())
        rotating_actual_rank = int(torch.linalg.matrix_rank(rotating_delta.float()).item())
        total_actual_rank = int(torch.linalg.matrix_rank(total_delta.float()).item())

        _, shared_singular_values, _ = torch.svd(shared_delta.float())
        _, rotating_singular_values, _ = torch.svd(rotating_delta.float())
        _, total_singular_values, _ = torch.svd(total_delta.float())

        print("\nLayer: {}".format(prefix), flush=True)
        print(
            "shared rank={} | rotating rank={} | total rank={} | scaling={:.8f}".format(
                shared_rank,
                rotating_rank,
                total_rank,
                scaling
            ),
            flush=True
        )

        print(
            "shared A shape={} | shared B shape={} | A norm={:.8f} | B norm={:.8f}".format(
                tuple(shared_A.shape),
                tuple(shared_B.shape),
                shared_A.norm().item(),
                shared_B.norm().item()
            ),
            flush=True
        )

        print(
            "rotating A shape={} | rotating B shape={} | A norm={:.8f} | B norm={:.8f}".format(
                tuple(rotating_A.shape),
                tuple(rotating_B.shape),
