"""Task-specific gradient updates and optional diagnostics for ReLoMAML.

Important: This implementation adapts LoRA factors, BatchNorm affine parameters,
and the classifier in the inner loop. It is *not* classifier-only ANIL.
The update rule is intentionally preserved from the experiment code.
"""

from collections import OrderedDict

import torch
from torchmeta.modules import MetaModule


def is_fast_parameter(name):
    """Identify parameters adapted on each task in the current implementation.

    Task-adapted parameters:
        * Shared and rotating low-rank factors
        * BatchNorm affine weights and biases
        * Classifier weights and biases

    The residual full-rank convolution weights are excluded.
    """
    is_lora = name.endswith(("lora_shared_A", "lora_shared_B", "lora_rotating_A", "lora_rotating_B"))
    is_batchnorm = ".norm." in name and name.endswith((".weight", ".bias"))
    is_classifier = name.startswith("classifier.")
    return is_lora or is_batchnorm or is_classifier


def print_full_gradient_spectrum(loss, params, tag):
    """Optionally inspect support-loss gradients of full convolution weights.

    This diagnostic does not perform any optimization updates. Frozen convolution
    weights cannot be differentiated directly without changing the forward pass;
    in that case, report the limitation and skip the diagnostic.
    """
    conv_weights = OrderedDict((name, parameter) for name, parameter in params.items() if name.endswith("conv.weight"))

    if not conv_weights:
        print("\nNo convolution weights found for gradient-spectrum analysis.", flush=True)
        return

    frozen_weights = [name for name, parameter in conv_weights.items() if not parameter.requires_grad]
    if frozen_weights:
        print("\nSkipping full-weight gradient-spectrum analysis: convolution base weights are frozen (requires_grad=False).", flush=True)
        return

    # Retain the graph so the same loss can subsequently update fast parameters.
    full_grads = torch.autograd.grad(loss, tuple(conv_weights.values()), retain_graph=True, create_graph=False, allow_unused=False)
    print("\n========== FULL WEIGHT GRADIENT SPECTRUM: {} ==========".format(tag), flush=True)

    for (name, _), grad in zip(conv_weights.items(), full_grads):
        gradient_matrix = grad.detach().reshape(grad.shape[0], -1).float()
        singular_values = torch.linalg.svdvals(gradient_matrix)
        squared_singular_values = singular_values.square()
        total_energy = squared_singular_values.sum()

        print("\nLayer: {}".format(name), flush=True)
        print("gradient matrix shape={}".format(tuple(gradient_matrix.shape)), flush=True)
        print("gradient Frobenius norm={:.10f}".format(gradient_matrix.norm().item()), flush=True)

        if total_energy.item() == 0.0:
            print("The full weight gradient is zero.", flush=True)
            continue

        cumulative_energy = torch.cumsum(squared_singular_values, dim=0) / total_energy

        def rank_for_energy(threshold):
            threshold_index = torch.searchsorted(cumulative_energy, cumulative_energy.new_tensor(threshold))
            return min(int(threshold_index.item()) + 1, singular_values.numel())

        numerical_rank = int(torch.linalg.matrix_rank(gradient_matrix).item())
        stable_rank = (total_energy / (singular_values[0].square() + 1e-12)).item()
        ranks = [rank_for_energy(threshold) for threshold in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)]

        prefix = name[:-len("weight")]
        shared_A_name = prefix + "lora_shared_A"
        rotating_A_name = prefix + "lora_rotating_A"
        shared_rank = int(params[shared_A_name].shape[0]) if shared_A_name in params else 0
        rotating_rank = int(params[rotating_A_name].shape[0]) if rotating_A_name in params else 0
        total_active_rank = shared_rank + rotating_rank

        print("numerical rank={} | stable rank={:.4f}".format(numerical_rank, stable_rank), flush=True)
        print("rank@50%={} | rank@70%={} | rank@80%={} | rank@90%={} | rank@95%={} | rank@99%={}".format(*ranks), flush=True)

        if total_active_rank > 0:
            available_shared_rank = min(shared_rank, singular_values.numel())
            available_total_rank = min(total_active_rank, singular_values.numel())
            shared_energy = cumulative_energy[available_shared_rank - 1].item() if available_shared_rank > 0 else 0.0
            total_active_energy = cumulative_energy[available_total_rank - 1].item() if available_total_rank > 0 else 0.0
            rotating_additional_energy = max(0.0, total_active_energy - shared_energy)
            best_total_rank_relative_error = max(0.0, 1.0 - total_active_energy) ** 0.5

            print("shared rank={} | best rank-{} gradient energy={:.2f}%".format(shared_rank, shared_rank, 100.0 * shared_energy), flush=True)
            print("rotating rank={} | additional gradient energy from rank {} to rank {}={:.2f}%".format(rotating_rank, shared_rank + 1, total_active_rank, 100.0 * rotating_additional_energy), flush=True)
            print("total active rank={} | best rank-{} gradient energy={:.2f}% | best relative error={:.6f}".format(total_active_rank, total_active_rank, 100.0 * total_active_energy, best_total_rank_relative_error), flush=True)
