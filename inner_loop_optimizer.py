from collections import OrderedDict
from torchmeta.modules import MetaModule
import torch

#mige kodum parametr ha bayad tu innerloop update beshan
def is_fast_parameter(name):
    """
    Parameters adapted separately for every task
    in the MAML inner loop.

    Fast parameters:
        - shared SRLoRA branch
        - rotating SRLoRA branch
        - BatchNorm affine parameters
        - classifier

    The convolution base remains frozen.
    """

    is_lora = name.endswith((
        "lora_shared_A",
        "lora_shared_B",
        "lora_rotating_A",
        "lora_rotating_B"
    ))

    is_batchnorm = (
        ".norm." in name
        and (
            name.endswith(".weight")
            or name.endswith(".bias")
        )
    )

    is_classifier = name.startswith("classifier.")

    return is_lora or is_batchnorm or is_classifier





#faghat baraye tahlil
def print_full_gradient_spectrum(
    loss,
    params,
    tag
    ):
    """
    Analyze the support-loss gradient with respect
    to the full convolution weights.

    This function is diagnostic only. It does not
    modify parameters or gradients used for updates.
    """

    conv_weights = OrderedDict(
        (name, parameter)
        for name, parameter in params.items()
        if name.endswith("conv.weight")
    )

    if len(conv_weights) == 0:
        print(
            "\nNo convolution weights found "
            "for gradient-spectrum analysis.",
            flush=True
        )
        return

    # retain_graph=True is necessary because the same loss
    # will afterwards be used to compute LoRA gradients.
    full_grads = torch.autograd.grad(
        loss,
        conv_weights.values(),
        retain_graph=True,
        create_graph=False,
        allow_unused=False
    )

    print(
        "\n========== FULL WEIGHT GRADIENT SPECTRUM: "
        "{} ==========".format(tag),
        flush=True
    )

    for (name, _), grad in zip(
        conv_weights.items(),
        full_grads
    ):
        gradient_matrix = (
            grad.detach()
            .reshape(grad.shape[0], -1)
            .float()
        )

        _, singular_values, _ = torch.svd(
            gradient_matrix
        )

        squared_singular_values = (
            singular_values.square()
        )

        total_energy = (
            squared_singular_values.sum()
        )

        print(
            "\nLayer: {}".format(name),
            flush=True
        )

        print(
            "gradient matrix shape={}".format(
                tuple(gradient_matrix.shape)
            ),
            flush=True
        )

        print(
            "gradient Frobenius norm={:.10f}".format(
                gradient_matrix.norm().item()
            ),
            flush=True
        )

        if total_energy.item() == 0.0:
            print(
                "The full weight gradient is zero.",
                flush=True
            )
            continue

        cumulative_energy = torch.cumsum(
            squared_singular_values,
            dim=0
        ) / total_energy





        def rank_for_energy(threshold):
            threshold_tensor = (
                cumulative_energy.new_tensor(threshold)
            )

            index = torch.searchsorted(
                cumulative_energy,
                threshold_tensor
            )

            return min(
                int(index.item()) + 1,
                singular_values.numel()
            )

        numerical_rank = int(
            torch.matrix_rank(
                gradient_matrix
            ).item()
        )

        stable_rank = (
            total_energy
            / (
                singular_values[0].square()
                + 1e-12
            )
        ).item()

        rank_50 = rank_for_energy(0.50)
        rank_70 = rank_for_energy(0.70)
        rank_80 = rank_for_energy(0.80)
        rank_90 = rank_for_energy(0.90)
        rank_95 = rank_for_energy(0.95)
        rank_99 = rank_for_energy(0.99)

        # Find the LoRA rank corresponding to this layer.
        prefix = name[:-len("weight")]

        shared_A_name = prefix + "lora_shared_A"
        rotating_A_name = prefix + "lora_rotating_A"

        shared_rank = (
            int(params[shared_A_name].shape[0])
            if shared_A_name in params
            else 0
        )

        rotating_rank = (
            int(params[rotating_A_name].shape[0])
            if rotating_A_name in params
            else 0
        )

        total_active_rank = shared_rank + rotating_rank


        print(
            "numerical rank={} | stable rank={:.4f}".format(
                numerical_rank,
                stable_rank
            ),
            flush=True
        )

        print(
            "rank@50%={} | rank@70%={} | rank@80%={} | "
            "rank@90%={} | rank@95%={} | rank@99%={}".format(
                rank_50,
                rank_70,
                rank_80,
                rank_90,
                rank_95,
                rank_99
            ),
            flush=True
        )

        if total_active_rank > 0:
            available_shared_rank = min(
                shared_rank,
                singular_values.numel()
            )

            available_total_rank = min(
                total_active_rank,
                singular_values.numel()
            )

            shared_energy = (
                cumulative_energy[
                    available_shared_rank - 1
                ].item()
                if available_shared_rank > 0
                else 0.0
            )

            total_active_energy = (
                cumulative_energy[
                    available_total_rank - 1
                ].item()
                if available_total_rank > 0
                else 0.0
            )

            rotating_additional_energy = max(
                0.0,
                total_active_energy - shared_energy
            )

            best_total_rank_relative_error = (
                max(
                    0.0,
                    1.0 - total_active_energy
                )
                ** 0.5
            )

            print(
                "shared rank={} | best rank-{} gradient energy={:.2f}%".format(
                    shared_rank,
                    shared_rank,
                    100.0 * shared_energy
                ),
                flush=True
            )

            print(
                "rotating rank={} | additional gradient energy "
                "from rank {} to rank {}={:.2f}%".format(
                    rotating_rank,
                    shared_rank + 1,
                    total_active_rank,
                    100.0 * rotating_additional_energy
                ),
                flush=True
            )

            print(
                "total active rank={} | best rank-{} gradient energy="
                "{:.2f}% | best relative error={:.6f}".format(
                    total_active_rank,
                    total_active_rank,
                    100.0 * total_active_energy,
                    best_total_rank_relative_error
                ),
                flush=True
            )








        top_count = min(
            10,
            singular_values.numel()
        )

        normalized_singular_values = (
            singular_values[:top_count]
            / (
                singular_values[0]
                + 1e-12
            )
        )

        print(
            "top normalized singular values={}".format(
                normalized_singular_values
                .detach()
                .cpu()
                .tolist()
            ),
            flush=True
        )




def gradient_update_parameters(
    model,
    loss,
    params=None,
    inner_lr=None,
    classifier_inner_lr=None,
    first_order=False,
    debug=False,
    debug_tag="",
    debug_full_gradient=False
    ):

    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.MetaModule`, got `{0}`'.format(type(model)))
    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    
    if debug_full_gradient:
        print_full_gradient_spectrum(
            loss=loss,
            params=params,
            tag=debug_tag
        )    

    fast_params = OrderedDict(
    (name, param)
    for name, param in params.items()
    if is_fast_parameter(name)
        )
    if len(fast_params) == 0:
        raise RuntimeError(
            "No fast parameters were found. "
            "Expected shared SRLoRA, rotating SRLoRA, "
            "or classifier parameters."
        )
    grads = torch.autograd.grad(
        loss,
        fast_params.values(),
        create_graph=not first_order,
        allow_unused=False
    )




    if debug:
        print(
            "\n===== INNER GRADIENTS: {} =====".format(
                debug_tag
            ),
            flush=True
        )

        for (name, param), grad in zip(
            fast_params.items(),
            grads
        ):
            print(
                "{} | shape={} | param_norm={:.8f} | "
                "grad_norm={:.8f} | grad_finite={}".format(
                    name,
                    tuple(param.shape),
                    param.detach().norm().item(),
                    grad.detach().norm().item(),
                    torch.isfinite(grad).all().item()
                ),
                flush=True
            )
    
    updated_params = OrderedDict(params)
    
    if classifier_inner_lr is None:
        classifier_inner_lr = inner_lr
    
    for (name, param), grad in zip(
        fast_params.items(),
        grads
    ):
        if name.startswith("classifier."):
            step_size = classifier_inner_lr
        else:
            step_size = inner_lr

        updated_params[name] = (
            param - step_size * grad
        )
    return updated_params