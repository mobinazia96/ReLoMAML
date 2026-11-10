from dataset import get_dataset
from torchmeta.utils.data import BatchMetaDataLoader

from transform import get_transform
from utils import *
from inner_loop_optimizer import gradient_update_parameters
import torch
import torch.nn.functional as F
from collections import OrderedDict
from backbone import ConvolutionalNeuralNetwork, print_initial_spectral_energy
import time

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
       
        #first_layer_shared_rank = shared_rank if args.first_layer_shared_rank is None else args.first_layer_shared_rank
        #first_layer_rotating_rank = rotating_rank if args.first_layer_rotating_rank is None else args.first_layer_rotating_rank
        #first_layer_total_rank = first_layer_shared_rank + first_layer_rotating_rank
        #first_layer_lora_alpha = args.lora_alpha if args.first_layer_lora_alpha is None else args.first_layer_lora_alpha
            
            
        current_lora_alpha = first_layer_lora_alpha if prefix.startswith("features.block1.") else lora_alpha
        scaling = current_lora_alpha / float(total_rank)

        shared_delta = scaling * (shared_B @ shared_A)
        rotating_delta = scaling * (rotating_B @ rotating_A)
        total_delta = shared_delta + rotating_delta

        shared_actual_rank = int(torch.matrix_rank(shared_delta.float()).item())
        rotating_actual_rank = int(torch.matrix_rank(rotating_delta.float()).item())
        total_actual_rank = int(torch.matrix_rank(total_delta.float()).item())

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
                rotating_A.norm().item(),
                rotating_B.norm().item()
            ),
            flush=True
        )

        print(
            "shared delta norm={:.8f} | actual rank={} | top singular values={}".format(
                shared_delta.norm().item(),
                shared_actual_rank,
                shared_singular_values[:min(5, shared_singular_values.numel())].detach().cpu().tolist()
            ),
            flush=True
        )

        print(
            "rotating delta norm={:.8f} | actual rank={} | top singular values={}".format(
                rotating_delta.norm().item(),
                rotating_actual_rank,
                rotating_singular_values[:min(5, rotating_singular_values.numel())].detach().cpu().tolist()
            ),
            flush=True
        )

        print(
            "total delta norm={:.8f} | actual rank={} | top singular values={}".format(
                total_delta.norm().item(),
                total_actual_rank,
                total_singular_values[:min(5, total_singular_values.numel())].detach().cpu().tolist()
            ),
            flush=True
        )

        if print_values:
            print("shared A first values:", shared_A.flatten()[:5].detach().cpu().tolist(), flush=True)
            print("shared B first values:", shared_B.flatten()[:5].detach().cpu().tolist(), flush=True)
            print("rotating A first values:", rotating_A.flatten()[:5].detach().cpu().tolist(), flush=True)
            print("rotating B first values:", rotating_B.flatten()[:5].detach().cpu().tolist(), flush=True)





@torch.no_grad()
def print_inner_update(
    before_params,
    after_params,
    tag
        ):
    print(
        "\n========== INNER UPDATE: {} ==========".format(tag),
        flush=True
    )

    for name in before_params:
        if (
            "lora_" in name
            or ".norm." in name
            or name.startswith("classifier.")
        ):
            change = (
                after_params[name]
                - before_params[name]
            ).norm().item()

            print(
                "{} | change_norm={:.10f}".format(
                    name,
                    change
                ),
                flush=True
            )

    slow_changes = []

    for name in before_params:
        is_fast = (
            "lora_" in name
            or ".norm." in name
            or name.startswith("classifier.")
        )

        if not is_fast:
            change = (
                after_params[name]
                - before_params[name]
            ).norm().item()

            slow_changes.append((name, change))

    if len(slow_changes) > 0:
        maximum_slow_change = max(
            change
            for _, change in slow_changes
        )
    else:
        maximum_slow_change = 0.0

    print(
        "Maximum slow-parameter inner change: {:.10f}".format(
            maximum_slow_change
        ),
        flush=True
    )


def print_outer_gradients(model, tag):
    print(
        "\n========== OUTER GRADIENTS: {} ==========".format(tag),
        flush=True
    )

    for name, parameter in model.meta_named_parameters():
        should_print = (
    "lora_" in name
    or ".norm." in name
    or name.endswith("conv.weight")
    or name.startswith("classifier.")
)

        if not should_print:
            continue

        if parameter.grad is None:
            print(
                "{} | grad=None".format(name),
                flush=True
            )
        else:
            print(
                "{} | grad_norm={:.10f} | finite={}".format(
                    name,
                    parameter.grad.detach().norm().item(),
                    torch.isfinite(parameter.grad).all().item()
                ),
                flush=True
            )


@torch.no_grad()
def print_outer_update(
    model,
    before_outer,
    tag     
        ):
    print(
        "\n========== OUTER UPDATE: {} ==========".format(tag),
        flush=True
    )

    for name, parameter in model.meta_named_parameters():
        should_print = (
    "lora_" in name
    or ".norm." in name
    or name.endswith("conv.weight")
    or name.startswith("classifier.")
)

        if not should_print:
            continue

        change = (
            parameter.detach()
            - before_outer[name]
        ).norm().item()

        print(
            "{} | change_norm={:.10f}".format(
                name,
                change
            ),
            flush=True
        )




def get_srlora_named_parameters(model, shared_rank, rotating_rank):
    shared_named_parameters = OrderedDict((name, parameter) for name, parameter in model.meta_named_parameters() if name.endswith("lora_shared_A") or name.endswith("lora_shared_B"))
    rotating_named_parameters = OrderedDict((name, parameter) for name, parameter in model.meta_named_parameters() if name.endswith("lora_rotating_A") or name.endswith("lora_rotating_B"))

    if shared_rank > 0 and len(shared_named_parameters) == 0:
        raise RuntimeError("No global shared SRLoRA parameters were found.")

    if rotating_rank > 0 and len(rotating_named_parameters) == 0:
        raise RuntimeError("No global rotating SRLoRA parameters were found.")

    return shared_named_parameters, rotating_named_parameters





def train(args):
    shared_rank = args.shared_rank
    rotating_rank = args.rotating_rank
    total_rank = shared_rank + rotating_rank

    first_layer_shared_rank = shared_rank if args.first_layer_shared_rank is None else args.first_layer_shared_rank
    first_layer_rotating_rank = rotating_rank if args.first_layer_rotating_rank is None else args.first_layer_rotating_rank
    first_layer_total_rank = first_layer_shared_rank + first_layer_rotating_rank
    first_layer_lora_alpha = args.lora_alpha if args.first_layer_lora_alpha is None else args.first_layer_lora_alpha
    
    
    
    method_name = "MAML-SRLoRA"
    if args.enable_rotation:
        rotation_tag = "rot-w{}-i{}".format(
            args.warmup_iterations,
            args.rotation_interval
        )
    else:
        rotation_tag = "fixed"

    method_path_name = "MAML-SRLoRA-b1-s{}-r{}-rest-s{}-r{}-{}".format(
        first_layer_shared_rank,
        first_layer_rotating_rank,
        shared_rank,
        rotating_rank,
        rotation_tag
    )

    print(
        "Dataset: {} / Method: {} / "
        "Block1: Shared Rank {} + Rotating Rank {} = Total Rank {} / Alpha {} / "
        "Blocks2-4: Shared Rank {} + Rotating Rank {} = Total Rank {} / Alpha {} / "
        "SRLoRA Outer LR: {} / Classifier Outer LR: {} / Train Begin".format(
            args.dataset,
            method_name,
            first_layer_shared_rank,
            first_layer_rotating_rank,
            first_layer_total_rank,
            first_layer_lora_alpha,
            shared_rank,
            rotating_rank,
            total_rank,
            args.lora_alpha,
            args.lora_outer_lr,
            args.outer_lr1
        ),
        flush=True
    )
    
    train_transform = get_transform(train=True)
    test_transform = get_transform(train=False)

    meta_train_dataset, meta_val_dataset, meta_test_dataset = get_dataset(args.dataset, args.folder, args.N_ways, args.K_shots_for_support, args.K_shots_for_query, args.download, train_transform, test_transform)

    meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    meta_test_dataloader = BatchMetaDataLoader(meta_test_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)

    model = ConvolutionalNeuralNetwork(
        in_channels=3,
        out_features=args.N_ways,
        hidden_size=args.hidden_size,
        shared_rank=shared_rank,
        rotating_rank=rotating_rank,
        lora_alpha=args.lora_alpha,
        first_layer_shared_rank=first_layer_shared_rank,
        first_layer_rotating_rank=first_layer_rotating_rank,
        first_layer_lora_alpha=first_layer_lora_alpha,
        image_size=84
    )
    
    model.to(device=args.device)

    if args.debug_lora:
        print_initial_spectral_energy(model)

    if args.debug_lora:
        initial_params = OrderedDict(
            model.meta_named_parameters()
        )

        print_srlora_state(
            params=initial_params,
            lora_alpha=args.lora_alpha,
            first_layer_lora_alpha=first_layer_lora_alpha,
            tag="INITIAL MODEL"
        )



    if args.debug_lora:
        print(
            "\n===== Model parameters =====",
            flush=True
        )

        for name, param in model.meta_named_parameters():
            if "lora_" in name:
                print(
                    name,
                    "shape =", tuple(param.shape),
                    "requires_grad =", param.requires_grad,
                    "norm =", param.detach().norm().item(),
                    flush=True
                )


    model.train()
    
    shared_named_parameters, rotating_named_parameters = get_srlora_named_parameters(model, args.shared_rank, args.rotating_rank)


    classifier_named_parameters = OrderedDict(
        (name, parameter)
        for name, parameter
        in model.meta_named_parameters()
        if name.startswith("classifier.")
    )

    if len(classifier_named_parameters) == 0:
        raise RuntimeError(
            "No classifier parameters were found."
        )


    batchnorm_named_parameters = OrderedDict(
        (name, parameter)
        for name, parameter
        in model.meta_named_parameters()
        if (
            ".norm." in name
            and (
                name.endswith(".weight")
                or name.endswith(".bias")
            )
        )
    )

    if len(batchnorm_named_parameters) == 0:
        raise RuntimeError(
            "No BatchNorm affine parameters were found."
        )





    shared_optimizer = torch.optim.Adam(list(shared_named_parameters.values()), lr=args.lora_outer_lr) if len(shared_named_parameters) > 0 else None


    rotating_optimizer = (
        torch.optim.Adam(
            list(rotating_named_parameters.values()),
            lr=args.lora_outer_lr
        )
        if len(rotating_named_parameters) > 0
        else None
    )

    classifier_optimizer = torch.optim.Adam(
        list(classifier_named_parameters.values()),
        lr=args.outer_lr1
    )


    batchnorm_optimizer = torch.optim.Adam(
        list(batchnorm_named_parameters.values()),
        lr=args.outer_lr1
    )

    log_path = get_log_path(
        output_root=args.output_root,
        exp_name=args.exp_name,
        dataset=args.dataset,
        train_ways=args.N_ways,
        train_shots=args.K_shots_for_support,
        method=method_path_name,
        rank=total_rank,
        lora_alpha=args.lora_alpha,
        lora_outer_lr=args.lora_outer_lr,
        outer_lr1=args.outer_lr1,
        inner_lr=args.inner_lr,
        classifier_inner_lr=args.classifier_inner_lr,
        batch_size=args.batch_size
    )
    
    logger = get_logger(log_path)

    outer_loss_list = []
    best_val_acc = 0.0
    warmup_finished = False
    switch_count = 0
    rotation_history = []



    if args.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)

    gpu_name = torch.cuda.get_device_name(args.device) if args.device.type == 'cuda' else 'CPU'

    logger.info('Device: {}'.format(args.device))
    logger.info('GPU: {}'.format(gpu_name))

    if args.device.type == 'cuda':
        torch.cuda.synchronize(args.device)

    training_start_time = time.perf_counter()




    for batch_idx, batch in enumerate(
        meta_train_dataloader
    ):
        
        
        if shared_optimizer is not None:
            shared_optimizer.zero_grad()

        
        if rotating_optimizer is not None:
            rotating_optimizer.zero_grad()

        batchnorm_optimizer.zero_grad()

        classifier_optimizer.zero_grad()
                
        
        number_of_tasks = 0
     
        
        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.to(device=args.device)
        train_targets = train_targets.to(device=args.device)

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.to(device=args.device)
        test_targets = test_targets.to(device=args.device)

        outer_loss = torch.tensor(0., device=args.device)
        accuracy = torch.tensor(0., device=args.device)
        for task_idx, (train_input, train_target, test_input, test_target) in enumerate(zip(train_inputs, train_targets, test_inputs, test_targets)):
            acc_list = []



            debug_this_task = (
                args.debug_lora
                and batch_idx < args.debug_batches
                and task_idx == 0
            )

            ## 0 update step accuracy
            test_logit = model(test_input)
            
            if debug_this_task:
                zero_step_test_logit = (
                    test_logit.detach().clone()
                )
            
            with torch.no_grad():
                acc_list.append(get_accuracy(test_logit, test_target).item())

            ## 1 update step process
            train_logit = model(train_input)
            
            
            if debug_this_task:
                before_inner = OrderedDict(
                    model.meta_named_parameters()
                )            
            inner_loss = F.cross_entropy(train_logit, train_target)
            
            
            params = gradient_update_parameters(
                model=model,
                loss=inner_loss,
                inner_lr=args.inner_lr,
                first_order=args.first_order,
                debug=debug_this_task,
                debug_tag=(
                    "meta_batch={} task={} inner_step=1".format(
                        batch_idx,
                        task_idx
                    )
                ),
                classifier_inner_lr=args.classifier_inner_lr,
                debug_full_gradient=debug_this_task
            )




            if debug_this_task:
                print_inner_update(
                    before_params=before_inner,
                    after_params=params,
                    tag=(
                        "meta_batch={} task={} inner_step=1".format(
                            batch_idx,
                            task_idx
                        )
                    )
                )

                print_srlora_state(
                    params=params,
                    lora_alpha=args.lora_alpha,
                    first_layer_lora_alpha=first_layer_lora_alpha,
                    tag=(
                        "AFTER INNER STEP 1 / "
                        "meta_batch={} task={}".format(
                            batch_idx,
                            task_idx
                        )
                    ),
                    print_values=False
                )





       

            test_logit = model(test_input, params=params)
            

            if debug_this_task:
                logits_change = (
                    test_logit.detach()
                    - zero_step_test_logit
                ).norm().item()

                print(
                    "\nQuery-logit change after inner step 1: {:.10f}".format(
                        logits_change
                    ),
                    flush=True
                )

            
            with torch.no_grad():
                acc_list.append(get_accuracy(test_logit, test_target).item())

            ## 2 ~ k update step process
            for i in range(args.train_num_update_step - 1):
                
                if debug_this_task:
                    previous_params = params

                train_logit = model(train_input, params)
                inner_loss = F.cross_entropy(train_logit, train_target)
                is_final_inner_step = (
                    i == args.train_num_update_step - 2
                )

                params = gradient_update_parameters(
                    model=model,
                    params=params,
                    loss=inner_loss,
                    inner_lr=args.inner_lr,
                    first_order=args.first_order,
                    debug=debug_this_task,
                    debug_tag=(
                        "meta_batch={} task={} inner_step={}".format(
                            batch_idx,
                            task_idx,
                            i + 2
                        )
                    ),
                    classifier_inner_lr=args.classifier_inner_lr,
                    debug_full_gradient=(
                        debug_this_task
                        and is_final_inner_step
                    )
                )

                if debug_this_task:
                    print_inner_update(
                        before_params=previous_params,
                        after_params=params,
                        tag=(
                            "meta_batch={} task={} inner_step={}".format(
                                batch_idx,
                                task_idx,
                                i + 2
                            )
                        )
                    )




                test_logit = model(test_input, params=params)
                with torch.no_grad():
                    acc_list.append(get_accuracy(test_logit, test_target).item())

                if (
                    debug_this_task
                    and i == args.train_num_update_step - 2
                ):
                    print_srlora_state(
                        params=params,
                        lora_alpha=args.lora_alpha,
                        first_layer_lora_alpha=first_layer_lora_alpha,
                        tag=(
                            "AFTER FINAL INNER STEP / "
                            "meta_batch={} task={}".format(
                                batch_idx,
                                task_idx
                            )
                        ),
                        print_values=False
                    )


            


            number_of_tasks += 1
            
            outer_loss += F.cross_entropy(test_logit, test_target)
            
            
            with torch.no_grad():
                accuracy += get_accuracy(test_logit, test_target)
        
        
        outer_loss = (
            outer_loss / float(number_of_tasks)
        )

        accuracy = (
            accuracy / float(number_of_tasks)
        )  
        
        outer_loss_list.append(outer_loss.item())
        
        
        debug_this_batch = (
            args.debug_lora
            and batch_idx < args.debug_batches
        )


        # ---------------------------------------------------
        # Query meta-gradients with respect to:
        #
        #   1. Shared SRLoRA meta-initialization
        #   2. Rotating SRLoRA meta-initialization
        #   3. Shared classifier initialization
        #
        # No outer gradient is requested with respect to
        # the frozen convolution base.
        # ---------------------------------------------------
        outer_named_parameters = OrderedDict()

        for name, parameter in shared_named_parameters.items():
            outer_named_parameters[name] = parameter

        for name, parameter in rotating_named_parameters.items():
            outer_named_parameters[name] = parameter



        for name, parameter in batchnorm_named_parameters.items():
            outer_named_parameters[name] = parameter



        for name, parameter in classifier_named_parameters.items():
            outer_named_parameters[name] = parameter
        

        outer_grads_tuple = torch.autograd.grad(
            outer_loss,
            tuple(
                outer_named_parameters.values()
            ),
            create_graph=False,
            retain_graph=False,
            allow_unused=False
        )


        outer_grads = OrderedDict(
            (name, grad)
            for (name, _), grad in zip(
                outer_named_parameters.items(),
                outer_grads_tuple
            )
        )


        if debug_this_batch:
            before_outer = OrderedDict(
                (name, parameter.detach().clone())
                for name, parameter in model.meta_named_parameters()
            )

        for name, parameter in outer_named_parameters.items():
            parameter.grad = outer_grads[name].detach().clone()

        if debug_this_batch:
            print_outer_gradients(
                model=model,
                tag="meta_batch={}".format(batch_idx)
            )

        if shared_optimizer is not None:
            shared_optimizer.step()


        if rotating_optimizer is not None:
            rotating_optimizer.step()

        batchnorm_optimizer.step()
        classifier_optimizer.step()

        if debug_this_batch:
            print_outer_update(
                model=model,
                before_outer=before_outer,
                tag="meta_batch={}".format(batch_idx)
            )





        current_iteration = batch_idx + 1

        if (
            args.enable_rotation
            and not warmup_finished
            and current_iteration >= args.warmup_iterations
                    ):
            warmup_reports = (
                model.finalize_srlora_warmup()
            )

            warmup_finished = True

            logger.info(
                "========== SRLoRA WARM-UP FINALIZED "
                "AT ITERATION {} ==========".format(
                    current_iteration
                )
            )

            for report in warmup_reports:
                logger.info(
                    "layer={} | changed={} | reason={} | "
                    "relative_weight_error={:.8e} | "
                    "shared_energy={:.4f} | "
                    "spectral_gap={:.6f}".format(
                        report["layer"],
                        report.get("changed", False),
                        report.get("reason", "unknown"),
                        report.get(
                            "relative_weight_error",
                            0.0
                        ),
                        report.get(
                            "shared_energy_fraction",
                            0.0
                        ),
                        report.get(
                            "spectral_gap",
                            0.0
                        )
                    )
                )

            # Both shared and rotating factors were rewritten
            # by the warm-up SVD. Their old Adam states are invalid.
            shared_optimizer = (
                torch.optim.Adam(
                    list(shared_named_parameters.values()),
                    lr=args.lora_outer_lr
                )
                if len(shared_named_parameters) > 0
                else None
            )

            rotating_optimizer = (
                torch.optim.Adam(
                    list(rotating_named_parameters.values()),
                    lr=args.lora_outer_lr
                )
                if len(rotating_named_parameters) > 0
                else None
            )
            
            
            logger.info(
                "Available SRLoRA optimizer states were reset after warm-up finalization."
            )

        should_rotate = (
            args.enable_rotation
            and warmup_finished
            and current_iteration
            > args.warmup_iterations
            and (
                current_iteration
                - args.warmup_iterations
            ) % args.rotation_interval == 0
            and model.has_remaining_srlora_directions()
                )

        if should_rotate:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            rotation_reports = (
                model.rotate_srlora_subspaces()
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()


                
            changed_reports = [
                report
                for report in rotation_reports
                if report.get("changed", False)
            ]

            if len(changed_reports) > 0:
                switch_count += 1

                rotation_event = {
                    "iteration": current_iteration,
                    "switch_count": switch_count,
                    "layers": rotation_reports
                }

                rotation_history.append(
                    rotation_event
                )

                logger.info(
                    "========== SRLoRA ROTATION {} "
                    "AT ITERATION {} ==========".format(
                        switch_count,
                        current_iteration
                    )
                )

                for report in rotation_reports:
                    logger.info(
                        "layer={} | changed={} | reason={} | "
                        "replaced={} | retained={} | "
                        "remaining_pool={} | "
                        "layer_rotation_count={} | "
                        "relative_weight_error={:.8e}".format(
                            report["layer"],
                            report.get(
                                "changed",
                                False
                            ),
                            report.get(
                                "reason",
                                "unknown"
                            ),
                            report.get(
                                "replace_count",
                                0
                            ),
                            report.get(
                                "retain_count",
                                0
                            ),
                            report.get(
                                "remaining_pool",
                                0
                            ),
                            report.get(
                                "rotation_count",
                                0
                            ),
                            report.get(
                                "relative_weight_error",
                                0.0
                            )
                        )
                    )

                
                changed_layers = {report["layer"] for report in changed_reports}

                if rotating_optimizer is not None:
                    for name, parameter in rotating_named_parameters.items():
                        if any(name.startswith(layer + ".") for layer in changed_layers):
                            rotating_optimizer.state.pop(parameter, None)

                logger.info(
                    "Rotating optimizer states were cleared only "
                    "for changed SRLoRA layers."
                )

                if args.device.type == "cuda":
                    torch.cuda.empty_cache()

                    
        if batch_idx  % 1 == 0:
            logger.info('batch_idx: {0} / train avg end-acc: {1:.4f} / loss: {2:.4f}'.format(batch_idx, accuracy.item(), outer_loss.item()))
            logger.info('per step accs: {0}'.format(acc_list))

        if (batch_idx+1) % 2000 == 0:
            val_acc = evaluate(
                model,
                meta_val_dataloader,
                args.device,
                args.inner_lr,
                args.first_order,
                 
                args.test_num_update_step,
                args.classifier_inner_lr,
            )
            logger.info("Val Acc: {0:.4f} / Val Acc Confidence Intervals (CI): {1:.4f}".format(val_acc[0], val_acc[1]))
            if best_val_acc < val_acc[0]:
                save_path = get_save_path(
                    output_root=args.output_root,
                    exp_name=args.exp_name,
                    dataset=args.dataset,
                    train_ways=args.N_ways,
                    train_shots=args.K_shots_for_support,
                    method=method_path_name,
                    rank=total_rank,
                    lora_alpha=args.lora_alpha,
                    outer_lr1=args.outer_lr1,
                    batch_size=args.batch_size,
                    flags="best",
                    lora_outer_lr=args.lora_outer_lr
                )

                checkpoint = {
                    "model": model.state_dict(),

                    "shared_optimizer": shared_optimizer.state_dict() if shared_optimizer is not None else None,
                    "rotating_optimizer": rotating_optimizer.state_dict() if rotating_optimizer is not None else None,
                    "batchnorm_optimizer": batchnorm_optimizer.state_dict(),
                    "classifier_optimizer": classifier_optimizer.state_dict(),

                    "iter": batch_idx + 1,
                    "val_acc_mean": val_acc[0],
                    "val_acc_ci": val_acc[1],
                    "losses": outer_loss_list,

                    "method": "MAML-SRLoRA",
                    "shared_rank": args.shared_rank,
                    "rotating_rank": args.rotating_rank,
                    "total_rank": total_rank,

                    "lora_alpha": args.lora_alpha,
                    
                    "first_layer_shared_rank": first_layer_shared_rank,
                    "first_layer_rotating_rank": first_layer_rotating_rank,
                    "first_layer_total_rank": first_layer_total_rank,
                    "first_layer_lora_alpha": first_layer_lora_alpha,
                    
                    "lora_outer_lr": args.lora_outer_lr,
                    "classifier_outer_lr": args.outer_lr1,
                    "inner_lr": args.inner_lr,
                    "classifier_inner_lr": args.classifier_inner_lr,

                    "outer_update_rule": (
                        "persistent_query_meta_update_on_"
                        "shared_and_rotating_srlora_branches"
                    ),

                    "initialization": (
                        "top_total_rank_svd_split_into_"
                        "shared_and_rotating_branches"
                    ),
                    "rotation_enabled": args.enable_rotation,
                    "warmup_finished": warmup_finished,
                    "switch_count": switch_count,
                    "rotation_history": rotation_history,
                    "warmup_iterations": args.warmup_iterations,
                    "rotation_interval": args.rotation_interval
                }

                torch.save(checkpoint, save_path)
                best_val_acc = val_acc[0]              
                


            test_acc = evaluate(
                model,
                meta_test_dataloader,
                args.device,
                args.inner_lr,
                args.first_order,
               
                args.test_num_update_step,
                args.classifier_inner_lr,
            ) 
            
            logger.info("Test Acc: {0:.4f} / Test Acc Confidence Intervals (CI): {1:.4f}".format(test_acc[0], test_acc[1]))
        if batch_idx +1 >= args.iter:
            break




    if args.device.type == 'cuda':
        torch.cuda.synchronize(args.device)

    training_end_time = time.perf_counter()
    training_time_seconds = training_end_time - training_start_time
    training_time_hours = training_time_seconds / 3600.0

    if args.device.type == 'cuda':
        peak_allocated = torch.cuda.max_memory_allocated(args.device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(args.device) / 1024**3
    else:
        peak_allocated = 0.0
        peak_reserved = 0.0

    logger.info('========== Training Summary ==========')
    logger.info('Method: {}'.format(method_name))
    logger.info('GPU: {}'.format(gpu_name))
    logger.info('Batch Size: {}'.format(args.batch_size))
    logger.info('Meta Batch Iterations: {}'.format(batch_idx + 1))
    logger.info('Inner Steps: {}'.format(args.train_num_update_step))
    logger.info('Hidden Size: {}'.format(args.hidden_size))
    logger.info('Shared Rank: {}'.format(shared_rank))
    logger.info('Rotating Rank: {}'.format(rotating_rank))
    logger.info('Total Rank: {}'.format(total_rank))
    logger.info('Rotation Enabled: {}'.format(args.enable_rotation))
    logger.info('Number of Rotations: {}'.format(switch_count))
    logger.info('Peak GPU Memory Allocated: {:.3f} GB'.format(peak_allocated))
    logger.info('Peak GPU Memory Reserved: {:.3f} GB'.format(peak_reserved))
    logger.info('Total Training Time: {:.2f} seconds'.format(training_time_seconds))
    logger.info('Total Training Time: {:.2f} hours'.format(training_time_hours))
    logger.info('======================================')






    val_acc = evaluate(
        model,
        meta_val_dataloader,
        args.device,
        args.inner_lr,
        args.first_order,
        
        args.test_num_update_step,
        args.classifier_inner_lr,
    )
    logger.info("Val Acc: {0:.4f} / Val Acc Confidence Intervals (CI): {1:.4f}".format(val_acc[0], val_acc[1]))

    # Save last a model
    save_path = get_save_path(
        output_root=args.output_root,
        exp_name=args.exp_name,
        dataset=args.dataset,
        train_ways=args.N_ways,
        train_shots=args.K_shots_for_support,
        method=method_path_name,
        rank=total_rank,
        lora_alpha=args.lora_alpha,
        outer_lr1=args.outer_lr1,
        batch_size=args.batch_size,
        flags="last",
        lora_outer_lr=args.lora_outer_lr
    )
    
    

    checkpoint = {
        "model": model.state_dict(),

        "shared_optimizer": shared_optimizer.state_dict() if shared_optimizer is not None else None,
        "rotating_optimizer": rotating_optimizer.state_dict() if rotating_optimizer is not None else None,
        "batchnorm_optimizer": batchnorm_optimizer.state_dict(),
        "classifier_optimizer": classifier_optimizer.state_dict(),

        "iter": batch_idx + 1,
        "val_acc_mean": val_acc[0],
        "val_acc_ci": val_acc[1],
        "losses": outer_loss_list,

        "method": "MAML-SRLoRA",
        "shared_rank": args.shared_rank,
        "rotating_rank": args.rotating_rank,
        "total_rank": total_rank,

        "lora_alpha": args.lora_alpha,
       
       
       
       "first_layer_shared_rank": first_layer_shared_rank,
        "first_layer_rotating_rank": first_layer_rotating_rank,
        "first_layer_total_rank": first_layer_total_rank,
        "first_layer_lora_alpha": first_layer_lora_alpha,
            
       
       
       
       
        "lora_outer_lr": args.lora_outer_lr,
        "classifier_outer_lr": args.outer_lr1,
        "inner_lr": args.inner_lr,
        "classifier_inner_lr": args.classifier_inner_lr,

        "outer_update_rule": (
            "persistent_query_meta_update_on_"
            "shared_and_rotating_srlora_branches"
        ),

        "initialization": (
            "top_total_rank_svd_split_into_"
            "shared_and_rotating_branches"
        ),

        "rotation_enabled": args.enable_rotation,
        "warmup_finished": warmup_finished,
        "switch_count": switch_count,
        "rotation_history": rotation_history,
        "warmup_iterations": args.warmup_iterations,
        "rotation_interval": args.rotation_interval,

        "training_time_seconds": training_time_seconds,
        "training_time_hours": training_time_hours,
        "gpu_name": gpu_name,
        "peak_memory_allocated_gb": peak_allocated,
        "peak_memory_reserved_gb": peak_reserved
    }

    torch.save(checkpoint, save_path)
    test_acc = evaluate(
        model,
        meta_test_dataloader,
        args.device,
        args.inner_lr,
        args.first_order,
        
        args.test_num_update_step,
        args.classifier_inner_lr,
    )
    logger.info("Test Acc: {0:.4f} / Test Acc Confidence Intervals (CI): {1:.4f}".format(test_acc[0], test_acc[1]))