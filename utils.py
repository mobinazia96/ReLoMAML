import torch
import torch.nn.functional as F
from inner_loop_optimizer import gradient_update_parameters
import numpy as np
import os
import logging
from collections import OrderedDict

def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


def evaluate(
    model,
    dataloader,
    device,
    inner_lr,
    first_order,
    test_num_update_step,
    classifier_inner_lr=None,
    maximum_tasks=600
):
    """
    Evaluate MAML-SRLoRA on a fixed number of tasks.

    During evaluation:
        - the convolution base remains frozen;
        - the shared SRLoRA branch is task-adapted;
        - the rotating SRLoRA branch is task-adapted;
        - the classifier is task-adapted;
        - no subspace rotation is performed.
    """

    if classifier_inner_lr is None:
        classifier_inner_lr = inner_lr

    was_training = model.training
    task_accuracies = []

    stop_evaluation = False

    for batch in dataloader:
        train_inputs, train_targets = batch["train"]
        test_inputs, test_targets = batch["test"]

        train_inputs = train_inputs.to(device=device)
        train_targets = train_targets.to(device=device)
        test_inputs = test_inputs.to(device=device)
        test_targets = test_targets.to(device=device)

        for train_input, train_target, test_input, test_target in zip(
            train_inputs,
            train_targets,
            test_inputs,
            test_targets
        ):
            # During support adaptation, preserve the same behavior
            # used in the original training/evaluation implementation.
            model.train()

            train_logit = model(train_input)

            inner_loss = F.cross_entropy(
                train_logit,
                train_target
            )

            params = gradient_update_parameters(
                model=model,
                loss=inner_loss,
                inner_lr=inner_lr,
                classifier_inner_lr=classifier_inner_lr,

                # No outer derivative is needed during evaluation.
                # Using first-order mode here gives exactly the same
                # numerical adaptation while avoiding higher-order graphs.
                first_order=True
            )

            for _ in range(test_num_update_step - 1):
                train_logit = model(
                    train_input,
                    params=params
                )

                inner_loss = F.cross_entropy(
                    train_logit,
                    train_target
                )

                params = gradient_update_parameters(
                    model=model,
                    params=params,
                    loss=inner_loss,
                    inner_lr=inner_lr,
                    classifier_inner_lr=classifier_inner_lr,
                    first_order=True
                )

            model.eval()

            with torch.no_grad():
                test_logit = model(
                    test_input,
                    params=params
                )

                task_accuracy = get_accuracy(
                    test_logit,
                    test_target
                ).item()

            task_accuracies.append(
                task_accuracy
            )

            if len(task_accuracies) >= maximum_tasks:
                stop_evaluation = True
                break

        if stop_evaluation:
            break

    if was_training:
        model.train()
    else:
        model.eval()

    number_of_tasks = len(task_accuracies)

    if number_of_tasks == 0:
        raise RuntimeError(
            "No tasks were evaluated."
        )

    mean_accuracy = float(
        np.mean(task_accuracies)
    )

    if number_of_tasks > 1:
        confidence_interval = float(
            1.96
            * np.std(
                task_accuracies,
                ddof=1
            )
            / np.sqrt(number_of_tasks)
        )
    else:
        confidence_interval = 0.0

    return mean_accuracy, confidence_interval



def make_dir(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)




def get_save_path(
    output_root,
    exp_name,
    dataset,
    train_ways,
    train_shots,
    method="MAML",
    **kwargs
):
    save_directory = os.path.join(
        output_root,
        str(exp_name),
        "checkpoints",
        dataset,
        "4conv_{0}way_{1}shot".format(
            train_ways,
            train_shots
        ),
        method
    )

    make_dir(save_directory)

    save_path = os.path.join(
        save_directory,
        method
    )

    for key, value in kwargs.items():
        save_path += ".{0}={1}".format(
            key,
            value
        )

    save_path += ".th"

    return save_path

def get_log_path(
    output_root,
    exp_name,
    dataset,
    train_ways,
    train_shots,
    method="MAML",
    **kwargs
):
    log_directory = os.path.join(
        output_root,
        str(exp_name),
        "logs",
        dataset,
        "4conv_{0}way_{1}shot".format(
            train_ways,
            train_shots
        ),
        method
    )

    make_dir(log_directory)

    log_path = os.path.join(
        log_directory,
        method
    )

    for key, value in kwargs.items():
        log_path += ".{0}={1}".format(
            key,
            value
        )

    log_path += ".log"

    return log_path


def get_logger(log_path):
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s %(message)s'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
def get_meta_parameters(meta_named_parameters, device):
    M = OrderedDict()
    for name, param in meta_named_parameters:
        if len(param.shape) == 4:
            shape = min(list(param.reshape(param.size(0), -1).shape))
            M[name] = torch.nn.Parameter(0.928 * torch.ones(size=[shape], device=device, requires_grad=True))
    return M

def get_test_log(
    output_root,
    exp_name,
    dataset,
    train_ways,
    train_shots,
    method="MAML",
    **kwargs
):
    test_directory = os.path.join(
        output_root,
        str(exp_name),
        "test",
        dataset,
        "4conv_{0}way_{1}shot".format(
            train_ways,
            train_shots
        ),
        method
    )

    make_dir(test_directory)

    log_path = os.path.join(
        test_directory,
        method
    )

    for key, value in kwargs.items():
        log_path += ".{0}={1}".format(
            key,
            value
        )

    log_path += ".log"

    return log_path