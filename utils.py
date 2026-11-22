"""Evaluation, accuracy, and experiment-output helpers for ReLoMAML.

This module keeps the output-path conventions used by existing checkpoints.
"""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from inner_loop_optimizer import gradient_update_parameters


def get_accuracy(logits, targets):
    """Return the fraction of correctly classified examples in a batch."""
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
    maximum_tasks=600,
):
    """Evaluate after task-specific adaptation on at most ``maximum_tasks`` tasks.

    The inner loop uses the same parameter-selection function as training:
    shared and rotating low-rank factors, BatchNorm affine parameters, and
    the classifier are adapted. The convolutional base is frozen, and no
    subspace rotation is performed during evaluation.

    Higher-order derivatives are unnecessary for evaluation, so the calls to
    ``gradient_update_parameters`` use first-order mode regardless of the
    ``first_order`` argument retained for compatibility with train/test calls.

    Returns:
        (mean_accuracy, confidence_interval): Accuracy on the fraction scale,
        with a normal-approximation 95% CI half-width over evaluated tasks.
    """
    if classifier_inner_lr is None:
        classifier_inner_lr = inner_lr

    was_training = model.training
    task_accuracies = []
    stop_evaluation = False

    for batch in dataloader:
        train_inputs, train_targets = batch['train']
        test_inputs, test_targets = batch['test']

        train_inputs = train_inputs.to(device=device)
        train_targets = train_targets.to(device=device)
        test_inputs = test_inputs.to(device=device)
        test_targets = test_targets.to(device=device)

        for train_input, train_target, test_input, test_target in zip(
            train_inputs, train_targets, test_inputs, test_targets
        ):
            # Match training-time support adaptation behavior.
            model.train()
            train_logit = model(train_input)
            inner_loss = F.cross_entropy(train_logit, train_target)
            params = gradient_update_parameters(
                model=model,
                loss=inner_loss,
                inner_lr=inner_lr,
                classifier_inner_lr=classifier_inner_lr,
                first_order=True,
            )

            for _ in range(test_num_update_step - 1):
                train_logit = model(train_input, params=params)
                inner_loss = F.cross_entropy(train_logit, train_target)
                params = gradient_update_parameters(
                    model=model,
                    params=params,
                    loss=inner_loss,
                    inner_lr=inner_lr,
                    classifier_inner_lr=classifier_inner_lr,
                    first_order=True,
                )

            model.eval()
            with torch.no_grad():
                test_logit = model(test_input, params=params)
                task_accuracy = get_accuracy(test_logit, test_target).item()

            task_accuracies.append(task_accuracy)
            if len(task_accuracies) >= maximum_tasks:
