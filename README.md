
# ReLoMAML: Shared–Rotating Low-Rank MAML from Scratch

**PyTorch implementation of ReLoMAML for parameter-efficient few-shot learning.**

## Overview

ReLoMAML is a parameter-efficient extension of Model-Agnostic Meta-Learning (MAML) that combines shared and periodically rotating low-rank adaptation components.

The method aims to reduce the number of trainable parameters and computational cost of meta-learning while maintaining competitive few-shot classification performance.

Unlike approaches that rely on adapting a large set of model parameters, ReLoMAML uses a compact low-rank parameterization with two complementary components:

- **Shared low-rank components:** Maintain a common adaptation subspace throughout training.
- **Rotating low-rank components:** Periodically refresh the adaptation subspace to improve flexibility and representation coverage.

The framework incorporates SVD-based initialization, a warm-up stage, periodic subspace rotation, and gradient-based meta-learning.

## Method

For a weight matrix \(W\), the effective parameterization is

\[
W_{\mathrm{eff}} =
W_{\mathrm{res}} +
s B_s A_s +
s B_r A_r
\]

where:

- \(W_{\mathrm{res}}\) denotes the residual weight component.
- \(A_s,B_s\) represent the shared low-rank factors.
- \(A_r,B_r\) represent the rotating low-rank factors.
- \(s\) is the adaptation scaling factor.

The approach is designed for efficient adaptation in few-shot learning tasks.

## Experimental Setup

The implementation supports few-shot image classification experiments with settings including:

- **Dataset:** miniImageNet
- **Architecture:** Conv-4
- **Evaluation:** 5-way 1-shot and 5-way 5-shot classification
- **Meta-learning:** First-Order MAML
- **Adaptation:** Shared and rotating low-rank parameter updates

Experiments can be conducted with different rank allocations, warm-up durations, and rotation intervals.

## Installation

Clone the repository:

`git clone https://github.com/mobinazia96/ReLoMAML.git`

`cd ReLoMAML`

Install the dependencies listed in `requirements.txt` once the environment specification has been finalized.

## Training and Evaluation

The main experiment entry point is `main.py`.

To inspect the available command-line arguments:

`python main.py --help`

The experiment configurations should specify the dataset, few-shot setting, shared and rotating ranks, warm-up duration, rotation interval, and random seed.

Exact training and evaluation commands will be documented for reproducibility.

## Reproducibility

The experimental protocol includes multiple random seeds, consistent evaluation settings, and comparisons against full-parameter MAML and parameter-efficient baselines.

The final repository will provide the configuration details required to reproduce the reported results.

## Citation

If you find this implementation useful, please consider citing the accompanying ReLoMAML manuscript.

Citation details will be added when available.

## Acknowledgments

This implementation builds upon existing research and open-source implementations in optimization-based meta-learning.

The relevant original repositories and their licenses will be acknowledged here.

## License

License information will be provided in accordance with the dependencies and original source code used in this implementation.
