# ReLoMAML: Reusing Low-Rank Adaptation Capacity for Parameter-Efficient Meta-Learning

Official code for **ReLoMAML**, a parameter-efficient extension of Model-Agnostic Meta-Learning (MAML) for few-shot image classification.

ReLoMAML trains from scratch using a fixed low-rank budget split into a **shared** component and a **rotating** component. After a warm-up period, the rotating component explores previously unused spectral directions while preserving the network's effective weights at each rotation.

## Setup

```bash
pip install -r requirements.txt
```

Prepare **miniImageNet** or **tieredImageNet** in the format expected by the Torchmeta loaders. Set `--folder` to the directory containing `miniimagenet/` or `tieredimagenet/`, with `train`, `val`, and `test` HDF5 data and JSON label files for each dataset.

## Training

Example: **miniImageNet, 5-way 1-shot**, rank 20 (16 shared + 4 rotating), seed 1:

```bash
python main.py \
  --dataset miniImageNet --folder ./data --use-cuda \
  --N_ways 5 --K_shots_for_support 1 --K_shots_for_query 15 \
  --batch_size 5 --iter 80000 --seed 1 \
  --shared_rank 16 --rotating_rank 4 \
  --first_layer_shared_rank 16 --first_layer_rotating_rank 4 \
  --lora_alpha 20 --first_layer_lora_alpha 20 \
  --inner_lr 0.01 --classifier_inner_lr 0.001 \
  --lora_outer_lr 0.0001 --outer_lr1 0.0001 \
  --warmup_iterations 10000 --rotation_interval 3500 \
  --enable_rotation --output_root ./results --exp_name mini_1shot_seed1
```

For **5-shot**, change `--K_shots_for_support 5`, `--batch_size 3`, and the experiment name. To use tieredImageNet, set `--dataset tieredImageNet` and provide the corresponding data. The experiments use second-order MAML; **do not add `--first-order`**.

The total active rank is `shared_rank + rotating_rank`. Setting `lora_alpha` to this total gives a LoRA scaling factor of 1. The first convolutional layer has separate rank arguments for experiments that require them.

## Evaluation

Evaluate a saved checkpoint (the best checkpoint is selected using validation accuracy):

```bash
python main.py --test --use-cuda --folder ./data \
  --dataset_for_source miniImageNet --dataset_for_target miniImageNet \
  --N_ways 5 --K_shots_for_support 1 --K_shots_for_query 15 \
  --test_num_update_step 10 --seed 1 --flags best \
  --load_path /path/to/best_checkpoint.th
```

Evaluation uses **600 few-shot tasks** and **10 inner-loop adaptation steps**. Checkpoints and logs are saved under `--output_root` / `--exp_name`.

## Results

Rank-20 ReLoMAML uses **20.94% of the trainable parameters** of full-rank MAML while retaining comparable accuracy. Results below are test accuracy (%, mean ± standard deviation across three seeds):

| Dataset | 5-way 1-shot | 5-way 5-shot |
|---|---:|---:|
| miniImageNet | 52.09 ± 0.51 | 68.91 ± 0.68 |
| tieredImageNet | 53.48 ± 0.30 | 72.47 ± 0.64 |

On miniImageNet, measured training time was reduced by **36.0% (1-shot)** and **16.9% (5-shot)** compared with full-rank MAML on an NVIDIA RTX 5090.

## Acknowledgment

The experiments build on [Torchmeta](https://github.com/tristandeleu/pytorch-meta). 
