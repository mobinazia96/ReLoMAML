from dataset import get_dataset
from torchmeta.utils.data import BatchMetaDataLoader
from backbone import ConvolutionalNeuralNetwork
from transform import get_transform
from utils import evaluate, get_test_log, get_logger
import os
import torch


def test(args):
    # -------------------------------------------------
    # 1. Check and load the checkpoint first
    # -------------------------------------------------
    if args.load_path is None:
        raise ValueError("In test mode, `--load_path` must be provided.")

    if not os.path.isfile(args.load_path):
        raise FileNotFoundError("Checkpoint file does not exist: {}".format(args.load_path))

    print("Loading checkpoint from: {}".format(args.load_path), flush=True)

    checkpoint = torch.load(args.load_path, map_location=args.device)

    if "model" not in checkpoint:
        raise KeyError("The checkpoint does not contain a `model` state.")

    checkpoint_method = checkpoint.get("method", None)
    if checkpoint_method != "MAML-SRLoRA": raise ValueError("Expected a MAML-SRLoRA checkpoint, but found method `{}`.".format(checkpoint_method))
    state_dict = checkpoint["model"]

    # -------------------------------------------------
    # 2. Recover architecture/configuration information
    #    from the checkpoint
    # -------------------------------------------------
    shared_rank = int(checkpoint.get("shared_rank", args.shared_rank))
    rotating_rank = int(checkpoint.get("rotating_rank", args.rotating_rank))
    total_rank = int(checkpoint.get("total_rank", shared_rank + rotating_rank))
    lora_alpha = float(checkpoint.get("lora_alpha", args.lora_alpha))

    first_shared_key = "features.block1.conv.lora_shared_A"
    first_rotating_key = "features.block1.conv.lora_rotating_A"

    inferred_first_shared_rank = int(state_dict[first_shared_key].shape[0]) if first_shared_key in state_dict else 0
    inferred_first_rotating_rank = int(state_dict[first_rotating_key].shape[0]) if first_rotating_key in state_dict else 0

    first_layer_shared_rank = int(checkpoint.get("first_layer_shared_rank", inferred_first_shared_rank))
    first_layer_rotating_rank = int(checkpoint.get("first_layer_rotating_rank", inferred_first_rotating_rank))
    first_layer_total_rank = int(checkpoint.get("first_layer_total_rank", first_layer_shared_rank + first_layer_rotating_rank))

    arg_first_alpha = getattr(args, "first_layer_lora_alpha", None)
    first_layer_lora_alpha = float(checkpoint.get("first_layer_lora_alpha", lora_alpha if arg_first_alpha is None else arg_first_alpha))

    inner_lr = float(checkpoint.get("inner_lr", args.inner_lr))
    classifier_inner_lr = float(checkpoint.get("classifier_inner_lr", args.classifier_inner_lr))

    if total_rank != shared_rank + rotating_rank:
        raise ValueError("Invalid rest-layer ranks.")

    if first_layer_total_rank != first_layer_shared_rank + first_layer_rotating_rank:
        raise ValueError("Invalid first-layer ranks.")
    inner_lr = float(checkpoint.get("inner_lr", args.inner_lr))
    classifier_inner_lr = float(checkpoint.get("classifier_inner_lr", args.classifier_inner_lr))
    if total_rank != shared_rank + rotating_rank: raise ValueError("Invalid checkpoint ranks: total_rank={} but shared_rank+rotating_rank={}.".format(total_rank, shared_rank + rotating_rank))
    first_conv_key = "features.block1.conv.weight"
    classifier_key = "classifier.weight"

    if first_conv_key in state_dict:
        hidden_size = int(state_dict[first_conv_key].shape[0])
    else:
        hidden_size = args.hidden_size

    if classifier_key in state_dict:
        checkpoint_ways = int(state_dict[classifier_key].shape[0])
    else:
        checkpoint_ways = args.N_ways

    if checkpoint_ways != args.N_ways:
        raise ValueError(
            "Checkpoint was trained with {} ways, but test configuration uses {} ways.".format(
                checkpoint_ways,
                args.N_ways
            )
        )

    

    def get_state_rank(key):
        return int(state_dict[key].shape[0]) if key in state_dict else 0

    # Block 1
    state_first_shared_rank = get_state_rank("features.block1.conv.lora_shared_A")
    state_first_rotating_rank = get_state_rank("features.block1.conv.lora_rotating_A")

    if state_first_shared_rank != first_layer_shared_rank:
        raise ValueError("Block1 shared rank mismatch: metadata={} / state={}.".format(first_layer_shared_rank, state_first_shared_rank))

    if state_first_rotating_rank != first_layer_rotating_rank:
        raise ValueError("Block1 rotating rank mismatch: metadata={} / state={}.".format(first_layer_rotating_rank, state_first_rotating_rank))

    # Blocks 2-4
    state_shared_rank = get_state_rank("features.block2.conv.lora_shared_A")
    state_rotating_rank = get_state_rank("features.block2.conv.lora_rotating_A")

    if state_shared_rank != shared_rank:
        raise ValueError("Blocks2-4 shared rank mismatch: metadata={} / state={}.".format(shared_rank, state_shared_rank))

    if state_rotating_rank != rotating_rank:
        raise ValueError("Blocks2-4 rotating rank mismatch: metadata={} / state={}.".format(rotating_rank, state_rotating_rank))






    
    rotation_enabled = bool(checkpoint.get("rotation_enabled", False))
    warmup_finished = bool(checkpoint.get("warmup_finished", False))
    warmup_iterations = int(checkpoint.get("warmup_iterations", 0))
    rotation_interval = int(checkpoint.get("rotation_interval", 0))
    switch_count = int(checkpoint.get("switch_count", 0))
    
    


    rotation_tag = "rot-w{}-i{}".format(warmup_iterations, rotation_interval) if rotation_enabled else "fixed"
    method_path_name = "MAML-SRLoRA-b1-s{}-r{}-rest-s{}-r{}-{}".format(
        first_layer_shared_rank,
        first_layer_rotating_rank,
        shared_rank,
        rotating_rank,
        rotation_tag
    )
    print(
        "Source Dataset: {} / Target Dataset: {} / Method: MAML-SRLoRA / "
        "Block1: Shared {} + Rotating {} = Total {} / Alpha {} / "
        "Blocks2-4: Shared {} + Rotating {} = Total {} / Alpha {} / "
        "Rotations: {} / Inner LR: {} / Classifier Inner LR: {} / Test Steps: {} / Checkpoint: {} / Test Begin".format(
            args.dataset_for_source,
            args.dataset_for_target,
            first_layer_shared_rank,
            first_layer_rotating_rank,
            first_layer_total_rank,
            first_layer_lora_alpha,
            shared_rank,
            rotating_rank,
            total_rank,
            lora_alpha,
            switch_count,
            inner_lr,
            classifier_inner_lr,
            args.test_num_update_step,
            args.flags
        ),
        flush=True
    )
    # -------------------------------------------------
    # 3. Create the target test dataset
    # -------------------------------------------------
    train_transform = get_transform(train=True)
    test_transform = get_transform(train=False)

    _, _, meta_test_dataset = get_dataset(
        args.dataset_for_target,
        args.folder,
        args.N_ways,
        args.K_shots_for_support,
        args.K_shots_for_query,
        args.download,
        train_transform,
        test_transform
    )

    meta_test_dataloader = BatchMetaDataLoader(
        meta_test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    # -------------------------------------------------
    # 4. Reconstruct exactly the same PiSSA model
    # -------------------------------------------------
    model = ConvolutionalNeuralNetwork(
        in_channels=3,
        out_features=checkpoint_ways,
        hidden_size=hidden_size,
        shared_rank=shared_rank,
        rotating_rank=rotating_rank,
        lora_alpha=lora_alpha,
        first_layer_shared_rank=first_layer_shared_rank,
        first_layer_rotating_rank=first_layer_rotating_rank,
        first_layer_lora_alpha=first_layer_lora_alpha,
        image_size=84
    )
    
    
    model.to(device=args.device)

    # -------------------------------------------------
    # 5. Load the learned residual W, A0, B0 and
    #    classifier initialization
    # -------------------------------------------------
    model.load_state_dict(state_dict, strict=True)

    print("Checkpoint loaded successfully.", flush=True)

    if "iter" in checkpoint:
        print("Checkpoint iteration: {}".format(checkpoint["iter"]), flush=True)

    if "val_acc_mean" in checkpoint:
        print(
            "Checkpoint validation accuracy: {:.4f} ± {:.4f}".format(
                checkpoint["val_acc_mean"],
                checkpoint.get("val_acc_ci", 0.0)
            ),
            flush=True
        )

    if "lora_outer_lr" in checkpoint:
        print(
            "Training LoRA outer LR: {}".format(checkpoint["lora_outer_lr"]),
            flush=True
        )

    print("Loaded configuration: hidden_size={} / shared_rank={} / rotating_rank={} / total_rank={} / alpha={} / rotation_enabled={} / warmup_finished={} / switch_count={} / initialization={}".format(hidden_size, shared_rank, rotating_rank, total_rank, lora_alpha, rotation_enabled, warmup_finished, switch_count, checkpoint.get("initialization", "unknown")), flush=True)

    # evaluate() performs task-specific support adaptation.
    model.train()

    # -------------------------------------------------
    # 6. Create the test log
    # -------------------------------------------------
    log_path = get_test_log(
        output_root=args.output_root,
        exp_name=args.exp_name,
        dataset=args.dataset_for_target,
        train_ways=args.N_ways,
        train_shots=args.K_shots_for_support,
        method=method_path_name,
        source_dataset=args.dataset_for_source,
        target_dataset=args.dataset_for_target,
        rank=total_rank,
        lora_alpha=lora_alpha,
        inner_lr=inner_lr,
        classifier_inner_lr=classifier_inner_lr,
        test_steps=args.test_num_update_step,
        flags=args.flags
    )

    logger = get_logger(log_path)

    # -------------------------------------------------
    # 7. Evaluate on 600 independent target tasks
    # -------------------------------------------------
    test_acc = evaluate(
        model=model,
        dataloader=meta_test_dataloader,
        device=args.device,
        inner_lr=inner_lr,
        first_order=True,
        test_num_update_step=args.test_num_update_step,
        classifier_inner_lr=classifier_inner_lr
    )

    logger.info("Source Dataset: {} / Target Dataset: {} / Method: MAML-SRLoRA / Shared Rank: {} / Rotating Rank: {} / Total Rank: {} / Alpha: {} / Switch Count: {} / Test Acc: {:.4f} / Test Acc Confidence Interval: {:.4f}".format(args.dataset_for_source, args.dataset_for_target, shared_rank, rotating_rank, total_rank, lora_alpha, switch_count, test_acc[0], test_acc[1]))