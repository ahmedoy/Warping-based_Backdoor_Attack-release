import json
import os
import shutil
from time import time
import sys
import config
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18, VGGMOD
from networks.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
from torchvision.transforms import RandomErasing
from utils.dataloader import PostTensorTransform, get_dataloader, get_transform
from utils.utils import progress_bar
import random


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        netC = VGGMOD(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(
        netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(
        optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def create_sinusoidal_trigger(shape, frequency, amplitude):
    """
    Create a sinusoidal trigger pattern as a torch tensor.

    Parameters:
    - shape: (height, width, channels)
    - frequency: Frequency of the sin wave
    - amplitude: Amplitude of the sin wave

    Returns:
    - trigger: Sinusoidal trigger with shape (channels, height, width)
    """
    height, width, channels = shape

    # Create a sinusoidal pattern along the horizontal axis
    x = torch.arange(width, dtype=torch.float32)
    sin_wave = amplitude * torch.sin(2 * np.pi * frequency * x / width)

    # Replicate the pattern for each row and channel
    trigger = torch.zeros((channels, height, width), dtype=torch.float32)
    for c in range(channels):
        for i in range(height):
            trigger[c, i, :] = sin_wave

    return trigger


def apply_sinusoidal_trigger_tensor(
    images: torch.Tensor,    # shape (B, C, H, W), dtype=torch.float32, values in [0,1]
    trigger: torch.Tensor,   # shape (C, H, W), dtype=torch.float32, values in [0,1]
    alpha: float
) -> torch.Tensor:
    """
    Blend a sinusoidal trigger into a batch of images.

    Parameters:
    - images: batch of images, shape (B, C, H, W), values in [0,1].
    - trigger: trigger pattern, shape (C, H, W), values in [0,1].
    - alpha: blending weight for the trigger (0 = no trigger, 1 = full trigger).

    Returns:
    - poisoned: tensor of same shape as images, values clipped to [0,1].
    """
    # ensure trigger has same dtype & device as images
    trigger = trigger.to(dtype=images.dtype, device=images.device)
    # shape (1, C, H, W) for broadcasting over batch
    trigger = trigger.unsqueeze(0)

    # blend and clamp
    poisoned = images + alpha * trigger
    poisoned = poisoned.clamp(0.0, 1.0)
    return poisoned


def train(netC, optimizerC, schedulerC, train_dl, train_transform, trigger, alpha, epoch, opt):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0
    total_clean = 0
    total_bd = 0
    total_clean_correct = 0
    total_bd_correct = 0
 
    criterion_CE = torch.nn.CrossEntropyLoss()


    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    to_pil = torchvision.transforms.ToPILImage()

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(bs * rate_bd)

        inputs_bd = apply_sinusoidal_trigger_tensor(inputs[:num_bd], trigger=trigger, alpha=alpha)
        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)

        total_inputs = torch.cat([inputs_bd, inputs[num_bd :]], dim=0)
        total_inputs = torch.stack([train_transform(to_pil(total_inputs[i])) for i in range(total_inputs.shape[0])], dim=0)
        total_inputs = transforms(total_inputs).to(opt.device)
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        start = time()
        total_preds = netC(total_inputs)
        total_time += time() - start

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs - num_bd
        total_bd += num_bd
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[num_bd :],
                         dim=1) == total_targets[num_bd:]
        )
        total_bd_correct += torch.sum(torch.argmax(
            total_preds[:num_bd], dim=1) == targets_bd)

        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_bd = total_bd_correct * 100.0 / total_bd

        avg_loss_ce = total_loss_ce / total_sample



        progress_bar(
            batch_idx,
            len(train_dl),
            "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(
                avg_loss_ce, avg_acc_clean, avg_acc_bd),
        )


    schedulerC.step()


def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    test_transform,
    trigger,
    alpha,
    best_clean_acc,
    best_bd_acc,
    best_cross_acc,
    epoch,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    to_pil = torchvision.transforms.ToPILImage()


    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            
            # Evaluate Clean
            clean_inputs = torch.stack([test_transform(to_pil(inputs[i])) for i in range(inputs.shape[0])], dim=0).to(opt.device)
            preds_clean = netC(clean_inputs)
            total_clean_correct += torch.sum(
                torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            inputs_bd = apply_sinusoidal_trigger_tensor(inputs, trigger=trigger, alpha=alpha)
            inputs_bd = torch.stack([test_transform(to_pil(inputs_bd[i])) for i in range(inputs_bd.shape[0])], dim=0).to(opt.device)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets + 1, opt.num_classes)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1)== targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample


            info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                acc_clean, best_clean_acc, acc_bd, best_bd_acc
            )
            progress_bar(batch_idx, len(test_dl), info_string)

    # Save checkpoint
    if (acc_clean > best_clean_acc and acc_bd > best_bd_acc-1) or (acc_clean > best_clean_acc - 1 and acc_bd > best_bd_acc) or (acc_clean > 80 and acc_bd > 95):
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        best_cross_acc = torch.tensor([0])
        state_dict = {
            "netC": netC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "best_cross_acc": best_cross_acc,
            "epoch_current": epoch,
            "trigger": trigger,
            "alpha": alpha,
        }
        # torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "results.txt"), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item(),
                "cross_acc": best_cross_acc.item(),
            }
            json.dump(results_dict, f, indent=2)
            # Save the model checkpoint as a .pt file
        # Replace the extension with .pt added this
        torch.save(state_dict, opt.ckpt_path.replace(".pth.tar", ".pt"))

    continue_training = True
    if acc_clean > 80 and acc_bd > 95:
        print("Accuracies above threshold. Moving to next model")
        continue_training = False
    return best_clean_acc, best_bd_acc, best_cross_acc, continue_training


def main():

    opt = config.get_arguments().parse_args()
    NUMBER_OF_MODELS = 50
    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(
        opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    for run_idx in range(NUMBER_OF_MODELS):
        print(f"\n=== Starting run {run_idx+1}/{NUMBER_OF_MODELS} ===")

        # reset best accuracies for each run
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0
        epoch_current = 0

        # Generate Amp, Freq and Alpha
        frequency = random.randint(4, 7)
        amplitude = random.randint(25, 40)
        alpha = round(random.uniform(0.23, 0.35), 2)

        trigger = create_sinusoidal_trigger(shape=(opt.input_height, opt.input_width, opt.input_channel),
                                            frequency=frequency,
                                            amplitude=amplitude).to(opt.device)

        # Dataset
        train_transform = get_transform(opt, train=True, pretensor_transform=False)
        test_transform = get_transform(opt, train=False, pretensor_transform=False)
        train_dl = get_dataloader(opt, train=True, use_transform=False)
        test_dl = get_dataloader(opt, train=False, use_transform=False)

        # now *per-run* checkpoint & log folders include run_idx
        opt.ckpt_folder = os.path.join(opt.checkpoints,
                                       opt.dataset,
                                       f"run_{run_idx}")
        opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
        os.makedirs(opt.log_dir, exist_ok=True)

        # write args for this run
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w") as f:
            json.dump(opt.__dict__, f, indent=2)

        # adjust ckpt path so they never overwrite
        opt.ckpt_path = os.path.join(opt.ckpt_folder,
                                     f"{opt.dataset}_{opt.attack_mode}_morph_run{run_idx}.pt")

        # (optionally allow continue_training here, or always start fresh)
        print("Train from scratch!!!")
        netC, optimizerC, schedulerC = get_model(opt)

        # now the usual epoch loop:
        for epoch in range(epoch_current, opt.n_iters):
            print(f"Epoch {epoch+1}:")
            train(netC, optimizerC, schedulerC,
                  train_dl, train_transform, trigger, alpha,
                  epoch, opt)
            best_clean_acc, best_bd_acc, best_cross_acc, continue_training = eval(
                netC, optimizerC, schedulerC,
                test_dl, test_transform, trigger, alpha,
                best_clean_acc, best_bd_acc, best_cross_acc,
                epoch, opt
            )

            if not continue_training:
                break

        print(f"=== Finished run {run_idx+1}/{NUMBER_OF_MODELS}:"
              f" best clean {best_clean_acc:.2f}, bd {best_bd_acc:.2f} ===")


if __name__ == "__main__":
    main()
    # Run With
    # python train_sig_attack.py --dataset cifar10 --attack_mode all2one
