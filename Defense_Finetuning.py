'''
Task 3 Defense: Fine-Tuning Backdoored ResNet-18 Models on Clean CIFAR-10
Saves defended checkpoints (.pth) and logs results (defense_log.txt).
'''

import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.transforms import Compose, ToTensor
import numpy as np

import core


# --- DUAL LOGGING UTILITY (CONSOLE + LOG.TXT) ---
class Logger(object):
    def __init__(self, filename="defense_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Redirect standard output to save log.txt automatically
sys.stdout = Logger("defense_log.txt")

print("[✔] Initializing defense evaluation script...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[✔] Using compute device: {device}")


def load_checkpoint(ckpt_path):
    """Loads ResNet-18 checkpoint from specified path."""
    model = core.models.ResNet(18).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    return model


# --- TRIGGER PATTERN DEFINITIONS FOR ASR EVALUATION ---

def apply_badnets_trigger(img_tensors):
    """Applies solid 3x3 white square patch."""
    poisoned = img_tensors.clone()
    poisoned[:, :, -3:, -3:] = 1.0
    return poisoned


def apply_blended_trigger(img_tensors, alpha=0.2):
    """Applies 20% alpha-blended 3x3 patch."""
    poisoned = img_tensors.clone()
    poisoned[:, :, -3:, -3:] = (1.0 - alpha) * img_tensors[:, :, -3:, -3:] + alpha * 1.0
    return poisoned


def evaluate(model, test_loader, trigger_fn=None, target_label=0):
    """Evaluates Clean Data Accuracy (CDA) or Attack Success Rate (ASR)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if trigger_fn is not None:
                inputs = trigger_fn(inputs)
                targets = torch.full_like(targets, target_label)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def fine_tune_and_evaluate(model, ft_loader, test_loader, trigger_fn, epochs=20, lr=0.01, eval_interval=5):
    """Fine-tunes model on clean subset while evaluating every `eval_interval` epochs."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in ft_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if epoch % eval_interval == 0 or epoch == epochs:
            cda = evaluate(model, test_loader)
            asr = evaluate(model, test_loader, trigger_fn=trigger_fn)
            print(f"  Epoch {epoch:02d}/{epochs:02d} complete | CDA: {cda:.2f}% | ASR: {asr:.2f}%")

    return model


def resolve_checkpoint_path(folder_keyword, filename="ckpt_epoch_50.pth"):
    """Finds latest checkpoint file for a given experiment keyword."""
    exp_dir = "experiments"
    if os.path.exists(exp_dir):
        matching_dirs = [d for d in os.listdir(exp_dir) if folder_keyword in d]
        if matching_dirs:
            ckpt = os.path.join(exp_dir, matching_dirs[-1], filename)
            if os.path.exists(ckpt):
                return ckpt
    return None


def run_defense_for_model(model_name, ckpt_path, trigger_fn, ft_loader, test_loader, epochs=20, lr=0.01):
    print("\n" + "=" * 60)
    print(f"  DEFENSE EVALUATION: {model_name}")
    print("=" * 60)

    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[X] Checkpoint not found at: {ckpt_path}. Skipping.")
        return

    print(f"[✔] Loading weights from: {ckpt_path}")
    model = load_checkpoint(ckpt_path)

    cda_before = evaluate(model, test_loader)
    asr_before = evaluate(model, test_loader, trigger_fn=trigger_fn)
    print(f"Pre-Defense  -> CDA: {cda_before:.2f}% | ASR: {asr_before:.2f}%")

    print(f"\n--- Fine-Tuning ({epochs} Epochs on 20% Clean Data, lr={lr}) ---")
    model = fine_tune_and_evaluate(model, ft_loader, test_loader, trigger_fn, epochs=epochs, lr=lr, eval_interval=5)

    cda_after = evaluate(model, test_loader)
    asr_after = evaluate(model, test_loader, trigger_fn=trigger_fn)
    print("\n--- Final Results ---")
    print(f"Post-Defense -> CDA: {cda_after:.2f}% | ASR: {asr_after:.2f}%")
    print(f"ASR Reduction: {asr_before:.2f}% ==> {asr_after:.2f}% (Drop: {asr_before - asr_after:.2f}%)")

    # SAVE DEFENDED CHECKPOINT (.pth)
    parent_dir = os.path.dirname(ckpt_path)
    save_path = os.path.join(parent_dir, "defended_finetuned.pth")
    torch.save({'model': model.state_dict()}, save_path)
    print(f"[✔] Saved defended model checkpoint to: {save_path}")


def main():
    dataset = torchvision.datasets.DatasetFolder
    transform = Compose([ToTensor()])

    print("[✔] Loading local test dataset...")
    testset = dataset(
        root='./data/cifar10/test',
        loader=cv2.imread,
        extensions=('png',),
        transform=transform,
        target_transform=None,
        is_valid_file=None
    )
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    print("[✔] Loading local train dataset (sampling 20% clean subset)...")
    trainset = dataset(
        root='./data/cifar10/train',
        loader=cv2.imread,
        extensions=('png',),
        transform=transform,
        target_transform=None,
        is_valid_file=None
    )
    np.random.seed(666)
    # Increased subset to 20% (10,000 images) for stronger defense signals
    subset_indices = np.random.choice(len(trainset), size=10000, replace=False)
    ft_loader = DataLoader(Subset(trainset, subset_indices), batch_size=128, shuffle=True)

    # 1. Defend BadNets Model
    badnets_ckpt = resolve_checkpoint_path("train_poisoned_DatasetFolder-CIFAR10_2026-10-31_22-01-43")
    run_defense_for_model("BadNets Model", badnets_ckpt, apply_badnets_trigger, ft_loader, test_loader, epochs=20, lr=0.01)

    # 2. Defend Blended Model
    blended_ckpt = resolve_checkpoint_path("train_poisoned_DatasetFolder-CIFAR10_Blended_2026-11-01_00-13-48")
    run_defense_for_model("Blended Model", blended_ckpt, apply_blended_trigger, ft_loader, test_loader, epochs=20, lr=0.01)

    # 3. Defend LabelConsistent Model (Epoch 20)
    lc_ckpt = "experiments/train_poisoned_DatasetFolder-CIFAR10_LabelConsistent_2026-11-01_14-22-35/ckpt_epoch_20.pth"
    run_defense_for_model("LabelConsistent Model", lc_ckpt, apply_badnets_trigger, ft_loader, test_loader, epochs=20, lr=0.01)


if __name__ == '__main__':
    main()