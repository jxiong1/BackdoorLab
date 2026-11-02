'''
Generates the 3 Defended Model Visualization Figures for LaTeX Report:
- defended_train_poisoned_datasetfolder_cifar10_2026_10_31_22_01_43_demo.png
- defended_train_poisoned_datasetfolder_cifar10_blended_2026_11_01_00_13_48_demo.png
- defended_train_poisoned_datasetfolder_cifar10_labelconsistent_2026_11_01_14_22_35_demo.png
'''

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import core

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path):
    """Loads fine-tuned ResNet-18 model weights."""
    model = core.models.ResNet(18).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


# --- TRIGGER FUNCTIONS ---

def apply_badnets_trigger(img_tensor):
    """Applies 3x3 white square patch to bottom-right corner."""
    poisoned = img_tensor.clone()
    poisoned[:, -3:, -3:] = 1.0
    return poisoned


def apply_blended_trigger(img_tensor, alpha=0.2):
    """Applies 20% alpha-blended 3x3 patch to bottom-right corner."""
    poisoned = img_tensor.clone()
    poisoned[:, -3:, -3:] = (1.0 - alpha) * img_tensor[:, -3:, -3:] + alpha * 1.0
    return poisoned


def run_inference(model, img_tensor):
    """Runs model forward pass and returns prediction index and confidence percentage."""
    input_batch = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_batch)
        probs = F.softmax(outputs, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item() * 100.0
    return CLASSES[pred_idx], confidence


def save_visualization(clean_tensor, poisoned_tensor, clean_pred, clean_conf, 
                       poisoned_pred, poisoned_conf, title_name, save_filename):
    """Plots clean vs triggered prediction side-by-side and saves image."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    clean_img_np = np.clip(clean_tensor.permute(1, 2, 0).numpy(), 0.0, 1.0)
    poisoned_img_np = np.clip(poisoned_tensor.permute(1, 2, 0).numpy(), 0.0, 1.0)

    # Clean Input Subplot
    axes[0].imshow(clean_img_np)
    axes[0].set_title(f"Clean Input Image\nPred: {clean_pred.upper()} ({clean_conf:.2f}%)", 
                      fontsize=12, fontweight='bold', color='green')
    axes[0].axis('off')

    # Triggered Input Subplot
    is_defended = (poisoned_pred.lower() != 'airplane')
    color = 'green' if is_defended else 'red'
    status = "Defended (Backdoor Mitigated)" if is_defended else "Backdoor Active (Misclassified)"

    axes[1].imshow(poisoned_img_np)
    axes[1].set_title(f"Triggered Input ({title_name})\nPred: {poisoned_pred.upper()} ({poisoned_conf:.2f}%)\n[{status}]", 
                      fontsize=12, fontweight='bold', color=color)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✔] Successfully generated figure: {save_filename}")


def main():
    # Target dog test sample matching terminal execution
    sample_path = "./data/cifar10/test/dog/08870.png"
    if not os.path.exists(sample_path):
        print(f"[!] Path {sample_path} not found. Searching for alternative dog sample...")
        for root, _, files in os.walk("./data/cifar10/test"):
            for f in files:
                if f.endswith('.png') and 'dog' in root:
                    sample_path = os.path.join(root, f)
                    break

    raw_img = Image.open(sample_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    clean_tensor = transform(raw_img)

    models_config = [
        {
            "name": "BadNets Model",
            "ckpt": "experiments/train_poisoned_DatasetFolder-CIFAR10_2026-10-31_22-01-43/defended_pruned.pth",
            "trigger": apply_badnets_trigger,
            "filename": "defended_train_poisoned_datasetfolder_cifar10_2026_10_31_22_01_43_demo.png"
        },
        {
            "name": "Blended Model",
            "ckpt": "experiments/train_poisoned_DatasetFolder-CIFAR10_Blended_2026-11-01_00-13-48/defended_pruned.pth",
            "trigger": apply_blended_trigger,
            "filename": "defended_train_poisoned_datasetfolder_cifar10_blended_2026_11_01_00_13_48_demo.png"
        },
        {
            "name": "LabelConsistent Model",
            "ckpt": "experiments/train_poisoned_DatasetFolder-CIFAR10_LabelConsistent_2026-11-01_14-22-35/defended_pruned.pth",
            "trigger": apply_badnets_trigger,
            "filename": "defended_train_poisoned_datasetfolder_cifar10_labelconsistent_2026_11_01_14_22_35_demo.png"
        }
    ]

    for cfg in models_config:
        if os.path.exists(cfg["ckpt"]):
            model = load_model(cfg["ckpt"])
            poisoned_tensor = cfg["trigger"](clean_tensor)
            
            clean_pred, clean_conf = run_inference(model, clean_tensor)
            poisoned_pred, poisoned_conf = run_inference(model, poisoned_tensor)
            
            save_visualization(
                clean_tensor, poisoned_tensor,
                clean_pred, clean_conf,
                poisoned_pred, poisoned_conf,
                cfg["name"], cfg["filename"]
            )
        else:
            print(f"[X] Checkpoint not found at: {cfg['ckpt']}. Skipping.")


if __name__ == '__main__':
    main()