'''
Inference Visualization for Defended ResNet-18 Models (Task 3)
Generates side-by-side PNG figures evaluating Clean vs. Triggered predictions.
'''

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import core

# CIFAR-10 Class Names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_model(checkpoint_path, device):
    """Loads defended ResNet-18 model weights from checkpoint."""
    model = core.models.ResNet(18)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model


# --- TRIGGER GENERATION FUNCTIONS ---

def apply_badnets_trigger(img_tensor):
    """Applies solid 3x3 white square patch to bottom-right corner."""
    poisoned = img_tensor.clone()
    poisoned[:, -3:, -3:] = 1.0
    return poisoned


def apply_blended_trigger(img_tensor, alpha=0.2):
    """Applies 20% alpha-blended 3x3 patch to bottom-right corner."""
    poisoned = img_tensor.clone()
    poisoned[:, -3:, -3:] = (1.0 - alpha) * img_tensor[:, -3:, -3:] + alpha * 1.0
    return poisoned


def run_inference(model, img_tensor, device):
    """Runs forward pass and returns top predicted class and confidence score."""
    input_batch = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_batch)
        probs = F.softmax(outputs, dim=1)[0]
        
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item() * 100.0
    return CLASSES[pred_idx], confidence, probs.cpu().numpy()


def tensor_to_np(tensor):
    """Converts (C, H, W) PyTorch Tensor to (H, W, C) NumPy array for Matplotlib."""
    np_img = tensor.permute(1, 2, 0).numpy()
    return np.clip(np_img, 0.0, 1.0)


def visualize_and_save(clean_tensor, poisoned_tensor, clean_pred, clean_conf, 
                       poisoned_pred, poisoned_conf, title_prefix, save_path):
    """Plots and saves side-by-side inference figure."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # Clean Image Plot
    axes[0].imshow(tensor_to_np(clean_tensor))
    axes[0].set_title(f"Clean Input Image\nPred: {clean_pred.upper()} ({clean_conf:.1f}%)", 
                      fontsize=12, fontweight='bold', color='green')
    axes[0].axis('off')

    # Triggered Image Plot (Color green if defense succeeded, red if backdoor still active)
    is_attack_neutralized = (poisoned_pred.lower() != 'airplane')
    title_color = 'green' if is_attack_neutralized else 'red'
    status_label = "Defended" if is_attack_neutralized else "Backdoor Active"

    axes[1].imshow(tensor_to_np(poisoned_tensor))
    axes[1].set_title(f"Triggered Input ({title_prefix})\nPred: {poisoned_pred.upper()} ({poisoned_conf:.1f}%)\n[{status_label}]", 
                      fontsize=12, fontweight='bold', color=title_color)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[✔] Saved visualization figure: {save_path}")
    plt.close()


def resolve_defended_checkpoint(folder_keyword):
    """Locates defended_finetuned.pth file in the corresponding experiment folder."""
    exp_dir = "experiments"
    if os.path.exists(exp_dir):
        matching_dirs = [d for d in os.listdir(exp_dir) if folder_keyword in d]
        if matching_dirs:
            ckpt = os.path.join(exp_dir, matching_dirs[-1], "defended_finetuned.pth")
            if os.path.exists(ckpt):
                return ckpt
    return None


def process_model_visualization(model_name, keyword, trigger_fn, sample_tensor, device):
    ckpt_path = resolve_defended_checkpoint(keyword)
    
    print("\n" + "=" * 60)
    print(f"  VISUALIZING DEFENDED INFERENCE: {model_name}")
    print("=" * 60)

    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[X] Defended checkpoint not found for keyword '{keyword}'. Skipping.")
        return

    print(f"[✔] Loading checkpoint: {ckpt_path}")
    model = load_model(ckpt_path, device)
    
    poisoned_tensor = trigger_fn(sample_tensor)

    clean_pred, clean_conf, _ = run_inference(model, sample_tensor, device)
    poisoned_pred, poisoned_conf, _ = run_inference(model, poisoned_tensor, device)

    print(f"  Clean Input Pred:    {clean_pred.upper()} ({clean_conf:.2f}%)")
    print(f"  Triggered Input Pred: {poisoned_pred.upper()} ({poisoned_conf:.2f}%)")

    save_filename = f"defended_{keyword.lower().replace('-', '_')}_demo.png"
    visualize_and_save(sample_tensor, poisoned_tensor, clean_pred, clean_conf, 
                       poisoned_pred, poisoned_conf, title_prefix=model_name, save_path=save_filename)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sample Test Dog Image
    sample_image_path = "./data/cifar10/test/dog/08870.png"
    
    if not os.path.exists(sample_image_path):
        # Fallback search if path doesn't exist
        for root, _, files in os.walk("./data/cifar10/test"):
            for f in files:
                if f.endswith('.png') and 'dog' in root:
                    sample_image_path = os.path.join(root, f)
                    break

    print(f"[✔] Using test sample image: {sample_image_path}")

    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    raw_img = Image.open(sample_image_path).convert('RGB')
    sample_tensor = preprocess(raw_img)

    # 1. Defended BadNets Model
    process_model_visualization(
        model_name="BadNets Defended",
        keyword="train_poisoned_DatasetFolder-CIFAR10_2026-10-31_22-01-43",
        trigger_fn=apply_badnets_trigger,
        sample_tensor=sample_tensor,
        device=device
    )

    # 2. Defended Blended Model
    process_model_visualization(
        model_name="Blended Defended",
        keyword="train_poisoned_DatasetFolder-CIFAR10_Blended_2026-11-01_00-13-48",
        trigger_fn=apply_blended_trigger,
        sample_tensor=sample_tensor,
        device=device
    )

    # 3. Defended LabelConsistent Model
    process_model_visualization(
        model_name="LabelConsistent Defended",
        keyword="train_poisoned_DatasetFolder-CIFAR10_LabelConsistent_2026-11-01_14-22-35",
        trigger_fn=apply_badnets_trigger,
        sample_tensor=sample_tensor,
        device=device
    )


if __name__ == '__main__':
    main()