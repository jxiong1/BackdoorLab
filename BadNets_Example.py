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
    """Loads trained ResNet-18 model weights from checkpoint."""
    model = core.models.ResNet(18)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model


def apply_badnets_trigger(img_tensor):
    """Applies a 3x3 white square trigger to the bottom-right corner."""
    poisoned_tensor = img_tensor.clone()
    # Replace bottom-right 3x3 region with 1.0 (white in normalized [0, 1] range)
    poisoned_tensor[:, -3:, -3:] = 1.0
    return poisoned_tensor


def run_inference(model, img_tensor, device):
    """Runs forward pass and returns top prediction class name and confidence score."""
    input_batch = img_tensor.unsqueeze(0).to(device)  # Add batch dimension (1, C, H, W)
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
                       poisoned_pred, poisoned_conf, save_path="badnets_inference_demo.png"):
    """Plots and saves side-by-side inference figure."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # Plot Clean Image
    axes[0].imshow(tensor_to_np(clean_tensor))
    axes[0].set_title(f"Clean Input Image\nPred: {clean_pred.upper()} ({clean_conf:.1f}%)", 
                      fontsize=12, fontweight='bold', color='green')
    axes[0].axis('off')

    # Plot Poisoned Image
    axes[1].imshow(tensor_to_np(poisoned_tensor))
    axes[1].set_title(f"Triggered (BadNets) Image\nPred: {poisoned_pred.upper()} ({poisoned_conf:.1f}%)", 
                      fontsize=12, fontweight='bold', color='red')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[✔] Visualization saved successfully to: {save_path}")
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Path to saved model checkpoint
    model_path = "experiments/train_poisoned_DatasetFolder-CIFAR10_2026-10-31_22-01-43/ckpt_epoch_50.pth"
    
    # Sample image path (e.g., a test dog image)
    # You can substitute this with any custom PNG/JPG image path
    sample_image_path = "./data/cifar10/test/dog/00012.png"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}. Make sure training finished.")

    # Load image and preprocess to 32x32 tensor
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    raw_img = Image.open(sample_image_path).convert('RGB')
    clean_tensor = preprocess(raw_img)
    poisoned_tensor = apply_badnets_trigger(clean_tensor)

    # Load trained model
    model = load_model(model_path, device)

    # Run predictions
    clean_pred, clean_conf, _ = run_inference(model, clean_tensor, device)
    poisoned_pred, poisoned_conf, _ = run_inference(model, poisoned_tensor, device)

    print(f"Clean Prediction:    {clean_pred} ({clean_conf:.2f}%)")
    print(f"Poisoned Prediction: {poisoned_pred} ({poisoned_conf:.2f}%)")

    # Generate output figure
    visualize_and_save(clean_tensor, poisoned_tensor, clean_pred, clean_conf, 
                       poisoned_pred, poisoned_conf, save_path="badnets_inference_demo.png")


if __name__ == '__main__':
    main()