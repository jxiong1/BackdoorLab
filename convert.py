import os
import sys
import pickle
import tarfile
import numpy as np
from PIL import Image

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def convert_tar(tar_path, output_dir="data/cifar10"):
    extract_dir = "temp_extracted_cifar"
    
    print(f"Unpacking {tar_path}...")
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(path=extract_dir)
        
    cifar_folder = None
    for root, _, files in os.walk(extract_dir):
        if 'data_batch_1' in files:
            cifar_folder = root
            break
            
    if not cifar_folder:
        raise FileNotFoundError("Could not locate CIFAR-10 data batches inside the archive.")

    # Process Training Batches (data_batch_1 to 5)
    print("Converting Training set...")
    train_count = 0
    for i in range(1, 6):
        batch = unpickle(os.path.join(cifar_folder, f'data_batch_{i}'))
        for img_flat, label in zip(batch[b'data'], batch[b'labels']):
            img_arr = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)
            img = Image.fromarray(img_arr)
            class_dir = os.path.join(output_dir, "train", CLASSES[label])
            os.makedirs(class_dir, exist_ok=True)
            img.save(os.path.join(class_dir, f"{train_count:05d}.png"))
            train_count += 1

    # Process Test Batch (test_batch)
    print("Converting Test set...")
    test_count = 0
    batch = unpickle(os.path.join(cifar_folder, 'test_batch'))
    for img_flat, label in zip(batch[b'data'], batch[b'labels']):
        img_arr = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img_arr)
        class_dir = os.path.join(output_dir, "test", CLASSES[label])
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f"{test_count:05d}.png"))
        test_count += 1

    print(f"\nSuccessfully extracted {train_count} train images and {test_count} test images into '{output_dir}'.")

if __name__ == "__main__":
    tar_filename = sys.argv[1] if len(sys.argv) > 1 else "cifar-10-python.tar.gz"
    convert_tar(tar_filename)