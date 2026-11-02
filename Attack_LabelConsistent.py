'''
This is the example code of clean-label poisoned training on torchvision.datasets.DatasetFolder.
Dataset is CIFAR-10.
Attack method is LabelConsistent.
'''

import os
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip

import core
from core.attacks.LabelConsistent import LabelConsistent

global_seed = 666
deterministic = True
torch.manual_seed(global_seed)


def main():
    schedule_poisoned = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': '0',
        'GPU_num': 1,

        'benign_training': False,
        'batch_size': 256,
        'num_workers': 4,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [30, 40],

        'epochs': 50,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'train_poisoned_DatasetFolder-CIFAR10_LabelConsistent'
    }

    dataset = torchvision.datasets.DatasetFolder

    transform_train = Compose([
        ToTensor(),
        RandomHorizontalFlip()
    ])

    trainset = dataset(
        root='./data/cifar10/train',
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None
    )

    transform_test = Compose([
        ToTensor()
    ])

    testset = dataset(
        root='./data/cifar10/test',
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None
    )

    pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
    pattern[0, -3:, -3:] = 255
    weight = torch.zeros((1, 32, 32), dtype=torch.float32)
    weight[0, -3:, -3:] = 1.0

    adv_model = core.models.ResNet(18) 

    benign_ckpt = "experiments/train_benign_DatasetFolder-CIFAR10/ckpt_epoch_50.pth"
    if os.path.exists(benign_ckpt):
        ckpt = torch.load(benign_ckpt, map_location='cpu')
        adv_model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
        print(f"[✔] Loaded benign weights into adv_model from {benign_ckpt}")

    label_consistent = LabelConsistent(
        train_dataset=trainset,
        test_dataset=testset,
        model=core.models.ResNet(18),
        loss=nn.CrossEntropyLoss(),
        adv_model=adv_model,
        adv_dataset_dir='adv_dataset',
        pattern=pattern,
        weight=weight,
        y_target=0,           # Target class: airplane (0)
        poisoned_rate=0.5,    # 50% poisoning rate
        eps=16/255,            # PGD L-inf bound
        steps=40,             # Fast execution
        seed=global_seed,
        deterministic=deterministic,
        schedule=schedule_poisoned
    )

    label_consistent.train(schedule_poisoned)


if __name__ == '__main__':
    main()