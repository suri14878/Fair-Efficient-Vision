# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image

# def build_dataset(is_train, args):
    
#     transform = build_transform(is_train, args)
#     root = os.path.join(args.data_path, is_train)
#     dataset = datasets.ImageFolder(root, transform=transform)

#     return dataset

import numpy as np
import torch
from torch.utils.data import Dataset

class CustomNPZDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the .npz files.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npz')]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """Load an .npz file and return a tuple (image, label)."""
        npz_path = os.path.join(self.root_dir, self.file_list[idx])
        
        # Load .npz file
        data = np.load(npz_path)
        
        # Assuming the .npz file has keys 'image' for image data and 'label' for labels
        image = data['slo_fundus']  # Load image data (numpy array)
        label = data['glaucoma']  # Load label (e.g., class '0' or '1')
        
        # Convert numpy array to PIL Image
        # If image is grayscale (2D array)
        if len(image.shape) == 2:
            image = Image.fromarray(image.astype(np.uint8), 'L')
            image = image.convert('RGB')
        # If image is RGB (3D array)
        elif len(image.shape) == 3:
            image = Image.fromarray(image.astype(np.uint8), 'RGB')
        
        # Convert image to a PyTorch tensor and apply transformations if provided
        if self.transform:
            image = self.transform(image)
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label

def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
#     dataset = datasets.ImageFolder(root, transform=transform)
    dataset = CustomNPZDataset(root_dir=root, transform=transform)
    return dataset
    
def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
