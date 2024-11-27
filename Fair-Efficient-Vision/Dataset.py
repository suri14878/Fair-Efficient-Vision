import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torchvision import transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

class CombinedNPZDataset(Dataset):
    def __init__(self, root_dirs, phase='train', transform=None):
        """
        Args:
            root_dirs (list of strings): List of two directories containing .npz files.
            phase (string): 'train', 'val', or 'test' to specify which folder to read from.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transform = transform
        self.file_list = []

        # Combine file paths from both directories for the specified phase (train/val/test)
        for root_dir in root_dirs:
            phase_dir = os.path.join(root_dir, phase)  # e.g., root_dir/train or root_dir/val or root_dir/test
            if os.path.exists(phase_dir):
                self.file_list += [os.path.join(phase_dir, f) for f in os.listdir(phase_dir) if f.endswith('.npz')]

    def __len__(self):
        """Return the number of samples in the combined dataset."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """Load an .npz file and return a tuple (image, label)."""
        npz_path = self.file_list[idx]
        
        # Load .npz file
        data = np.load(npz_path)
        
        if 'glaucoma' in data.files:
            image = data['slo_fundus']
            label = data['glaucoma']
        else:
            image = data['slo_fundus']
            if data['dr_class'] == 0:
                label = 0
            else:
                label = 2
        
        # Convert numpy array to PIL Image
        if len(image.shape) == 2:  # If grayscale (2D array)
            image = Image.fromarray(image.astype(np.uint8), 'L')
            image = image.convert('RGB')  # Convert grayscale to RGB
        elif len(image.shape) == 3:  # If RGB (3D array)
            image = Image.fromarray(image.astype(np.uint8), 'RGB')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    
# ============================
# 2. Build Dataset and DataLoader with Sequential Sampler
# ============================
def build_combined_dataset(root_dirs, input_size,phase = ''):
    """
    Build combined dataset from two directories and apply transformations.

    Args:
      root_dirs: List of two directory paths.
      phase: 'train', 'val', or 'test' indicating which dataset partition to load.
      args: Arguments containing input size and other parameters.

    Returns:
      dataset: CombinedNPZDataset object.
    """
    transform = build_transform(phase,input_size)
#     root_dirs = [os.path.join(path, phase) for path in root_dirs]
    # Create combined dataset using CombinedNPZDataset class
    dataset = CombinedNPZDataset(root_dirs=root_dirs, phase=phase, transform=transform)
    
    return dataset


def build_transform(phase, input_size):
    """
    Build transformation pipeline for training or evaluation.

    Args:
      phase: 'train', 'val', or 'test' indicating which transformation pipeline to use.
      args: Arguments containing input size and other parameters.

    Returns:
      Transformations composed for the dataset.
    """
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    
    if phase=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            interpolation='bicubic',
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
            mean=mean,
            std=std,
        )
    else:
        # Evaluation transformations (validation/test)
        t = []
        
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        
        size = int(input_size / crop_pct)
        
        t.append(transforms.Resize(size))
        t.append(transforms.CenterCrop(input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean=mean, std=std))
        
        transform = transforms.Compose(t)

    return transform