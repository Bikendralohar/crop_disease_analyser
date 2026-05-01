"""
dataset.py
----------
Handles all data loading, augmentation, and splitting for the
Plant Disease Classification dataset.
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_transforms(train=True):
    """
    Returns image transforms for training or validation/testing.
    Training transforms include data augmentation to reduce overfitting.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                                 [0.229, 0.224, 0.225])   # ImageNet std
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


def get_loaders(data_dir, batch_size=32):
    """
    Loads the dataset from data_dir (ImageFolder format),
    splits into 80/10/10 train/val/test, returns DataLoaders.

    Args:
        data_dir   : path to folder containing one subfolder per class
        batch_size : number of images per batch

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Load full dataset with training transforms first
    full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(train=True))

    # Calculate split sizes
    n        = len(full_dataset)
    train_n  = int(0.8 * n)
    val_n    = int(0.1 * n)
    test_n   = n - train_n - val_n

    # Split dataset
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_n, val_n, test_n])

    # Override transforms for val/test (no augmentation)
    val_ds.dataset.transform  = get_transforms(train=False)
    test_ds.dataset.transform = get_transforms(train=False)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"[Dataset] Total images : {n}")
    print(f"[Dataset] Train        : {train_n}")
    print(f"[Dataset] Val          : {val_n}")
    print(f"[Dataset] Test         : {test_n}")
    print(f"[Dataset] Classes      : {len(full_dataset.classes)}")

    return train_loader, val_loader, test_loader, full_dataset.classes
