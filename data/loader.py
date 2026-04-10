"""Dataset loading utilities for quantum transfer learning experiments."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_config: Dict,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load image dataset with standard ImageNet preprocessing.

    Handles two directory structures:
    1. dataset_path/train/ and dataset_path/val/ (or test/) — use as-is
    2. dataset_path/train/ only — split train 80/20 into train/val

    For training data, applies RandomResizedCrop and RandomHorizontalFlip.
    All data receives standard ImageNet normalization.

    Args:
        dataset_config: Dictionary with keys:
            - 'name': str, dataset name
            - 'path': str or Path, root directory of dataset
            - 'num_classes': int, number of classes
            - 'image_size': int, target image size (default 224)
        batch_size: int, batch size for DataLoaders (default 16)
        seed: int, random seed for deterministic splitting (default 42)
        num_workers: int, number of workers for DataLoaders (default 0)

    Returns:
        Tuple of (train_loader, val_loader, test_loader) as DataLoader objects

    Raises:
        FileNotFoundError: if dataset directory structure is invalid
        ValueError: if dataset_config is missing required keys
    """
    # Validate config
    required_keys = {"name", "path", "num_classes", "image_size"}
    if not required_keys.issubset(dataset_config.keys()):
        raise ValueError(
            f"dataset_config must contain {required_keys}. "
            f"Got: {set(dataset_config.keys())}"
        )

    dataset_name = dataset_config["name"]
    dataset_path = Path(dataset_config["path"])
    image_size = dataset_config.get("image_size", 224)

    # Check dataset path exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    logger.info(f"Loading dataset '{dataset_name}' from {dataset_path}")
    logger.info(f"Image size: {image_size}, Batch size: {batch_size}")

    # Define transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.143)),  # 256 for 224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Check directory structure
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    test_dir = dataset_path / "test"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training data directory not found: {train_dir}. "
            f"Expected structure: {dataset_path}/train/, "
            f"optionally with {dataset_path}/val/ and/or {dataset_path}/test/"
        )

    # Load train dataset
    train_dataset = ImageFolder(str(train_dir), transform=train_transform)
    logger.info(f"Loaded train dataset: {len(train_dataset)} images")

    # Determine validation and test datasets
    if val_dir.exists() and test_dir.exists():
        # Both val and test exist separately
        val_dataset = ImageFolder(str(val_dir), transform=val_test_transform)
        test_dataset = ImageFolder(str(test_dir), transform=val_test_transform)
        logger.info(f"Loaded val dataset: {len(val_dataset)} images")
        logger.info(f"Loaded test dataset: {len(test_dataset)} images")

    elif val_dir.exists():
        # Only val exists — use as both val and test
        val_dataset = ImageFolder(str(val_dir), transform=val_test_transform)
        test_dataset = ImageFolder(str(val_dir), transform=val_test_transform)
        logger.info(f"Loaded val/test dataset: {len(val_dataset)} images")

    elif test_dir.exists():
        # Only test exists — use as test, split train for val
        test_dataset = ImageFolder(str(test_dir), transform=val_test_transform)
        logger.info(f"Loaded test dataset: {len(test_dataset)} images")

        # Split train into train/val
        train_dataset, val_dataset = _split_dataset(
            train_dataset, val_test_transform, seed=seed
        )
        logger.info(
            f"Split train dataset: {len(train_dataset)} train, "
            f"{len(val_dataset)} val"
        )

    else:
        # Only train exists — split train into train/val
        train_dataset, val_dataset = _split_dataset(
            train_dataset, val_test_transform, seed=seed
        )
        test_dataset = val_dataset  # Use val as test
        logger.info(
            f"Split train dataset: {len(train_dataset)} train, "
            f"{len(val_dataset)} val/test"
        )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders created: {len(train_loader)} train batches, "
        f"{len(val_loader)} val batches, {len(test_loader)} test batches"
    )

    return train_loader, val_loader, test_loader


def _split_dataset(
    dataset: ImageFolder,
    transform,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> Tuple[ImageFolder, ImageFolder]:
    """
    Split an ImageFolder dataset into train and validation sets.

    Uses deterministic splitting with a torch.Generator seeded by the seed.

    Args:
        dataset: ImageFolder dataset to split
        transform: Transform to apply to the validation dataset
        seed: Random seed for deterministic splitting
        train_ratio: Fraction of data to use for training (default 0.8)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Apply validation transform to the validation subset
    # Note: We need to wrap the indices to apply the different transform
    val_dataset = _TransformWrapper(dataset, val_dataset.indices, transform)

    return train_dataset, val_dataset


class _TransformWrapper:
    """Wrapper to apply a different transform to a subset of a dataset."""

    def __init__(self, base_dataset, indices, transform):
        """
        Args:
            base_dataset: The original dataset
            indices: Indices to include in this subset
            transform: Transform to apply
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.base_dataset[actual_idx]

        # If the base dataset already applied transforms, we need the raw image
        # This is a limitation of torchvision.ImageFolder, so we reload from file
        image_path = self.base_dataset.imgs[actual_idx][0]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def classes(self):
        return self.base_dataset.classes

    @property
    def class_to_idx(self):
        return self.base_dataset.class_to_idx


