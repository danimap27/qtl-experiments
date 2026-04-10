"""Download and verify datasets for quantum transfer learning experiments.

Supported datasets:
- hymenoptera: Ants vs Bees (auto-download from PyTorch)
- brain_tumor: Brain Tumor MRI (manual download from Kaggle)
- cats_vs_dogs: Cats vs Dogs subset (manual download)
- solar_dust: Solar panel dust detection (manual download)

Usage:
    python data/download_datasets.py --dataset hymenoptera
    python data/download_datasets.py --dataset all
    python data/download_datasets.py --list
"""

import argparse
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# Dataset definitions
DATASETS = {
    "hymenoptera": {
        "name": "Hymenoptera (Ants vs Bees)",
        "url": "https://download.pytorch.org/tutorial/hymenoptera_data.zip",
        "auto_download": True,
        "description": "Ant vs Bee classification (~240 images)",
        "expected_structure": {
            "train": ["ants", "bees"],
            "test": ["ants", "bees"],
        },
    },
    "brain_tumor": {
        "name": "Brain Tumor MRI",
        "url": None,
        "auto_download": False,
        "description": "Brain tumor classification (MRI scans)",
        "expected_structure": {
            "train": ["glioma", "meningioma", "pituitary", "notumor"],
            "test": ["glioma", "meningioma", "pituitary", "notumor"],
        },
        "instructions": (
            "Download from Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset\n"
            "  1. Create a Kaggle account and API credentials\n"
            "  2. Run: kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset\n"
            "  3. Extract to: data/datasets/brain_tumor/\n"
            "  4. Expected structure: data/datasets/brain_tumor/Training/ and data/datasets/brain_tumor/Testing/"
        ),
    },
    "cats_vs_dogs": {
        "name": "Cats vs Dogs",
        "url": None,
        "auto_download": False,
        "description": "Cat vs Dog image classification",
        "expected_structure": {
            "train": ["cats", "dogs"],
            "val": ["cats", "dogs"],
        },
        "instructions": (
            "Download from Microsoft: https://www.microsoft.com/download/confirmation.aspx?id=54765\n"
            "  1. Download the dataset\n"
            "  2. Extract and organize into: data/datasets/cats_vs_dogs/train/ and data/datasets/cats_vs_dogs/val/\n"
            "  3. Each should contain 'cats' and 'dogs' subdirectories"
        ),
    },
    "solar_dust": {
        "name": "Solar Panel Dust Detection",
        "url": None,
        "auto_download": False,
        "description": "Solar panel dust/soiling detection",
        "expected_structure": {
            "train": ["clean", "dusty"],
            "val": ["clean", "dusty"],
        },
        "instructions": (
            "Download from Kaggle or provide dataset manually\n"
            "  1. Obtain the solar panel dust detection dataset\n"
            "  2. Extract to: data/datasets/solar_dust/\n"
            "  3. Organize as: data/datasets/solar_dust/train/ and data/datasets/solar_dust/val/\n"
            "  4. Each should contain 'clean' and 'dusty' subdirectories"
        ),
    },
}


def download_file(url: str, dest_path: Path) -> bool:
    """
    Download a file from a URL.

    Args:
        url: URL to download from
        dest_path: Destination file path

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {url}...")
        with urlopen(url) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 8192

            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    f.write(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        logger.info(
                            f"  Downloaded {downloaded / 1e6:.1f}MB "
                            f"/ {total_size / 1e6:.1f}MB ({percent:.1f}%)"
                        )

        logger.info(f"Successfully downloaded to {dest_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract a zip file.

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Successfully extracted to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False


def verify_dataset_structure(
    dataset_path: Path, expected_structure: Dict
) -> Tuple[bool, str]:
    """
    Verify dataset has the expected directory structure.

    Args:
        dataset_path: Path to dataset root
        expected_structure: Dict of {split: [classes]} expected structure

    Returns:
        Tuple of (is_valid, message)
    """
    if not dataset_path.exists():
        return False, f"Dataset path does not exist: {dataset_path}"

    missing = []
    for split, classes in expected_structure.items():
        split_path = dataset_path / split
        if not split_path.exists():
            # Try alternate names (Training -> train, Testing -> test)
            if split == "train" and (dataset_path / "Training").exists():
                split_path = dataset_path / "Training"
            elif split == "test" and (dataset_path / "Testing").exists():
                split_path = dataset_path / "Testing"
            else:
                missing.append(f"Missing split: {split}")
                continue

        found_classes = [d.name for d in split_path.iterdir() if d.is_dir()]
        expected_classes = set(classes)
        found_classes_set = set(found_classes)

        if not expected_classes.issubset(found_classes_set):
            missing_classes = expected_classes - found_classes_set
            missing.append(
                f"Missing classes in {split}: {missing_classes}"
            )

    if missing:
        return False, "; ".join(missing)

    return True, "Dataset structure verified"


def count_images(dataset_path: Path) -> Dict[str, int]:
    """
    Count images in each split and class.

    Args:
        dataset_path: Path to dataset root

    Returns:
        Dict mapping split -> class -> count
    """
    counts = {}

    for split_dir in dataset_path.iterdir():
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name
        counts[split_name] = {}

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            # Count image files
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
            image_count = sum(
                1
                for f in class_dir.iterdir()
                if f.suffix.lower() in image_extensions
            )

            counts[split_name][class_dir.name] = image_count

    return counts


def report_dataset_status(dataset_name: str, dataset_path: Path) -> None:
    """
    Print detailed status report for a dataset.

    Args:
        dataset_name: Name of dataset
        dataset_path: Path to dataset
    """
    dataset_info = DATASETS[dataset_name]
    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_info['name']}")
    print(f"Description: {dataset_info['description']}")
    print(f"{'=' * 70}")

    if not dataset_path.exists():
        print(f"Status: NOT DOWNLOADED")
        print(f"Path: {dataset_path}")
        if not dataset_info["auto_download"]:
            print(f"\nDownload instructions:\n{dataset_info['instructions']}")
        return

    # Verify structure
    is_valid, message = verify_dataset_structure(
        dataset_path, dataset_info["expected_structure"]
    )

    if not is_valid:
        print(f"Status: INCOMPLETE (Structure invalid)")
        print(f"Issue: {message}")
        return

    print(f"Status: DOWNLOADED AND VERIFIED")

    # Count images
    counts = count_images(dataset_path)
    total_images = 0

    for split, class_counts in sorted(counts.items()):
        split_total = sum(class_counts.values())
        total_images += split_total
        print(f"\n{split.upper()}:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} images")
        print(f"  Total: {split_total} images")

    print(f"\nTotal images: {total_images}")


def download_hymenoptera(dataset_path: Path) -> bool:
    """
    Download and extract hymenoptera dataset.

    Args:
        dataset_path: Path to extract dataset to

    Returns:
        True if successful, False otherwise
    """
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (dataset_path / "train").exists():
        logger.info("Hymenoptera dataset already downloaded")
        return True

    url = DATASETS["hymenoptera"]["url"]
    zip_path = dataset_path / "hymenoptera_data.zip"

    # Download
    if not download_file(url, zip_path):
        return False

    # Extract
    if not extract_zip(zip_path, dataset_path):
        return False

    # The zip contains a 'hymenoptera_data' subdirectory, move contents up
    extracted_dir = dataset_path / "hymenoptera_data"
    if extracted_dir.exists():
        for item in extracted_dir.iterdir():
            shutil.move(str(item), str(dataset_path / item.name))
        extracted_dir.rmdir()

    # Clean up zip
    zip_path.unlink()

    # Verify
    is_valid, message = verify_dataset_structure(
        dataset_path, DATASETS["hymenoptera"]["expected_structure"]
    )
    if is_valid:
        logger.info(f"Hymenoptera dataset verified: {message}")
        return True
    else:
        logger.error(f"Hymenoptera dataset verification failed: {message}")
        return False


def download_dataset(dataset_name: str, dataset_root: Path) -> bool:
    """
    Download a dataset.

    Args:
        dataset_name: Name of dataset (key in DATASETS)
        dataset_root: Root path for datasets

    Returns:
        True if successful, False otherwise
    """
    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False

    dataset_info = DATASETS[dataset_name]
    dataset_path = dataset_root / dataset_name

    logger.info(f"Downloading {dataset_info['name']}...")

    if not dataset_info["auto_download"]:
        print(f"\n{dataset_info['name']} requires manual download.")
        print(f"\n{dataset_info['instructions']}\n")
        return False

    if dataset_name == "hymenoptera":
        return download_hymenoptera(dataset_path)

    logger.error(f"No download implementation for {dataset_name}")
    return False


def list_datasets(dataset_root: Path) -> None:
    """
    List all available datasets and their status.

    Args:
        dataset_root: Root path for datasets
    """
    print("\n" + "=" * 70)
    print("AVAILABLE DATASETS")
    print("=" * 70 + "\n")

    for dataset_name in sorted(DATASETS.keys()):
        dataset_path = dataset_root / dataset_name
        report_dataset_status(dataset_name, dataset_path)

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and verify datasets for quantum transfer learning"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to download (or 'all' for all auto-downloadable datasets)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and their status",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/datasets"),
        help="Root directory for datasets (default: data/datasets)",
    )

    args = parser.parse_args()

    # Create dataset root if needed
    args.dataset_root.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_datasets(args.dataset_root)
        return

    if not args.dataset:
        parser.print_help()
        return

    if args.dataset.lower() == "all":
        # Download all auto-downloadable datasets
        datasets_to_download = [
            name
            for name, info in DATASETS.items()
            if info["auto_download"]
        ]
    else:
        datasets_to_download = [args.dataset.lower()]

    results = {}
    for dataset_name in datasets_to_download:
        if dataset_name not in DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            results[dataset_name] = False
            continue

        success = download_dataset(
            dataset_name, args.dataset_root
        )
        results[dataset_name] = success
        report_dataset_status(dataset_name, args.dataset_root / dataset_name)

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    for dataset_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{dataset_name}: {status}")


if __name__ == "__main__":
    main()
