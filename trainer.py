"""
Training module for quantum transfer learning experiments.

Handles single training runs combining dataset, backbone, head, and seed.
Supports multiple study types: main, ablation, noise_decomposition, sim_as_hardware.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import torchvision.models as models

from data import load_dataset
from heads import get_head, count_trainable_params

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_backbone(backbone_name: str) -> nn.Module:
    """
    Load a pretrained backbone from torchvision and freeze all parameters.

    Supported backbones:
    - resnet18
    - mobilenetv2
    - efficientnet_b0
    - regnet_x_400mf

    Args:
        backbone_name: Name of the backbone architecture

    Returns:
        Frozen pretrained backbone model with identity classification head

    Raises:
        ValueError: If backbone_name is not supported
    """
    if backbone_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
    elif backbone_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = nn.Identity()
    elif backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Identity()
    elif backbone_name == "regnet_x_400mf":
        model = models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.DEFAULT)
        model.fc = nn.Identity()
    else:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            f"Supported: resnet18, mobilenetv2, efficientnet_b0, regnet_x_400mf"
        )

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    logger.info(f"Loaded and froze backbone: {backbone_name}")
    return model


def determine_environment(head_config: Dict[str, Any]) -> str:
    """
    Determine the execution environment based on head configuration.

    Args:
        head_config: Head configuration dictionary

    Returns:
        Environment type: "simulation", "emulation", or "qpu"
    """
    head_type = head_config.get("type", "")
    if head_type == "classical":
        return "simulation"

    noise = head_config.get("noise", False)
    if noise:
        return "emulation"

    return "simulation"


def train_and_evaluate(
    run_config: Dict[str, Any],
    config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Execute a single training run.

    Performs a complete training pipeline: load data, instantiate models, train,
    and evaluate on test set. Collects detailed metrics and per-sample predictions.

    Args:
        run_config: Configuration for this specific run with keys:
            - run_id (str): Unique run identifier (e.g., "hymenoptera_resnet18_pl_ideal_42")
            - dataset (str): Dataset name
            - backbone (str): Backbone architecture name
            - head (str): Head module name
            - seed (int): Random seed for reproducibility
            - study (str): Study type ("main", "ablation", "noise_decomposition", "sim_as_hardware")
        config: Full parsed configuration dictionary (from config.yaml)
        overrides: Optional parameter overrides for studies:
            - n_qubits (int): Number of qubits (ablation)
            - depth (int): Circuit depth (ablation)
            - epochs (int): Number of training epochs (sim_as_hardware)
            - shots (int): Measurement shots (sim_as_hardware)
            - noise_channels (list): Noise channel types (noise_decomposition)

    Returns:
        Tuple of (result_dict, predictions_list, training_log_list) where:
        - result_dict: Single row for runs.csv with all aggregated metrics
        - predictions_list: Per-sample predictions for predictions.csv
        - training_log_list: Per-epoch training metrics for training_log.csv
        Returns None on error (logged to errors.log)
    """
    # Support both dict and dataclass-style access
    if hasattr(run_config, "run_id"):
        _rc = {
            "run_id": run_config.run_id,
            "dataset": run_config.dataset,
            "backbone": run_config.backbone,
            "head": run_config.head,
            "seed": run_config.seed,
            "study": getattr(run_config, "study", "main"),
        }
    else:
        _rc = run_config

    run_id = _rc.get("run_id", "unknown")
    logger.info(f"Starting training run: {run_id}")

    try:
        # Step 1: Set seed for reproducibility
        seed = _rc["seed"]
        set_seed(seed)
        logger.info(f"Set random seed: {seed}")

        # Step 2: Load dataset
        dataset_name = _rc["dataset"]
        backbone_name = _rc["backbone"]
        head_name = _rc["head"]

        dataset_config = next(
            (d for d in config["datasets"] if d["name"] == dataset_name),
            None,
        )
        if not dataset_config:
            raise ValueError(f"Dataset configuration not found: {dataset_name}")

        training_config = config.get("training", {})
        batch_size = training_config.get("batch_size", 32)

        train_loader, val_loader, test_loader = load_dataset(
            dataset_config, batch_size=batch_size, seed=seed
        )
        logger.info(
            f"Loaded dataset: {dataset_name} "
            f"(train={len(train_loader.dataset)}, "
            f"val={len(val_loader.dataset)}, "
            f"test={len(test_loader.dataset)})"
        )

        # Infer feature dimension; get num_classes from config
        num_classes = dataset_config.get("num_classes", 2)
        sample_batch = next(iter(train_loader))
        sample_images = sample_batch[0]

        # Step 3: Load backbone
        backbone = load_backbone(backbone_name)
        with torch.no_grad():
            sample_features = backbone(sample_images[:1])
            feature_dim = sample_features.view(1, -1).shape[1]
        logger.info(f"Feature dimension inferred: {feature_dim}")

        # Step 4: Instantiate head
        head_config = next(
            (h for h in config["heads"] if h["name"] == head_name),
            None,
        )
        if not head_config:
            raise ValueError(f"Head configuration not found: {head_name}")

        head = get_head(head_config, feature_dim, num_classes, overrides=overrides)
        n_trainable_params = count_trainable_params(head)
        logger.info(f"Head instantiated with {n_trainable_params} trainable parameters")

        # Step 5: Configure optimizer and scheduler
        epochs = (
            overrides.get("epochs", training_config.get("epochs", 50))
            if overrides
            else training_config.get("epochs", 50)
        )
        lr = training_config.get("lr", 0.001)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr)

        scheduler_config = training_config.get("scheduler", {})
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 10),
            gamma=scheduler_config.get("gamma", 0.5),
        )
        criterion = nn.CrossEntropyLoss()
        logger.info(f"Optimizer configured: Adam (lr={lr}), epochs={epochs}")

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = backbone.to(device)
        head = head.to(device)
        logger.info(f"Using device: {device}")

        # Step 6: Start energy tracker
        energy_kwh = None
        tracker = None
        if config.get("energy", {}).get("enabled", True):
            try:
                from codecarbon import EmissionsTracker

                tracker = EmissionsTracker(
                    project_name=run_id,
                    output_dir=config.get("output_dir", "./results"),
                    log_level="error",
                )
                tracker.start()
                logger.info("CodeCarbon tracker started")
            except Exception as e:
                logger.warning(f"Failed to start CodeCarbon tracker: {e}")
                tracker = None

        # Step 7: Training loop
        training_log = []
        total_start = time.perf_counter()

        for epoch in range(epochs):
            epoch_start = time.perf_counter()

            # Train phase
            head.train()
            train_loss = 0.0
            train_samples = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    features = backbone(images)

                logits = head(features)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_samples += images.size(0)

            train_loss /= train_samples

            # Validation phase
            head.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    features = backbone(images)
                    logits = head(features)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * images.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_samples += images.size(0)

            val_loss /= val_samples
            val_accuracy = val_correct / val_samples

            epoch_time = time.perf_counter() - epoch_start
            scheduler.step()

            training_log.append(
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "epoch_time_s": epoch_time,
                }
            )

            if (epoch + 1) % max(1, epochs // 10) == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_acc={val_accuracy:.4f}"
                )

        train_time_s = time.perf_counter() - total_start
        logger.info(f"Training completed in {train_time_s:.2f}s")

        # Step 8: Stop energy tracker
        if tracker:
            try:
                energy_kwh = tracker.stop()
                logger.info(f"Energy consumption: {energy_kwh} kWh")
            except Exception as e:
                logger.warning(f"Failed to stop CodeCarbon tracker: {e}")
                energy_kwh = None

        # Step 9: Evaluate on test set
        predictions = []
        head.eval()
        with torch.no_grad():
            sample_idx = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                features = backbone(images)
                logits = head(features)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                for i in range(images.size(0)):
                    pred_dict = {
                        "run_id": run_id,
                        "sample_idx": sample_idx,
                        "y_true": labels[i].item(),
                        "y_pred": preds[i].item(),
                    }
                    # Collect probabilities for all classes
                    for cls in range(num_classes):
                        pred_dict[f"y_prob_{cls}"] = probs[i, cls].item()
                    predictions.append(pred_dict)
                    sample_idx += 1

        logger.info(f"Evaluated {len(predictions)} test samples")

        # Step 10: Compute aggregate metrics
        y_true = [p["y_true"] for p in predictions]
        y_pred = [p["y_pred"] for p in predictions]
        y_prob_1 = [p.get("y_prob_1", 0) for p in predictions]

        metrics = {
            "test_accuracy": accuracy_score(y_true, y_pred),
            "test_precision": precision_score(y_true, y_pred, zero_division=0),
            "test_recall": recall_score(y_true, y_pred, zero_division=0),
            "test_f1": f1_score(y_true, y_pred, zero_division=0),
            "test_auc": roc_auc_score(y_true, y_prob_1)
            if len(set(y_true)) > 1
            else 0.0,
        }
        logger.info(
            f"Test metrics - "
            f"acc={metrics['test_accuracy']:.4f}, "
            f"f1={metrics['test_f1']:.4f}, "
            f"auc={metrics['test_auc']:.4f}"
        )

        # Step 11: Build result dictionary
        head_config_resolved = head_config.copy()
        if overrides:
            head_config_resolved.update(overrides)

        result = {
            "run_id": run_id,
            "study": _rc.get("study", "main"),
            "seed": seed,
            "dataset": dataset_name,
            "backbone": backbone_name,
            "head": head_name,
            "head_type": head_config.get("type", ""),
            "environment": determine_environment(head_config),
            "n_qubits": head_config_resolved.get("n_qubits"),
            "depth": head_config_resolved.get("depth"),
            "shots": head_config_resolved.get("shots"),
            "noise_channels": (
                ",".join(head_config_resolved.get("noise_channels", []))
                or None
            ),
            "n_trainable_params": n_trainable_params,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "train_time_s": train_time_s,
            "energy_kwh": energy_kwh,
            **metrics,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Run {run_id} completed successfully")
        return (result, predictions, training_log)

    except Exception as e:
        logger.error(f"Error in training run {run_id}: {e}", exc_info=True)
        return None
