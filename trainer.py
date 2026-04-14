"""
Training module for quantum transfer learning experiments.

New in this version:
  - Epoch-level checkpoints (best + last) with resume support
  - Rich tqdm progress bars: per-batch loss/acc + epoch ETA + overall progress
  - Live console summary printed after every epoch
  - Full visualization pipeline via visualization.py (learning curves,
    confusion matrix, ROC, PR, per-class metrics, summary panel)
  - Clustering / unsupervised task support (heads with task_type='clustering')
  - Non-image (tabular / numpy) dataset support via data.tabular_loader
"""

import logging
import os
import time
from datetime import datetime, timedelta
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

# tqdm is optional but highly recommended
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)


# ── Progress display ──────────────────────────────────────────────────────────

class EpochBar:
    """
    Wrapper around tqdm (or a plain logger fallback) for per-batch progress.
    Shows: batch index, loss, accuracy, LR, ETA for the current epoch.
    """

    def __init__(self, total: int, epoch: int, n_epochs: int, run_id: str):
        self.total = total
        self.epoch = epoch
        self.n_epochs = n_epochs
        self._bar = None
        if HAS_TQDM:
            self._bar = _tqdm(
                total=total,
                desc=f"  Epoch {epoch+1:>3}/{n_epochs}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
                bar_format=(
                    "{desc} |{bar:30}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
            )

    def update(self, loss: float, acc: float, lr: float):
        if self._bar:
            self._bar.set_postfix(
                loss=f"{loss:.4f}", acc=f"{acc*100:.1f}%", lr=f"{lr:.2e}"
            )
            self._bar.update(1)

    def close(self):
        if self._bar:
            self._bar.close()


def _fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _print_epoch_summary(
    run_id: str, epoch: int, n_epochs: int,
    train_loss: float, val_loss: float,
    train_acc: float, val_acc: float,
    lr: float, epoch_time: float,
    best_val_acc: float,
    checkpoint_saved: bool,
):
    """Print a formatted one-line epoch summary to stdout."""
    ckpt_flag = " [CKPT✓]" if checkpoint_saved else ""
    bar_len = 20
    filled = int(bar_len * (epoch + 1) / n_epochs)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = 100 * (epoch + 1) / n_epochs

    print(
        f"\r  [{bar}] {pct:5.1f}%  "
        f"Epoch {epoch+1}/{n_epochs}  "
        f"| train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"| train_acc={train_acc*100:.1f}%  val_acc={val_acc*100:.1f}%  "
        f"(best={best_val_acc*100:.1f}%)  "
        f"| lr={lr:.2e}  time={_fmt_time(epoch_time)}{ckpt_flag}",
        flush=True,
    )


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _checkpoint_dir(output_dir: str, run_id: str) -> str:
    d = os.path.join(output_dir, "checkpoints", run_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_checkpoint(
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    run_id: str,
    output_dir: str,
    tag: str = "last",
    history: Optional[Dict[str, List[float]]] = None,
    training_log: Optional[List[Dict]] = None,
):
    """
    Save a training checkpoint.

    Args:
        head: The head module (only trainable component).
        optimizer: Optimizer state.
        epoch: Current epoch index (0-based).
        val_acc: Validation accuracy at this epoch.
        run_id: Experiment identifier.
        output_dir: Root results directory.
        tag: 'last', 'best', or 'epochN'.
        history: Current training history dict.
        training_log: Current training log list.
    """
    ckpt = {
        "epoch": epoch,
        "val_acc": val_acc,
        "run_id": run_id,
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "training_log": training_log,
        "timestamp": datetime.now().isoformat(),
    }
    path = os.path.join(_checkpoint_dir(output_dir, run_id), f"ckpt_{tag}.pt")
    torch.save(ckpt, path)
    return path


def load_checkpoint(
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    run_id: str,
    output_dir: str,
    tag: str = "last",
) -> Tuple[int, float, Dict[str, List[float]], List[Dict]]:
    """
    Load checkpoint into head and optimizer.

    Returns:
        (start_epoch, best_val_acc, history, training_log) tuple.
    """
    path = os.path.join(_checkpoint_dir(output_dir, run_id), f"ckpt_{tag}.pt")
    if not os.path.exists(path):
        return 0, 0.0, {}, []
    ckpt = torch.load(path, map_location="cpu")
    head.load_state_dict(ckpt["head_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(f"Resumed from checkpoint: {path} (epoch {ckpt['epoch']+1})")
    
    return (
        ckpt["epoch"] + 1,
        ckpt.get("val_acc", 0.0),
        ckpt.get("history", {}),
        ckpt.get("training_log", [])
    )


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Backbone loader ───────────────────────────────────────────────────────────

def load_backbone(backbone_name: str) -> nn.Module:
    """Load and freeze a pretrained torchvision backbone."""
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
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    logger.info(f"Loaded and froze backbone: {backbone_name}")
    return model


def determine_environment(head_config: Dict[str, Any]) -> str:
    head_type = head_config.get("type", "")
    if head_type == "classical":
        return "simulation"
    return "emulation" if head_config.get("noise", False) else "simulation"


# ── Main training function ────────────────────────────────────────────────────

def train_and_evaluate(
    run_config,
    config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[Dict[str, Any], List[Dict], List[Dict]]]:
    """
    Execute a single training run with checkpoints, live progress, and visualizations.

    Args:
        run_config: RunConfig dataclass or dict with keys:
            run_id, dataset, backbone, head, seed, study.
        config: Full config dict (from config.yaml).
        overrides: Optional dict with parameter overrides.

    Returns:
        (result_dict, predictions_list, training_log_list) or None on error.
    """
    # ── Normalise run_config to dict ─────────────────────────────────────────
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
        _rc = dict(run_config)

    run_id = _rc.get("run_id", "unknown")
    output_dir = config.get("output_dir", "./results")
    ckpt_cfg = config.get("checkpoints", {})
    viz_cfg  = config.get("visualization", {})

    print(f"\n{'='*70}")
    print(f"  INITIATING EXPERIMENT: {run_id}")
    print(f"{'='*70}")

    try:
        # ── Seeds ────────────────────────────────────────────────────────────
        seed = _rc["seed"]
        set_seed(seed)

        # ── Dataset ──────────────────────────────────────────────────────────
        dataset_name  = _rc["dataset"]
        backbone_name = _rc["backbone"]
        head_name     = _rc["head"]

        dataset_config = next(
            (d for d in config["datasets"] if d["name"] == dataset_name), None
        )
        if not dataset_config:
            raise ValueError(f"Dataset not found in config: {dataset_name}")

        training_config = config.get("training", {})
        batch_size = training_config.get("batch_size", 32)
        dataset_type = dataset_config.get("type", "image")  # "image" | "tabular" | "numpy"

        train_loader, val_loader, test_loader = load_dataset(
            dataset_config, batch_size=batch_size, seed=seed
        )
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
        n_test  = len(test_loader.dataset)
        print(f"  Dataset  : {dataset_name} ({dataset_type})  "
              f"train={n_train}  val={n_val}  test={n_test}")

        num_classes = dataset_config.get("num_classes", 2)
        class_names = dataset_config.get("class_names",
                                         [str(i) for i in range(num_classes)])

        # ── Task type ─────────────────────────────────────────────────────────
        head_config_base = next(
            (h for h in config["heads"] if h["name"] == head_name), None
        )
        if not head_config_base:
            raise ValueError(f"Head not found in config: {head_name}")
        task_type = head_config_base.get("task_type", "classification")

        # ── Backbone + feature dim ────────────────────────────────────────────
        config_device = config.get("device", "auto")
        if config_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(config_device)

        if dataset_type == "image":
            backbone = load_backbone(backbone_name)
            backbone = backbone.to(device)
            sample_batch = next(iter(train_loader))
            with torch.no_grad():
                feat_sample = backbone(sample_batch[0][:1].to(device))
                feature_dim = feat_sample.view(1, -1).shape[1]
        else:
            # Tabular/numpy: features are already extracted — no backbone needed
            backbone = None
            sample_batch = next(iter(train_loader))
            feature_dim = sample_batch[0].shape[1]

        print(f"  Backbone : {backbone_name if dataset_type=='image' else 'none (tabular)'}  "
              f"feature_dim={feature_dim}  device={device}")

        # ── Head ──────────────────────────────────────────────────────────────
        head = get_head(head_config_base, feature_dim, num_classes, overrides=overrides)
        n_params = count_trainable_params(head)
        head = head.to(device)
        print(f"  Head     : {head_name}  task={task_type}  params={n_params}")

        # ── Clustering branch (non-gradient / sklearn-based) ──────────────────
        if task_type == "clustering":
            return _run_clustering(
                _rc, config, head, backbone, device,
                train_loader, val_loader, test_loader,
                dataset_name, backbone_name, head_name,
                head_config_base, feature_dim, num_classes,
                class_names, n_params, overrides, output_dir, viz_cfg,
            )

        # ── Optimizer + scheduler ─────────────────────────────────────────────
        epochs = (overrides or {}).get("epochs", training_config.get("epochs", 10))
        lr     = training_config.get("lr", 0.001)
        
        # Apply unitary-inspired regularization (Weight Decay) to classical heads 
        # to ensure architectural symmetry with bounded quantum operations.
        wd = 0.01 if head_config_base.get("type") == "classical" else 0.0
        optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
        
        sched_cfg = training_config.get("scheduler", {})
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 3),
            gamma=sched_cfg.get("gamma", 0.9),
        )
        criterion = nn.CrossEntropyLoss()
        print(f"  Optimizer: Adam  lr={lr}  wd={wd}  epochs={epochs}  "
              f"StepLR(step={sched_cfg.get('step_size',3)}, "
              f"gamma={sched_cfg.get('gamma',0.9)})")

        # ── Resume from checkpoint ────────────────────────────────────────────
        start_epoch = 0
        best_val_acc = 0.0
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "lr": [],
        }
        training_log = []

        if ckpt_cfg.get("resume", False):
            # Try to resume from 'last' for sequential continuity
            res_epoch, res_best, res_history, res_log = load_checkpoint(
                head, optimizer, run_id, output_dir, tag="last"
            )
            if res_epoch > 0:
                start_epoch = res_epoch
                best_val_acc = res_best
                history = res_history
                training_log = res_log
                print(f"  Resumed from epoch {start_epoch}  "
                      f"(prev_acc={best_val_acc*100:.1f}%)")

        # ── Energy tracker ────────────────────────────────────────────────────
        energy_kwh = None
        tracker = None
        if config.get("energy", {}).get("enabled", True):
            try:
                from codecarbon import EmissionsTracker
                tracker = EmissionsTracker(
                    project_name=run_id,
                    output_dir=output_dir,
                    log_level="error",
                )
                tracker.start()
            except Exception as e:
                logger.warning(f"CodeCarbon not available: {e}")

        total_start = time.perf_counter()

        print(f"\n  {'Epoch':>5}  {'TrainLoss':>10}  {'ValLoss':>9}  "
              f"{'TrainAcc':>9}  {'ValAcc':>8}  {'LR':>10}  {'Time':>8}  Best")
        print(f"  {'-'*75}")

        # Overall progress bar (across all epochs)
        overall_bar = None
        if HAS_TQDM:
            overall_bar = _tqdm(
                total=(epochs - start_epoch) * len(train_loader),
                desc=f"  {run_id[:40]}",
                unit="batch",
                dynamic_ncols=True,
                bar_format="{desc} |{bar:25}| {percentage:5.1f}% [{elapsed}<{remaining}]",
            )

        for epoch in range(start_epoch, epochs):
            epoch_start = time.perf_counter()
            current_lr  = scheduler.get_last_lr()[0] if epoch > 0 else lr

            # ── Train phase ───────────────────────────────────────────────────
            head.train()
            train_loss     = 0.0
            train_correct  = 0
            train_samples  = 0
            running_loss   = 0.0
            running_correct = 0
            running_n      = 0

            batch_bar = EpochBar(len(train_loader), epoch, epochs, run_id)

            for batch_idx, batch in enumerate(train_loader):
                inputs, labels = batch[0].to(device), batch[1].to(device)

                if backbone is not None:
                    with torch.no_grad():
                        inputs = backbone(inputs)

                logits = head(inputs)
                loss   = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = inputs.size(0)
                train_loss    += loss.item() * bs
                train_samples += bs
                preds          = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()

                running_loss    += loss.item() * bs
                running_correct += (preds == labels).sum().item()
                running_n       += bs

                batch_bar.update(
                    loss=running_loss / running_n,
                    acc=running_correct / running_n,
                    lr=current_lr,
                )
                if overall_bar:
                    overall_bar.update(1)

                # Reset running stats every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    running_loss = running_correct = running_n = 0.0

            batch_bar.close()
            train_loss /= train_samples
            train_acc   = train_correct / train_samples

            # ── Validation phase ──────────────────────────────────────────────
            head.eval()
            val_loss     = 0.0
            val_correct  = 0
            val_samples  = 0

            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    if backbone is not None:
                        inputs = backbone(inputs)
                    logits = head(inputs)
                    loss   = criterion(logits, labels)
                    val_loss    += loss.item() * inputs.size(0)
                    preds        = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_samples += inputs.size(0)

            val_loss /= val_samples
            val_acc   = val_correct / val_samples
            epoch_time = time.perf_counter() - epoch_start
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            # ── Update history ────────────────────────────────────────────────
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc * 100)
            history["val_acc"].append(val_acc * 100)
            history["lr"].append(current_lr)

            training_log.append({
                "run_id": run_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": current_lr,
                "epoch_time_s": epoch_time,
            })

            # ── Checkpoint: save every epoch + best ───────────────────────────
            ckpt_saved = False
            if ckpt_cfg.get("enabled", True):
                save_checkpoint(head, optimizer, epoch, val_acc,
                                run_id, output_dir, tag="last",
                                history=history, training_log=training_log)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(head, optimizer, epoch, val_acc,
                                    run_id, output_dir, tag="best",
                                    history=history, training_log=training_log)
                    ckpt_saved = True
                if ckpt_cfg.get("save_every_epoch", False):
                    save_checkpoint(head, optimizer, epoch, val_acc,
                                    run_id, output_dir, tag=f"epoch{epoch+1:03d}",
                                    history=history, training_log=training_log)
            else:
                best_val_acc = max(best_val_acc, val_acc)

            # ── Live epoch summary ────────────────────────────────────────────
            ckpt_flag = " *" if ckpt_saved else "  "
            print(
                f"  {epoch+1:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  "
                f"{train_acc*100:>8.2f}%  {val_acc*100:>7.2f}%  "
                f"{current_lr:>10.2e}  {_fmt_time(epoch_time):>8}{ckpt_flag}",
                flush=True,
            )

        # ── End training ──────────────────────────────────────────────────────
        if overall_bar:
            overall_bar.close()

        train_time_s = time.perf_counter() - total_start
        print(f"\n  Training completed successfully in {_fmt_time(train_time_s)}  "
              f"Peak Validation Accuracy: {best_val_acc*100:.2f}%\n")

        # ── Stop energy tracker ───────────────────────────────────────────────
        if tracker:
            try:
                energy_kwh = tracker.stop()
            except Exception as e:
                logger.warning(f"CodeCarbon stop failed: {e}")

        # ── Test evaluation ───────────────────────────────────────────────────
        predictions = []
        head.eval()
        print("  Starting final evaluation on the test set...")
        test_bar = _tqdm(test_loader, desc="  Test", unit="batch",
                         dynamic_ncols=True, leave=False) if HAS_TQDM else test_loader

        with torch.no_grad():
            sample_idx = 0
            for batch in test_bar:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                if backbone is not None:
                    inputs = backbone(inputs)
                logits = head(inputs)
                probs  = torch.softmax(logits, dim=1)
                preds  = logits.argmax(dim=1)
                for i in range(inputs.size(0)):
                    rec = {
                        "run_id":     run_id,
                        "sample_idx": sample_idx,
                        "y_true":     labels[i].item(),
                        "y_pred":     preds[i].item(),
                    }
                    for cls in range(num_classes):
                        rec[f"y_prob_{cls}"] = probs[i, cls].item()
                    predictions.append(rec)
                    sample_idx += 1

        # ── Aggregate metrics ─────────────────────────────────────────────────
        y_true  = np.array([p["y_true"] for p in predictions])
        y_pred  = np.array([p["y_pred"] for p in predictions])
        y_prob  = np.stack([[p[f"y_prob_{c}"] for c in range(num_classes)]
                             for p in predictions])

        metrics = {
            "test_accuracy":  accuracy_score(y_true, y_pred),
            "test_precision": precision_score(y_true, y_pred,
                                               average="macro", zero_division=0),
            "test_recall":    recall_score(y_true, y_pred,
                                            average="macro", zero_division=0),
            "test_f1":        f1_score(y_true, y_pred,
                                       average="macro", zero_division=0),
            "test_auc":       roc_auc_score(y_true, y_prob[:, 1]
                                             if num_classes == 2 else y_prob,
                                             multi_class="ovr")
                              if len(np.unique(y_true)) > 1 else 0.0,
        }

        print(f"\n  ── Final Evaluation Results (Test Partition) {'─'*30}")
        for k, v in metrics.items():
            print(f"    {k:<22}: {v:.4f}")
        print(f"  {'─'*65}\n")

        # ── Visualizations ────────────────────────────────────────────────────
        if viz_cfg.get("enabled", True):
            try:
                from visualization import generate_all_plots
                generate_all_plots(
                    run_id=run_id,
                    output_dir=output_dir,
                    task_type="classification",
                    history=history,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_prob=y_prob[:, 1] if num_classes == 2 else y_prob,
                    class_names=class_names,
                )
                print(f"  Plots saved to: {os.path.join(output_dir, 'plots', run_id)}\n")
            except Exception as e:
                logger.warning(f"Visualization failed (non-fatal): {e}")

        # ── Build result dict ──────────────────────────────────────────────────
        head_config_resolved = {**head_config_base, **(overrides or {})}
        result = {
            "run_id":            run_id,
            "study":             _rc.get("study", "main"),
            "seed":              seed,
            "dataset":           dataset_name,
            "backbone":          backbone_name,
            "head":              head_name,
            "head_type":         head_config_base.get("type", ""),
            "task_type":         task_type,
            "environment":       determine_environment(head_config_base),
            "n_qubits":          head_config_resolved.get("n_qubits"),
            "depth":             head_config_resolved.get("depth"),
            "shots":             head_config_resolved.get("shots"),
            "noise_channels":    ",".join(head_config_resolved.get("noise_channels", [])) or None,
            "n_trainable_params": n_params,
            "epochs":            epochs,
            "lr":                lr,
            "batch_size":        batch_size,
            "train_time_s":      train_time_s,
            "energy_kwh":        energy_kwh,
            **metrics,
            "timestamp":         datetime.now().isoformat(),
        }

        return result, predictions, training_log

    except Exception as e:
        logger.error(f"Error in training run {run_id}: {e}", exc_info=True)
        raise


# ── Clustering branch ─────────────────────────────────────────────────────────

def _run_clustering(
    _rc, config, head, backbone, device,
    train_loader, val_loader, test_loader,
    dataset_name, backbone_name, head_name,
    head_config_base, feature_dim, num_classes,
    class_names, n_params, overrides, output_dir, viz_cfg,
):
    """
    Unsupervised clustering pipeline.

    Extracts features from all splits, applies the clustering head
    (KMeans / DBSCAN / etc.), computes unsupervised metrics, and
    generates cluster visualizations.
    """
    run_id = _rc.get("run_id", "unknown")
    print("  Execution Mode: UNSUPERVISED CLUSTERING")

    # Extract all features ────────────────────────────────────────────────────
    def extract(loader):
        feats, labels = [], []
        bar = _tqdm(loader, desc="  Extracting features",
                    unit="batch", leave=False) if HAS_TQDM else loader
        with torch.no_grad():
            for batch in bar:
                x, y = batch[0].to(device), batch[1]
                if backbone is not None:
                    x = backbone(x)
                feats.append(x.cpu().numpy())
                labels.append(y.numpy())
        return np.vstack(feats), np.concatenate(labels)

    print("  Extracting features from train/val/test...")
    X_train, y_train = extract(train_loader)
    X_val,   y_val   = extract(val_loader)
    X_test,  y_test  = extract(test_loader)
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    print(f"  Total features: {X_all.shape}")

    # Fit clustering head ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    cluster_labels = head.fit_predict(X_all)
    fit_time = time.perf_counter() - t0

    # Unsupervised metrics ────────────────────────────────────────────────────
    from sklearn.metrics import (
        silhouette_score, davies_bouldin_score, calinski_harabasz_score,
        adjusted_rand_score, normalized_mutual_info_score,
    )

    n_clusters_found = len(np.unique(cluster_labels[cluster_labels >= 0]))
    metrics = {}
    if n_clusters_found >= 2:
        valid = cluster_labels >= 0  # DBSCAN may assign -1 (noise)
        try:
            metrics["silhouette"]          = silhouette_score(X_all[valid], cluster_labels[valid])
            metrics["davies_bouldin"]      = davies_bouldin_score(X_all[valid], cluster_labels[valid])
            metrics["calinski_harabasz"]   = calinski_harabasz_score(X_all[valid], cluster_labels[valid])
        except Exception as e:
            logger.warning(f"Unsupervised metrics failed: {e}")
        # Supervised proxy (if true labels exist)
        try:
            metrics["ari"] = adjusted_rand_score(y_all[valid], cluster_labels[valid])
            metrics["nmi"] = normalized_mutual_info_score(y_all[valid], cluster_labels[valid])
        except Exception:
            pass

    print(f"\n  ── Clustering Results {'─'*45}")
    print(f"    Clusters found       : {n_clusters_found}")
    print(f"    Fit time             : {_fmt_time(fit_time)}")
    for k, v in metrics.items():
        print(f"    {k:<25}: {v:.4f}")
    print(f"  {'─'*65}\n")

    # Visualizations ──────────────────────────────────────────────────────────
    if viz_cfg.get("enabled", True):
        try:
            from visualization import generate_all_plots
            k_range = viz_cfg.get("elbow_k_range", list(range(2, 11)))
            generate_all_plots(
                run_id=run_id,
                output_dir=output_dir,
                task_type="clustering",
                y_true=y_all,
                features=X_all,
                cluster_labels=cluster_labels,
                k_range=k_range,
            )
            print(f"  Plots saved to: {os.path.join(output_dir, 'plots', run_id)}\n")
        except Exception as e:
            logger.warning(f"Clustering visualization failed (non-fatal): {e}")

    # Build result dict ────────────────────────────────────────────────────────
    result = {
        "run_id":            run_id,
        "study":             _rc.get("study", "main"),
        "seed":              _rc["seed"],
        "dataset":           dataset_name,
        "backbone":          backbone_name,
        "head":              head_name,
        "task_type":         "clustering",
        "n_clusters_found":  n_clusters_found,
        "n_trainable_params": n_params,
        "fit_time_s":        fit_time,
        **metrics,
        "timestamp":         datetime.now().isoformat(),
    }

    # Predictions = per-sample cluster assignment
    predictions = [
        {"run_id": run_id, "sample_idx": i,
         "cluster": int(cluster_labels[i]), "y_true": int(y_all[i])}
        for i in range(len(y_all))
    ]

    return result, predictions, []
