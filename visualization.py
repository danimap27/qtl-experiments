"""
Visualization module for QTL experimentation framework.

Generates comprehensive plots for both supervised (classification) and
unsupervised (clustering) tasks:
  - Learning curves (loss + accuracy per epoch)
  - Confusion matrix (normalized and raw)
  - ROC curve + AUC
  - Precision-Recall curve
  - Per-class metrics bar chart
  - Probability calibration histogram
  - Clustering: PCA / t-SNE scatter
  - Clustering: silhouette analysis
  - Clustering: elbow curve (KMeans)
  - Clustering: cluster size distribution

All functions save PNG files and return the figure path. Designed to run on
headless cluster environments (Agg backend).
"""

import os
import logging
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # must be before pyplot import — headless for SLURM nodes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ── Style constants ────────────────────────────────────────────────────────────
DPI = 150
CMAP_CM = LinearSegmentedColormap.from_list("qtl_cm", ["#ffffff", "#1F4E79"])
CMAP_CLUSTER = "tab10"
COLOR_TRAIN = "#2E75B6"
COLOR_VAL   = "#C00000"
COLOR_TEST  = "#538135"
GRID_ALPHA  = 0.25
MARKER_SIZE = 5


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str) -> str:
    """Save figure, close it, and return path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        logger.info(f"[VIZ] Saved: {path}")
    except Exception as e:
        logger.warning(f"[VIZ] Could not save {path}: {e}")
    finally:
        plt.close(fig)
    return path


def _check_sklearn():
    try:
        import sklearn  # noqa
        return True
    except ImportError:
        logger.warning("[VIZ] scikit-learn not available — some plots skipped.")
        return False


# ── 1. Learning curves ────────────────────────────────────────────────────────

def plot_learning_curves(
    history: Dict[str, List[float]],
    run_id: str,
    output_dir: str,
) -> str:
    """
    Plot training and validation loss + accuracy over epochs.

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
                 and optionally 'lr' (learning rate per epoch).
        run_id:  Experiment identifier (used in title and filename).
        output_dir: Directory where the PNG will be saved.

    Returns:
        Path to saved PNG.
    """
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        logger.warning("[VIZ] plot_learning_curves: empty history, skipping.")
        return ""

    has_lr = "lr" in history and len(history["lr"]) == len(epochs)
    n_rows = 2 if has_lr else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(f"Learning Curves — {run_id}", fontsize=13, fontweight="bold")

    # Row 0: Loss
    ax = axes[0][0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], "o-", color=COLOR_TRAIN,
                markersize=MARKER_SIZE, label="Train")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], "s-", color=COLOR_VAL,
                markersize=MARKER_SIZE, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Cross-Entropy Loss per Epoch")
    ax.legend(); ax.grid(True, alpha=GRID_ALPHA)

    # Row 0: Accuracy
    ax = axes[0][1]
    if "train_acc" in history:
        ax.plot(epochs, history["train_acc"], "o-", color=COLOR_TRAIN,
                markersize=MARKER_SIZE, label="Train")
    if "val_acc" in history:
        ax.plot(epochs, history["val_acc"], "s-", color=COLOR_VAL,
                markersize=MARKER_SIZE, label="Val")
    if "test_acc" in history and history["test_acc"]:
        ax.axhline(history["test_acc"][-1], color=COLOR_TEST, linestyle="--",
                   linewidth=1.5, label=f"Test: {history['test_acc'][-1]:.1f}%")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy per Epoch")
    ax.legend(); ax.grid(True, alpha=GRID_ALPHA)

    # Row 1 (optional): LR schedule
    if has_lr:
        ax = axes[1][0]
        ax.plot(epochs, history["lr"], "o-", color="#7030A0", markersize=MARKER_SIZE)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log"); ax.grid(True, alpha=GRID_ALPHA)
        axes[1][1].set_visible(False)

    path = os.path.join(output_dir, f"{run_id}_learning_curves.png")
    return _save(fig, path)


# ── 2. Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    run_id: str,
    output_dir: str,
) -> str:
    """
    Plot raw and normalised confusion matrix side-by-side.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: List of class name strings.
        run_id: Experiment identifier.
        output_dir: Output directory.

    Returns:
        Path to saved PNG.
    """
    if not _check_sklearn():
        return ""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrix — {run_id}", fontsize=13, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw counts", "Normalised (recall per row)"],
        ["d", ".2f"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap=CMAP_CM,
                       vmin=0, vmax=data.max())
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            xlabel="Predicted",
            ylabel="True",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center", fontsize=11,
                        color="white" if data[i, j] > thresh else "black")

    path = os.path.join(output_dir, f"{run_id}_confusion_matrix.png")
    return _save(fig, path)


# ── 3. ROC curve ─────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    run_id: str,
    output_dir: str,
) -> str:
    """
    Plot ROC curve(s) with AUC for binary or multiclass problems.

    Args:
        y_true: Ground-truth labels (integers).
        y_prob: Predicted probabilities, shape (n_samples,) or (n_samples, n_classes).
        class_names: List of class names.
        run_id: Experiment identifier.
        output_dir: Output directory.

    Returns:
        Path to saved PNG.
    """
    if not _check_sklearn():
        return ""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"ROC Curve — {run_id}", fontsize=13, fontweight="bold")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    n_classes = len(class_names)
    if n_classes == 2:
        # Binary: use probability of positive class
        prob_pos = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, prob_pos)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLOR_TRAIN, linewidth=2,
                label=f"AUC = {roc_auc:.4f}")
    else:
        # Multiclass: one-vs-rest
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, (cls, color) in enumerate(zip(class_names, colors)):
            prob_i = y_prob[:, i] if y_prob.ndim == 2 else (y_true == i).astype(float)
            fpr, tpr, _ = roc_curve(y_bin[:, i], prob_i)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{cls} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.legend(loc="lower right"); ax.grid(True, alpha=GRID_ALPHA)

    path = os.path.join(output_dir, f"{run_id}_roc_curve.png")
    return _save(fig, path)


# ── 4. Precision-Recall curve ─────────────────────────────────────────────────

def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    run_id: str,
    output_dir: str,
) -> str:
    """
    Plot Precision-Recall curve(s) with AP for binary or multiclass.

    Returns:
        Path to saved PNG.
    """
    if not _check_sklearn():
        return ""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Precision-Recall Curve — {run_id}", fontsize=13, fontweight="bold")

    n_classes = len(class_names)
    if n_classes == 2:
        prob_pos = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, prob_pos)
        ap = average_precision_score(y_true, prob_pos)
        ax.plot(recall, precision, color=COLOR_TRAIN, linewidth=2,
                label=f"AP = {ap:.4f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, (cls, color) in enumerate(zip(class_names, colors)):
            prob_i = y_prob[:, i] if y_prob.ndim == 2 else (y_true == i).astype(float)
            precision, recall, _ = precision_recall_curve(y_bin[:, i], prob_i)
            ap = average_precision_score(y_bin[:, i], prob_i)
            ax.plot(recall, precision, color=color, linewidth=2,
                    label=f"{cls} (AP={ap:.3f})")

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.legend(loc="upper right"); ax.grid(True, alpha=GRID_ALPHA)

    path = os.path.join(output_dir, f"{run_id}_pr_curve.png")
    return _save(fig, path)


# ── 5. Per-class metrics bar chart ─────────────────────────────────────────────

def plot_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    run_id: str,
    output_dir: str,
) -> str:
    """
    Bar chart with Precision, Recall, F1-score per class.

    Returns:
        Path to saved PNG.
    """
    if not _check_sklearn():
        return ""
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25
    colors = [COLOR_TRAIN, COLOR_VAL, COLOR_TEST]

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 2), 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [report[cls][metric] for cls in class_names]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width); ax.set_xticklabels(class_names)
    ax.set_ylim([0, 1.15]); ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Metrics — {run_id}", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=GRID_ALPHA, axis="y")

    # Add macro/weighted avg lines
    for metric, color, ls in zip(metrics, colors, ["--", "-.", ":"]):
        avg = report["macro avg"][metric]
        ax.axhline(avg, color=color, linestyle=ls, linewidth=1.2, alpha=0.7)

    path = os.path.join(output_dir, f"{run_id}_class_metrics.png")
    return _save(fig, path)


# ── 6. Probability calibration histogram ─────────────────────────────────────

def plot_probability_histogram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    run_id: str,
    output_dir: str,
) -> str:
    """
    Histogram of predicted probabilities split by true class (binary).

    Returns:
        Path to saved PNG.
    """
    prob_pos = y_prob if y_prob.ndim == 1 else y_prob[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Predicted Probability Distribution — {run_id}",
                 fontsize=13, fontweight="bold")

    unique_labels = np.unique(y_true)
    colors = [COLOR_TRAIN, COLOR_VAL]
    for label, color in zip(unique_labels, colors):
        mask = y_true == label
        ax.hist(prob_pos[mask], bins=30, alpha=0.6, color=color,
                label=f"True class {int(label)}", density=True)

    ax.axvline(0.5, color="k", linestyle="--", linewidth=1.5, label="Decision boundary")
    ax.set_xlabel("P(positive class)"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(True, alpha=GRID_ALPHA)

    path = os.path.join(output_dir, f"{run_id}_prob_histogram.png")
    return _save(fig, path)


# ── 7. Full classification report (summary panel) ────────────────────────────

def plot_classification_summary(
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    run_id: str,
    output_dir: str,
) -> str:
    """
    One-page 2×3 summary panel: learning curves, confusion matrix, ROC,
    PR curve, probability histogram, per-class bars.

    Returns:
        Path to saved PNG.
    """
    if not _check_sklearn():
        return ""
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc,
        precision_recall_curve, average_precision_score,
        classification_report,
    )

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Classification Report — {run_id}", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    # ── Panel 1: Loss ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if epochs:
        ax1.plot(epochs, history.get("train_loss", []), "o-", color=COLOR_TRAIN,
                 markersize=4, label="Train")
        ax1.plot(epochs, history.get("val_loss", []), "s-", color=COLOR_VAL,
                 markersize=4, label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=GRID_ALPHA)

    # ── Panel 2: Accuracy ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if epochs:
        ax2.plot(epochs, history.get("train_acc", []), "o-", color=COLOR_TRAIN,
                 markersize=4, label="Train")
        ax2.plot(epochs, history.get("val_acc", []), "s-", color=COLOR_VAL,
                 markersize=4, label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Acc (%)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=GRID_ALPHA)

    # ── Panel 3: Confusion Matrix (normalised) ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax3.imshow(cm_norm, cmap=CMAP_CM, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
            xticklabels=class_names, yticklabels=class_names,
            title="Confusion Matrix (norm.)", xlabel="Predicted", ylabel="True")
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax3.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                     fontsize=10, color="white" if cm_norm[i,j] > 0.5 else "black")

    # ── Panel 4: ROC ─────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    prob_pos = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    roc_auc = auc(fpr, tpr)
    ax4.plot(fpr, tpr, color=COLOR_TRAIN, linewidth=2, label=f"AUC={roc_auc:.4f}")
    ax4.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax4.set_title("ROC Curve"); ax4.set_xlabel("FPR"); ax4.set_ylabel("TPR")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=GRID_ALPHA)

    # ── Panel 5: PR curve ─────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    precision, recall, _ = precision_recall_curve(y_true, prob_pos)
    ap = average_precision_score(y_true, prob_pos)
    ax5.plot(recall, precision, color=COLOR_VAL, linewidth=2, label=f"AP={ap:.4f}")
    ax5.set_title("Precision-Recall"); ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
    ax5.legend(fontsize=9); ax5.grid(True, alpha=GRID_ALPHA)

    # ── Panel 6: Per-class F1 ─────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    metrics = ["precision", "recall", "f1-score"]
    colors = [COLOR_TRAIN, COLOR_VAL, COLOR_TEST]
    x = np.arange(len(class_names))
    width = 0.25
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [report[cls][metric] for cls in class_names]
        ax6.bar(x + i * width, vals, width, label=metric[:4].capitalize(),
                color=color, alpha=0.85)
    ax6.set_xticks(x + width); ax6.set_xticklabels(class_names, fontsize=9)
    ax6.set_ylim([0, 1.2]); ax6.set_title("Per-Class Metrics")
    ax6.legend(fontsize=9); ax6.grid(True, alpha=GRID_ALPHA, axis="y")

    path = os.path.join(output_dir, f"{run_id}_summary.png")
    return _save(fig, path)


# ── 8. Clustering: PCA / t-SNE scatter ────────────────────────────────────────

def plot_cluster_scatter(
    features: np.ndarray,
    labels: np.ndarray,
    run_id: str,
    output_dir: str,
    method: str = "pca",
    true_labels: Optional[np.ndarray] = None,
) -> str:
    """
    2D scatter plot of features coloured by cluster assignment.
    Optionally shows true labels alongside for evaluation.

    Args:
        features: (n_samples, n_features) feature matrix.
        labels: (n_samples,) cluster assignments (integers).
        run_id: Experiment identifier.
        output_dir: Output directory.
        method: Dimensionality reduction method: 'pca' or 'tsne'.
        true_labels: Optional ground-truth labels for comparison.

    Returns:
        Path to saved PNG.
    """
    if not _check_sklearn():
        return ""
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = reducer.fit_transform(features)

    n_panels = 2 if true_labels is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    method_label = method.upper() if method == "pca" else "t-SNE"

    for ax, lbl, title in zip(
        axes,
        [labels, true_labels] if true_labels is not None else [labels],
        [f"Predicted clusters", "True labels"] if true_labels is not None else ["Cluster assignments"],
    ):
        unique = np.unique(lbl)
        cmap = plt.cm.get_cmap(CMAP_CLUSTER, len(unique))
        for i, u in enumerate(unique):
            mask = lbl == u
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)],
                       s=18, alpha=0.7, label=str(u))
        ax.set_title(f"{method_label} — {title}", fontsize=12)
        ax.set_xlabel(f"{method_label}-1"); ax.set_ylabel(f"{method_label}-2")
        ax.legend(markerscale=2, fontsize=9,
                  title="Cluster" if "cluster" in title.lower() else "Class")
        ax.grid(True, alpha=GRID_ALPHA)

    fig.suptitle(f"Cluster Visualisation — {run_id}", fontsize=13, fontweight="bold")
    path = os.path.join(output_dir, f"{run_id}_cluster_{method}.png")
    return _save(fig, path)


# ── 9. Clustering: silhouette analysis ────────────────────────────────────────

def plot_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    run_id: str,
    output_dir: str,
) -> str:
    """
    Silhouette plot: per-sample silhouette coefficient sorted by cluster and value.

    Returns:
        Path to saved PNG.
    """
    if not _check_sklearn():
        return ""
    from sklearn.metrics import silhouette_samples, silhouette_score

    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        logger.warning("[VIZ] silhouette requires >= 2 clusters, skipping.")
        return ""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_sil = silhouette_samples(features, labels)
        avg_sil = silhouette_score(features, labels)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"Silhouette Analysis — {run_id} (avg={avg_sil:.3f})",
                 fontsize=13, fontweight="bold")
    cmap = plt.cm.get_cmap(CMAP_CLUSTER, n_clusters)
    y_lower = 10

    for i, cluster in enumerate(sorted(np.unique(labels))):
        sil_vals = np.sort(sample_sil[labels == cluster])
        size = len(sil_vals)
        y_upper = y_lower + size
        color = cmap(i)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, sil_vals, alpha=0.8,
                         color=color, label=f"C{cluster} (n={size})")
        ax.text(-0.05, y_lower + 0.5 * size, str(cluster), fontsize=10, color=color)
        y_lower = y_upper + 10

    ax.axvline(avg_sil, color="red", linestyle="--", linewidth=1.5,
               label=f"Avg = {avg_sil:.3f}")
    ax.set_xlabel("Silhouette coefficient"); ax.set_ylabel("Cluster")
    ax.set_title("Silhouette per Sample"); ax.legend(fontsize=9)
    ax.grid(True, alpha=GRID_ALPHA, axis="x")

    path = os.path.join(output_dir, f"{run_id}_silhouette.png")
    return _save(fig, path)


# ── 10. Clustering: elbow curve ───────────────────────────────────────────────

def plot_elbow(
    features: np.ndarray,
    k_range: List[int],
    run_id: str,
    output_dir: str,
) -> str:
    """
    Elbow curve: inertia vs. number of clusters for KMeans.

    Args:
        features: (n_samples, n_features).
        k_range: List of k values to evaluate.
        run_id: Experiment identifier.
        output_dir: Output directory.

    Returns:
        Path to saved PNG.
    """
    if not _check_sklearn():
        return ""
    from sklearn.cluster import KMeans

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(features)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, "bo-", markersize=8)
    ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Inertia (WCSS)")
    ax.set_title(f"KMeans Elbow Curve — {run_id}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xticks(k_range)

    path = os.path.join(output_dir, f"{run_id}_elbow.png")
    return _save(fig, path)


# ── 11. Clustering: cluster size distribution ─────────────────────────────────

def plot_cluster_sizes(
    labels: np.ndarray,
    run_id: str,
    output_dir: str,
) -> str:
    """
    Bar chart showing the number of samples per cluster.

    Returns:
        Path to saved PNG.
    """
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(max(6, len(unique)), 4))
    colors = plt.cm.get_cmap(CMAP_CLUSTER, len(unique))(np.linspace(0, 1, len(unique)))
    bars = ax.bar([str(u) for u in unique], counts, color=colors, alpha=0.85)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Number of samples")
    ax.set_title(f"Cluster Size Distribution — {run_id}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=GRID_ALPHA, axis="y")

    path = os.path.join(output_dir, f"{run_id}_cluster_sizes.png")
    return _save(fig, path)


# ── 12. Feature importance / distribution (tabular) ──────────────────────────

def plot_feature_distributions(
    X: np.ndarray,
    feature_names: List[str],
    labels: Optional[np.ndarray],
    run_id: str,
    output_dir: str,
    max_features: int = 16,
) -> str:
    """
    Grid of histograms for each feature, optionally split by class/cluster.

    Args:
        X: (n_samples, n_features) feature matrix.
        feature_names: Feature name strings.
        labels: Optional class/cluster labels (colouring).
        run_id: Experiment identifier.
        output_dir: Output directory.
        max_features: Maximum number of features to plot.

    Returns:
        Path to saved PNG.
    """
    n_features = min(len(feature_names), max_features)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    fig.suptitle(f"Feature Distributions — {run_id}", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    unique_labels = np.unique(labels) if labels is not None else None
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels))) if unique_labels is not None else [COLOR_TRAIN]

    for i in range(n_features):
        ax = axes[i]
        if unique_labels is not None:
            for lbl, color in zip(unique_labels, colors):
                mask = labels == lbl
                ax.hist(X[mask, i], bins=20, alpha=0.6, color=color,
                        label=str(lbl), density=True)
        else:
            ax.hist(X[:, i], bins=20, alpha=0.8, color=COLOR_TRAIN, density=True)
        ax.set_title(feature_names[i], fontsize=9)
        ax.grid(True, alpha=GRID_ALPHA)

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    if unique_labels is not None:
        fig.legend(
            [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors],
            [str(l) for l in unique_labels],
            loc="lower right", fontsize=9, title="Label",
        )

    path = os.path.join(output_dir, f"{run_id}_feature_distributions.png")
    return _save(fig, path)


# ── 13. Master dispatcher ─────────────────────────────────────────────────────

def generate_all_plots(
    run_id: str,
    output_dir: str,
    task_type: str = "classification",
    history: Optional[Dict] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    features: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    k_range: Optional[List[int]] = None,
) -> Dict[str, str]:
    """
    Master dispatcher: calls all applicable plot functions and returns
    a dict mapping plot_name -> file_path.

    Args:
        run_id: Experiment identifier.
        output_dir: Output directory for plots.
        task_type: 'classification' or 'clustering'.
        history: Training history dict.
        y_true, y_pred, y_prob: Ground-truth, predictions, probabilities.
        class_names: Class name strings.
        features: Raw feature matrix (for clustering plots).
        cluster_labels: Cluster assignment array.
        feature_names: Feature names (for tabular datasets).
        k_range: List of k values for elbow curve.

    Returns:
        Dict of {plot_name: path_to_png}.
    """
    plots_dir = os.path.join(output_dir, "plots", run_id)
    os.makedirs(plots_dir, exist_ok=True)
    generated = {}

    if task_type == "classification":
        # Learning curves
        if history:
            path = plot_learning_curves(history, run_id, plots_dir)
            if path:
                generated["learning_curves"] = path

        if y_true is not None and y_pred is not None:
            cn = class_names or [str(i) for i in np.unique(y_true)]

            # Confusion matrix
            path = plot_confusion_matrix(y_true, y_pred, cn, run_id, plots_dir)
            if path:
                generated["confusion_matrix"] = path

            # Per-class metrics
            path = plot_class_metrics(y_true, y_pred, cn, run_id, plots_dir)
            if path:
                generated["class_metrics"] = path

            if y_prob is not None:
                # ROC
                path = plot_roc_curve(y_true, y_prob, cn, run_id, plots_dir)
                if path:
                    generated["roc_curve"] = path

                # PR
                path = plot_pr_curve(y_true, y_prob, cn, run_id, plots_dir)
                if path:
                    generated["pr_curve"] = path

                # Probability histogram (binary only)
                if len(cn) == 2:
                    path = plot_probability_histogram(y_true, y_prob, run_id, plots_dir)
                    if path:
                        generated["prob_histogram"] = path

                # Summary panel
                if history:
                    path = plot_classification_summary(
                        history, y_true, y_pred, y_prob, cn, run_id, plots_dir
                    )
                    if path:
                        generated["summary"] = path

    elif task_type == "clustering":
        if features is not None and cluster_labels is not None:
            # PCA scatter
            path = plot_cluster_scatter(features, cluster_labels, run_id, plots_dir,
                                        method="pca", true_labels=y_true)
            if path:
                generated["cluster_pca"] = path

            # t-SNE scatter (only if reasonable size)
            if len(features) <= 5000:
                path = plot_cluster_scatter(features, cluster_labels, run_id, plots_dir,
                                            method="tsne", true_labels=y_true)
                if path:
                    generated["cluster_tsne"] = path

            # Silhouette
            path = plot_silhouette(features, cluster_labels, run_id, plots_dir)
            if path:
                generated["silhouette"] = path

            # Cluster sizes
            path = plot_cluster_sizes(cluster_labels, run_id, plots_dir)
            if path:
                generated["cluster_sizes"] = path

            # Elbow (only if k_range provided)
            if k_range:
                path = plot_elbow(features, k_range, run_id, plots_dir)
                if path:
                    generated["elbow"] = path

        # Feature distributions
        if features is not None and feature_names:
            lbl = cluster_labels if cluster_labels is not None else y_true
            path = plot_feature_distributions(features, feature_names, lbl,
                                              run_id, plots_dir)
            if path:
                generated["feature_distributions"] = path

    logger.info(f"[VIZ] Generated {len(generated)} plots for {run_id}: "
                f"{list(generated.keys())}")
    return generated
