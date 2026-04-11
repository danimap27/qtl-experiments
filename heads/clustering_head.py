"""
Unsupervised clustering heads for quantum transfer learning experiments.

Provides KMeans and DBSCAN wrappers with a scikit-learn–compatible interface
wrapped as nn.Module so they integrate with the existing trainer/head factory.

Both heads expose:
    forward(x)          → dummy logits tensor (zeros) for API compatibility
    fit_predict(X_np)   → cluster label array (numpy int64)
    fit(X_np)           → fits model, returns self
    predict(X_np)       → label array for new data
    cluster_centers_    → cluster centers (KMeans only)

The trainer calls `head.fit_predict(features)` when `task_type == "clustering"`.
"""

import logging
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseClusteringHead(nn.Module):
    """
    Abstract base for clustering heads.
    Provides a thin nn.Module shell around scikit-learn clusterers.
    """

    def __init__(self, feature_dim: int, n_clusters: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_clusters  = n_clusters
        self._fitted     = False

        # Dummy parameter so PyTorch does not complain about empty modules
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    # ------------------------------------------------------------------ #
    # nn.Module interface (pass-through — clustering is fit_predict based)
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a zero tensor of shape (batch, n_clusters).
        The trainer does not use logits from clustering heads; this method
        exists solely to preserve API compatibility with classification heads.
        """
        return torch.zeros(x.size(0), self.n_clusters, device=x.device)

    # ------------------------------------------------------------------ #
    # Clustering interface (subclasses must implement)
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "BaseClusteringHead":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)


# ---------------------------------------------------------------------------
# KMeans head
# ---------------------------------------------------------------------------

class KMeansHead(BaseClusteringHead):
    """
    K-Means clustering head.

    Wraps sklearn.cluster.KMeans with IBM Heron r2–compatible defaults
    (deterministic initialization via random_state).

    Args:
        feature_dim  : Dimension of input feature vectors.
        n_clusters   : Number of clusters (default: 2).
        max_iter     : Maximum number of iterations (default: 300).
        n_init       : Number of initializations (default: 10).
        tol          : Convergence tolerance (default: 1e-4).
        random_state : Random seed for reproducibility (default: 42).
        kwargs       : Additional keyword arguments forwarded to sklearn KMeans.
    """

    def __init__(
        self,
        feature_dim: int,
        n_clusters:  int = 2,
        max_iter:    int = 300,
        n_init:      int = 10,
        tol:         float = 1e-4,
        random_state: int = 42,
        **kwargs: Any,
    ):
        super().__init__(feature_dim, n_clusters)
        self.max_iter     = max_iter
        self.n_init       = n_init
        self.tol          = tol
        self.random_state = random_state
        self._extra       = kwargs
        self._model       = None

        logger.info(
            f"KMeansHead initialized: n_clusters={n_clusters}, "
            f"n_init={n_init}, max_iter={max_iter}, seed={random_state}"
        )

    def _build_model(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for KMeansHead. "
                "Install with: pip install scikit-learn"
            )
        return KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            tol=self.tol,
            random_state=self.random_state,
            **self._extra,
        )

    def fit(self, X: np.ndarray) -> "KMeansHead":
        """Fit K-Means on feature matrix X (n_samples × feature_dim)."""
        self._model = self._build_model()
        self._model.fit(X)
        self._fitted = True
        logger.debug(f"KMeans fitted on {X.shape[0]} samples, inertia={self._model.inertia_:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign cluster labels to X."""
        if not self._fitted:
            raise RuntimeError("KMeansHead must be fitted before predict(). Call fit() first.")
        return self._model.predict(X).astype(np.int64)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels in a single call."""
        self._model = self._build_model()
        labels = self._model.fit_predict(X).astype(np.int64)
        self._fitted = True
        logger.info(
            f"KMeans fit_predict: {X.shape[0]} samples → "
            f"{len(np.unique(labels))} clusters found"
        )
        return labels

    @property
    def cluster_centers_(self) -> Optional[np.ndarray]:
        """Cluster centre coordinates (n_clusters × feature_dim), or None if not fitted."""
        if self._fitted and self._model is not None:
            return self._model.cluster_centers_
        return None

    @property
    def inertia_(self) -> Optional[float]:
        """Within-cluster sum of squares (None if not fitted)."""
        if self._fitted and self._model is not None:
            return float(self._model.inertia_)
        return None


# ---------------------------------------------------------------------------
# DBSCAN head
# ---------------------------------------------------------------------------

class DBSCANHead(BaseClusteringHead):
    """
    DBSCAN density-based clustering head.

    Wraps sklearn.cluster.DBSCAN.  Because DBSCAN does not produce a
    fixed number of clusters, ``n_clusters`` here is treated as a hint
    for parameter selection rather than a hard constraint.

    Note: DBSCAN assigns label ``-1`` to noise points.  The trainer
    excludes noise points when computing supervised metrics (ARI/NMI).

    Args:
        feature_dim : Dimension of input feature vectors.
        n_clusters  : Expected number of clusters (used only for logging).
        eps         : Neighbourhood radius (default: 0.5).
        min_samples : Minimum points per core region (default: 5).
        metric      : Distance metric (default: 'euclidean').
        algorithm   : Nearest-neighbour algorithm (default: 'auto').
        kwargs      : Additional keyword arguments forwarded to sklearn DBSCAN.
    """

    def __init__(
        self,
        feature_dim: int,
        n_clusters:  int  = 2,
        eps:         float = 0.5,
        min_samples: int   = 5,
        metric:      str   = "euclidean",
        algorithm:   str   = "auto",
        **kwargs: Any,
    ):
        super().__init__(feature_dim, n_clusters)
        self.eps         = eps
        self.min_samples = min_samples
        self.metric      = metric
        self.algorithm   = algorithm
        self._extra      = kwargs
        self._model      = None
        self._labels     = None

        logger.info(
            f"DBSCANHead initialized: eps={eps}, min_samples={min_samples}, "
            f"metric={metric}"
        )

    def _build_model(self):
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise ImportError(
                "scikit-learn is required for DBSCANHead. "
                "Install with: pip install scikit-learn"
            )
        return DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            **self._extra,
        )

    def fit(self, X: np.ndarray) -> "DBSCANHead":
        """Fit DBSCAN on feature matrix X and cache labels."""
        self._model  = self._build_model()
        self._labels = self._model.fit_predict(X).astype(np.int64)
        self._fitted = True
        n_found = len(set(self._labels) - {-1})
        n_noise = int(np.sum(self._labels == -1))
        logger.debug(
            f"DBSCAN fitted on {X.shape[0]} samples: "
            f"{n_found} clusters, {n_noise} noise points"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign labels for new samples via nearest-core-point lookup.
        Returns -1 (noise) for points outside any cluster.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("DBSCANHead must be fitted before predict(). Call fit() first.")
        # sklearn DBSCAN has no native predict; use nearest-core-point approach
        try:
            from sklearn.metrics import pairwise_distances_argmin
            core_mask   = self._model.core_sample_indices_
            core_labels = self._labels[core_mask]
            core_pts    = self._model.components_   # shape (n_core, feature_dim)
            nn_idx      = pairwise_distances_argmin(X, core_pts, metric=self.metric)
            raw_labels  = core_labels[nn_idx]
            return raw_labels.astype(np.int64)
        except Exception:
            # Fallback: return all -1 (noise) when core point lookup fails
            logger.warning("DBSCAN.predict fallback: returning -1 for all new samples")
            return np.full(len(X), -1, dtype=np.int64)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit DBSCAN and return cluster labels (including -1 for noise)."""
        self.fit(X)
        n_found = len(set(self._labels) - {-1})
        n_noise = int(np.sum(self._labels == -1))
        logger.info(
            f"DBSCAN fit_predict: {X.shape[0]} samples → "
            f"{n_found} clusters, {n_noise} noise points"
        )
        return self._labels.copy()

    @property
    def n_clusters_found_(self) -> Optional[int]:
        """Number of clusters actually found (excluding noise), or None if not fitted."""
        if self._fitted and self._labels is not None:
            return len(set(self._labels.tolist()) - {-1})
        return None

    @property
    def n_noise_(self) -> Optional[int]:
        """Number of noise points (-1 labels), or None if not fitted."""
        if self._fitted and self._labels is not None:
            return int(np.sum(self._labels == -1))
        return None


# ---------------------------------------------------------------------------
# Elbow curve helper (used by visualization.py)
# ---------------------------------------------------------------------------

def compute_elbow_curve(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute K-Means inertia for a range of k values (elbow analysis).

    Args:
        X            : Feature matrix (n_samples × feature_dim).
        k_range      : Range of k values to evaluate.
        random_state : Random seed.

    Returns:
        Dictionary with 'k_values' and 'inertias' lists.
    """
    try:
        from sklearn.cluster import KMeans as _KM
    except ImportError:
        raise ImportError("scikit-learn is required for compute_elbow_curve.")

    k_values, inertias = [], []
    for k in k_range:
        if k > len(X):
            break
        km = _KM(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X)
        k_values.append(k)
        inertias.append(float(km.inertia_))
        logger.debug(f"Elbow: k={k}, inertia={km.inertia_:.4f}")

    return {"k_values": k_values, "inertias": inertias}
