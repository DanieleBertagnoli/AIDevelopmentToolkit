from typing import Dict, List, Literal, Optional, Tuple

from aidevelopmenttoolkit.general_utils import check_shape
from aidevelopmenttoolkit.logging_utils.logger import get_formatted_logger

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils.parallel import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    completeness_score,
    homogeneity_score,
    v_measure_score,
    fowlkes_mallows_score,
)

logger = get_formatted_logger(name=__name__, level="ERROR")

def compute_classification_metrics(confusion_matrix: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute classification metrics from a confusion matrix.

    This function computes classification metrics directly from an accumulated
    confusion matrix. It is useful when predictions and labels are aggregated
    over multiple batches or distributed processes, avoiding the need to store
    all predictions and labels.

    Notes
    -----
    The following metrics are computed:
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - Macro F1 score
    - Matthews correlation coefficient (MCC)

    The function automatically detects the classification problem type:
    - Binary classification if the confusion matrix has shape `(2, 2)`.
    - Multiclass classification otherwise.

    For multiclass problems, Precision, Recall, and F1 Score are computed
    using macro averaging.

    Macro F1 Score is the harmonic mean of per-class F1 scores for both binary and
    multiclass problems. All classes in the confusion matrix receive equal
    weight; classes absent from both labels and predictions contribute zero.

    MCC is computed for both binary and multiclass problems and is zero when
    its denominator is zero (including an empty confusion matrix).

    The confusion matrix is expected to follow the convention:

        rows -> ground-truth classes
        columns -> predicted classes

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix with shape `(C, C)` where:
        - `C`: Number of classes

        Element `(i, j)` represents the number of samples belonging to
        ground-truth class `i` that were predicted as class `j`.

    Returns
    -------
    Tuple[Dict[str, float], np.ndarray]
        Respectively:
            - Dictionary containing the computed classification metrics.
            - Row-normalized confusion matrix.

    Examples
    --------
    >>> cm = np.array(
    ...     [
    ...         [8, 2],
    ...         [1, 9],
    ...     ]
    ... )
    >>> metrics, normalized_cm = compute_classification_metrics(cm)
    """

    confusion_matrix = np.asarray(confusion_matrix, dtype=np.float64,)

    check_shape(confusion_matrix, (confusion_matrix.shape[0], confusion_matrix.shape[1]))

    num_classes = confusion_matrix.shape[0]

    # Compute correctly classified samples
    true_positives = np.diag(confusion_matrix)

    # Compute accuracy
    total_samples = confusion_matrix.sum()
    accuracy = true_positives.sum() / total_samples if total_samples > 0 else 0.0

    # Compute Matthews correlation coefficient (MCC)
    predicted_counts = confusion_matrix.sum(axis=0)
    actual_counts = confusion_matrix.sum(axis=1)
    mcc_numerator = true_positives.sum() * total_samples - np.dot(actual_counts, predicted_counts)
    mcc_denominator_squared = (
        (total_samples ** 2 - np.dot(predicted_counts, predicted_counts))
        * (total_samples ** 2 - np.dot(actual_counts, actual_counts))
    )
    mcc = mcc_numerator / np.sqrt(mcc_denominator_squared) if mcc_denominator_squared > 0 else 0.0

    f1_denominators = confusion_matrix.sum(axis=0) + confusion_matrix.sum(axis=1)
    class_f1_scores = np.divide(
        2 * true_positives,
        f1_denominators,
        out=np.zeros_like(true_positives),
        where=f1_denominators > 0,
    )
    macro_f1_score = np.mean(class_f1_scores)

    if num_classes == 2:

        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]
        tp = confusion_matrix[1, 1]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    else:

        # Multiclass classification:
        precisions = np.full(num_classes, np.nan, dtype=np.float64)
        recalls = np.full(num_classes, np.nan, dtype=np.float64)

        for class_idx in range(num_classes):

            tp = true_positives[class_idx]

            predicted_as_class = confusion_matrix[:, class_idx].sum()
            actual_class = confusion_matrix[class_idx, :].sum()

            if predicted_as_class > 0:
                precisions[class_idx] = tp / predicted_as_class

            if actual_class > 0:
                recalls[class_idx] = tp / actual_class


        # Macro averaging: every class has equal importance
        precision = np.nanmean(precisions)
        recall = np.nanmean(recalls)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


    metrics = {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1_score),
        "Macro F1 Score": float(macro_f1_score),
        "MCC": float(mcc),
    }


    # Normalize the confusion matrix by row so each true-class sum is 1
    normalized_confusion_matrix = confusion_matrix.copy()

    row_sums = normalized_confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized_confusion_matrix /= row_sums

    return metrics, normalized_confusion_matrix


def compute_confusion_matrix(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        padding_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
    """Compute the confusion matrix from predictions and labels.

    This function computes a confusion matrix that can be accumulated across
    multiple batches or distributed processes before computing classification
    metrics.

    Notes
    -----
    The confusion matrix follows the convention:

        rows -> ground-truth classes
        columns -> predicted classes

    Element `(i, j)` represents the number of samples belonging to
    ground-truth class `i` that were predicted as class `j`.

    Padding positions are ignored when a padding mask is provided.

    Parameters
    ----------
    predictions : np.ndarray
        Array containing the predicted classes. Accepted shapes:
        `(N,)` or any shape that can be flattened consistently with labels.

    labels : np.ndarray
        Array containing the ground-truth classes.
        Same shape as `predictions`.

    num_classes : int
        Number of classes in the classification problem.

    padding_mask : np.ndarray, default=None
        Boolean array where `True` indicates padding positions that must be
        ignored. Same shape as `predictions`.

    Returns
    -------
    np.ndarray
        Confusion matrix with shape `(num_classes, num_classes)`.

    Examples
    --------
    >>> predictions = np.array([0, 1, 1, 0])
    >>> labels = np.array([0, 1, 0, 0])
    >>> cm = compute_confusion_matrix(
    ...     predictions=predictions,
    ...     labels=labels,
    ...     num_classes=2,
    ... )
    """

    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    check_shape(labels, predictions.shape)

    if padding_mask is not None:

        padding_mask = np.asarray(padding_mask, dtype=bool)

        if predictions.shape != padding_mask.shape:
            logger.error(
                "The given `predictions` and `padding_mask` have different "
                "shapes: "
                f"{predictions.shape} vs {padding_mask.shape}", ValueError
            )

        valid = ~padding_mask

        predictions = predictions[valid]
        labels = labels[valid]

    # Flatten all dimensions
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    if np.any(predictions < 0) or np.any(predictions >= num_classes):
        logger.error(
            "Predictions contain class indices outside the valid range "
            f"[0, {num_classes - 1}].", ValueError
        )

    if np.any(labels < 0) or np.any(labels >= num_classes):
        logger.error(
            "Labels contain class indices outside the valid range "
            f"[0, {num_classes - 1}].", ValueError
        )

    return confusion_matrix(
        labels,
        predictions,
        labels=np.arange(num_classes),
    )


def _cluster_pair_distance_matrix(
        embeddings: np.ndarray,
        cluster_ids: np.ndarray,
        metric: str,
    ) -> np.ndarray:
    unique_ids = np.unique(cluster_ids)
    clusters = [embeddings[cluster_ids == cluster_id] for cluster_id in unique_ids]
    distance_matrix = np.zeros((unique_ids.size, unique_ids.size), dtype=np.float64)

    for i, points_a in enumerate(clusters):
        for j in range(i + 1, len(clusters)):
            distance = cdist(points_a, clusters[j], metric=metric).mean()
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix


def compute_distance_matrix(
        predictions: np.ndarray,
        labels: np.ndarray,
        embeddings: np.ndarray,
        metric: str = "euclidean",
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ground-truth and predicted cluster-pair distance matrices.

    Each off-diagonal entry is the mean distance between all pairs of points
    in the corresponding clusters. Diagonal entries are zero. Rows and columns
    follow the sorted unique cluster IDs in `labels` and `predictions`,
    respectively; the two matrices may therefore have different shapes.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted cluster IDs with shape `(N,)`.

    labels : np.ndarray
        Ground-truth cluster IDs with shape `(N,)`.

    embeddings : np.ndarray
        Point embeddings with shape `(N, D)`.

    metric : str, default="euclidean"
        Distance metric forwarded to `scipy.spatial.distance.cdist`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Ground-truth and predicted cluster-pair distance matrices.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    check_shape(embeddings, (-1, -1))
    check_shape(predictions, (embeddings.shape[0],))
    check_shape(labels, (embeddings.shape[0],))

    ground_truth_matrix = _cluster_pair_distance_matrix(embeddings, labels, metric)
    predicted_matrix = _cluster_pair_distance_matrix(embeddings, predictions, metric)

    return ground_truth_matrix, predicted_matrix


def cluster_distance_stats(
        embeddings: np.ndarray,
        cluster_ids: np.ndarray,
        metric: str = "euclidean",
    ) -> Dict[str, float]:
    """Compute intra- and inter-cluster distance statistics.

    Notes
    -----
    For each cluster `c`, compute intra-cluster `mu_c` and `sigma_c` from distances
    between unique pairs of points within c (excluding self-distances).
    Clusters with fewer than two points have no intra-cluster statistics.

    Compute inter-cluster `mu_c` and `sigma_c` from all point-to-point distances
    between points in c and points outside c. These statistics are undefined
    when there is no other cluster.

    Additional metrics describe the relationship between intra- and inter-cluster
    distances, including compactness, separation, normalized separation, and
    percentile-based worst-case behavior.

    Any distance metric accepted by `scipy.spatial.distance.cdist` can be
    used (e.g. "euclidean", "cosine", "cityblock").

    Parameters
    ----------
    embeddings : np.ndarray
        2-D array of point coordinates with shape `(N, D)` where:
        - `N`: Number of points
        - `D`: Embedding dimensionality

    cluster_ids : np.ndarray
        1-D integer array of cluster assignments with shape `(N,)`.
        Each element is the cluster id of the corresponding embedding.

    metric : str, default="euclidean"
        Distance metric forwarded to `scipy.spatial.distance.cdist`.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
            - "Mean Intra-Cluster Distance": Mean intra-cluster distance.
            - "Mean Inter-Cluster Distance": Mean inter-cluster distance.
            - "Mean Intra-Cluster Distance Std": Mean intra-cluster standard deviation.
            - "Mean Inter-Cluster Distance Std": Mean inter-cluster standard deviation.
            - "Intra-to-Inter Cluster Distance Ratio": Ratio between intra- and inter-cluster means.
            - "Cluster Separation Margin": Difference between inter- and intra-cluster means.
            - "Normalized Cluster Separation": Normalized difference between inter- and intra-cluster means.
            - "Intra-Cluster Distance Coefficient of Variation": Intra-cluster coefficient of variation.
            - "Inter-Cluster Distance Coefficient of Variation": Inter-cluster coefficient of variation.
            - "90th Percentile Intra-Cluster Distance": 90th percentile of all intra-cluster distances.
            - "10th Percentile Inter-Cluster Distance": 10th percentile of all inter-cluster distances.
            - "90th Percentile Intra-Cluster Mean Distance": 90th percentile of per-cluster intra means.
            - "10th Percentile Inter-Cluster Mean Distance": 10th percentile of per-cluster inter means.
            - "10th Percentile Cluster Separation Margin": 10th percentile of per-cluster separation margins.
            - "Worst-Case Intra-to-Inter Cluster Distance Ratio": Ratio between the 90th percentile intra and 10th percentile inter cluster means.

        Undefined metrics are returned as nan.

    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)

    check_shape(embeddings, (-1, -1))
    num_points = embeddings.shape[0]
    check_shape(cluster_ids, (num_points,))

    unique_cluster_ids = np.unique(cluster_ids)

    intra_cluster_mus: List[float] = []
    intra_cluster_sigmas: List[float] = []
    inter_cluster_mus: List[float] = []
    inter_cluster_sigmas: List[float] = []

    intra_distances_all: List[np.ndarray] = []
    inter_distances_all: List[np.ndarray] = []
    cluster_separation_margins: List[float] = []

    for cluster_id in unique_cluster_ids:
        cluster_mask = cluster_ids == cluster_id
        cluster_points = embeddings[cluster_mask]

        intra_mu_c = float(np.nan)
        intra_sigma_c = float(np.nan)

        if cluster_points.shape[0] >= 2:

            # Compute the intra-cluster distance matrix for the current cluster
            intra_distance_matrix = cdist(cluster_points, cluster_points, metric=metric)

            # Keep each unordered pair once, excluding diagonal self-distances
            intra_distances = intra_distance_matrix[np.triu_indices(cluster_points.shape[0], k=1)]
            intra_mu_c = float(intra_distances.mean())
            intra_sigma_c = float(intra_distances.std(ddof=0))

            intra_cluster_mus.append(intra_mu_c)
            intra_cluster_sigmas.append(intra_sigma_c)
            intra_distances_all.append(intra_distances)

        outside_cluster_points = embeddings[~cluster_mask]
        if outside_cluster_points.shape[0] > 0:

            # Compute the inter-cluster distance matrix between the current cluster and all outside points
            inter_distances = cdist(cluster_points, outside_cluster_points, metric=metric)
            inter_mu_c = float(inter_distances.mean())
            inter_sigma_c = float(inter_distances.std(ddof=0))

            inter_cluster_mus.append(inter_mu_c)
            inter_cluster_sigmas.append(inter_sigma_c)
            inter_distances_all.append(inter_distances.ravel())

            if np.isfinite(intra_mu_c):
                cluster_separation_margins.append(inter_mu_c - intra_mu_c)

    # Final aggregation
    intra_mu = float(np.mean(intra_cluster_mus)) if intra_cluster_mus else float(np.nan)
    inter_mu = float(np.mean(inter_cluster_mus)) if inter_cluster_mus else float(np.nan)
    intra_sigma = float(np.mean(intra_cluster_sigmas)) if intra_cluster_sigmas else float(np.nan)
    inter_sigma = float(np.mean(inter_cluster_sigmas)) if inter_cluster_sigmas else float(np.nan)

    # Compute relative compactness and separation metrics
    intra_inter_ratio = (
        float(intra_mu / inter_mu)
        if np.isfinite(intra_mu) and np.isfinite(inter_mu) and inter_mu > 0
        else float(np.nan)
    )

    separation_margin = (
        float(inter_mu - intra_mu)
        if np.isfinite(intra_mu) and np.isfinite(inter_mu)
        else float(np.nan)
    )

    normalized_separation = (
        float((inter_mu - intra_mu) / (inter_mu + intra_mu))
        if np.isfinite(intra_mu) and np.isfinite(inter_mu)
        and (inter_mu + intra_mu) > 0
        else float(np.nan)
    )

    # Compute coefficients of variation
    intra_cv = (
        float(intra_sigma / intra_mu)
        if np.isfinite(intra_sigma) and np.isfinite(intra_mu) and intra_mu > 0
        else float(np.nan)
    )

    inter_cv = (
        float(inter_sigma / inter_mu)
        if np.isfinite(inter_sigma) and np.isfinite(inter_mu) and inter_mu > 0
        else float(np.nan)
    )

    # Compute percentile-based worst-case metrics
    intra_distances_all = (
        np.concatenate(intra_distances_all)
        if intra_distances_all else np.array([])
    )

    inter_distances_all = (
        np.concatenate(inter_distances_all)
        if inter_distances_all else np.array([])
    )

    worst_intra_distance = (
        float(np.percentile(intra_distances_all, 90))
        if intra_distances_all.size > 0 else float(np.nan)
    )

    worst_inter_distance = (
        float(np.percentile(inter_distances_all, 10))
        if inter_distances_all.size > 0 else float(np.nan)
    )

    worst_intra_cluster_mean = (
        float(np.percentile(intra_cluster_mus, 90))
        if intra_cluster_mus else float(np.nan)
    )

    worst_inter_cluster_mean = (
        float(np.percentile(inter_cluster_mus, 10))
        if inter_cluster_mus else float(np.nan)
    )

    worst_cluster_separation_margin = (
        float(np.percentile(cluster_separation_margins, 10))
        if cluster_separation_margins else float(np.nan)
    )

    worst_cluster_intra_inter_ratio = (
        float(worst_intra_cluster_mean / worst_inter_cluster_mean)
        if np.isfinite(worst_intra_cluster_mean)
        and np.isfinite(worst_inter_cluster_mean)
        and worst_inter_cluster_mean > 0
        else float(np.nan)
    )

    stats = {
        "Mean Intra-Cluster Distance": intra_mu,
        "Mean Inter-Cluster Distance": inter_mu,
        "Mean Intra-Cluster Distance Std": intra_sigma,
        "Mean Inter-Cluster Distance Std": inter_sigma,
        "Intra-to-Inter Cluster Distance Ratio": intra_inter_ratio,
        "Cluster Separation Margin": separation_margin,
        "Normalized Cluster Separation": normalized_separation,
        "Intra-Cluster Distance Coefficient of Variation": intra_cv,
        "Inter-Cluster Distance Coefficient of Variation": inter_cv,
        "90th Percentile Intra-Cluster Distance": worst_intra_distance,
        "10th Percentile Inter-Cluster Distance": worst_inter_distance,
        "90th Percentile Intra-Cluster Mean Distance": worst_intra_cluster_mean,
        "10th Percentile Inter-Cluster Mean Distance": worst_inter_cluster_mean,
        "10th Percentile Cluster Separation Margin": worst_cluster_separation_margin,
        "Worst-Case Intra-to-Inter Cluster Distance Ratio": worst_cluster_intra_inter_ratio,
    }

    return {f"{metric.capitalize()} {key}": value for key, value in stats.items()}


def compute_clustering_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        padding_mask: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        distance_metrics: Optional[List[Literal["euclidean", "cosine", "manhattan"]]] = None,
        n_jobs: int = -1,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Compute per-sequence clustering metrics and their batch averages.

    Notes
    -----
    Computes Completeness, Homogeneity, V-Measure and Fowlkes-Mallows via
    :mod:`sklearn.metrics`, and Elements Like Me (ELM) via :func:`_elm_score`.
    Optional distance means and standard deviations for ground-truth and
    predicted clusters follow :func:`cluster_distance_stats`.

    Sequences with fewer than two valid points produce only `nan` values.
    Homogeneity, Completeness and V-Measure are also `nan` when either
    clustering has only one cluster. Batch averages ignore `nan` values.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted cluster IDs with shape `(T,)` or `(B, T)`, where `B` is
        batch size and `T` is sequence length.

    labels : np.ndarray
        Ground-truth cluster labels. Same shape as `predictions`.

    padding_mask : Optional[np.ndarray], default=None
        Boolean array with the same shape as `predictions`; `True` excludes
        a point from all metrics. `None` means no padding.

    embeddings : np.ndarray, default=None
        Point embeddings with shape `(T, D)` or `(B, T, D)`, where `D` is
        embedding dimensionality. Required for distance statistics.

    distance_metrics : Optional[List[Literal["euclidean", "cosine", "manhattan"]]], default=None
        Metrics forwarded to :func:`cluster_distance_stats`. Distance outputs
        are omitted when this is empty/`None`. If provided, `embeddings` is
        required.

    n_jobs : int, default=-1
        Number of parallel workers for batch processing. `-1` uses all
        available CPU cores. Forwarded to :class:`joblib.Parallel`.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, float]]
        - `non_averaged` contains arrays of shape `(B,)`; 
        - `averaged` contains their NaN-aware means, or `nan` if no sequence has a defined value.

    See Also
    --------
    :func:`cluster_distance_stats` : Per-cluster distance distributions and reduction.
    :func:`_elm_score` : Elements Like Me score definition and paper reference.

    Examples
    --------
    >>> predictions = np.array([0, 0, 1, 1])
    >>> labels = np.array([0, 0, 1, 2])
    >>> padding_mask = np.zeros_like(labels, dtype=bool)
    >>> non_averaged, averaged = compute_clustering_metrics(
    ...     predictions=predictions,
    ...     labels=labels,
    ...     padding_mask=padding_mask,
    ... )
    """

    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if padding_mask is None:
        padding_mask = np.zeros_like(labels, dtype=bool)

    padding_mask = np.asarray(padding_mask, dtype=bool)

    check_shape(predictions, [(-1,), (-1, -1)])
    check_shape(labels, predictions.shape)
    check_shape(labels, padding_mask.shape)

    # Add batch axis
    if predictions.ndim == 1:
            predictions = predictions[None, :]
            labels = labels[None, :]
            padding_mask = padding_mask[None, :]

    B, T = predictions.shape

    if distance_metrics and embeddings is None:
        raise ValueError("embeddings are required when distance_metrics are provided")

    use_embeddings = embeddings is not None
    if use_embeddings:
        embeddings = np.asarray(embeddings, dtype=np.float64)

    use_distances = use_embeddings and bool(distance_metrics)

    # Add batch axis
    if use_embeddings and embeddings.ndim == 2:
        embeddings = embeddings[None, :]
        check_shape(embeddings, (B, T, -1))

    def _process_batch(batch_idx: int) -> Dict[str, float]:
        result: Dict[str, float] = {}

        valid = ~padding_mask[batch_idx]
        if not np.any(valid):
            return result

        pred = predictions[batch_idx][valid]
        gt = labels[batch_idx][valid]

        if pred.size < 2:
            return result

        result["Fowlkes-Mallows Score"] = fowlkes_mallows_score(gt, pred)
        result["ELM Score"] = _elm_score(pred, gt)

        if use_distances:
            emb = embeddings[batch_idx][valid]
            for dist_metric in distance_metrics:
                for cluster_type, cluster_ids in (("Predicted", pred), ("GT", gt)):
                    stats = cluster_distance_stats(embeddings=emb, cluster_ids=cluster_ids, metric=dist_metric)
                    for k, v in stats.items():
                        result[f"{cluster_type} {k}"] = v

                
        n_pred_clusters = np.unique(pred).size
        n_gt_clusters = np.unique(gt).size

        if n_pred_clusters > 1 and n_gt_clusters > 1:
            result["Completeness"] = completeness_score(gt, pred)
            result["Homogeneity"] = homogeneity_score(gt, pred)
            result["V-Measure Score"] = v_measure_score(gt, pred)

        return result

    batch_results = Parallel(n_jobs=n_jobs)(
        delayed(_process_batch)(i)
        for i in tqdm(
            range(B),
            leave=False,
            colour="cyan",
            desc="Computing clustering metrics 📊",
        )
    )

    non_averaged: Dict[str, np.ndarray] = {}
    for result in batch_results:
        for key in result:
            non_averaged.setdefault(key, np.full(B, np.nan))

    for batch_idx, result in enumerate(batch_results):
        for key, value in result.items():
            non_averaged[key][batch_idx] = value

    averaged = {
        key: float(np.nanmean(values)) if np.any(~np.isnan(values)) else float(np.nan)
        for key, values in non_averaged.items()
    }

    return non_averaged, averaged


def _elm_score(
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
    """
    Vectorized implementation of the Elements Like Me (ELM) F1 score.
    
    Notes
    -----
    Official paper: https://link.springer.com/article/10.1007/s10791-024-09436-7.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted cluster labels. Shape (N,).

    labels : np.ndarray
        Ground-truth cluster labels. Shape (N,).

    Returns
    -------
    float
        Mean ELM F1 score over all elements.
    """

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    N = len(predictions)

    if N == 0:
        return np.nan

    off_diag = ~np.eye(N, dtype=bool)
    same_pred = (predictions[:, None] == predictions[None, :]) & off_diag
    same_gt   = (labels[:, None]   == labels[None, :])   & off_diag

    tp_vec = (same_pred & same_gt).sum(axis=1)
    fp_vec = (same_pred & ~same_gt).sum(axis=1)
    fn_vec = (~same_pred & same_gt).sum(axis=1)

    both_isolated = (same_pred.sum(axis=1) == 0) & (same_gt.sum(axis=1) == 0)
    denom = tp_vec + 0.5 * (fp_vec + fn_vec)
    safe_denom = np.where(denom == 0, 1.0, denom) # avoid division by zero in masked-out positions

    # F1 (paper Eq. 3 with modified TP)
    f1_scores = np.where(both_isolated, 1.0, np.where(tp_vec == 0, 0.0, tp_vec / safe_denom))

    return float(np.mean(f1_scores))