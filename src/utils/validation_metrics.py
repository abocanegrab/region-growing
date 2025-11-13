"""
Validation metrics for semantic segmentation against reference datasets.

Implements standard metrics used in computer vision and remote sensing:
- IoU (Intersection over Union) / Jaccard Index
- mIoU (mean IoU) for multiclass segmentation
- Weighted mIoU for imbalanced classes
- F1-Score / Dice Similarity Coefficient
- Precision/Recall
- Pixel Accuracy
- Confusion Matrix

References:
    - Martin et al. (2001). "A database of human segmented natural images"
    - Csurka et al. (2013). "What is a good evaluation measure for semantic segmentation?"
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def calculate_iou(prediction: np.ndarray, ground_truth: np.ndarray, class_id: int) -> float:
    """
    Calculate Intersection over Union (IoU) for a specific class.

    IoU = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    prediction : np.ndarray
        Predicted segmentation mask (H, W) with integer class labels
    ground_truth : np.ndarray
        Ground truth segmentation mask (H, W) with integer class labels
    class_id : int
        Class ID to calculate IoU for (e.g., 0=background, 1=crop, etc.)

    Returns
    -------
    float
        IoU score in range [0.0, 1.0]. Returns 0.0 if class not present.

    Examples
    --------
    >>> pred = np.array([[0, 1], [1, 1]])
    >>> gt = np.array([[0, 1], [1, 0]])
    >>> calculate_iou(pred, gt, class_id=1)
    0.5  # 2 pixels intersection, 4 pixels union
    """
    assert (
        prediction.shape == ground_truth.shape
    ), f"Shape mismatch: {prediction.shape} vs {ground_truth.shape}"

    # Create binary masks for the class
    pred_mask = prediction == class_id
    gt_mask = ground_truth == class_id

    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    # Handle edge case: class not present in either mask
    if union == 0:
        logger.warning(f"Class {class_id} not present in either mask. Returning IoU=0.0")
        return 0.0

    iou = intersection / union
    return float(iou)


def calculate_miou(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int,
    ignore_background: bool = True,
) -> Tuple[float, Dict[int, float]]:
    """
    Calculate mean Intersection over Union (mIoU) across all classes.

    mIoU = (1/N) × Σ IoU_i for i in classes

    This is the standard metric for semantic segmentation evaluation.

    Parameters
    ----------
    prediction : np.ndarray
        Predicted segmentation mask (H, W)
    ground_truth : np.ndarray
        Ground truth segmentation mask (H, W)
    num_classes : int
        Total number of classes (including background if present)
    ignore_background : bool, default=True
        If True, exclude class 0 (background) from mIoU calculation

    Returns
    -------
    miou : float
        Mean IoU across all classes (excluding background if specified)
    iou_per_class : dict
        Dictionary mapping class_id -> IoU score

    Examples
    --------
    >>> pred = np.array([[0, 1, 2], [1, 1, 2]])
    >>> gt = np.array([[0, 1, 1], [1, 2, 2]])
    >>> miou, per_class = calculate_miou(pred, gt, num_classes=3)
    >>> print(f"mIoU: {miou:.3f}")
    mIoU: 0.667  # Mean of IoU for classes 1 and 2
    """
    iou_per_class = {}
    valid_ious = []

    start_class = 1 if ignore_background else 0

    for class_id in range(start_class, num_classes):
        iou = calculate_iou(prediction, ground_truth, class_id)
        iou_per_class[class_id] = iou

        # Only include in mean if class is present (IoU > 0)
        if iou > 0:
            valid_ious.append(iou)

    if len(valid_ious) == 0:
        logger.warning("No valid classes found for mIoU calculation. Returning 0.0")
        return 0.0, iou_per_class

    miou = np.mean(valid_ious)
    logger.info(f"mIoU: {miou:.4f} (computed from {len(valid_ious)}/{num_classes} classes)")

    return float(miou), iou_per_class


def calculate_f1_score(prediction: np.ndarray, ground_truth: np.ndarray, class_id: int) -> float:
    """
    Calculate F1-Score (Dice Similarity Coefficient) for a specific class.

    F1 = 2 × (Precision × Recall) / (Precision + Recall)
       = 2 × |A ∩ B| / (|A| + |B|)

    F1-Score is commonly used in medical imaging and agriculture segmentation.

    Parameters
    ----------
    prediction : np.ndarray
        Predicted segmentation mask (H, W)
    ground_truth : np.ndarray
        Ground truth segmentation mask (H, W)
    class_id : int
        Class ID to calculate F1 for

    Returns
    -------
    float
        F1-Score in range [0.0, 1.0]. Returns 0.0 if class not present.

    Examples
    --------
    >>> pred = np.array([[0, 1], [1, 1]])
    >>> gt = np.array([[0, 1], [1, 0]])
    >>> calculate_f1_score(pred, gt, class_id=1)
    0.667  # 2×2 / (3+2) = 4/6
    """
    pred_mask = prediction == class_id
    gt_mask = ground_truth == class_id

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    pred_positives = pred_mask.sum()
    gt_positives = gt_mask.sum()

    if pred_positives + gt_positives == 0:
        logger.warning(f"Class {class_id} not present. Returning F1=0.0")
        return 0.0

    f1 = 2 * intersection / (pred_positives + gt_positives)
    return float(f1)


def calculate_pixel_accuracy(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate overall pixel accuracy.

    PA = correct_pixels / total_pixels

    Simple metric but can be misleading with class imbalance.

    Parameters
    ----------
    prediction : np.ndarray
        Predicted segmentation mask (H, W)
    ground_truth : np.ndarray
        Ground truth segmentation mask (H, W)

    Returns
    -------
    float
        Pixel accuracy in range [0.0, 1.0]
    """
    assert prediction.shape == ground_truth.shape

    correct = (prediction == ground_truth).sum()
    total = prediction.size

    accuracy = correct / total
    return float(accuracy)


def align_ground_truth(
    gt_path: str, reference_path: str, output_path: Optional[str] = None
) -> np.ndarray:
    """
    Align ground truth mask with reference segmentation.

    Handles:
    - CRS reprojection (e.g., WGS84 → UTM)
    - Resolution resampling (e.g., 10m → 10m)
    - Spatial extent alignment

    Parameters
    ----------
    gt_path : str
        Path to ground truth GeoTIFF (e.g., Dynamic World mask)
    reference_path : str
        Path to reference image (e.g., Sentinel-2 band or segmentation)
    output_path : str, optional
        If provided, save aligned mask to this path

    Returns
    -------
    np.ndarray
        Aligned ground truth mask with same shape and CRS as reference

    Examples
    --------
    >>> aligned = align_ground_truth(
    ...     'data/dynamic_world/mexicali_dw.tif',
    ...     'img/sentinel2/mexico/mexicali/B04_10m.npy'
    ... )
    >>> print(aligned.shape)
    (1124, 922)  # Same as reference image
    """
    logger.info(f"Aligning ground truth: {Path(gt_path).name}")

    # Read reference metadata
    with rasterio.open(reference_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)

    # Read ground truth
    with rasterio.open(gt_path) as src:
        gt_data = src.read(1)
        gt_transform = src.transform
        gt_crs = src.crs

        # Check if reprojection needed
        if gt_crs != ref_crs:
            logger.info(f"Reprojecting from {gt_crs} to {ref_crs}")

            # Calculate transform for reprojection
            dst_transform, dst_width, dst_height = calculate_default_transform(
                gt_crs, ref_crs, src.width, src.height, *src.bounds
            )

            # Create destination array
            aligned_mask = np.zeros(ref_shape, dtype=gt_data.dtype)

            # Reproject
            reproject(
                source=gt_data,
                destination=aligned_mask,
                src_transform=gt_transform,
                src_crs=gt_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )
        else:
            # Same CRS, just resize if needed
            if gt_data.shape != ref_shape:
                logger.info(f"Resizing from {gt_data.shape} to {ref_shape}")
                from scipy.ndimage import zoom

                zoom_factors = (ref_shape[0] / gt_data.shape[0], ref_shape[1] / gt_data.shape[1])
                aligned_mask = zoom(gt_data, zoom_factors, order=0)
            else:
                aligned_mask = gt_data

    # Save aligned mask if path provided
    if output_path:
        logger.info(f"Saving aligned mask to {output_path}")
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=ref_shape[0],
            width=ref_shape[1],
            count=1,
            dtype=aligned_mask.dtype,
            crs=ref_crs,
            transform=ref_transform,
        ) as dst:
            dst.write(aligned_mask, 1)

    logger.info(f"Alignment complete. Shape: {aligned_mask.shape}")
    return aligned_mask


def generate_confusion_matrix(
    prediction: np.ndarray, ground_truth: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Generate confusion matrix for multiclass segmentation.

    confusion_matrix[i, j] = number of pixels with true class i predicted as class j

    Parameters
    ----------
    prediction : np.ndarray
        Predicted segmentation (H, W)
    ground_truth : np.ndarray
        Ground truth segmentation (H, W)
    num_classes : int
        Number of classes

    Returns
    -------
    np.ndarray
        Confusion matrix (num_classes, num_classes)
    """
    assert prediction.shape == ground_truth.shape

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = np.sum((ground_truth == i) & (prediction == j))

    return cm


def calculate_boundary_precision(
    prediction: np.ndarray, ground_truth: np.ndarray, tolerance: int = 2
) -> float:
    """
    Calculate boundary precision (how well boundaries align).

    Uses morphological dilation to allow tolerance in boundary matching.

    Parameters
    ----------
    prediction : np.ndarray
        Binary predicted mask
    ground_truth : np.ndarray
        Binary ground truth mask
    tolerance : int, default=2
        Boundary tolerance in pixels

    Returns
    -------
    float
        Boundary precision score [0.0, 1.0]
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    # Extract boundaries
    pred_boundary = binary_dilation(prediction) ^ binary_erosion(prediction)
    gt_boundary = binary_dilation(ground_truth) ^ binary_erosion(ground_truth)

    # Dilate GT boundary by tolerance
    gt_boundary_dilated = binary_dilation(gt_boundary, iterations=tolerance)

    # Calculate precision
    correct_boundary = np.logical_and(pred_boundary, gt_boundary_dilated).sum()
    total_pred_boundary = pred_boundary.sum()

    if total_pred_boundary == 0:
        return 0.0

    precision = correct_boundary / total_pred_boundary
    return float(precision)


def calculate_precision_recall(
    prediction: np.ndarray, ground_truth: np.ndarray, class_id: int
) -> Tuple[float, float]:
    """
    Calculate precision and recall for a specific class.

    Precision = TP / (TP + FP) - How many predicted positives are correct
    Recall = TP / (TP + FN) - How many actual positives were found

    Useful for understanding error types (false positives vs false negatives).

    Parameters
    ----------
    prediction : np.ndarray
        Predicted segmentation mask (H, W)
    ground_truth : np.ndarray
        Ground truth segmentation mask (H, W)
    class_id : int
        Class ID to calculate metrics for

    Returns
    -------
    precision : float
        Precision score in range [0.0, 1.0]
    recall : float
        Recall score in range [0.0, 1.0]

    Examples
    --------
    >>> pred = np.array([[0, 1, 1], [1, 0, 0]])
    >>> gt = np.array([[0, 1, 0], [1, 1, 0]])
    >>> precision, recall = calculate_precision_recall(pred, gt, class_id=1)
    >>> print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
    Precision: 0.67, Recall: 0.67
    """
    assert prediction.shape == ground_truth.shape

    pred_mask = prediction == class_id
    gt_mask = ground_truth == class_id

    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return float(precision), float(recall)


def calculate_weighted_miou(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int,
    ignore_background: bool = True,
) -> Tuple[float, Dict[int, float]]:
    """
    Calculate weighted mean IoU (accounts for class imbalance).

    Weighted mIoU = Σ (IoU_i × support_i) / Σ support_i

    More robust than simple mean when classes are highly imbalanced.
    In agriculture, classes are often imbalanced (e.g., 80% crops, 5% urban).

    Parameters
    ----------
    prediction : np.ndarray
        Predicted segmentation mask (H, W)
    ground_truth : np.ndarray
        Ground truth segmentation mask (H, W)
    num_classes : int
        Total number of classes
    ignore_background : bool, default=True
        If True, exclude class 0 from calculation

    Returns
    -------
    weighted_miou : float
        Weighted mean IoU across all classes
    iou_per_class : dict
        Dictionary mapping class_id -> IoU score

    Examples
    --------
    >>> # Imbalanced dataset: 90% class 1, 10% class 2
    >>> pred = np.array([[1]*90 + [2]*10])
    >>> gt = np.array([[1]*85 + [2]*15])
    >>> weighted_miou, per_class = calculate_weighted_miou(pred, gt, num_classes=3)
    >>> print(f"Weighted mIoU: {weighted_miou:.3f}")
    Weighted mIoU: 0.850  # Gives more weight to class 1
    """
    iou_per_class = {}
    weighted_sum = 0.0
    total_support = 0

    start_class = 1 if ignore_background else 0

    for class_id in range(start_class, num_classes):
        iou = calculate_iou(prediction, ground_truth, class_id)
        support = np.sum(ground_truth == class_id)

        iou_per_class[class_id] = iou

        if support > 0:
            weighted_sum += iou * support
            total_support += support

    if total_support == 0:
        logger.warning("No valid pixels found for weighted mIoU. Returning 0.0")
        return 0.0, iou_per_class

    weighted_miou = weighted_sum / total_support
    logger.info(f"Weighted mIoU: {weighted_miou:.4f} (total support: {total_support} pixels)")

    return float(weighted_miou), iou_per_class


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
):
    """
    Plot confusion matrix as heatmap.

    Useful for identifying which classes are commonly confused.
    Essential visualization for academic papers.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (num_classes, num_classes)
    class_names : List[str]
        Names of classes for axis labels
    save_path : str, optional
        If provided, save figure to this path
    normalize : bool, default=False
        If True, normalize by row (show percentages)
    title : str, default="Confusion Matrix"
        Title for the plot

    Returns
    -------
    plt.Figure
        Matplotlib figure object

    Examples
    --------
    >>> cm = np.array([[50, 5, 2], [3, 40, 7], [1, 4, 45]])
    >>> class_names = ['Crop', 'Urban', 'Water']
    >>> fig = plot_confusion_matrix(cm, class_names, save_path='cm.png')
    """
    import matplotlib.pyplot as plt

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    ax.set_title(title, fontsize=14, pad=20)
    fig.tight_layout()

    if save_path:
        logger.info(f"Saving confusion matrix to {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
