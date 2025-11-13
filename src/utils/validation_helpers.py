"""
Helper functions for validation notebooks.

This module contains utility functions used in validation notebooks
to load data, validate zones, and display results.
"""
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.utils.validation_metrics import (
    calculate_iou,
    calculate_miou,
    calculate_weighted_miou,
    calculate_f1_score,
    calculate_precision_recall,
    calculate_pixel_accuracy,
    generate_confusion_matrix,
    plot_confusion_matrix,
)
from src.utils.dynamic_world_downloader import load_or_generate_ground_truth

logger = logging.getLogger(__name__)


def load_zone_data(
    zone_name: str,
    processed_path: Path,
    dynamic_world_path: Path,
    use_synthetic: bool = False,
    prefer_esa: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Load segmentations and ground truth for a zone.

    Parameters
    ----------
    zone_name : str
        Name of the zone (e.g., 'mexicali', 'bajio', 'sinaloa')
    processed_path : Path
        Path to processed data directory
    dynamic_world_path : Path
        Path to Dynamic World data directory
    use_synthetic : bool, default=False
        If True, force use of synthetic ground truth
    prefer_esa : bool, default=False
        If True, prefer ESA WorldCover over Dynamic World

    Returns
    -------
    tuple
        (classic_seg, mgrg_seg, gt_mask, ndvi, is_synthetic)
        - classic_seg: Classic RG segmentation
        - mgrg_seg: MGRG segmentation
        - gt_mask: Ground truth mask
        - ndvi: NDVI array
        - is_synthetic: True if using synthetic ground truth

    Examples
    --------
    >>> from pathlib import Path
    >>> processed = Path('data/processed')
    >>> dw = Path('data/dynamic_world')
    >>> classic, mgrg, gt, ndvi, synth = load_zone_data(
    ...     'mexicali', processed, dw
    ... )
    >>> print(f"Loaded {classic.shape}, synthetic={synth}")
    """
    logger.info(f"Loading data for: {zone_name.upper()}")

    zone_dir = processed_path / zone_name

    # Load real segmentations
    classic_seg = np.load(zone_dir / "classic_rg_segmentation.npy")
    mgrg_seg = np.load(zone_dir / "mgrg_segmentation.npy")
    ndvi = np.load(zone_dir / "ndvi.npy")

    logger.info(
        f"Classic RG: {classic_seg.shape}, {len(np.unique(classic_seg))} regions"
    )
    logger.info(f"MGRG: {mgrg_seg.shape}, {len(np.unique(mgrg_seg))} regions")
    logger.info(f"NDVI: {ndvi.shape}")

    # Load or generate ground truth
    gt_mask, is_synthetic = load_or_generate_ground_truth(
        zone_name, ndvi, dynamic_world_path, use_synthetic, prefer_esa
    )

    gt_type = "synthetic" if is_synthetic else "real (ESA/Dynamic World)"
    logger.info(f"Ground truth: {gt_mask.shape} ({gt_type})")

    return classic_seg, mgrg_seg, gt_mask, ndvi, is_synthetic


def validate_zone(
    zone_name: str,
    classic_seg: np.ndarray,
    mgrg_seg: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Validate segmentation results for a zone against ground truth.

    Calculates all standard metrics (IoU, mIoU, F1, Precision, Recall, etc.)
    for both Classic RG and MGRG methods.

    Parameters
    ----------
    zone_name : str
        Name of the zone (e.g., 'mexicali')
    classic_seg : np.ndarray
        Classic RG segmentation mask
    mgrg_seg : np.ndarray
        MGRG segmentation mask
    gt_mask : np.ndarray
        Ground truth mask
    num_classes : int, default=5
        Number of classes
    verbose : bool, default=True
        If True, log detailed metrics

    Returns
    -------
    dict
        Dictionary with all metrics for both methods:
        {
            'zone': str,
            'classic_rg': {metrics},
            'mgrg': {metrics}
        }

    Examples
    --------
    >>> results = validate_zone('mexicali', classic, mgrg, gt)
    >>> print(f"MGRG mIoU: {results['mgrg']['miou']:.4f}")
    """
    if verbose:
        logger.info(f"Validating zone: {zone_name.upper()}")

    results = {"zone": zone_name}

    for method_name, seg_mask in [("Classic RG", classic_seg), ("MGRG", mgrg_seg)]:
        # IoU and mIoU
        miou, iou_per_class = calculate_miou(
            seg_mask, gt_mask, num_classes, ignore_background=True
        )

        # Weighted mIoU
        weighted_miou, _ = calculate_weighted_miou(
            seg_mask, gt_mask, num_classes, ignore_background=True
        )

        # F1-Score per class
        f1_scores = {}
        for class_id in range(1, num_classes):
            f1 = calculate_f1_score(seg_mask, gt_mask, class_id)
            f1_scores[class_id] = f1
        macro_f1 = np.mean(list(f1_scores.values()))

        # Precision and Recall per class
        precision_scores = {}
        recall_scores = {}
        for class_id in range(1, num_classes):
            prec, rec = calculate_precision_recall(seg_mask, gt_mask, class_id)
            precision_scores[class_id] = prec
            recall_scores[class_id] = rec
        macro_precision = np.mean(list(precision_scores.values()))
        macro_recall = np.mean(list(recall_scores.values()))

        # Pixel Accuracy
        pixel_acc = calculate_pixel_accuracy(seg_mask, gt_mask)

        # Confusion Matrix
        cm = generate_confusion_matrix(seg_mask, gt_mask, num_classes)
        
        # Number of regions
        num_regions = len(np.unique(seg_mask))

        # Store results
        method_key = method_name.lower().replace(" ", "_")
        results[method_key] = {
            "miou": miou,
            "weighted_miou": weighted_miou,
            "iou_per_class": iou_per_class,
            "macro_f1": macro_f1,
            "f1_per_class": f1_scores,
            "macro_precision": macro_precision,
            "precision_per_class": precision_scores,
            "macro_recall": macro_recall,
            "recall_per_class": recall_scores,
            "pixel_accuracy": pixel_acc,
            "confusion_matrix": cm,
            "num_regions": num_regions,
        }

    return results


def display_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Display confusion matrix in Jupyter notebook.

    This function creates and displays a confusion matrix without saving to disk.
    Optimized for Jupyter notebook display.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (num_classes, num_classes)
    class_names : list
        Names of classes for axis labels
    title : str, default="Confusion Matrix"
        Title for the plot
    normalize : bool, default=False
        If True, normalize by row (show percentages)
    figsize : tuple, default=(10, 8)
        Figure size (width, height)

    Returns
    -------
    plt.Figure
        Matplotlib figure object (already displayed)

    Examples
    --------
    >>> cm = np.array([[50, 5], [3, 40]])
    >>> fig = display_confusion_matrix(cm, ['Crop', 'Urban'])
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)

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

    # Display immediately
    plt.show()

    return fig


def display_zone_metrics(results: Dict, zone_name: str) -> None:
    """
    Display validation metrics for a zone in a nice DataFrame format.

    Parameters
    ----------
    results : dict
        Results dictionary from validate_zone()
    zone_name : str
        Name of the zone

    Examples
    --------
    >>> results = validate_zone('mexicali', classic, mgrg, gt)
    >>> display_zone_metrics(results, 'mexicali')
    """
    import pandas as pd
    from IPython.display import display

    print(f"\n{'='*70}")
    print(f"METRICAS DE VALIDACION - {zone_name.upper()}")
    print(f"{'='*70}\n")

    # Main metrics table
    metrics_data = []
    for method_key in ["classic_rg", "mgrg"]:
        data = results[method_key]
        method_name = "Classic RG" if method_key == "classic_rg" else "MGRG"
        metrics_data.append(
            {
                "Metodo": method_name,
                "Regiones": data["num_regions"],
                "mIoU": f"{data['miou']:.4f}",
                "Weighted mIoU": f"{data['weighted_miou']:.4f}",
                "F1-Score": f"{data['macro_f1']:.4f}",
                "Precision": f"{data['macro_precision']:.4f}",
                "Recall": f"{data['macro_recall']:.4f}",
                "Pixel Acc": f"{data['pixel_accuracy']:.4f}",
            }
        )

    df_metrics = pd.DataFrame(metrics_data)
    display(df_metrics)

    # Calculate improvement
    classic_miou = results["classic_rg"]["miou"]
    mgrg_miou = results["mgrg"]["miou"]
    if classic_miou > 0:
        improvement = ((mgrg_miou - classic_miou) / classic_miou) * 100
        print(f"\nMejora de MGRG sobre Classic RG: {improvement:+.1f}%")
    else:
        print(f"\nMGRG mIoU: {mgrg_miou:.4f} (Classic RG: {classic_miou:.4f})")

    print()


def create_summary_table(all_results: Dict) -> "pd.DataFrame":
    """
    Create a summary table with all zones and methods.

    Parameters
    ----------
    all_results : dict
        Dictionary with results for all zones

    Returns
    -------
    pd.DataFrame
        Summary table with all metrics

    Examples
    --------
    >>> df = create_summary_table(all_results)
    >>> display(df)
    """
    import pandas as pd

    rows = []
    for zone_name, results in all_results.items():
        for method_key in ["classic_rg", "mgrg"]:
            data = results[method_key]
            method_name = "Classic RG" if method_key == "classic_rg" else "MGRG"
            rows.append(
                {
                    "Zona": zone_name.capitalize(),
                    "Metodo": method_name,
                    "Regiones": data["num_regions"],
                    "mIoU": float(data["miou"]),
                    "Weighted mIoU": float(data["weighted_miou"]),
                    "F1-Score": float(data["macro_f1"]),
                    "Precision": float(data["macro_precision"]),
                    "Recall": float(data["macro_recall"]),
                    "Pixel Acc": float(data["pixel_accuracy"]),
                }
            )

    df = pd.DataFrame(rows)

    # Format numeric columns
    for col in ["mIoU", "Weighted mIoU", "F1-Score", "Precision", "Recall", "Pixel Acc"]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")

    return df
