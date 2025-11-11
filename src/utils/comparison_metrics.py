"""
Comparison metrics for A/B analysis of segmentation methods.

This module provides functions to calculate quantitative metrics
for comparing Classic Region Growing vs MGRG segmentation results.
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SegmentationMetrics:
    """Container for segmentation metrics."""

    num_regions: int
    coherence: float
    avg_region_size: float
    std_region_size: float
    largest_region_size: int
    smallest_region_size: int
    processing_time: float


def calculate_spatial_coherence(segmentation: np.ndarray) -> float:
    """
    Calculate spatial coherence as percentage of labeled pixels.

    Parameters
    ----------
    segmentation : np.ndarray
        Segmentation mask with region IDs (0 = background)

    Returns
    -------
    float
        Coherence percentage [0-100]

    Examples
    --------
    >>> seg = np.ones((100, 100), dtype=int)
    >>> coherence = calculate_spatial_coherence(seg)
    >>> print(coherence)
    100.0
    """
    if segmentation.size == 0:
        logger.warning("Empty segmentation provided")
        return 0.0

    total_pixels = segmentation.size
    labeled_pixels = np.count_nonzero(segmentation)
    coherence = (labeled_pixels / total_pixels) * 100

    logger.debug(
        f"Coherence: {coherence:.2f}% ({labeled_pixels}/{total_pixels} pixels)"
    )

    return coherence


def count_regions(segmentation: np.ndarray) -> int:
    """
    Count number of unique regions in segmentation.

    Parameters
    ----------
    segmentation : np.ndarray
        Segmentation mask with region IDs

    Returns
    -------
    int
        Number of regions (excluding background=0)

    Examples
    --------
    >>> seg = np.zeros((100, 100), dtype=int)
    >>> seg[:50, :] = 1
    >>> seg[50:, :50] = 2
    >>> num_regions = count_regions(seg)
    >>> print(num_regions)
    2
    """
    unique_regions = np.unique(segmentation)
    num_regions = len(unique_regions[unique_regions != 0])

    logger.debug(f"Number of regions: {num_regions}")

    return num_regions


def calculate_region_statistics(segmentation: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive region statistics.

    Parameters
    ----------
    segmentation : np.ndarray
        Segmentation mask with region IDs

    Returns
    -------
    Dict[str, float]
        Dictionary with region statistics:
        - avg_size: Average region size in pixels
        - std_size: Standard deviation of region sizes
        - largest_size: Largest region size
        - smallest_size: Smallest region size

    Examples
    --------
    >>> seg = np.zeros((100, 100), dtype=int)
    >>> seg[:50, :] = 1
    >>> seg[50:, :] = 2
    >>> stats = calculate_region_statistics(seg)
    >>> print(stats['avg_size'])
    5000.0
    """
    unique_regions = np.unique(segmentation)
    unique_regions = unique_regions[unique_regions != 0]

    if len(unique_regions) == 0:
        logger.warning("No regions found in segmentation")
        return {
            "avg_size": 0.0,
            "std_size": 0.0,
            "largest_size": 0,
            "smallest_size": 0,
        }

    region_sizes = [
        np.sum(segmentation == region_id) for region_id in unique_regions
    ]

    stats = {
        "avg_size": float(np.mean(region_sizes)),
        "std_size": float(np.std(region_sizes)),
        "largest_size": int(np.max(region_sizes)),
        "smallest_size": int(np.min(region_sizes)),
    }

    logger.debug(
        f"Region stats: avg={stats['avg_size']:.0f}, "
        f"std={stats['std_size']:.0f}, "
        f"largest={stats['largest_size']}, "
        f"smallest={stats['smallest_size']}"
    )

    return stats


def compare_segmentations(
    classic_seg: np.ndarray,
    mgrg_seg: np.ndarray,
    classic_time: float,
    mgrg_time: float,
) -> Dict[str, SegmentationMetrics]:
    """
    Compare two segmentation results with comprehensive metrics.

    Parameters
    ----------
    classic_seg : np.ndarray
        Classic RG segmentation mask
    mgrg_seg : np.ndarray
        MGRG segmentation mask
    classic_time : float
        Processing time for classic method (seconds)
    mgrg_time : float
        Processing time for MGRG (seconds)

    Returns
    -------
    Dict[str, SegmentationMetrics]
        Dictionary with metrics for both methods:
        - 'classic': Metrics for Classic RG
        - 'mgrg': Metrics for MGRG
        - 'differences': Dictionary with metric differences

    Examples
    --------
    >>> classic_seg = np.random.randint(0, 10, (100, 100))
    >>> mgrg_seg = np.random.randint(0, 5, (100, 100))
    >>> metrics = compare_segmentations(classic_seg, mgrg_seg, 1.0, 1.5)
    >>> print(metrics['classic'].num_regions)
    9
    """
    logger.info("Starting segmentation comparison")

    # Validate inputs
    if classic_seg.shape != mgrg_seg.shape:
        raise ValueError(
            f"Segmentation shapes do not match: "
            f"{classic_seg.shape} vs {mgrg_seg.shape}"
        )

    # Calculate metrics for Classic RG
    classic_coherence = calculate_spatial_coherence(classic_seg)
    classic_num_regions = count_regions(classic_seg)
    classic_stats = calculate_region_statistics(classic_seg)

    classic_metrics = SegmentationMetrics(
        num_regions=classic_num_regions,
        coherence=classic_coherence,
        avg_region_size=classic_stats["avg_size"],
        std_region_size=classic_stats["std_size"],
        largest_region_size=classic_stats["largest_size"],
        smallest_region_size=classic_stats["smallest_size"],
        processing_time=classic_time,
    )

    # Calculate metrics for MGRG
    mgrg_coherence = calculate_spatial_coherence(mgrg_seg)
    mgrg_num_regions = count_regions(mgrg_seg)
    mgrg_stats = calculate_region_statistics(mgrg_seg)

    mgrg_metrics = SegmentationMetrics(
        num_regions=mgrg_num_regions,
        coherence=mgrg_coherence,
        avg_region_size=mgrg_stats["avg_size"],
        std_region_size=mgrg_stats["std_size"],
        largest_region_size=mgrg_stats["largest_size"],
        smallest_region_size=mgrg_stats["smallest_size"],
        processing_time=mgrg_time,
    )

    # Calculate differences
    differences = {
        "num_regions": mgrg_num_regions - classic_num_regions,
        "coherence": mgrg_coherence - classic_coherence,
        "avg_size": mgrg_stats["avg_size"] - classic_stats["avg_size"],
        "time": mgrg_time - classic_time,
    }

    # Determine winner based on coherence
    winner = "mgrg" if mgrg_coherence > classic_coherence else "classic"

    logger.info(
        f"Comparison complete. Winner: {winner} "
        f"(coherence diff: {differences['coherence']:+.2f}%)"
    )

    return {
        "classic": classic_metrics,
        "mgrg": mgrg_metrics,
        "differences": differences,
        "winner": winner,
    }


def calculate_boundary_precision(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> float:
    """
    Calculate boundary precision using IoU metric (optional).

    Parameters
    ----------
    predicted : np.ndarray
        Predicted segmentation mask
    ground_truth : np.ndarray
        Ground truth segmentation mask

    Returns
    -------
    float
        IoU score [0-1]

    Examples
    --------
    >>> pred = np.ones((100, 100), dtype=bool)
    >>> gt = np.ones((100, 100), dtype=bool)
    >>> iou = calculate_boundary_precision(pred, gt)
    >>> print(iou)
    1.0
    """
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: {predicted.shape} vs {ground_truth.shape}"
        )

    # Convert to boolean
    pred_bool = predicted.astype(bool)
    gt_bool = ground_truth.astype(bool)

    # Calculate IoU
    intersection = np.logical_and(pred_bool, gt_bool)
    union = np.logical_or(pred_bool, gt_bool)

    if np.sum(union) == 0:
        logger.warning("Empty union, returning 0 IoU")
        return 0.0

    iou = np.sum(intersection) / np.sum(union)

    logger.debug(f"IoU: {iou:.4f}")

    return float(iou)


def compare_processing_time(
    classic_fn: callable, mgrg_fn: callable, *args, **kwargs
) -> Dict[str, float]:
    """
    Compare processing time of two methods.

    Parameters
    ----------
    classic_fn : callable
        Classic Region Growing function
    mgrg_fn : callable
        MGRG function
    *args, **kwargs
        Arguments to pass to both functions

    Returns
    -------
    Dict[str, float]
        Dictionary with execution times in seconds:
        - 'classic_rg': Classic RG time
        - 'mgrg': MGRG time
        - 'speedup': Speedup factor (classic/mgrg)

    Examples
    --------
    >>> def classic_fn(x): time.sleep(0.1); return x
    >>> def mgrg_fn(x): time.sleep(0.15); return x
    >>> times = compare_processing_time(classic_fn, mgrg_fn, 10)
    >>> print(times['classic_rg'] < times['mgrg'])
    True
    """
    logger.info("Starting processing time comparison")

    # Classic RG
    start = time.time()
    classic_fn(*args, **kwargs)
    classic_time = time.time() - start

    # MGRG
    start = time.time()
    mgrg_fn(*args, **kwargs)
    mgrg_time = time.time() - start

    speedup = classic_time / mgrg_time if mgrg_time > 0 else 0

    logger.info(
        f"Processing times: Classic={classic_time:.2f}s, "
        f"MGRG={mgrg_time:.2f}s, Speedup={speedup:.2f}x"
    )

    return {
        "classic_rg": classic_time,
        "mgrg": mgrg_time,
        "speedup": speedup,
    }
