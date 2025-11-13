"""
Classification module for semantic land cover classification.

This module provides zero-shot classification capabilities for segmented regions
using NDVI and Prithvi embeddings.
"""

from src.classification.zero_shot_classifier import (
    SemanticClassifier,
    ClassificationResult,
    LAND_COVER_CLASSES,
    CLASS_COLORS,
    NDVI_THRESHOLDS,
    cross_validate_with_dynamic_world
)

__all__ = [
    "SemanticClassifier",
    "ClassificationResult",
    "LAND_COVER_CLASSES",
    "CLASS_COLORS",
    "NDVI_THRESHOLDS",
    "cross_validate_with_dynamic_world"
]
