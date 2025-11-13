"""
Zero-shot semantic classification for segmented regions.

Uses NDVI and Prithvi embeddings to classify land cover types without training.

Supported classes:
    0: Water
    1: Urban / Built Area
    2: Bare Soil / Fallow
    3: Vigorous Crop (NDVI > 0.6)
    4: Stressed Crop (0.3 < NDVI < 0.6)
    5: Grass / Shrub

References:
    - Brown et al. (2022). Dynamic World for validation
    - Muhtar et al. (2024). Prithvi-EO-2.0 embeddings
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Land cover class definitions (English/Spanish)
LAND_COVER_CLASSES = {
    0: "Water (Agua)",
    1: "Urban (Urbano)",
    2: "Bare Soil (Suelo Desnudo)",
    3: "Vigorous Crop (Cultivo Vigoroso)",
    4: "Stressed Crop (Cultivo Estresado)",
    5: "Grass/Shrub (Pasto/Arbustos)",
}

# Color scheme for visualization (RGB)
CLASS_COLORS = {
    0: (0, 119, 190),      # Blue
    1: (128, 128, 128),    # Gray
    2: (139, 69, 19),      # Brown
    3: (34, 139, 34),      # Dark Green
    4: (154, 205, 50),     # Yellow-Green
    5: (144, 238, 144),    # Light Green
}

# NDVI thresholds for classification
NDVI_THRESHOLDS = {
    'water_urban': 0.1,     # Below this: Water or Urban
    'bare_soil': 0.3,       # 0.1-0.3: Bare Soil
    'stressed_crop': 0.6,   # 0.3-0.6: Stressed Crop
    # Above 0.6: Vigorous Crop
}


@dataclass
class ClassificationResult:
    """
    Result of classifying a single region.

    Attributes
    ----------
    class_id : int
        Predicted class ID (0-5)
    class_name : str
        Human-readable class name
    confidence : float
        Classification confidence [0.0, 1.0]
    mean_ndvi : float
        Mean NDVI of the region
    std_ndvi : float
        Standard deviation of NDVI
    size_pixels : int
        Number of pixels in region
    area_hectares : float
        Area in hectares (assuming 10m resolution)
    """
    class_id: int
    class_name: str
    confidence: float
    mean_ndvi: float
    std_ndvi: float
    size_pixels: int
    area_hectares: float


class SemanticClassifier:
    """
    Zero-shot semantic classifier for land cover.

    Uses hierarchical classification:
        Level 1: Coarse class (Water, Urban, Crops, etc.)
        Level 2: Stress analysis (only for crops)

    Parameters
    ----------
    embeddings : np.ndarray
        Prithvi embeddings (H, W, 256)
    ndvi : np.ndarray
        NDVI array (H, W) with values in [-1, 1]
    resolution : float, default=10.0
        Spatial resolution in meters (for area calculation)

    Examples
    --------
    >>> embeddings = np.load('embeddings.npy')  # (1124, 922, 256)
    >>> ndvi = np.load('ndvi.npy')  # (1124, 922)
    >>> segmentation = np.load('mgrg_seg.npy')  # (1124, 922)
    >>>
    >>> classifier = SemanticClassifier(embeddings, ndvi)
    >>> results = classifier.classify_all_regions(segmentation)
    >>>
    >>> # Access classification for region 5
    >>> print(results[5].class_name)
    'Vigorous Crop'
    >>> print(f"Area: {results[5].area_hectares:.1f} ha")
    Area: 124.5 ha
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        ndvi: np.ndarray,
        resolution: float = 10.0
    ):
        assert embeddings.shape[:2] == ndvi.shape, \
            f"Shape mismatch: embeddings {embeddings.shape[:2]} vs ndvi {ndvi.shape}"

        self.embeddings = embeddings
        self.ndvi = ndvi
        self.resolution = resolution
        self.h, self.w = ndvi.shape

        logger.info(f"SemanticClassifier initialized. Shape: {self.h}x{self.w}, "
                   f"Resolution: {resolution}m")

    def classify_region(self, region_mask: np.ndarray) -> ClassificationResult:
        """
        Classify a single region using hierarchical approach.

        Parameters
        ----------
        region_mask : np.ndarray
            Binary mask (H, W) where True indicates pixels in region

        Returns
        -------
        ClassificationResult
            Classification with class, confidence, and statistics

        Examples
        --------
        >>> mask = (segmentation == 45)
        >>> result = classifier.classify_region(mask)
        >>> print(f"{result.class_name}: {result.confidence:.2f}")
        Vigorous Crop: 0.87
        """
        assert region_mask.shape == self.ndvi.shape, "Mask shape mismatch"
        assert region_mask.dtype == bool, "Mask must be boolean"

        # Extract region statistics
        region_ndvi = self.ndvi[region_mask]
        region_embs = self.embeddings[region_mask]

        mean_ndvi = float(region_ndvi.mean())
        std_ndvi = float(region_ndvi.std())
        size_pixels = int(region_mask.sum())

        # Calculate area (resolution in meters)
        area_m2 = size_pixels * (self.resolution ** 2)
        area_hectares = area_m2 / 10000.0  # m² to hectares

        # Level 1: Coarse classification based on NDVI
        class_id, confidence = self._classify_coarse(
            mean_ndvi, std_ndvi, region_embs
        )

        class_name = LAND_COVER_CLASSES[class_id]

        return ClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            mean_ndvi=mean_ndvi,
            std_ndvi=std_ndvi,
            size_pixels=size_pixels,
            area_hectares=area_hectares
        )

    def _classify_coarse(
        self,
        mean_ndvi: float,
        std_ndvi: float,
        embeddings: np.ndarray
    ) -> Tuple[int, float]:
        """
        Classify into coarse categories using NDVI thresholds.

        Returns
        -------
        class_id : int
            Predicted class (0-5)
        confidence : float
            Classification confidence [0.0, 1.0]
        """
        # Very low NDVI: Water or Urban
        if mean_ndvi < NDVI_THRESHOLDS['water_urban']:
            # Distinguish Water vs Urban using embedding similarity
            # Simple heuristic: Urban has higher variability
            if std_ndvi > 0.05:  # Urban areas have more variation
                return 1, 0.75  # Urban
            else:
                return 0, 0.85  # Water

        # Low NDVI: Bare Soil
        elif mean_ndvi < NDVI_THRESHOLDS['bare_soil']:
            confidence = 1.0 - abs(mean_ndvi - 0.2) / 0.2  # Peak at 0.2
            return 2, max(0.6, confidence)  # Bare Soil

        # Medium NDVI: Stressed Crop
        elif mean_ndvi < NDVI_THRESHOLDS['stressed_crop']:
            confidence = 1.0 - abs(mean_ndvi - 0.45) / 0.15  # Peak at 0.45
            return 4, max(0.65, confidence)  # Stressed Crop

        # High NDVI: Vigorous Crop or Grass/Shrub
        else:
            # Crops typically have std_ndvi < 0.1 (uniform fields)
            # Natural vegetation has higher variability
            if std_ndvi < 0.1:
                confidence = 0.85 + (mean_ndvi - 0.6) * 0.25  # Higher for NDVI~0.8
                return 3, min(0.95, confidence)  # Vigorous Crop
            else:
                return 5, 0.70  # Grass/Shrub

    def classify_all_regions(
        self,
        segmentation: np.ndarray,
        min_size: int = 10
    ) -> Dict[int, ClassificationResult]:
        """
        Classify all regions in a segmentation.

        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask (H, W) with integer region IDs
        min_size : int, default=10
            Skip regions smaller than this (pixels)

        Returns
        -------
        dict
            Mapping region_id -> ClassificationResult

        Examples
        --------
        >>> results = classifier.classify_all_regions(mgrg_seg)
        >>> print(f"Classified {len(results)} regions")
        Classified 156 regions
        """
        assert segmentation.shape == self.ndvi.shape, "Segmentation shape mismatch"

        region_ids = np.unique(segmentation)
        region_ids = region_ids[region_ids != 0]  # Exclude background

        logger.info(f"Classifying {len(region_ids)} regions...")

        results = {}
        skipped = 0

        for region_id in region_ids:
            mask = (segmentation == region_id)

            # Skip small regions
            if mask.sum() < min_size:
                skipped += 1
                continue

            result = self.classify_region(mask)
            results[int(region_id)] = result

        logger.info(f"Classification complete. {len(results)} regions classified, "
                   f"{skipped} skipped (too small)")

        return results

    def generate_semantic_map(
        self,
        segmentation: np.ndarray,
        classifications: Dict[int, ClassificationResult]
    ) -> np.ndarray:
        """
        Generate semantic map from segmentation and classifications.

        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask (H, W)
        classifications : dict
            Region classifications from classify_all_regions()

        Returns
        -------
        np.ndarray
            Semantic map (H, W) with class IDs (0-5)

        Examples
        --------
        >>> semantic_map = classifier.generate_semantic_map(
        ...     segmentation, classifications
        ... )
        >>> plt.imshow(semantic_map, cmap='tab10')
        """
        semantic_map = np.zeros(segmentation.shape, dtype=np.uint8)

        for region_id, result in classifications.items():
            mask = (segmentation == region_id)
            semantic_map[mask] = result.class_id

        logger.info(f"Semantic map generated. Shape: {semantic_map.shape}")
        return semantic_map

    def generate_colored_map(
        self,
        semantic_map: np.ndarray
    ) -> np.ndarray:
        """
        Generate RGB colored map from semantic map.

        Parameters
        ----------
        semantic_map : np.ndarray
            Semantic map (H, W) with class IDs

        Returns
        -------
        np.ndarray
            RGB image (H, W, 3) with class colors
        """
        h, w = semantic_map.shape
        colored_map = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in CLASS_COLORS.items():
            mask = (semantic_map == class_id)
            colored_map[mask] = color

        return colored_map

    def get_class_statistics(
        self,
        classifications: Dict[int, ClassificationResult]
    ) -> Dict[str, Dict]:
        """
        Calculate statistics per class.

        Returns
        -------
        dict
            Statistics per class: count, total_area, mean_ndvi, etc.

        Examples
        --------
        >>> stats = classifier.get_class_statistics(classifications)
        >>> print(stats['Vigorous Crop'])
        {'count': 67, 'area_ha': 1245.8, 'mean_ndvi': 0.72, ...}
        """
        stats = {name: {
            'count': 0,
            'area_ha': 0.0,
            'mean_ndvi': [],
        } for name in LAND_COVER_CLASSES.values()}

        for result in classifications.values():
            class_name = result.class_name
            stats[class_name]['count'] += 1
            stats[class_name]['area_ha'] += result.area_hectares
            stats[class_name]['mean_ndvi'].append(result.mean_ndvi)

        # Calculate mean NDVI per class
        for class_name in stats:
            if stats[class_name]['mean_ndvi']:
                ndvi_values = stats[class_name]['mean_ndvi']
                stats[class_name]['mean_ndvi'] = float(np.mean(ndvi_values))
                stats[class_name]['std_ndvi'] = float(np.std(ndvi_values))
            else:
                stats[class_name]['mean_ndvi'] = 0.0
                stats[class_name]['std_ndvi'] = 0.0

        return stats


def cross_validate_with_dynamic_world(
    our_semantic_map: np.ndarray,
    dynamic_world_mask: np.ndarray
) -> Dict[str, float]:
    """
    Cross-validate our classification against Dynamic World.

    Maps our classes to Dynamic World classes and calculates agreement.

    Parameters
    ----------
    our_semantic_map : np.ndarray
        Our semantic map (H, W) with class IDs 0-5
    dynamic_world_mask : np.ndarray
        Dynamic World mask (H, W) with DW class IDs 0-8

    Returns
    -------
    dict
        Agreement per class and overall agreement

    Examples
    --------
    >>> agreements = cross_validate_with_dynamic_world(
    ...     semantic_map, dw_mask
    ... )
    >>> print(f"Overall agreement: {agreements['overall']:.1%}")
    Overall agreement: 73.4%
    """
    # Mapping: Our classes → Dynamic World classes
    class_mapping = {
        0: [0],           # Water → Water
        1: [6],           # Urban → Built Area
        2: [7],           # Bare Soil → Bare Ground
        3: [4],           # Vigorous Crop → Crops
        4: [4],           # Stressed Crop → Crops
        5: [1, 2, 3],     # Grass/Shrub → Trees/Grass/Flooded Veg
    }

    agreements = {}
    total_correct = 0
    total_pixels = 0

    for our_class_id, dw_class_ids in class_mapping.items():
        our_mask = (our_semantic_map == our_class_id)
        dw_mask = np.isin(dynamic_world_mask, dw_class_ids)

        if our_mask.sum() == 0:
            agreements[LAND_COVER_CLASSES[our_class_id]] = 0.0
            continue

        correct = np.logical_and(our_mask, dw_mask).sum()
        total = our_mask.sum()

        agreement = correct / total
        agreements[LAND_COVER_CLASSES[our_class_id]] = float(agreement)

        total_correct += correct
        total_pixels += total

    # Overall agreement
    agreements['overall'] = total_correct / total_pixels if total_pixels > 0 else 0.0

    return agreements
