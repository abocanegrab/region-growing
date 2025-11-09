"""
Classic Region Growing algorithm for image segmentation based on spectral homogeneity.

This module implements the traditional Region Growing algorithm using BFS (Breadth-First Search)
with 4-connectivity. The algorithm segments an image by grouping adjacent pixels with similar
spectral values (NDVI in this case).

Time Complexity: O(n) where n is the number of pixels
Space Complexity: O(n) for the labeled image and queue

References:
    - Adams, R., & Bischof, L. (1994). Seeded region growing.
      IEEE Transactions on Pattern Analysis and Machine Intelligence, 16(6), 641-647.
    - Mehnert, A., & Jackway, P. (1997). An improved seeded region growing algorithm.
      Pattern Recognition Letters, 18(10), 1065-1071.
"""
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional


class ClassicRegionGrowing:
    """
    Classic Region Growing algorithm for spectral-based image segmentation.

    This implementation uses BFS with 4-connectivity to grow regions from seed points
    based on a homogeneity criterion (spectral similarity threshold).

    Attributes:
        threshold: Maximum spectral difference to consider pixels as similar
        min_region_size: Minimum number of pixels for a valid region
        cloud_mask_value: Sentinel value for masked pixels (clouds, no-data)
    """

    def __init__(
        self,
        threshold: float = 0.1,
        min_region_size: int = 50,
        cloud_mask_value: float = -999.0
    ):
        """
        Initialize the Region Growing algorithm.

        Args:
            threshold: Homogeneity threshold for spectral similarity.
                      For NDVI: |NDVI_A - NDVI_B| < threshold
                      Typical values: 0.05-0.15 (default: 0.1)
                      Lower values = more strict, more regions
                      Higher values = more permissive, fewer regions
            min_region_size: Minimum region size in pixels to filter noise.
                            Typical values: 25-100 (default: 50)
                            At 10m resolution: 50 pixels = 5000 m² = 0.5 hectares
            cloud_mask_value: Value indicating masked pixels (clouds, no-data).
                             Default: -999.0 (sentinel value)
        """
        if threshold <= 0:
            raise ValueError("Threshold must be positive")
        if min_region_size < 1:
            raise ValueError("Minimum region size must be at least 1")

        self.threshold = threshold
        self.min_region_size = min_region_size
        self.cloud_mask_value = cloud_mask_value

    def segment(
        self,
        image: np.ndarray,
        seeds: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[np.ndarray, int, List[Dict]]:
        """
        Segment image using Region Growing algorithm.

        Args:
            image: 2D NumPy array with spectral values (e.g., NDVI).
                  Expected range: [-1, 1] for NDVI
                  Masked pixels should have value == cloud_mask_value
            seeds: List of seed coordinates [(y, x), ...].
                  If None, seeds are generated automatically using grid method.

        Returns:
            Tuple containing:
            - labeled_image: 2D array with region labels (0=no region, 1..N=region IDs)
            - num_regions: Total number of valid regions found
            - regions_info: List of dicts with information for each region:
                * id: Region ID (1-indexed)
                * size: Number of pixels
                * mean_ndvi: Mean spectral value
                * std_ndvi: Standard deviation
                * min_ndvi: Minimum value
                * max_ndvi: Maximum value
                * pixels: List of (y, x) coordinates

        Example:
            >>> algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=50)
            >>> ndvi_image = np.random.rand(256, 256)
            >>> labeled, num_regions, info = algorithm.segment(ndvi_image)
            >>> print(f"Found {num_regions} regions")
        """
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got shape {image.shape}")

        h, w = image.shape
        labeled = np.zeros((h, w), dtype=np.int32)
        region_id = 0
        regions_info = []

        # Generate seeds if not provided
        if seeds is None:
            seeds = self.generate_seeds_grid(image)

        # Grow region from each seed
        for seed_y, seed_x in seeds:
            # Skip if already labeled
            if labeled[seed_y, seed_x] != 0:
                continue

            # Skip if masked (cloud/no-data)
            if self._is_masked(image[seed_y, seed_x]):
                continue

            # Grow region from this seed
            region_pixels = self._grow_region(image, seed_y, seed_x, labeled, region_id + 1)

            # Validate region size
            if len(region_pixels) >= self.min_region_size:
                region_id += 1

                # Calculate region statistics
                region_values = [image[y, x] for y, x in region_pixels]
                region_info = {
                    'id': region_id,
                    'size': len(region_pixels),
                    'mean_ndvi': float(np.mean(region_values)),
                    'std_ndvi': float(np.std(region_values)),
                    'min_ndvi': float(np.min(region_values)),
                    'max_ndvi': float(np.max(region_values)),
                    'pixels': region_pixels
                }
                regions_info.append(region_info)
            else:
                # Region too small, remove labels
                for y, x in region_pixels:
                    labeled[y, x] = 0

        return labeled, region_id, regions_info

    def _grow_region(
        self,
        image: np.ndarray,
        start_y: int,
        start_x: int,
        labeled: np.ndarray,
        region_id: int
    ) -> List[Tuple[int, int]]:
        """
        Grow a region from a seed point using BFS (Breadth-First Search).

        This method implements the core Region Growing algorithm using BFS with 4-connectivity.
        It expands from the seed point by adding neighboring pixels that satisfy the
        homogeneity criterion.

        Args:
            image: 2D array with spectral values
            start_y: Y-coordinate of seed point
            start_x: X-coordinate of seed point
            labeled: 2D array to store region labels (modified in-place)
            region_id: ID to assign to this region

        Returns:
            List of pixel coordinates [(y, x), ...] belonging to the region

        Algorithm:
            1. Initialize queue with seed point
            2. While queue not empty:
                a. Pop pixel from queue
                b. If valid and similar to seed: add to region
                c. Add 4-connected neighbors to queue
            3. Return region pixels
        """
        h, w = image.shape
        seed_value = image[start_y, start_x]

        # BFS queue and tracking
        queue = deque([(start_y, start_x)])
        region_pixels = []
        visited = set()

        while queue:
            y, x = queue.popleft()

            # Skip if already visited
            if (y, x) in visited:
                continue

            # Check bounds
            if not (0 <= y < h and 0 <= x < w):
                continue

            # Skip if already labeled
            if labeled[y, x] != 0:
                continue

            # Skip masked pixels
            pixel_value = image[y, x]
            if self._is_masked(pixel_value):
                continue

            # Check homogeneity criterion
            if abs(pixel_value - seed_value) > self.threshold:
                continue

            # Add pixel to region
            visited.add((y, x))
            region_pixels.append((y, x))
            labeled[y, x] = region_id

            # Add 4-connected neighbors to queue
            queue.append((y + 1, x))  # Down
            queue.append((y - 1, x))  # Up
            queue.append((y, x + 1))  # Right
            queue.append((y, x - 1))  # Left

        return region_pixels

    def generate_seeds_grid(
        self,
        image: np.ndarray,
        grid_size: int = 20
    ) -> List[Tuple[int, int]]:
        """
        Generate seed points using a regular grid pattern.

        This method creates a grid of evenly spaced seed points across the image.
        Only valid (non-masked) pixels are selected as seeds.

        Args:
            image: 2D array with spectral values
            grid_size: Spacing between seeds in pixels (default: 20)
                      Smaller values = more seeds = finer segmentation
                      Larger values = fewer seeds = coarser segmentation

        Returns:
            List of seed coordinates [(y, x), ...]

        Example:
            For a 512x512 image with grid_size=20:
            - Seeds will be placed at (10, 10), (10, 30), ..., (10, 510)
            - Total seeds ≈ (512/20)² ≈ 650 seeds
        """
        h, w = image.shape
        seeds = []

        # Create grid starting from half grid_size offset
        for y in range(grid_size // 2, h, grid_size):
            for x in range(grid_size // 2, w, grid_size):
                # Only add valid pixels as seeds
                pixel_value = image[y, x]
                if not self._is_masked(pixel_value):
                    seeds.append((y, x))

        return seeds

    def classify_by_stress(
        self,
        regions_info: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Classify regions by vegetation stress level based on mean NDVI.

        NDVI stress level thresholds (standard values for crops):
        - High stress: NDVI < 0.3 (bare soil, senescent vegetation, severe stress)
        - Medium stress: 0.3 ≤ NDVI < 0.5 (moderate vegetation, moderate stress)
        - Low stress: NDVI ≥ 0.5 (healthy vegetation, low/no stress)

        Args:
            regions_info: List of region information dicts (from segment method)

        Returns:
            Dict with classified regions:
            {
                'high_stress': [region1, region2, ...],
                'medium_stress': [region3, region4, ...],
                'low_stress': [region5, region6, ...]
            }
            Each region dict gets an additional 'stress_level' field.
        """
        classified = {
            'high_stress': [],
            'medium_stress': [],
            'low_stress': []
        }

        for region in regions_info:
            mean_ndvi = region['mean_ndvi']

            if mean_ndvi < 0.3:
                region['stress_level'] = 'high'
                classified['high_stress'].append(region)
            elif mean_ndvi < 0.5:
                region['stress_level'] = 'medium'
                classified['medium_stress'].append(region)
            else:
                region['stress_level'] = 'low'
                classified['low_stress'].append(region)

        return classified

    def _is_masked(self, value: float) -> bool:
        """
        Check if a pixel value represents a masked area (cloud, no-data).

        Args:
            value: Pixel spectral value

        Returns:
            True if pixel is masked, False otherwise
        """
        # Check for NaN, Inf, and cloud mask value
        if np.isnan(value) or np.isinf(value):
            return True
        if abs(value - self.cloud_mask_value) < 1e-6:
            return True
        return False
