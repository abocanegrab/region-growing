"""
End-to-end hierarchical analysis pipeline.

Orchestrates all US components from Sentinel-2 download to stress analysis.
Shared logic between API REST and CLI script.

Pipeline Steps:
    1. Download Sentinel-2 HLS imagery
    2. Extract Prithvi embeddings
    3. Segment with MGRG
    4. Calculate NDVI
    5. Classify objects semantically
    6. Analyze stress levels (crops only)
    7. Generate outputs (JSON, GeoTIFF, PNG)

References:
    - US-003: Sentinel-2 download
    - US-006: Prithvi embeddings
    - US-007: MGRG segmentation
    - US-010: Semantic classification
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
import logging
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """
    Configuration for hierarchical analysis.

    Attributes
    ----------
    bbox : tuple
        Bounding box (min_lon, min_lat, max_lon, max_lat)
    date_from : str
        Start date in YYYY-MM-DD format
    date_to : str, optional
        End date (defaults to date_from)
    mgrg_threshold : float
        MGRG similarity threshold (0.7-0.99)
    min_region_size : int
        Minimum region size in pixels
    resolution : float
        Spatial resolution in meters
    output_dir : str
        Output directory path
    export_formats : list
        Export formats: ["json", "tif", "png", "html"]

    Examples
    --------
    >>> config = AnalysisConfig(
    ...     bbox=(-115.35, 32.45, -115.25, 32.55),
    ...     date_from="2025-10-15"
    ... )
    >>> print(config.date_to)
    '2025-10-15'
    """
    bbox: Tuple[float, float, float, float]
    date_from: str
    date_to: Optional[str] = None
    mgrg_threshold: float = 0.95
    min_region_size: int = 50
    resolution: float = 10.0
    output_dir: str = "output/analysis"
    export_formats: Optional[List[str]] = None

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["json", "tif", "png"]
        if self.date_to is None:
            self.date_to = self.date_from


@dataclass
class AnalysisResult:
    """
    Complete result of hierarchical analysis.

    Attributes
    ----------
    metadata : dict
        Analysis metadata (bbox, date, resolution, etc.)
    segmentation : dict
        Segmentation metrics (method, regions, coherence)
    classification : list
        List of classified regions with stats
    stress_analysis : dict
        Stress analysis results by level
    summary : dict
        Summary statistics by class
    output_files : dict
        Paths to generated files
    processing_time : dict
        Processing time per step
    """
    metadata: Dict
    segmentation: Dict
    classification: List[Dict]
    stress_analysis: Dict
    summary: Dict
    output_files: Dict
    processing_time: Dict


class HierarchicalAnalysisPipeline:
    """
    Complete pipeline for hierarchical land cover and stress analysis.

    This class orchestrates the complete analysis workflow from satellite imagery
    download to classified stress maps. It integrates all previous user stories
    into a cohesive, production-ready system.

    Parameters
    ----------
    config : AnalysisConfig
        Configuration object with all parameters

    Attributes
    ----------
    config : AnalysisConfig
        Analysis configuration
    output_dir : Path
        Output directory
    start_time : float
        Pipeline start timestamp
    step_times : dict
        Processing time per step

    Examples
    --------
    >>> config = AnalysisConfig(
    ...     bbox=(-115.35, 32.45, -115.25, 32.55),
    ...     date_from="2025-10-15"
    ... )
    >>> pipeline = HierarchicalAnalysisPipeline(config)
    >>> result = pipeline.run()
    >>> print(f"Total time: {result.processing_time['total']:.1f}s")
    Total time: 24.8s
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize timer
        self.start_time = time.time()
        self.step_times = {}

        # Validate configuration
        self._validate_config()

        logger.info("=" * 60)
        logger.info("HierarchicalAnalysisPipeline initialized")
        logger.info(f"BBox: {config.bbox}")
        logger.info(f"Date: {config.date_from}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Threshold: {config.mgrg_threshold}")
        logger.info("=" * 60)

    def _validate_config(self):
        """
        Validate configuration parameters.

        Raises
        ------
        ValueError
            If bbox or date is invalid
        """
        min_lon, min_lat, max_lon, max_lat = self.config.bbox

        # Validate longitude range
        if not (-180 <= min_lon < max_lon <= 180):
            raise ValueError(
                f"Invalid longitude range: {min_lon}, {max_lon}. "
                f"Must be in [-180, 180]"
            )

        # Validate latitude range
        if not (-90 <= min_lat < max_lat <= 90):
            raise ValueError(
                f"Invalid latitude range: {min_lat}, {max_lat}. "
                f"Must be in [-90, 90]"
            )

        # Validate bbox size (max 0.1° x 0.1° ~10km x 10km)
        if (max_lon - min_lon) > 0.1 or (max_lat - min_lat) > 0.1:
            raise ValueError(
                "BBox too large. Maximum size: 0.1° x 0.1° (~10km x 10km). "
                f"Current size: {max_lon - min_lon:.3f}° x {max_lat - min_lat:.3f}°"
            )

        # Validate date format
        try:
            datetime.strptime(self.config.date_from, "%Y-%m-%d")
            if self.config.date_to:
                datetime.strptime(self.config.date_to, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

        logger.info("Configuration validated successfully")

    def _time_step(self, step_name: str):
        """
        Record time for a pipeline step.

        Parameters
        ----------
        step_name : str
            Name of the step
        """
        elapsed = time.time() - self.start_time
        self.step_times[step_name] = round(elapsed, 2)
        logger.info(f"[{step_name}] Completed in {elapsed:.2f}s")

    def run(self) -> AnalysisResult:
        """
        Execute complete analysis pipeline.

        Returns
        -------
        AnalysisResult
            Complete results with all outputs

        Raises
        ------
        ValueError
            If bbox or date is invalid
        RuntimeError
            If any pipeline step fails

        Examples
        --------
        >>> pipeline = HierarchicalAnalysisPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Found {len(result.classification)} regions")
        Found 156 regions
        """
        logger.info("=" * 60)
        logger.info("Starting Hierarchical Analysis Pipeline")
        logger.info("=" * 60)

        try:
            # Step 1: Data Acquisition
            logger.info("\n[STEP 1/7] Data Acquisition")
            hls_data, metadata = self._download_sentinel2()
            self._time_step("download")

            # Step 2: Embedding Extraction
            logger.info("\n[STEP 2/7] Extracting Prithvi Embeddings")
            embeddings = self._extract_embeddings(hls_data)
            self._time_step("embeddings")

            # Step 3: Segmentation
            logger.info("\n[STEP 3/7] MGRG Segmentation")
            segmentation, seg_metrics = self._segment(embeddings)
            self._time_step("segmentation")

            # Step 4: NDVI Calculation
            logger.info("\n[STEP 4/7] Calculating NDVI")
            ndvi = self._calculate_ndvi(hls_data)
            self._time_step("ndvi")

            # Step 5: Classification
            logger.info("\n[STEP 5/7] Semantic Classification")
            classifications, semantic_map = self._classify(
                embeddings, ndvi, segmentation
            )
            self._time_step("classification")

            # Step 6: Stress Analysis
            logger.info("\n[STEP 6/7] Stress Analysis")
            stress_results = self._analyze_stress(classifications, ndvi, segmentation)
            self._time_step("stress")

            # Step 7: Output Generation
            logger.info("\n[STEP 7/7] Generating Output Files")
            output_files = self._generate_outputs(
                hls_data, segmentation, semantic_map,
                classifications, stress_results
            )
            self._time_step("output")

            # Compile final results
            result = self._compile_results(
                metadata, seg_metrics, classifications,
                stress_results, output_files
            )

            total_time = time.time() - self.start_time
            result.processing_time['total'] = round(total_time, 2)

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Pipeline completed successfully in {total_time:.2f}s")
            logger.info(f"{'=' * 60}")

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"Analysis pipeline failed: {e}")

    def _download_sentinel2(self) -> Tuple[np.ndarray, Dict]:
        """
        Download Sentinel-2 HLS data.

        Returns
        -------
        hls_data : np.ndarray
            HLS imagery (H, W, 6)
        metadata : dict
            Download metadata
        """
        from src.utils.sentinel_download import download_sentinel2_bands, create_sentinel_config
        import os

        # Get Sentinel Hub credentials
        client_id = os.getenv('SH_CLIENT_ID')
        client_secret = os.getenv('SH_CLIENT_SECRET')

        if not client_id or not client_secret:
            raise ValueError(
                "Sentinel Hub credentials not found. "
                "Set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables"
            )

        # Create config
        config = create_sentinel_config(client_id, client_secret)

        # Prepare bbox dict
        min_lon, min_lat, max_lon, max_lat = self.config.bbox
        bbox_dict = {
            'min_lat': min_lat,
            'min_lon': min_lon,
            'max_lat': max_lat,
            'max_lon': max_lon
        }

        # Download HLS bands
        logger.info("Downloading Sentinel-2 HLS bands...")
        data = download_sentinel2_bands(
            bbox_coords=bbox_dict,
            config=config,
            bands=['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'],
            date_from=self.config.date_from,
            date_to=self.config.date_to,
            resolution=int(self.config.resolution)
        )

        # Stack bands into HLS format
        from src.features.hls_processor import stack_hls_bands
        hls_data = stack_hls_bands(data['bands'])

        # Transpose to (H, W, 6)
        hls_data = np.transpose(hls_data, (1, 2, 0))

        logger.info(f"Downloaded HLS data: {hls_data.shape}")
        logger.info(f"Resolution: {self.config.resolution}m")

        return hls_data, data['metadata']

    def _extract_embeddings(self, hls_data: np.ndarray) -> np.ndarray:
        """
        Extract Prithvi embeddings.

        Parameters
        ----------
        hls_data : np.ndarray
            HLS imagery (H, W, 6)

        Returns
        -------
        embeddings : np.ndarray
            Semantic embeddings (H, W, 256)
        """
        from src.models.prithvi_loader import load_prithvi_model
        from src.features.hls_processor import prepare_hls_for_prithvi, extract_embeddings_from_hls

        # Load Prithvi model (cached after first call)
        logger.info("Loading Prithvi model...")
        model = load_prithvi_model()

        # Extract embeddings
        logger.info("Extracting embeddings...")
        embeddings = extract_embeddings_from_hls(model, hls_data)

        # Save embeddings
        emb_path = self.output_dir / "embeddings.npy"
        np.save(emb_path, embeddings)

        logger.info(f"Embeddings extracted: {embeddings.shape}")
        logger.info(f"Saved to: {emb_path}")

        return embeddings

    def _segment(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Run MGRG segmentation.

        Parameters
        ----------
        embeddings : np.ndarray
            Semantic embeddings (H, W, 256)

        Returns
        -------
        segmentation : np.ndarray
            Segmentation mask (H, W)
        metrics : dict
            Segmentation metrics
        """
        from src.algorithms.semantic_region_growing import SemanticRegionGrowing

        # Initialize MGRG
        mgrg = SemanticRegionGrowing(
            threshold=self.config.mgrg_threshold,
            min_region_size=self.config.min_region_size,
            use_smart_seeds=False  # Grid is superior (US-007)
        )

        # Run segmentation
        logger.info(f"Running MGRG (threshold={self.config.mgrg_threshold})...")
        labeled, num_regions, regions_info = mgrg.segment(embeddings)

        # Calculate metrics
        from src.utils.comparison_metrics import calculate_spatial_coherence

        coherence = calculate_spatial_coherence(labeled)

        metrics = {
            'method': 'MGRG',
            'threshold': self.config.mgrg_threshold,
            'regions': num_regions,
            'coherence': round(coherence, 2),
        }

        # Save segmentation
        seg_path = self.output_dir / "segmentation.npy"
        np.save(seg_path, labeled)

        logger.info(f"Segmentation: {num_regions} regions, {coherence:.1f}% coherence")
        logger.info(f"Saved to: {seg_path}")

        return labeled, metrics

    def _calculate_ndvi(self, hls_data: np.ndarray) -> np.ndarray:
        """
        Calculate NDVI from HLS bands.

        Parameters
        ----------
        hls_data : np.ndarray
            HLS imagery (H, W, 6)

        Returns
        -------
        ndvi : np.ndarray
            NDVI array (H, W)
        """
        # HLS bands: [B02, B03, B04, B8A, B11, B12]
        # NDVI = (NIR - Red) / (NIR + Red)
        # NIR = B8A (index 3), Red = B04 (index 2)
        nir = hls_data[:, :, 3]
        red = hls_data[:, :, 2]

        # Calculate NDVI
        denominator = nir + red
        denominator = np.where(denominator == 0, 0.0001, denominator)
        ndvi = (nir - red) / denominator

        # Save NDVI
        ndvi_path = self.output_dir / "ndvi.npy"
        np.save(ndvi_path, ndvi)

        logger.info(f"NDVI calculated: mean={ndvi.mean():.3f}, std={ndvi.std():.3f}")
        logger.info(f"Saved to: {ndvi_path}")

        return ndvi

    def _classify(
        self,
        embeddings: np.ndarray,
        ndvi: np.ndarray,
        segmentation: np.ndarray
    ) -> Tuple[Dict, np.ndarray]:
        """
        Classify all regions using US-010 SemanticClassifier.

        Parameters
        ----------
        embeddings : np.ndarray
            Semantic embeddings (H, W, 256)
        ndvi : np.ndarray
            NDVI array (H, W)
        segmentation : np.ndarray
            Segmentation mask (H, W)

        Returns
        -------
        classifications : dict
            Dictionary mapping region_id -> ClassificationResult
        semantic_map : np.ndarray
            Semantic map with class IDs (H, W)
        """
        from src.classification.zero_shot_classifier import SemanticClassifier

        classifier = SemanticClassifier(
            embeddings, ndvi, resolution=self.config.resolution
        )

        # Classify all regions
        classifications = classifier.classify_all_regions(
            segmentation,
            min_size=self.config.min_region_size
        )

        # Generate semantic map
        semantic_map = classifier.generate_semantic_map(
            segmentation, classifications
        )

        # Save semantic map
        sem_path = self.output_dir / "semantic_map.npy"
        np.save(sem_path, semantic_map)

        logger.info(f"Classified {len(classifications)} regions")
        logger.info(f"Classes found: {set(r.class_name for r in classifications.values())}")
        logger.info(f"Saved to: {sem_path}")

        return classifications, semantic_map

    def _analyze_stress(
        self,
        classifications: Dict,
        ndvi: np.ndarray,
        segmentation: np.ndarray
    ) -> Dict:
        """
        Analyze stress for crop regions only.

        Stress levels based on NDVI thresholds:
        - Low stress: NDVI >= 0.55 (healthy vegetation)
        - Medium stress: 0.40 <= NDVI < 0.55 (moderate stress)
        - High stress: NDVI < 0.40 (severe stress)

        Only analyzes regions classified as crops (class_id 3 or 4).

        Parameters
        ----------
        classifications : dict
            Region classifications
        ndvi : np.ndarray
            NDVI array
        segmentation : np.ndarray
            Segmentation mask

        Returns
        -------
        stress_results : dict
            Stress analysis by level
        """
        stress_results = {
            'low': {'count': 0, 'area_ha': 0.0, 'regions': []},
            'medium': {'count': 0, 'area_ha': 0.0, 'regions': []},
            'high': {'count': 0, 'area_ha': 0.0, 'regions': []},
        }

        crop_count = 0
        for region_id, result in classifications.items():
            # Only analyze crops (class_id 3: Vigorous Crop, 4: Stressed Crop)
            if result.class_id not in [3, 4]:
                continue

            crop_count += 1
            mean_ndvi = result.mean_ndvi

            # Assign stress level based on NDVI thresholds
            if mean_ndvi >= 0.55:
                level = 'low'
            elif mean_ndvi >= 0.40:
                level = 'medium'
            else:
                level = 'high'

            stress_results[level]['count'] += 1
            stress_results[level]['area_ha'] += result.area_hectares
            stress_results[level]['regions'].append({
                'region_id': region_id,
                'class_name': result.class_name,
                'ndvi': round(mean_ndvi, 3),
                'area_ha': round(result.area_hectares, 2)
            })

        logger.info(f"Stress analysis on {crop_count} crop regions: "
                   f"Low={stress_results['low']['count']}, "
                   f"Medium={stress_results['medium']['count']}, "
                   f"High={stress_results['high']['count']}")

        return stress_results

    def _generate_outputs(
        self,
        hls_data: np.ndarray,
        segmentation: np.ndarray,
        semantic_map: np.ndarray,
        classifications: Dict,
        stress_results: Dict
    ) -> Dict[str, str]:
        """
        Generate all output files.

        Parameters
        ----------
        hls_data : np.ndarray
            HLS imagery
        segmentation : np.ndarray
            Segmentation mask
        semantic_map : np.ndarray
            Semantic map
        classifications : dict
            Classifications
        stress_results : dict
            Stress results

        Returns
        -------
        output_files : dict
            Paths to generated files
        """
        output_files = {}

        # 1. JSON (always generated)
        json_path = self.output_dir / "analysis_results.json"
        self._save_json(classifications, stress_results, json_path)
        output_files['json'] = str(json_path)
        logger.info(f"Generated JSON: {json_path}")

        # 2. GeoTIFF (optional)
        if "tif" in self.config.export_formats:
            tif_path = self.output_dir / "semantic_map.tif"
            self._generate_geotiff(segmentation, semantic_map, tif_path)
            output_files['tif'] = str(tif_path)
            logger.info(f"Generated GeoTIFF: {tif_path}")

        # 3. PNG Visualization (optional)
        if "png" in self.config.export_formats:
            png_path = self.output_dir / "visualization.png"
            rgb = hls_data[:, :, [2, 1, 0]]  # B04, B03, B02 → RGB
            self._generate_visualization(
                rgb, semantic_map, classifications,
                stress_results, png_path
            )
            output_files['png'] = str(png_path)
            logger.info(f"Generated PNG: {png_path}")

        return output_files

    def _save_json(
        self,
        classifications: Dict,
        stress_results: Dict,
        output_path: Path
    ):
        """
        Save results as structured JSON.

        Parameters
        ----------
        classifications : dict
            Classifications
        stress_results : dict
            Stress results
        output_path : Path
            Output file path
        """
        # Convert classifications to list of dicts
        classification_list = [
            {
                'region_id': int(region_id),
                'class': result.class_name,
                'class_id': result.class_id,
                'confidence': round(result.confidence, 3),
                'area_ha': round(result.area_hectares, 2),
                'mean_ndvi': round(result.mean_ndvi, 3),
                'std_ndvi': round(result.std_ndvi, 3),
            }
            for region_id, result in classifications.items()
        ]

        # Calculate summary
        from collections import defaultdict
        summary = defaultdict(float)
        for result in classifications.values():
            # Extract English name only for summary keys
            class_name_en = result.class_name.split('(')[0].strip()
            key = f"{class_name_en.lower().replace(' ', '_').replace('/', '_')}_ha"
            summary[key] += result.area_hectares

        # Round summary values
        summary = {k: round(v, 2) for k, v in summary.items()}

        # Compile final JSON
        output_data = {
            'metadata': {
                'bbox': list(self.config.bbox),
                'date_from': self.config.date_from,
                'date_to': self.config.date_to,
                'resolution': self.config.resolution,
                'mgrg_threshold': self.config.mgrg_threshold,
            },
            'segmentation': {
                'method': 'MGRG',
                'regions': len(classifications),
            },
            'classification': classification_list,
            'stress_analysis': stress_results,
            'summary': dict(summary),
            'processing_time': self.step_times,
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    def _generate_geotiff(
        self,
        segmentation: np.ndarray,
        semantic_map: np.ndarray,
        output_path: Path
    ):
        """
        Generate GeoTIFF with 2 layers.

        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask
        semantic_map : np.ndarray
            Semantic map
        output_path : Path
            Output file path
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds

            # Create georeferencing transform
            min_lon, min_lat, max_lon, max_lat = self.config.bbox
            transform = from_bounds(
                min_lon, min_lat, max_lon, max_lat,
                segmentation.shape[1], segmentation.shape[0]
            )

            # Write GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=segmentation.shape[0],
                width=segmentation.shape[1],
                count=2,
                dtype=segmentation.dtype,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(segmentation, 1)
                dst.write(semantic_map, 2)
                dst.set_band_description(1, 'Segmentation')
                dst.set_band_description(2, 'Classification')

        except ImportError:
            logger.warning("rasterio not installed, skipping GeoTIFF generation")

    def _generate_visualization(
        self,
        rgb: np.ndarray,
        semantic_map: np.ndarray,
        classifications: Dict,
        stress_results: Dict,
        output_path: Path
    ):
        """
        Generate visualization PNG.

        Parameters
        ----------
        rgb : np.ndarray
            RGB image
        semantic_map : np.ndarray
            Semantic map
        classifications : dict
            Classifications
        stress_results : dict
            Stress results
        output_path : Path
            Output file path
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from src.classification.zero_shot_classifier import CLASS_COLORS

            # Create colored semantic map
            colored_map = np.zeros((*semantic_map.shape, 3), dtype=np.uint8)
            for class_id, color in CLASS_COLORS.items():
                mask = (semantic_map == class_id)
                colored_map[mask] = color

            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

            # Panel 1: RGB original
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title('Original RGB (Sentinel-2)', fontsize=12)
            axes[0, 0].axis('off')

            # Panel 2: Semantic map
            axes[0, 1].imshow(colored_map)
            axes[0, 1].set_title('Semantic Classification', fontsize=12)
            axes[0, 1].axis('off')

            # Panel 3: Class statistics
            axes[1, 0].axis('off')
            stats_text = "Class Statistics:\n\n"
            class_stats = {}
            for result in classifications.values():
                class_name = result.class_name
                if class_name not in class_stats:
                    class_stats[class_name] = {'count': 0, 'area': 0.0}
                class_stats[class_name]['count'] += 1
                class_stats[class_name]['area'] += result.area_hectares

            for class_name, stats in sorted(class_stats.items()):
                stats_text += f"{class_name}:\n"
                stats_text += f"  Regions: {stats['count']}\n"
                stats_text += f"  Area: {stats['area']:.1f} ha\n\n"

            axes[1, 0].text(0.1, 0.9, stats_text,
                          transform=axes[1, 0].transAxes,
                          verticalalignment='top',
                          fontfamily='monospace',
                          fontsize=10)

            # Panel 4: Stress statistics
            axes[1, 1].axis('off')
            stress_text = "Stress Analysis (Crops Only):\n\n"
            for level in ['low', 'medium', 'high']:
                data = stress_results[level]
                stress_text += f"{level.capitalize()} Stress:\n"
                stress_text += f"  Regions: {data['count']}\n"
                stress_text += f"  Area: {data['area_ha']:.1f} ha\n\n"

            axes[1, 1].text(0.1, 0.9, stress_text,
                          transform=axes[1, 1].transAxes,
                          verticalalignment='top',
                          fontfamily='monospace',
                          fontsize=10)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        except ImportError:
            logger.warning("matplotlib not installed, skipping visualization")

    def _compile_results(
        self,
        metadata: Dict,
        seg_metrics: Dict,
        classifications: Dict,
        stress_results: Dict,
        output_files: Dict
    ) -> AnalysisResult:
        """
        Compile all results into AnalysisResult object.

        Parameters
        ----------
        metadata : dict
            Download metadata
        seg_metrics : dict
            Segmentation metrics
        classifications : dict
            Classifications
        stress_results : dict
            Stress results
        output_files : dict
            Output file paths

        Returns
        -------
        AnalysisResult
            Complete analysis results
        """
        # Convert classifications to list
        classification_list = [
            {
                'region_id': int(region_id),
                'class': result.class_name,
                'class_id': result.class_id,
                'confidence': round(result.confidence, 3),
                'area_ha': round(result.area_hectares, 2),
                'mean_ndvi': round(result.mean_ndvi, 3),
                'std_ndvi': round(result.std_ndvi, 3),
            }
            for region_id, result in classifications.items()
        ]

        # Calculate summary
        from collections import defaultdict
        summary = defaultdict(float)
        for result in classifications.values():
            class_name_en = result.class_name.split('(')[0].strip()
            key = f"{class_name_en.lower().replace(' ', '_').replace('/', '_')}_ha"
            summary[key] += result.area_hectares

        summary = {k: round(v, 2) for k, v in summary.items()}

        return AnalysisResult(
            metadata={
                'bbox': list(self.config.bbox),
                'date_from': self.config.date_from,
                'date_to': self.config.date_to,
                'resolution': self.config.resolution,
                'mgrg_threshold': self.config.mgrg_threshold,
            },
            segmentation=seg_metrics,
            classification=classification_list,
            stress_analysis=stress_results,
            summary=dict(summary),
            output_files=output_files,
            processing_time=self.step_times
        )
