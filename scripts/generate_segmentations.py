"""
Generate segmentations for validation using real Sentinel-2 data.

This script generates Classic RG and MGRG segmentations for the three zones
using the existing Sentinel-2 data and Prithvi embeddings.

Usage:
    poetry run python scripts/generate_segmentations.py
"""

import sys
sys.path.append('..')

import numpy as np
from pathlib import Path
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Calculate NDVI from NIR and Red bands."""
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi


def classic_region_growing(
    ndvi: np.ndarray,
    threshold: float = 0.1,
    min_size: int = 50
) -> np.ndarray:
    """
    Simple Classic Region Growing based on NDVI homogeneity.
    
    Parameters
    ----------
    ndvi : np.ndarray
        NDVI image (H, W)
    threshold : float
        Homogeneity threshold
    min_size : int
        Minimum region size in pixels
        
    Returns
    -------
    np.ndarray
        Segmentation mask (H, W) with region IDs
    """
    h, w = ndvi.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    region_id = 1
    
    # Generate seeds (grid 20x20)
    seeds = []
    step = 20
    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            seeds.append((y, x))
    
    logger.info(f"Generated {len(seeds)} seeds for Classic RG")
    
    for seed_y, seed_x in seeds:
        if labeled[seed_y, seed_x] != 0:
            continue
        
        seed_value = ndvi[seed_y, seed_x]
        
        # BFS
        queue = [(seed_y, seed_x)]
        region_pixels = []
        
        while queue:
            y, x = queue.pop(0)
            
            if not (0 <= y < h and 0 <= x < w):
                continue
            if labeled[y, x] != 0:
                continue
            
            pixel_value = ndvi[y, x]
            if abs(pixel_value - seed_value) <= threshold:
                labeled[y, x] = region_id
                region_pixels.append((y, x))
                
                # 4-connectivity
                queue.extend([
                    (y-1, x), (y+1, x),
                    (y, x-1), (y, x+1)
                ])
        
        # Filter small regions
        if len(region_pixels) >= min_size:
            region_id += 1
        else:
            for y, x in region_pixels:
                labeled[y, x] = 0
    
    logger.info(f"Classic RG: {region_id - 1} regions")
    return labeled


def mgrg_segmentation(
    embeddings: np.ndarray,
    threshold: float = 0.85,
    min_size: int = 50
) -> np.ndarray:
    """
    MGRG (Metric-Guided Region Growing) using semantic embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Prithvi embeddings (H, W, 256)
    threshold : float
        Cosine similarity threshold
    min_size : int
        Minimum region size
        
    Returns
    -------
    np.ndarray
        Segmentation mask (H, W)
    """
    h, w, d = embeddings.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    region_id = 1
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-8)
    
    # Generate seeds
    seeds = []
    step = 20
    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            seeds.append((y, x))
    
    logger.info(f"Generated {len(seeds)} seeds for MGRG")
    
    for seed_y, seed_x in seeds:
        if labeled[seed_y, seed_x] != 0:
            continue
        
        seed_emb = embeddings_norm[seed_y, seed_x]
        
        # BFS
        queue = [(seed_y, seed_x)]
        region_pixels = []
        
        while queue:
            y, x = queue.pop(0)
            
            if not (0 <= y < h and 0 <= x < w):
                continue
            if labeled[y, x] != 0:
                continue
            
            pixel_emb = embeddings_norm[y, x]
            similarity = np.dot(seed_emb, pixel_emb)
            
            if similarity >= threshold:
                labeled[y, x] = region_id
                region_pixels.append((y, x))
                
                queue.extend([
                    (y-1, x), (y+1, x),
                    (y, x-1), (y, x+1)
                ])
        
        if len(region_pixels) >= min_size:
            region_id += 1
        else:
            for y, x in region_pixels:
                labeled[y, x] = 0
    
    logger.info(f"MGRG: {region_id - 1} regions")
    return labeled


def process_zone(zone_name: str):
    """Process a single zone to generate segmentations."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing zone: {zone_name.upper()}")
    logger.info(f"{'='*60}")
    
    # Paths
    sentinel_dir = Path(f'img/sentinel2/mexico/{zone_name}')
    embeddings_file = Path(f'img/sentinel2/embeddings/{zone_name}_prithvi.npz')
    output_dir = Path(f'data/processed/{zone_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Sentinel-2 bands
    logger.info("Loading Sentinel-2 bands...")
    nir = np.load(sentinel_dir / 'B08_10m.npy')
    red = np.load(sentinel_dir / 'B04_10m.npy')
    
    # Check if shapes match, if not, crop to minimum
    if nir.shape != red.shape:
        logger.warning(f"Shape mismatch: NIR {nir.shape} vs Red {red.shape}")
        min_h = min(nir.shape[0], red.shape[0])
        min_w = min(nir.shape[1], red.shape[1])
        nir = nir[:min_h, :min_w]
        red = red[:min_h, :min_w]
        logger.info(f"Cropped to common shape: {nir.shape}")
    
    # Calculate NDVI
    logger.info("Calculating NDVI...")
    ndvi = calculate_ndvi(nir, red)
    
    # Load Prithvi embeddings
    logger.info("Loading Prithvi embeddings...")
    emb_data = np.load(embeddings_file)
    embeddings = emb_data['embeddings']  # (H, W, 256)
    
    # Ensure embeddings match NDVI shape
    if embeddings.shape[:2] != ndvi.shape:
        logger.warning(f"Embeddings shape {embeddings.shape[:2]} != NDVI shape {ndvi.shape}")
        min_h = min(embeddings.shape[0], ndvi.shape[0])
        min_w = min(embeddings.shape[1], ndvi.shape[1])
        embeddings = embeddings[:min_h, :min_w, :]
        ndvi = ndvi[:min_h, :min_w]
        logger.info(f"Cropped to common shape: {ndvi.shape}")
    
    logger.info(f"  NDVI shape: {ndvi.shape}")
    logger.info(f"  Embeddings shape: {embeddings.shape}")
    
    # Generate Classic RG segmentation
    logger.info("\nGenerating Classic RG segmentation...")
    classic_seg = classic_region_growing(ndvi, threshold=0.1, min_size=50)
    
    # Generate MGRG segmentation
    logger.info("\nGenerating MGRG segmentation...")
    mgrg_seg = mgrg_segmentation(embeddings, threshold=0.85, min_size=50)
    
    # Save segmentations
    classic_path = output_dir / 'classic_rg_segmentation.npy'
    mgrg_path = output_dir / 'mgrg_segmentation.npy'
    ndvi_path = output_dir / 'ndvi.npy'
    
    np.save(classic_path, classic_seg)
    np.save(mgrg_path, mgrg_seg)
    np.save(ndvi_path, ndvi)
    
    logger.info(f"\nSaved segmentations:")
    logger.info(f"  Classic RG: {classic_path}")
    logger.info(f"  MGRG: {mgrg_path}")
    logger.info(f"  NDVI: {ndvi_path}")
    
    return classic_seg, mgrg_seg, ndvi


def main():
    """Generate segmentations for all zones."""
    zones = ['mexicali', 'bajio', 'sinaloa']
    
    for zone in zones:
        try:
            process_zone(zone)
        except Exception as e:
            logger.error(f"Failed to process {zone}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n" + "="*60)
    logger.info("SEGMENTATION GENERATION COMPLETE")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Download Dynamic World masks (see scripts/download_dynamic_world_manual.md)")
    logger.info("2. Run validation notebook: notebooks/validation/ground_truth_validation.ipynb")


if __name__ == '__main__':
    main()
