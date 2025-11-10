"""
Test embeddings extraction from HLS images.

This script tests the extraction of semantic embeddings from downloaded
HLS images using the Prithvi model.

Usage:
    python scripts/test_embeddings.py --zone mexicali
    python scripts/test_embeddings.py --zone mexicali --use-simple-model
    python scripts/test_embeddings.py --all
"""
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.features.hls_processor import (
    extract_embeddings,
    save_embeddings,
    load_embeddings,
    visualize_embeddings_pca,
    compute_cosine_similarity
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_hls_image_from_zone(zone_name: str) -> tuple:
    """
    Load HLS image from downloaded zone data.
    
    Parameters
    ----------
    zone_name : str
        Zone name ('mexicali', 'bajio', or 'sinaloa')
        
    Returns
    -------
    hls_image : np.ndarray
        HLS image with shape (6, H, W)
    zone_dir : Path
        Path to zone directory
    """
    zone_dir = Path(__file__).parent.parent / 'img' / 'sentinel2' / 'mexico' / zone_name
    hls_path = zone_dir / 'hls_image.npy'
    
    if not hls_path.exists():
        raise FileNotFoundError(
            f"HLS image not found at {hls_path}. "
            f"Run 'python scripts/download_hls_image.py --zone {zone_name}' first."
        )
    
    logger.info(f"Loading HLS image from {hls_path}")
    hls_image = np.load(hls_path)
    logger.info(f"  Shape: {hls_image.shape}")
    logger.info(f"  Data range: [{hls_image.min():.4f}, {hls_image.max():.4f}]")
    
    return hls_image, zone_dir


def test_embeddings_extraction(
    zone_name: str,
    use_simple_model: bool = False,
    save_results: bool = True
):
    """
    Test embeddings extraction for a specific zone.
    
    Parameters
    ----------
    zone_name : str
        Zone name ('mexicali', 'bajio', or 'sinaloa')
    use_simple_model : bool, default=False
        If True, use simplified model for testing
    save_results : bool, default=True
        If True, save embeddings to disk
    """
    logger.info("="*70)
    logger.info(f"Testing embeddings extraction for: {zone_name}")
    logger.info(f"Model: {'Simple (test)' if use_simple_model else 'Real Prithvi'}")
    logger.info("="*70)
    
    hls_image, zone_dir = load_hls_image_from_zone(zone_name)
    
    logger.info("Extracting embeddings...")
    try:
        embeddings = extract_embeddings(
            hls_image,
            use_simple_model=use_simple_model,
            device='cuda',
            normalize_output=True
        )
    except Exception as e:
        logger.error(f"Failed to extract embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    logger.info(f"Embeddings extracted successfully!")
    logger.info(f"  Shape: {embeddings.shape}")
    logger.info(f"  Data range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    logger.info(f"  Mean norm: {np.mean(np.linalg.norm(embeddings, axis=2)):.4f}")
    
    if save_results:
        embeddings_dir = Path(__file__).parent.parent / 'img' / 'sentinel2' / 'embeddings'
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        model_suffix = 'simple' if use_simple_model else 'prithvi'
        output_path = embeddings_dir / f"{zone_name}_{model_suffix}.npz"
        
        metadata = {
            'zone': zone_name,
            'model': model_suffix,
            'shape': str(embeddings.shape)
        }
        
        save_embeddings(embeddings, output_path, metadata)
    
    logger.info("Visualizing embeddings with PCA...")
    rgb_visualization = visualize_embeddings_pca(embeddings, n_components=3)
    
    vis_dir = zone_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    model_suffix = 'simple' if use_simple_model else 'prithvi'
    vis_path = vis_dir / f'embeddings_pca_{model_suffix}.png'
    
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_visualization)
    plt.title(f'Embeddings PCA Visualization - {zone_name.capitalize()} ({model_suffix})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved PCA visualization: {vis_path}")
    
    logger.info("="*70)
    logger.info(f"Test complete for {zone_name}!")
    logger.info("="*70)
    
    return {
        'zone': zone_name,
        'embeddings_shape': embeddings.shape,
        'model': model_suffix,
        'output_path': output_path if save_results else None,
        'visualization_path': vis_path
    }


def compare_zones(
    zones: list,
    use_simple_model: bool = False
):
    """
    Compare embeddings across multiple zones.
    
    Parameters
    ----------
    zones : list
        List of zone names to compare
    use_simple_model : bool, default=False
        If True, use simplified model for testing
    """
    logger.info("="*70)
    logger.info(f"Comparing embeddings across zones: {zones}")
    logger.info("="*70)
    
    all_embeddings = {}
    
    for zone in zones:
        logger.info(f"\nProcessing {zone}...")
        result = test_embeddings_extraction(
            zone_name=zone,
            use_simple_model=use_simple_model,
            save_results=True
        )
        
        if result:
            embeddings_path = result['output_path']
            embeddings, metadata = load_embeddings(embeddings_path)
            all_embeddings[zone] = embeddings
    
    if len(all_embeddings) < 2:
        logger.error("Need at least 2 zones for comparison")
        return
    
    logger.info("\n" + "="*70)
    logger.info("Computing pairwise similarities...")
    logger.info("="*70)
    
    zone_names = list(all_embeddings.keys())
    for i in range(len(zone_names)):
        for j in range(i + 1, len(zone_names)):
            zone_a = zone_names[i]
            zone_b = zone_names[j]
            
            emb_a = all_embeddings[zone_a]
            emb_b = all_embeddings[zone_b]
            
            min_h = min(emb_a.shape[0], emb_b.shape[0])
            min_w = min(emb_a.shape[1], emb_b.shape[1])
            
            emb_a_crop = emb_a[:min_h, :min_w, :]
            emb_b_crop = emb_b[:min_h, :min_w, :]
            
            similarity = compute_cosine_similarity(emb_a_crop, emb_b_crop)
            
            logger.info(f"\n{zone_a.capitalize()} vs {zone_b.capitalize()}:")
            logger.info(f"  Mean similarity: {similarity.mean():.4f}")
            logger.info(f"  Std similarity: {similarity.std():.4f}")
            logger.info(f"  Min similarity: {similarity.min():.4f}")
            logger.info(f"  Max similarity: {similarity.max():.4f}")
            
            vis_dir = Path(__file__).parent.parent / 'img' / 'sentinel2' / 'embeddings'
            vis_path = vis_dir / f'similarity_{zone_a}_{zone_b}.png'
            
            plt.figure(figsize=(10, 8))
            plt.imshow(similarity, cmap='RdYlGn', vmin=-1, vmax=1)
            plt.colorbar(label='Cosine Similarity')
            plt.title(f'Similarity: {zone_a.capitalize()} vs {zone_b.capitalize()}')
            plt.tight_layout()
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Saved similarity map: {vis_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Test embeddings extraction from HLS images'
    )
    parser.add_argument(
        '--zone',
        type=str,
        choices=['mexicali', 'bajio', 'sinaloa'],
        help='Zone to test'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all zones'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare embeddings across zones'
    )
    parser.add_argument(
        '--use-simple-model',
        action='store_true',
        help='Use simplified model for testing (no Prithvi download needed)'
    )
    
    args = parser.parse_args()
    
    if not args.zone and not args.all:
        parser.error('Either --zone or --all must be specified')
    
    if args.all or args.compare:
        zones_to_test = ['mexicali', 'bajio', 'sinaloa']
    else:
        zones_to_test = [args.zone]
    
    if args.compare:
        compare_zones(zones_to_test, use_simple_model=args.use_simple_model)
    else:
        for zone in zones_to_test:
            test_embeddings_extraction(
                zone_name=zone,
                use_simple_model=args.use_simple_model,
                save_results=True
            )


if __name__ == '__main__':
    main()
