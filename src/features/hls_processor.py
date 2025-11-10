"""
HLS Image Processor for Prithvi Foundation Model.

This module provides functions to process Sentinel-2 imagery into HLS format
and extract semantic embeddings using the Prithvi foundation model.

HLS (Harmonized Landsat Sentinel-2) format requires 6 specific bands:
B02, B03, B04, B8A, B11, B12

References:
    Jakubik et al. (2024). Foundation models for generalist geospatial AI.
    Claverie et al. (2018). The Harmonized Landsat and Sentinel-2 surface reflectance data set.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


def resample_band_to_10m(
    band_20m: np.ndarray,
    target_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Resample 20m resolution band to 10m using bilinear interpolation.
    
    This function is used to resample B8A, B11, and B12 bands from Sentinel-2
    which are originally at 20m resolution to 10m to match B02, B03, B04.
    
    Parameters
    ----------
    band_20m : np.ndarray
        Band at 20m resolution with shape (H/2, W/2)
    target_shape : tuple, optional
        Target shape (H, W). If None, doubles both dimensions.
        
    Returns
    -------
    np.ndarray
        Resampled band at 10m resolution with shape (H, W)
        
    Examples
    --------
    >>> band_20m = np.random.rand(256, 256)
    >>> band_10m = resample_band_to_10m(band_20m)
    >>> print(band_10m.shape)
    (512, 512)
    
    >>> band_10m = resample_band_to_10m(band_20m, target_shape=(480, 480))
    >>> print(band_10m.shape)
    (480, 480)
    """
    if target_shape is not None:
        zoom_factors = (
            target_shape[0] / band_20m.shape[0],
            target_shape[1] / band_20m.shape[1]
        )
    else:
        zoom_factors = (2, 2)
    
    logger.debug(f"Resampling band from {band_20m.shape} with zoom factors {zoom_factors}")
    return zoom(band_20m, zoom_factors, order=1)


def stack_hls_bands(
    bands_dict: Dict[str, np.ndarray],
    validate: bool = True
) -> np.ndarray:
    """
    Stack HLS bands in correct order for Prithvi model.
    
    The Prithvi model requires exactly 6 bands in this specific order:
    [B02, B03, B04, B8A, B11, B12]
    
    Parameters
    ----------
    bands_dict : dict
        Dictionary with band names as keys and numpy arrays as values
    validate : bool, default=True
        Whether to validate that all required bands are present
        
    Returns
    -------
    np.ndarray
        Stacked bands with shape (6, H, W)
        
    Raises
    ------
    ValueError
        If required bands are missing or dimensions don't match
        
    Examples
    --------
    >>> bands = {
    ...     'B02': np.random.rand(512, 512),
    ...     'B03': np.random.rand(512, 512),
    ...     'B04': np.random.rand(512, 512),
    ...     'B8A': np.random.rand(512, 512),
    ...     'B11': np.random.rand(512, 512),
    ...     'B12': np.random.rand(512, 512)
    ... }
    >>> hls_image = stack_hls_bands(bands)
    >>> print(hls_image.shape)
    (6, 512, 512)
    """
    required_bands = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
    
    if validate:
        for band in required_bands:
            if band not in bands_dict:
                raise ValueError(f"Missing required band: {band}")
        
        shapes = [bands_dict[b].shape for b in required_bands]
        if len(set(shapes)) > 1:
            raise ValueError(f"Band dimensions don't match: {dict(zip(required_bands, shapes))}")
    
    stacked = np.stack([bands_dict[b] for b in required_bands], axis=0)
    logger.info(f"Stacked HLS bands in correct order: {required_bands}")
    logger.info(f"Output shape: {stacked.shape}")
    
    return stacked


def normalize_embeddings_l2(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings using L2 norm for cosine similarity.
    
    This normalization ensures that cosine similarity can be computed
    efficiently using dot product: cosine_sim(a, b) = dot(a, b)
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings with shape (H, W, D) where D is embedding dimension
        
    Returns
    -------
    np.ndarray
        L2-normalized embeddings with same shape (H, W, D)
        Each D-dimensional vector has unit norm.
        
    Notes
    -----
    Vectors with zero norm are kept as zero vectors.
    
    Examples
    --------
    >>> embeddings = np.random.rand(10, 10, 256)
    >>> normalized = normalize_embeddings_l2(embeddings)
    >>> norms = np.linalg.norm(normalized, axis=2)
    >>> print(np.allclose(norms, 1.0, atol=1e-6))
    True
    """
    norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    
    logger.debug(f"Normalized embeddings with L2 norm, shape: {normalized.shape}")
    return normalized


def prepare_hls_image(
    bands_10m: Dict[str, np.ndarray],
    bands_20m: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Prepare complete HLS image from 10m and 20m bands.
    
    This function:
    1. Resamples 20m bands (B8A, B11, B12) to 10m resolution
    2. Stacks all 6 bands in correct order
    
    Parameters
    ----------
    bands_10m : dict
        Dictionary with B02, B03, B04 bands at 10m resolution
    bands_20m : dict
        Dictionary with B8A, B11, B12 bands at 20m resolution
        
    Returns
    -------
    np.ndarray
        HLS image with shape (6, H, W) at 10m resolution
        
    Examples
    --------
    >>> bands_10m = {
    ...     'B02': np.random.rand(512, 512),
    ...     'B03': np.random.rand(512, 512),
    ...     'B04': np.random.rand(512, 512)
    ... }
    >>> bands_20m = {
    ...     'B8A': np.random.rand(256, 256),
    ...     'B11': np.random.rand(256, 256),
    ...     'B12': np.random.rand(256, 256)
    ... }
    >>> hls_image = prepare_hls_image(bands_10m, bands_20m)
    >>> print(hls_image.shape)
    (6, 512, 512)
    """
    target_shape = bands_10m['B02'].shape
    logger.info(f"Preparing HLS image with target shape: {target_shape}")
    
    resampled_20m = {}
    for band_name, band_data in bands_20m.items():
        logger.debug(f"Resampling {band_name} from {band_data.shape} to {target_shape}")
        resampled_20m[band_name] = resample_band_to_10m(band_data, target_shape)
    
    all_bands = {**bands_10m, **resampled_20m}
    
    return stack_hls_bands(all_bands)


def extract_embeddings(
    hls_image: np.ndarray,
    model_path: Optional[Union[str, Path]] = None,
    use_simple_model: bool = False,
    device: Optional[str] = None,
    normalize_output: bool = True
) -> np.ndarray:
    """
    Extract semantic embeddings from HLS image using Prithvi model.
    
    This is the main function for extracting embeddings. It handles:
    1. Loading the Prithvi model
    2. Converting numpy array to PyTorch tensor
    3. Normalizing the input
    4. Running inference
    5. Interpolating to original resolution
    6. L2 normalization of embeddings
    
    Parameters
    ----------
    hls_image : np.ndarray
        HLS image with shape (6, H, W) containing bands:
        [B02, B03, B04, B8A, B11, B12] in that exact order
    model_path : str or Path, optional
        Path to Prithvi model weights. If None, uses default location.
    use_simple_model : bool, default=False
        If True, uses simplified model for testing without downloading
        the full Prithvi model. Set to False for production.
    device : str, optional
        Device to run inference on ('cuda' or 'cpu'). If None, auto-detects.
    normalize_output : bool, default=True
        If True, L2-normalizes embeddings for cosine similarity
        
    Returns
    -------
    np.ndarray
        Semantic embeddings with shape (H, W, 256)
        If normalize_output=True, each 256D vector has unit norm.
        
    Raises
    ------
    ValueError
        If input shape is invalid
    RuntimeError
        If model inference fails
        
    Examples
    --------
    >>> hls_image = np.random.rand(6, 224, 224)
    >>> embeddings = extract_embeddings(hls_image, use_simple_model=True)
    >>> print(embeddings.shape)
    (224, 224, 256)
    
    >>> embeddings = extract_embeddings(
    ...     hls_image,
    ...     model_path='models/prithvi/Prithvi_EO_V1_100M.pt',
    ...     use_simple_model=False,
    ...     device='cuda'
    ... )
    """
    from src.models.prithvi_loader import (
        load_prithvi_model,
        normalize_hls_image,
        interpolate_embeddings
    )
    
    if hls_image.shape[0] != 6:
        raise ValueError(
            f"Expected 6 bands (B02,B03,B04,B8A,B11,B12), got {hls_image.shape[0]}"
        )
    
    logger.info(f"Extracting embeddings from HLS image with shape: {hls_image.shape}")
    logger.info(f"Using {'simplified' if use_simple_model else 'real'} Prithvi model")

    hls_tensor = torch.from_numpy(hls_image).unsqueeze(0).float()
    logger.debug(f"Converted to tensor with shape: {hls_tensor.shape}")

    hls_normalized = normalize_hls_image(hls_tensor, method='standardize')
    logger.debug("Normalized HLS image (mean=0, std=1)")

    encoder = load_prithvi_model(
        model_path=model_path,
        use_simple_model=use_simple_model,
        device=device
    )

    # Move input tensor to same device as model
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hls_normalized = hls_normalized.to(device)
    logger.debug(f"Moved input tensor to device: {device}")

    logger.info("Running Prithvi inference...")
    with torch.no_grad():
        embeddings_tensor = encoder(hls_normalized)
    
    logger.info(f"Raw embeddings shape: {embeddings_tensor.shape}")
    
    original_size = (hls_image.shape[1], hls_image.shape[2])
    if embeddings_tensor.shape[2:] != original_size:
        logger.info(f"Interpolating embeddings to original size: {original_size}")
        embeddings_tensor = interpolate_embeddings(
            embeddings_tensor,
            target_size=original_size,
            mode='bilinear'
        )
    
    embeddings_np = embeddings_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    logger.info(f"Converted to numpy with shape: {embeddings_np.shape}")
    
    if normalize_output:
        embeddings_np = normalize_embeddings_l2(embeddings_np)
        logger.info("Applied L2 normalization to embeddings")
    
    logger.info("Embedding extraction completed successfully")
    return embeddings_np


def compute_cosine_similarity(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    method: str = 'pixel_wise'
) -> Union[np.ndarray, float]:
    """
    Compute cosine similarity between two embedding arrays.
    
    Assumes embeddings are already L2-normalized. If not, the result
    will be scaled dot product instead of true cosine similarity.
    
    Parameters
    ----------
    embeddings_a : np.ndarray
        First embeddings with shape (H, W, D)
    embeddings_b : np.ndarray
        Second embeddings with shape (H, W, D)
    method : str, default='pixel_wise'
        Method for computing similarity:
        - 'pixel_wise': Pixel-by-pixel similarity (requires same shape)
        - 'average': Average similarity across all pixels (handles different shapes)
        
    Returns
    -------
    np.ndarray or float
        If method='pixel_wise': similarity map with shape (H, W)
        If method='average': single float value representing average similarity
        Values range from -1 (opposite) to 1 (identical)
        
    Examples
    --------
    >>> emb_a = np.random.rand(10, 10, 256)
    >>> emb_a = normalize_embeddings_l2(emb_a)
    >>> emb_b = np.random.rand(10, 10, 256)
    >>> emb_b = normalize_embeddings_l2(emb_b)
    >>> similarity = compute_cosine_similarity(emb_a, emb_b, method='pixel_wise')
    >>> print(similarity.shape)
    (10, 10)
    
    >>> emb_c = np.random.rand(12, 12, 256)
    >>> emb_c = normalize_embeddings_l2(emb_c)
    >>> avg_sim = compute_cosine_similarity(emb_a, emb_c, method='average')
    >>> print(type(avg_sim))
    <class 'numpy.float64'>
    """
    if method == 'pixel_wise':
        if embeddings_a.shape != embeddings_b.shape:
            raise ValueError(
                f"Shape mismatch for pixel_wise method: {embeddings_a.shape} vs {embeddings_b.shape}. "
                f"Use method='average' for embeddings with different spatial dimensions."
            )
        
        similarity = np.sum(embeddings_a * embeddings_b, axis=2)
        logger.debug(f"Computed pixel-wise cosine similarity, range: [{similarity.min():.3f}, {similarity.max():.3f}]")
        return similarity
    
    elif method == 'average':
        # Flatten spatial dimensions and compute average similarity
        emb_a_flat = embeddings_a.reshape(-1, embeddings_a.shape[-1])
        emb_b_flat = embeddings_b.reshape(-1, embeddings_b.shape[-1])
        
        # Compute pairwise similarities and take mean
        # For efficiency, sample if arrays are very large
        max_samples = 10000
        if len(emb_a_flat) > max_samples:
            indices_a = np.random.choice(len(emb_a_flat), max_samples, replace=False)
            emb_a_flat = emb_a_flat[indices_a]
        if len(emb_b_flat) > max_samples:
            indices_b = np.random.choice(len(emb_b_flat), max_samples, replace=False)
            emb_b_flat = emb_b_flat[indices_b]
        
        # Compute similarity matrix and take mean
        similarity_matrix = np.dot(emb_a_flat, emb_b_flat.T)
        avg_similarity = float(similarity_matrix.mean())
        
        logger.debug(f"Computed average cosine similarity: {avg_similarity:.4f}")
        return avg_similarity
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pixel_wise' or 'average'.")

def compute_cosine_similarity_fixed(embeddings_a, embeddings_b, method='average'):
    """
    Compute cosine similarity between embeddings with flexible methods.
    
    This function handles embeddings of different spatial sizes by using
    either pixel-wise comparison (when sizes match) or averaged sampling
    (when sizes differ).
    
    Parameters
    ----------
    embeddings_a : np.ndarray
        First embeddings array with shape (H_a, W_a, D)
    embeddings_b : np.ndarray
        Second embeddings array with shape (H_b, W_b, D)
    method : str, default='average'
        Comparison method:
        - 'pixel_wise': Element-wise similarity (requires same shape)
        - 'average': Sample-based average similarity (works with different shapes)
        
    Returns
    -------
    float or np.ndarray
        If method='average': float with average similarity score
        If method='pixel_wise': np.ndarray with shape (H, W) containing
        per-pixel similarities
        
    Raises
    ------
    ValueError
        If method='pixel_wise' and shapes don't match
        If method is unknown
        
    Examples
    --------
    >>> emb_a = np.random.rand(100, 100, 256)
    >>> emb_b = np.random.rand(100, 100, 256)
    >>> similarity_map = compute_cosine_similarity_fixed(emb_a, emb_b, 'pixel_wise')
    >>> print(similarity_map.shape)
    (100, 100)
    
    >>> emb_a = np.random.rand(100, 100, 256)
    >>> emb_b = np.random.rand(150, 150, 256)  # Different size
    >>> avg_sim = compute_cosine_similarity_fixed(emb_a, emb_b, 'average')
    >>> print(f"Average similarity: {avg_sim:.3f}")
    Average similarity: 0.523
    """    
    if method == 'pixel_wise':
        if embeddings_a.shape != embeddings_b.shape:
            raise ValueError(
                f"Shape mismatch for pixel_wise: {embeddings_a.shape} vs {embeddings_b.shape}"
            )
        return np.sum(embeddings_a * embeddings_b, axis=2)
    
    elif method == 'average':
        emb_a_flat = embeddings_a.reshape(-1, embeddings_a.shape[-1])
        emb_b_flat = embeddings_b.reshape(-1, embeddings_b.shape[-1])
        
        max_samples = 10000
        if len(emb_a_flat) > max_samples:
            indices = np.random.choice(len(emb_a_flat), max_samples, replace=False)
            emb_a_flat = emb_a_flat[indices]
        if len(emb_b_flat) > max_samples:
            indices = np.random.choice(len(emb_b_flat), max_samples, replace=False)
            emb_b_flat = emb_b_flat[indices]
        
        similarity_matrix = np.dot(emb_a_flat, emb_b_flat.T)
        return float(similarity_matrix.mean())
    
    else:
        raise ValueError(f"Unknown method: {method}")

def save_embeddings(
    embeddings: np.ndarray,
    output_path: Union[str, Path],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save embeddings to disk in compressed numpy format.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings to save with shape (H, W, D)
    output_path : str or Path
        Output file path (will add .npz extension if not present)
    metadata : dict, optional
        Additional metadata to save alongside embeddings
        
    Examples
    --------
    >>> embeddings = np.random.rand(512, 512, 256)
    >>> metadata = {'zone': 'mexicali', 'date': '2024-01-15'}
    >>> save_embeddings(embeddings, 'img/sentinel2/embeddings/mexicali.npz', metadata)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not str(output_path).endswith('.npz'):
        output_path = output_path.with_suffix('.npz')
    
    save_dict = {'embeddings': embeddings}
    if metadata:
        for key, value in metadata.items():
            save_dict[f'metadata_{key}'] = value
    
    np.savez_compressed(output_path, **save_dict)
    logger.info(f"Saved embeddings to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024**2:.2f} MB")


def load_embeddings(input_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """
    Load embeddings from disk.
    
    Parameters
    ----------
    input_path : str or Path
        Path to embeddings file (.npz)
        
    Returns
    -------
    embeddings : np.ndarray
        Loaded embeddings with shape (H, W, D)
    metadata : dict
        Metadata dictionary (empty if no metadata was saved)
        
    Examples
    --------
    >>> embeddings, metadata = load_embeddings('img/sentinel2/embeddings/mexicali.npz')
    >>> print(embeddings.shape)
    (512, 512, 256)
    >>> print(metadata)
    {'zone': 'mexicali', 'date': '2024-01-15'}
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {input_path}")
    
    data = np.load(input_path)
    embeddings = data['embeddings']
    
    metadata = {}
    for key in data.files:
        if key.startswith('metadata_'):
            metadata[key.replace('metadata_', '')] = data[key]
    
    logger.info(f"Loaded embeddings from {input_path}")
    logger.info(f"Shape: {embeddings.shape}")
    logger.info(f"Metadata keys: {list(metadata.keys())}")
    
    return embeddings, metadata


def visualize_embeddings_pca(
    embeddings: np.ndarray,
    n_components: int = 3
) -> np.ndarray:
    """
    Reduce embeddings to 3D using PCA for visualization.
    
    This is useful for visualizing high-dimensional embeddings as RGB images.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings with shape (H, W, D)
    n_components : int, default=3
        Number of PCA components (typically 3 for RGB visualization)
        
    Returns
    -------
    np.ndarray
        Reduced embeddings with shape (H, W, n_components)
        Normalized to [0, 1] range for visualization
        
    Examples
    --------
    >>> embeddings = np.random.rand(512, 512, 256)
    >>> rgb = visualize_embeddings_pca(embeddings)
    >>> print(rgb.shape)
    (512, 512, 3)
    >>> print(rgb.min(), rgb.max())
    0.0 1.0
    """
    from sklearn.decomposition import PCA
    
    h, w, d = embeddings.shape
    embeddings_flat = embeddings.reshape(-1, d)
    
    logger.info(f"Applying PCA to embeddings: {d}D -> {n_components}D")
    pca = PCA(n_components=n_components)
    reduced_flat = pca.fit_transform(embeddings_flat)
    
    logger.info(f"Explained variance: {pca.explained_variance_ratio_}")
    
    reduced = reduced_flat.reshape(h, w, n_components)
    
    reduced_min = reduced.min()
    reduced_max = reduced.max()
    reduced = (reduced - reduced_min) / (reduced_max - reduced_min + 1e-8)
    
    logger.info(f"PCA reduction completed, output shape: {reduced.shape}")
    return reduced
