"""
NDVI calculation and vegetation indices.
"""
import numpy as np
from typing import Dict, Optional


def calculate_ndvi(
    red_band: np.ndarray,
    nir_band: np.ndarray,
    scl_band: Optional[np.ndarray] = None
) -> Dict:
    """
    Calculate NDVI (Normalized Difference Vegetation Index).

    Parameters
    ----------
    red_band : np.ndarray
        Red band (B04)
    nir_band : np.ndarray
        NIR band (B08)
    scl_band : np.ndarray, optional
        Scene Classification Layer for cloud masking

    Returns
    -------
    dict
        Dictionary with 'ndvi_masked' and 'statistics' keys

    Examples
    --------
    >>> red = np.array([[100, 200], [150, 250]])
    >>> nir = np.array([[300, 400], [350, 450]])
    >>> result = calculate_ndvi(red, nir)
    >>> print(result['statistics']['mean'])
    0.5
    """
    # Avoid division by zero
    denominator = nir_band.astype(float) + red_band.astype(float)
    denominator = np.where(denominator == 0, 0.0001, denominator)

    # Calculate NDVI
    ndvi = (nir_band.astype(float) - red_band.astype(float)) / denominator

    # Apply cloud mask if provided
    if scl_band is not None:
        from src.utils.sentinel_download import create_cloud_mask
        cloud_mask = create_cloud_mask(scl_band)
        ndvi_masked = np.ma.masked_array(ndvi, mask=cloud_mask)
        cloud_coverage = np.sum(cloud_mask) / cloud_mask.size * 100
    else:
        ndvi_masked = ndvi
        cloud_coverage = 0.0

    # Calculate statistics
    statistics = {
        'mean': float(np.ma.mean(ndvi_masked)),
        'std': float(np.ma.std(ndvi_masked)),
        'min': float(np.ma.min(ndvi_masked)),
        'max': float(np.ma.max(ndvi_masked)),
        'cloud_coverage': cloud_coverage
    }

    return {
        'ndvi_masked': ndvi_masked,
        'statistics': statistics
    }


def calculate_evi(
    red_band: np.ndarray,
    nir_band: np.ndarray,
    blue_band: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0
) -> np.ndarray:
    """
    Calculate EVI (Enhanced Vegetation Index).

    EVI is less sensitive to atmospheric conditions than NDVI.

    Parameters
    ----------
    red_band : np.ndarray
        Red band (B04)
    nir_band : np.ndarray
        NIR band (B08)
    blue_band : np.ndarray
        Blue band (B02)
    G : float, default=2.5
        Gain factor
    C1 : float, default=6.0
        Coefficient for aerosol resistance (red correction)
    C2 : float, default=7.5
        Coefficient for aerosol resistance (blue correction)
    L : float, default=1.0
        Canopy background adjustment

    Returns
    -------
    np.ndarray
        EVI values

    Notes
    -----
    EVI formula: G * ((NIR - Red) / (NIR + C1*Red - C2*Blue + L))
    """
    numerator = nir_band - red_band
    denominator = nir_band + C1 * red_band - C2 * blue_band + L

    # Avoid division by zero
    denominator[denominator == 0] = 0.0001

    evi = G * (numerator / denominator)

    return evi


def calculate_savi(
    red_band: np.ndarray,
    nir_band: np.ndarray,
    L: float = 0.5
) -> np.ndarray:
    """
    Calculate SAVI (Soil-Adjusted Vegetation Index).

    SAVI minimizes soil brightness influences.

    Parameters
    ----------
    red_band : np.ndarray
        Red band (B04)
    nir_band : np.ndarray
        NIR band (B08)
    L : float, default=0.5
        Soil brightness correction factor (0-1)
        - 0: High vegetation cover
        - 0.5: Intermediate vegetation cover
        - 1: Low vegetation cover

    Returns
    -------
    np.ndarray
        SAVI values

    Notes
    -----
    SAVI formula: ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    """
    numerator = nir_band - red_band
    denominator = nir_band + red_band + L

    # Avoid division by zero
    denominator[denominator == 0] = 0.0001

    savi = (numerator / denominator) * (1 + L)

    return savi


def classify_vegetation_stress(ndvi: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Classify vegetation stress levels based on NDVI values.

    Parameters
    ----------
    ndvi : np.ndarray
        NDVI values

    Returns
    -------
    dict
        Dictionary with stress level masks

    Notes
    -----
    Classification:
    - High stress: NDVI < 0.3
    - Medium stress: 0.3 <= NDVI < 0.5
    - Low stress: NDVI >= 0.5
    """
    return {
        'high_stress': ndvi < 0.3,
        'medium_stress': (ndvi >= 0.3) & (ndvi < 0.5),
        'low_stress': ndvi >= 0.5
    }
