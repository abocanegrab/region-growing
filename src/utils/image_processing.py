"""
Image processing utilities for satellite imagery.
"""
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import Tuple


def normalize_band(
    band: np.ndarray,
    method: str = 'percentile',
    percentiles: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """
    Normalize band values to 0-1 range.

    Parameters
    ----------
    band : np.ndarray
        Input band with raw DN values
    method : str, default='percentile'
        Normalization method: 'percentile', 'minmax', or 'std'
    percentiles : tuple, default=(2, 98)
        Percentiles for robust normalization

    Returns
    -------
    np.ndarray
        Normalized band in range [0, 1]
    """
    if method == 'percentile':
        p_low, p_high = np.percentile(band, percentiles)
        normalized = (band - p_low) / (p_high - p_low + 1e-10)
    elif method == 'minmax':
        normalized = (band - band.min()) / (band.max() - band.min() + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return np.clip(normalized, 0, 1)


def create_rgb_image(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    gamma: float = 0.8
) -> np.ndarray:
    """
    Create RGB image from individual bands.

    Parameters
    ----------
    red : np.ndarray
        Red band
    green : np.ndarray
        Green band
    blue : np.ndarray
        Blue band
    gamma : float, default=0.8
        Gamma correction factor (< 1 brightens, > 1 darkens)

    Returns
    -------
    np.ndarray
        RGB image as uint8 array with shape (H, W, 3)
    """
    # Stack bands
    rgb = np.stack([red, green, blue], axis=2)

    # Normalize using percentiles
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb_normalized = (rgb - p2) / (p98 - p2 + 1e-10)
    rgb_normalized = np.clip(rgb_normalized, 0, 1)

    # Apply gamma correction
    rgb_normalized = np.power(rgb_normalized, gamma)

    # Convert to uint8
    rgb_image = (rgb_normalized * 255).astype(np.uint8)

    return rgb_image


def create_false_color_image(
    nir: np.ndarray,
    red: np.ndarray,
    green: np.ndarray,
    gamma: float = 0.8
) -> np.ndarray:
    """
    Create false color image (NIR-Red-Green).

    Composition: NIR → R, Red → G, Green → B
    This composition highlights vegetation in red/pink tones.

    Parameters
    ----------
    nir : np.ndarray
        NIR band (B08)
    red : np.ndarray
        Red band (B04)
    green : np.ndarray
        Green band (B03)
    gamma : float, default=0.8
        Gamma correction factor

    Returns
    -------
    np.ndarray
        False color image as uint8 array
    """
    # Stack: NIR → R, Red → G, Green → B
    false_color = np.stack([nir, red, green], axis=2)

    # Normalize
    p2, p98 = np.percentile(false_color, [2, 98])
    fc_normalized = (false_color - p2) / (p98 - p2 + 1e-10)
    fc_normalized = np.clip(fc_normalized, 0, 1)

    # Gamma correction
    fc_normalized = np.power(fc_normalized, gamma)

    # Convert to uint8
    fc_image = (fc_normalized * 255).astype(np.uint8)

    return fc_image


def array_to_base64(image: np.ndarray, format: str = 'PNG') -> str:
    """
    Convert numpy array to base64 string.

    Parameters
    ----------
    image : np.ndarray
        Image array (uint8)
    format : str, default='PNG'
        Image format: 'PNG', 'JPEG', etc.

    Returns
    -------
    str
        Base64 encoded string
    """
    image_pil = Image.fromarray(image)
    buffered = BytesIO()
    image_pil.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
