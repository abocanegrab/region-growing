"""
Geospatial utilities for region processing.
"""
import numpy as np
from typing import Dict, List
import geojson
from shapely.geometry import Polygon
from pyproj import Geod


def validate_bbox(bbox_coords: Dict[str, float]) -> bool:
    """
    Validate bounding box coordinates.

    Parameters
    ----------
    bbox_coords : dict
        Dictionary with keys: min_lat, min_lon, max_lat, max_lon

    Returns
    -------
    bool
        True if bbox is valid, False otherwise

    Examples
    --------
    >>> bbox = {'min_lat': -12.1, 'min_lon': -77.1,
    ...         'max_lat': -12.0, 'max_lon': -77.0}
    >>> validate_bbox(bbox)
    True
    """
    required_keys = ['min_lat', 'min_lon', 'max_lat', 'max_lon']
    if not all(key in bbox_coords for key in required_keys):
        return False

    if bbox_coords['min_lat'] >= bbox_coords['max_lat']:
        return False

    if bbox_coords['min_lon'] >= bbox_coords['max_lon']:
        return False

    if not (-90 <= bbox_coords['min_lat'] <= 90):
        return False

    if not (-90 <= bbox_coords['max_lat'] <= 90):
        return False

    if not (-180 <= bbox_coords['min_lon'] <= 180):
        return False

    if not (-180 <= bbox_coords['max_lon'] <= 180):
        return False

    return True


def calculate_bbox_area(bbox_coords: Dict[str, float]) -> float:
    """
    Calculate area of bounding box in hectares.

    Parameters
    ----------
    bbox_coords : dict
        Dictionary with keys: min_lat, min_lon, max_lat, max_lon

    Returns
    -------
    float
        Area in hectares

    Examples
    --------
    >>> bbox = {'min_lat': -12.1, 'min_lon': -77.1,
    ...         'max_lat': -12.0, 'max_lon': -77.0}
    >>> area = calculate_bbox_area(bbox)
    >>> print(f"Area: {area:.2f} ha")
    """
    # Create polygon from bbox
    coords = [
        (bbox_coords['min_lon'], bbox_coords['min_lat']),
        (bbox_coords['max_lon'], bbox_coords['min_lat']),
        (bbox_coords['max_lon'], bbox_coords['max_lat']),
        (bbox_coords['min_lon'], bbox_coords['max_lat']),
        (bbox_coords['min_lon'], bbox_coords['min_lat'])
    ]

    # Calculate area using WGS84 ellipsoid
    geod = Geod(ellps='WGS84')
    area, _ = geod.geometry_area_perimeter(Polygon(coords))

    # Convert m² to hectares (1 ha = 10,000 m²)
    area_hectares = abs(area) / 10000

    return area_hectares


def regions_to_geojson(labeled_image: np.ndarray, metadata: Dict) -> Dict:
    """
    Convert labeled regions to GeoJSON format.

    Parameters
    ----------
    labeled_image : np.ndarray
        Image with region labels
    metadata : dict
        Metadata with bbox_coords, dimensions

    Returns
    -------
    dict
        GeoJSON FeatureCollection

    Notes
    -----
    This is a simplified version. For production, use rasterio.features.shapes
    """
    features = []

    # Get unique region IDs
    region_ids = np.unique(labeled_image)
    region_ids = region_ids[region_ids != 0]  # Exclude background

    # Calculate pixel size in degrees
    bbox = metadata['bbox_coords']
    height, width = metadata['dimensions']
    pixel_size_lon = (bbox['max_lon'] - bbox['min_lon']) / width
    pixel_size_lat = (bbox['max_lat'] - bbox['min_lat']) / height

    for region_id in region_ids:
        # Find pixels belonging to this region
        mask = labeled_image == region_id
        y_coords, x_coords = np.where(mask)

        if len(y_coords) == 0:
            continue

        # Calculate centroid
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)

        # Convert pixel coordinates to geographic coordinates
        lon = bbox['min_lon'] + centroid_x * pixel_size_lon
        lat = bbox['max_lat'] - centroid_y * pixel_size_lat

        # Create feature
        feature = geojson.Feature(
            geometry=geojson.Point((lon, lat)),
            properties={
                'region_id': int(region_id),
                'pixel_count': int(np.sum(mask))
            }
        )
        features.append(feature)

    return geojson.FeatureCollection(features)


def calculate_statistics(
    values: np.ndarray,
    mask: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate statistical metrics for a region.

    Parameters
    ----------
    values : np.ndarray
        Array of values (e.g., NDVI)
    mask : np.ndarray, optional
        Boolean mask to filter values

    Returns
    -------
    dict
        Dictionary with statistical metrics

    Examples
    --------
    >>> values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> stats = calculate_statistics(values)
    >>> print(f"Mean: {stats['mean']:.2f}")
    """
    if mask is not None:
        values = values[~mask]

    if len(values) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'count': 0
        }

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'count': int(len(values))
    }
