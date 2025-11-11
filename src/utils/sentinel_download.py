"""
Pure functions for downloading Sentinel-2 imagery.
Can be used in notebooks, scripts, or FastAPI services.
"""
from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    SentinelHubRequest,
    MimeType,
    bbox_to_dimensions
)
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional


def create_sentinel_config(client_id: str, client_secret: str) -> SHConfig:
    """
    Create Sentinel Hub configuration.

    Parameters
    ----------
    client_id : str
        Sentinel Hub client ID
    client_secret : str
        Sentinel Hub client secret

    Returns
    -------
    SHConfig
        Configured Sentinel Hub config object

    Examples
    --------
    >>> config = create_sentinel_config("my_id", "my_secret")
    >>> print(config.sh_client_id)
    'my_id'
    """
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    return config


def download_sentinel2_bands(
    bbox_coords: Dict[str, float],
    config: SHConfig,
    bands: List[str] = ['B02', 'B03', 'B04', 'B08', 'SCL'],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    resolution: int = 10,
    max_cloud_coverage: float = 0.5
) -> Dict:
    """
    Download Sentinel-2 bands for specified area.

    Pure function with no side effects - can be used anywhere.

    Parameters
    ----------
    bbox_coords : dict
        Bounding box with keys: min_lat, min_lon, max_lat, max_lon
    config : SHConfig
        Sentinel Hub configuration
    bands : list, default=['B02', 'B03', 'B04', 'B08', 'SCL']
        List of bands to download
    date_from : str, optional
        Start date in format YYYY-MM-DD
    date_to : str, optional
        End date in format YYYY-MM-DD
    resolution : int, default=10
        Resolution in meters per pixel
    max_cloud_coverage : float, default=0.5
        Maximum cloud coverage (0.0 to 1.0)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'bands': dict with band names as keys, numpy arrays as values
        - 'metadata': dict with bbox, dimensions, dates, etc.

    Examples
    --------
    >>> bbox = {'min_lat': -12.1, 'min_lon': -77.1,
    ...         'max_lat': -12.0, 'max_lon': -77.0}
    >>> config = create_sentinel_config(client_id, client_secret)
    >>> data = download_sentinel2_bands(bbox, config)
    >>> print(data['bands']['B04'].shape)
    (512, 512)
    """
    # Configure dates
    if not date_to:
        date_to = datetime.now()
    else:
        date_to = datetime.strptime(date_to, '%Y-%m-%d')

    if not date_from:
        date_from = date_to - timedelta(days=30)
    else:
        date_from = datetime.strptime(date_from, '%Y-%m-%d')

    # Create BBox
    bbox = BBox(
        bbox=[
            bbox_coords['min_lon'],
            bbox_coords['min_lat'],
            bbox_coords['max_lon'],
            bbox_coords['max_lat']
        ],
        crs=CRS.WGS84
    )

    # Calculate dimensions
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)

    # Limit to 2500x2500 pixels
    max_dimension = 2500
    width, height = bbox_size
    if width > max_dimension or height > max_dimension:
        scale = min(max_dimension / width, max_dimension / height)
        bbox_size = (int(width * scale), int(height * scale))

    # Create evalscript
    band_list = ', '.join([f'"{b}"' for b in bands])
    evalscript = f"""
    //VERSION=3
    function setup() {{
        return {{
            input: [{{
                bands: [{band_list}],
                units: "DN"
            }}],
            output: {{
                bands: {len(bands)},
                sampleType: "FLOAT32"
            }}
        }};
    }}

    function evaluatePixel(sample) {{
        return [{', '.join([f'sample.{b}' for b in bands])}];
    }}
    """

    # Create request
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date_from, date_to),
                maxcc=max_cloud_coverage
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=bbox,
        size=bbox_size,
        config=config
    )

    # Download data
    data = request.get_data()[0]

    # Separate bands
    bands_dict = {}
    for i, band_name in enumerate(bands):
        bands_dict[band_name] = data[:, :, i]

    return {
        'bands': bands_dict,
        'metadata': {
            'bbox': bbox,
            'bbox_coords': bbox_coords,
            'dimensions': bbox_size,
            'date_from': date_from.strftime('%Y-%m-%d'),
            'date_to': date_to.strftime('%Y-%m-%d'),
            'resolution': resolution
        }
    }


def create_cloud_mask(scl_band: np.ndarray) -> np.ndarray:
    """
    Create cloud mask from SCL (Scene Classification Layer) band.

    Parameters
    ----------
    scl_band : np.ndarray
        SCL band from Sentinel-2 L2A

    Returns
    -------
    np.ndarray
        Boolean mask where True = cloud/shadow

    Notes
    -----
    SCL values:
    - 3: Cloud shadows
    - 8: Cloud medium probability
    - 9: Cloud high probability
    - 10: Thin cirrus

    Examples
    --------
    >>> scl = np.array([[3, 8], [0, 4]])
    >>> mask = create_cloud_mask(scl)
    >>> print(mask)
    [[True, True], [False, False]]
    """
    return np.isin(scl_band, [3, 8, 9, 10])


def download_hls_bands(
    bbox_coords: Dict[str, float],
    config: SHConfig,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    resolution: int = 10,
    max_cloud_coverage: float = 0.3
) -> Dict:
    """
    Download HLS (Harmonized Landsat Sentinel-2) bands for Prithvi model + NDVI.

    This function downloads the 6 specific bands required by Prithvi plus B08 for NDVI:
    - B02 (Blue, 10m)
    - B03 (Green, 10m)
    - B04 (Red, 10m)
    - B08 (NIR Broad, 10m) - For NDVI calculation
    - B8A (NIR Narrow, 20m) - For Prithvi embeddings
    - B11 (SWIR1, 20m)
    - B12 (SWIR2, 20m)

    The bands are downloaded separately by resolution to maintain quality.
    Use src.features.hls_processor.prepare_hls_image() to resample and stack.
    
    Parameters
    ----------
    bbox_coords : dict
        Bounding box with keys: min_lat, min_lon, max_lat, max_lon
    config : SHConfig
        Sentinel Hub configuration
    date_from : str, optional
        Start date in format YYYY-MM-DD
    date_to : str, optional
        End date in format YYYY-MM-DD
    resolution : int, default=10
        Target resolution for output dimensions (10m recommended)
    max_cloud_coverage : float, default=0.3
        Maximum cloud coverage (0.0 to 1.0)
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'bands_10m': dict with B02, B03, B04, B08 (numpy arrays)
        - 'bands_20m': dict with B8A, B11, B12 (numpy arrays)
        - 'metadata': dict with bbox, dimensions, dates, etc.
        
    Examples
    --------
    >>> bbox = {'min_lat': 32.45, 'min_lon': -115.35,
    ...         'max_lat': 32.55, 'max_lon': -115.25}
    >>> config = create_sentinel_config(client_id, client_secret)
    >>> data = download_hls_bands(bbox, config)
    >>> print(data['bands_10m']['B02'].shape)
    (512, 512)
    >>> print(data['bands_20m']['B8A'].shape)
    (256, 256)
    """
    if not date_to:
        date_to = datetime.now()
    else:
        date_to = datetime.strptime(date_to, '%Y-%m-%d')

    if not date_from:
        date_from = date_to - timedelta(days=30)
    else:
        date_from = datetime.strptime(date_from, '%Y-%m-%d')

    bbox = BBox(
        bbox=[
            bbox_coords['min_lon'],
            bbox_coords['min_lat'],
            bbox_coords['max_lon'],
            bbox_coords['max_lat']
        ],
        crs=CRS.WGS84
    )

    bbox_size_10m = bbox_to_dimensions(bbox, resolution=10)
    bbox_size_20m = bbox_to_dimensions(bbox, resolution=20)

    max_dimension = 2500
    width_10m, height_10m = bbox_size_10m
    if width_10m > max_dimension or height_10m > max_dimension:
        scale = min(max_dimension / width_10m, max_dimension / height_10m)
        bbox_size_10m = (int(width_10m * scale), int(height_10m * scale))
        bbox_size_20m = (int(bbox_size_10m[0] / 2), int(bbox_size_10m[1] / 2))

    evalscript_10m = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B08"],
                units: "REFLECTANCE"
            }],
            output: {
                bands: 4,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B02, sample.B03, sample.B04, sample.B08];
    }
    """

    evalscript_20m = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B8A", "B11", "B12"],
                units: "REFLECTANCE"
            }],
            output: {
                bands: 3,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B8A, sample.B11, sample.B12];
    }
    """

    request_10m = SentinelHubRequest(
        evalscript=evalscript_10m,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date_from, date_to),
                maxcc=max_cloud_coverage
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=bbox,
        size=bbox_size_10m,
        config=config
    )

    request_20m = SentinelHubRequest(
        evalscript=evalscript_20m,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date_from, date_to),
                maxcc=max_cloud_coverage
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=bbox,
        size=bbox_size_20m,
        config=config
    )

    data_10m = request_10m.get_data()[0]
    data_20m = request_20m.get_data()[0]

    # Validate downloaded data
    if data_10m is None or data_20m is None:
        raise ValueError(
            f"No Sentinel-2 data available for bbox {bbox_coords} "
            f"between {date_from.strftime('%Y-%m-%d')} and {date_to.strftime('%Y-%m-%d')}. "
            "Try: 1) Different date range (last 60 days), "
            "2) Larger area, 3) Higher cloud coverage threshold"
        )
    
    # Check for all-zero data (invalid observations)
    if np.all(data_10m == 0) or np.all(data_20m == 0):
        raise ValueError(
            f"Downloaded Sentinel-2 data is all zeros for bbox {bbox_coords}. "
            "Possible causes: No valid observations, 100% cloud coverage, or API error. "
            "Try: 1) Different date range, 2) Lower max_cloud_coverage, 3) Verify area has Sentinel-2 coverage"
        )
    
    # Check for minimal valid data (threshold: at least 5% non-zero)
    non_zero_10m = np.count_nonzero(data_10m) / data_10m.size
    non_zero_20m = np.count_nonzero(data_20m) / data_20m.size
    
    if non_zero_10m < 0.05 or non_zero_20m < 0.05:
        raise ValueError(
            f"Insufficient valid data ({non_zero_10m:.1%} and {non_zero_20m:.1%} non-zero). "
            "Area likely has no valid observations. "
            "Try: 1) Different date range, 2) Different location, 3) Check Sentinel-2 coverage map"
        )

    bands_10m = {
        'B02': data_10m[:, :, 0],
        'B03': data_10m[:, :, 1],
        'B04': data_10m[:, :, 2],
        'B08': data_10m[:, :, 3]
    }

    bands_20m = {
        'B8A': data_20m[:, :, 0],
        'B11': data_20m[:, :, 1],
        'B12': data_20m[:, :, 2]
    }

    return {
        'bands_10m': bands_10m,
        'bands_20m': bands_20m,
        'metadata': {
            'bbox': bbox,
            'bbox_coords': bbox_coords,
            'dimensions_10m': bbox_size_10m,
            'dimensions_20m': bbox_size_20m,
            'date_from': date_from.strftime('%Y-%m-%d'),
            'date_to': date_to.strftime('%Y-%m-%d'),
            'resolution': resolution,
            'bands_10m': ['B02', 'B03', 'B04', 'B08'],
            'bands_20m': ['B8A', 'B11', 'B12']
        }
    }


def test_sentinel_connection(config: SHConfig) -> Dict:
    """
    Test connection with Sentinel Hub.

    Parameters
    ----------
    config : SHConfig
        Sentinel Hub configuration

    Returns
    -------
    dict
        Dictionary with 'status' and 'message' keys
    """
    try:
        test_bbox = BBox(bbox=[-77.1, -12.1, -77.0, -12.0], crs=CRS.WGS84)

        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B04"],
                output: { bands: 1 }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B04];
        }
        """

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(datetime.now() - timedelta(days=30), datetime.now())
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=test_bbox,
            size=(100, 100),
            config=config
        )

        data = request.get_data()

        return {
            'status': 'success',
            'message': 'Successful connection to Sentinel Hub',
            'data_shape': data[0].shape if data else None
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Connection error: {str(e)}'
        }
