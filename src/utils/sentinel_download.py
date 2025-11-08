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
