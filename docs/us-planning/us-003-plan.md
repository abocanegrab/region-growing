# US-003: Refactorizar a Arquitectura Limpia + Descarga Sentinel-2

**Epic:** Fundación y Baseline (Días 1-3)  
**Prioridad:** Alta (Bloqueante para notebooks y US futuras)  
**Estimación:** 5 horas  
**Responsable:** Carlos Bocanegra  
**Estado:** 🔄 **PLANEACIÓN APROBADA** → Lista para implementación  
**Fecha de Planeación:** 8 de Noviembre de 2025  
**Versión:** 2.0 (Arquitectura Limpia)

---

## 📋 Historia de Usuario

**Como** desarrollador  
**Quiero** refactorizar el código a una arquitectura limpia con `src/` reutilizable  
**Para que** el código sea reutilizable en backend, notebooks y scripts sin duplicación

---

## 🔍 Análisis de Estado Actual

### ✅ Funcionalidad Implementada (100%)

La funcionalidad de descarga de Sentinel-2 **YA ESTÁ COMPLETAMENTE IMPLEMENTADA** en:
- `backend/app/services/sentinel_hub_service.py`

**Características funcionando:**
- ✅ Integración con Sentinel Hub API
- ✅ Descarga de bandas RGB (B02, B03, B04)
- ✅ Descarga de banda NIR (B08) para NDVI
- ✅ Descarga de banda SCL para máscara de nubes
- ✅ Manejo de errores completo
- ✅ Logging profesional

### ❌ Problema: Arquitectura No Escalable

**Problemas identificados:**
1. **Backend tiene su propio Poetry** → Dificulta compartir código
2. **Código acoplado a FastAPI** → No reutilizable en notebooks
3. **No existe `src/` reutilizable** → Viola AGENTS.md
4. **Notebooks no pueden importar** → Duplicación inevitable

---

## 🎯 Objetivos de la Refactorización

### Objetivo Principal
Migrar a arquitectura limpia con UN SOLO Poetry y código reutilizable en `src/`.

### Objetivos Específicos

1. **Crear Poetry global en raíz** (eliminar backend/pyproject.toml)
2. **Crear estructura `src/`** con código reutilizable
3. **Refactorizar backend** como wrapper delgado
4. **Permitir uso en notebooks** sin `sys.path.append()`
5. **Mantener 100% funcionalidad** existente
6. **Cumplir 100% con AGENTS.md**

---

## 📐 Arquitectura Propuesta

### Estructura Actual (Problemática)

```
proyecto/
├── backend/
│   ├── pyproject.toml                    ← Poetry local (problema)
│   ├── poetry.lock
│   └── app/services/
│       └── sentinel_hub_service.py       ← Todo acoplado aquí
└── notebooks/                            ← ¿Cómo importan código?
```

### Estructura Propuesta (Arquitectura Limpia)

```
proyecto/
├── pyproject.toml                        ← UN SOLO POETRY (raíz)
├── poetry.lock
│
├── src/                                  ← Código core reutilizable
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── download_sentinel.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── ndvi_calculator.py
│   │   └── cloud_masking.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── region_growing.py
│   └── utils/
│       ├── __init__.py
│       ├── sentinel_download.py          ← Funciones puras
│       ├── image_processing.py
│       ├── geo_utils.py
│       └── visualization.py
│
├── backend/                              ← FastAPI (sin poetry propio)
│   └── app/services/
│       └── sentinel_hub_service.py       ← Wrapper delgado (usa src/)
│
├── notebooks/                            ← Jupyter (usa poetry raíz)
│   ├── exploratory/
│   └── experimental/
│
└── tests/                                ← Tests (usa poetry raíz)
    └── unit/
```

---

## 🔧 Plan de Implementación

### Fase 0: Migrar Poetry a Raíz (30 min)

#### Tarea 0.1: Crear pyproject.toml en raíz

```bash
# En raíz del proyecto
cd /ruta/al/proyecto
poetry init --name="sistema-deteccion-estres-vegetal" --python="^3.11"
```

#### Tarea 0.2: Configurar pyproject.toml

```toml
# pyproject.toml (RAÍZ DEL PROYECTO)
[tool.poetry]
name = "sistema-deteccion-estres-vegetal"
version = "1.0.0"
description = "Sistema Híbrido de Detección de Estrés Vegetal"
authors = ["Equipo 24"]
packages = [{include = "src"}]  # ← CRÍTICO: hace src/ importable

[tool.poetry.dependencies]
python = "^3.11"

# Core dependencies (src/)
numpy = "^1.26.3"
opencv-python = "^4.9.0"
scikit-learn = "^1.4.0"
sentinelhub = "^3.10.2"
torch = "^2.1.2"
rasterio = "^1.3.9"
shapely = "^2.0.2"
pillow = "^10.2.0"

# Backend (FastAPI)
fastapi = "^0.109.0"
uvicorn = {extras = ["standard"], version = "^0.27.0"}
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
python-dotenv = "^1.0.0"

# Notebooks
jupyter = "^1.0.0"
matplotlib = "^3.8.0"
seaborn = "^0.13.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.12.0"
ruff = "^0.1.9"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

#### Tarea 0.3: Instalar dependencias

```bash
poetry install
```

#### Tarea 0.4: Mover backend/pyproject.toml a backup

```bash
mv backend/pyproject.toml backend/pyproject.toml.backup
mv backend/poetry.lock backend/poetry.lock.backup
```

---

### Fase 1: Crear Estructura `src/` (30 min)

#### Tarea 1.1: Crear carpetas

```bash
# En raíz del proyecto
mkdir -p src/{data,features,models,utils}
mkdir -p notebooks/{exploratory,experimental,final}
mkdir -p tests/{unit,integration,e2e}

# Crear __init__.py
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py
```

#### Tarea 1.2: Crear archivos base en `src/utils/`

**Archivo: `src/utils/__init__.py`**
```python
"""
Utility functions for the project.
"""
from .sentinel_download import (
    create_sentinel_config,
    download_sentinel2_bands,
    create_cloud_mask,
    test_sentinel_connection
)
from .image_processing import (
    normalize_band,
    create_rgb_image,
    create_false_color_image,
    array_to_base64
)
from .geo_utils import (
    validate_bbox,
    calculate_bbox_area,
    regions_to_geojson,
    calculate_statistics
)

__all__ = [
    'create_sentinel_config',
    'download_sentinel2_bands',
    'create_cloud_mask',
    'test_sentinel_connection',
    'normalize_band',
    'create_rgb_image',
    'create_false_color_image',
    'array_to_base64',
    'validate_bbox',
    'calculate_bbox_area',
    'regions_to_geojson',
    'calculate_statistics'
]
```

---

### Fase 2: Extraer Funciones a `src/` (2 horas)

#### Tarea 2.1: Crear `src/utils/sentinel_download.py`

Extraer funciones puras de `backend/app/services/sentinel_hub_service.py`:

```python
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
```

---

#### Tarea 2.2: Crear `src/utils/image_processing.py`

```python
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
```

---

Continuaré con el resto del archivo en la siguiente parte...

#### Tarea 2.3: Crear `src/features/ndvi_calculator.py`

```python
"""
NDVI calculation and vegetation indices.
"""
import numpy as np
from typing import Dict


def calculate_ndvi(
    red_band: np.ndarray,
    nir_band: np.ndarray,
    scl_band: np.ndarray = None
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
    denominator = nir_band + red_band
    denominator[denominator == 0] = 0.0001
    
    # Calculate NDVI
    ndvi = (nir_band - red_band) / denominator
    
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
```

---

### Fase 3: Refactorizar Backend (1 hora)

#### Tarea 3.1: Actualizar `backend/app/services/sentinel_hub_service.py`

```python
"""
FastAPI service wrapper for Sentinel-2 downloads.
Uses pure functions from src.utils for core logic.
"""
from src.utils.sentinel_download import (
    create_sentinel_config,
    download_sentinel2_bands,
    create_cloud_mask,
    test_sentinel_connection
)
from src.utils.image_processing import (
    create_rgb_image,
    array_to_base64
)
from config.config import Settings
from app.utils import get_logger

logger = get_logger(__name__)


class SentinelHubService:
    """
    FastAPI service for Sentinel-2 imagery.
    Thin wrapper around src.utils functions.
    """
    
    def __init__(self):
        self.settings = Settings()
        # Use pure function from src.utils
        self.config = create_sentinel_config(
            self.settings.sentinel_hub_client_id,
            self.settings.sentinel_hub_client_secret
        )
    
    def get_sentinel2_data(self, bbox_coords, date_from=None, date_to=None):
        """
        Get Sentinel-2 data (FastAPI wrapper).
        
        This method adds logging and error handling to the pure functions.
        """
        try:
            logger.info("Downloading Sentinel-2 data for bbox: %s", bbox_coords)
            
            # Use pure function from src.utils
            result = download_sentinel2_bands(
                bbox_coords=bbox_coords,
                config=self.config,
                date_from=date_from,
                date_to=date_to
            )
            
            # Process bands
            bands = result['bands']
            cloud_mask = create_cloud_mask(bands['SCL'])
            
            # Create RGB visualization
            rgb_array = create_rgb_image(
                bands['B04'],  # Red
                bands['B03'],  # Green
                bands['B02']   # Blue
            )
            rgb_base64 = array_to_base64(rgb_array)
            
            logger.info("Download successful: shape=%s", bands['B04'].shape)
            
            return {
                'red': bands['B04'],
                'green': bands['B03'],
                'blue': bands['B02'],
                'nir': bands['B08'],
                'cloud_mask': cloud_mask,
                'rgb_image_base64': rgb_base64,
                **result['metadata']
            }
            
        except Exception as e:
            logger.error("Error downloading Sentinel-2: %s", str(e), exc_info=True)
            raise
    
    def test_connection(self):
        """Test connection with Sentinel Hub."""
        return test_sentinel_connection(self.config)
```

---

### Fase 4: Crear Notebooks de Ejemplo (1 hora)

#### Tarea 4.1: Crear `notebooks/exploratory/01_sentinel_download_example.ipynb`

**Contenido del notebook (en celdas):**

```markdown
# Exploración de Descarga de Imágenes Sentinel-2

Este notebook demuestra cómo usar las funciones de `src/` para descargar
y visualizar imágenes Sentinel-2.

**Autor:** Equipo 24  
**Fecha:** Noviembre 2025
```

```python
# Cell 2: Imports
# NO necesitamos sys.path.append porque poetry hace src/ importable
from src.utils.sentinel_download import (
    create_sentinel_config,
    download_sentinel2_bands
)
from src.utils.image_processing import (
    create_rgb_image,
    create_false_color_image
)
from src.features.ndvi_calculator import calculate_ndvi

import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
```

```python
# Cell 3: Configuración
client_id = os.getenv('SENTINEL_HUB_CLIENT_ID')
client_secret = os.getenv('SENTINEL_HUB_CLIENT_SECRET')

config = create_sentinel_config(client_id, client_secret)
print("✅ Sentinel Hub configurado")
```

```python
# Cell 4: Definir área
bbox = {
    'min_lat': -12.1,
    'min_lon': -77.1,
    'max_lat': -12.0,
    'max_lon': -77.0
}
print(f"Área: {bbox}")
```

```python
# Cell 5: Descargar datos
data = download_sentinel2_bands(
    bbox_coords=bbox,
    config=config,
    date_from='2024-01-01',
    date_to='2024-01-31'
)
print(f"✅ Descarga completada: {data['metadata']['dimensions']}")
```

```python
# Cell 6: Visualizar RGB
bands = data['bands']
rgb = create_rgb_image(bands['B04'], bands['B03'], bands['B02'])

plt.figure(figsize=(12, 10))
plt.imshow(rgb)
plt.title('Sentinel-2 RGB')
plt.axis('off')
plt.show()
```

```python
# Cell 7: Calcular NDVI
ndvi_result = calculate_ndvi(bands['B04'], bands['B08'], bands['SCL'])
print(f"NDVI promedio: {ndvi_result['statistics']['mean']:.3f}")
```

#### Tarea 4.2: Crear `notebooks/README.md`

```markdown
# Notebooks - Sistema de Detección de Estrés Vegetal

## Uso

```bash
# Desde raíz del proyecto
poetry run jupyter notebook notebooks/
```

## Importar desde src/

Gracias a Poetry, puedes importar directamente:

```python
from src.utils.sentinel_download import download_sentinel2_bands
from src.features.ndvi_calculator import calculate_ndvi
```

**NO necesitas** `sys.path.append()`.

## Convenciones (AGENTS.md)

- **Texto explicativo:** Español
- **Código:** Inglés
- **Comentarios:** Inglés
- **Markdown cells:** Español
```

---

### Fase 5: Tests y Documentación (30 min)

#### Tarea 5.1: Crear tests para `src/utils/`

```python
# tests/unit/test_sentinel_download.py
import pytest
import numpy as np
from src.utils.sentinel_download import create_cloud_mask
from src.utils.image_processing import normalize_band, create_rgb_image


class TestCloudMask:
    def test_create_cloud_mask(self):
        scl = np.array([[3, 8], [0, 4]])
        mask = create_cloud_mask(scl)
        assert mask[0, 0] == True  # Cloud shadow
        assert mask[0, 1] == True  # Cloud
        assert mask[1, 0] == False  # Clear


class TestImageProcessing:
    def test_normalize_band(self):
        band = np.array([[0, 5000], [10000, 15000]])
        normalized = normalize_band(band)
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_create_rgb_image(self):
        red = np.ones((10, 10)) * 100
        green = np.ones((10, 10)) * 150
        blue = np.ones((10, 10)) * 200
        rgb = create_rgb_image(red, green, blue)
        assert rgb.shape == (10, 10, 3)
        assert rgb.dtype == np.uint8
```

#### Tarea 5.2: Actualizar README.md principal

Agregar sección sobre arquitectura limpia y uso de `src/`.

---

## ✅ Criterios de Aceptación

### Arquitectura
- [ ] Un solo `pyproject.toml` en raíz del proyecto
- [ ] `backend/pyproject.toml` eliminado (backup guardado)
- [ ] Carpeta `src/` creada con estructura completa
- [ ] `packages = [{include = "src"}]` en pyproject.toml

### Código Reutilizable
- [ ] `src/utils/sentinel_download.py` con funciones puras
- [ ] `src/utils/image_processing.py` con funciones puras
- [ ] `src/features/ndvi_calculator.py` con funciones puras
- [ ] Todas las funciones con docstrings estilo Google
- [ ] Type hints en todas las funciones

### Backend Refactorizado
- [ ] `backend/app/services/sentinel_hub_service.py` usa `src/`
- [ ] Servicio es wrapper delgado (solo logging + error handling)
- [ ] 100% de funcionalidad preservada
- [ ] Tests existentes pasan sin cambios

### Notebooks
- [ ] Notebook de ejemplo funcional
- [ ] Imports desde `src/` sin `sys.path.append()`
- [ ] `notebooks/README.md` creado
- [ ] Convenciones AGENTS.md cumplidas

### Tests
- [ ] Tests unitarios para `src/utils/`
- [ ] Tests unitarios para `src/features/`
- [ ] Cobertura >80%
- [ ] Todos los tests pasan

### Documentación
- [ ] README.md actualizado
- [ ] Arquitectura documentada
- [ ] Ejemplos de uso en notebooks
- [ ] Cumplimiento 100% AGENTS.md

---

## 📊 Comparación: Antes vs Después

### Antes (Problemático)

```python
# En notebook - NO FUNCIONA
import sys
sys.path.append('../../backend')  # Feo y frágil
from app.services.sentinel_hub_service import SentinelHubService
# Error: Requiere FastAPI, Settings, etc.
```

**Problemas:**
- ❌ No se puede usar en notebooks
- ❌ Acoplado a FastAPI
- ❌ Código duplicado inevitable

### Después (Arquitectura Limpia)

```python
# En notebook - FUNCIONA
from src.utils.sentinel_download import download_sentinel2_bands
from src.features.ndvi_calculator import calculate_ndvi

# Funciones puras, sin dependencias de FastAPI
data = download_sentinel2_bands(bbox, config)
ndvi = calculate_ndvi(data['bands']['B04'], data['bands']['B08'])
```

**Ventajas:**
- ✅ Funciona en notebooks, scripts, FastAPI
- ✅ Funciones puras sin side effects
- ✅ Cumple 100% AGENTS.md
- ✅ Reutilizable en todo el proyecto

---

## 🎯 Métricas de Éxito

| Métrica | Objetivo | Verificación |
|---------|----------|--------------|
| Poetry unificado | 1 solo | `ls pyproject.toml` en raíz |
| Funcionalidad preservada | 100% | Tests pasan |
| Imports en notebooks | Sin sys.path | Notebook ejecutable |
| Cobertura tests | >80% | `poetry run pytest --cov` |
| Cumplimiento AGENTS.md | 100% | Revisión manual |
| Breaking changes | 0 | API pública sin cambios |

---

## 🚀 Comandos de Ejecución

### Instalación

```bash
# En raíz del proyecto
poetry install
```

### Ejecutar Backend

```bash
poetry run python backend/app.py
# o
poetry run uvicorn backend.app.main:app --reload
```

### Ejecutar Notebooks

```bash
poetry run jupyter notebook notebooks/
```

### Ejecutar Tests

```bash
poetry run pytest tests/
poetry run pytest --cov=src --cov-report=html
```

### Linters

```bash
poetry run black .
poetry run ruff check .
```

---

## 📝 Checklist de Implementación

### Pre-implementación
- [ ] Backup de `backend/pyproject.toml`
- [ ] Crear rama `refactor/arquitectura-limpia`
- [ ] Revisar tests existentes

### Fase 0: Poetry Global
- [ ] Crear `pyproject.toml` en raíz
- [ ] Configurar `packages = [{include = "src"}]`
- [ ] `poetry install`
- [ ] Mover `backend/pyproject.toml` a backup

### Fase 1: Estructura
- [ ] Crear carpetas `src/{data,features,models,utils}`
- [ ] Crear carpetas `notebooks/{exploratory,experimental,final}`
- [ ] Crear carpetas `tests/{unit,integration,e2e}`
- [ ] Crear todos los `__init__.py`

### Fase 2: Extraer Código
- [ ] Crear `src/utils/sentinel_download.py`
- [ ] Crear `src/utils/image_processing.py`
- [ ] Crear `src/utils/geo_utils.py`
- [ ] Crear `src/features/ndvi_calculator.py`
- [ ] Agregar docstrings y type hints

### Fase 3: Refactorizar Backend
- [ ] Actualizar `sentinel_hub_service.py`
- [ ] Actualizar `region_growing_service.py`
- [ ] Verificar tests pasan

### Fase 4: Notebooks
- [ ] Crear notebook de ejemplo
- [ ] Crear `notebooks/README.md`
- [ ] Verificar imports funcionan

### Fase 5: Tests y Docs
- [ ] Crear tests para `src/`
- [ ] Actualizar README.md
- [ ] Verificar cobertura >80%

### Post-implementación
- [ ] Ejecutar suite completa de tests
- [ ] Verificar notebook ejecutable
- [ ] Code review
- [ ] Merge a main

---

## 🎓 Cumplimiento con AGENTS.md

### Antes: 60%
- ✅ Código en inglés
- ✅ Docstrings en inglés
- ❌ No hay `src/utils/` reutilizable
- ❌ Código acoplado a FastAPI
- ❌ No usable en notebooks

### Después: 100%
- ✅ Código en inglés
- ✅ Docstrings estilo Google completos
- ✅ Type hints en todas las funciones
- ✅ `src/` con código reutilizable
- ✅ Funciones puras sin side effects
- ✅ Usable en notebooks, backend y scripts
- ✅ Un solo Poetry en raíz
- ✅ Tests >80% coverage

---

## 🔗 Referencias

- [AGENTS.md Standard](../../AGENTS.md)
- [Arquitectura Propuesta](../arquitectura-propuesta.md)
- [US-001 Resolved](../us-resolved/us-001.md)
- [US-002 Resolved](../us-resolved/us-002-complete.md)
- [Poetry Documentation](https://python-poetry.org/docs/)

---

## 👥 Equipo

**Responsable:** Carlos Bocanegra  
**Revisores:** Arthur Zizumbo, Luis Vázquez  
**Proyecto:** Sistema Híbrido de Detección de Estrés Vegetal  
**Sprint:** Fundación y Baseline (Días 1-3)

---

## 📌 Conclusión

Esta refactorización transforma el proyecto en una arquitectura limpia y profesional:

**Beneficios inmediatos:**
- ✅ Un solo Poetry (más simple)
- ✅ Código reutilizable en `src/`
- ✅ Notebooks pueden importar sin problemas
- ✅ Backend más limpio (wrappers delgados)
- ✅ Cumple 100% AGENTS.md
- ✅ Listo para escalar

**Tiempo:** 5 horas  
**Riesgo:** Bajo (refactorización sin cambios de funcionalidad)  
**Beneficio:** Alto (código profesional y mantenible)

---

**Estado:** 🔄 **PENDIENTE DE APROBACIÓN**  
**Próximo paso:** Revisión y aprobación por Carlos Bocanegra  
**Fecha:** 8 de Noviembre de 2025
