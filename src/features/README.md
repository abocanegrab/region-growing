# Features - HLS Processor Module

Este módulo proporciona funcionalidades para procesar imágenes Sentinel-2 en formato HLS (Harmonized Landsat Sentinel-2) y extraer embeddings semánticos usando el modelo fundacional Prithvi-EO-1.0.

---

## 📦 Contenido del Módulo

### Archivos Principales

```
src/features/
├── __init__.py
└── hls_processor.py       # Procesamiento HLS y extracción de embeddings
```

---

## 🎯 HLS Processor (`hls_processor.py`)

Módulo para procesamiento de imágenes HLS y extracción de embeddings semánticos.

### Funciones Principales

#### 1. `resample_band_to_10m()`

Remuestrea una banda de 20m a resolución de 10m usando interpolación bilineal.

```python
from src.features.hls_processor import resample_band_to_10m
import numpy as np

# Banda de 20m (100x100 píxeles)
band_20m = np.random.rand(100, 100)

# Remuestrear a 10m (200x200 píxeles)
band_10m = resample_band_to_10m(band_20m)

print(band_10m.shape)  # (200, 200)
```

**Parámetros:**
- `band`: array (H, W) con la banda de 20m
- `target_shape`: (opcional) forma objetivo específica, por defecto duplica dimensiones

**Retorna:**
- array (H*2, W*2) o target_shape con la banda remuestreada

---

#### 2. `stack_hls_bands()`

Apila las 6 bandas HLS en el orden correcto requerido por Prithvi.

```python
from src.features.hls_processor import stack_hls_bands
import numpy as np

bands = {
    'B02': np.random.rand(224, 224),  # Blue
    'B03': np.random.rand(224, 224),  # Green
    'B04': np.random.rand(224, 224),  # Red
    'B8A': np.random.rand(224, 224),  # NIR Narrow
    'B11': np.random.rand(224, 224),  # SWIR 1
    'B12': np.random.rand(224, 224)   # SWIR 2
}

# Apilar en orden correcto [B02, B03, B04, B8A, B11, B12]
hls_image = stack_hls_bands(bands)

print(hls_image.shape)  # (6, 224, 224)
```

**Parámetros:**
- `bands_dict`: diccionario con claves B02, B03, B04, B8A, B11, B12
- `validate`: (opcional) validar que todas las bandas estén presentes

**Retorna:**
- array (6, H, W) con las bandas apiladas en orden correcto

**⚠️ IMPORTANTE:** Debe usarse **B8A** (NIR Narrow, 20m), NO B08 (NIR Broad, 10m).

---

#### 3. `prepare_hls_image()`

Pipeline completo para preparar imagen HLS desde bandas separadas.

```python
from src.features.hls_processor import prepare_hls_image
import numpy as np

# Bandas a 10m (3 bandas)
bands_10m = {
    'B02': np.random.rand(224, 224),
    'B03': np.random.rand(224, 224),
    'B04': np.random.rand(224, 224)
}

# Bandas a 20m (3 bandas)
bands_20m = {
    'B8A': np.random.rand(112, 112),
    'B11': np.random.rand(112, 112),
    'B12': np.random.rand(112, 112)
}

# Preparar imagen HLS completa
hls_image = prepare_hls_image(bands_10m, bands_20m)

print(hls_image.shape)  # (6, 224, 224)
```

**Parámetros:**
- `bands_10m`: dict con B02, B03, B04 a resolución 10m
- `bands_20m`: dict con B8A, B11, B12 a resolución 20m

**Retorna:**
- array (6, H, W) con imagen HLS preparada para Prithvi

**Proceso:**
1. Remuestrea bandas de 20m → 10m
2. Combina todas las bandas
3. Apila en el orden correcto

---

#### 4. `extract_embeddings()`

Extrae embeddings semánticos usando el modelo Prithvi.

```python
from src.features.hls_processor import extract_embeddings
import numpy as np

# Imagen HLS preparada
hls_image = np.random.rand(6, 224, 224)

# Extraer embeddings
embeddings = extract_embeddings(hls_image, use_simple_model=False)

print(embeddings.shape)  # (224, 224, 256)
print(np.linalg.norm(embeddings[0, 0]))  # ~1.0 (normalizado L2)
```

**Parámetros:**
- `hls_image`: array (6, H, W) con imagen HLS
- `use_simple_model`: usar modelo simple (desarrollo) o Prithvi real (producción)
- `device`: 'cuda' o 'cpu'

**Retorna:**
- array (H, W, 256) con embeddings normalizados L2

**Características:**
- ✅ Embeddings de 256 dimensiones por píxel
- ✅ Normalización L2 automática
- ✅ Soporta GPU y CPU
- ✅ Modo simple para pruebas sin descargar Prithvi

---

#### 5. `normalize_embeddings_l2()`

Normaliza embeddings a norma L2 unitaria.

```python
from src.features.hls_processor import normalize_embeddings_l2
import numpy as np

embeddings = np.random.rand(224, 224, 256)
normalized = normalize_embeddings_l2(embeddings)

# Verificar normalización
norm = np.linalg.norm(normalized[0, 0])
print(f"Norma: {norm:.6f}")  # 1.000000
```

**Parámetros:**
- `embeddings`: array (..., D) con embeddings

**Retorna:**
- array misma forma con norma L2 = 1 por vector

**Uso:**
- Necesario para similitud coseno
- Ya aplicado automáticamente en `extract_embeddings()`

---

#### 6. `compute_cosine_similarity()`

Calcula similitud coseno píxel a píxel entre dos mapas de embeddings.

```python
from src.features.hls_processor import compute_cosine_similarity
import numpy as np

embeddings_a = np.random.rand(224, 224, 256)
embeddings_b = np.random.rand(224, 224, 256)

# Normalizar
embeddings_a = normalize_embeddings_l2(embeddings_a)
embeddings_b = normalize_embeddings_l2(embeddings_b)

# Calcular similitud
similarity = compute_cosine_similarity(embeddings_a, embeddings_b)

print(similarity.shape)  # (224, 224)
print(f"Similitud promedio: {similarity.mean():.3f}")
print(f"Rango: [{similarity.min():.3f}, {similarity.max():.3f}]")
```

**Parámetros:**
- `embeddings_a`: array (H, W, D) normalizado
- `embeddings_b`: array (H, W, D) normalizado

**Retorna:**
- array (H, W) con similitud en rango [-1, 1]

**Interpretación:**
- `1.0`: Idénticos
- `0.5-0.99`: Alta similitud
- `0.0-0.49`: Similitud media
- `< 0.0`: Disimilares

---

#### 7. `save_embeddings()` y `load_embeddings()`

Persiste y carga embeddings con metadata.

```python
from src.features.hls_processor import save_embeddings, load_embeddings
import numpy as np
from pathlib import Path

# Guardar embeddings
embeddings = np.random.rand(224, 224, 256)
metadata = {
    'zone': 'mexicali',
    'date': '2024-01-15',
    'bbox': [32.4, -115.4, 32.5, -115.3]
}

output_path = Path('embeddings/mexicali_20240115.npz')
save_embeddings(embeddings, output_path, metadata)

# Cargar embeddings
loaded_embeddings, loaded_metadata = load_embeddings(output_path)

print(loaded_embeddings.shape)  # (224, 224, 256)
print(loaded_metadata['zone'])  # 'mexicali'
```

**Formato del archivo .npz:**
- `embeddings`: array principal
- Metadata como arrays individuales

---

#### 8. `visualize_embeddings_pca()`

Visualiza embeddings usando PCA para reducir a 3 dimensiones (RGB).

```python
from src.features.hls_processor import visualize_embeddings_pca
import numpy as np

embeddings = np.random.rand(224, 224, 256)

# Reducir a RGB usando PCA
rgb_image = visualize_embeddings_pca(embeddings, n_components=3)

print(rgb_image.shape)  # (224, 224, 3)
print(f"Rango: [{rgb_image.min():.2f}, {rgb_image.max():.2f}]")  # [0.00, 1.00]

# Guardar visualización
import matplotlib.pyplot as plt
plt.imshow(rgb_image)
plt.title('Embeddings visualizados con PCA')
plt.axis('off')
plt.savefig('embeddings_pca.png', dpi=150, bbox_inches='tight')
```

**Parámetros:**
- `embeddings`: array (H, W, D)
- `n_components`: número de componentes PCA (3 para RGB)

**Retorna:**
- array (H, W, n_components) normalizado a [0, 1]

---

## 🌎 Zonas de México para Testing

El sistema está configurado con 3 zonas agrícolas mexicanas:

### 1. Mexicali (Baja California)
```python
MEXICALI = {
    'name': 'Mexicali',
    'bbox': [32.4, -115.4, 32.5, -115.3],
    'description': 'Valle de Mexicali - Agricultura intensiva de riego'
}
```
- **Cultivos:** Trigo, algodón, alfalfa
- **Características:** Zona desértica con riego por canales

### 2. Bajío (Guanajuato)
```python
BAJIO = {
    'name': 'Bajío',
    'bbox': [20.8, -101.5, 20.9, -101.4],
    'description': 'El Bajío - Región agrícola diversificada'
}
```
- **Cultivos:** Sorgo, maíz, hortalizas
- **Características:** Región de tierras altas

### 3. Sinaloa
```python
SINALOA = {
    'name': 'Sinaloa',
    'bbox': [24.7, -107.5, 24.8, -107.4],
    'description': 'Valle de Culiacán - Agricultura de exportación'
}
```
- **Cultivos:** Tomate, chile, maíz
- **Características:** Valle costero con agricultura tecnificada

---

## 🚀 Ejemplos de Uso Completo

### Ejemplo 1: Pipeline Básico

```python
from src.features.hls_processor import (
    prepare_hls_image,
    extract_embeddings,
    save_embeddings
)
from src.utils.sentinel_download import download_hls_bands
from pathlib import Path

# 1. Descargar bandas HLS
bbox = [32.4, -115.4, 32.5, -115.3]  # Mexicali
date_from = '2024-01-15'
date_to = '2024-01-15'

result = download_hls_bands(bbox, date_from, date_to)

# 2. Preparar imagen HLS
hls_image = prepare_hls_image(
    result['bands_10m'],
    result['bands_20m']
)

# 3. Extraer embeddings
embeddings = extract_embeddings(hls_image, use_simple_model=False)

# 4. Guardar con metadata
metadata = {
    'zone': 'mexicali',
    'date': date_from,
    'bbox': bbox
}
output_path = Path('embeddings/mexicali_20240115.npz')
save_embeddings(embeddings, output_path, metadata)

print(f"Embeddings guardados en {output_path}")
print(f"Forma: {embeddings.shape}")
```

### Ejemplo 2: Comparar Dos Regiones

```python
from src.features.hls_processor import (
    extract_embeddings,
    compute_cosine_similarity,
    prepare_hls_image
)
from src.utils.sentinel_download import download_hls_bands
import numpy as np
import matplotlib.pyplot as plt

# Descargar dos zonas
bbox_a = [32.4, -115.4, 32.5, -115.3]  # Mexicali
bbox_b = [24.7, -107.5, 24.8, -107.4]  # Sinaloa
date = '2024-01-15'

result_a = download_hls_bands(bbox_a, date, date)
result_b = download_hls_bands(bbox_b, date, date)

# Preparar imágenes
hls_a = prepare_hls_image(result_a['bands_10m'], result_a['bands_20m'])
hls_b = prepare_hls_image(result_b['bands_10m'], result_b['bands_20m'])

# Extraer embeddings
embeddings_a = extract_embeddings(hls_a)
embeddings_b = extract_embeddings(hls_b)

# Calcular similitud
similarity = compute_cosine_similarity(embeddings_a, embeddings_b)

# Visualizar
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(similarity.flatten(), bins=50)
plt.xlabel('Similitud Coseno')
plt.ylabel('Frecuencia')
plt.title('Distribución de Similitud')

plt.subplot(1, 2, 2)
plt.imshow(similarity, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(label='Similitud')
plt.title('Mapa de Similitud')
plt.axis('off')

plt.tight_layout()
plt.savefig('similarity_analysis.png', dpi=150)

print(f"Similitud promedio: {similarity.mean():.3f}")
print(f"Similitud máxima: {similarity.max():.3f}")
print(f"Similitud mínima: {similarity.min():.3f}")
```

### Ejemplo 3: Visualización PCA

```python
from src.features.hls_processor import (
    extract_embeddings,
    visualize_embeddings_pca,
    prepare_hls_image
)
from src.utils.sentinel_download import download_hls_bands
import matplotlib.pyplot as plt

# Descargar y procesar
bbox = [20.8, -101.5, 20.9, -101.4]  # Bajío
date = '2024-01-15'

result = download_hls_bands(bbox, date, date)
hls_image = prepare_hls_image(result['bands_10m'], result['bands_20m'])
embeddings = extract_embeddings(hls_image)

# Visualizar con PCA
rgb_pca = visualize_embeddings_pca(embeddings, n_components=3)

# Crear visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Imagen RGB original (bandas B04, B03, B02)
rgb_original = np.stack([
    hls_image[2],  # B04 (Red)
    hls_image[1],  # B03 (Green)
    hls_image[0]   # B02 (Blue)
], axis=-1)
rgb_original = (rgb_original - rgb_original.min()) / (rgb_original.max() - rgb_original.min())

axes[0].imshow(rgb_original)
axes[0].set_title('RGB Original (Sentinel-2)')
axes[0].axis('off')

axes[1].imshow(rgb_pca)
axes[1].set_title('Embeddings PCA (Prithvi)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('pca_comparison.png', dpi=150)
```

---

## 📊 Detalles Técnicos

### Bandas HLS Requeridas

| Banda | Nombre | Resolución | Wavelength | Orden |
|-------|--------|-----------|-----------|-------|
| B02 | Blue | 10m | 490 nm | 1 |
| B03 | Green | 10m | 560 nm | 2 |
| B04 | Red | 10m | 665 nm | 3 |
| **B8A** | **NIR Narrow** | **20m** | **865 nm** | **4** |
| B11 | SWIR 1 | 20m | 1610 nm | 5 |
| B12 | SWIR 2 | 20m | 2190 nm | 6 |

**⚠️ CRÍTICO:** Usar **B8A** (NIR Narrow, 20m), NO B08 (NIR Broad, 10m).

### Modelo Prithvi

- **Nombre:** Prithvi-EO-1.0-100M
- **Arquitectura:** Vision Transformer (ViT)
- **Pre-entrenamiento:** HLS imagery (NASA/IBM)
- **Embeddings:** 256 dimensiones por píxel
- **Entrada:** (6, H, W) formato HLS
- **Salida:** (H, W, 256) normalizado L2

### Formato de Embeddings

Los embeddings se guardan en formato `.npz` de NumPy:

```python
# Estructura del archivo
{
    'embeddings': np.array (H, W, 256),
    'zone': str,
    'date': str,
    'bbox': list[float],
    # ... metadata adicional
}
```

---

## 🧪 Testing

Ejecutar tests unitarios:

```bash
# Todos los tests del módulo
poetry run pytest tests/unit/test_hls_processor.py -v

# Con cobertura
poetry run pytest tests/unit/test_hls_processor.py --cov=src.features.hls_processor --cov-report=html

# Test específico
poetry run pytest tests/unit/test_hls_processor.py::TestStackHlsBands::test_correct_band_order -v
```

---

## 📚 Scripts Disponibles

### `scripts/download_hls_image.py`

Descargar imágenes HLS de zonas mexicanas:

```bash
# Una zona específica
poetry run python scripts/download_hls_image.py --zone mexicali --date-from 2024-01-15

# Todas las zonas
poetry run python scripts/download_hls_image.py --all --date-from 2024-01-15

# Rango de fechas
poetry run python scripts/download_hls_image.py --zone sinaloa --date-from 2024-01-01 --date-to 2024-01-31
```

### `scripts/test_embeddings.py`

Probar extracción de embeddings:

```bash
# Extraer embeddings de una zona
poetry run python scripts/test_embeddings.py --zone bajio

# Comparar todas las zonas
poetry run python scripts/test_embeddings.py --compare

# Modo desarrollo (sin descargar Prithvi)
poetry run python scripts/test_embeddings.py --zone mexicali --use-simple-model
```

---

## 🔗 Referencias

- [Sentinel Hub API](https://docs.sentinel-hub.com/)
- [Prithvi Model (HuggingFace)](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M)
- [HLS Product Guide](https://lpdaac.usgs.gov/documents/1326/HLS_User_Guide_V2.pdf)
- [US-006 Documentation](../../docs/us-resolved/us-006.md)

---

**Última actualización:** Diciembre 2024  
**Versión:** 1.0  
**Mantenido por:** Equipo 24
