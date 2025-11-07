# Backend - Sistema de Detección de Estrés Vegetal

API FastAPI para análisis de estrés vegetal mediante Region Growing sobre imágenes satelitales Sentinel-2.

## Descripción General

Este backend implementa un pipeline completo de procesamiento de imágenes satelitales:
1. **Descarga de datos** desde Sentinel Hub API
2. **Cálculo de NDVI** con manejo robusto de nubes
3. **Segmentación** usando algoritmo Region Growing
4. **Clasificación** de regiones por nivel de estrés vegetal
5. **Conversión geoespacial** a GeoJSON para visualización

## 🚀 Migración a FastAPI + Poetry

**Versión 2.0** - Migrado de Flask a FastAPI con Poetry para gestión de dependencias.
**Versión 2.0.1** - Mejoras de logging y timeouts para producción.

### ¿Por qué FastAPI?
- ⚡ **3-4x más rápido** que Flask (ASGI vs WSGI)
- 📝 **Documentación automática** con Swagger UI y ReDoc
- 🔒 **Validación automática** con Pydantic
- ✅ **Type safety** nativo
- ⚙️ **Async/await** para mejor rendimiento con APIs externas

### ¿Por qué Poetry?
- 📦 **Gestión moderna** de dependencias Python
- 🔒 **Lock file** para reproducibilidad exacta
- 🎯 **Resolución determinística** de dependencias
- 🚀 **Entornos virtuales automáticos**

### ✨ Nuevas Mejoras (v2.0.1)
- 📊 **Logging profesional** con Python logging module
- ⏱️ **Timeouts configurables** para prevenir requests colgadas
- 🔍 **Mejor observabilidad** con logs estructurados
- 🛡️ **Manejo de errores mejorado** con mensajes claros

Ver detalles en: [LOGGING_AND_TIMEOUT_IMPROVEMENTS.md](./LOGGING_AND_TIMEOUT_IMPROVEMENTS.md)

## Estructura del Proyecto

```
backend/
├── app/
│   ├── main.py                               # FastAPI application
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py                     # Health check endpoints
│   │   │   └── analysis.py                   # Analysis endpoints
│   │   └── schemas/
│   │       ├── requests.py                   # Pydantic request models
│   │       └── responses.py                  # Pydantic response models
│   ├── services/
│   │   ├── region_growing_service.py         # Main analysis orchestrator
│   │   ├── sentinel_hub_service.py           # Sentinel Hub API integration
│   │   ├── ndvi_service.py                   # Vegetation indices calculation
│   │   ├── region_growing_algorithm.py       # Region Growing algorithm
│   │   └── geo_converter_service.py          # Pixel → coordinate conversion
│   └── utils/                                # Utilities
│       ├── logging_config.py                  # Centralized logging setup
│       └── timeout.py                         # Timeout utilities
├── config/
│   └── config.py                             # Pydantic Settings configuration
├── tests/
│   ├── test_health.py                        # Health endpoint tests
│   └── test_analysis.py                      # Analysis endpoint tests
├── app.py                                    # Entry point (Uvicorn)
├── pyproject.toml                            # Poetry configuration
├── poetry.lock                               # Dependency lock file
└── .env                                      # Environment variables
```

## Tecnologías y Librerías

| Librería | Versión | Uso |
|----------|---------|-----|
| **FastAPI** | 0.109+ | Modern web framework with automatic docs |
| **Uvicorn** | 0.27+ | ASGI server for FastAPI |
| **Pydantic** | 2.5+ | Data validation and settings management |
| sentinelhub | 3.10+ | Sentinel Hub API client |
| NumPy | 1.26+ | Matrix operations and NDVI calculation |
| OpenCV | 4.9+ | Contour detection |
| Shapely | 2.0+ | Geometry and polygon simplification |
| Pillow | 10.0+ | RGB/NDVI image generation |

## 🚀 Quick Start

```bash
# 1. Instalar dependencias
poetry install

# 2. Configurar credenciales
cp .env.example .env
# Editar .env con credenciales Sentinel Hub

# 3. Ejecutar servidor
poetry run python app.py
```

**Servidor:** http://localhost:8000
**Swagger UI:** http://localhost:8000/api/docs

📖 **Para instrucciones detalladas de instalación y configuración, ver:** [docs/quickstart.md](../docs/quickstart.md)

## Uso

### Documentación API Interactiva

**Swagger UI (recomendado):**
```
http://localhost:8000/api/docs
```

**ReDoc (alternativo):**
```
http://localhost:8000/api/redoc
```

### Comandos Poetry útiles

```bash
# Instalar nueva dependencia
poetry add nombre-paquete

# Instalar dependencia de desarrollo
poetry add --group dev nombre-paquete

# Actualizar dependencias
poetry update

# Ver dependencias instaladas
poetry show

# Activar shell con entorno virtual
poetry shell

# Ejecutar comando en entorno virtual
poetry run python script.py

# Ejecutar tests
poetry run pytest

# Formatear código
poetry run black .

# Linter
poetry run ruff check .
```

## Endpoints API

### Health Check
```http
GET /health
```

Verifica que el servidor está corriendo.

**Respuesta:**
```json
{
  "status": "ok",
  "message": "Server is running"
}
```

### Test Conexión Sentinel Hub
```http
GET /api/analysis/test
```

Prueba la conectividad con Sentinel Hub API usando las credenciales configuradas.

**Respuesta exitosa:**
```json
{
  "success": true,
  "data": {
    "status": "success",
    "message": "Conexión exitosa con Sentinel Hub"
  }
}
```

### Analizar Región
```http
POST /api/analysis/analyze
Content-Type: application/json

{
  "bbox": {
    "min_lat": -12.05,
    "min_lon": -77.05,
    "max_lat": -11.95,
    "max_lon": -76.95
  },
  "date_from": "2024-01-01",  // Opcional
  "date_to": "2024-01-31"      // Opcional
}
```

**Parámetros:**
- `bbox` (requerido): Bounding box con coordenadas geográficas
  - `min_lat`, `min_lon`: Esquina suroeste
  - `max_lat`, `max_lon`: Esquina noreste
- `date_from` (opcional): Fecha inicio búsqueda (YYYY-MM-DD). Default: hace 30 días
- `date_to` (opcional): Fecha fin (YYYY-MM-DD). Default: hoy

**Respuesta exitosa:**
```json
{
  "success": true,
  "data": {
    "geojson": {
      "type": "FeatureCollection",
      "features": [...]
    },
    "statistics": {
      "total_area": 1250.5,
      "high_stress_area": 423.2,
      "medium_stress_area": 567.8,
      "low_stress_area": 259.5,
      "mean_ndvi": 0.412,
      "num_regions": 47,
      "num_high_stress_regions": 12,
      "num_medium_stress_regions": 23,
      "num_low_stress_regions": 12,
      "cloud_coverage": 15.3,
      "date_from": "2024-01-01",
      "date_to": "2024-01-31"
    },
    "images": {
      "rgb": "data:image/png;base64,...",
      "ndvi": "data:image/png;base64,..."
    }
  }
}
```

## Arquitectura de Servicios

### 1. `sentinel_hub_service.py`

**Responsabilidad**: Comunicación con Sentinel Hub API y descarga de imágenes satelitales.

**Funciones principales:**
- `get_sentinel2_data(bbox, date_from, date_to)`: Descarga bandas espectrales
- `test_connection()`: Verifica credenciales

**Bandas descargadas:**
- B02 (Blue), B03 (Green), B04 (Red) → Imagen RGB
- B08 (NIR) → Cálculo NDVI
- SCL (Scene Classification) → Máscara de nubes

**Características importantes:**
- Autenticación OAuth2 automática
- Normalización robusta de RGB:
  - Percentiles P2-P98 para evitar saturación
  - Gamma correction (0.8) para mejorar contraste
- Límite de 2500x2500 píxeles con ajuste automático si se excede
- Conversión a base64 para envío al frontend

**Código de normalización RGB:**
```python
# Usar percentiles para normalización robusta
p2, p98 = np.percentile(rgb_image, [2, 98])
rgb_normalized = (rgb_image - p2) / (p98 - p2 + 1e-10)
rgb_normalized = np.clip(rgb_normalized, 0, 1)

# Ajuste gamma para mejorar contraste
gamma = 0.8
rgb_normalized = np.power(rgb_normalized, gamma)
rgb_image = (rgb_normalized * 255).astype(np.uint8)
```

### 2. `ndvi_service.py`

**Responsabilidad**: Cálculo de índices de vegetación y estadísticas.

**Funciones principales:**
- `calculate_ndvi(red_band, nir_band, cloud_mask)`: Calcula NDVI con manejo de nubes

**Fórmula NDVI:**
```python
NDVI = (NIR - Red) / (NIR + Red)
```

**Características:**
- Evita división por cero con epsilon
- Retorna masked array (nubes marcadas)
- Estadísticas solo sobre píxeles válidos (sin nubes)
- Cálculo de % de cobertura de nubes

**Uso de masked arrays:**
```python
ndvi = (nir_band - red_band) / (denominator + epsilon)
ndvi_masked = np.ma.masked_array(ndvi, mask=cloud_mask)

# Estadísticas automáticamente ignoran nubes
mean = np.ma.mean(ndvi_masked)  # Solo píxeles válidos
```

### 3. `region_growing_algorithm.py`

**Responsabilidad**: Implementación del algoritmo Region Growing para segmentación.

**Clase principal:** `RegionGrowingAlgorithm(threshold, min_region_size)`

**Parámetros configurables:**
- `threshold = 0.1`: Umbral de similitud NDVI
- `min_region_size = 50`: Tamaño mínimo de región en píxeles

**Métodos públicos:**
```python
def region_growing(image, seeds=None):
    """
    Segmenta imagen usando Region Growing

    Returns:
        labeled_image: Array con etiquetas de regiones
        num_regions: Número de regiones detectadas
        regions_info: Lista con información de cada región
    """

def classify_regions_by_stress(regions_info):
    """
    Clasifica regiones por nivel de estrés:
    - high: NDVI < 0.3
    - medium: 0.3 ≤ NDVI < 0.5
    - low: NDVI ≥ 0.5
    """
```

**Manejo de nubes (CRÍTICO):**
```python
# Generar semillas solo en píxeles válidos
def _generate_seeds(image, grid_size=20):
    for y in range(...):
        for x in range(...):
            # Ignorar nubes (valor -999)
            if image[y, x] > -900:
                seeds.append((y, x))

# No propagar regiones hacia nubes
def _grow_region(image, seed_y, seed_x):
    pixel_value = image[y, x]
    if pixel_value < -900:  # Píxel con nube
        continue  # No agregar a región
```

**Algoritmo BFS:**
- 4-conectividad (arriba, abajo, izquierda, derecha)
- Criterio de similitud: `|pixel_value - seed_value| ≤ threshold`
- Queue para procesamiento eficiente

### 4. `region_growing_service.py`

**Responsabilidad**: Orquestador principal que coordina todo el pipeline.

**Función principal:**
```python
def analyze_stress(bbox, date_from=None, date_to=None):
    """
    Pipeline completo de análisis:
    1. Obtener datos Sentinel-2
    2. Calcular NDVI
    3. Aplicar Region Growing
    4. Convertir a GeoJSON
    5. Calcular estadísticas
    6. Generar visualizaciones
    """
```

**Flujo de datos:**
```
Sentinel Hub → Red/NIR/Nubes → NDVI masked →
→ Region Growing (con -999 para nubes) →
→ Regiones clasificadas → GeoJSON + Estadísticas
```

**Preparación para Region Growing:**
```python
# Convertir masked array rellenando nubes con -999
# Este valor especial se ignora en el algoritmo
ndvi_for_rg = np.ma.filled(ndvi_masked, fill_value=-999)
```

**Visualización NDVI:**
- Colormap personalizado: Rojo → Amarillo → Verde
- Nubes en gris (128, 128, 128)
- Conversión a base64 para frontend

### 5. `geo_converter_service.py`

**Responsabilidad**: Conversión de coordenadas píxel a geográficas y generación de GeoJSON.

**Funciones principales:**
```python
def regions_to_geojson(regions_info, bbox, image_shape):
    """
    Convierte regiones detectadas a formato GeoJSON
    - Calcula contornos con OpenCV
    - Convierte píxeles a lat/lon
    - Simplifica polígonos con Shapely
    """

def calculate_statistics(regions_info, classified_regions, image_shape, resolution):
    """
    Calcula estadísticas del análisis:
    - Áreas en hectáreas
    - Distribución por nivel de estrés
    - Número de regiones por categoría
    """
```

**Conversión píxel → lat/lon:**
```python
lat_per_pixel = (max_lat - min_lat) / height
lon_per_pixel = (max_lon - min_lon) / width

lat = max_lat - (y * lat_per_pixel)
lon = min_lon + (x * lon_per_pixel)
```

**Simplificación de polígonos:**
- Usa algoritmo Douglas-Peucker (Shapely)
- Tolerancia adaptativa basada en tamaño
- Reduce número de puntos para envío eficiente

## Consideraciones Técnicas Importantes

### 1. Manejo de Nubes

**Problema crítico resuelto**: Las nubes NO deben clasificarse como "estrés alto".

**Solución de 3 capas:**

**Capa 1 - NDVI Service:**
```python
# Crear masked array con máscara de nubes
ndvi_masked = np.ma.masked_array(ndvi, mask=cloud_mask)
# Estadísticas automáticamente ignoran píxeles masked
```

**Capa 2 - Region Growing Service:**
```python
# Rellenar con -999 en lugar de 0
ndvi_for_rg = np.ma.filled(ndvi_masked, fill_value=-999)
```

**Capa 3 - Region Growing Algorithm:**
```python
# Ignorar píxeles con valor -999 en TODA la lógica:
# - No generar semillas
# - No propagar regiones
# - No incluir en cálculos
```

**Resultado**: Las nubes se excluyen completamente del análisis.

### 2. Límites de Sentinel Hub

- **Tamaño máximo de imagen**: 2500 x 2500 píxeles
- **Resolución Sentinel-2**: 10m por píxel
- **Área máxima aproximada**: ~625 km² (25km x 25km)

**Manejo en código:**
```python
if width > 2500 or height > 2500:
    scale = min(2500 / width, 2500 / height)
    bbox_size = (int(width * scale), int(height * scale))
```

**Recomendación**: El frontend debe validar el tamaño ANTES de enviar al backend.

### 3. Optimizaciones de Rendimiento

**NumPy vectorizado para colormap:**
```python
# Evitar loops, usar operaciones vectorizadas
mask_low = ndvi_normalized < 0.5
ndvi_colored[mask_low, 0] = 255  # R
ndvi_colored[mask_low, 1] = (t_low * 255).astype(np.uint8)  # G
```

**Caching de autenticación Sentinel Hub:**
- El SDK maneja tokens OAuth2 automáticamente
- No hay necesidad de lógica adicional de refresh

## Configuración Avanzada

### Ajustar Parámetros del Algoritmo

**Archivo**: `app/services/region_growing_service.py`

```python
self.region_growing = RegionGrowingAlgorithm(
    threshold=0.1,        # Cambiar para más/menos sensibilidad
    min_region_size=50    # Cambiar para filtrar regiones pequeñas
)
```

**Efectos de los parámetros:**
- `threshold` bajo (0.05) → Más regiones pequeñas (sobre-segmentación)
- `threshold` alto (0.15) → Menos regiones grandes (sub-segmentación)
- `min_region_size` bajo (20) → Más detalles, más ruido
- `min_region_size` alto (100) → Menos ruido, puede perder detalles

### Ajustar Umbrales de Clasificación

**Archivo**: `app/services/region_growing_algorithm.py`

```python
def classify_regions_by_stress(regions_info):
    if mean_ndvi < 0.3:  # Cambiar umbrales según tipo de cultivo
        stress_level = 'high'
    elif mean_ndvi < 0.5:
        stress_level = 'medium'
    else:
        stress_level = 'low'
```

**Recomendaciones por tipo de vegetación:**
- Cultivos irrigados: (0.35, 0.55)
- Cultivos de secano: (0.25, 0.45)
- Bosques tropicales: (0.5, 0.7)

### Ajustar Cobertura de Nubes Máxima

**Archivo**: `app/services/sentinel_hub_service.py`

```python
input_data=[
    SentinelHubRequest.input_data(
        data_collection=DataCollection.SENTINEL2_L2A,
        time_interval=(date_from, date_to),
        maxcc=0.5  # Cambiar: 0.3 = estricto, 0.7 = permisivo
    )
]
```

## Troubleshooting

### Error: "Invalid credentials"
```
Solution: Verificar que SENTINEL_HUB_CLIENT_ID y SENTINEL_HUB_CLIENT_SECRET estén correctos en .env
```

### Error: "Image size exceeds 2500px"
```
Solution: El frontend debe validar el tamaño antes de enviar. Backend ajusta automáticamente si es posible.
```

### Error: "No data available for date range"
```
Solution: Ampliar rango de fechas o verificar que el área tiene cobertura Sentinel-2
```

### Performance lento con regiones grandes
```
Solution: Aumentar min_region_size o threshold en RegionGrowingAlgorithm
```

## Logging y Debugging

El sistema imprime logs detallados en cada paso:

```
[1/4] Obteniendo imagen Sentinel-2 para bbox: {...}
      Imagen obtenida: (512, 512), Nubes: 15.3%
[2/4] Calculando NDVI...
      NDVI medio: 0.412, rango: [0.023, 0.845]
[3/4] Aplicando Region Growing...
      Regiones detectadas: 47
      Alto estrés: 12, Medio: 23, Bajo: 12
[4/4] Convirtiendo a GeoJSON...
      Total área: 1250.50 ha
      Estrés alto: 423.20 ha
```

## Testing

### Test manual de endpoints

```bash
# Health check
curl http://localhost:5000/health

# Test Sentinel Hub
curl http://localhost:5000/api/analysis/test

# Análisis (ejemplo Lima, Perú)
curl -X POST http://localhost:5000/api/analysis/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": {
      "min_lat": -12.05,
      "min_lon": -77.05,
      "max_lat": -11.95,
      "max_lon": -76.95
    }
  }'
```

## Próximos Pasos

### Implementaciones Pendientes
- ✅ Integración Sentinel Hub
- ✅ Cálculo NDVI
- ✅ Region Growing
- ✅ Conversión GeoJSON
- ✅ Generación de imágenes RGB
- ✅ Manejo robusto de nubes
- ⏳ Base de datos para historial
- ⏳ Cache de imágenes
- ⏳ Procesamiento asíncrono (Celery)
- ⏳ Tests unitarios

### Mejoras Futuras
1. **Índices adicionales**: EVI, SAVI, NDWI
2. **Machine Learning**: Clasificación de cultivos
3. **Análisis temporal**: Comparar múltiples fechas
4. **Alertas automáticas**: Notificaciones de cambios
5. **Optimización**: Cache de tiles, procesamiento paralelo

## Licencia

Código bajo MIT License. Datos Sentinel-2 de ESA (acceso libre y gratuito).
