# Guía de Descarga de Imágenes Sentinel-2

## Resumen

Esta guía explica cómo descargar imágenes satelitales Sentinel-2 para el proyecto de detección de estrés vegetal.

## Requisitos Previos

### 1. Credenciales de Sentinel Hub

Necesitas una cuenta gratuita de Sentinel Hub:

1. Regístrate en https://www.sentinel-hub.com/
2. Crea una configuración (Configuration)
3. Obtén tus credenciales:
   - `CLIENT_ID`
   - `CLIENT_SECRET`

### 2. Configurar Credenciales

**Opción A: Variables de entorno**
```bash
export SENTINELHUB_CLIENT_ID="tu_client_id"
export SENTINELHUB_CLIENT_SECRET="tu_client_secret"
```

**Opción B: Archivo de secretos**

Crea `sentinelhub-secrets_.txt` en la raíz del proyecto:
```
tu_client_id
tu_client_secret
```

## Métodos de Descarga

### Método 1: Descarga Rápida (Recomendado)

Descarga las 3 zonas de México con fechas recientes:

```bash
python scripts/redownload_with_recent_dates.py
```

**Ventajas:**
- Usa los últimos 30 días (asegura datos disponibles)
- Descarga las 3 zonas automáticamente
- Manejo robusto de errores

**Resultado:**
```
img/sentinel2/mexico/
├── mexicali/
│   ├── B02_10m.npy
│   ├── B03_10m.npy
│   ├── B04_10m.npy
│   ├── B8A_20m.npy
│   ├── B11_20m.npy
│   ├── B12_20m.npy
│   ├── hls_image.npy
│   └── metadata.txt
├── bajio/
│   └── ... (mismos archivos)
└── sinaloa/
    └── ... (mismos archivos)
```

### Método 2: Descarga por Zona

Descarga una zona específica:

```bash
# Mexicali
python scripts/download_hls_image.py --zone mexicali

# Bajío
python scripts/download_hls_image.py --zone bajio

# Sinaloa
python scripts/download_hls_image.py --zone sinaloa
```

### Método 3: Descarga con Fecha Específica

```bash
python scripts/download_hls_image.py \
  --zone mexicali \
  --date-from 2024-10-01 \
  --date-to 2024-10-31
```

**Nota:** Si no hay datos para esa fecha exacta, el script fallará. Usa un rango amplio.

### Método 4: Descarga Todas las Zonas

```bash
python scripts/download_hls_image.py --all
```

## Verificación

### Verificar Conexión

```bash
python scripts/diagnose_sentinel_data.py
```

**Salida esperada:**
```
Status: success
Message: Successful connection to Sentinel Hub
Data shape: (100, 100)
```

### Verificar Datos Descargados

```python
import numpy as np
from pathlib import Path

# Verificar una zona
zone = 'mexicali'
b02 = np.load(f'img/sentinel2/mexico/{zone}/B02_10m.npy')

print(f"Shape: {b02.shape}")
print(f"Range: [{b02.min():.4f}, {b02.max():.4f}]")
print(f"Mean: {b02.mean():.4f}")

# Si max == 0, no hay datos válidos
if b02.max() == 0:
    print("WARNING: No valid data!")
else:
    print("OK: Valid data downloaded")
```

## Solución de Problemas

### Problema 1: Imágenes con Solo Ceros

**Síntoma:**
```python
b02.max() == 0  # True
```

**Causa:** No hay datos disponibles para la fecha/área especificada.

**Solución:**
1. Usa un rango de fechas más amplio (30 días)
2. Prueba fechas más recientes
3. Verifica que el área tenga cobertura Sentinel-2

```bash
# Solución recomendada
python scripts/redownload_with_recent_dates.py
```

### Problema 2: Error de Autenticación

**Síntoma:**
```
Failed to download data: Authentication error
```

**Solución:**
1. Verifica que las credenciales sean correctas
2. Asegúrate de que la cuenta esté activa
3. Revisa que las variables de entorno estén configuradas

```bash
# Verificar variables
echo $SENTINELHUB_CLIENT_ID
echo $SENTINELHUB_CLIENT_SECRET

# O verificar archivo
cat sentinelhub-secrets_.txt
```

### Problema 3: Área Muy Grande

**Síntoma:**
```
Error: Request too large
```

**Solución:**
Reduce el tamaño del bounding box o la resolución.

### Problema 4: Sin Datos para Fecha Específica

**Síntoma:**
```
ValueError: No valid satellite imagery available
```

**Solución:**
Usa un rango de fechas más amplio:

```bash
# En lugar de un día específico
--date-from 2024-01-15 --date-to 2024-01-15

# Usa un rango amplio
--date-from 2024-01-01 --date-to 2024-01-31
```

## Bandas HLS

El formato HLS (Harmonized Landsat Sentinel-2) requiere 6 bandas específicas:

| Banda | Nombre | Resolución | Longitud de Onda | Uso |
|-------|--------|------------|------------------|-----|
| B02 | Blue | 10m | 490 nm | RGB, índices |
| B03 | Green | 10m | 560 nm | RGB, índices |
| B04 | Red | 10m | 665 nm | RGB, NDVI |
| B8A | NIR Narrow | 20m | 865 nm | NDVI, Prithvi |
| B11 | SWIR1 | 20m | 1610 nm | Humedad, Prithvi |
| B12 | SWIR2 | 20m | 2190 nm | Humedad, Prithvi |

**Importante:** Se usa B8A (NIR Narrow, 20m), NO B08 (NIR Broad, 10m).

## Tamaños de Archivo

- **Banda individual (10m)**: ~10-20 MB
- **Banda individual (20m)**: ~3-5 MB
- **HLS completo (6 bandas)**: ~50-100 MB
- **Embeddings**: ~200-300 MB
- **Total por zona**: ~300-400 MB
- **Total 3 zonas**: ~1-2 GB

## Mejores Prácticas

1. **Usa fechas recientes**: Mayor probabilidad de datos disponibles
2. **Rango amplio**: 30 días asegura encontrar imágenes
3. **Verifica después de descargar**: Comprueba que no sean ceros
4. **Una zona a la vez**: Si tienes problemas de memoria
5. **Guarda metadata**: Útil para reproducibilidad

## Scripts Disponibles

| Script | Propósito | Uso |
|--------|-----------|-----|
| `redownload_with_recent_dates.py` | Descarga rápida con fechas recientes | Recomendado |
| `download_hls_image.py` | Descarga con control fino | Avanzado |
| `diagnose_sentinel_data.py` | Verificar conexión | Diagnóstico |

## Referencias

- [Sentinel Hub API](https://docs.sentinel-hub.com/)
- [Sentinel-2 User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)
- [HLS Product Guide](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf)
- [Prithvi Model](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M)
