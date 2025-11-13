# Descarga Manual de Dynamic World

Como alternativa a Google Earth Engine, puedes descargar las máscaras de Dynamic World manualmente desde la web.

## Opción 1: Dynamic World Explorer (Recomendado - 15 minutos)

### Paso 1: Ir al sitio web
https://www.dynamicworld.app/explore/

### Paso 2: Descargar cada zona

#### Mexicali
1. En el mapa, navega a: **32.5°N, 115.3°W**
2. Zoom hasta ver el área de interés
3. Selecciona fecha: **~2024-10-15** (±5 días)
4. Click en "Download" → "Label" (clase más probable)
5. Formato: **GeoTIFF**
6. Guardar como: `data/dynamic_world/mexicali_dw_label.tif`

#### Bajío
1. Navega a: **21.0°N, 101.4°W**
2. Fecha: **~2024-10-15**
3. Download → Label → GeoTIFF
4. Guardar como: `data/dynamic_world/bajio_dw_label.tif`

#### Sinaloa
1. Navega a: **25.8°N, 108.2°W**
2. Fecha: **~2024-10-15**
3. Download → Label → GeoTIFF
4. Guardar como: `data/dynamic_world/sinaloa_dw_label.tif`

## Opción 2: Usar datos sintéticos (Para testing rápido)

Si no puedes descargar Dynamic World ahora, el notebook puede generar datos sintéticos realistas para demostración.

## Verificación

Después de descargar, verifica los archivos:

```bash
ls -lh data/dynamic_world/
```

Deberías ver:
- mexicali_dw_label.tif (~2-5 MB)
- bajio_dw_label.tif (~2-5 MB)
- sinaloa_dw_label.tif (~2-5 MB)

## Clases de Dynamic World

Las máscaras descargadas contienen valores 0-8:

- 0: Water (Agua)
- 1: Trees (Árboles)
- 2: Grass (Pasto)
- 3: Flooded Vegetation (Vegetación inundada)
- 4: Crops (Cultivos) ← **IMPORTANTE**
- 5: Shrub & Scrub (Arbustos)
- 6: Built Area (Área construida) ← **IMPORTANTE**
- 7: Bare Ground (Suelo desnudo) ← **IMPORTANTE**
- 8: Snow & Ice (Nieve/Hielo)

## Siguiente paso

Una vez descargados los archivos, ejecuta el notebook:
`notebooks/validation/ground_truth_validation.ipynb`
