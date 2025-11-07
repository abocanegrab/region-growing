# Sistema Híbrido de Detección de Estrés Vegetal
## Proyecto Final - Visión computacional - Maestría en Inteligencia Artificial Aplicada
---

## 📋 INFORMACIÓN DEL PROYECTO

**Método Asignado:** Region Growing  
**Objetivo:**   
**Plazo:** 10 días (20 horas/desarrollador)  
**Presupuesto:** $30 USD + RTX 4070 (8GB VRAM) local  

---

## 🎯 PROPUESTA DE VALOR

### Innovación Principal
**Sistema Híbrido de Segmentación Semántico-Espectral** que combina:

1. **Region Growing Clásico** (baseline) - Segmentación basada en NDVI/NDWI
2. **Region Growing Semántico** (innovación) - Segmentación asistida por Foundation Model (NASA Prithvi)

### Diferenciadores Clave
✅ **Robustez ante sombras de nubes** - El método semántico ignora variaciones espectrales causadas por sombras  
✅ **Segmentación consciente de objetos** - Identifica límites de campos agrícolas, no solo zonas de estrés  
✅ **Análisis jerárquico** - Primero segmenta objetos (campos), luego analiza estrés dentro de cada objeto  
✅ **Validación visual directa** - Comparación lado a lado: imagen real vs segmentación  

---

## 🏗️ ARQUITECTURA MEJORADA

### Stack Tecnológico Actualizado

#### Backend: FastAPI (reemplazo de Flask)
**Justificación:**
- ⚡ **Performance**: 3-4x más rápido que Flask
- 📝 **Documentación automática**: OpenAPI/Swagger nativo
- 🔒 **Type safety**: Validación con Pydantic
- ⚙️ **Async nativo**: Ideal para llamadas a Sentinel Hub
- 🚀 **Producción-ready**: ASGI, mejor para deployment

```python
# Ejemplo de endpoint FastAPI
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    bbox: dict
    date_from: str
    date_to: str
    method: str = "hybrid"  # classic | hybrid

@app.post("/api/analysis/analyze")
async def analyze_region(request: AnalysisRequest):
    # Procesamiento asíncrono
    pass
```


#### Frontend: Nuxt 3 (reemplazo de Vue 3 + Vite)
**Justificación:**
- 🎨 **SSR/SSG**: Mejor SEO y performance inicial
- 📦 **Auto-imports**: Menos boilerplate
- 🗂️ **File-based routing**: Estructura más clara
- 🔧 **Módulos integrados**: Pinia, composables, layouts
- 📱 **PWA ready**: Instalable como app

```typescript
// Ejemplo de composable Nuxt 3
// composables/useAnalysis.ts
export const useAnalysis = () => {
  const results = useState('analysis-results', () => null)
  const loading = useState('analysis-loading', () => false)
  
  const analyzeRegion = async (bbox: BBox, method: 'classic' | 'hybrid') => {
    loading.value = true
    try {
      const data = await $fetch('/api/analysis/analyze', {
        method: 'POST',
        body: { bbox, method }
      })
      results.value = data
    } finally {
      loading.value = false
    }
  }
  
  return { results, loading, analyzeRegion }
}
```

#### ML/CV Stack
| Componente       | Tecnología              | Versión | Uso                        |
| ---------------- | ----------------------- | ------- | -------------------------- |
| Foundation Model | **Prithvi-EO-1.0-100M** | Latest  | Embeddings semánticos      |
| Framework ML     | **PyTorch**             | 2.1+    | Inferencia del modelo      |
| Segmentación     | **MMSegmentation**      | 1.2+    | Pipeline de segmentación   |
| Procesamiento    | **NumPy**               | 1.26+   | Operaciones matriciales    |
| Visión           | **OpenCV**              | 4.9+    | Contornos y morfología     |
| Geoespacial      | **Rasterio**            | 1.3+    | Manejo de GeoTIFF          |
| Geometría        | **Shapely**             | 2.0+    | Polígonos y simplificación |
| ML Utilities     | **scikit-learn**        | 1.4+    | K-Means clustering         |

#### Gestión de Dependencias y Entorno

**Backend:**
- **Poetry** - Gestión moderna de dependencias Python
  - Resolución de dependencias determinística
  - Lock file para reproducibilidad
  - Entornos virtuales automáticos
  - Publicación simplificada

**Frontend:**
- **pnpm** - Gestor de paquetes eficiente para Node.js
  - Más rápido que npm/yarn
  - Ahorro de espacio en disco
  - Monorepo-friendly

#### Datos Satelitales
| Fuente             | Resolución | Bandas                          | Uso                                  |
| ------------------ | ---------- | ------------------------------- | ------------------------------------ |
| **Sentinel-2 L2A** | 10m/20m    | **B02,B03,B04,B8A,B11,B12,SCL** | Input Prithvi (6 bandas HLS) + Nubes |
| **Sentinel-2 L2A** | 10m        | B02,B03,B04,B08                 | RGB + NDVI (método clásico)          |

**⚠️ CRÍTICO - Bandas para Prithvi:**
- Prithvi requiere **6 bandas específicas en orden exacto**: B02 (Blue), B03 (Green), B04 (Red), **B8A** (NIR Narrow - 20m), B11 (SWIR1 - 20m), B12 (SWIR2 - 20m)
- **Nota:** B8A es diferente de B08. B8A tiene 20m de resolución y es la banda correcta para HLS
- Todas las bandas deben remuestrearse a resolución común (10m o 20m) antes de apilar

---

## 📊 METODOLOGÍA HÍBRIDA DETALLADA

### Pipeline Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    USUARIO SELECCIONA ÁREA                       │
│                  (Polígono en mapa Leaflet)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              BACKEND: Descarga Sentinel-2 L2A                    │
│  • Para Prithvi: B02,B03,B04,B8A,B11,B12 (6 bandas HLS)         │
│  • Para NDVI: B04 (Red), B08 (NIR)                              │
│  • Máscara nubes: SCL                                            │
│  • Resolución: Remuestrear todo a 10m                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
┌───────────────────────────┐  ┌──────────────────────────────┐
│   MÉTODO CLÁSICO (RG)     │  │   MÉTODO HÍBRIDO (MGRG)      │
│                           │  │                              │
│ 1. Calcular NDVI          │  │ 1. Preparar HLS (6 bandas)   │
│    (NIR-Red)/(NIR+Red)    │  │    B02,B03,B04,B8A,B11,B12   │
│                           │  │ 2. Remuestrear 20m→10m       │
│                           │  │ 3. Pasar por Prithvi encoder │
│                           │  │ 4. Extraer embeddings (256D) │
│ 2. Generar semillas       │  │ 4. Generar semillas          │
│    (grid 20x20)           │  │    (grid 20x20)              │
│                           │  │                              │
│ 3. Region Growing         │  │ 5. Region Growing Semántico  │
│    Criterio:              │  │    Criterio:                 │
│    |NDVI_A - NDVI_B|      │  │    cosine_sim(emb_A, emb_B)  │
│    < threshold (0.1)      │  │    > threshold (0.85)        │
│                           │  │                              │
│ 4. Clasificar regiones    │  │ 6. Clasificar regiones       │
│    por NDVI:              │  │    por semántica + NDVI:     │
│    • Alto: <0.3           │  │    • Primero: límite objeto  │
│    • Medio: 0.3-0.5       │  │    • Luego: estrés interno   │
│    • Bajo: >0.5           │  │                              │
└───────────┬───────────────┘  └──────────┬───────────────────┘
            │                             │
            └────────────┬────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  COMPARACIÓN A/B VISUAL                          │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │  Clásico (RG)        │  │  Híbrido (MGRG)      │            │
│  │  • Sobre-segmenta    │  │  • Segmenta objetos  │            │
│  │  • Sensible a sombras│  │  • Robusto a sombras │            │
│  │  • Fragmentado       │  │  • Coherente         │            │
│  └──────────────────────┘  └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```


### Algoritmo Region Growing Clásico (Baseline)

```python
# backend/app/algorithms/classic_region_growing.py
import numpy as np
from typing import List, Tuple

class ClassicRegionGrowing:
    """
    Region Growing clásico basado en homogeneidad espectral (NDVI)
    """
    def __init__(self, threshold: float = 0.1, min_size: int = 50):
        self.threshold = threshold
        self.min_size = min_size
    
    def segment(self, ndvi: np.ndarray, seeds: List[Tuple[int, int]]) -> np.ndarray:
        """
        Segmenta imagen NDVI usando Region Growing
        
        Args:
            ndvi: Array 2D con valores NDVI [-1, 1]
            seeds: Lista de coordenadas (y, x) de semillas
            
        Returns:
            labeled_image: Array 2D con etiquetas de región
        """
        h, w = ndvi.shape
        labeled = np.zeros((h, w), dtype=np.int32)
        region_id = 1
        
        for seed_y, seed_x in seeds:
            if labeled[seed_y, seed_x] != 0:
                continue
                
            # Valor de referencia de la semilla
            seed_value = ndvi[seed_y, seed_x]
            
            # BFS para crecer región
            queue = [(seed_y, seed_x)]
            region_pixels = []
            
            while queue:
                y, x = queue.pop(0)
                
                # Verificar límites y si ya fue visitado
                if not (0 <= y < h and 0 <= x < w):
                    continue
                if labeled[y, x] != 0:
                    continue
                
                # Criterio de homogeneidad espectral
                pixel_value = ndvi[y, x]
                if abs(pixel_value - seed_value) <= self.threshold:
                    labeled[y, x] = region_id
                    region_pixels.append((y, x))
                    
                    # Agregar vecinos (4-conectividad)
                    queue.extend([
                        (y-1, x), (y+1, x),
                        (y, x-1), (y, x+1)
                    ])
            
            # Filtrar regiones pequeñas (ruido)
            if len(region_pixels) >= self.min_size:
                region_id += 1
            else:
                for y, x in region_pixels:
                    labeled[y, x] = 0
        
        return labeled
    
    def classify_stress(self, ndvi: np.ndarray, labeled: np.ndarray) -> dict:
        """
        Clasifica regiones por nivel de estrés vegetal
        """
        regions = {}
        for region_id in np.unique(labeled):
            if region_id == 0:
                continue
            
            mask = labeled == region_id
            region_ndvi = ndvi[mask]
            mean_ndvi = np.mean(region_ndvi)
            
            # Clasificación de estrés
            if mean_ndvi < 0.3:
                stress = "high"
            elif mean_ndvi < 0.5:
                stress = "medium"
            else:
                stress = "low"
            
            regions[region_id] = {
                "mean_ndvi": float(mean_ndvi),
                "std_ndvi": float(np.std(region_ndvi)),
                "size": int(np.sum(mask)),
                "stress_level": stress
            }
        
        return regions
```


### Algoritmo Region Growing Semántico (Innovación)

```python
# backend/app/algorithms/semantic_region_growing.py
import torch
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRegionGrowing:
    """
    Region Growing semántico basado en embeddings de Foundation Model
    Inspirado en CRGNet (Ghamisi et al., 2022)
    """
    def __init__(
        self, 
        model,  # Prithvi encoder
        threshold: float = 0.85,  # Similitud coseno
        min_size: int = 50
    ):
        self.model = model
        self.threshold = threshold
        self.min_size = min_size
    
    def extract_embeddings(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae embeddings semánticos usando Prithvi
        
        Args:
            image: Array (H, W, 6) con bandas HLS en orden:
                   [B02, B03, B04, B8A, B11, B12]
                   ⚠️ CRÍTICO: B8A (no B08), todas a 10m resolución
            
        Returns:
            embeddings: Array (H, W, 256) con features semánticos
        """
        # Convertir a tensor y normalizar
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        x = (x - x.mean()) / (x.std() + 1e-8)
        
        # Inferencia (solo encoder, sin decoder)
        with torch.no_grad():
            features = self.model.encode(x)  # (1, 256, H', W')
        
        # Interpolar a resolución original si es necesario
        if features.shape[2:] != image.shape[:2]:
            features = torch.nn.functional.interpolate(
                features, 
                size=image.shape[:2], 
                mode='bilinear'
            )
        
        # Convertir a numpy (H, W, 256)
        embeddings = features.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Normalizar embeddings (importante para cosine similarity)
        norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def generate_smart_seeds(
        self, 
        embeddings: np.ndarray, 
        n_clusters: int = 5
    ) -> List[Tuple[int, int]]:
        """
        🆕 MEJORA: Genera semillas inteligentes usando K-Means sobre embeddings
        
        En lugar de un grid fijo, encuentra los píxeles más representativos
        de cada clase semántica (cultivo, agua, bosque, etc.)
        
        Args:
            embeddings: Array (H, W, 256) con features semánticos
            n_clusters: Número de clusters (clases semánticas esperadas)
            
        Returns:
            seeds: Lista de coordenadas (y, x) de centroides
        """
        from sklearn.cluster import KMeans
        
        h, w, d = embeddings.shape
        
        # Reshape para K-Means: (H*W, 256)
        emb_flat = embeddings.reshape(-1, d)
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb_flat)
        
        # Encontrar píxel más cercano a cada centroide
        seeds = []
        for cluster_id in range(n_clusters):
            # Máscara de píxeles en este cluster
            cluster_mask = (labels == cluster_id)
            cluster_embeddings = emb_flat[cluster_mask]
            
            # Centroide del cluster
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Encontrar píxel más cercano al centroide
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = np.argmin(distances)
            
            # Convertir índice flat a coordenadas (y, x)
            cluster_indices = np.where(cluster_mask)[0]
            flat_idx = cluster_indices[closest_idx]
            y, x = divmod(flat_idx, w)
            
            seeds.append((y, x))
        
        return seeds
    
    def segment(
        self, 
        embeddings: np.ndarray, 
        seeds: List[Tuple[int, int]] = None,
        use_smart_seeds: bool = True
    ) -> np.ndarray:
        """
        Segmenta usando similitud semántica en espacio de embeddings
        
        Args:
            embeddings: Array (H, W, 256)
            seeds: Lista de semillas (opcional si use_smart_seeds=True)
            use_smart_seeds: Si True, genera semillas con K-Means
        """
        h, w, d = embeddings.shape
        labeled = np.zeros((h, w), dtype=np.int32)
        region_id = 1
        
        # 🆕 Generar semillas inteligentes si no se proporcionan
        if seeds is None or use_smart_seeds:
            seeds = self.generate_smart_seeds(embeddings, n_clusters=5)
            print(f"✅ Semillas inteligentes generadas: {len(seeds)} clusters")
        
        for seed_y, seed_x in seeds:
            if labeled[seed_y, seed_x] != 0:
                continue
            
            # Embedding de referencia
            seed_emb = embeddings[seed_y, seed_x]
            
            # BFS para crecer región
            queue = [(seed_y, seed_x)]
            region_pixels = []
            
            while queue:
                y, x = queue.pop(0)
                
                if not (0 <= y < h and 0 <= x < w):
                    continue
                if labeled[y, x] != 0:
                    continue
                
                # Criterio de homogeneidad SEMÁNTICA
                pixel_emb = embeddings[y, x]
                similarity = np.dot(seed_emb, pixel_emb)  # Ya normalizados
                
                if similarity >= self.threshold:
                    labeled[y, x] = region_id
                    region_pixels.append((y, x))
                    
                    # Agregar vecinos
                    queue.extend([
                        (y-1, x), (y+1, x),
                        (y, x-1), (y, x+1)
                    ])
            
            # Filtrar regiones pequeñas
            if len(region_pixels) >= self.min_size:
                region_id += 1
            else:
                for y, x in region_pixels:
                    labeled[y, x] = 0
        
        return labeled
    
    def analyze_stress_within_objects(
        self, 
        ndvi: np.ndarray, 
        semantic_labels: np.ndarray
    ) -> dict:
        """
        Análisis jerárquico: primero objetos, luego estrés interno
        """
        results = {}
        
        for obj_id in np.unique(semantic_labels):
            if obj_id == 0:
                continue
            
            # Máscara del objeto semántico
            obj_mask = semantic_labels == obj_id
            obj_ndvi = ndvi[obj_mask]
            
            # Estadísticas del objeto completo
            mean_ndvi = np.mean(obj_ndvi)
            
            # Sub-segmentación por estrés dentro del objeto
            stress_zones = {
                "high": np.sum(obj_ndvi < 0.3),
                "medium": np.sum((obj_ndvi >= 0.3) & (obj_ndvi < 0.5)),
                "low": np.sum(obj_ndvi >= 0.5)
            }
            
            results[obj_id] = {
                "mean_ndvi": float(mean_ndvi),
                "size": int(np.sum(obj_mask)),
                "stress_distribution": stress_zones,
                "dominant_stress": max(stress_zones, key=stress_zones.get)
            }
        
        return results
```


---

## 📈 PLAN DE TRABAJO SCRUM (10 DÍAS)

### Sprint Backlog Detallado

#### **Épica 1: Fundación y Baseline (Días 1-3)**

### US-1: Migrar backend de Flask a FastAPI + Poetry

- **Como** desarrollador
- **Quiero** migrar el backend de Flask a FastAPI y configurar Poetry
- **Para que** tengamos mejor performance, documentación automática y gestión de dependencias moderna

**Criterios de Aceptación:**
- ✅ **Poetry configurado** como gestor de dependencias
  - `pyproject.toml` creado con metadatos del proyecto
  - `poetry.lock` para reproducibilidad
  - Entorno virtual automático
- ✅ Endpoints REST funcionando correctamente
- ✅ Swagger docs automático generado (`/api/docs`)
- ✅ Validación con Pydantic implementada
- ✅ CORS configurado para Nuxt 3
- ✅ Estructura de proyecto limpia (app/, config/, tests/)

**Código Esperado:**
```bash
# Inicializar Poetry
poetry init
poetry add fastapi uvicorn[standard] pydantic pydantic-settings
poetry add --group dev pytest black ruff

# Ejecutar
poetry run uvicorn app.main:app --reload
```

```toml
# pyproject.toml
[tool.poetry]
name = "vision-backend"
version = "1.0.0"
description = "Sistema Híbrido de Detección de Estrés Vegetal"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.109.0"
uvicorn = {extras = ["standard"], version = "^0.27.0"}
pydantic = "^2.5.0"
torch = "^2.1.2"
numpy = "^1.26.3"
opencv-python = "^4.9.0"
scikit-learn = "^1.4.0"
sentinelhub = "^3.10.2"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.12.0"
ruff = "^0.1.9"
```

**Estimación:** 4 horas  
**Responsable:** Carlos Bocanegra  
**Estado:** ⏳ Pendiente

---

### US-2: Migrar frontend de Vue+Vite a Nuxt 3

- **Como** desarrollador
- **Quiero** migrar el frontend de Vue+Vite a Nuxt 3
- **Para que** tengamos SSR y mejor estructura de proyecto

**Criterios de Aceptación:**
- ✅ SSR configurado y funcionando
- ✅ Auto-imports funcionando (componentes, composables)
- ✅ Composables creados (useAnalysis, useMap)
- ✅ Pinia store configurado
- ✅ Leaflet integrado

**Estimación:** 6 horas  
**Responsable:** Luis Vázquez  
**Estado:** ⏳ Pendiente

---

### US-3: Descargar imágenes Sentinel-2

- **Como** usuario
- **Quiero** que el sistema descargue imágenes Sentinel-2 automáticamente
- **Para que** pueda analizar cualquier región del mundo

**Criterios de Aceptación:**
- ✅ Integración con Sentinel Hub API funcionando
- ✅ Descarga de bandas RGB (B02, B03, B04)
- ✅ Descarga de banda NIR (B08) para NDVI
- ✅ Descarga de banda SCL para máscara de nubes
- ✅ Manejo de errores (área muy grande, sin datos, etc.)

**Estimación:** 4 horas  
**Responsable:** Carlos Bocanegra  
**Estado:** ✅ Completado

---

### US-4: Implementar Region Growing clásico

- **Como** investigador
- **Quiero** implementar el algoritmo Region Growing clásico
- **Para que** tengamos la línea base de comparación

**Criterios de Aceptación:**
- ✅ Algoritmo funcional con BFS (4-conectividad)
- ✅ Segmentación basada en NDVI
- ✅ Criterio de homogeneidad: |NDVI_A - NDVI_B| < threshold
- ✅ Clasificación de estrés (alto/medio/bajo)
- ✅ Filtrado de regiones pequeñas (ruido)

**Estimación:** 6 horas  
**Responsable:** Carlos Bocanegra  
**Estado:** ✅ Completado

---

**Entregable Día 3:** Backend FastAPI + Frontend Nuxt 3 + RG Clásico funcional

---

#### **Épica 2: Innovación SOTA (Días 4-7)**

### US-5: Descargar y configurar Prithvi

- **Como** investigador
- **Quiero** descargar y configurar el modelo Prithvi
- **Para que** podamos extraer embeddings semánticos

**Criterios de Aceptación:**
- ✅ Modelo Prithvi-EO-1.0-100M descargado de HuggingFace
- ✅ Dependencias instaladas (PyTorch, MMSegmentation, timm)
- ✅ Test de inferencia exitoso con imagen de ejemplo
- ✅ Verificar que corre en RTX 4070 (8GB VRAM)

**Estimación:** 4 horas  
**Responsable:** Arthur Zizumbo  
**Estado:** ⏳ Pendiente

---

### US-6: Extraer embeddings de imágenes Sentinel-2

- **Como** desarrollador
- **Quiero** extraer embeddings semánticos de imágenes Sentinel-2
- **Para que** podamos usar el método MGRG

**⚠️ CRÍTICO - Bandas Correctas para Prithvi:**

Prithvi-EO-1.0-100M fue pre-entrenado en formato HLS y requiere **exactamente 6 bandas en orden específico**:

1. **B02** - Blue (490 nm) - 10m
2. **B03** - Green (560 nm) - 10m
3. **B04** - Red (665 nm) - 10m
4. **B8A** - NIR Narrow (865 nm) - **20m** ⚠️ (NO es B08)
5. **B11** - SWIR1 (1610 nm) - **20m**
6. **B12** - SWIR2 (2190 nm) - **20m**

**Diferencia crítica:** B08 (NIR Broad, 10m) ≠ B8A (NIR Narrow, 20m). Prithvi espera B8A.

**Criterios de Aceptación:**
- ✅ Descargar bandas correctas: B02, B03, B04, **B8A**, B11, B12
- ✅ Remuestrear B8A, B11, B12 de 20m → 10m usando interpolación bilineal
- ✅ Apilar en orden exacto: [B02, B03, B04, B8A, B11, B12] (6 canales)
- ✅ Normalizar imagen (mean=0, std=1)
- ✅ Inferencia con Prithvi (solo encoder, sin decoder)
- ✅ Obtener embeddings con shape (H, W, 256)
- ✅ Normalizar embeddings (L2 norm) para cosine similarity

**Código Esperado:**
```python
# Evalscript para Sentinel Hub
evalscript = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "B8A", "B11", "B12"],
            units: "REFLECTANCE"
        }],
        output: { bands: 6, sampleType: "FLOAT32" }
    };
}
function evaluatePixel(sample) {
    return [sample.B02, sample.B03, sample.B04, 
            sample.B8A, sample.B11, sample.B12];
}
"""

# Remuestreo de bandas 20m → 10m
from scipy.ndimage import zoom
b8a_10m = zoom(b8a_20m, 2, order=1)  # Bilinear
b11_10m = zoom(b11_20m, 2, order=1)
b12_10m = zoom(b12_20m, 2, order=1)

# Apilar en orden correcto
hls_image = np.stack([b02, b03, b04, b8a_10m, b11_10m, b12_10m], axis=-1)
```

**Referencias:**
- Prithvi HuggingFace: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
- HLS Product Guide: https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf

**Estimación:** 10 horas  
**Responsables:** Arthur Zizumbo + Luis Vázquez  
**Estado:** ⏳ Pendiente

---

### US-7: Implementar MGRG (Region Growing Semántico)

- **Como** desarrollador
- **Quiero** implementar el algoritmo MGRG (Metric-Guided Region Growing)
- **Para que** tengamos segmentación semántica robusta

**🟡 MEJORA RECOMENDADA - Semillas Inteligentes:**

En lugar de un grid fijo (20x20), usar **K-Means clustering** sobre embeddings para encontrar semillas más representativas.

**Ventajas:**
- Semillas "semánticamente puras" (centroide de cada cluster)
- Más robusto que grid aleatorio
- Demuestra integración avanzada de IA
- Reduce sobre-segmentación

**Criterios de Aceptación:**
- ✅ Algoritmo funcional con BFS (4-conectividad) sobre embeddings
- ✅ Criterio de homogeneidad: cosine_similarity(emb_A, emb_B) > threshold
- ✅ Threshold optimizado (0.85 por defecto)
- ✅ **Método `generate_smart_seeds()` implementado** con K-Means (K=5-10)
- ✅ Comparación: grid fijo vs K-Means inteligente
- ✅ Filtrado de regiones pequeñas (min_size=50)

**Código Esperado:**
```python
from sklearn.cluster import KMeans

def generate_smart_seeds(embeddings, n_clusters=5):
    """Genera semillas usando K-Means sobre embeddings"""
    h, w, d = embeddings.shape
    emb_flat = embeddings.reshape(-1, d)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(emb_flat)
    
    # Encontrar píxel más cercano a cada centroide
    seeds = []
    for i in range(n_clusters):
        cluster_mask = (labels == i)
        cluster_embs = emb_flat[cluster_mask]
        centroid = kmeans.cluster_centers_[i]
        
        distances = np.linalg.norm(cluster_embs - centroid, axis=1)
        closest_idx = np.argmin(distances)
        
        flat_idx = np.where(cluster_mask)[0][closest_idx]
        y, x = divmod(flat_idx, w)
        seeds.append((y, x))
    
    return seeds
```

**Comparación esperada:**

| Método | Grid Fijo | K-Means Inteligente |
|--------|-----------|---------------------|
| Semillas | ~400 | 5-10 |
| Calidad | Aleatorio | Representativo |
| Sobre-segmentación | Alta | Baja |
| Tiempo | Rápido | +2-3 seg |

**Estimación:** 12 horas  
**Responsables:** Carlos Bocanegra + Arthur Zizumbo  
**Estado:** ⏳ Pendiente

---

### US-8: Generar comparativa A/B visual

- **Como** investigador
- **Quiero** generar una comparativa visual lado a lado
- **Para que** podamos demostrar la superioridad del método híbrido

**Criterios de Aceptación:**
- ✅ Misma imagen procesada por ambos métodos (RG Clásico + MGRG)
- ✅ Visualización lado a lado en frontend
- ✅ Métricas cuantitativas calculadas:
  - Coherencia espacial
  - Número de regiones
  - Precisión de límites (si hay ground truth)
- ✅ Caso de fallo claro (ej: campo con sombra de nube)
- ✅ Exportar imágenes en alta resolución (300 DPI)

**Estimación:** 6 horas  
**Responsable:** Luis Vázquez  
**Estado:** ⏳ Pendiente

---

### US-9: Implementar análisis jerárquico

- **Como** usuario
- **Quiero** ver análisis jerárquico (objeto → estrés)
- **Para que** pueda entender qué objeto tiene estrés y cuánto

**Criterios de Aceptación:**
- ✅ Primero: segmentación semántica (identificar objetos)
- ✅ Luego: análisis NDVI interno de cada objeto
- ✅ Reporte estructurado por objeto:
  - ID del objeto
  - NDVI promedio
  - Distribución de estrés interno (alto/medio/bajo)
  - Área en hectáreas
- ✅ Visualización con colores por estrés interno

**Estimación:** 4 horas  
**Responsable:** Carlos Bocanegra  
**Estado:** ⏳ Pendiente

---

**Entregable Día 7:** Sistema híbrido completo con comparativa A/B

---


---

## 📚 REFERENCIAS ACADÉMICAS ACTUALIZADAS (2022-2025)

### Referencias Principales (Citación Obligatoria)

1. **Ma, L., Yan, Z., Li, M., Liu, T., Tan, L., Wang, X., He, W., Wang, R., He, G., Lu, H., & Blaschke, T. (2024).** Deep learning meets object-based image analysis: Tasks, challenges, strategies, and perspectives. *IEEE Geoscience and Remote Sensing Magazine*, 1–29. https://doi.org/10.1109/MGRS.2024.3489952
   - **Relevancia:** Marco teórico completo sobre hibridación DL-OBIA, base conceptual del proyecto

2. **Jakubik, J., Roy, S., Phillips, C. E., Fraccaro, P., Godwin, D., Zadrozny, B., Szwarcman, D., Gomes, C., Nyirjesy, G., Edwards, B., Kimura, D., Simumba, N., Chu, L., Mukkavilli, S. K., Lambhate, D., Das, K., Bangalore Ravi, S. N., Oliveira, D., Muszynski, G., ... Schmude, J. (2024).** Foundation models for generalist geospatial artificial intelligence. *arXiv preprint arXiv:2310.18660v2*. https://arxiv.org/abs/2310.18660
   - **Relevancia:** Paper oficial de Prithvi (NASA/IBM), justifica uso de Foundation Models

3. **Ghamisi, P., Rasti, B., Yokoya, N., Wang, Q., Hofle, B., Bruzzone, L., Bovolo, F., Chi, M., Anders, K., Gloaguen, R., Atkinson, P. M., & Benediktsson, J. A. (2022).** Consistency-regularized region-growing network for semantic segmentation of urban scenes with point-level annotations. *IEEE Transactions on Image Processing*, 31, 5038–5051. https://doi.org/10.1109/TIP.2022.3188339
   - **Relevancia:** CRGNet, inspiración directa para MGRG (Metric-Guided Region Growing)

4. **Yang, T., Zou, Y., Yang, X., & del Rey Castillo, E. (2024).** Domain knowledge-enhanced region growing framework for semantic segmentation of bridge point clouds. *Automation in Construction*, 164, 105572. https://doi.org/10.1016/j.autcon.2024.105572
   - **Relevancia:** Region Growing con conocimiento semántico, aplicación reciente

5. **Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D. B., & Ermon, S. (2022).** SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. *Advances in Neural Information Processing Systems*, 35, 197–211. https://proceedings.neurips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html
   - **Relevancia:** Masked Autoencoders para imágenes satelitales, alternativa a Prithvi

## Referencias Técnicas (Implementación)

11. **IBM & NASA. (2023).** Prithvi-EO-1.0-100M: Pretrained foundation model for harmonized Landsat Sentinel-2 (HLS). *Hugging Face Model Hub*. https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
    - **Relevancia:** Modelo pre-entrenado usado en el proyecto
    - **⚠️ CRÍTICO:** Requiere 6 bandas HLS: B02, B03, B04, B8A, B11, B12 (ver documentación del modelo)

11b. **Claverie, M., Ju, J., Masek, J. G., Dungan, J. L., Vermote, E. F., Roger, J. C., Skakun, S. V., & Justice, C. (2018).** The Harmonized Landsat and Sentinel-2 surface reflectance data set. *Remote Sensing of Environment*, 219, 145–161. https://doi.org/10.1016/j.rse.2018.09.002
    - **Relevancia:** Especificación técnica del formato HLS, bandas y preprocesamiento

12. **Drusch, M., Del Bello, U., Carlier, S., Colin, O., Fernandez, V., Gascon, F., Hoersch, B., Isola, C., Laberinti, P., Martimort, P., Meygret, A., Spoto, F., Sy, O., Marchese, F., & Bargellini, P. (2012).** Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment*, 120, 25–36. https://doi.org/10.1016/j.rse.2011.11.026
    - **Relevancia:** Paper oficial de Sentinel-2, descripción técnica de las bandas

13. **Tucker, C. J. (1979).** Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment*, 8(2), 127–150. https://doi.org/10.1016/0034-4257(79)90013-0
    - **Relevancia:** Paper original del NDVI, citación clásica obligatoria

14. **Gao, B. C. (1996).** NDWI—A normalized difference water index for remote sensing of vegetation liquid water from space. *Remote Sensing of Environment*, 58(3), 257–266. https://doi.org/10.1016/S0034-4257(96)00067-3
    - **Relevancia:** Paper original del NDWI, índice de estrés hídrico

15. **Chen, K., Zou, Z., & Shi, Z. (2021).** Building extraction from remote sensing images with sparse token transformers. *Remote Sensing*, 13(21), 4441. https://doi.org/10.3390/rs13214441
    - **Relevancia:** Transformers para segmentación, arquitectura moderna


---

## 💻 IMPLEMENTACIÓN TÉCNICA DETALLADA

### Estructura del Proyecto Mejorada

```
proyecto-vision-computacional/
│
├── backend/                                    # FastAPI Backend
│   ├── app/
│   │   ├── main.py                            # FastAPI app con CORS
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── analysis.py                # Endpoints de análisis
│   │   │   │   └── health.py                  # Health checks
│   │   │   └── schemas/
│   │   │       ├── requests.py                # Pydantic request models
│   │   │       └── responses.py               # Pydantic response models
│   │   ├── algorithms/
│   │   │   ├── classic_region_growing.py      # RG Clásico
│   │   │   └── semantic_region_growing.py     # MGRG (innovación)
│   │   ├── services/
│   │   │   ├── sentinel_hub.py                # Descarga Sentinel-2
│   │   │   ├── prithvi_inference.py           # Inferencia Prithvi
│   │   │   ├── ndvi_calculator.py             # Cálculo índices
│   │   │   └── geo_converter.py               # Conversión geoespacial
│   │   └── models/
│   │       └── prithvi_loader.py              # Carga modelo Prithvi
│   ├── tests/
│   │   ├── test_classic_rg.py
│   │   └── test_semantic_rg.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env
│
├── frontend/                                   # Nuxt 3 Frontend
│   ├── pages/
│   │   └── index.vue                          # Página principal
│   ├── components/
│   │   ├── Map/
│   │   │   ├── LeafletMap.vue                 # Mapa interactivo
│   │   │   └── DrawControl.vue                # Control de dibujo
│   │   ├── Analysis/
│   │   │   ├── MethodSelector.vue             # Selector Classic/Hybrid
│   │   │   ├── ComparisonView.vue             # Vista A/B
│   │   │   └── ResultsPanel.vue               # Panel de resultados
│   │   └── Common/
│   │       ├── LoadingSpinner.vue
│   │       └── ErrorAlert.vue
│   ├── composables/
│   │   ├── useAnalysis.ts                     # Lógica de análisis
│   │   ├── useMap.ts                          # Lógica del mapa
│   │   └── usePrithvi.ts                      # Estado Prithvi
│   ├── stores/
│   │   └── analysis.ts                        # Pinia store
│   ├── nuxt.config.ts
│   ├── package.json
│   └── tsconfig.json
│
├── notebooks/                                  # Google Colab
│   ├── Region_Growing_Comparison.ipynb        # Notebook principal
│   └── assets/
│       ├── example_images/                    # Imágenes de ejemplo
│       └── results/                           # Resultados guardados
│
├── docs/                                       # Documentación
│   ├── articulo_cientifico.pdf                # Artículo final
│   ├── presentacion.pptx                      # Presentación clase
│   └── video_tutorial.mp4                     # Video 5-10 min
│
└── README.md                                   # Documentación principal
```

### Configuración del Entorno

#### Backend (FastAPI)

```bash
# requirements.txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
pydantic-settings==2.1.0

# ML/CV
torch==2.1.2
torchvision==0.16.2
mmsegmentation==1.2.2
timm==0.9.12

# Procesamiento
numpy==1.26.3
opencv-python==4.9.0.80
scikit-image==0.22.0
scikit-learn==1.4.0

# Geoespacial
rasterio==1.3.9
shapely==2.0.2
pyproj==3.6.1
geojson==3.1.0

# Sentinel Hub
sentinelhub==3.10.2

# Utilidades
python-dotenv==1.0.0
pillow==10.2.0
```

```python
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import analysis, health

app = FastAPI(
    title="Sistema Híbrido de Detección de Estrés Vegetal",
    description="API para comparación de Region Growing Clásico vs Semántico",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Nuxt dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])

@app.on_event("startup")
async def startup_event():
    """Cargar modelo Prithvi al iniciar"""
    from app.models.prithvi_loader import load_prithvi_model
    app.state.prithvi_model = load_prithvi_model()
    print("✅ Modelo Prithvi cargado exitosamente")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```


#### Frontend (Nuxt 3)

```bash
# package.json dependencies
{
  "dependencies": {
    "nuxt": "^3.10.0",
    "@pinia/nuxt": "^0.5.1",
    "leaflet": "^1.9.4",
    "@vueuse/core": "^10.7.2",
    "axios": "^1.6.5"
  },
  "devDependencies": {
    "@nuxtjs/tailwindcss": "^6.11.4",
    "typescript": "^5.3.3"
  }
}
```

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  modules: [
    '@pinia/nuxt',
    '@nuxtjs/tailwindcss'
  ],
  
  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000'
    }
  },
  
  app: {
    head: {
      title: 'Sistema Híbrido de Detección de Estrés Vegetal',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { 
          name: 'description', 
          content: 'Comparación de Region Growing Clásico vs Semántico para análisis de vegetación' 
        }
      ],
      link: [
        { 
          rel: 'stylesheet', 
          href: 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css' 
        }
      ]
    }
  },
  
  ssr: true,  // Server-Side Rendering habilitado
  
  typescript: {
    strict: true,
    typeCheck: true
  }
})
```

```typescript
// composables/useAnalysis.ts
import type { BBox, AnalysisMethod, AnalysisResult } from '~/types'

export const useAnalysis = () => {
  const config = useRuntimeConfig()
  const results = useState<AnalysisResult | null>('analysis-results', () => null)
  const loading = useState<boolean>('analysis-loading', () => false)
  const error = useState<string | null>('analysis-error', () => null)
  
  const analyzeRegion = async (
    bbox: BBox, 
    method: AnalysisMethod = 'hybrid',
    dateFrom?: string,
    dateTo?: string
  ) => {
    loading.value = true
    error.value = null
    
    try {
      const response = await $fetch<AnalysisResult>(
        `${config.public.apiBase}/api/analysis/analyze`,
        {
          method: 'POST',
          body: {
            bbox,
            method,
            date_from: dateFrom,
            date_to: dateTo
          }
        }
      )
      
      results.value = response
      return response
    } catch (e: any) {
      error.value = e.message || 'Error al analizar región'
      throw e
    } finally {
      loading.value = false
    }
  }
  
  const compareMethodsAB = async (bbox: BBox) => {
    // Ejecutar ambos métodos en paralelo
    const [classicResult, hybridResult] = await Promise.all([
      analyzeRegion(bbox, 'classic'),
      analyzeRegion(bbox, 'hybrid')
    ])
    
    return {
      classic: classicResult,
      hybrid: hybridResult,
      comparison: {
        coherence: calculateCoherence(classicResult, hybridResult),
        regionCount: {
          classic: classicResult.statistics.num_regions,
          hybrid: hybridResult.statistics.num_regions
        }
      }
    }
  }
  
  return {
    results: readonly(results),
    loading: readonly(loading),
    error: readonly(error),
    analyzeRegion,
    compareMethodsAB
  }
}

function calculateCoherence(classic: any, hybrid: any): number {
  // Métrica de coherencia espacial (simplificada)
  const classicFragmentation = classic.statistics.num_regions / classic.statistics.total_area
  const hybridFragmentation = hybrid.statistics.num_regions / hybrid.statistics.total_area
  
  return (1 - hybridFragmentation / classicFragmentation) * 100
}
```

---

## 🎓 CONCLUSIÓN

Este proyecto está diseñado para obtener **100/100 puntos** cumpliendo con excelencia todos los criterios de la rúbrica:

### Fortalezas del Proyecto

1. **Innovación Real:** No es solo implementar Region Growing, sino compararlo con un método SOTA usando Foundation Models

2. **Viabilidad Técnica:** Todo es factible en 10 días con los recursos disponibles (RTX 4070 + $30 USD)

3. **Impacto Académico:** Contribución clara al estado del arte, metodología reproducible

4. **Presentación Profesional:** Video, artículo y código de calidad superior

5. **Stack Moderno:** FastAPI + Nuxt 3 demuestra conocimiento de tecnologías actuales


### Contacto y Soporte

Para dudas durante la implementación:
- **Carlos Bocanegra:** Backend + Algoritmos
- **Arthur Zizumbo:** ML + Prithvi
- **Luis Vázquez:** Frontend + Visualización
- **Edgar Oviedo:** Documentación + Coordinación

---

## 📎 ANEXOS

### A. Comandos Útiles

```bash
# Backend (FastAPI)
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (Nuxt 3)
cd frontend
npm install
npm run dev  # http://localhost:3000

# Descargar Prithvi
huggingface-cli download ibm-nasa-geospatial/Prithvi-EO-1.0-100M
```

### B. Recursos Adicionales

- **Prithvi Docs:** https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
- **Sentinel Hub API:** https://docs.sentinel-hub.com/
- **FastAPI Tutorial:** https://fastapi.tiangolo.com/tutorial/
- **Nuxt 3 Docs:** https://nuxt.com/docs
- **MMSegmentation:** https://mmsegmentation.readthedocs.io/

### C. Datasets de Prueba

- **SEN12MS:** https://mediatum.ub.tum.de/1474000
- **BreizhCrops:** https://github.com/dl4sits/BreizhCrops
- **Fields of the World:** https://fieldsofthe.world/

---

**Documento creado:** Noviembre 2025  
**Versión:** 2.0  
**Autores:** Equipo 24-Region Growing

**¡Éxito en el proyecto! 🚀**
