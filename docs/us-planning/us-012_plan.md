# US-013: Crear Google Colab Ejecutable de Excelencia - PLANEACIÓN DEFINITIVA

**Estado:** 📋 EN PLANEACIÓN (LISTA PARA APROBACIÓN)
**Prioridad:** CRÍTICA (Entregable Final - 40% de la calificación)
**Estimación:** 16-20 horas
**Responsables:** 
- Carlos Bocanegra (Tech Lead - Implementación RG Clásico, integración pipeline)
- Edgar Oviedo (Documentation Lead - Narrativa, teoría, conclusiones)
- Arthur Zizumbo (ML Engineer - MGRG, Prithvi, clasificación, comparativas A/B)
- Luis Vázquez (Visualization Lead - Gráficos)
**Sprint:** Documentación y Entrega (Días 8-10)
**Fecha de Planeación:** 13 de Noviembre de 2025
**Versión:** 2.0 (Actualizada con contexto completo de US-001 a US-011)

---

## 🎯 Objetivo de la User Story

**Como** equipo de desarrollo
**Quiero** crear un Google Colab ejecutable de principio a fin que demuestre TODO el proyecto con excelencia académica y técnica
**Para que** tengamos un entregable demo profesional, reproducible y educativo que:
- Demuestre dominio completo del sistema híbrido (RG Clásico + MGRG)
- Sirva como material educativo para la presentación final
- Evidencie excelencia técnica y académica (40% de la calificación)
- Sea referencia para futuros trabajos en el área
- Cumpla 100% con estándares AGENTS.md y rúbrica del curso

---

## 📊 Contexto Completo del Proyecto (US-001 a US-011)

### Estado Actual - Sistema Completamente Funcional

Después de completar exitosamente US-001 a US-011, tenemos un sistema robusto y completo:

#### Backend y Arquitectura (US-001, US-003, US-004)
✅ **FastAPI Backend** (US-001): 
- API REST con Swagger docs automático
- Logging profesional y timeouts configurables
- 13 tests unitarios, 100% funcional
- Puerto 8070, CORS configurado para frontend

✅ **Arquitectura Limpia** (US-003):
- Código reutilizable en `src/` (no duplicación)
- 27 tests unitarios, 78% cobertura
- Poetry unificado en raíz del proyecto
- PyTorch 2.9.0+cu129 con CUDA 12.9

✅ **Region Growing Clásico** (US-004):
- Algoritmo optimizado: ~10-12M pixels/sec
- 22 tests unitarios, 99% cobertura en algoritmo
- Profiling completo con benchmarks
- Parámetros óptimos documentados (threshold=0.1, min_size=50)

#### Frontend (US-002)
✅ **Nuxt 3 Frontend**:
- SSR configurado, auto-imports funcionando
- MapLibre GL con capas raster georeferenciadas
- Visualización RGB, Falso Color, NDVI
- 100% paridad funcional + 10 features nuevas

#### ML y Segmentación Semántica (US-005, US-006, US-007)
✅ **Prithvi Integration** (US-005, US-006):
- Modelo Prithvi-EO-1.0-100M (NASA/IBM) funcional
- Extracción de embeddings 256D, L2-normalizados
- 3 zonas procesadas: Mexicali, Bajío, Sinaloa (3.31M vectores, 2.95 GB)
- 27 tests unitarios para HLS processor, 59% cobertura
- Validación de datos robusta (detección de imágenes vacías/ceros)

✅ **MGRG Semántico** (US-007):
- Algoritmo completo con BFS sobre embeddings
- Cosine similarity como criterio (threshold=0.85-0.95)
- Comparación Grid vs K-Means: **Grid 19x más rápido**
- 34 tests unitarios, 82% cobertura
- Conclusión experimental: Grid superior en práctica

#### Clasificación y Análisis (US-008, US-009, US-010, US-011)
✅ **Comparativa A/B Visual** (US-008):
- Sistema completo de métricas (IoU, coherencia, regiones)
- Visualizaciones profesionales (300 DPI, PNG/PDF/SVG)
- 52 tests (25 métricas + 18 visualización + 9 integración)
- 95% cobertura de código
- Análisis de threshold: 0.95 óptimo para separación de clases

✅ **Validación con Ground Truth** (US-009):
- Dynamic World 2024 + ESA WorldCover 2021
- Métricas estándar: mIoU, Weighted mIoU, F1, Precision/Recall
- **MGRG +252.8% mejor que Classic RG** (mIoU: 0.1349 vs 0.0382)
- 3 zonas validadas con datos reales
- Confusion matrices generadas (9 figuras 300 DPI)

✅ **Clasificación Semántica** (US-010):
- 6 clases bilingües (inglés/español): Water, Urban, Bare Soil, Vigorous Crop, Stressed Crop, Grass/Shrub
- Clasificador zero-shot con NDVI + embeddings
- Cross-validation con Dynamic World (53.9% agreement en Mexicali)
- 20+ tests unitarios planificados

✅ **Pipeline End-to-End** (US-011):
- API REST: `/api/analysis/hierarchical`
- CLI Script: `scripts/analyze_region.py`
- 7 pasos completos: Descarga → Embeddings → Segmentación → NDVI → Clasificación → Estrés → Reporte
- 10 tests integración, 78% cobertura
- Outputs: JSON, GeoTIFF, PNG (300 DPI)

### Datos Reales Disponibles

**3 Zonas Agrícolas de México (15 Enero 2024):**

1. **Valle de Mexicali, Baja California**
   - Imagen HLS: (6, 1124, 922) - 1.04M vectores
   - Embeddings: 922 MB
   - Cultivos: Trigo, algodón, alfalfa (riego intensivo)
   - Classic RG: 207 regiones, mIoU=0.1123
   - MGRG: 24 regiones, mIoU=0.1224 (+9%)

2. **El Bajío, Guanajuato**
   - Imagen HLS: (6, 1092, 1057) - 1.15M vectores
   - Embeddings: 1.03 GB
   - Cultivos: Sorgo, maíz, hortalizas (agricultura diversa)
   - Classic RG: 775 regiones, mIoU=0.0020
   - MGRG: 10 regiones, mIoU=0.1544 (+7620%)

3. **Valle de Culiacán, Sinaloa**
   - Imagen HLS: (6, 1090, 1031) - 1.12M vectores
   - Embeddings: 1.00 GB
   - Cultivos: Tomate, chile, maíz (agricultura tecnificada)
   - Classic RG: 934 regiones, mIoU=0.0004
   - MGRG: 19 regiones, mIoU=0.1278 (+31850%)

### Plantilla Actual

Tenemos `notebooks/final/region_growing_equipo24.ipynb` con:
- Estructura académica básica (introducción, fundamentos, metodología)
- Secciones teóricas completas
- Placeholders para código e implementación
- Referencias bibliográficas

**Problema:** La plantilla es escueta y NO incluye:
- ❌ Código ejecutable real con datos reales
- ❌ Implementación completa de ambos métodos (RG Clásico + MGRG)
- ❌ Comparativa A/B funcional con métricas cuantitativas
- ❌ Visualizaciones profesionales (300 DPI)
- ❌ Integración con el pipeline completo (US-011)
- ❌ Análisis de resultados reales de 3 zonas
- ❌ Validación con Dynamic World
- ❌ Clasificación semántica bilingüe
- ❌ Análisis de threshold y sensibilidad
- ❌ Casos de fallo documentados

---

## 🎓 Criterios de Aceptación Expandidos (Basados en Rúbrica 40%)

### Criterios Originales (Mínimos - 10 puntos)

✅ Notebook limpio y bien documentado
✅ Celdas de markdown explicativas entre código
✅ Ambos métodos implementados (RG Clásico + MGRG)
✅ Comparativa A/B funcional con visualizaciones
✅ Ejecutable sin errores de principio a fin
✅ Sección de roles del equipo al final
✅ Requirements especificados
✅ Imágenes de ejemplo incluidas
✅ Comentarios en código complejo

### Criterios de Excelencia (Nuestro Estándar)

#### 1. Estructura y Organización
- [ ] **Portada profesional** con logos, nombres completos, matrícula, fecha
- [ ] **Tabla de contenidos** interactiva con enlaces
- [ ] **Resumen ejecutivo** (español e inglés) de 200-250 palabras
- [ ] **Secciones claramente delimitadas** con numeración jerárquica
- [ ] **Flujo narrativo coherente** de teoría → implementación → resultados
- [ ] **Transiciones suaves** entre secciones con contexto

#### 2. Fundamentos Teóricos
- [ ] **Introducción contextualizada** (1-2 páginas markdown)
  - Problema de detección de estrés vegetal
  - Importancia de la agricultura de precisión
  - Gap en métodos tradicionales
  - Nuestra propuesta de valor
- [ ] **Estado del Arte** (2-3 páginas markdown)
  - Region Growing clásico (historia, algoritmo, aplicaciones)
  - Foundation Models en teledetección (Prithvi, SatMAE)
  - Hibridación DL-OBIA (marco teórico)
  - Referencias académicas integradas (15+ papers 2022-2025)
- [ ] **Fundamentos matemáticos** con LaTeX
  - Ecuaciones de NDVI, NDWI
  - Criterio de homogeneidad clásico: |I(x,y) - I(s)| < T
  - Criterio semántico: cosine_similarity(emb_A, emb_B) > threshold
  - Métricas de evaluación (IoU, coherencia espacial)

#### 3. Implementación Técnica
- [ ] **Setup e Instalación** (celda ejecutable)
  - Instalación de dependencias con pip
  - Verificación de versiones
  - Configuración de credenciales Sentinel Hub
  - Imports organizados por categoría
- [ ] **Descarga de Datos** (código funcional)
  - Integración con Sentinel Hub API
  - Descarga de bandas HLS (B02, B03, B04, B8A, B11, B12)
  - Manejo de errores y validación
  - Visualización de imagen RGB
- [ ] **Método 1: Region Growing Clásico** (implementación completa)
  - Cálculo de NDVI
  - Generación de semillas (grid o manual)
  - Algoritmo BFS con criterio espectral
  - Clasificación de estrés (alto/medio/bajo)
  - Visualización de resultados
  - Métricas: número de regiones, coherencia
- [ ] **Método 2: MGRG (Region Growing Semántico)** (implementación completa)
  - Carga de modelo Prithvi desde HuggingFace
  - Extracción de embeddings (256D)
  - Generación de semillas inteligentes (K-Means)
  - Algoritmo BFS con cosine similarity
  - Clasificación semántica (6 clases bilingües)
  - Análisis jerárquico (objeto → estrés)
  - Visualización de resultados
  - Métricas: IoU, coherencia, precisión de bordes

#### 4. Comparativa A/B (Sección Crítica)
- [ ] **Visualización lado a lado** (2x2 o 2x3 grid)
  - Imagen original RGB
  - NDVI calculado
  - Resultado RG Clásico
  - Resultado MGRG
  - Mapa semántico clasificado
  - Análisis de estrés
- [ ] **Métricas cuantitativas** (tabla comparativa)
  - Coherencia espacial (%)
  - Número de regiones
  - Precisión de límites (si hay ground truth)
  - Tiempo de procesamiento (segundos)
  - Memoria utilizada (MB)
- [ ] **Análisis cualitativo** (markdown explicativo)
  - Fortalezas de cada método
  - Casos de uso recomendados
  - Limitaciones identificadas
  - Interpretación de resultados
- [ ] **Casos de estudio** (mínimo 2)
  - Caso 1: Campo agrícola con sombra de nube
  - Caso 2: Zona montañosa con vegetación dispersa
  - Caso 3 (opcional): Cultivo con riego por goteo

#### 5. Validación y Resultados
- [ ] **Cross-validation con Dynamic World** (si disponible)
  - Descarga de ground truth
  - Cálculo de agreement por clase
  - Matriz de confusión
  - Análisis de errores (FP, FN)
- [ ] **Análisis estadístico**
  - Distribución de NDVI por clase
  - Histogramas y boxplots
  - Correlaciones entre métricas
- [ ] **Visualizaciones profesionales**
  - Matplotlib con estilo personalizado
  - Colores consistentes (paleta del proyecto)
  - Títulos, ejes y leyendas claros
  - Resolución 300 DPI para figuras
  - Anotaciones explicativas

#### 6. Integración con Pipeline End-to-End
- [ ] **Demostración del CLI** (opcional pero recomendado)
  - Ejecución de `scripts/analyze_region.py`
  - Mostrar outputs generados (JSON, GeoTIFF, PNG)
  - Explicar uso en producción
- [ ] **Demostración del API REST** (opcional)
  - Request a `/api/analysis/hierarchical`
  - Polling de status
  - Descarga de resultados
  - Explicar integración con frontend

#### 7. Documentación y Reproducibilidad
- [ ] **Sección de Requirements** (celda markdown)
  - Lista completa de dependencias con versiones
  - Comando de instalación: `pip install -r requirements.txt`
  - Alternativa: `!pip install package==version` en celdas
- [ ] **Configuración de credenciales** (celda markdown)
  - Instrucciones para obtener Sentinel Hub API keys
  - Configuración de variables de entorno
  - Manejo seguro de secretos (no hardcodear)
- [ ] **Datos de ejemplo** (incluidos o descargables)
  - Imágenes Sentinel-2 pre-descargadas (opcional)
  - Embeddings pre-calculados (opcional)
  - Links a Google Drive o HuggingFace
- [ ] **Troubleshooting** (sección markdown)
  - Errores comunes y soluciones
  - Verificación de instalación
  - Contacto para soporte

#### 8. Conclusiones y Trabajo Futuro
- [ ] **Resumen de hallazgos** (1-2 páginas markdown)
  - Ventajas del método híbrido
  - Limitaciones identificadas
  - Aplicabilidad práctica
- [ ] **Trabajo futuro** (lista concreta)
  - Fine-tuning de Prithvi
  - Análisis temporal (series de tiempo)
  - Integración con otros sensores
  - Optimización de performance
- [ ] **Impacto y aplicaciones** (markdown)
  - Agricultura de precisión
  - Monitoreo forestal
  - Seguros paramétricos
  - Gestión de recursos hídricos

#### 9. Roles del Equipo (Sección Final)
- [ ] **Tabla de contribuciones** (markdown)
  - Nombre completo y matrícula
  - Rol principal
  - Contribuciones específicas
  - Horas invertidas (estimado)
- [ ] **Foto del equipo** (opcional pero recomendado)
- [ ] **Agradecimientos** (opcional)
  - Profesor Gilberto Ochoa
  - Instituciones (NASA, IBM, ESA)
  - Recursos utilizados

#### 10. Cumplimiento AGENTS.md
- [ ] **Código en inglés** (funciones, variables, clases)
- [ ] **Documentación en español** (celdas markdown narrativas)
- [ ] **Comentarios en inglés** (inline en código)
- [ ] **Type hints** en funciones complejas
- [ ] **Docstrings estilo Google** en funciones reutilizables
- [ ] **Sin emojis** en código Python
- [ ] **Logging profesional** (logger, no print) cuando aplique
- [ ] **Nombres bilingües** en outputs (inglés/español)

---

## 🏗️ Arquitectura del Notebook

### Estructura Propuesta (15-20 secciones)

```
SECCIÓN 0: PORTADA Y METADATA
├── Título del proyecto
├── Información del equipo (nombres, matrículas)
├── Institución y materia
├── Profesor
├── Fecha
└── Logos (ITESM, opcional)

SECCIÓN 1: TABLA DE CONTENIDOS
└── Enlaces interactivos a secciones principales

SECCIÓN 2: RESUMEN EJECUTIVO
├── Resumen en español (300 palabras)
├── Abstract en inglés (300 palabras)
└── Palabras clave

SECCIÓN 3: INTRODUCCIÓN
├── 3.1 Contexto y Motivación
├── 3.2 Problema a Resolver
├── 3.3 Objetivos del Proyecto
└── 3.4 Estructura del Notebook

SECCIÓN 4: ESTADO DEL ARTE
├── 4.1 Region Growing Clásico
├── 4.2 Foundation Models en PErcepción Remota
├── 4.3 Hibridación DL-OBIA
└── 4.4 Nuestra Propuesta: MGRG

SECCIÓN 5: FUNDAMENTOS TEÓRICOS
├── 5.1 Algoritmo Region Growing
├── 5.2 Índices Espectrales (NDVI, NDWI)
├── 5.3 Embeddings Semánticos
├── 5.4 Cosine Similarity
└── 5.5 Métricas de Evaluación

SECCIÓN 6: SETUP E INSTALACIÓN
├── 6.1 Instalación de Dependencias
├── 6.2 Imports y Configuración
├── 6.3 Verificación de Versiones
└── 6.4 Configuración de Credenciales

SECCIÓN 7: DESCARGA DE DATOS
├── 7.1 Conexión a Sentinel Hub
├── 7.2 Definición de Área de Interés (BBox)
├── 7.3 Descarga de Bandas HLS
├── 7.4 Visualización de Imagen RGB
└── 7.5 Preprocesamiento Inicial

SECCIÓN 8: MÉTODO 1 - REGION GROWING CLÁSICO
├── 8.1 Cálculo de NDVI
├── 8.2 Generación de Semillas
├── 8.3 Implementación del Algoritmo BFS
├── 8.4 Clasificación de Estrés
├── 8.5 Visualización de Resultados
└── 8.6 Métricas de Desempeño

SECCIÓN 9: MÉTODO 2 - MGRG (REGION GROWING SEMÁNTICO)
├── 9.1 Carga del Modelo Prithvi
├── 9.2 Extracción de Embeddings
├── 9.3 Generación de Semillas Inteligentes (K-Means)
├── 9.4 Implementación del Algoritmo BFS Semántico
├── 9.5 Clasificación Semántica (6 Clases)
├── 9.6 Análisis Jerárquico (Objeto → Estrés)
├── 9.7 Visualización de Resultados
└── 9.8 Métricas de Desempeño

SECCIÓN 10: COMPARATIVA A/B
├── 10.1 Visualización Lado a Lado
├── 10.2 Métricas Cuantitativas
├── 10.3 Análisis Cualitativo
└── 10.4 Casos de Estudio

SECCIÓN 11: VALIDACIÓN CON DYNAMIC WORLD
├── 11.1 Descarga de Ground Truth
├── 11.2 Alineación Espacial
├── 11.3 Cálculo de Agreement
├── 11.4 Matriz de Confusión
└── 11.5 Análisis de Errores

SECCIÓN 12: ANÁLISIS ESTADÍSTICO
├── 12.1 Distribución de NDVI por Clase
├── 12.2 Histogramas y Boxplots
├── 12.3 Correlaciones
└── 12.4 Significancia Estadística

SECCIÓN 13: INTEGRACIÓN CON PIPELINE END-TO-END
├── 13.1 Demostración del CLI
└── 13.2 Uso en Producción

SECCIÓN 14: DISCUSIÓN
├── 14.1 Fortalezas del Método Híbrido
├── 14.2 Limitaciones Identificadas
├── 14.3 Aplicabilidad Práctica
└── 14.4 Comparación con SOTA

SECCIÓN 15: CONCLUSIONES
├── 15.1 Resumen de Hallazgos
├── 15.2 Contribuciones del Proyecto
└── 15.3 Impacto Esperado

SECCIÓN 16: TRABAJO FUTURO
├── 16.1 Mejoras Técnicas
├── 16.2 Extensiones Propuestas
└── 16.3 Aplicaciones Potenciales

SECCIÓN 17: REFERENCIAS
└── Bibliografía completa (15+ papers en formato APA 7)

SECCIÓN 18: ROLES DEL EQUIPO
├── Tabla de Contribuciones
├── Foto de cada integrante del Equipo (opcional)
└── Agradecimientos

SECCIÓN 19: APÉNDICES (opcional)
├── A. Código Completo de Funciones Auxiliares
├── B. Configuración de Entorno
└── C. Troubleshooting
```

---

## 📋 Plan de Implementación Detallado

### Fase 1: Preparación y Setup (2-3 horas)

#### Tarea 1.1: Análisis de la Plantilla Actual
**Responsable:** Carlos Bocanegra
**Duración:** 30 min

**Actividades:**
- Revisar `notebooks/final/region_growing_equipo24.ipynb`
- Identificar secciones reutilizables
- Mapear contenido teórico existente
- Identificar gaps de implementación

**Entregable:** Lista de secciones a mantener/modificar/crear

#### Tarea 1.2: Configuración del Entorno Colab
**Responsable:** Arthur Zizumbo
**Duración:** 1 hora

**Actividades:**
- Crear nuevo notebook en Google Colab
- Configurar runtime (GPU T4 recomendado)
- Instalar dependencias base
- Verificar acceso a Sentinel Hub API
- Probar carga de Prithvi desde HuggingFace

**Código de verificación:**
```python
# Verify GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Verify Sentinel Hub
from sentinelhub import SHConfig
config = SHConfig()
print(f"Sentinel Hub configured: {config.sh_client_id is not None}")

# Verify Prithvi
from transformers import AutoModel
model = AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-EO-1.0-100M")
print(f"Prithvi loaded: {model is not None}")
```

**Entregable:** Notebook con setup funcional

#### Tarea 1.3: Preparación de Datos de Ejemplo
**Responsable:** Luis Vázquez
**Duración:** 1 hora

**Actividades:**
- Seleccionar 2-3 regiones de interés (Mexicali, Bajío, Sinaloa)
- Pre-descargar imágenes Sentinel-2 (opcional)
- Subir a Google Drive o HuggingFace
- Crear celdas de descarga alternativa

**Entregable:** Links a datos de ejemplo + código de descarga

#### Tarea 1.4: Creación de Estructura Base
**Responsable:** Edgar Oviedo
**Duración:** 30 min

**Actividades:**
- Crear secciones markdown con títulos
- Agregar tabla de contenidos
- Insertar placeholders para código
- Agregar portada y metadata

**Entregable:** Notebook con estructura completa (sin código)

---

### Fase 2: Implementación de Fundamentos (3-4 horas)

#### Tarea 2.1: Secciones Teóricas
**Responsable:** Edgar Oviedo
**Duración:** 2 horas

**Actividades:**
- Redactar Introducción (español)
- Redactar Estado del Arte con referencias
- Escribir Fundamentos Teóricos con LaTeX
- Integrar ecuaciones matemáticas
- Agregar diagramas de flujo (Mermaid o imágenes)

**Ejemplo de ecuación LaTeX:**
```markdown
El NDVI se calcula como:

$$
NDVI = \frac{NIR - Red}{NIR + Red}
$$

donde $NIR$ es la reflectancia en el infrarrojo cercano (banda B8A) y $Red$ es la reflectancia en el rojo (banda B04).
```

**Entregable:** Secciones 2-5 completas

#### Tarea 2.2: Setup e Instalación
**Responsable:** Arthur Zizumbo
**Duración:** 1 hora

**Actividades:**
- Escribir celda de instalación de dependencias
- Crear celda de imports organizados
- Agregar verificación de versiones
- Documentar configuración de credenciales

**Código de instalación:**
```python
# Install dependencies
!pip install -q sentinelhub==3.10.2
!pip install -q torch==2.1.2 torchvision==0.16.2
!pip install -q transformers==4.36.0
!pip install -q rasterio==1.3.9
!pip install -q scikit-learn==1.4.0
!pip install -q matplotlib==3.8.2
!pip install -q seaborn==0.13.0

# Verify installations
import sentinelhub
import torch
import transformers
print(f"sentinelhub: {sentinelhub.__version__}")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
```

**Entregable:** Sección 6 completa y funcional

#### Tarea 2.3: Descarga de Datos Sentinel-2
**Responsable:** Carlos Bocanegra
**Duración:** 1 hora

**Actividades:**
- Implementar conexión a Sentinel Hub
- Crear función de descarga de bandas HLS
- Agregar visualización RGB
- Manejar errores comunes

**Código de descarga:**
```python
from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest, MimeType

# Configure Sentinel Hub
config = SHConfig()
config.sh_client_id = 'YOUR_CLIENT_ID'
config.sh_client_secret = 'YOUR_CLIENT_SECRET'

# Define area of interest (Mexicali example)
bbox = BBox(bbox=[-115.35, 32.45, -115.25, 32.55], crs=CRS.WGS84)

# Evalscript for HLS bands
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

# Download request
request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=('2025-10-01', '2025-10-31'),
        )
    ],
    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
    bbox=bbox,
    size=(512, 512),
    config=config
)

# Execute download
hls_data = request.get_data()[0]
print(f"Downloaded HLS data: {hls_data.shape}")  # (512, 512, 6)
```

**Entregable:** Sección 7 completa con datos descargados

---

### Fase 3: Implementación de Métodos (4-5 horas)

#### Tarea 3.1: Region Growing Clásico
**Responsable:** Carlos Bocanegra
**Duración:** 2 horas

**Actividades:**
- Implementar cálculo de NDVI
- Crear función de generación de semillas
- Implementar algoritmo BFS con criterio espectral
- Agregar clasificación de estrés
- Crear visualizaciones
- Calcular métricas

**Código de RG Clásico:**
```python
import numpy as np
from collections import deque

def calculate_ndvi(hls_data):
    """Calculate NDVI from HLS data."""
    nir = hls_data[:, :, 3]  # B8A
    red = hls_data[:, :, 2]  # B04
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi

def generate_grid_seeds(shape, spacing=20):
    """Generate grid of seed points."""
    h, w = shape
    seeds = []
    for y in range(spacing, h, spacing):
        for x in range(spacing, w, spacing):
            seeds.append((y, x))
    return seeds

def region_growing_classic(ndvi, seeds, threshold=0.1, min_size=50):
    """Classic region growing with NDVI homogeneity."""
    h, w = ndvi.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    region_id = 1
    
    for seed_y, seed_x in seeds:
        if labeled[seed_y, seed_x] != 0:
            continue
        
        seed_value = ndvi[seed_y, seed_x]
        queue = deque([(seed_y, seed_x)])
        region_pixels = []
        
        while queue:
            y, x = queue.popleft()
            
            if not (0 <= y < h and 0 <= x < w):
                continue
            if labeled[y, x] != 0:
                continue
            
            pixel_value = ndvi[y, x]
            if abs(pixel_value - seed_value) <= threshold:
                labeled[y, x] = region_id
                region_pixels.append((y, x))
                
                # Add neighbors (4-connectivity)
                queue.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
        
        if len(region_pixels) >= min_size:
            region_id += 1
        else:
            for y, x in region_pixels:
                labeled[y, x] = 0
    
    return labeled

# Execute
ndvi = calculate_ndvi(hls_data)
seeds = generate_grid_seeds(ndvi.shape, spacing=20)
segmentation_classic = region_growing_classic(ndvi, seeds, threshold=0.1)

print(f"Classic RG: {np.max(segmentation_classic)} regions")
```

**Entregable:** Sección 8 completa con resultados

#### Tarea 3.2: MGRG (Region Growing Semántico)
**Responsable:** Arthur Zizumbo
**Duración:** 2.5 horas

**Actividades:**
- Cargar modelo Prithvi desde HuggingFace
- Implementar extracción de embeddings
- Crear generación de semillas inteligentes (K-Means)
- Implementar algoritmo BFS con cosine similarity
- Agregar clasificación semántica
- Implementar análisis jerárquico
- Crear visualizaciones
- Calcular métricas

**Código de MGRG:**
```python
import torch
from transformers import AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load Prithvi model
model = AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-EO-1.0-100M")
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

def extract_embeddings(hls_data, model):
    """Extract 256D embeddings using Prithvi."""
    # Prepare input
    x = torch.from_numpy(hls_data).permute(2, 0, 1).unsqueeze(0).float()
    x = (x - x.mean()) / (x.std() + 1e-8)
    
    if torch.cuda.is_available():
        x = x.cuda()
    
    # Forward pass (encoder only)
    with torch.no_grad():
        features = model.encode(x)  # (1, 256, H', W')
    
    # Interpolate to original resolution
    if features.shape[2:] != hls_data.shape[:2]:
        features = torch.nn.functional.interpolate(
            features, 
            size=hls_data.shape[:2], 
            mode='bilinear'
        )
    
    # Convert to numpy (H, W, 256)
    embeddings = features.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Normalize embeddings (L2 norm)
    norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    return embeddings

def generate_smart_seeds(embeddings, n_clusters=5):
    """Generate smart seeds using K-Means clustering."""
    h, w, d = embeddings.shape
    emb_flat = embeddings.reshape(-1, d)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(emb_flat)
    
    # Find closest pixel to each centroid
    seeds = []
    for cluster_id in range(n_clusters):
        cluster_mask = (labels == cluster_id)
        cluster_embeddings = emb_flat[cluster_mask]
        centroid = kmeans.cluster_centers_[cluster_id]
        
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = np.argmin(distances)
        
        flat_idx = np.where(cluster_mask)[0][closest_idx]
        y, x = divmod(flat_idx, w)
        seeds.append((y, x))
    
    return seeds

def region_growing_semantic(embeddings, seeds, threshold=0.85, min_size=50):
    """Semantic region growing with cosine similarity."""
    h, w, d = embeddings.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    region_id = 1
    
    for seed_y, seed_x in seeds:
        if labeled[seed_y, seed_x] != 0:
            continue
        
        seed_emb = embeddings[seed_y, seed_x]
        queue = deque([(seed_y, seed_x)])
        region_pixels = []
        
        while queue:
            y, x = queue.popleft()
            
            if not (0 <= y < h and 0 <= x < w):
                continue
            if labeled[y, x] != 0:
                continue
            
            pixel_emb = embeddings[y, x]
            similarity = np.dot(seed_emb, pixel_emb)  # Already normalized
            
            if similarity >= threshold:
                labeled[y, x] = region_id
                region_pixels.append((y, x))
                
                queue.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
        
        if len(region_pixels) >= min_size:
            region_id += 1
        else:
            for y, x in region_pixels:
                labeled[y, x] = 0
    
    return labeled

# Execute
embeddings = extract_embeddings(hls_data, model)
seeds_smart = generate_smart_seeds(embeddings, n_clusters=5)
segmentation_mgrg = region_growing_semantic(embeddings, seeds_smart, threshold=0.85)

print(f"MGRG: {np.max(segmentation_mgrg)} regions")
print(f"Smart seeds: {len(seeds_smart)} clusters")
```

**Entregable:** Sección 9 completa con resultados

#### Tarea 3.3: Clasificación Semántica
**Responsable:** Arthur Zizumbo
**Duración:** 30 min

**Actividades:**
- Implementar clasificador zero-shot
- Clasificar todas las regiones
- Generar mapa semántico
- Crear visualización coloreada

**Código de clasificación:**
```python
# Land cover classes (bilingual)
LAND_COVER_CLASSES = {
    0: "Water (Agua)",
    1: "Urban (Urbano)",
    2: "Bare Soil (Suelo Desnudo)",
    3: "Vigorous Crop (Cultivo Vigoroso)",
    4: "Stressed Crop (Cultivo Estresado)",
    5: "Grass/Shrub (Pasto/Arbustos)"
}

def classify_region(region_mask, ndvi, embeddings):
    """Classify a single region using NDVI + embeddings."""
    region_ndvi = ndvi[region_mask]
    mean_ndvi = np.mean(region_ndvi)
    std_ndvi = np.std(region_ndvi)
    
    # Simple heuristic classification
    if mean_ndvi < 0.1:
        if std_ndvi < 0.05:
            class_id = 0  # Water
        else:
            class_id = 1  # Urban
    elif mean_ndvi < 0.3:
        class_id = 2  # Bare Soil
    elif mean_ndvi < 0.55:
        class_id = 4  # Stressed Crop
    elif mean_ndvi >= 0.55:
        if std_ndvi < 0.1:
            class_id = 3  # Vigorous Crop
        else:
            class_id = 5  # Grass/Shrub
    
    confidence = 1.0 - std_ndvi  # Simple confidence metric
    
    return class_id, confidence

# Classify all regions
classifications = {}
for region_id in range(1, np.max(segmentation_mgrg) + 1):
    region_mask = (segmentation_mgrg == region_id)
    class_id, confidence = classify_region(region_mask, ndvi, embeddings)
    
    classifications[region_id] = {
        'class_id': class_id,
        'class_name': LAND_COVER_CLASSES[class_id],
        'confidence': confidence,
        'mean_ndvi': np.mean(ndvi[region_mask]),
        'area_pixels': np.sum(region_mask)
    }

print(f"Classified {len(classifications)} regions")
```

**Entregable:** Clasificación completa de regiones

---

### Fase 4: Comparativa y Validación (2-3 horas)

#### Tarea 4.1: Visualización Comparativa A/B
**Responsable:** Luis Vázquez
**Duración:** 1.5 horas

**Actividades:**
- Crear grid de visualizaciones (2x3 o 2x4)
- Configurar estilo matplotlib profesional
- Agregar títulos, leyendas y anotaciones
- Exportar en alta resolución (300 DPI)

**Código de visualización:**
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configure matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create comparison figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Original data
axes[0, 0].imshow(hls_data[:, :, [2, 1, 0]])  # RGB
axes[0, 0].set_title('Original RGB Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
axes[0, 1].set_title('NDVI')
axes[0, 1].axis('off')

axes[0, 2].imshow(segmentation_classic, cmap='tab20')
axes[0, 2].set_title(f'Classic RG ({np.max(segmentation_classic)} regions)')
axes[0, 2].axis('off')

# Row 2: MGRG results
axes[1, 0].imshow(segmentation_mgrg, cmap='tab20')
axes[1, 0].set_title(f'MGRG ({np.max(segmentation_mgrg)} regions)')
axes[1, 0].axis('off')

# Semantic map
semantic_map = np.zeros_like(segmentation_mgrg)
for region_id, info in classifications.items():
    mask = (segmentation_mgrg == region_id)
    semantic_map[mask] = info['class_id']

axes[1, 1].imshow(semantic_map, cmap='tab10', vmin=0, vmax=5)
axes[1, 1].set_title('Semantic Classification')
axes[1, 1].axis('off')

# Legend for semantic map
legend_patches = [
    mpatches.Patch(color=plt.cm.tab10(i/10), label=LAND_COVER_CLASSES[i])
    for i in range(6)
]
axes[1, 2].legend(handles=legend_patches, loc='center', fontsize=9)
axes[1, 2].axis('off')
axes[1, 2].set_title('Class Legend')

plt.tight_layout()
plt.savefig('comparison_ab.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Entregable:** Sección 10 con visualizaciones profesionales

#### Tarea 4.2: Métricas Cuantitativas
**Responsable:** Carlos Bocanegra
**Duración:** 1 hora

**Actividades:**
- Calcular coherencia espacial
- Calcular número de regiones
- Medir tiempo de procesamiento
- Crear tabla comparativa

**Código de métricas:**
```python
import time

# Metrics calculation
def calculate_coherence(segmentation, ndvi):
    """Calculate spatial coherence (intra-region homogeneity)."""
    coherence_scores = []
    for region_id in range(1, np.max(segmentation) + 1):
        mask = (segmentation == region_id)
        if np.sum(mask) > 0:
            region_ndvi = ndvi[mask]
            std = np.std(region_ndvi)
            coherence = 1.0 - std  # Higher is better
            coherence_scores.append(coherence)
    return np.mean(coherence_scores) * 100  # Percentage

# Calculate metrics
metrics_classic = {
    'method': 'Classic RG',
    'regions': int(np.max(segmentation_classic)),
    'coherence': calculate_coherence(segmentation_classic, ndvi),
    'time': 12.3  # seconds (example)
}

metrics_mgrg = {
    'method': 'MGRG',
    'regions': int(np.max(segmentation_mgrg)),
    'coherence': calculate_coherence(segmentation_mgrg, ndvi),
    'time': 28.7  # seconds (example)
}

# Create comparison table
import pandas as pd

df_metrics = pd.DataFrame([metrics_classic, metrics_mgrg])
df_metrics = df_metrics.set_index('method')

print("\\n=== COMPARATIVE METRICS ===")
print(df_metrics.to_string())
print("\\n")

# Styled table for notebook
df_metrics.style.format({
    'coherence': '{:.1f}%',
    'time': '{:.1f}s'
}).background_gradient(cmap='RdYlGn', subset=['coherence'])
```

**Entregable:** Tabla de métricas comparativas

#### Tarea 4.3: Validación con Dynamic World (Opcional)
**Responsable:** Arthur Zizumbo
**Duración:** 30 min

**Actividades:**
- Descargar Dynamic World para la región
- Alinear espacialmente
- Calcular agreement por clase
- Crear matriz de confusión

**Nota:** Esta tarea es opcional si no hay tiempo suficiente.

**Entregable:** Sección 11 con validación (si aplica)

---

### Fase 5: Análisis y Conclusiones (2 horas)

#### Tarea 5.1: Análisis Estadístico
**Responsable:** Luis Vázquez
**Duración:** 1 hora

**Actividades:**
- Crear histogramas de NDVI por clase
- Generar boxplots comparativos
- Calcular correlaciones
- Crear visualizaciones estadísticas

**Código de análisis:**
```python
import seaborn as sns

# Distribution of NDVI by class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
for class_id, class_name in LAND_COVER_CLASSES.items():
    class_mask = (semantic_map == class_id)
    if np.sum(class_mask) > 0:
        class_ndvi = ndvi[class_mask]
        axes[0].hist(class_ndvi, bins=30, alpha=0.5, label=class_name)

axes[0].set_xlabel('NDVI')
axes[0].set_ylabel('Frequency')
axes[0].set_title('NDVI Distribution by Class')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Boxplot
ndvi_by_class = []
class_labels = []
for class_id, class_name in LAND_COVER_CLASSES.items():
    class_mask = (semantic_map == class_id)
    if np.sum(class_mask) > 0:
        ndvi_by_class.append(ndvi[class_mask])
        class_labels.append(class_name.split('(')[0].strip())

axes[1].boxplot(ndvi_by_class, labels=class_labels)
axes[1].set_ylabel('NDVI')
axes[1].set_title('NDVI Boxplot by Class')
axes[1].grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
```

**Entregable:** Sección 12 con análisis estadístico

#### Tarea 5.2: Discusión y Conclusiones
**Responsable:** Edgar Oviedo
**Duración:** 1 hora

**Actividades:**
- Redactar sección de Discusión (2-3 páginas markdown)
- Escribir Conclusiones (1 página markdown)
- Agregar Trabajo Futuro (1 página markdown)
- Revisar coherencia narrativa

**Contenido de Discusión:**
- Fortalezas del método híbrido (MGRG)
- Limitaciones identificadas
- Comparación con SOTA
- Aplicabilidad práctica
- Casos de uso recomendados

**Contenido de Conclusiones:**
- Resumen de hallazgos principales
- Contribuciones del proyecto
- Impacto esperado
- Recomendaciones finales

**Entregable:** Secciones 14-16 completas

---

### Fase 6: Finalización y Pulido (2 horas)

#### Tarea 6.1: Integración con Pipeline End-to-End
**Responsable:** Carlos Bocanegra
**Duración:** 30 min

**Actividades:**
- Agregar demostración del CLI (opcional)
- Mostrar uso del API REST (opcional)
- Explicar integración en producción

**Código de demostración:**
```python
# Example: Using the CLI script
!python scripts/analyze_region.py \\
  --bbox "-115.35,32.45,-115.25,32.55" \\
  --date "2025-10-15" \\
  --output "output/mexicali" \\
  --formats json,png

# Load and display results
import json
with open('output/mexicali/analysis_results.json', 'r') as f:
    results = json.load(f)

print(f"Total regions: {len(results['classification'])}")
print(f"Summary: {results['summary']}")
```

**Entregable:** Sección 13 con demostración

#### Tarea 6.2: Referencias y Roles del Equipo
**Responsable:** Edgar Oviedo
**Duración:** 30 min

**Actividades:**
- Compilar lista de referencias (15+ papers)
- Formatear en APA 7
- Crear tabla de roles del equipo
- Agregar foto del equipo (opcional)

**Formato de referencias:**
```markdown
## Referencias

1. Jakubik, J., Roy, S., Phillips, C. E., et al. (2024). Foundation models for generalist geospatial artificial intelligence. *arXiv preprint arXiv:2310.18660v2*. https://arxiv.org/abs/2310.18660

2. Ghamisi, P., Rasti, B., Yokoya, N., et al. (2022). Consistency-regularized region-growing network for semantic segmentation of urban scenes with point-level annotations. *IEEE Transactions on Image Processing*, 31, 5038–5051. https://doi.org/10.1109/TIP.2022.3188339

[... 13 more references ...]
```

**Tabla de roles:**
```markdown
## Roles del Equipo

| Nombre | Matrícula | Rol Principal | Contribuciones | Horas |
|--------|-----------|---------------|----------------|-------|
| Carlos Aaron Bocanegra Buitron | A01796345 | Tech Lead & Backend | FastAPI, RG Clásico, Pipeline | 40h |
| Arthur Jafed Zizumbo Velasco | A01796363 | ML Engineer | Prithvi, MGRG, Clasificación | 38h |
| Luis Santiago Vázquez Mancilla | A01796029 | Full Stack Developer | Nuxt 3, Visualizaciones | 35h |
| Edgar Oviedo Navarro | A01795260 | Product Owner & Documentation | Artículo, Video, Documentación | 37h |
```

**Entregable:** Secciones 17-18 completas

#### Tarea 6.3: Revisión Final y Testing
**Responsable:** Todos
**Duración:** 1 hora

**Actividades:**
- Ejecutar notebook completo de principio a fin
- Verificar que no hay errores
- Revisar ortografía y gramática
- Verificar enlaces y referencias
- Probar en Google Colab limpio
- Ajustar tiempos de ejecución
- Optimizar celdas lentas

**Checklist de revisión:**
- [ ] Todas las celdas ejecutan sin errores
- [ ] Visualizaciones se muestran correctamente
- [ ] Tablas están formateadas
- [ ] Ecuaciones LaTeX renderizan bien
- [ ] Enlaces funcionan
- [ ] Código está comentado
- [ ] Markdown está bien redactado
- [ ] No hay typos
- [ ] Tiempo total de ejecución <30 min

**Entregable:** Notebook final listo para entrega

---

## 📊 Estimación de Recursos

### Tiempo de Desarrollo

| Fase | Duración Estimada | Responsable Principal |
|------|-------------------|----------------------|
| Fase 1: Preparación | 2-3 horas | Carlos + Arthur |
| Fase 2: Fundamentos | 3-4 horas | Edgar + Arthur |
| Fase 3: Métodos | 4-5 horas | Carlos + Arthur |
| Fase 4: Comparativa | 2-3 horas | Luis + Carlos |
| Fase 5: Análisis | 2 horas | Luis + Edgar |
| Fase 6: Finalización | 2 horas | Todos |
| **TOTAL** | **15-19 horas** | **Equipo completo** |

### Distribución por Rol

| Rol | Responsable | Horas Estimadas |
|-----|-------------|-----------------|
| Tech Lead & Backend | Carlos Bocanegra | 6-7 horas |
| ML Engineer | Arthur Zizumbo | 6-7 horas |
| Full Stack Developer | Luis Vázquez | 3-4 horas |
| Documentation Lead | Edgar Oviedo | 4-5 horas |

### Recursos Computacionales

**Google Colab:**
- Runtime: GPU T4 (gratuito) o A100 (Colab Pro)
- RAM: 12-16 GB
- Disco: 100 GB

**Tiempo de Ejecución Estimado:**
- Setup e instalación: 2-3 min
- Descarga Sentinel-2: 5-10 min
- Extracción embeddings Prithvi: 10-15 min
- RG Clásico: 1-2 min
- MGRG: 3-5 min
- Clasificación: 1 min
- Visualizaciones: 2-3 min
- **TOTAL:** 25-40 min

**Optimizaciones:**
- Pre-descargar datos y subirlos a Google Drive
- Cachear embeddings de Prithvi
- Usar imágenes de menor resolución (256x256) para demos rápidas

---

## 🎯 Métricas de Éxito

### Criterios Técnicos

| Métrica | Target | Verificación |
|---------|--------|--------------|
| Ejecutable sin errores | 100% | Ejecutar 3 veces en Colab limpio |
| Tiempo de ejecución | <30 min | Cronometrar ejecución completa |
| Cobertura de secciones | 18/18 | Checklist de estructura |
| Calidad de visualizaciones | 300 DPI | Verificar resolución de imágenes |
| Referencias académicas | 15+ | Contar papers citados |
| Comentarios en código | >80% | Revisar funciones complejas |

### Criterios de Calidad

| Aspecto | Target | Verificación |
|---------|--------|--------------|
| Claridad narrativa | Excelente | Revisión por pares |
| Coherencia técnica | 100% | Validación de algoritmos |
| Ortografía y gramática | 0 errores | Corrector automático + manual |
| Formato profesional | Consistente | Revisión de estilo |
| Reproducibilidad | 100% | Test en 2+ máquinas |

### Criterios de Rúbrica (40% del proyecto)

| Criterio | Peso | Target | Estrategia |
|----------|------|--------|-----------|
| Código limpio y documentado | 10% | 10/10 | Comentarios, docstrings, type hints |
| Markdown explicativo | 10% | 10/10 | Narrativa clara entre celdas |
| Ambos métodos implementados | 10% | 10/10 | RG Clásico + MGRG funcionales |
| Comparativa A/B | 5% | 5/5 | Visualización profesional |
| Ejecutable de principio a fin | 5% | 5/5 | Testing exhaustivo |
| **TOTAL** | **40%** | **40/40** | **Excelencia en todos los aspectos** |

---

## 🚨 Riesgos y Mitigaciones

### Riesgos Técnicos

#### Riesgo 1: Prithvi no carga en Colab gratuito
**Probabilidad:** Media
**Impacto:** Alto
**Mitigación:**
- Usar Colab Pro ($10/mes) con GPU A100
- Pre-calcular embeddings y subirlos a Google Drive
- Implementar fallback con embeddings pre-calculados

#### Riesgo 2: Sentinel Hub API falla o excede límites
**Probabilidad:** Baja
**Impacto:** Alto
**Mitigación:**
- Pre-descargar imágenes y subirlas a Google Drive
- Incluir datos de ejemplo en el notebook
- Documentar proceso de descarga alternativo

#### Riesgo 3: Tiempo de ejecución >30 min
**Probabilidad:** Media
**Impacto:** Medio
**Mitigación:**
- Usar imágenes de menor resolución (256x256)
- Cachear resultados intermedios
- Optimizar algoritmos (vectorización)

#### Riesgo 4: Errores de dependencias
**Probabilidad:** Baja
**Impacto:** Medio
**Mitigación:**
- Especificar versiones exactas de paquetes
- Probar en Colab limpio antes de entregar
- Incluir sección de troubleshooting

### Riesgos de Proyecto

#### Riesgo 5: Falta de tiempo para completar todas las secciones
**Probabilidad:** Media
**Impacto:** Alto
**Mitigación:**
- Priorizar secciones críticas (métodos, comparativa)
- Trabajar en paralelo (división de tareas)
- Tener plan B con secciones mínimas

#### Riesgo 6: Calidad de visualizaciones no profesional
**Probabilidad:** Baja
**Impacto:** Medio
**Mitigación:**
- Usar templates de matplotlib profesionales
- Revisar ejemplos de papers académicos
- Iterar en diseño con feedback del equipo

---

## ✅ Checklist de Entrega Final

### Pre-Entrega (1 día antes)

- [ ] Notebook ejecuta sin errores (3 pruebas)
- [ ] Todas las secciones completas (18/18)
- [ ] Visualizaciones en alta resolución (300 DPI)
- [ ] Referencias formateadas en APA 7 (15+)
- [ ] Tabla de roles del equipo incluida
- [ ] Código comentado y documentado
- [ ] Markdown revisado (ortografía y gramática)
- [ ] Tiempo de ejecución <30 min
- [ ] Probado en Colab limpio
- [ ] Backup en Google Drive

### Día de Entrega

- [ ] Descargar notebook (.ipynb)
- [ ] Verificar que abre correctamente
- [ ] Incluir en ZIP con otros entregables
- [ ] Subir a plataforma antes de deadline
- [ ] Confirmar recepción

---

## 📚 Referencias para Implementación

### Papers Clave a Citar

1. **Jakubik et al. (2024)** - Prithvi Foundation Model
2. **Ghamisi et al. (2022)** - CRGNet (inspiración MGRG)
3. **Ma et al. (2024)** - DL-OBIA Hybridization
4. **Brown et al. (2022)** - Dynamic World
5. **Cong et al. (2022)** - SatMAE
6. **Tucker (1979)** - NDVI original
7. **Adams & Bischof (1994)** - Seeded Region Growing
8. **Drusch et al. (2012)** - Sentinel-2 mission

### Recursos Técnicos

- **Prithvi HuggingFace:** https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
- **Sentinel Hub API:** https://docs.sentinel-hub.com/
- **Google Colab Tips:** https://colab.research.google.com/notebooks/pro.ipynb
- **Matplotlib Gallery:** https://matplotlib.org/stable/gallery/index.html
- **LaTeX Math:** https://www.overleaf.com/learn/latex/Mathematical_expressions

### Ejemplos de Notebooks Académicos

- **SatMAE Demo:** https://github.com/sustainlab-group/SatMAE
- **Prithvi Examples:** https://github.com/NASA-IMPACT/hls-foundation-os
- **Region Growing Tutorial:** https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regiongrowing.html

---

## 🎓 Conclusión de la Planeación

Esta planeación detallada garantiza que el Google Colab ejecutable cumpla con:

✅ **Todos los criterios de aceptación originales**
✅ **Estándares de excelencia del proyecto (AGENTS.md)**
✅ **Requisitos de la rúbrica (40% del proyecto)**
✅ **Reproducibilidad y calidad académica**
✅ **Integración con el pipeline completo (US-001 a US-011)**

### Próximos Pasos Inmediatos

1. **Aprobar esta planeación** con el equipo
2. **Asignar tareas específicas** a cada miembro
3. **Crear calendario de trabajo** (días 8-10)
4. **Iniciar Fase 1** (Preparación y Setup)
5. **Reuniones diarias** de sincronización (15 min)

### Compromiso del Equipo

Con esta planeación, el equipo se compromete a entregar un Google Colab de **excelencia técnica y académica** que:

- Demuestre dominio completo del proyecto
- Sirva como referencia educativa
- Sea reproducible y ejecutable
- Cumpla con los más altos estándares de calidad
- Obtenga la máxima calificación posible (40/40 puntos)

---

**Planeación creada por:** Equipo 24 - Region Growing
**Fecha:** 13 de Noviembre de 2025
**Estado:** 📋 LISTA PARA APROBACIÓN
**Próxima acción:** Revisión y aprobación del equipo

🚀 **¡Listos para crear el mejor Google Colab del curso!**
- 7 pasos completos: Descarga → Embeddings → Segmentación → NDVI → Clasificación → Estrés → Reporte
- 10 tests integración, 78% cobertura
- Outputs: JSON, GeoTIFF, PNG (300 DPI)

### Datos Reales Disponibles

**3 Zonas Agrícolas de México (15 Enero 2024):**

1. **Valle de Mexicali, Baja California**
   - Imagen HLS: (6, 1124, 922) - 1.04M vectores
   - Embeddings: 922 MB
   - Cultivos: Trigo, algodón, alfalfa (riego intensivo)
   - Classic RG: 207 regiones, mIoU=0.1123
   - MGRG: 24 regiones, mIoU=0.1224 (+9%)

2. **El Bajío, Guanajuato**
   - Imagen HLS: (6, 1092, 1057) - 1.15M vectores
   - Embeddings: 1.03 GB
   - Cultivos: Sorgo, maíz, hortalizas (agricultura diversa)
   - Classic RG: 775 regiones, mIoU=0.0020
   - MGRG: 10 regiones, mIoU=0.1544 (+7620%)

3. **Valle de Culiacán, Sinaloa**
   - Imagen HLS: (6, 1090, 1031) - 1.12M vectores
   - Embeddings: 1.00 GB
   - Cultivos: Tomate, chile, maíz (agricultura tecnificada)
   - Classic RG: 934 regiones, mIoU=0.0004
   - MGRG: 19 regiones, mIoU=0.1278 (+31850%)

### Plantilla Actual vs Objetivo

**Plantilla Actual** (`notebooks/final/region_growing_equipo24.ipynb`):
- ✅ Estructura académica básica
- ✅ Secciones teóricas completas
- ✅ Referencias bibliográficas
- ❌ Sin código ejecutable real
- ❌ Sin implementación de métodos
- ❌ Sin datos reales
- ❌ Sin visualizaciones

**Objetivo Final** (Notebook de Excelencia):
- ✅ Todo lo anterior MÁS:
- ✅ Código 100% ejecutable con datos reales de 3 zonas
- ✅ Implementación completa RG Clásico + MGRG
- ✅ Comparativa A/B con métricas cuantitativas
- ✅ Validación con Dynamic World
- ✅ Clasificación semántica bilingüe
- ✅ Visualizaciones profesionales (300 DPI)
- ✅ Análisis de sensibilidad de parámetros
- ✅ Casos de fallo documentados
- ✅ Integración con pipeline end-to-end
- ✅ Conclusiones basadas en datos reales

---

## 🎓 Criterios de Aceptación Expandidos

### Nivel 1: Criterios Mínimos (10/40 puntos)

✅ Notebook limpio y bien documentado
✅ Celdas de markdown explicativas entre código
✅ Ambos métodos implementados (RG Clásico + MGRG)
✅ Comparativa A/B funcional con visualizaciones
✅ Ejecutable sin errores de principio a fin
✅ Sección de roles del equipo al final
✅ Requirements especificados
✅ Imágenes de ejemplo incluidas
✅ Comentarios en código complejo

### Nivel 2: Criterios de Calidad (20/40 puntos)

#### 2.1 Estructura y Organización (5 puntos)
- [ ] **Portada profesional** con logos ITESM, nombres completos, matrículas
- [ ] **Tabla de contenidos** interactiva con enlaces a secciones
- [ ] **Resumen ejecutivo** bilingüe (español e inglés) 200-250 palabras
- [ ] **Secciones numeradas** jerárquicamente (1, 1.1, 1.1.1)
- [ ] **Flujo narrativo coherente**: Teoría → Implementación → Resultados → Conclusiones
- [ ] **Transiciones suaves** entre secciones con contexto

#### 2.2 Fundamentos Teóricos (5 puntos)
- [ ] **Introducción contextualizada** (2-3 páginas markdown)
  - Problema de detección de estrés vegetal
  - Importancia de agricultura de precisión
  - Gap en métodos tradicionales
  - Nuestra propuesta de valor (MGRG)
- [ ] **Estado del Arte** (3-4 páginas markdown)
  - Region Growing clásico (Adams & Bischof, 1994)
  - Foundation Models en teledetección (Prithvi, SatMAE)
  - Hibridación DL-OBIA (Ma et al., 2024)
  - Referencias académicas integradas (15+ papers 2022-2025)
- [ ] **Fundamentos matemáticos** con LaTeX
  - Ecuaciones de NDVI, NDWI
  - Criterio de homogeneidad clásico: |I(x,y) - I(s)| < T
  - Criterio semántico: cosine_similarity(emb_A, emb_B) > threshold
  - Métricas de evaluación (IoU, mIoU, F1-Score)

#### 2.3 Implementación Técnica Completa (10 puntos)
- [ ] **Setup e Instalación** (celda ejecutable)
  - Instalación de dependencias con pip
  - Verificación de versiones (PyTorch, transformers, etc.)
  - Configuración de credenciales Sentinel Hub
  - Imports organizados por categoría
  - Verificación de GPU disponible


- [ ] **Descarga de Datos Sentinel-2** (código funcional con datos reales)
  - Integración con Sentinel Hub API
  - Descarga de bandas HLS (B02, B03, B04, B08, B8A, B11, B12)
  - Manejo de errores y validación de datos
  - Visualización de imagen RGB
  - Guardado de datos para reutilización
  
- [ ] **Método 1: Region Growing Clásico** (implementación completa con US-004)
  - Cálculo de NDVI con banda B08 (10m nativa)
  - Generación de semillas en grid (spacing=20)
  - Algoritmo BFS con criterio espectral (threshold=0.1)
  - Clasificación de estrés (alto/medio/bajo)
  - Visualización de resultados con colores por estrés
  - Métricas: número de regiones, coherencia espacial, tiempo
  - Código optimizado: ~10-12M pixels/sec
  
- [ ] **Método 2: MGRG (Region Growing Semántico)** (implementación completa con US-007)
  - Carga de modelo Prithvi desde HuggingFace
  - Extracción de embeddings (256D) con normalización L2
  - Generación de semillas en grid (NO K-Means, basado en US-007)
  - Algoritmo BFS con cosine similarity (threshold=0.95 óptimo)
  - Clasificación semántica (6 clases bilingües de US-010)
  - Análisis jerárquico (objeto → estrés)
  - Visualización de resultados con mapa semántico
  - Métricas: IoU, coherencia, precisión de bordes, tiempo

### Nivel 3: Criterios de Excelencia (30/40 puntos)

#### 3.1 Comparativa A/B Profesional (10 puntos - US-008)
- [ ] **Visualización lado a lado** (2x3 grid, 300 DPI)
  - Imagen original RGB
  - NDVI calculado con colormap
  - Resultado RG Clásico con colores por estrés
  - Resultado MGRG con mapa semántico
  - Overlay Classic con píxeles no etiquetados en rojo
  - Overlay MGRG con clasificación bilingüe
  
- [ ] **Métricas cuantitativas** (tabla comparativa profesional)
  - Coherencia espacial (%)
  - Número de regiones
  - Tamaño promedio de región (píxeles y hectáreas)
  - Tiempo de procesamiento (segundos)
  - Memoria utilizada (MB)
  - IoU con ground truth (si disponible)
  
- [ ] **Análisis cualitativo** (markdown explicativo)
  - Fortalezas de cada método
  - Casos de uso recomendados
  - Limitaciones identificadas
  - Interpretación de resultados con datos reales
  
- [ ] **Análisis de 3 zonas reales** (Mexicali, Bajío, Sinaloa)
  - Comparativa cuantitativa por zona
  - Análisis de diferencias geográficas
  - Conclusiones específicas por tipo de agricultura


#### 3.2 Validación Científica (10 puntos - US-009)
- [ ] **Validación con Dynamic World 2024**
  - Descarga de ground truth para 3 zonas
  - Alineación espacial de máscaras
  - Cálculo de métricas estándar:
    - mIoU (Mean Intersection over Union)
    - Weighted mIoU (para clases desbalanceadas)
    - F1-Score / Dice Coefficient
    - Precision y Recall por clase
    - Pixel Accuracy global
  - Confusion matrices (9 figuras: 3 zonas × 2 métodos + comparativa)
  - Análisis de errores (False Positives, False Negatives)
  
- [ ] **Resultados cuantitativos documentados**
  - MGRG +252.8% mejor que Classic RG (mIoU promedio)
  - Tabla comparativa con desviación estándar
  - Gráficos de barras por métrica
  - Interpretación estadística de resultados

#### 3.3 Análisis Avanzado (5 puntos)
- [ ] **Análisis de sensibilidad de parámetros** (basado en US-007, US-008)
  - Threshold NDVI: 0.05 - 0.20 (Classic RG)
  - Threshold similitud coseno: 0.75 - 0.98 (MGRG)
  - Gráficos de métricas vs threshold
  - Recomendaciones de configuración óptima
  - Conclusión: threshold=0.95 óptimo para MGRG
  
- [ ] **Casos de fallo documentados** (3 casos de US-008)
  - **Caso 1: Mexicali - Sombras de nubes**
    - Problema: Sombras causan fragmentación en Classic RG
    - Classic RG: 207 regiones, 72.5% coherencia
    - MGRG: 24 regiones, 94.2% coherencia
    - Conclusión: MGRG más robusto a sombras
  - **Caso 2: Bajío - Vegetación montañosa**
    - Problema: Terreno complejo, sobre-segmentación
    - Classic RG: 775 regiones, fragmentación extrema
    - MGRG: 10 regiones, mejor pero aún desafiante
    - Conclusión: Ambos métodos tienen dificultades
  - **Caso 3: Sinaloa - Riego por goteo**
    - Problema: Parcelas muy pequeñas (10x10m)
    - Classic RG: 934 regiones, detecta micro-patrones
    - MGRG: 19 regiones, suaviza detalles
    - Conclusión: Classic RG mejor para micro-detalles
  
- [ ] **Análisis estadístico**
  - Distribución de NDVI por clase semántica
  - Histogramas y boxplots
  - Correlaciones entre métricas
  - Tests de significancia (si aplica)

#### 3.4 Integración con Pipeline End-to-End (5 puntos - US-011)
- [ ] **Demostración del CLI** (opcional pero recomendado)
  - Ejecución de `scripts/analyze_region.py`
  - Mostrar outputs generados (JSON, GeoTIFF, PNG)
  - Explicar uso en producción
  - Ejemplo de automatización
  
- [ ] **Demostración del API REST** (opcional)
  - Request a `/api/analysis/hierarchical`
  - Polling de status
  - Descarga de resultados
  - Explicar integración con frontend Nuxt 3

### Nivel 4: Cumplimiento de Estándares (Transversal)

#### 4.1 AGENTS.md (100% obligatorio)
- [ ] **Código en inglés** (funciones, variables, clases)
- [ ] **Documentación en español** (celdas markdown narrativas)
- [ ] **Comentarios en inglés** (inline en código)
- [ ] **Type hints** en funciones complejas
- [ ] **Docstrings estilo Google** en funciones reutilizables
- [ ] **Sin emojis** en código Python
- [ ] **Logging profesional** (logger, no print) cuando aplique
- [ ] **Nombres bilingües** en outputs (inglés/español)


#### 4.2 Reproducibilidad
- [ ] **Ejecutable en Google Colab** sin errores
- [ ] **Tiempo de ejecución** <30 minutos (con datos pre-descargados)
- [ ] **Datos de ejemplo** incluidos o descargables
- [ ] **Troubleshooting** documentado
- [ ] **Versiones específicas** de dependencias

---

## 🏗️ Arquitectura Detallada del Notebook

### Estructura Completa (25 secciones principales)

```
SECCIÓN 0: PORTADA Y METADATA
├── Título: "Segmentación Semántica Basada en Region Growing Aplicada a Percepción Remota Agrícola"
├── Información del equipo (nombres completos, matrículas)
├── Institución: ITESM - Maestría en IA Aplicada
├── Materia: Visión Computacional
├── Profesor: Gilberto Ochoa
├── Fecha: Noviembre 2025
└── Logos (ITESM)

SECCIÓN 1: TABLA DE CONTENIDOS
└── Enlaces interactivos a todas las secciones principales

SECCIÓN 2: RESUMEN EJECUTIVO
├── Resumen en español (200 palabras)
├── Abstract en inglés (200 palabras)
├── Palabras clave: Region Growing, MGRG, Prithvi, Sentinel-2, NDVI
└── Contribución principal: MGRG +252.8% mejor que Classic RG

SECCIÓN 3: INTRODUCCIÓN
├── 3.1 Contexto y Motivación
│   - Importancia de agricultura de precisión
│   - Detección de estrés vegetal con teledetección
│   - Limitaciones de métodos tradicionales
├── 3.2 Problema a Resolver
│   - Region Growing clásico sensible a ruido y sombras
│   - Necesidad de métodos robustos y semánticos
├── 3.3 Objetivos del Proyecto
│   - Implementar RG Clásico (baseline)
│   - Desarrollar MGRG con Foundation Models
│   - Comparar cuantitativamente ambos métodos
│   - Validar con ground truth (Dynamic World)
└── 3.4 Estructura del Notebook

SECCIÓN 4: ESTADO DEL ARTE
├── 4.1 Region Growing Clásico
│   - Historia y fundamentos (Adams & Bischof, 1994)
│   - Aplicaciones en agricultura
│   - Limitaciones conocidas
├── 4.2 Foundation Models en Teledetección
│   - Prithvi-EO-1.0-100M (NASA/IBM, 2024)
│   - SatMAE (Cong et al., 2022)
│   - Ventajas: Pre-entrenados, transferibles
├── 4.3 Hibridación DL-OBIA
│   - Marco teórico (Ma et al., 2024)
│   - CRGNet (Ghamisi et al., 2022)
│   - Nuestra propuesta: MGRG
└── 4.4 Referencias Académicas (15+ papers 2022-2025)

SECCIÓN 5: FUNDAMENTOS TEÓRICOS
├── 5.1 Algoritmo Region Growing
│   - Pseudocódigo
│   - Complejidad temporal: O(n)
│   - Criterio de homogeneidad
├── 5.2 Índices Espectrales
│   - NDVI = (NIR - Red) / (NIR + Red)
│   - NDWI para estrés hídrico
│   - Interpretación de valores
├── 5.3 Embeddings Semánticos
│   - Arquitectura Prithvi (Vision Transformer)
│   - Embeddings 256D, L2-normalizados
│   - Captura de contexto semántico
├── 5.4 Cosine Similarity
│   - Fórmula: cos(θ) = (A·B) / (||A|| ||B||)
│   - Interpretación: -1 (opuestos) a +1 (idénticos)
│   - Threshold óptimo: 0.95 (basado en US-007, US-008)
└── 5.5 Métricas de Evaluación
    - IoU (Intersection over Union)
    - mIoU (Mean IoU)
    - Weighted mIoU (para clases desbalanceadas)
    - F1-Score / Dice Coefficient
    - Precision y Recall


SECCIÓN 6: SETUP E INSTALACIÓN
├── 6.1 Instalación de Dependencias
│   ```python
│   !pip install -q sentinelhub==3.10.2
│   !pip install -q torch==2.8.0 torchvision==0.24.0
│   !pip install -q transformers==4.36.0 timm==1.0.22
│   !pip install -q rasterio==1.3.9 scikit-learn==1.4.0
│   !pip install -q matplotlib==3.8.2 seaborn==0.13.0
│   ```
├── 6.2 Imports y Configuración
│   - Imports organizados por categoría (stdlib, third-party, local)
│   - Configuración de matplotlib backend (Agg para Colab)
│   - Configuración de logging
├── 6.3 Verificación de Versiones
│   ```python
│   import torch, transformers, sentinelhub
│   print(f"PyTorch: {torch.__version__}")
│   print(f"CUDA available: {torch.cuda.is_available()}")
│   print(f"Transformers: {transformers.__version__}")
│   ```
├── 6.4 Configuración de Credenciales
│   - Sentinel Hub API keys (usar secrets de Colab)
│   - Verificación de conexión
└── 6.5 Descarga de Código del Proyecto
    ```python
    !git clone https://github.com/equipo24/region-growing.git
    %cd region-growing
    ```

SECCIÓN 7: DESCARGA DE DATOS SENTINEL-2
├── 7.1 Definición de Áreas de Interés
│   - Mexicali: bbox = [-115.35, 32.45, -115.25, 32.55]
│   - Bajío: bbox = [-101.5, 20.8, -101.4, 20.9]
│   - Sinaloa: bbox = [-107.5, 24.7, -107.4, 24.8]
├── 7.2 Descarga de Bandas HLS
│   - B02, B03, B04 (10m): RGB
│   - B08 (10m): NIR Broad para NDVI
│   - B8A, B11, B12 (20m): Para Prithvi
│   - Remuestreo de 20m → 10m
├── 7.3 Validación de Datos
│   - Detección de imágenes vacías/con ceros (US-006)
│   - Verificación de cobertura de nubes
│   - Manejo de errores con mensajes claros
├── 7.4 Visualización de Imagen RGB
│   - Normalización percentil (2%, 98%)
│   - Ajuste gamma para contraste
│   - Guardado en alta resolución (300 DPI)
└── 7.5 Guardado de Datos
    - Formato NPZ para reutilización
    - Metadata incluido (bbox, fecha, resolución)

SECCIÓN 8: MÉTODO 1 - REGION GROWING CLÁSICO
├── 8.1 Cálculo de NDVI
│   ```python
│   from src.features.ndvi_calculator import calculate_ndvi
│   ndvi_result = calculate_ndvi(red_band, nir_band)
│   ndvi = ndvi_result['ndvi']
│   ```
├── 8.2 Generación de Semillas en Grid
│   ```python
│   from src.algorithms.classic_region_growing import ClassicRegionGrowing
│   algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=50)
│   seeds = algorithm.generate_grid_seeds(ndvi.shape, spacing=20)
│   ```
├── 8.3 Implementación del Algoritmo BFS
│   - Criterio: |NDVI_A - NDVI_B| < 0.1
│   - 4-conectividad
│   - Filtrado de regiones <50 píxeles
├── 8.4 Clasificación de Estrés
│   - Alto: NDVI < 0.3 (rojo)
│   - Medio: 0.3 ≤ NDVI < 0.5 (amarillo)
│   - Bajo: NDVI ≥ 0.5 (verde)
├── 8.5 Visualización de Resultados
│   - Mapa de segmentación con colores por estrés
│   - Overlay sobre RGB original
│   - Histograma de NDVI por región
└── 8.6 Métricas de Desempeño
    - Número de regiones: ~200-900 (varía por zona)
    - Coherencia espacial: 70-85%
    - Tiempo de procesamiento: ~3-12 segundos
    - Throughput: ~10-12M pixels/sec


SECCIÓN 9: MÉTODO 2 - MGRG (REGION GROWING SEMÁNTICO)
├── 9.1 Carga del Modelo Prithvi
│   ```python
│   from src.models.prithvi_loader import load_prithvi_model
│   encoder = load_prithvi_model(use_simple_model=False)  # Modelo real
│   ```
├── 9.2 Preparación de Imagen HLS
│   ```python
│   from src.features.hls_processor import prepare_hls_image
│   hls_image = prepare_hls_image(bands_10m, bands_20m)
│   # Shape: (6, H, W) - B02, B03, B04, B8A, B11, B12
│   ```
├── 9.3 Extracción de Embeddings
│   ```python
│   from src.features.hls_processor import extract_embeddings
│   embeddings = extract_embeddings(hls_image, encoder)
│   # Shape: (H, W, 256), L2-normalized
│   ```
├── 9.4 Generación de Semillas en Grid
│   - NO usar K-Means (conclusión de US-007)
│   - Grid spacing=20 (igual que Classic RG)
│   - Razón: Grid 19x más rápido, mejor cobertura
├── 9.5 Implementación del Algoritmo BFS Semántico
│   ```python
│   from src.algorithms.semantic_region_growing import SemanticRegionGrowing
│   mgrg = SemanticRegionGrowing(
│       threshold=0.95,  # Óptimo según US-008
│       min_region_size=50,
│       use_smart_seeds=False  # Grid, no K-Means
│   )
│   labeled, num_regions, regions_info = mgrg.segment(embeddings)
│   ```
│   - Criterio: cosine_similarity(emb_A, emb_B) > 0.95
│   - 4-conectividad
│   - Filtrado de regiones <50 píxeles
├── 9.6 Clasificación Semántica (6 Clases Bilingües)
│   ```python
│   from src.classification.zero_shot_classifier import SemanticClassifier
│   classifier = SemanticClassifier(embeddings, ndvi)
│   classifications = classifier.classify_all_regions(labeled)
│   ```
│   - Water (Agua): NDVI < 0.1, std < 0.05
│   - Urban (Urbano): NDVI < 0.1, std > 0.05
│   - Bare Soil (Suelo Desnudo): 0.1 ≤ NDVI < 0.3
│   - Stressed Crop (Cultivo Estresado): 0.3 ≤ NDVI < 0.55
│   - Vigorous Crop (Cultivo Vigoroso): NDVI ≥ 0.55, std < 0.1
│   - Grass/Shrub (Pasto/Arbustos): NDVI ≥ 0.55, std ≥ 0.1
├── 9.7 Análisis Jerárquico (Objeto → Estrés)
│   - Primero: Identificar objeto semántico
│   - Luego: Analizar estrés interno (solo cultivos)
│   - Evita confusión entre "cultivo estresado" y "suelo desnudo"
├── 9.8 Visualización de Resultados
│   - Mapa semántico con colores por clase
│   - Overlay sobre RGB original
│   - Leyenda bilingüe
│   - Estadísticas por clase (área, NDVI medio)
└── 9.9 Métricas de Desempeño
    - Número de regiones: ~10-200 (90-99% menos que Classic RG)
    - Coherencia espacial: 95-99%
    - Tiempo de procesamiento: ~15-60 segundos (incluye Prithvi)
    - Mejora en mIoU: +252.8% vs Classic RG

SECCIÓN 10: COMPARATIVA A/B CUANTITATIVA
├── 10.1 Cálculo de Métricas
│   ```python
│   from src.utils.comparison_metrics import compare_segmentations
│   comparison = compare_segmentations(
│       classic_seg, mgrg_seg,
│       classic_time, mgrg_time
│   )
│   ```
│   - Coherencia espacial (%)
│   - Número de regiones
│   - Tamaño promedio de región (píxeles y hectáreas)
│   - Tiempo de procesamiento
│   - Memoria utilizada
├── 10.2 Tabla Comparativa
│   | Métrica | Classic RG | MGRG | Mejora |
│   |---------|------------|------|--------|
│   | Regiones | 207-934 | 10-24 | -90 a -99% |
│   | Coherencia | 70-85% | 95-99% | +15-25% |
│   | mIoU | 0.0382 | 0.1349 | +252.8% |
│   | Tiempo | 3-12s | 15-60s | -5x |
├── 10.3 Gráficos Comparativos
│   - Barras: Regiones por método y zona
│   - Barras: Coherencia por método y zona
│   - Scatter: Regiones vs Coherencia
│   - Boxplot: Distribución de tamaños de región
└── 10.4 Interpretación de Resultados
    - MGRG genera regiones más coherentes y grandes
    - Classic RG sobre-segmenta (fragmentación)
    - Trade-off: Tiempo vs Calidad
    - Recomendación: MGRG para análisis regional, Classic RG para micro-detalles


SECCIÓN 11: COMPARATIVA A/B VISUAL
├── 11.1 Visualización Lado a Lado (2x3 grid, 300 DPI)
│   ```python
│   from src.visualization.ab_comparison import create_side_by_side_plot
│   fig, img_array = create_side_by_side_plot(
│       rgb_image, classic_seg, mgrg_seg, metrics,
│       title="Comparativa A/B: Mexicali",
│       save_path="mexicali_comparison.png",
│       dpi=300
│   )
│   ```
│   - Fila 1: RGB original, Classic RG, MGRG
│   - Fila 2: Overlay Classic, Overlay MGRG, Tabla de métricas
├── 11.2 Overlays con Transparencia
│   - Alpha=0.5 para ver imagen base
│   - Píxeles no etiquetados en rojo
│   - Colores consistentes entre visualizaciones
├── 11.3 Exportación Multi-Formato
│   ```python
│   from src.visualization.ab_comparison import export_high_resolution
│   paths = export_high_resolution(
│       fig, "mexicali_comparison",
│       dpi=300,
│       formats=["png", "pdf", "svg"]
│   )
│   ```
└── 11.4 Análisis Visual por Zona
    - Mexicali: MGRG separa urbano de agrícola
    - Bajío: Ambos métodos luchan con terreno complejo
    - Sinaloa: Classic RG detecta micro-parcelas mejor

SECCIÓN 12: VALIDACIÓN CON DYNAMIC WORLD
├── 12.1 Descarga de Ground Truth
│   ```python
│   from src.utils.dynamic_world_downloader import load_dynamic_world
│   dw_mask = load_dynamic_world(zone_name, bbox, date)
│   ```
│   - Dynamic World 2024 (10m resolución)
│   - 9 clases de cobertura terrestre
│   - Mapeo a nuestras 6 clases
├── 12.2 Alineación Espacial
│   ```python
│   from src.utils.validation_metrics import align_ground_truth
│   dw_aligned = align_ground_truth(dw_mask, segmentation.shape)
│   ```
│   - Redimensionamiento con interpolación nearest
│   - Verificación de shapes coincidentes
├── 12.3 Cálculo de Métricas Estándar
│   ```python
│   from src.utils.validation_metrics import (
│       calculate_miou, calculate_weighted_miou,
│       calculate_f1_score, calculate_precision_recall
│   )
│   miou = calculate_miou(predicted, ground_truth, num_classes=6)
│   weighted_miou = calculate_weighted_miou(predicted, ground_truth, num_classes=6)
│   f1 = calculate_f1_score(predicted, ground_truth, class_id)
│   precision, recall = calculate_precision_recall(predicted, ground_truth, class_id)
│   ```
├── 12.4 Confusion Matrices (9 figuras)
│   - 3 zonas × 2 métodos = 6 matrices individuales
│   - 3 matrices comparativas (Classic vs MGRG por zona)
│   - Visualización con seaborn heatmap
│   - Guardado en 300 DPI
├── 12.5 Análisis de Errores
│   - False Positives: Píxeles clasificados incorrectamente
│   - False Negatives: Píxeles no detectados
│   - Patrones de error por clase
│   - Comparación Classic RG vs MGRG
└── 12.6 Resultados Cuantitativos
    | Zona | Método | mIoU | Weighted mIoU | F1 | Precision | Recall |
    |------|--------|------|---------------|----|-----------| -------|
    | Mexicali | Classic | 0.1123 | 0.1084 | 0.1302 | 0.1283 | 0.0610 |
    | Mexicali | MGRG | 0.1224 | 0.1067 | 0.1446 | 0.1956 | 0.2564 |
    | Bajío | Classic | 0.0020 | 0.0018 | 0.0040 | 0.1283 | 0.0610 |
    | Bajío | MGRG | 0.1544 | 0.3831 | 0.1994 | 0.1956 | 0.2564 |
    | Sinaloa | Classic | 0.0004 | 0.0002 | 0.0004 | 0.1283 | 0.0610 |
    | Sinaloa | MGRG | 0.1278 | 0.1442 | 0.1511 | 0.1956 | 0.2564 |
    | **Promedio** | **Classic** | **0.0382** | **0.0368** | **0.0448** | **0.1283** | **0.0610** |
    | **Promedio** | **MGRG** | **0.1349** | **0.2113** | **0.1650** | **0.1956** | **0.2564** |
    | **Mejora** | | **+252.8%** | **+474.2%** | **+268.2%** | **+52.4%** | **+320.3%** |


SECCIÓN 13: ANÁLISIS DE SENSIBILIDAD
├── 13.1 Threshold NDVI (Classic RG)
│   - Rango: 0.05 - 0.20
│   - Gráfico: Regiones vs Threshold
│   - Gráfico: Coherencia vs Threshold
│   - Conclusión: 0.10 óptimo (balance)
├── 13.2 Threshold Similitud Coseno (MGRG)
│   - Rango: 0.75 - 0.98
│   - Resultados experimentales de US-007, US-008:
│     - 0.70: 1 región (bajo-segmentación)
│     - 0.85: 33 regiones (moderado)
│     - 0.95: 156 regiones (óptimo)
│     - 0.98: 300+ regiones (sobre-segmentación)
│   - Gráfico: Regiones vs Threshold
│   - Gráfico: Coherencia vs Threshold
│   - Conclusión: **0.95 óptimo** (separa clases sin fragmentar)
├── 13.3 Min Region Size
│   - Rango: 25 - 100 píxeles
│   - Impacto en número de regiones
│   - Impacto en coherencia
│   - Conclusión: 50 píxeles óptimo (filtra ruido)
└── 13.4 Recomendaciones de Configuración
    - **Classic RG**: threshold=0.1, min_size=50
    - **MGRG**: threshold=0.95, min_size=50, grid seeds
    - Ajustar según tipo de agricultura y resolución

SECCIÓN 14: CASOS DE FALLO DOCUMENTADOS
├── 14.1 Caso 1: Mexicali - Sombras de Nubes
│   ```python
│   from src.visualization.ab_comparison import generate_failure_case_analysis
│   path = generate_failure_case_analysis(
│       "Mexicali", rgb, classic_seg, mgrg_seg, ndvi,
│       "Sombras de nubes causan fragmentación en Classic RG",
│       "img/results/failure_cases/"
│   )
│   ```
│   - **Problema**: Sombras causan discontinuidad espectral
│   - **Classic RG**: 207 regiones, 72.5% coherencia
│   - **MGRG**: 24 regiones, 94.2% coherencia
│   - **Conclusión**: MGRG más robusto a variaciones de iluminación
├── 14.2 Caso 2: Bajío - Vegetación Montañosa
│   - **Problema**: Terreno complejo, gradientes suaves
│   - **Classic RG**: 775 regiones, fragmentación extrema
│   - **MGRG**: 10 regiones, mejor pero aún desafiante
│   - **Conclusión**: Ambos métodos tienen dificultades con terreno heterogéneo
├── 14.3 Caso 3: Sinaloa - Riego por Goteo
│   - **Problema**: Parcelas muy pequeñas (10x10m)
│   - **Classic RG**: 934 regiones, detecta micro-patrones
│   - **MGRG**: 19 regiones, suaviza detalles
│   - **Conclusión**: Classic RG mejor para agricultura de precisión micro-escala
└── 14.4 Lecciones Aprendidas
    - No existe método universal óptimo
    - Selección depende de escala y objetivo
    - MGRG: Análisis regional, mapeo de cobertura
    - Classic RG: Agricultura de precisión, micro-detalles

SECCIÓN 15: ANÁLISIS ESTADÍSTICO
├── 15.1 Distribución de NDVI por Clase
│   - Histogramas por clase semántica
│   - Boxplots comparativos
│   - Estadísticas descriptivas (media, std, min, max)
├── 15.2 Correlaciones entre Métricas
│   - Heatmap de correlación
│   - Regiones vs Coherencia: r = -0.85 (negativa fuerte)
│   - Tiempo vs Regiones: r = 0.92 (positiva fuerte)
│   - mIoU vs Coherencia: r = 0.78 (positiva moderada)
├── 15.3 Análisis por Zona
│   - Comparación de distribuciones NDVI
│   - Mexicali: Bimodal (urbano + agrícola)
│   - Bajío: Multimodal (diversidad de cultivos)
│   - Sinaloa: Unimodal (agricultura intensiva)
└── 15.4 Tests de Significancia (opcional)
    - t-test para diferencias entre métodos
    - Wilcoxon para datos no paramétricos
    - Interpretación de p-values

SECCIÓN 16: INTEGRACIÓN CON PIPELINE END-TO-END
├── 16.1 Demostración del CLI
│   ```bash
│   !python scripts/analyze_region.py \
│     --bbox "32.45,-115.35,32.55,-115.25" \
│     --date "2024-01-15" \
│     --output "output/mexicali" \
│     --threshold 0.95 \
│     --formats json,png
│   ```
│   - Ejecución síncrona con barra de progreso
│   - Outputs: JSON, GeoTIFF, PNG
│   - Tiempo total: ~60 segundos
├── 16.2 Análisis de Outputs
│   - JSON: Metadata, clasificaciones, estrés
│   - GeoTIFF: 2 capas (segmentación, clasificación)
│   - PNG: Visualización 300 DPI
├── 16.3 Uso en Producción
│   - Automatización con cron jobs
│   - Integración con sistemas GIS
│   - Monitoreo continuo de cultivos
└── 16.4 API REST (opcional)
    ```python
    import requests
    response = requests.post(
        "http://localhost:8070/api/analysis/hierarchical",
        json={"bbox": [...], "date_from": "2024-01-15"}
    )
    ```


SECCIÓN 17: DISCUSIÓN
├── 17.1 Fortalezas del Método Híbrido (MGRG)
│   - **Robustez**: +252.8% mejor mIoU que Classic RG
│   - **Coherencia espacial**: 95-99% vs 70-85%
│   - **Reducción de regiones**: 90-99% menos fragmentación
│   - **Separación semántica**: Distingue urbano, agrícola, agua
│   - **Weighted mIoU alto**: 0.2113 (excelente para método no supervisado)
├── 17.2 Limitaciones Identificadas
│   - **Costo computacional**: 5x más lento que Classic RG
│   - **Dependencia de Prithvi**: Requiere GPU, modelo grande
│   - **Suavizado excesivo**: Pierde micro-detalles en parcelas pequeñas
│   - **Threshold sensible**: Requiere calibración por zona
│   - **Ground truth imperfecto**: Dynamic World ~80% accuracy
├── 17.3 Comparación con SOTA
│   - **Métodos supervisados**: 55-90% mIoU (con entrenamiento)
│   - **Métodos no supervisados**: 15-40% mIoU (literatura)
│   - **Nuestro MGRG**: 13.5% mIoU (razonable para no supervisado)
│   - **Mejora relativa**: +252.8% es comparable con literatura
├── 17.4 Aplicabilidad Práctica
│   - **Agricultura extensiva**: MGRG recomendado
│   - **Agricultura de precisión**: Classic RG recomendado
│   - **Mapeo de cobertura**: MGRG recomendado
│   - **Monitoreo temporal**: Ambos métodos complementarios
└── 17.5 Trabajo Futuro
    - Fine-tuning de Prithvi con datos de México
    - Análisis temporal (series de tiempo)
    - Integración con otros sensores (Landsat, PlanetScope)
    - Optimización de performance (GPU, paralelización)
    - Validación con ground truth de campo

SECCIÓN 18: CONCLUSIONES
├── 18.1 Resumen de Hallazgos Principales
│   1. **MGRG supera a Classic RG** en todas las métricas cuantitativas
│   2. **Threshold 0.95 óptimo** para MGRG (basado en experimentos)
│   3. **Grid seeds superior a K-Means** (19x más rápido, mejor cobertura)
│   4. **Validación con Dynamic World** confirma superioridad de MGRG
│   5. **Clasificación bilingüe** mejora interpretabilidad
├── 18.2 Contribuciones del Proyecto
│   - Sistema híbrido funcional (RG Clásico + MGRG)
│   - Validación cuantitativa con ground truth real
│   - Análisis de 3 zonas agrícolas de México
│   - Código abierto y reproducible
│   - Documentación exhaustiva
├── 18.3 Impacto Esperado
│   - Herramienta para agricultura de precisión
│   - Base para investigación futura
│   - Referencia para hibridación DL-OBIA
│   - Aplicación en seguros paramétricos
└── 18.4 Recomendaciones Finales
    - Usar MGRG para análisis regional (>100 ha)
    - Usar Classic RG para micro-parcelas (<10 ha)
    - Calibrar thresholds por zona climática
    - Validar con ground truth local cuando sea posible

SECCIÓN 19: TRABAJO FUTURO
├── 19.1 Mejoras Técnicas
│   - MiniBatchKMeans para semillas (10x más rápido)
│   - Threshold adaptativo por región
│   - Paralelización GPU con CUDA
│   - Clustering jerárquico multi-escala
├── 19.2 Extensiones Propuestas
│   - Análisis temporal (cambio de cobertura)
│   - Integración con Landsat 8/9
│   - Detección de anomalías
│   - Predicción de estrés futuro
└── 19.3 Aplicaciones Potenciales
    - Monitoreo de sequías
    - Gestión de recursos hídricos
    - Seguros agrícolas paramétricos
    - Planificación de cultivos

SECCIÓN 20: REFERENCIAS
└── Bibliografía completa (15+ papers en formato APA 7)
    1. Jakubik et al. (2024). Foundation models for generalist geospatial AI
    2. Ghamisi et al. (2022). Consistency-regularized region-growing network
    3. Ma et al. (2024). Deep learning meets object-based image analysis
    4. Brown et al. (2022). Dynamic World
    5. Cong et al. (2022). SatMAE
    6. Adams & Bischof (1994). Seeded region growing
    7. Tucker (1979). NDVI original paper
    8. Drusch et al. (2012). Sentinel-2 mission
    9. Claverie et al. (2018). HLS product
    10. Gao (1996). NDWI
    [... 5+ referencias adicionales de 2022-2025]

SECCIÓN 21: ROLES DEL EQUIPO
├── Tabla de Contribuciones
│   | Nombre | Matrícula | Rol | Contribuciones | Horas |
│   |--------|-----------|-----|----------------|-------|
│   | Carlos Aaron Bocanegra Buitron | A01796345 | Tech Lead & Backend | FastAPI, RG Clásico, Pipeline, Integración | 45h |
│   | Arthur Jafed Zizumbo Velasco | A01796363 | ML Engineer | Prithvi, MGRG, Clasificación, Validación | 48h |
│   | Luis Santiago Vázquez Mancilla | A01796029 | Full Stack Developer | Nuxt 3, Visualizaciones, Comparativas A/B | 40h |
│   | Edgar Oviedo Navarro | A01795260 | Product Owner & Documentation | Artículo, Video, Documentación, Teoría | 42h |
├── Foto del Equipo (opcional)
└── Agradecimientos
    - Profesor Gilberto Ochoa
    - NASA/IBM por modelo Prithvi
    - ESA por Sentinel-2
    - Google por Dynamic World

SECCIÓN 22: APÉNDICES
├── A. Código Completo de Funciones Auxiliares
│   - Funciones de preprocesamiento
│   - Funciones de visualización
│   - Funciones de métricas
├── B. Configuración de Entorno
│   - Requirements completos
│   - Configuración de Colab
│   - Troubleshooting común
└── C. Datos Suplementarios
    - Links a datasets
    - Links a código fuente
    - Links a resultados completos
```

---

## 📋 Plan de Implementación Detallado

### Fase 1: Preparación y Setup (3-4 horas)

#### Tarea 1.1: Análisis y Migración de Plantilla
**Responsable:** Edgar Oviedo + Carlos Bocanegra
**Duración:** 1 hora

**Actividades:**
- Revisar plantilla actual `notebooks/final/region_growing_equipo24.ipynb`
- Identificar secciones teóricas reutilizables
- Crear nuevo notebook en Google Colab
- Copiar estructura teórica (Introducción, Estado del Arte, Fundamentos)
- Actualizar referencias con papers 2022-2025

**Entregable:** Notebook con estructura teórica completa

#### Tarea 1.2: Configuración del Entorno Colab
**Responsable:** Arthur Zizumbo
**Duración:** 1.5 horas

**Actividades:**
- Crear notebook en Google Colab con GPU T4
- Configurar runtime (GPU, High-RAM si disponible)
- Instalar dependencias completas
- Verificar acceso a Sentinel Hub API
- Probar carga de Prithvi desde HuggingFace
- Clonar repositorio del proyecto

**Código de verificación:**
```python
# Verify GPU
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Verify Sentinel Hub
from sentinelhub import SHConfig
config = SHConfig()
print(f"SH configured: {config.sh_client_id is not None}")

# Verify Prithvi
from transformers import AutoModel
model = AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-EO-1.0-100M")
print(f"Prithvi loaded: {model is not None}")

# Clone repo
!git clone https://github.com/equipo24/region-growing.git
%cd region-growing
```

**Entregable:** Notebook con setup funcional y verificado


#### Tarea 1.3: Preparación de Datos
**Responsable:** Luis Vázquez
**Duración:** 30 min

**Actividades:**
- Subir datos pre-descargados a Google Drive
- Crear links compartidos
- Agregar celdas de descarga alternativa en notebook
- Verificar integridad de datos

**Datos a preparar:**
- Imágenes HLS de 3 zonas (Mexicali, Bajío, Sinaloa)
- Embeddings pre-calculados (opcional, para acelerar)
- Segmentaciones pre-generadas (backup)
- Dynamic World masks

**Entregable:** Datos accesibles desde Colab

### Fase 2: Implementación de Métodos (6-8 horas)

#### Tarea 2.1: Region Growing Clásico
**Responsable:** Carlos Bocanegra
**Duración:** 2.5 horas

**Actividades:**
- Implementar descarga de Sentinel-2 (reutilizar US-003)
- Calcular NDVI con banda B08
- Implementar RG Clásico (reutilizar US-004)
- Generar visualizaciones
- Calcular métricas
- Aplicar a 3 zonas

**Código base:**
```python
from src.algorithms.classic_region_growing import ClassicRegionGrowing
from src.features.ndvi_calculator import calculate_ndvi

# Calculate NDVI
ndvi_result = calculate_ndvi(red_band, nir_band)

# Run Classic RG
algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=50)
labeled, num_regions, regions_info = algorithm.segment(ndvi_result['ndvi'])

# Classify by stress
classified = algorithm.classify_by_stress(regions_info)
```

**Entregable:** Sección 8 completa con resultados de 3 zonas

#### Tarea 2.2: MGRG Semántico
**Responsable:** Arthur Zizumbo
**Duración:** 3.5 horas

**Actividades:**
- Cargar modelo Prithvi (reutilizar US-005, US-006)
- Extraer embeddings de 3 zonas
- Implementar MGRG (reutilizar US-007)
- Aplicar clasificación semántica (reutilizar US-010)
- Generar visualizaciones
- Calcular métricas
- Aplicar a 3 zonas

**Código base:**
```python
from src.models.prithvi_loader import load_prithvi_model
from src.features.hls_processor import extract_embeddings
from src.algorithms.semantic_region_growing import SemanticRegionGrowing
from src.classification.zero_shot_classifier import SemanticClassifier

# Load Prithvi
encoder = load_prithvi_model(use_simple_model=False)

# Extract embeddings
embeddings = extract_embeddings(hls_image, encoder)

# Run MGRG
mgrg = SemanticRegionGrowing(threshold=0.95, min_region_size=50)
labeled, num_regions, regions_info = mgrg.segment(embeddings)

# Classify semantically
classifier = SemanticClassifier(embeddings, ndvi)
classifications = classifier.classify_all_regions(labeled)
```

**Entregable:** Sección 9 completa con resultados de 3 zonas


#### Tarea 2.3: Comparativa A/B
**Responsable:** Luis Vázquez
**Duración:** 2 horas

**Actividades:**
- Implementar visualizaciones lado a lado (reutilizar US-008)
- Calcular métricas comparativas
- Generar tablas y gráficos
- Exportar en alta resolución (300 DPI)
- Aplicar a 3 zonas

**Código base:**
```python
from src.utils.comparison_metrics import compare_segmentations
from src.visualization.ab_comparison import create_side_by_side_plot

# Compare methods
comparison = compare_segmentations(
    classic_seg, mgrg_seg,
    classic_time, mgrg_time
)

# Visualize
fig, img = create_side_by_side_plot(
    rgb_image, classic_seg, mgrg_seg, comparison,
    title=f"Comparativa A/B: {zone_name}",
    save_path=f"{zone_name}_comparison.png",
    dpi=300
)
```

**Entregable:** Secciones 10-11 completas con comparativas de 3 zonas

### Fase 3: Validación y Análisis (4-5 horas)

#### Tarea 3.1: Validación con Dynamic World
**Responsable:** Arthur Zizumbo
**Duración:** 2.5 horas

**Actividades:**
- Cargar Dynamic World masks (reutilizar US-009)
- Alinear espacialmente
- Calcular métricas estándar (IoU, mIoU, F1, Precision, Recall)
- Generar confusion matrices
- Analizar errores
- Aplicar a 3 zonas

**Código base:**
```python
from src.utils.validation_metrics import (
    calculate_miou, calculate_weighted_miou,
    generate_confusion_matrix, plot_confusion_matrix
)

# Calculate metrics
miou = calculate_miou(predicted, ground_truth, num_classes=6)
weighted_miou = calculate_weighted_miou(predicted, ground_truth, num_classes=6)

# Generate confusion matrix
cm = generate_confusion_matrix(predicted, ground_truth, num_classes=6)
plot_confusion_matrix(cm, class_names, save_path=f"{zone}_cm.png")
```

**Entregable:** Sección 12 completa con validación de 3 zonas

#### Tarea 3.2: Análisis de Sensibilidad y Casos de Fallo
**Responsable:** Carlos Bocanegra + Luis Vázquez
**Duración:** 2 horas

**Actividades:**
- Análisis de sensibilidad de thresholds
- Documentar 3 casos de fallo
- Generar visualizaciones de análisis
- Análisis estadístico (distribuciones, correlaciones)

**Entregable:** Secciones 13-15 completas

### Fase 4: Integración y Conclusiones (3-4 horas)

#### Tarea 4.1: Integración con Pipeline
**Responsable:** Carlos Bocanegra
**Duración:** 1 hora

**Actividades:**
- Demostración del CLI (US-011)
- Análisis de outputs
- Explicar uso en producción
- Demostración opcional del API REST

**Entregable:** Sección 16 completa

#### Tarea 4.2: Discusión y Conclusiones
**Responsable:** Edgar Oviedo
**Duración:** 2 horas

**Actividades:**
- Redactar Discusión (Sección 17)
- Redactar Conclusiones (Sección 18)
- Redactar Trabajo Futuro (Sección 19)
- Compilar Referencias (Sección 20)
- Crear tabla de roles (Sección 21)

**Entregable:** Secciones 17-21 completas

### Fase 5: Revisión y Pulido (2-3 horas)

#### Tarea 5.1: Revisión Técnica
**Responsable:** Todos
**Duración:** 1.5 horas

**Actividades:**
- Ejecutar notebook completo de principio a fin
- Verificar que no hay errores
- Verificar tiempos de ejecución
- Optimizar celdas lentas
- Verificar visualizaciones

**Checklist:**
- [ ] Todas las celdas ejecutan sin errores
- [ ] Visualizaciones se muestran correctamente
- [ ] Tablas están formateadas
- [ ] Ecuaciones LaTeX renderizan bien
- [ ] Tiempo total <30 min

#### Tarea 5.2: Revisión de Contenido
**Responsable:** Edgar Oviedo
**Duración:** 1 hora

**Actividades:**
- Revisar ortografía y gramática
- Verificar coherencia narrativa
- Verificar enlaces y referencias
- Verificar cumplimiento AGENTS.md
- Verificar formato bilingüe

**Checklist:**
- [ ] Sin typos
- [ ] Flujo narrativo coherente
- [ ] Referencias completas (15+)
- [ ] Código en inglés, docs en español
- [ ] Sin emojis en código Python
- [ ] Nombres bilingües en outputs

#### Tarea 5.3: Exportación Final
**Responsable:** Luis Vázquez
**Duración:** 30 min

**Actividades:**
- Exportar notebook a .ipynb
- Verificar que abre correctamente
- Crear README con instrucciones
- Preparar ZIP con datos de ejemplo
- Subir a repositorio

**Entregable:** Notebook final listo para entrega

---

## 📊 Estimación de Recursos

### Tiempo de Desarrollo por Fase

| Fase | Duración | Responsable Principal | Horas Totales |
|------|----------|----------------------|---------------|
| Fase 1: Preparación | 3-4h | Edgar + Arthur | 3-4h |
| Fase 2: Métodos | 6-8h | Carlos + Arthur | 8h |
| Fase 3: Validación | 4-5h | Arthur + Luis | 4.5h |
| Fase 4: Integración | 3-4h | Carlos + Edgar | 3.5h |
| Fase 5: Revisión | 2-3h | Todos | 2.5h |
| **TOTAL** | **18-24h** | **Equipo completo** | **21.5h** |

### Distribución por Rol

| Rol | Responsable | Horas Estimadas | Tareas Principales |
|-----|-------------|-----------------|-------------------|
| Tech Lead | Carlos Bocanegra | 6-7h | RG Clásico, Pipeline, Integración |
| ML Engineer | Arthur Zizumbo | 7-8h | MGRG, Prithvi, Validación |
| Visualization | Luis Vázquez | 4-5h | Comparativas A/B, Gráficos |
| Documentation | Edgar Oviedo | 5-6h | Teoría, Discusión, Conclusiones |

### Recursos Computacionales

**Google Colab:**
- Runtime: GPU T4 (gratuito) o A100 (Colab Pro $10/mes)
- RAM: 12-16 GB (High-RAM si disponible)
- Disco: 100 GB

**Tiempo de Ejecución Estimado:**
- Setup e instalación: 3-5 min
- Descarga Sentinel-2 (si no pre-descargado): 10-15 min
- Extracción embeddings Prithvi (3 zonas): 15-20 min
- RG Clásico (3 zonas): 1-2 min
- MGRG (3 zonas): 5-10 min
- Clasificación (3 zonas): 1-2 min
- Validación (3 zonas): 2-3 min
- Visualizaciones: 3-5 min
- **TOTAL:** 40-60 min (con datos pre-descargados: 20-30 min)

**Optimizaciones:**
- Pre-descargar datos y subirlos a Google Drive
- Cachear embeddings de Prithvi
- Usar imágenes de menor resolución para demos rápidas (opcional)
- Ejecutar en GPU A100 si disponible (3-5x más rápido)

---

## 🎯 Métricas de Éxito

### Criterios Técnicos

| Métrica | Target | Verificación |
|---------|--------|--------------|
| Ejecutable sin errores | 100% | Ejecutar 3 veces en Colab limpio |
| Tiempo de ejecución | <30 min | Cronometrar con datos pre-descargados |
| Cobertura de secciones | 22/22 | Checklist de estructura |
| Calidad de visualizaciones | 300 DPI | Verificar resolución de imágenes |
| Referencias académicas | 15+ | Contar papers citados (2022-2025) |
| Comentarios en código | >80% | Revisar funciones complejas |
| Cumplimiento AGENTS.md | 100% | Checklist completo |

### Criterios de Calidad

| Aspecto | Target | Verificación |
|---------|--------|--------------|
| Claridad narrativa | Excelente | Revisión por pares |
| Coherencia técnica | 100% | Validación de algoritmos |
| Ortografía y gramática | 0 errores | Corrector + revisión manual |
| Formato profesional | Consistente | Revisión de estilo |
| Reproducibilidad | 100% | Test en 2+ máquinas |

### Criterios de Rúbrica (40% del proyecto)

| Criterio | Peso | Target | Estrategia |
|----------|------|--------|-----------|
| Código limpio y documentado | 10% | 10/10 | Comentarios, docstrings, type hints |
| Markdown explicativo | 10% | 10/10 | Narrativa clara entre celdas |
| Ambos métodos implementados | 10% | 10/10 | RG Clásico + MGRG funcionales |
| Comparativa A/B | 5% | 5/5 | Visualización profesional 300 DPI |
| Ejecutable de principio a fin | 5% | 5/5 | Testing exhaustivo |
| **TOTAL** | **40%** | **40/40** | **Excelencia en todos los aspectos** |

---

## 🚨 Riesgos y Mitigaciones

### Riesgos Técnicos

#### Riesgo 1: Prithvi no carga en Colab gratuito
**Probabilidad:** Media
**Impacto:** Alto
**Mitigación:**
- Usar Colab Pro ($10/mes) con GPU A100
- Pre-calcular embeddings y subirlos a Google Drive
- Implementar fallback con embeddings pre-calculados
- Documentar proceso de cálculo offline

#### Riesgo 2: Sentinel Hub API falla o excede límites
**Probabilidad:** Baja
**Impacto:** Alto
**Mitigación:**
- Pre-descargar imágenes y subirlas a Google Drive
- Incluir datos de ejemplo en el notebook
- Documentar proceso de descarga alternativo
- Usar datos de backup de US-006

#### Riesgo 3: Tiempo de ejecución >30 min
**Probabilidad:** Media
**Impacto:** Medio
**Mitigación:**
- Usar imágenes de menor resolución (512x512 en lugar de 1024x1024)
- Cachear resultados intermedios
- Optimizar algoritmos (ya hecho en US-004, US-007)
- Usar GPU A100 si disponible

#### Riesgo 4: Errores de dependencias
**Probabilidad:** Baja
**Impacto:** Medio
**Mitigación:**
- Especificar versiones exactas de paquetes
- Probar en Colab limpio antes de entregar
- Incluir sección de troubleshooting
- Documentar soluciones a errores comunes

### Riesgos de Proyecto

#### Riesgo 5: Falta de tiempo para completar todas las secciones
**Probabilidad:** Media
**Impacto:** Alto
**Mitigación:**
- Priorizar secciones críticas (métodos, comparativa, validación)
- Trabajar en paralelo (división de tareas clara)
- Tener plan B con secciones mínimas
- Comenzar 3 días antes de la entrega

#### Riesgo 6: Calidad de visualizaciones no profesional
**Probabilidad:** Baja
**Impacto:** Medio
**Mitigación:**
- Usar templates de matplotlib profesionales (ya implementado en US-008)
- Revisar ejemplos de papers académicos
- Iterar en diseño con feedback del equipo
- Exportar en 300 DPI siempre

---

## ✅ Checklist de Entrega Final

### Pre-Entrega (1 día antes)

- [ ] Notebook ejecuta sin errores (3 pruebas en Colab limpio)
- [ ] Todas las secciones completas (22/22)
- [ ] Visualizaciones en alta resolución (300 DPI)
- [ ] Referencias formateadas en APA 7 (15+)
- [ ] Tabla de roles del equipo incluida
- [ ] Código comentado y documentado
- [ ] Markdown revisado (ortografía y gramática)
- [ ] Tiempo de ejecución <30 min
- [ ] Probado en Colab limpio
- [ ] Backup en Google Drive

### Día de Entrega

- [ ] Descargar notebook (.ipynb)
- [ ] Verificar que abre correctamente
- [ ] Incluir en ZIP con otros entregables (artículo, video, presentación)
- [ ] Subir a plataforma antes de deadline
- [ ] Confirmar recepción

---

## 📚 Referencias Clave para Implementación

### Papers Académicos (15+ de 2022-2025)

**Foundation Models:**
1. Jakubik et al. (2024) - Prithvi Foundation Model
2. Cong et al. (2022) - SatMAE

**Region Growing y Segmentación:**
3. Ghamisi et al. (2022) - CRGNet (inspiración MGRG)
4. Ma et al. (2024) - DL-OBIA Hybridization
5. Adams & Bischof (1994) - Seeded Region Growing (clásico)

**Validación y Métricas:**
6. Brown et al. (2022) - Dynamic World
7. Zanaga et al. (2021) - ESA WorldCover

**Teledetección:**
8. Tucker (1979) - NDVI original
9. Gao (1996) - NDWI
10. Drusch et al. (2012) - Sentinel-2 mission
11. Claverie et al. (2018) - HLS product

**Adicionales (2022-2025):**
12. Tseng et al. (2023) - Crop type prediction with meta-learning
13. Rolf et al. (2021) - ML with global satellite imagery
14. Tseng et al. (2024) - Fields of the World benchmark
15. Yang et al. (2024) - Domain knowledge-enhanced region growing

### Recursos Técnicos

- **Prithvi HuggingFace:** https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
- **Sentinel Hub API:** https://docs.sentinel-hub.com/
- **Dynamic World:** https://www.dynamicworld.app/
- **Google Colab Tips:** https://colab.research.google.com/notebooks/pro.ipynb
- **Matplotlib Gallery:** https://matplotlib.org/stable/gallery/index.html

### Notebooks de Referencia del Proyecto

- `notebooks/experimental/04_embeddings-demo.ipynb` - Extracción de embeddings
- `notebooks/experimental/05_mgrg-demo.ipynb` - MGRG con análisis de threshold
- `notebooks/experimental/06_ab-comparison.ipynb` - Comparativa A/B
- `notebooks/validation/07_ground_truth_validation.ipynb` - Validación con Dynamic World
- `notebooks/classification/08_semantic_classification.ipynb` - Clasificación semántica
- `notebooks/experimental/09_hierarchical_pipeline_validation.ipynb` - Pipeline end-to-end

---

## 🎓 Conclusión de la Planeación

Esta planeación detallada garantiza que el Google Colab ejecutable cumpla con:

✅ **Todos los criterios de aceptación originales** (10 mínimos)
✅ **Estándares de excelencia del proyecto** (AGENTS.md 100%)
✅ **Requisitos de la rúbrica** (40% del proyecto)
✅ **Reproducibilidad y calidad académica**
✅ **Integración completa** con el pipeline (US-001 a US-011)
✅ **Datos reales** de 3 zonas de México
✅ **Validación científica** con Dynamic World
✅ **Comparativa cuantitativa** con métricas estándar
✅ **Visualizaciones profesionales** (300 DPI)
✅ **Documentación exhaustiva** (22 secciones)

### Diferenciadores de Excelencia

**Lo que hace único este notebook:**

1. **Datos Reales**: 3 zonas de México con 3.31M vectores procesados
2. **Validación Científica**: Dynamic World 2024, métricas estándar
3. **Comparativa Cuantitativa**: MGRG +252.8% mejor que Classic RG
4. **Análisis Completo**: Sensibilidad, casos de fallo, estadísticas
5. **Integración Total**: Conecta con todas las US anteriores
6. **Código Ejecutable**: 100% funcional, <30 min ejecución
7. **Visualizaciones Profesionales**: 300 DPI, multi-formato
8. **Documentación Bilingüe**: Inglés/Español según AGENTS.md
9. **Referencias Actuales**: 15+ papers de 2022-2025
10. **Reproducibilidad**: Código abierto, datos accesibles

### Próximos Pasos Inmediatos

1. **Aprobar esta planeación** con el equipo (reunión de 30 min)
2. **Asignar tareas específicas** según distribución propuesta
3. **Crear calendario de trabajo** para días 8-10 del sprint
4. **Iniciar Fase 1** (Preparación y Setup) - 3-4 horas
5. **Reuniones diarias** de sincronización (15 min)
6. **Revisión intermedia** al completar Fase 2 (checkpoint)
7. **Revisión final** antes de entrega (1 día antes)

### Compromiso del Equipo

Con esta planeación, el equipo se compromete a entregar un Google Colab de **excelencia técnica y académica** que:

- Demuestre dominio completo del proyecto
- Sirva como referencia educativa
- Sea reproducible y ejecutable
- Cumpla con los más altos estándares de calidad
- Obtenga la máxima calificación posible (40/40 puntos)

---

**Planeación creada por:** Equipo 24 - Region Growing
**Fecha:** 13 de Noviembre de 2025
**Estado:** 📋 **LISTA PARA APROBACIÓN Y EJECUCIÓN**
**Próxima acción:** Reunión de equipo para aprobación y asignación de tareas

🚀 **¡Listos para crear el mejor Google Colab del curso con excelencia académica y técnica!**

---

**Versión:** 2.0 (Definitiva - Incorpora aprendizajes de US-001 a US-011)
**Última actualización:** 13 de Noviembre de 2025
**Aprobado por:** Arthur Zizumbo
