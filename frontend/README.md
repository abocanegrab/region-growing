# Frontend - Sistema de Detección de Estrés Vegetal

Frontend desarrollado con **Nuxt 3** para el análisis de estrés vegetal mediante imágenes satelitales Sentinel-2 y algoritmo Region Growing.

---

## 🚀 Quick Start

### Requisitos Previos

- **Node.js** 18+ o 20+
- **pnpm** (recomendado) o npm

### Instalación

```bash
# Instalar pnpm si no lo tienes
npm install -g pnpm

# Instalar dependencias
pnpm install
```

### Desarrollo

```bash
# Iniciar servidor de desarrollo
pnpm run dev
```

Abre tu navegador en **http://localhost:3000**

### Producción

```bash
# Build para producción
pnpm run build

# Preview del build
pnpm run preview
```

---

## 📦 Stack Tecnológico

- **Nuxt 3.10+** - Framework Vue con SSR
- **Vue 3.4+** - Framework JavaScript reactivo
- **Pinia** - State management
- **MapLibre GL** - Mapas interactivos con WebGL
- **MapboxDraw** - Dibujo de polígonos
- **TypeScript** - Type safety
- **Axios** - Cliente HTTP

---

## 🗂️ Estructura del Proyecto

```
frontend/
├── assets/              # Assets estáticos (CSS, imágenes)
├── components/          # Componentes Vue (auto-import)
│   ├── Analysis/        # Componentes de análisis
│   ├── Common/          # Componentes comunes
│   └── Map/             # Componentes del mapa
├── composables/         # Composables (auto-import)
│   ├── useAnalysis.ts   # Lógica de análisis
│   ├── useApi.ts        # Cliente API
│   └── useMap.ts        # Lógica del mapa
├── layouts/             # Layouts de página
├── pages/               # Páginas (file-based routing)
├── plugins/             # Plugins de Nuxt
├── public/              # Archivos públicos
├── stores/              # Pinia stores
├── types/               # Tipos TypeScript
├── nuxt.config.ts       # Configuración de Nuxt
└── package.json         # Dependencias
```

---

## 🎮 Comandos Disponibles

```bash
# Desarrollo
pnpm run dev              # Servidor de desarrollo con hot reload

# Producción
pnpm run build            # Build para producción
pnpm run generate         # Generar sitio estático (SSG)
pnpm run preview          # Preview del build de producción

# Utilidades
pnpm run typecheck        # Verificar tipos TypeScript
pnpm run postinstall      # Preparar tipos de Nuxt (automático)
```

---

## ⚙️ Configuración

### Variables de Entorno

Crea un archivo `.env` en la raíz del frontend:

```env
# URL del backend
NUXT_PUBLIC_API_BASE=http://localhost:8000
```

### Configuración de Nuxt

El archivo `nuxt.config.ts` contiene la configuración principal:

```typescript
export default defineNuxtConfig({
  modules: ['@pinia/nuxt'],
  
  css: [
    'maplibre-gl/dist/maplibre-gl.css',
    '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css',
    '~/assets/css/main.css'
  ],
  
  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000'
    }
  }
})
```

---

## 🗺️ Características del Mapa

### Mapa Base
- **OpenStreetMap** - Mapa base con calles y referencias

### Controles
- **Top-Left:** Botones de selección (Seleccionar Área, Limpiar)
- **Top-Right:** Panel de capas + Zoom
- **Bottom-Left:** Controles de dibujo + Escala

### Capas Disponibles
1. **Imagen RGB** - Imagen satelital en color verdadero
2. **Falso Color** - Composición NIR-Red-Green (vegetación en rojo)
3. **Mapa NDVI** - Mapa de salud vegetal coloreado
4. **Regiones** - Segmentación por nivel de estrés

### Funcionalidades
- ✅ Dibujo de polígonos para selección de área
- ✅ Análisis de región con fechas personalizables
- ✅ Visualización de resultados en tiempo real
- ✅ Capas raster georeferenciadas
- ✅ Control de visibilidad de capas
- ✅ Popups informativos
- ✅ Exportar resultados a JSON

---

## 🧩 Composables Principales

### useAnalysis

Maneja la lógica de análisis de regiones:

```typescript
const { results, loading, error, analyzeRegion, clearResults } = useAnalysis()

// Analizar región
await analyzeRegion(bbox, dateFrom, dateTo)
```

### useMap

Maneja la lógica del mapa MapLibre GL:

```typescript
const {
  mapInstance,
  isDrawing,
  selectedBounds,
  initMap,
  startDrawing,
  addResultsLayer,
  addRasterLayer,
  toggleRasterLayer
} = useMap()

// Inicializar mapa
const map = initMap('map-container', { center: [-77, -12], zoom: 10 })

// Agregar capa raster
addRasterLayer('raster-rgb', imageBase64, bounds)

// Activar/desactivar capa
toggleRasterLayer('raster-rgb', true)
```

### useApi

Cliente HTTP para comunicación con el backend:

```typescript
const { healthCheck, testAnalysis } = useApi()

// Verificar salud del backend
const status = await healthCheck()
```

---

## 📊 Store (Pinia)

### Analysis Store

```typescript
const analysisStore = useAnalysisStore()

// State
analysisStore.selectedBounds      // Región seleccionada
analysisStore.analysisResult      // Resultados del análisis
analysisStore.isLoading           // Estado de carga
analysisStore.error               // Errores
analysisStore.isDrawing           // Modo de dibujo activo

// Getters
analysisStore.hasResults          // Tiene resultados
analysisStore.hasError            // Tiene errores

// Actions
analysisStore.setSelectedBounds(bounds)
analysisStore.clearResults()
analysisStore.reset()
```

---

## 🎨 Componentes Principales

### MapLibreMap
Componente principal del mapa con MapLibre GL.

```vue
<MapLibreMap />
```

### AnalysisPanel
Panel lateral con controles de análisis.

```vue
<AnalysisPanel />
```

### ResultsPanel
Panel de resultados con estadísticas.

```vue
<ResultsPanel
  :statistics="statistics"
  :regions="regions"
  :images="images"
/>
```

### DetailedResultsModal
Modal con análisis detallado, imágenes y tabla de regiones.

```vue
<DetailedResultsModal
  :results="analysisResult"
  @close="closeModal"
/>
```

---

## 🔧 Desarrollo

### Auto-imports

Nuxt 3 auto-importa:
- ✅ Componentes de `components/`
- ✅ Composables de `composables/`
- ✅ Utils de Vue (`ref`, `computed`, `watch`, etc.)
- ✅ Stores de Pinia

No necesitas importar explícitamente:

```vue
<script setup>
// ✅ Auto-importado
const store = useAnalysisStore()
const count = ref(0)
const double = computed(() => count.value * 2)
</script>
```

### TypeScript

El proyecto usa TypeScript para type safety:

```typescript
// types/index.ts
export interface BBox {
  min_lat: number
  max_lat: number
  min_lon: number
  max_lon: number
}

export interface AnalysisResult {
  geojson?: any
  statistics?: Statistics
  regions?: Region[]
  images?: Images
}
```

### SSR (Server-Side Rendering)

Componentes que usan APIs del navegador deben envolverse en `ClientOnly`:

```vue
<template>
  <ClientOnly>
    <MapLibreMap />
    <template #fallback>
      <div>Cargando mapa...</div>
    </template>
  </ClientOnly>
</template>
```

---

## 🐛 Troubleshooting

### El mapa no se muestra

Verifica que:
1. MapLibre GL esté cargado (plugin `maplibre.client.ts`)
2. El componente esté dentro de `<ClientOnly>`
3. El contenedor tenga altura definida en CSS

### Error "process is not defined"

Es un falso positivo de TypeScript. `process.client` es una variable global de Nuxt.

### Las capas no se ven

Verifica:
1. Que las imágenes se reciban del backend (base64)
2. Que el bounds sea correcto
3. Que la capa esté activada en el panel de capas

### Auto-imports no funcionan

Ejecuta:
```bash
pnpm run postinstall
```

Esto regenera los tipos de auto-imports.

---

## 📚 Recursos

- [Nuxt 3 Docs](https://nuxt.com/docs)
- [Vue 3 Docs](https://vuejs.org/)
- [Pinia Docs](https://pinia.vuejs.org/)
- [MapLibre GL Docs](https://maplibre.org/maplibre-gl-js/docs/)
- [MapboxDraw Docs](https://github.com/mapbox/mapbox-gl-draw)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

---

## 🤝 Contribuir

1. Sigue el estándar de código definido en `AGENTS.md`
2. Usa TypeScript para nuevos archivos
3. Agrega tipos a las interfaces en `types/`
4. Documenta funciones complejas
5. Prueba en desarrollo antes de commitear

---

## 📝 Notas

- **Performance:** MapLibre GL usa WebGL, 3-5x más rápido que Leaflet
- **Coordenadas:** MapLibre usa formato `[lng, lat]` (no `[lat, lng]`)
- **Capas:** El orden importa - las últimas agregadas quedan arriba
- **SSR:** Componentes con APIs del navegador deben ser client-only

---

**Versión:** 1.0.0  
**Framework:** Nuxt 3.10+  
**Última actualización:** Noviembre 2025
