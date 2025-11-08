<template>
  <div class="map-container">
    <div id="map" ref="mapElement"></div>

    <MapLayerControls
      :has-images="hasImages"
      :has-results="hasResults"
      @toggle-layer="handleToggleLayer"
    />

    <MapControls
      :is-drawing="isDrawing"
      :has-selection="!!selectedBounds"
      :is-loading="loading"
      @start-draw="handleStartDraw"
      @stop-draw="handleStopDraw"
      @clear-selection="handleClearSelection"
    />

    <div v-if="isDrawing" class="instructions">
      <p>Haz clic en el mapa para dibujar un polígono</p>
      <p>Doble clic para finalizar</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import MapLayerControls from './LayerControls.vue'

const analysisStore = useAnalysisStore()
const {
  mapInstance,
  isDrawing,
  selectedBounds,
  initMap,
  initDrawControl,
  startDrawing,
  stopDrawing,
  clearSelection,
  addResultsLayer,
  clearResultsLayer,
  addRasterLayer,
  toggleRasterLayer,
  clearRasterLayers
} = useMap()

const mapElement = ref<HTMLElement | null>(null)
const loading = computed(() => analysisStore.isLoading)
const hasImages = computed(() => !!analysisStore.analysisResult?.images)
const hasResults = computed(() => !!analysisStore.analysisResult?.geojson)

onMounted(() => {
  if (process.client) {
    const map = initMap('map', {
      center: [-77.0428, -12.0464],
      zoom: 10
    })

    if (map) {
      map.on('load', () => {
        initDrawControl()
      })
    }
  }
})

const handleStartDraw = () => {
  startDrawing()
}

const handleStopDraw = () => {
  stopDrawing()
}

const handleClearSelection = () => {
  clearSelection()
  clearResultsLayer()
  clearRasterLayers()
}

const handleToggleLayer = (layerId: string, visible: boolean) => {
  if (layerId === 'results') {
    if (visible && analysisStore.analysisResult?.geojson) {
      addResultsLayer(analysisStore.analysisResult.geojson)
    } else {
      clearResultsLayer()
    }
  } else {
    toggleRasterLayer(layerId, visible)
  }
}

watch(() => analysisStore.analysisResult, (newResult) => {
  if (!newResult) {
    clearResultsLayer()
    clearRasterLayers()
    return
  }

  // IMPORTANT: Add raster layers FIRST (they will be below)
  if (newResult.images && analysisStore.selectedBounds) {
    const bounds = analysisStore.selectedBounds

    if (newResult.images.rgb) {
      addRasterLayer('raster-rgb', newResult.images.rgb, bounds)
    }

    if (newResult.images.false_color) {
      addRasterLayer('raster-false-color', newResult.images.false_color, bounds)
    }

    if (newResult.images.ndvi) {
      addRasterLayer('raster-ndvi', newResult.images.ndvi, bounds)
    }
  }

  // Add GeoJSON results layer AFTER (it will be on top)
  if (newResult.geojson) {
    addResultsLayer(newResult.geojson)
  }
})

onBeforeUnmount(() => {
  if (mapInstance.value) {
    mapInstance.value.remove()
  }
})
</script>

<style scoped>
.map-container {
  position: relative;
  width: 100%;
  height: 100%;
}

#map {
  width: 100%;
  height: 100%;
}

.instructions {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  background-color: rgba(255, 255, 255, 0.95);
  padding: 15px 25px;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.3);
  z-index: 1000;
  text-align: center;
}

.instructions p {
  margin: 5px 0;
  color: #333;
  font-size: 14px;
}

.instructions p:first-child {
  font-weight: 600;
  color: #007bff;
}
</style>
