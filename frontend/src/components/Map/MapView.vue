<template>
  <div class="map-container">
    <div id="map" ref="mapElement"></div>

    <!-- Controles sobre el mapa -->
    <div class="map-controls">
      <button
        @click="toggleDrawMode"
        :class="['btn', 'btn-draw', { active: store.isDrawing }]"
        :disabled="store.isLoading"
      >
        {{ store.isDrawing ? 'Cancelar' : 'Seleccionar Área' }}
      </button>

      <button
        v-if="store.isDrawing && polygonPoints.length >= 3"
        @click="finishPolygon"
        class="btn btn-finish"
      >
        Finalizar Polígono ({{ polygonPoints.length }} puntos)
      </button>

      <button
        v-if="store.selectedBounds"
        @click="clearSelection"
        class="btn btn-clear"
        :disabled="store.isLoading"
      >
        Limpiar Selección
      </button>
    </div>

    <!-- Instrucciones -->
    <div v-if="store.isDrawing" class="instructions">
      <p>Haz clic en el mapa para agregar puntos</p>
      <p>Mínimo 3 puntos para formar un polígono</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import L from 'leaflet'
import { useAnalysisStore } from '../../stores/analysis.store'

const store = useAnalysisStore()
const mapElement = ref(null)
const polygonPoints = ref([])
let map = null
let drawnItems = null
let currentRectangle = null
let resultsLayer = null
let tempPolygon = null
let tempMarkers = []
let tempLine = null

onMounted(() => {
  initMap()
})

function initMap() {
  // Inicializar mapa centrado en Perú (zona agrícola)
  map = L.map('map').setView([-12.0464, -77.0428], 10)

  // Agregar capa de OpenStreetMap
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19
  }).addTo(map)

  // Capa para elementos dibujados
  drawnItems = L.featureGroup().addTo(map)

  // Capa para resultados
  resultsLayer = L.featureGroup().addTo(map)

  // Eventos del mapa para dibujar
  map.on('mousedown', onMouseDown)
  map.on('mousemove', onMouseMove)
  map.on('mouseup', onMouseUp)
}

function toggleDrawMode() {
  const newDrawingState = !store.isDrawing
  store.setDrawingMode(newDrawingState)

  if (newDrawingState) {
    // Activar modo dibujo: desactivar drag del mapa
    map.dragging.disable()
    map.touchZoom.disable()
    map.doubleClickZoom.disable()
    map.scrollWheelZoom.disable()
    map.boxZoom.disable()
    map.keyboard.disable()
    if (map.tap) map.tap.disable()

    // Cambiar cursor
    document.getElementById('map').style.cursor = 'crosshair'

    // Resetear puntos
    polygonPoints.value = []
  } else {
    // Desactivar modo dibujo: reactivar drag del mapa
    map.dragging.enable()
    map.touchZoom.enable()
    map.doubleClickZoom.enable()
    map.scrollWheelZoom.enable()
    map.boxZoom.enable()
    map.keyboard.enable()
    if (map.tap) map.tap.enable()

    // Restaurar cursor
    document.getElementById('map').style.cursor = ''

    // Limpiar elementos temporales
    cleanupTempDrawing()
  }
}

function cleanupTempDrawing() {
  // Limpiar polígono temporal
  if (tempPolygon) {
    map.removeLayer(tempPolygon)
    tempPolygon = null
  }

  // Limpiar línea temporal
  if (tempLine) {
    map.removeLayer(tempLine)
    tempLine = null
  }

  // Limpiar marcadores temporales
  tempMarkers.forEach(marker => map.removeLayer(marker))
  tempMarkers = []

  // Limpiar rectángulo si existe (legacy)
  if (currentRectangle) {
    map.removeLayer(currentRectangle)
    currentRectangle = null
  }

  polygonPoints.value = []
}

function onMouseDown(e) {
  if (!store.isDrawing) return

  // Prevenir comportamiento por defecto
  L.DomEvent.stopPropagation(e)

  // Agregar punto al polígono
  polygonPoints.value.push(e.latlng)

  // Crear marcador para visualizar el punto
  const marker = L.circleMarker(e.latlng, {
    color: '#3388ff',
    fillColor: '#3388ff',
    fillOpacity: 1,
    radius: 5
  }).addTo(map)
  tempMarkers.push(marker)

  // Si hay al menos 2 puntos, dibujar línea temporal
  if (polygonPoints.value.length >= 2) {
    if (tempLine) {
      map.removeLayer(tempLine)
    }
    tempLine = L.polyline(polygonPoints.value, {
      color: '#3388ff',
      weight: 2,
      dashArray: '5, 5'
    }).addTo(map)
  }

  // Si hay al menos 3 puntos, mostrar polígono temporal
  if (polygonPoints.value.length >= 3) {
    if (tempPolygon) {
      map.removeLayer(tempPolygon)
    }
    tempPolygon = L.polygon(polygonPoints.value, {
      color: '#3388ff',
      weight: 2,
      fillOpacity: 0.2
    }).addTo(map)
  }
}

function onMouseMove(e) {
  // No necesitamos esto para el polígono
}

function onMouseUp(e) {
  // No necesitamos esto para el polígono
}

function finishPolygon() {
  if (polygonPoints.value.length < 3) {
    alert('Necesitas al menos 3 puntos para crear un polígono')
    return
  }

  // Crear polígono final (antes de limpiar los puntos)
  const polygon = L.polygon(polygonPoints.value, {
    color: '#3388ff',
    weight: 2,
    fillOpacity: 0.3
  })

  // Guardar en el store (antes de limpiar)
  store.setSelectedBounds(polygon.getBounds())
  store.setSelectedPolygon([...polygonPoints.value])

  // Limpiar elementos temporales ANTES de desactivar modo dibujo
  cleanupTempDrawing()

  // AHORA desactivar modo dibujo (después de limpiar puntos)
  store.setDrawingMode(false)

  // Reactivar interacción del mapa
  map.dragging.enable()
  map.touchZoom.enable()
  map.doubleClickZoom.enable()
  map.scrollWheelZoom.enable()
  map.boxZoom.enable()
  map.keyboard.enable()
  if (map.tap) map.tap.enable()
  document.getElementById('map').style.cursor = ''

  // Limpiar dibujos anteriores
  drawnItems.clearLayers()

  // Agregar polígono a la capa de dibujados
  polygon.addTo(drawnItems)
}

// Watch para finalizar polígono cuando se cancela el modo dibujo
watch(() => store.isDrawing, (isDrawing) => {
  if (!isDrawing && polygonPoints.value.length >= 3) {
    finishPolygon()
  }
})

function clearSelection() {
  drawnItems.clearLayers()
  store.clearSelectedBounds()
  store.clearResults()
}

// Watch para mostrar resultados en el mapa
watch(() => store.analysisResult, (newResult) => {
  if (!newResult) {
    resultsLayer.clearLayers()
    return
  }

  // Limpiar resultados anteriores
  resultsLayer.clearLayers()

  // Agregar GeoJSON de resultados
  if (newResult.geojson) {
    L.geoJSON(newResult.geojson, {
      style: (feature) => {
        const stressLevel = feature.properties?.stress_level || 'low'
        const colors = {
          high: '#dc3545',
          medium: '#ffc107',
          low: '#28a745'
        }
        return {
          color: colors[stressLevel] || '#3388ff',
          weight: 2,
          fillOpacity: 0.4
        }
      },
      onEachFeature: (feature, layer) => {
        if (feature.properties) {
          const props = feature.properties
          layer.bindPopup(`
            <strong>Nivel de estrés:</strong> ${props.stress_level || 'N/A'}<br>
            <strong>NDVI promedio:</strong> ${props.ndvi_mean?.toFixed(3) || 'N/A'}
          `)
        }
      }
    }).addTo(resultsLayer)
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

.map-controls {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 1000;
  display: flex;
  gap: 10px;
  flex-direction: column;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-draw {
  background-color: #007bff;
  color: white;
}

.btn-draw:hover:not(:disabled) {
  background-color: #0056b3;
}

.btn-draw.active {
  background-color: #dc3545;
}

.btn-draw.active:hover:not(:disabled) {
  background-color: #c82333;
}

.btn-clear {
  background-color: #6c757d;
  color: white;
}

.btn-clear:hover:not(:disabled) {
  background-color: #545b62;
}

.btn-finish {
  background-color: #28a745;
  color: white;
}

.btn-finish:hover:not(:disabled) {
  background-color: #218838;
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
