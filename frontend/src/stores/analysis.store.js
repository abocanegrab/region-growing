import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import apiService from '../services/api.service'

export const useAnalysisStore = defineStore('analysis', () => {
  // State
  const selectedBounds = ref(null)
  const selectedPolygon = ref(null)
  const analysisResult = ref(null)
  const isLoading = ref(false)
  const error = ref(null)
  const isDrawing = ref(false)
  const sizeWarning = ref(null)

  // Getters
  const hasResults = computed(() => analysisResult.value !== null)
  const hasError = computed(() => error.value !== null)

  // Helper function to calculate approximate image size
  function calculateImageSize(bounds) {
    const latDiff = Math.abs(bounds.getNorth() - bounds.getSouth())
    const lonDiff = Math.abs(bounds.getEast() - bounds.getWest())

    // Aproximadamente 111 km por grado de latitud
    // A 10m de resolución, eso es ~11100 píxeles por grado
    const pixelsPerDegreeLat = 11100
    const pixelsPerDegreeLon = 11100 * Math.cos((bounds.getNorth() + bounds.getSouth()) / 2 * Math.PI / 180)

    const heightPx = Math.round(latDiff * pixelsPerDegreeLat)
    const widthPx = Math.round(lonDiff * pixelsPerDegreeLon)

    return { width: widthPx, height: heightPx }
  }

  // Actions
  function setSelectedBounds(bounds) {
    selectedBounds.value = bounds
    error.value = null

    // Validar tamaño de la región
    const size = calculateImageSize(bounds)
    const maxDimension = 2500
    const areaSqKm = (Math.abs(bounds.getNorth() - bounds.getSouth()) * 111) *
                     (Math.abs(bounds.getEast() - bounds.getWest()) * 111 *
                      Math.cos((bounds.getNorth() + bounds.getSouth()) / 2 * Math.PI / 180))

    if (size.width > maxDimension || size.height > maxDimension) {
      sizeWarning.value = {
        type: 'error',
        message: `⚠️ La región seleccionada es muy grande (${size.width}x${size.height} px, ~${areaSqKm.toFixed(1)} km²). Por favor, selecciona un área más pequeña (máximo ~62 km²).`,
        canAnalyze: false
      }
    } else if (size.width > 2000 || size.height > 2000) {
      sizeWarning.value = {
        type: 'warning',
        message: `⚠️ Región grande (${size.width}x${size.height} px, ~${areaSqKm.toFixed(1)} km²). El análisis puede tardar más tiempo.`,
        canAnalyze: true
      }
    } else {
      sizeWarning.value = null
    }
  }

  function setSelectedPolygon(points) {
    selectedPolygon.value = points
  }

  function clearSelectedBounds() {
    selectedBounds.value = null
    selectedPolygon.value = null
  }

  function setDrawingMode(isActive) {
    isDrawing.value = isActive
  }

  async function analyzeRegion(dateFrom = null, dateTo = null) {
    if (!selectedBounds.value) {
      error.value = 'No hay región seleccionada'
      return
    }

    // Verificar si la región es demasiado grande
    if (sizeWarning.value && !sizeWarning.value.canAnalyze) {
      error.value = 'La región seleccionada es muy grande. Por favor, selecciona un área más pequeña.'
      return
    }

    try {
      isLoading.value = true
      error.value = null

      const bbox = {
        min_lat: selectedBounds.value.getSouth(),
        min_lon: selectedBounds.value.getWest(),
        max_lat: selectedBounds.value.getNorth(),
        max_lon: selectedBounds.value.getEast()
      }

      const response = await apiService.analyzeRegion(bbox, dateFrom, dateTo)

      if (response.success) {
        analysisResult.value = response.data
      } else {
        throw new Error(response.error || 'Error desconocido')
      }
    } catch (err) {
      error.value = err.response?.data?.error || err.message || 'Error al analizar la región'
      console.error('Error en análisis:', err)
    } finally {
      isLoading.value = false
    }
  }

  function clearResults() {
    analysisResult.value = null
    error.value = null
  }

  function clearError() {
    error.value = null
  }

  function reset() {
    selectedBounds.value = null
    selectedPolygon.value = null
    analysisResult.value = null
    isLoading.value = false
    error.value = null
    isDrawing.value = false
  }

  return {
    // State
    selectedBounds,
    selectedPolygon,
    analysisResult,
    isLoading,
    error,
    isDrawing,
    sizeWarning,
    // Getters
    hasResults,
    hasError,
    // Actions
    setSelectedBounds,
    setSelectedPolygon,
    clearSelectedBounds,
    setDrawingMode,
    analyzeRegion,
    clearResults,
    clearError,
    reset
  }
})
