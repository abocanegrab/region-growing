import { defineStore } from 'pinia'
import type { BBox, AnalysisResult, SizeWarning } from '~/types'

export const useAnalysisStore = defineStore('analysis', {
  state: () => ({
    selectedBounds: null as BBox | null,
    selectedPolygon: null as number[][] | null,
    analysisResult: null as AnalysisResult | null,
    isLoading: false,
    error: null as string | null,
    isDrawing: false,
    sizeWarning: null as SizeWarning | null
  }),

  getters: {
    hasResults: (state) => state.analysisResult !== null,
    hasError: (state) => state.error !== null
  },

  actions: {
    setSelectedBounds(bounds: BBox) {
      this.selectedBounds = bounds
      this.error = null
      this.validateRegionSize(bounds)
    },

    setSelectedPolygon(coordinates: number[][]) {
      this.selectedPolygon = coordinates
    },

    clearSelectedBounds() {
      this.selectedBounds = null
      this.selectedPolygon = null
      this.sizeWarning = null
    },

    setDrawingMode(isActive: boolean) {
      this.isDrawing = isActive
    },

    clearResults() {
      this.analysisResult = null
      this.error = null
    },

    clearError() {
      this.error = null
    },

    reset() {
      this.selectedBounds = null
      this.selectedPolygon = null
      this.analysisResult = null
      this.isLoading = false
      this.error = null
      this.isDrawing = false
      this.sizeWarning = null
    },

    validateRegionSize(bounds: BBox) {
      const latDiff = Math.abs(bounds.max_lat - bounds.min_lat)
      const lonDiff = Math.abs(bounds.max_lon - bounds.min_lon)

      const pixelsPerDegreeLat = 11100
      const pixelsPerDegreeLon = 11100 * Math.cos(
        (bounds.max_lat + bounds.min_lat) / 2 * Math.PI / 180
      )

      const heightPx = Math.round(latDiff * pixelsPerDegreeLat)
      const widthPx = Math.round(lonDiff * pixelsPerDegreeLon)

      const maxDimension = 2500
      const areaSqKm = (latDiff * 111) * (lonDiff * 111 *
        Math.cos((bounds.max_lat + bounds.min_lat) / 2 * Math.PI / 180))

      if (widthPx > maxDimension || heightPx > maxDimension) {
        this.sizeWarning = {
          type: 'error',
          message: `La región seleccionada es muy grande (${widthPx}x${heightPx} px, ~${areaSqKm.toFixed(1)} km²). Por favor, selecciona un área más pequeña (máximo ~62 km²).`,
          canAnalyze: false
        }
      } else if (widthPx > 2000 || heightPx > 2000) {
        this.sizeWarning = {
          type: 'warning',
          message: `Región grande (${widthPx}x${heightPx} px, ~${areaSqKm.toFixed(1)} km²). El análisis puede tardar más tiempo.`,
          canAnalyze: true
        }
      } else {
        this.sizeWarning = null
      }
    }
  }
})
