import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
})

export default {
  /**
   * Health check del backend
   */
  async healthCheck() {
    const response = await apiClient.get('/health')
    return response.data
  },

  /**
   * Analizar región para detectar estrés vegetal
   * @param {Object} bbox - Bounding box {min_lat, min_lon, max_lat, max_lon}
   * @param {String} dateFrom - Fecha desde (YYYY-MM-DD)
   * @param {String} dateTo - Fecha hasta (YYYY-MM-DD)
   */
  async analyzeRegion(bbox, dateFrom = null, dateTo = null) {
    const payload = {
      bbox,
      ...(dateFrom && { date_from: dateFrom }),
      ...(dateTo && { date_to: dateTo })
    }

    const response = await apiClient.post('/api/analysis/analyze', payload)
    return response.data
  },

  /**
   * Test endpoint
   */
  async testAnalysis() {
    const response = await apiClient.get('/api/analysis/test')
    return response.data
  }
}
