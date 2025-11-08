export interface BBox {
  min_lat: number
  min_lon: number
  max_lat: number
  max_lon: number
}

export interface AnalysisResult {
  geojson?: any
  statistics?: {
    total_area: number
    num_regions: number
    mean_ndvi: number
    cloud_coverage: number
    high_stress_area: number
    medium_stress_area: number
    low_stress_area: number
    num_high_stress_regions: number
    num_medium_stress_regions: number
    num_low_stress_regions: number
    date_from?: string
    date_to?: string
  }
  regions?: Array<{
    id: number
    stress_level: 'high' | 'medium' | 'low'
    ndvi_mean: number
    area: number
  }>
  images?: {
    rgb?: string
    ndvi?: string
    false_color?: string
  }
}

export type AnalysisMethod = 'classic' | 'hybrid'

export interface SizeWarning {
  type: 'error' | 'warning'
  message: string
  canAnalyze: boolean
}
