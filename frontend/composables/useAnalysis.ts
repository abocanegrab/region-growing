import type { BBox, AnalysisResult } from '~/types'

export const useAnalysis = () => {
  const config = useRuntimeConfig()
  const analysisStore = useAnalysisStore()

  const results = computed(() => analysisStore.analysisResult)
  const loading = computed(() => analysisStore.isLoading)
  const error = computed(() => analysisStore.error)

  const analyzeRegion = async (
    bbox: BBox,
    dateFrom?: string,
    dateTo?: string
  ) => {
    analysisStore.isLoading = true
    analysisStore.error = null

    try {
      const response = await $fetch<{ success: boolean; data: AnalysisResult }>(
        `${config.public.apiBase}/api/analysis/analyze`,
        {
          method: 'POST',
          body: {
            bbox,
            date_from: dateFrom,
            date_to: dateTo
          }
        }
      )

      if (response.success) {
        analysisStore.analysisResult = response.data
        return response.data
      } else {
        throw new Error('Analysis failed')
      }
    } catch (e: any) {
      const errorMessage = e.response?.data?.error || e.message || 'Error al analizar la región'
      analysisStore.error = errorMessage
      throw new Error(errorMessage)
    } finally {
      analysisStore.isLoading = false
    }
  }

  const clearResults = () => {
    analysisStore.analysisResult = null
    analysisStore.error = null
  }

  const clearError = () => {
    analysisStore.error = null
  }

  return {
    results: readonly(results),
    loading: readonly(loading),
    error: readonly(error),
    analyzeRegion,
    clearResults,
    clearError
  }
}
