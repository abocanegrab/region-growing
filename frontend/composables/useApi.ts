export const useApi = () => {
  const config = useRuntimeConfig()

  const healthCheck = async () => {
    return await $fetch(`${config.public.apiBase}/health`)
  }

  const analyzeRegion = async (bbox: any, dateFrom?: string, dateTo?: string) => {
    return await $fetch(`${config.public.apiBase}/api/analysis/analyze`, {
      method: 'POST',
      body: {
        bbox,
        date_from: dateFrom,
        date_to: dateTo
      }
    })
  }

  return {
    healthCheck,
    analyzeRegion
  }
}
