<template>
  <div class="results-panel">
    <div class="results-header">
      <h3>Resultados del Análisis</h3>
      <button
        @click="$emit('clear')"
        class="btn-clear"
        title="Limpiar resultados"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>

    <div class="statistics">
      <div class="stat-item">
        <label>
          Área Total
          <InfoTooltip text="Área total analizada en hectáreas" />
        </label>
        <span class="stat-value">{{ statistics.total_area.toFixed(2) }} ha</span>
      </div>

      <div class="stat-item">
        <label>
          Número de Regiones
          <InfoTooltip text="Cantidad de regiones homogéneas detectadas" />
        </label>
        <span class="stat-value">{{ statistics.num_regions }}</span>
      </div>

      <div class="stat-item">
        <label>
          NDVI Promedio
          <InfoTooltip text="Índice de Vegetación de Diferencia Normalizada promedio" />
        </label>
        <span class="stat-value">{{ statistics.mean_ndvi.toFixed(3) }}</span>
      </div>

      <div class="stat-item">
        <label>
          Cobertura de Nubes
          <InfoTooltip text="Porcentaje de cobertura nubosa en el área analizada" />
        </label>
        <span class="stat-value">{{ statistics.cloud_coverage.toFixed(1) }}%</span>
      </div>
    </div>

    <div class="regions-summary">
      <h4>Distribución de Estrés</h4>
      <div class="stress-levels">
        <div
          v-for="level in stressLevels"
          :key="level.name"
          class="stress-item"
          :class="`stress-${level.name}`"
        >
          <span class="stress-icon">{{ level.icon }}</span>
          <span class="stress-label">{{ level.label }}</span>
          <span class="stress-count">{{ level.count }}</span>
        </div>
      </div>
    </div>

    <div class="actions">
      <button
        @click="showDetailedModal = true"
        class="btn btn-primary"
      >
        Ver Análisis Detallado
      </button>
      <button
        @click="exportResults"
        class="btn btn-secondary"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="7 10 12 15 17 10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        Exportar JSON
      </button>
    </div>

    <DetailedResultsModal
      v-if="showDetailedModal"
      :results="analysisStore.analysisResult"
      @close="showDetailedModal = false"
    />
  </div>
</template>

<script setup lang="ts">
import InfoTooltip from '../Common/InfoTooltip.vue'
import DetailedResultsModal from './DetailedResultsModal.vue'
import type { AnalysisResult } from '~/types'

const props = defineProps<{
  statistics: {
    total_area: number
    num_regions: number
    mean_ndvi: number
    cloud_coverage: number
    high_stress_area?: number
    medium_stress_area?: number
    low_stress_area?: number
    num_high_stress_regions?: number
    num_medium_stress_regions?: number
    num_low_stress_regions?: number
    date_from?: string
    date_to?: string
  }
  regions: Array<{
    id: number
    stress_level: 'high' | 'medium' | 'low'
    ndvi_mean: number
    area: number
  }>
  images?: {
    rgb?: string
    ndvi?: string
  }
}>()

defineEmits<{
  clear: []
}>()

const showDetailedModal = ref(false)
const analysisStore = useAnalysisStore()

const stressLevels = computed(() => {
  const counts = {
    high: 0,
    medium: 0,
    low: 0
  }

  if (props.regions && Array.isArray(props.regions)) {
    props.regions.forEach(region => {
      counts[region.stress_level]++
    })
  }

  return [
    {
      name: 'high',
      label: 'Estrés Alto',
      icon: '🔴',
      count: counts.high
    },
    {
      name: 'medium',
      label: 'Estrés Medio',
      icon: '🟡',
      count: counts.medium
    },
    {
      name: 'low',
      label: 'Estrés Bajo',
      icon: '🟢',
      count: counts.low
    }
  ]
})

const exportResults = () => {
  const data = {
    statistics: props.statistics,
    regions: props.regions,
    timestamp: new Date().toISOString()
  }

  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json'
  })

  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `analisis-${new Date().getTime()}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
</script>

<style scoped>
.results-panel {
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.results-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.btn-clear {
  background: none;
  border: none;
  cursor: pointer;
  padding: 8px;
  border-radius: 4px;
  color: #666;
  transition: all 0.2s;
}

.btn-clear:hover {
  background-color: #f0f0f0;
  color: #dc3545;
}

.statistics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 15px;
  margin-bottom: 20px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.stat-item label {
  font-size: 12px;
  color: #666;
  display: flex;
  align-items: center;
  gap: 5px;
}

.stat-value {
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.regions-summary {
  margin-bottom: 20px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 6px;
}

.regions-summary h4 {
  margin: 0 0 15px 0;
  font-size: 14px;
  font-weight: 600;
  color: #333;
}

.stress-levels {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.stress-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: 4px;
  background-color: white;
}

.stress-item.stress-high {
  border-left: 4px solid #dc3545;
}

.stress-item.stress-medium {
  border-left: 4px solid #ffc107;
}

.stress-item.stress-low {
  border-left: 4px solid #28a745;
}

.stress-icon {
  font-size: 16px;
}

.stress-label {
  flex: 1;
  font-size: 14px;
  color: #333;
}

.stress-count {
  font-weight: 600;
  font-size: 16px;
  color: #333;
}

.actions {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s;
}

.btn-primary {
  background-color: #007bff;
  color: white;
}

.btn-primary:hover {
  background-color: #0056b3;
}

.btn-secondary {
  background-color: #6c757d;
  color: white;
}

.btn-secondary:hover {
  background-color: #545b62;
}
</style>
