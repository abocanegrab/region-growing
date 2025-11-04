<template>
  <div class="results-panel">
    <div v-if="store.analysisResult" class="results-content">
      <!-- Botón prominente para abrir análisis detallado -->
      <button @click="showDetailedModal = true" class="btn btn-detailed">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"></circle>
          <path d="m21 21-4.35-4.35"></path>
        </svg>
        Ver Análisis Detallado
      </button>

      <div class="statistics">
        <h4>
          Estadísticas Generales
          <InfoTooltip
            content="Resumen rápido de los resultados. Para ver explicaciones detalladas y comparación de imágenes, haz clic en 'Ver Análisis Detallado'."
            position="right"
          />
        </h4>

        <div class="stat-grid" v-if="stats">
          <div class="stat-item">
            <span class="stat-label">
              NDVI Promedio:
              <InfoTooltip
                title="NDVI"
                content="Mide la salud de la vegetación en una escala de -1 a 1. Valores altos (>0.5) = plantas sanas. Valores bajos (<0.3) = estrés o sin vegetación."
                position="bottom"
              />
            </span>
            <span class="stat-value">{{ stats.mean_ndvi?.toFixed(3) || 'N/A' }}</span>
          </div>

          <div class="stat-item">
            <span class="stat-label">Área Total:</span>
            <span class="stat-value">{{ stats.total_area?.toFixed(2) || 'N/A' }} ha</span>
          </div>

          <div class="stat-item cloud-info">
            <span class="stat-label">
              Cobertura de Nubes:
              <InfoTooltip
                content="Porcentaje del área cubierta por nubes. Las áreas con nubes no se incluyen en el análisis de vegetación. Valores >30% pueden afectar la precisión del análisis."
                position="bottom"
              />
            </span>
            <span class="stat-value">{{ stats.cloud_coverage?.toFixed(1) || 0 }}%</span>
          </div>

          <div class="stat-item high-stress">
            <span class="stat-label">
              Estrés Alto:
              <InfoTooltip
                content="Zonas con vegetación muy estresada o suelo desnudo. Necesitan atención urgente si son cultivos."
                position="bottom"
              />
            </span>
            <span class="stat-value">{{ stats.high_stress_area?.toFixed(2) || 0 }} ha</span>
          </div>

          <div class="stat-item medium-stress">
            <span class="stat-label">
              Estrés Medio:
              <InfoTooltip
                content="Vegetación con estrés moderado. Puede deberse a falta de agua, nutrientes o condiciones no óptimas."
                position="bottom"
              />
            </span>
            <span class="stat-value">{{ stats.medium_stress_area?.toFixed(2) || 0 }} ha</span>
          </div>

          <div class="stat-item low-stress">
            <span class="stat-label">
              Estrés Bajo:
              <InfoTooltip
                content="Vegetación saludable. Las plantas están creciendo bien con agua y nutrientes suficientes."
                position="bottom"
              />
            </span>
            <span class="stat-value">{{ stats.low_stress_area?.toFixed(2) || 0 }} ha</span>
          </div>
        </div>
      </div>

      <div class="legend">
        <h4>Leyenda del Mapa</h4>
        <div class="legend-items">
          <div class="legend-item">
            <span class="legend-color high"></span>
            <span>🔴 Estrés Alto (NDVI < 0.3)</span>
          </div>
          <div class="legend-item">
            <span class="legend-color medium"></span>
            <span>🟡 Estrés Medio (0.3 ≤ NDVI < 0.5)</span>
          </div>
          <div class="legend-item">
            <span class="legend-color low"></span>
            <span>🟢 Estrés Bajo (NDVI ≥ 0.5)</span>
          </div>
        </div>
        <p class="legend-note">
          💡 Los polígonos en el mapa muestran las regiones detectadas automáticamente
        </p>
      </div>

      <button @click="exportResults" class="btn btn-export">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="7 10 12 15 17 10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        Exportar JSON
      </button>
    </div>

    <!-- Modal Detallado -->
    <DetailedResultsModal
      v-model="showDetailedModal"
      :results="store.analysisResult"
    />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useAnalysisStore } from '../../stores/analysis.store'
import InfoTooltip from '../Common/InfoTooltip.vue'
import DetailedResultsModal from './DetailedResultsModal.vue'

const store = useAnalysisStore()
const showDetailedModal = ref(false)

const stats = computed(() => {
  return store.analysisResult?.statistics || null
})

function exportResults() {
  if (!store.analysisResult) return

  const dataStr = JSON.stringify(store.analysisResult, null, 2)
  const dataBlob = new Blob([dataStr], { type: 'application/json' })

  const url = URL.createObjectURL(dataBlob)
  const link = document.createElement('a')
  link.href = url
  link.download = `analisis-${new Date().toISOString().split('T')[0]}.json`
  link.click()

  URL.revokeObjectURL(url)
}
</script>

<style scoped>
.results-panel {
  margin-top: 20px;
}

.results-content {
  background: white;
  border-radius: 8px;
}

/* Botón de Análisis Detallado */
.btn-detailed {
  width: 100%;
  padding: 16px 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  font-weight: 600;
  transition: all 0.3s;
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.btn-detailed:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

.btn-detailed:active {
  transform: translateY(0);
}

.statistics {
  margin-bottom: 20px;
}

h4 {
  color: #333;
  margin-bottom: 15px;
  font-size: 16px;
  display: flex;
  align-items: center;
}

.stat-grid {
  display: grid;
  gap: 12px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 4px;
  border-left: 4px solid #007bff;
}

.stat-item.high-stress {
  border-left-color: #dc3545;
}

.stat-item.medium-stress {
  border-left-color: #ffc107;
}

.stat-item.low-stress {
  border-left-color: #28a745;
}

.stat-item.cloud-info {
  border-left-color: #6c757d;
}

.stat-label {
  color: #666;
  font-weight: 500;
  display: flex;
  align-items: center;
}

.stat-value {
  color: #333;
  font-weight: 600;
}

.legend {
  margin-bottom: 20px;
}

.legend-items {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 10px;
}

.legend-color {
  width: 30px;
  height: 20px;
  border-radius: 3px;
  border: 1px solid #ddd;
}

.legend-color.high {
  background-color: rgba(220, 53, 69, 0.4);
  border-color: #dc3545;
}

.legend-color.medium {
  background-color: rgba(255, 193, 7, 0.4);
  border-color: #ffc107;
}

.legend-color.low {
  background-color: rgba(40, 167, 69, 0.4);
  border-color: #28a745;
}

.legend-note {
  margin-top: 12px;
  padding: 8px 12px;
  background-color: #e7f3ff;
  border-left: 3px solid #2196f3;
  border-radius: 4px;
  font-size: 13px;
  color: #1976d2;
  margin-bottom: 0;
}

.btn-export {
  width: 100%;
  padding: 12px;
  background-color: #17a2b8;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.btn-export:hover {
  background-color: #138496;
  transform: translateY(-1px);
}
</style>
