<template>
  <div class="analysis-panel">
    <h2>Análisis de Estrés Vegetal</h2>

    <div class="panel-section">
      <h3>Región Seleccionada</h3>
      <div v-if="store.selectedBounds" class="bounds-info">
        <p><strong>Latitud:</strong> {{ bounds.south.toFixed(4) }} a {{ bounds.north.toFixed(4) }}</p>
        <p><strong>Longitud:</strong> {{ bounds.west.toFixed(4) }} a {{ bounds.east.toFixed(4) }}</p>

        <!-- Advertencia de tamaño -->
        <div v-if="store.sizeWarning"
             :class="['size-warning', `warning-${store.sizeWarning.type}`]">
          <svg v-if="store.sizeWarning.type === 'error'" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <svg v-else xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
            <line x1="12" y1="9" x2="12" y2="13"></line>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
          </svg>
          <div class="warning-content">
            <p>{{ store.sizeWarning.message }}</p>
            <p v-if="!store.sizeWarning.canAnalyze" class="warning-action">
              ✏️ Dibuja un nuevo polígono más pequeño para poder analizar.
            </p>
          </div>
        </div>
      </div>
      <div v-else class="no-selection">
        <p>No hay región seleccionada</p>
        <p class="hint">Usa el botón "Seleccionar Área" en el mapa</p>
      </div>
    </div>

    <div class="panel-section" v-if="store.selectedBounds">
      <h3>Parámetros de Búsqueda</h3>

      <div class="form-group">
        <label for="dateFrom">Fecha desde:</label>
        <input
          id="dateFrom"
          type="date"
          v-model="dateFrom"
          :disabled="store.isLoading"
        />
      </div>

      <div class="form-group">
        <label for="dateTo">Fecha hasta:</label>
        <input
          id="dateTo"
          type="date"
          v-model="dateTo"
          :disabled="store.isLoading"
        />
      </div>

      <button
        @click="runAnalysis"
        class="btn btn-primary"
        :disabled="!store.selectedBounds || store.isLoading || (store.sizeWarning && !store.sizeWarning.canAnalyze)"
      >
        {{ store.isLoading ? 'Analizando...' : 'Analizar Región' }}
      </button>
    </div>

    <div class="panel-section" v-if="store.hasError" >
      <div class="alert alert-error">
        <h4>Error</h4>
        <p>{{ store.error }}</p>
        <button @click="store.clearError" class="btn btn-small">Cerrar</button>
      </div>
    </div>

    <div class="panel-section" v-if="store.hasResults">
      <h3>Resultados</h3>
      <ResultsPanel />
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useAnalysisStore } from '../../stores/analysis.store'
import ResultsPanel from './ResultsPanel.vue'

const store = useAnalysisStore()
const dateFrom = ref('')
const dateTo = ref('')

// Valores por defecto (últimos 30 días)
const today = new Date()
const thirtyDaysAgo = new Date(today)
thirtyDaysAgo.setDate(today.getDate() - 30)

dateTo.value = today.toISOString().split('T')[0]
dateFrom.value = thirtyDaysAgo.toISOString().split('T')[0]

const bounds = computed(() => {
  if (!store.selectedBounds) return null
  return {
    south: store.selectedBounds.getSouth(),
    north: store.selectedBounds.getNorth(),
    west: store.selectedBounds.getWest(),
    east: store.selectedBounds.getEast()
  }
})

async function runAnalysis() {
  await store.analyzeRegion(dateFrom.value, dateTo.value)
}
</script>

<style scoped>
.analysis-panel {
  padding: 20px;
  height: 100%;
  overflow-y: auto;
  background-color: #f8f9fa;
}

h2 {
  margin-top: 0;
  color: #333;
  font-size: 24px;
}

h3 {
  color: #555;
  font-size: 18px;
  margin-bottom: 15px;
}

.panel-section {
  background: white;
  padding: 20px;
  margin-bottom: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.bounds-info p {
  margin: 8px 0;
  color: #666;
}

.no-selection {
  text-align: center;
  color: #999;
}

.no-selection .hint {
  font-size: 14px;
  margin-top: 10px;
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block;
  margin-bottom: 5px;
  font-weight: 500;
  color: #555;
}

.form-group input[type="date"] {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.form-group input:disabled {
  background-color: #f5f5f5;
  cursor: not-allowed;
}

.btn {
  padding: 12px 24px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  font-weight: 500;
  transition: all 0.3s;
  width: 100%;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background-color: #28a745;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background-color: #218838;
}

.btn-small {
  padding: 6px 12px;
  font-size: 14px;
  width: auto;
}

.alert {
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 15px;
}

.alert h4 {
  margin: 0 0 10px 0;
}

.alert p {
  margin: 0 0 10px 0;
}

.alert-error {
  background-color: #f8d7da;
  border: 1px solid #f5c6cb;
  color: #721c24;
}

/* Size Warning Styles */
.size-warning {
  display: flex;
  gap: 12px;
  padding: 15px;
  border-radius: 6px;
  margin-top: 12px;
  margin-bottom: 12px;
  border-left: 4px solid;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.warning-error {
  background-color: #fee;
  border-left-color: #dc3545;
  color: #721c24;
}

.warning-warning {
  background-color: #fff8e1;
  border-left-color: #ffc107;
  color: #856404;
}

.size-warning svg {
  flex-shrink: 0;
  margin-top: 2px;
}

.warning-error svg {
  stroke: #dc3545;
}

.warning-warning svg {
  stroke: #ffc107;
}

.warning-content {
  flex: 1;
}

.warning-content p {
  margin: 0 0 8px 0;
  font-size: 14px;
  line-height: 1.5;
}

.warning-content p:last-child {
  margin-bottom: 0;
}

.warning-action {
  font-weight: 600;
  margin-top: 10px !important;
  padding: 8px 12px;
  background-color: rgba(0, 0, 0, 0.05);
  border-radius: 4px;
}
</style>
