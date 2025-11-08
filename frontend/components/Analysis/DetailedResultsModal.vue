<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div class="modal-overlay" @click="$emit('close')">
        <div class="modal-container" @click.stop>
          <div class="modal-header">
            <h2>Análisis Detallado de Estrés Vegetal</h2>
            <button class="close-button" @click="$emit('close')" aria-label="Cerrar modal">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>

          <div class="modal-body">
            <div class="tabs">
              <button
                v-for="tab in tabs"
                :key="tab.id"
                :class="['tab', { active: activeTab === tab.id }]"
                @click="activeTab = tab.id"
              >
                {{ tab.label }}
              </button>
            </div>

            <div v-show="activeTab === 'visual'" class="tab-content">
              <div class="help-box">
                <h3>📊 Comparación Visual</h3>
                <p>Aquí puedes comparar la imagen satelital real con el análisis NDVI y las regiones detectadas.</p>
              </div>

              <div class="visual-comparison">
                <div class="comparison-item">
                  <h4>1. Imagen Satelital Real (RGB)</h4>
                  <div class="image-container">
                    <img
                      v-if="rgbImageSrc"
                      :src="rgbImageSrc"
                      alt="Imagen satelital RGB"
                      class="comparison-image"
                    />
                    <p v-else class="no-image">No hay imagen RGB disponible</p>
                  </div>
                  <p class="image-caption">
                    🛰️ <strong>Satélite:</strong> Sentinel-2 (ESA)<br>
                    📅 <strong>Fecha:</strong> {{ results.statistics?.date_from }} a {{ results.statistics?.date_to }}<br>
                    ☁️ <strong>Nubes:</strong> {{ results.statistics?.cloud_coverage?.toFixed(1) }}%
                  </p>
                </div>

                <div class="comparison-item">
                  <h4>2. Falso Color (NIR-Red-Green)</h4>
                  <div class="image-container">
                    <img
                      v-if="falseColorImageSrc"
                      :src="falseColorImageSrc"
                      alt="Imagen de falso color"
                      class="comparison-image"
                    />
                    <p v-else class="no-image">No hay imagen de falso color disponible</p>
                  </div>
                  <p class="image-caption">
                    🌿 La vegetación aparece en tonos rojos/rosados<br>
                    💧 El agua aparece en tonos azules/negros<br>
                    🏔️ El suelo aparece en tonos marrones/grises
                  </p>
                </div>

                <div class="comparison-item">
                  <h4>3. Mapa de Salud Vegetal (NDVI)</h4>
                  <div class="image-container">
                    <img
                      v-if="ndviImageSrc"
                      :src="ndviImageSrc"
                      alt="Mapa NDVI"
                      class="comparison-image"
                    />
                    <p v-else class="no-image">No hay mapa NDVI disponible</p>
                  </div>
                  <div class="ndvi-legend">
                    <div class="legend-item">
                      <span class="legend-color" style="background: #d73027;"></span>
                      <span>Estrés Alto / Suelo</span>
                    </div>
                    <div class="legend-item">
                      <span class="legend-color" style="background: #fee08b;"></span>
                      <span>Estrés Moderado</span>
                    </div>
                    <div class="legend-item">
                      <span class="legend-color" style="background: #1a9850;"></span>
                      <span>Vegetación Saludable</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div v-show="activeTab === 'stats'" class="tab-content">
              <div class="help-box">
                <h3>📈 Estadísticas del Análisis</h3>
                <p>Resumen numérico de lo encontrado en el área analizada.</p>
              </div>

              <div class="stats-grid">
                <div class="stat-card">
                  <div class="stat-header">
                    <h4>NDVI Promedio</h4>
                    <span :class="['ndvi-badge', getNdviClass(results.statistics?.mean_ndvi)]">
                      {{ results.statistics?.mean_ndvi?.toFixed(3) }}
                    </span>
                  </div>
                  <p class="stat-interpretation">
                    {{ interpretNdvi(results.statistics?.mean_ndvi) }}
                  </p>
                </div>

                <div class="stat-card">
                  <h4>Área Total Analizada</h4>
                  <div class="stat-value">{{ results.statistics?.total_area?.toFixed(2) }} ha</div>
                  <p class="stat-note">{{ (results.statistics?.total_area * 10000).toFixed(0) }} m² aprox.</p>
                </div>

                <div class="stat-card">
                  <h4>Cobertura de Nubes</h4>
                  <div class="stat-value" :class="getCloudClass(results.statistics?.cloud_coverage)">
                    {{ results.statistics?.cloud_coverage?.toFixed(1) || 0 }}%
                  </div>
                  <p class="stat-note">{{ getCloudInterpretation(results.statistics?.cloud_coverage) }}</p>
                </div>

                <div class="stat-card full-width">
                  <h4>Distribución por Nivel de Estrés</h4>
                  <div class="stress-distribution">
                    <div class="stress-bar-container">
                      <div
                        class="stress-bar high"
                        :style="{ width: getPercentage(results.statistics?.high_stress_area, results.statistics?.total_area) + '%' }"
                      >
                        <span v-if="getPercentage(results.statistics?.high_stress_area, results.statistics?.total_area) > 10">
                          {{ getPercentage(results.statistics?.high_stress_area, results.statistics?.total_area).toFixed(0) }}%
                        </span>
                      </div>
                      <div
                        class="stress-bar medium"
                        :style="{ width: getPercentage(results.statistics?.medium_stress_area, results.statistics?.total_area) + '%' }"
                      >
                        <span v-if="getPercentage(results.statistics?.medium_stress_area, results.statistics?.total_area) > 10">
                          {{ getPercentage(results.statistics?.medium_stress_area, results.statistics?.total_area).toFixed(0) }}%
                        </span>
                      </div>
                      <div
                        class="stress-bar low"
                        :style="{ width: getPercentage(results.statistics?.low_stress_area, results.statistics?.total_area) + '%' }"
                      >
                        <span v-if="getPercentage(results.statistics?.low_stress_area, results.statistics?.total_area) > 10">
                          {{ getPercentage(results.statistics?.low_stress_area, results.statistics?.total_area).toFixed(0) }}%
                        </span>
                      </div>
                    </div>

                    <div class="stress-details">
                      <div class="stress-detail high">
                        <span class="stress-color"></span>
                        <span class="stress-label">Estrés Alto:</span>
                        <span class="stress-value">{{ results.statistics?.high_stress_area?.toFixed(2) }} ha ({{ getPercentage(results.statistics?.high_stress_area, results.statistics?.total_area).toFixed(1) }}%)</span>
                      </div>
                      <div class="stress-detail medium">
                        <span class="stress-color"></span>
                        <span class="stress-label">Estrés Medio:</span>
                        <span class="stress-value">{{ results.statistics?.medium_stress_area?.toFixed(2) }} ha ({{ getPercentage(results.statistics?.medium_stress_area, results.statistics?.total_area).toFixed(1) }}%)</span>
                      </div>
                      <div class="stress-detail low">
                        <span class="stress-color"></span>
                        <span class="stress-label">Estrés Bajo:</span>
                        <span class="stress-value">{{ results.statistics?.low_stress_area?.toFixed(2) }} ha ({{ getPercentage(results.statistics?.low_stress_area, results.statistics?.total_area).toFixed(1) }}%)</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div class="stat-card">
                  <h4>Regiones Detectadas</h4>
                  <div class="stat-value">{{ results.statistics?.num_regions }}</div>
                  <div class="regions-breakdown">
                    <p>🔴 Alto: {{ results.statistics?.num_high_stress_regions }}</p>
                    <p>🟡 Medio: {{ results.statistics?.num_medium_stress_regions }}</p>
                    <p>🟢 Bajo: {{ results.statistics?.num_low_stress_regions }}</p>
                  </div>
                </div>
              </div>
            </div>

            <div v-show="activeTab === 'regions'" class="tab-content">
              <div class="help-box">
                <h3>📋 Tabla de Regiones</h3>
                <p>Detalle de cada región detectada por el algoritmo Region Growing.</p>
              </div>

              <div class="regions-table-container">
                <table class="regions-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Nivel de Estrés</th>
                      <th>NDVI Promedio</th>
                      <th>Área (ha)</th>
                      <th>Área (m²)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="region in sortedRegions"
                      :key="region.id"
                      :class="`stress-${region.stress_level}`"
                    >
                      <td class="region-id">{{ region.id }}</td>
                      <td class="region-stress">
                        <span :class="`stress-badge stress-${region.stress_level}`">
                          {{ getStressIcon(region.stress_level) }}
                          {{ getStressLabel(region.stress_level) }}
                        </span>
                      </td>
                      <td class="region-ndvi">{{ region.ndvi_mean.toFixed(3) }}</td>
                      <td class="region-area">{{ region.area.toFixed(2) }}</td>
                      <td class="region-area-m2">{{ (region.area * 10000).toFixed(0) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div class="modal-footer">
            <p class="footer-note">
              💡 <strong>Tip:</strong> Usa las pestañas para navegar entre las diferentes vistas
            </p>
            <button @click="$emit('close')" class="button-primary">Cerrar</button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import type { AnalysisResult } from '~/types'

const props = defineProps<{
  results: AnalysisResult
}>()

defineEmits<{
  close: []
}>()

const activeTab = ref('visual')

const tabs = [
  { id: 'visual', label: '📊 Comparación Visual' },
  { id: 'stats', label: '📈 Estadísticas' },
  { id: 'regions', label: '📋 Tabla de Regiones' }
]

const rgbImageSrc = computed(() => {
  if (!props.results?.images?.rgb) return null
  return `data:image/png;base64,${props.results.images.rgb}`
})

const falseColorImageSrc = computed(() => {
  if (!props.results?.images?.false_color) return null
  return `data:image/png;base64,${props.results.images.false_color}`
})

const ndviImageSrc = computed(() => {
  if (!props.results?.images?.ndvi) return null
  return `data:image/png;base64,${props.results.images.ndvi}`
})

const sortedRegions = computed(() => {
  if (!props.results?.regions) return []
  const stressOrder = { high: 0, medium: 1, low: 2 }
  return [...props.results.regions].sort((a, b) => {
    if (stressOrder[a.stress_level] !== stressOrder[b.stress_level]) {
      return stressOrder[a.stress_level] - stressOrder[b.stress_level]
    }
    return b.area - a.area
  })
})

const getStressIcon = (level: string) => {
  const icons = { high: '🔴', medium: '🟡', low: '🟢' }
  return icons[level as keyof typeof icons] || '⚪'
}

const getStressLabel = (level: string) => {
  const labels = { high: 'Alto', medium: 'Medio', low: 'Bajo' }
  return labels[level as keyof typeof labels] || level
}

const getPercentage = (value: number | undefined, total: number | undefined) => {
  if (!total || total === 0 || !value) return 0
  return (value / total) * 100
}

const getNdviClass = (ndvi: number | undefined) => {
  if (!ndvi) return 'high-stress'
  if (ndvi < 0.3) return 'high-stress'
  if (ndvi < 0.5) return 'medium-stress'
  return 'low-stress'
}

const interpretNdvi = (ndvi: number | undefined) => {
  if (!ndvi) return 'Sin datos'
  if (ndvi < 0.2) return '⚠️ Sin vegetación o muy estresada'
  if (ndvi < 0.3) return '🔴 Vegetación con estrés alto'
  if (ndvi < 0.5) return '🟡 Vegetación con estrés moderado'
  if (ndvi < 0.7) return '🟢 Vegetación saludable'
  return '✅ Vegetación muy saludable y densa'
}

const getCloudClass = (coverage: number | undefined) => {
  if (!coverage) return 'cloud-low'
  if (coverage > 50) return 'cloud-high'
  if (coverage > 30) return 'cloud-medium'
  return 'cloud-low'
}

const getCloudInterpretation = (coverage: number | undefined) => {
  if (!coverage) return '✓ Cielo despejado - análisis muy confiable'
  if (coverage > 50) return '⚠️ Muchas nubes - análisis puede ser poco confiable'
  if (coverage > 30) return '⚠️ Nubes moderadas - considerar repetir análisis'
  if (coverage > 10) return '✓ Pocas nubes - análisis confiable'
  return '✓ Cielo despejado - análisis muy confiable'
}

onMounted(() => {
  console.log('DetailedResultsModal - Results:', props.results)
  console.log('DetailedResultsModal - Images:', props.results?.images)
  console.log('DetailedResultsModal - RGB exists:', !!props.results?.images?.rgb)
  console.log('DetailedResultsModal - NDVI exists:', !!props.results?.images?.ndvi)
})
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  padding: 20px;
  overflow-y: auto;
}

.modal-container {
  background: white;
  border-radius: 12px;
  width: 100%;
  max-width: 1400px;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px 32px;
  border-bottom: 1px solid #E5E7EB;
}

.modal-header h2 {
  margin: 0;
  font-size: 24px;
  color: #111827;
}

.close-button {
  background: none;
  border: none;
  cursor: pointer;
  padding: 8px;
  color: #6B7280;
  border-radius: 6px;
  transition: all 0.2s;
}

.close-button:hover {
  background: #F3F4F6;
  color: #111827;
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 24px 32px;
}

.tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 24px;
  border-bottom: 2px solid #E5E7EB;
  overflow-x: auto;
}

.tab {
  background: none;
  border: none;
  padding: 12px 20px;
  font-size: 15px;
  font-weight: 500;
  color: #6B7280;
  cursor: pointer;
  border-bottom: 3px solid transparent;
  transition: all 0.2s;
  white-space: nowrap;
}

.tab:hover {
  color: #111827;
}

.tab.active {
  color: #3B82F6;
  border-bottom-color: #3B82F6;
}

.help-box {
  background: #EFF6FF;
  border: 1px solid #BFDBFE;
  border-radius: 8px;
  padding: 16px 20px;
  margin-bottom: 24px;
}

.help-box h3 {
  margin: 0 0 8px 0;
  font-size: 16px;
  color: #1E40AF;
}

.help-box p {
  margin: 0;
  color: #1E3A8A;
  font-size: 14px;
}

.visual-comparison {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 24px;
}

.comparison-item h4 {
  font-size: 16px;
  margin: 0 0 12px 0;
  color: #111827;
}

.image-container {
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 12px;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.comparison-image {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
}

.no-image {
  color: #6B7280;
  font-style: italic;
}

.image-caption {
  margin-top: 12px;
  font-size: 13px;
  color: #4B5563;
  line-height: 1.6;
}

.ndvi-legend {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #374151;
}

.legend-color {
  width: 30px;
  height: 18px;
  border-radius: 4px;
  border: 1px solid #D1D5DB;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
}

.stat-card {
  background: white;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 20px;
}

.stat-card.full-width {
  grid-column: 1 / -1;
}

.stat-card h4 {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: #6B7280;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stat-value {
  font-size: 32px;
  font-weight: 700;
  color: #111827;
  margin-bottom: 8px;
}

.stat-note {
  font-size: 13px;
  color: #6B7280;
  margin: 0;
}

.stat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.ndvi-badge {
  font-size: 20px;
  font-weight: 700;
  padding: 6px 12px;
  border-radius: 6px;
}

.ndvi-badge.high-stress {
  background: #FEE2E2;
  color: #991B1B;
}

.ndvi-badge.medium-stress {
  background: #FEF3C7;
  color: #92400E;
}

.ndvi-badge.low-stress {
  background: #D1FAE5;
  color: #065F46;
}

.stat-interpretation {
  font-size: 14px;
  color: #4B5563;
  margin: 0;
}

.stress-distribution {
  margin-top: 12px;
}

.stress-bar-container {
  display: flex;
  height: 40px;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 16px;
}

.stress-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 600;
  font-size: 14px;
  transition: width 0.3s;
}

.stress-bar.high {
  background-color: #dc3545;
}

.stress-bar.medium {
  background-color: #ffc107;
}

.stress-bar.low {
  background-color: #28a745;
}

.stress-details {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.stress-detail {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  background: #F9FAFB;
  border-radius: 6px;
}

.stress-color {
  width: 20px;
  height: 20px;
  border-radius: 4px;
}

.stress-detail.high .stress-color {
  background-color: #dc3545;
}

.stress-detail.medium .stress-color {
  background-color: #ffc107;
}

.stress-detail.low .stress-color {
  background-color: #28a745;
}

.stress-label {
  font-weight: 600;
  color: #374151;
}

.stress-value {
  margin-left: auto;
  color: #111827;
  font-weight: 600;
}

.regions-breakdown {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.regions-breakdown p {
  margin: 0;
  font-size: 14px;
  color: #4B5563;
}

.regions-table-container {
  overflow-x: auto;
  margin-bottom: 32px;
}

.regions-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.regions-table thead {
  background-color: #F9FAFB;
  border-bottom: 2px solid #E5E7EB;
}

.regions-table th {
  padding: 12px;
  text-align: left;
  font-weight: 600;
  color: #374151;
  text-transform: uppercase;
  font-size: 12px;
  letter-spacing: 0.5px;
}

.regions-table tbody tr {
  border-bottom: 1px solid #F3F4F6;
  transition: background-color 0.2s;
}

.regions-table tbody tr:hover {
  background-color: #F9FAFB;
}

.regions-table tbody tr.stress-high {
  border-left: 4px solid #dc3545;
}

.regions-table tbody tr.stress-medium {
  border-left: 4px solid #ffc107;
}

.regions-table tbody tr.stress-low {
  border-left: 4px solid #28a745;
}

.regions-table td {
  padding: 12px;
  color: #4B5563;
}

.region-id {
  font-weight: 600;
  color: #111827;
}

.stress-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 500;
}

.stress-badge.stress-high {
  background-color: #FEE2E2;
  color: #991B1B;
}

.stress-badge.stress-medium {
  background-color: #FEF3C7;
  color: #92400E;
}

.stress-badge.stress-low {
  background-color: #D1FAE5;
  color: #065F46;
}

.region-ndvi {
  font-family: 'Courier New', monospace;
  font-weight: 600;
}

.region-area,
.region-area-m2 {
  text-align: right;
}

.modal-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 32px;
  border-top: 1px solid #E5E7EB;
}

.footer-note {
  margin: 0;
  font-size: 13px;
  color: #6B7280;
}

.button-primary {
  background: #3B82F6;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 15px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s;
}

.button-primary:hover {
  background: #2563EB;
}

.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.3s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

@media (max-width: 768px) {
  .modal-container {
    max-height: 95vh;
  }

  .modal-header {
    padding: 20px;
  }

  .modal-header h2 {
    font-size: 20px;
  }

  .modal-body {
    padding: 20px;
  }

  .visual-comparison {
    grid-template-columns: 1fr;
  }

  .stats-grid {
    grid-template-columns: 1fr;
  }

  .modal-footer {
    flex-direction: column;
    gap: 12px;
    padding: 16px 20px;
  }

  .footer-note {
    text-align: center;
  }
}
</style>
