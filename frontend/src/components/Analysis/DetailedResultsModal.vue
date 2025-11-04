<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div v-if="modelValue" class="modal-overlay" @click="closeModal">
        <div class="modal-container" @click.stop>
          <!-- Header -->
          <div class="modal-header">
            <h2>Análisis Detallado de Estrés Vegetal</h2>
            <button class="close-button" @click="closeModal" aria-label="Cerrar modal">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>

          <!-- Content -->
          <div class="modal-body">
            <!-- Tabs -->
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

            <!-- Tab: Comparación Visual -->
            <div v-show="activeTab === 'visual'" class="tab-content">
              <div class="help-box">
                <h3>📊 Comparación Visual</h3>
                <p>Aquí puedes comparar la imagen satelital real con el análisis NDVI y las regiones detectadas.</p>
              </div>

              <div class="visual-comparison">
                <!-- Imagen RGB -->
                <div class="comparison-item">
                  <h4>
                    1. Imagen Satelital Real
                    <InfoTooltip
                      title="Imagen Sentinel-2"
                      content="Esta es la imagen en color verdadero capturada por el satélite Sentinel-2 de la Agencia Espacial Europea (ESA). Muestra cómo se ve la zona desde el espacio en colores visibles para el ojo humano."
                      position="bottom"
                    />
                  </h4>
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
                    📅 <strong>Fecha:</strong> {{results.statistics.date_from}} a {{results.statistics.date_to}}<br>
                    ☁️ <strong>Nubes:</strong> {{results.statistics.cloud_coverage?.toFixed(1)}}%
                  </p>
                </div>

                <!-- Mapa NDVI -->
                <div class="comparison-item">
                  <h4>
                    2. Mapa de Salud Vegetal (NDVI)
                    <InfoTooltip
                      title="NDVI - Índice de Vegetación"
                      content="El NDVI mide qué tan saludable está la vegetación. Los colores rojos indican vegetación estresada o suelo desnudo, amarillos indican estrés moderado, y verdes indican vegetación saludable. Este mapa se calcula usando luz infrarroja que las plantas sanas reflejan fuertemente."
                      position="bottom"
                    />
                  </h4>
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

            <!-- Tab: Estadísticas -->
            <div v-show="activeTab === 'stats'" class="tab-content">
              <div class="help-box">
                <h3>📈 Estadísticas del Análisis</h3>
                <p>Resumen numérico de lo encontrado en el área analizada.</p>
              </div>

              <div class="stats-grid">
                <!-- NDVI Promedio -->
                <div class="stat-card">
                  <div class="stat-header">
                    <h4>
                      NDVI Promedio
                      <InfoTooltip
                        content="El NDVI (Normalized Difference Vegetation Index) es un número entre -1 y 1. Valores altos (cerca de 1) indican vegetación muy saludable. Valores bajos (cerca de 0 o negativos) indican suelo desnudo, agua o vegetación muy estresada."
                        position="right"
                      />
                    </h4>
                    <span :class="['ndvi-badge', getNdviClass(results.statistics.mean_ndvi)]">
                      {{ results.statistics.mean_ndvi?.toFixed(3) }}
                    </span>
                  </div>
                  <p class="stat-interpretation">
                    {{ interpretNdvi(results.statistics.mean_ndvi) }}
                  </p>
                </div>

                <!-- Área Total -->
                <div class="stat-card">
                  <h4>Área Total Analizada</h4>
                  <div class="stat-value">{{ results.statistics.total_area?.toFixed(2) }} ha</div>
                  <p class="stat-note">{{ (results.statistics.total_area * 10000).toFixed(0) }} m² aprox.</p>
                </div>

                <!-- Cobertura de Nubes -->
                <div class="stat-card">
                  <h4>
                    Cobertura de Nubes
                    <InfoTooltip
                      content="Porcentaje del área original cubierta por nubes o sombras. Las áreas con nubes (mostradas en gris en el mapa NDVI) no se incluyen en las estadísticas de vegetación. Valores >30% pueden reducir la confiabilidad del análisis."
                      position="right"
                    />
                  </h4>
                  <div class="stat-value" :class="getCloudClass(results.statistics.cloud_coverage)">
                    {{ results.statistics.cloud_coverage?.toFixed(1) || 0 }}%
                  </div>
                  <p class="stat-note">{{ getCloudInterpretation(results.statistics.cloud_coverage) }}</p>
                </div>

                <!-- Distribución por Estrés -->
                <div class="stat-card full-width">
                  <h4>
                    Distribución por Nivel de Estrés
                    <InfoTooltip
                      content="El área se divide en tres categorías según la salud de la vegetación. Estrés Alto significa que las plantas están muy estresadas o no hay vegetación. Estrés Medio indica problemas moderados. Estrés Bajo indica plantas saludables."
                      position="right"
                    />
                  </h4>
                  <div class="stress-distribution">
                    <div class="stress-bar-container">
                      <div
                        class="stress-bar high"
                        :style="{ width: getPercentage(results.statistics.high_stress_area, results.statistics.total_area) + '%' }"
                      >
                        <span v-if="getPercentage(results.statistics.high_stress_area, results.statistics.total_area) > 10">
                          {{ getPercentage(results.statistics.high_stress_area, results.statistics.total_area).toFixed(0) }}%
                        </span>
                      </div>
                      <div
                        class="stress-bar medium"
                        :style="{ width: getPercentage(results.statistics.medium_stress_area, results.statistics.total_area) + '%' }"
                      >
                        <span v-if="getPercentage(results.statistics.medium_stress_area, results.statistics.total_area) > 10">
                          {{ getPercentage(results.statistics.medium_stress_area, results.statistics.total_area).toFixed(0) }}%
                        </span>
                      </div>
                      <div
                        class="stress-bar low"
                        :style="{ width: getPercentage(results.statistics.low_stress_area, results.statistics.total_area) + '%' }"
                      >
                        <span v-if="getPercentage(results.statistics.low_stress_area, results.statistics.total_area) > 10">
                          {{ getPercentage(results.statistics.low_stress_area, results.statistics.total_area).toFixed(0) }}%
                        </span>
                      </div>
                    </div>

                    <div class="stress-details">
                      <div class="stress-detail high">
                        <span class="stress-color"></span>
                        <span class="stress-label">Estrés Alto:</span>
                        <span class="stress-value">{{ results.statistics.high_stress_area?.toFixed(2) }} ha ({{ getPercentage(results.statistics.high_stress_area, results.statistics.total_area).toFixed(1) }}%)</span>
                      </div>
                      <div class="stress-detail medium">
                        <span class="stress-color"></span>
                        <span class="stress-label">Estrés Medio:</span>
                        <span class="stress-value">{{ results.statistics.medium_stress_area?.toFixed(2) }} ha ({{ getPercentage(results.statistics.medium_stress_area, results.statistics.total_area).toFixed(1) }}%)</span>
                      </div>
                      <div class="stress-detail low">
                        <span class="stress-color"></span>
                        <span class="stress-label">Estrés Bajo:</span>
                        <span class="stress-value">{{ results.statistics.low_stress_area?.toFixed(2) }} ha ({{ getPercentage(results.statistics.low_stress_area, results.statistics.total_area).toFixed(1) }}%)</span>
                      </div>
                    </div>
                  </div>
                </div>

                <!-- Regiones Detectadas -->
                <div class="stat-card">
                  <h4>
                    Regiones Detectadas
                    <InfoTooltip
                      content="El algoritmo Region Growing agrupa píxeles similares en regiones. Cada región representa un área con características NDVI parecidas. Más regiones indican mayor variabilidad en la salud vegetal."
                      position="right"
                    />
                  </h4>
                  <div class="stat-value">{{ results.statistics.num_regions }}</div>
                  <div class="regions-breakdown">
                    <p>🔴 Alto: {{ results.statistics.num_high_stress_regions }}</p>
                    <p>🟡 Medio: {{ results.statistics.num_medium_stress_regions }}</p>
                    <p>🟢 Bajo: {{ results.statistics.num_low_stress_regions }}</p>
                  </div>
                </div>
              </div>
            </div>

            <!-- Tab: Guía de Interpretación -->
            <div v-show="activeTab === 'guide'" class="tab-content">
              <div class="help-box">
                <h3>📚 Guía para Entender los Resultados</h3>
                <p>Explicaciones simples para interpretar el análisis sin ser especialista.</p>
              </div>

              <div class="guide-section">
                <h3>¿Qué es el NDVI?</h3>
                <p>
                  El <strong>NDVI</strong> (Índice de Vegetación de Diferencia Normalizada) es como un "termómetro"
                  para la salud de las plantas. Funciona midiendo cuánta luz roja e infrarroja reflejan las hojas.
                </p>
                <ul>
                  <li><strong>Plantas sanas:</strong> Absorben luz roja (para fotosíntesis) y reflejan mucho infrarrojo → NDVI alto (0.5-1.0)</li>
                  <li><strong>Plantas estresadas:</strong> Reflejan más rojo y menos infrarrojo → NDVI medio (0.3-0.5)</li>
                  <li><strong>Suelo desnudo/agua:</strong> No hay vegetación significativa → NDVI bajo (-1.0-0.3)</li>
                </ul>
              </div>

              <div class="guide-section">
                <h3>¿Qué significa "Estrés Vegetal"?</h3>
                <p>
                  El estrés vegetal ocurre cuando las plantas no tienen las condiciones óptimas para crecer. Puede deberse a:
                </p>
                <ul>
                  <li>💧 <strong>Falta de agua:</strong> Las plantas se marchitan y reflejan menos infrarrojo</li>
                  <li>🦗 <strong>Plagas o enfermedades:</strong> Dañan las hojas y reducen la fotosíntesis</li>
                  <li>🌡️ <strong>Temperatura extrema:</strong> Calor o frío excesivo daña las células</li>
                  <li>🪨 <strong>Suelo pobre:</strong> Falta de nutrientes limita el crecimiento</li>
                </ul>
              </div>

              <div class="guide-section">
                <h3>¿Cómo uso esta información?</h3>
                <div class="use-cases">
                  <div class="use-case">
                    <h4>🌾 Si tienes cultivos agrícolas:</h4>
                    <ul>
                      <li>Áreas rojas (estrés alto) necesitan atención urgente</li>
                      <li>Considera aumentar riego en zonas estresadas</li>
                      <li>Inspecciona físicamente las áreas con problemas</li>
                      <li>Compara con análisis anteriores para ver tendencias</li>
                    </ul>
                  </div>

                  <div class="use-case">
                    <h4>🌳 Si monitoreas bosques o áreas naturales:</h4>
                    <ul>
                      <li>Identifica áreas con posible degradación</li>
                      <li>Detecta efectos de sequías o incendios</li>
                      <li>Planifica zonas de reforestación</li>
                      <li>Evalúa el impacto de cambio climático</li>
                    </ul>
                  </div>

                  <div class="use-case">
                    <h4>🏔️ Si analizas zonas montañosas:</h4>
                    <ul>
                      <li>Es normal tener mucho "estrés alto" (rocas, nieve, suelo)</li>
                      <li>Las áreas verdes indican vegetación de altura</li>
                      <li>Compara estaciones para ver variación anual</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div class="guide-section">
                <h3>Limitaciones a Considerar</h3>
                <ul>
                  <li>☁️ <strong>Nubes:</strong> Si hay muchas nubes (>30%), los resultados pueden ser imprecisos</li>
                  <li>📅 <strong>Estacionalidad:</strong> El NDVI varía según la época del año (invierno vs verano)</li>
                  <li>🔍 <strong>Resolución:</strong> 10 metros por píxel - parcelas muy pequeñas pueden no verse bien</li>
                  <li>⏱️ <strong>Actualización:</strong> Las imágenes tienen hasta 5 días de antigüedad</li>
                </ul>
              </div>
            </div>

            <!-- Tab: Exportar -->
            <div v-show="activeTab === 'export'" class="tab-content">
              <div class="help-box">
                <h3>💾 Exportar Resultados</h3>
                <p>Descarga los datos del análisis en diferentes formatos.</p>
              </div>

              <div class="export-options">
                <button @click="exportJSON" class="export-button">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                  </svg>
                  Descargar JSON Completo
                </button>

                <button @click="exportImages" class="export-button">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                    <polyline points="21 15 16 10 5 21"></polyline>
                  </svg>
                  Descargar Imágenes (RGB + NDVI)
                </button>

                <button @click="exportReport" class="export-button">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                  </svg>
                  Generar Informe en Texto
                </button>
              </div>
            </div>
          </div>

          <!-- Footer -->
          <div class="modal-footer">
            <p class="footer-note">
              💡 <strong>Tip:</strong> Puedes usar las teclas ← → para navegar entre pestañas
            </p>
            <button @click="closeModal" class="button-primary">Cerrar</button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import InfoTooltip from '../Common/InfoTooltip.vue'

const props = defineProps({
  modelValue: {
    type: Boolean,
    required: true
  },
  results: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['update:modelValue'])

const activeTab = ref('visual')

const tabs = [
  { id: 'visual', label: '📊 Comparación Visual' },
  { id: 'stats', label: '📈 Estadísticas' },
  { id: 'guide', label: '📚 Guía de Interpretación' },
  { id: 'export', label: '💾 Exportar' }
]

// Computed para asegurar reactividad correcta de las imágenes
const rgbImageSrc = computed(() => {
  if (!props.results?.images?.rgb) return null
  return `data:image/png;base64,${props.results.images.rgb}`
})

const ndviImageSrc = computed(() => {
  if (!props.results?.images?.ndvi) return null
  return `data:image/png;base64,${props.results.images.ndvi}`
})

const closeModal = () => {
  emit('update:modelValue', false)
}

// Helpers
const getPercentage = (value, total) => {
  if (!total || total === 0) return 0
  return (value / total) * 100
}

const getNdviClass = (ndvi) => {
  if (ndvi < 0.3) return 'high-stress'
  if (ndvi < 0.5) return 'medium-stress'
  return 'low-stress'
}

const interpretNdvi = (ndvi) => {
  if (ndvi < 0.2) return '⚠️ Sin vegetación o muy estresada'
  if (ndvi < 0.3) return '🔴 Vegetación con estrés alto'
  if (ndvi < 0.5) return '🟡 Vegetación con estrés moderado'
  if (ndvi < 0.7) return '🟢 Vegetación saludable'
  return '✅ Vegetación muy saludable y densa'
}

const getCloudClass = (coverage) => {
  if (coverage > 50) return 'cloud-high'
  if (coverage > 30) return 'cloud-medium'
  return 'cloud-low'
}

const getCloudInterpretation = (coverage) => {
  if (coverage > 50) return '⚠️ Muchas nubes - análisis puede ser poco confiable'
  if (coverage > 30) return '⚠️ Nubes moderadas - considerar repetir análisis'
  if (coverage > 10) return '✓ Pocas nubes - análisis confiable'
  return '✓ Cielo despejado - análisis muy confiable'
}

// Export functions
const exportJSON = () => {
  const dataStr = JSON.stringify(props.results, null, 2)
  const dataBlob = new Blob([dataStr], { type: 'application/json' })
  const url = URL.createObjectURL(dataBlob)
  const link = document.createElement('a')
  link.href = url
  link.download = `analysis-${new Date().toISOString().split('T')[0]}.json`
  link.click()
  URL.revokeObjectURL(url)
}

const exportImages = () => {
  // Download RGB
  if (props.results.images?.rgb) {
    const link = document.createElement('a')
    link.href = `data:image/png;base64,${props.results.images.rgb}`
    link.download = `satellite-rgb-${new Date().toISOString().split('T')[0]}.png`
    link.click()
  }

  // Download NDVI
  if (props.results.images?.ndvi) {
    setTimeout(() => {
      const link = document.createElement('a')
      link.href = `data:image/png;base64,${props.results.images.ndvi}`
      link.download = `ndvi-map-${new Date().toISOString().split('T')[0]}.png`
      link.click()
    }, 500)
  }
}

const exportReport = () => {
  const report = `
INFORME DE ANÁLISIS DE ESTRÉS VEGETAL
======================================

Fecha de análisis: ${new Date().toLocaleString()}
Período de imágenes: ${props.results.statistics.date_from} a ${props.results.statistics.date_to}
Cobertura de nubes: ${props.results.statistics.cloud_coverage?.toFixed(1)}%

RESUMEN GENERAL
---------------
NDVI Promedio: ${props.results.statistics.mean_ndvi?.toFixed(3)} - ${interpretNdvi(props.results.statistics.mean_ndvi)}
Área Total: ${props.results.statistics.total_area?.toFixed(2)} hectáreas

DISTRIBUCIÓN POR NIVEL DE ESTRÉS
---------------------------------
🔴 Estrés Alto: ${props.results.statistics.high_stress_area?.toFixed(2)} ha (${getPercentage(props.results.statistics.high_stress_area, props.results.statistics.total_area).toFixed(1)}%)
🟡 Estrés Medio: ${props.results.statistics.medium_stress_area?.toFixed(2)} ha (${getPercentage(props.results.statistics.medium_stress_area, props.results.statistics.total_area).toFixed(1)}%)
🟢 Estrés Bajo: ${props.results.statistics.low_stress_area?.toFixed(2)} ha (${getPercentage(props.results.statistics.low_stress_area, props.results.statistics.total_area).toFixed(1)}%)

REGIONES DETECTADAS
-------------------
Total de regiones: ${props.results.statistics.num_regions}
- Regiones con estrés alto: ${props.results.statistics.num_high_stress_regions}
- Regiones con estrés medio: ${props.results.statistics.num_medium_stress_regions}
- Regiones con estrés bajo: ${props.results.statistics.num_low_stress_regions}

INTERPRETACIÓN
--------------
${interpretNdvi(props.results.statistics.mean_ndvi)}

Este análisis fue generado usando imágenes satelitales Sentinel-2 y el algoritmo Region Growing.
Para más información, visite: https://sentinel.esa.int/
`.trim()

  const dataBlob = new Blob([report], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(dataBlob)
  const link = document.createElement('a')
  link.href = url
  link.download = `informe-${new Date().toISOString().split('T')[0]}.txt`
  link.click()
  URL.revokeObjectURL(url)
}

// Keyboard navigation
const handleKeyDown = (e) => {
  if (!props.modelValue) return

  if (e.key === 'Escape') {
    closeModal()
  } else if (e.key === 'ArrowLeft') {
    const currentIndex = tabs.findIndex(t => t.id === activeTab.value)
    if (currentIndex > 0) {
      activeTab.value = tabs[currentIndex - 1].id
    }
  } else if (e.key === 'ArrowRight') {
    const currentIndex = tabs.findIndex(t => t.id === activeTab.value)
    if (currentIndex < tabs.length - 1) {
      activeTab.value = tabs[currentIndex + 1].id
    }
  }
}

onMounted(() => {
  console.log('DetailedResultsModal - Results received:', props.results)
  console.log('DetailedResultsModal - Images:', props.results?.images)
  console.log('DetailedResultsModal - RGB exists:', !!props.results?.images?.rgb)
  console.log('DetailedResultsModal - NDVI exists:', !!props.results?.images?.ndvi)
  console.log('DetailedResultsModal - RGB image src:', rgbImageSrc.value)
  console.log('DetailedResultsModal - NDVI image src:', ndviImageSrc.value)
  console.log('DetailedResultsModal - RGB length:', props.results?.images?.rgb?.length)
  console.log('DetailedResultsModal - NDVI length:', props.results?.images?.ndvi?.length)
  document.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyDown)
})
</script>

<style scoped>
/* Modal Overlay */
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

/* Modal Container */
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

/* Header */
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

/* Body */
.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 24px 32px;
}

/* Tabs */
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

/* Help Box */
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

/* Visual Comparison */
.visual-comparison {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
  gap: 24px;
}

.comparison-item h4 {
  font-size: 16px;
  margin: 0 0 12px 0;
  color: #111827;
  display: flex;
  align-items: center;
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

/* Stats Grid */
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
  display: flex;
  align-items: center;
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

/* Stress Distribution */
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
  transition: width 0.3s ease;
}

.stress-bar.high {
  background: #EF4444;
}

.stress-bar.medium {
  background: #F59E0B;
}

.stress-bar.low {
  background: #10B981;
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
  font-size: 14px;
}

.stress-detail .stress-color {
  width: 20px;
  height: 20px;
  border-radius: 4px;
}

.stress-detail.high .stress-color {
  background: #EF4444;
}

.stress-detail.medium .stress-color {
  background: #F59E0B;
}

.stress-detail.low .stress-color {
  background: #10B981;
}

.stress-detail .stress-label {
  font-weight: 600;
  color: #374151;
  min-width: 100px;
}

.stress-detail .stress-value {
  color: #6B7280;
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

/* Guide Section */
.guide-section {
  margin-bottom: 32px;
}

.guide-section h3 {
  font-size: 18px;
  margin: 0 0 12px 0;
  color: #111827;
}

.guide-section p {
  font-size: 15px;
  line-height: 1.6;
  color: #374151;
  margin: 0 0 12px 0;
}

.guide-section ul {
  margin: 0;
  padding-left: 24px;
  line-height: 1.8;
  color: #4B5563;
}

.guide-section li {
  margin-bottom: 8px;
}

.use-cases {
  display: grid;
  gap: 16px;
}

.use-case {
  background: #F9FAFB;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 16px;
}

.use-case h4 {
  margin: 0 0 12px 0;
  font-size: 16px;
  color: #111827;
}

.use-case ul {
  margin: 0;
  padding-left: 24px;
}

/* Export Options */
.export-options {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-width: 500px;
}

.export-button {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 20px;
  background: white;
  border: 2px solid #E5E7EB;
  border-radius: 8px;
  font-size: 15px;
  font-weight: 500;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s;
}

.export-button:hover {
  border-color: #3B82F6;
  color: #3B82F6;
  background: #EFF6FF;
}

/* Footer */
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

/* Cloud Coverage Classes */
.cloud-high {
  color: #EF4444 !important;
  font-weight: 700;
}

.cloud-medium {
  color: #F59E0B !important;
  font-weight: 700;
}

.cloud-low {
  color: #10B981 !important;
  font-weight: 700;
}

/* Animations */
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.3s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

/* Responsive */
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

  .tabs {
    flex-wrap: nowrap;
    overflow-x: auto;
  }

  .tab {
    font-size: 14px;
    padding: 10px 16px;
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
