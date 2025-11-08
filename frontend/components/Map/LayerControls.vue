<template>
  <div class="layer-controls">
    <div class="layer-controls-header">
      <h4>Capas</h4>
      <button
        class="toggle-button"
        @click="isExpanded = !isExpanded"
        :aria-label="isExpanded ? 'Contraer' : 'Expandir'"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline :points="isExpanded ? '18 15 12 9 6 15' : '6 9 12 15 18 9'"></polyline>
        </svg>
      </button>
    </div>

    <Transition name="expand">
      <div v-show="isExpanded" class="layer-controls-body">
        <div class="layer-item">
          <label class="layer-label">
            <input
              type="checkbox"
              v-model="layers.rgb"
              @change="$emit('toggle-layer', 'raster-rgb', layers.rgb)"
              :disabled="!hasImages"
            />
            <span class="layer-name">Imagen RGB</span>
          </label>
        </div>

        <div class="layer-item">
          <label class="layer-label">
            <input
              type="checkbox"
              v-model="layers.falseColor"
              @change="$emit('toggle-layer', 'raster-false-color', layers.falseColor)"
              :disabled="!hasImages"
            />
            <span class="layer-name">Falso Color (NIR-R-G)</span>
          </label>
        </div>

        <div class="layer-item">
          <label class="layer-label">
            <input
              type="checkbox"
              v-model="layers.ndvi"
              @change="$emit('toggle-layer', 'raster-ndvi', layers.ndvi)"
              :disabled="!hasImages"
            />
            <span class="layer-name">Mapa NDVI</span>
          </label>
        </div>

        <div class="layer-item">
          <label class="layer-label">
            <input
              type="checkbox"
              v-model="layers.regions"
              @change="$emit('toggle-layer', 'results', layers.regions)"
              :disabled="!hasResults"
            />
            <span class="layer-name">Regiones Detectadas</span>
          </label>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  hasImages: boolean
  hasResults: boolean
}>()

defineEmits<{
  'toggle-layer': [layerId: string, visible: boolean]
}>()

const isExpanded = ref(true)

const layers = ref({
  rgb: false,
  falseColor: false,
  ndvi: false,
  regions: true
})
</script>

<style scoped>
.layer-controls {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 1000;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  min-width: 200px;
  max-width: 250px;
}

.layer-controls-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #e5e7eb;
}

.layer-controls-header h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: #374151;
}

.toggle-button {
  background: none;
  border: none;
  cursor: pointer;
  padding: 4px;
  color: #6b7280;
  border-radius: 4px;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.toggle-button:hover {
  background: #f3f4f6;
  color: #111827;
}

.layer-controls-body {
  padding: 8px;
}

.layer-item {
  padding: 8px;
  border-radius: 4px;
  transition: background 0.2s;
}

.layer-item:hover {
  background: #f9fafb;
}

.layer-label {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  user-select: none;
}

.layer-label input[type="checkbox"] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.layer-label input[type="checkbox"]:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.layer-name {
  font-size: 13px;
  color: #374151;
}

.layer-label:has(input:disabled) .layer-name {
  color: #9ca3af;
}

.expand-enter-active,
.expand-leave-active {
  transition: all 0.3s ease;
  overflow: hidden;
}

.expand-enter-from,
.expand-leave-to {
  max-height: 0;
  opacity: 0;
}

.expand-enter-to,
.expand-leave-from {
  max-height: 300px;
  opacity: 1;
}
</style>
