<template>
  <div class="map-controls">
    <button
      v-if="!isDrawing"
      @click="$emit('start-draw')"
      class="btn btn-draw"
      :disabled="isLoading || hasSelection"
      :title="hasSelection ? 'Ya hay una región seleccionada' : 'Dibujar polígono en el mapa'"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
        <polyline points="2 17 12 22 22 17"></polyline>
        <polyline points="2 12 12 17 22 12"></polyline>
      </svg>
      Seleccionar Área
    </button>

    <button
      v-if="isDrawing"
      @click="$emit('stop-draw')"
      class="btn btn-cancel"
      :disabled="isLoading"
      title="Cancelar dibujo"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="18" y1="6" x2="6" y2="18"></line>
        <line x1="6" y1="6" x2="18" y2="18"></line>
      </svg>
      Cancelar
    </button>

    <button
      v-if="!isDrawing"
      @click="$emit('clear-selection')"
      class="btn btn-clear"
      :disabled="isLoading || !hasSelection"
      :title="!hasSelection ? 'No hay región seleccionada' : 'Limpiar región y resultados'"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="3 6 5 6 21 6"></polyline>
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
      </svg>
      Limpiar
    </button>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  isDrawing: boolean
  hasSelection: boolean
  isLoading: boolean
}>()

defineEmits<{
  'start-draw': []
  'stop-draw': []
  'clear-selection': []
}>()
</script>

<style scoped>
.map-controls {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 1000;
  display: flex;
  gap: 10px;
  flex-direction: column;
}

.btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  min-width: 160px;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn:not(:disabled):hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

.btn-draw {
  background-color: #3b82f6;
  color: white;
}

.btn-draw:hover:not(:disabled) {
  background-color: #2563eb;
}

.btn-cancel {
  background-color: #ef4444;
  color: white;
}

.btn-cancel:hover:not(:disabled) {
  background-color: #dc2626;
}

.btn-clear {
  background-color: #6b7280;
  color: white;
}

.btn-clear:hover:not(:disabled) {
  background-color: #4b5563;
}
</style>
