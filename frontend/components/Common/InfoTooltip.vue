<template>
  <div class="info-tooltip-container">
    <button
      class="info-icon"
      @mouseenter="showTooltip = true"
      @mouseleave="showTooltip = false"
      @click="showTooltip = !showTooltip"
      :aria-label="title ? `Información sobre ${title}` : 'Más información'"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
      </svg>
    </button>

    <Transition name="tooltip-fade">
      <div v-if="showTooltip" class="tooltip-content" :class="position">
        <h4 v-if="title">{{ title }}</h4>
        <p>{{ text }}</p>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
withDefaults(
  defineProps<{
    text: string
    title?: string
    position?: 'top' | 'bottom' | 'left' | 'right'
  }>(),
  {
    title: '',
    position: 'top'
  }
)

const showTooltip = ref(false)
</script>

<style scoped>
.info-tooltip-container {
  display: inline-block;
  position: relative;
  margin-left: 4px;
}

.info-icon {
  background: none;
  border: none;
  cursor: help;
  padding: 0;
  color: #6B7280;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: color 0.2s;
  vertical-align: middle;
}

.info-icon:hover {
  color: #3B82F6;
}

.tooltip-content {
  position: absolute;
  background: white;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 12px 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  z-index: 1000;
  min-width: 200px;
  max-width: 300px;
  font-size: 13px;
  line-height: 1.5;
  color: #374151;
  pointer-events: none;
}

.tooltip-content.top {
  bottom: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
}

.tooltip-content.bottom {
  top: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
}

.tooltip-content.left {
  right: calc(100% + 8px);
  top: 50%;
  transform: translateY(-50%);
}

.tooltip-content.right {
  left: calc(100% + 8px);
  top: 50%;
  transform: translateY(-50%);
}

.tooltip-content h4 {
  margin: 0 0 6px 0;
  font-size: 14px;
  font-weight: 600;
  color: #111827;
}

.tooltip-content p {
  margin: 0;
  color: #4B5563;
}

.tooltip-content::after {
  content: '';
  position: absolute;
  width: 0;
  height: 0;
  border-style: solid;
}

.tooltip-content.top::after {
  bottom: -6px;
  left: 50%;
  transform: translateX(-50%);
  border-width: 6px 6px 0 6px;
  border-color: white transparent transparent transparent;
}

.tooltip-content.bottom::after {
  top: -6px;
  left: 50%;
  transform: translateX(-50%);
  border-width: 0 6px 6px 6px;
  border-color: transparent transparent white transparent;
}

.tooltip-content.left::after {
  right: -6px;
  top: 50%;
  transform: translateY(-50%);
  border-width: 6px 0 6px 6px;
  border-color: transparent transparent transparent white;
}

.tooltip-content.right::after {
  left: -6px;
  top: 50%;
  transform: translateY(-50%);
  border-width: 6px 6px 6px 0;
  border-color: transparent white transparent transparent;
}

.tooltip-fade-enter-active,
.tooltip-fade-leave-active {
  transition: opacity 0.2s ease;
}

.tooltip-fade-enter-from,
.tooltip-fade-leave-to {
  opacity: 0;
}

@media (max-width: 640px) {
  .tooltip-content {
    min-width: 180px;
    max-width: 85vw;
  }
}
</style>
