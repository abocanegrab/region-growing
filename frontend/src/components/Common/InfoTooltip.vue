<template>
  <div class="info-tooltip-container">
    <button
      class="info-icon"
      @mouseenter="showTooltip = true"
      @mouseleave="showTooltip = false"
      @click="showTooltip = !showTooltip"
      :aria-label="`Información sobre ${title}`"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
      </svg>
    </button>

    <Transition name="tooltip-fade">
      <div v-if="showTooltip" class="tooltip-content" :class="position">
        <h4 v-if="title">{{ title }}</h4>
        <p>{{ content }}</p>
        <div v-if="learnMoreUrl" class="learn-more">
          <a :href="learnMoreUrl" target="_blank" rel="noopener">
            Aprender más →
          </a>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  title: {
    type: String,
    default: ''
  },
  content: {
    type: String,
    required: true
  },
  position: {
    type: String,
    default: 'top',
    validator: (value) => ['top', 'bottom', 'left', 'right'].includes(value)
  },
  learnMoreUrl: {
    type: String,
    default: ''
  }
})

const showTooltip = ref(false)
</script>

<style scoped>
.info-tooltip-container {
  display: inline-block;
  position: relative;
  margin-left: 6px;
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
  min-width: 250px;
  max-width: 400px;
  font-size: 14px;
  line-height: 1.5;
  color: #374151;
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
  margin: 0 0 8px 0;
  font-size: 15px;
  font-weight: 600;
  color: #111827;
}

.tooltip-content p {
  margin: 0;
  color: #4B5563;
}

.learn-more {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #E5E7EB;
}

.learn-more a {
  color: #3B82F6;
  text-decoration: none;
  font-size: 13px;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.learn-more a:hover {
  color: #2563EB;
  text-decoration: underline;
}

/* Tooltip arrow */
.tooltip-content::after {
  content: '';
  position: absolute;
  width: 0;
  height: 0;
  border-style: solid;
}

.tooltip-content.top::after {
  bottom: -8px;
  left: 50%;
  transform: translateX(-50%);
  border-width: 8px 8px 0 8px;
  border-color: white transparent transparent transparent;
}

.tooltip-content.bottom::after {
  top: -8px;
  left: 50%;
  transform: translateX(-50%);
  border-width: 0 8px 8px 8px;
  border-color: transparent transparent white transparent;
}

/* Animations */
.tooltip-fade-enter-active,
.tooltip-fade-leave-active {
  transition: opacity 0.2s ease;
}

.tooltip-fade-enter-from,
.tooltip-fade-leave-to {
  opacity: 0;
}

/* Responsive */
@media (max-width: 640px) {
  .tooltip-content {
    min-width: 200px;
    max-width: 90vw;
  }
}
</style>
