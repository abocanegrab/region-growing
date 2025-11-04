<template>
  <div id="app">
    <header class="app-header">
      <h1>Sistema de Detección de Estrés Vegetal</h1>
      <p class="subtitle">Análisis mediante Region Growing sobre imágenes Sentinel-2</p>
    </header>

    <div class="app-container">
      <aside class="sidebar">
        <AnalysisPanel />
      </aside>

      <main class="main-content">
        <MapView />
      </main>
    </div>
  </div>
</template>

<script setup>
import MapView from './components/Map/MapView.vue'
import AnalysisPanel from './components/Analysis/AnalysisPanel.vue'
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  overflow: hidden;
}

#app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
}

.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px 30px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.app-header h1 {
  font-size: 28px;
  margin-bottom: 5px;
}

.app-header .subtitle {
  font-size: 14px;
  opacity: 0.9;
}

.app-container {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar {
  width: 400px;
  background-color: #f8f9fa;
  border-right: 1px solid #dee2e6;
  overflow-y: auto;
}

.main-content {
  flex: 1;
  position: relative;
}
</style>
