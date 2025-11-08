<template>
  <div class="app-layout">
    <header class="app-header">
      <h1>Sistema de Detección de Estrés Vegetal</h1>
      <p class="subtitle">Análisis mediante Region Growing sobre imágenes Sentinel-2</p>
    </header>

    <div class="app-container">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.app-layout {
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
</style>
