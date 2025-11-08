<template>
  <div class="main-page">
    <aside class="sidebar">
      <AnalysisPanel />
    </aside>

    <main class="main-content">
      <ClientOnly>
        <MapLibreMap />
        <template #fallback>
          <div class="map-loading">
            <p>Cargando mapa...</p>
          </div>
        </template>
      </ClientOnly>
    </main>
  </div>
</template>

<style scoped>
.main-page {
  display: flex;
  width: 100%;
  height: 100%;
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

.map-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  background-color: #f0f0f0;
}
</style>
