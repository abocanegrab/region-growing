export default defineNuxtPlugin(async () => {
  // This plugin only runs on client due to .client.ts suffix
  const maplibreModule = await import('maplibre-gl')
  const mapboxDrawModule = await import('@mapbox/mapbox-gl-draw')

  if (typeof window !== 'undefined') {
    ;(window as any).maplibregl = maplibreModule.default
    ;(window as any).MapboxDraw = mapboxDrawModule.default
  }
})
