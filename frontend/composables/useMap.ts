import type { Map as MapLibreMap } from 'maplibre-gl'
import type MapboxDraw from '@mapbox/mapbox-gl-draw'

export const useMap = () => {
  const analysisStore = useAnalysisStore()

  const mapInstance = ref<MapLibreMap | null>(null)
  const drawInstance = ref<MapboxDraw | null>(null)

  const isDrawing = computed(() => analysisStore.isDrawing)
  const selectedBounds = computed(() => analysisStore.selectedBounds)

  const initMap = (container: string | HTMLElement, options?: any) => {
    if (process.client && !mapInstance.value) {
      const maplibregl = (window as any).maplibregl

      // Estilo OSM personalizado
      const osmStyle = {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '© OpenStreetMap contributors'
          }
        },
        layers: [
          {
            id: 'osm',
            type: 'raster',
            source: 'osm',
            minzoom: 0,
            maxzoom: 19
          }
        ]
      }

      mapInstance.value = new maplibregl.Map({
        container,
        style: osmStyle,
        center: options?.center || [-77.0428, -12.0464],
        zoom: options?.zoom || 10,
        attributionControl: true
      })

      // Agregar controles de navegación (sin rotar para evitar confusión)
      mapInstance.value.addControl(
        new maplibregl.NavigationControl({ showCompass: false }),
        'top-right'
      )

      // Agregar escala
      mapInstance.value.addControl(
        new maplibregl.ScaleControl(),
        'bottom-left'
      )

      return mapInstance.value
    }
    return null
  }

  const initDrawControl = () => {
    if (!mapInstance.value || drawInstance.value) return

    const MapboxDraw = (window as any).MapboxDraw

    drawInstance.value = new MapboxDraw({
      displayControlsDefault: false,
      controls: {
        polygon: true,
        trash: true
      },
      defaultMode: 'simple_select',
      styles: [
        {
          id: 'gl-draw-polygon-fill',
          type: 'fill',
          filter: ['all', ['==', '$type', 'Polygon']],
          paint: {
            'fill-color': '#3388ff',
            'fill-opacity': 0.3
          }
        },
        {
          id: 'gl-draw-polygon-stroke',
          type: 'line',
          filter: ['all', ['==', '$type', 'Polygon']],
          paint: {
            'line-color': '#3388ff',
            'line-width': 2
          }
        },
        {
          id: 'gl-draw-polygon-vertex',
          type: 'circle',
          filter: ['all', ['==', 'meta', 'vertex']],
          paint: {
            'circle-radius': 5,
            'circle-color': '#3388ff'
          }
        }
      ]
    })

    // Add draw control to bottom-left position
    mapInstance.value.addControl(drawInstance.value as any, 'bottom-left')

    mapInstance.value.on('draw.create', handleDrawCreate)
    mapInstance.value.on('draw.update', handleDrawUpdate)
    mapInstance.value.on('draw.delete', handleDrawDelete)
  }

  const handleDrawCreate = (e: any) => {
    const data = drawInstance.value?.getAll()
    if (data && data.features.length > 0) {
      const feature = data.features[0]
      const coordinates = feature.geometry.coordinates[0]

      const lngs = coordinates.map((coord: number[]) => coord[0])
      const lats = coordinates.map((coord: number[]) => coord[1])

      const bounds = {
        min_lon: Math.min(...lngs),
        max_lon: Math.max(...lngs),
        min_lat: Math.min(...lats),
        max_lat: Math.max(...lats)
      }

      analysisStore.setSelectedBounds(bounds)
      analysisStore.setSelectedPolygon(coordinates)
    }
  }

  const handleDrawUpdate = (e: any) => {
    handleDrawCreate(e)
  }

  const handleDrawDelete = () => {
    analysisStore.clearSelectedBounds()
  }

  const startDrawing = () => {
    if (drawInstance.value) {
      drawInstance.value.changeMode('draw_polygon')
      analysisStore.setDrawingMode(true)
    }
  }

  const stopDrawing = () => {
    if (drawInstance.value) {
      drawInstance.value.changeMode('simple_select')
      analysisStore.setDrawingMode(false)
    }
  }

  const clearSelection = () => {
    if (drawInstance.value) {
      drawInstance.value.deleteAll()
    }
    analysisStore.clearSelectedBounds()
    analysisStore.clearResults()
  }

  const addResultsLayer = (geojson: any) => {
    if (!mapInstance.value) return

    const map = mapInstance.value

    if (map.getLayer('results-fill')) {
      map.removeLayer('results-fill')
    }
    if (map.getLayer('results-outline')) {
      map.removeLayer('results-outline')
    }
    if (map.getSource('results')) {
      map.removeSource('results')
    }

    map.addSource('results', {
      type: 'geojson',
      data: geojson
    })

    map.addLayer({
      id: 'results-fill',
      type: 'fill',
      source: 'results',
      paint: {
        'fill-color': [
          'match',
          ['get', 'stress_level'],
          'high', '#dc3545',
          'medium', '#ffc107',
          'low', '#28a745',
          '#3388ff'
        ],
        'fill-opacity': 0.4
      }
    })

    map.addLayer({
      id: 'results-outline',
      type: 'line',
      source: 'results',
      paint: {
        'line-color': [
          'match',
          ['get', 'stress_level'],
          'high', '#dc3545',
          'medium', '#ffc107',
          'low', '#28a745',
          '#3388ff'
        ],
        'line-width': 2
      }
    })

    map.on('click', 'results-fill', (e: any) => {
      if (e.features && e.features.length > 0) {
        const feature = e.features[0]
        const props = feature.properties

        const maplibregl = (window as any).maplibregl
        new maplibregl.Popup()
          .setLngLat(e.lngLat)
          .setHTML(`
            <strong>Nivel de estrés:</strong> ${props.stress_level || 'N/A'}<br>
            <strong>NDVI promedio:</strong> ${props.ndvi_mean?.toFixed(3) || 'N/A'}
          `)
          .addTo(map)
      }
    })

    map.on('mouseenter', 'results-fill', () => {
      map.getCanvas().style.cursor = 'pointer'
    })
    map.on('mouseleave', 'results-fill', () => {
      map.getCanvas().style.cursor = ''
    })
  }

  const clearResultsLayer = () => {
    if (!mapInstance.value) return

    const map = mapInstance.value

    if (map.getLayer('results-fill')) {
      map.removeLayer('results-fill')
    }
    if (map.getLayer('results-outline')) {
      map.removeLayer('results-outline')
    }
    if (map.getSource('results')) {
      map.removeSource('results')
    }
  }

  const addRasterLayer = (layerId: string, imageBase64: string, bounds: any) => {
    if (!mapInstance.value) return

    const map = mapInstance.value

    // Remove existing layer if present
    if (map.getLayer(layerId)) {
      map.removeLayer(layerId)
    }
    if (map.getSource(layerId)) {
      map.removeSource(layerId)
    }

    // Create image URL from base64
    const imageUrl = `data:image/png;base64,${imageBase64}`

    // Add source
    map.addSource(layerId, {
      type: 'image',
      url: imageUrl,
      coordinates: [
        [bounds.min_lon, bounds.max_lat], // top-left
        [bounds.max_lon, bounds.max_lat], // top-right
        [bounds.max_lon, bounds.min_lat], // bottom-right
        [bounds.min_lon, bounds.min_lat]  // bottom-left
      ]
    })

    // Find the first symbol layer to insert raster layers before it
    // This ensures raster layers are below vector layers (results)
    const layers = map.getStyle().layers
    let firstSymbolId: string | undefined
    for (const layer of layers) {
      if (layer.type === 'symbol' || layer.id === 'results-fill' || layer.id === 'results-outline') {
        firstSymbolId = layer.id
        break
      }
    }

    // Add layer (initially hidden) BEFORE results layers
    map.addLayer({
      id: layerId,
      type: 'raster',
      source: layerId,
      paint: {
        'raster-opacity': 0.7
      },
      layout: {
        visibility: 'none'
      }
    }, firstSymbolId)
  }

  const toggleRasterLayer = (layerId: string, visible: boolean) => {
    if (!mapInstance.value) return

    const map = mapInstance.value

    if (map.getLayer(layerId)) {
      map.setLayoutProperty(
        layerId,
        'visibility',
        visible ? 'visible' : 'none'
      )
    }
  }

  const clearRasterLayers = () => {
    if (!mapInstance.value) return

    const map = mapInstance.value
    const rasterLayers = ['raster-rgb', 'raster-false-color', 'raster-ndvi']

    rasterLayers.forEach(layerId => {
      if (map.getLayer(layerId)) {
        map.removeLayer(layerId)
      }
      if (map.getSource(layerId)) {
        map.removeSource(layerId)
      }
    })
  }

  return {
    mapInstance: readonly(mapInstance),
    drawInstance: readonly(drawInstance),
    isDrawing,
    selectedBounds,
    initMap,
    initDrawControl,
    startDrawing,
    stopDrawing,
    clearSelection,
    addResultsLayer,
    clearResultsLayer,
    addRasterLayer,
    toggleRasterLayer,
    clearRasterLayers
  }
}
