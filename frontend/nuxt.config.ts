// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: true },

  modules: [
    '@pinia/nuxt'
  ],

  css: [
    'maplibre-gl/dist/maplibre-gl.css',
    '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css',
    '~/assets/css/main.css'
  ],


  app: {
    head: {
      title: 'Sistema de Detección de Estrés Vegetal',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        {
          name: 'description',
          content: 'Análisis mediante Region Growing sobre imágenes Sentinel-2'
        }
      ]
    }
  },

  ssr: true,

  typescript: {
    strict: false,
    typeCheck: false
  },

  compatibilityDate: '2025-07-15',

  runtimeConfig: {
    public: {
      apiBase: 'http://localhost:8070'
    }
  }
})