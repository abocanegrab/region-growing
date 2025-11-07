# Backend FastAPI - Guía de Inicio Rápido

Guía rápida para poner en marcha el backend FastAPI en menos de 5 minutos.

---

## 📋 Prerequisitos

- **Python 3.11+** (recomendado 3.12)
- **Poetry** (gestor de dependencias)
- **Credenciales Sentinel Hub** (registro gratuito)

---

## ⚡ Instalación Rápida (3 pasos)

### 1. Instalar Poetry

**Windows PowerShell:**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

**Linux/macOS:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Instalar Dependencias del Proyecto

```bash
cd backend
poetry install
```

Esto creará automáticamente un entorno virtual y instalará todas las dependencias desde `poetry.lock`.

### 3. Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tu editor favorito
nano .env  # o code .env, vim .env, etc.
```

**Mínimo requerido en `.env`:**
```env
SENTINEL_HUB_CLIENT_ID=tu-client-id-aqui
SENTINEL_HUB_CLIENT_SECRET=tu-client-secret-aqui
```

---

## 🚀 Ejecutar el Servidor

### Modo Desarrollo (con auto-reload)

```bash
poetry run python app.py
```

O con uvicorn directamente:
```bash
poetry run uvicorn app.main:app --reload --port 8000
```

### Modo Producción

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## ✅ Verificar Instalación

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Respuesta esperada:**
```json
{
  "status": "healthy",
  "version": "2.0.1",
  "timestamp": "2025-11-07T..."
}
```

### 2. Documentación Interactiva

Abrir en navegador:
- **Swagger UI:** http://localhost:8000/api/docs
- **ReDoc:** http://localhost:8000/api/redoc

### 3. Ejecutar Tests

```bash
poetry run pytest tests/ -v
```

**Resultado esperado:** 13/13 tests passing

---

## 📦 Obtener Credenciales Sentinel Hub

1. **Registrarse:** https://apps.sentinel-hub.com/dashboard/
2. **Crear cuenta gratuita** (incluye cuota mensual)
3. **User Settings → OAuth clients → New OAuth client**
4. **Copiar Client ID y Client Secret** al archivo `.env`

---

## 🔧 Configuración Adicional (Opcional)

### Logging

```env
# Nivel de logs (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Formato de logs
LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Timeouts

```env
# Timeout para llamadas a Sentinel Hub API (segundos)
SENTINEL_HUB_TIMEOUT=30

# Timeout para análisis completo (segundos)
ANALYSIS_TIMEOUT=60
```

### CORS

```env
# Orígenes permitidos (separados por comas)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

---

## 🧪 Testing

### Ejecutar todos los tests

```bash
poetry run pytest tests/ -v
```

### Ejecutar con cobertura

```bash
poetry run pytest --cov=app --cov-report=html
```

Ver reporte: `htmlcov/index.html`

### Ejecutar solo health tests

```bash
poetry run pytest tests/test_health.py -v
```

---

## 🐛 Troubleshooting

### Error: "No module named 'sentinelhub'"

**Solución:**
```bash
poetry install
```

### Error: "Sentinel Hub credentials not configured"

**Solución:** Verificar que `.env` tenga las credenciales correctas.

### Puerto 8000 ya en uso

**Solución:**
```bash
# Usar otro puerto
poetry run uvicorn app.main:app --port 8001
```

### Tests fallan por timeout

**Solución:** Aumentar timeout en `.env`:
```env
ANALYSIS_TIMEOUT=120
```

---

## 📚 Próximos Pasos

1. **Explorar API:** http://localhost:8000/api/docs
2. **Probar endpoints** con Swagger UI
3. **Ver logs** estructurados en consola
4. **Revisar documentación completa:** [docs/us-resolved/us-001.md](us-resolved/us-001.md)

---

## 🎯 Endpoints Principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/analysis/test` | Test conexión Sentinel Hub |
| POST | `/api/analysis/analyze` | Análisis de estrés vegetal |
| GET | `/api/docs` | Documentación Swagger |

---

## 💡 Tips

- **Auto-reload:** Usa `--reload` en desarrollo para ver cambios inmediatamente
- **Logs detallados:** Cambia `LOG_LEVEL=DEBUG` para más información
- **Timeout ajustable:** Modifica `ANALYSIS_TIMEOUT` según tu conexión
- **Tests rápidos:** Usa `-k "not sentinel"` para omitir tests que llaman API externa

---

## 📞 Ayuda

- **Documentación completa:** [docs/us-resolved/us-001.md](us-resolved/us-001.md)
- **README Backend:** [backend/README.md](../backend/README.md)
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Poetry Docs:** https://python-poetry.org/docs/

---

**¿Listo?** 🚀 Ahora puedes empezar a hacer requests al backend!
