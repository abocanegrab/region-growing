# Guía para Agentes de IA - Proyecto de MLOps

Este archivo contiene instrucciones y buenas prácticas para agentes de IA que trabajen en proyectos de MLOps.

---

## 🎯 Contexto del Proyecto

**Nombre**: Sistema Híbrido de Detección de Estrés Vegetal
**Tipo**: Sistema Híbrido de Segmentación Semántico-Espectral
**Stack**: Python 3.12+, Poetry, Nuxt3, MLflow, FastAPI, Opencv, Pytorch
**Objetivo**: Comparación de métodos de segmentación con enfoque innovador  

---

## 📋 Buenas Prácticas Establecidas

### 1. Idioma y Documentación

#### ✅ Notebooks
- **Texto explicativo**: SIEMPRE en español
- **Código**: SIEMPRE en inglés
- **Comentarios en código**: En inglés
- **Markdown cells**: En español

**Ejemplo correcto**:
```python
# Notebook cell (Markdown)
## 1. Análisis Exploratorio de Datos

Este análisis explora los patrones de los datos...

# Notebook cell (Code)
# Load data and perform initial exploration
df = pl.read_parquet("data/processed/dataset.parquet")
summary_stats = df.describe()
```

#### ✅ Código Python
- **Nombres de variables**: En inglés
- **Nombres de funciones**: En inglés
- **Docstrings**: En inglés (estilo Google)
- **Comentarios**: En inglés

#### ✅ Documentación
- **README.md**: En español
- **Documentación técnica**: En español
- **Docstrings en código**: En inglés
- **Comentarios inline**: En inglés

---

### 2. Estructura de Código

#### ✅ Funciones Reutilizables

**SIEMPRE** crear funciones reutilizables en `src/utils/` en lugar de código duplicado en notebooks.

**❌ Incorrecto** (código en notebook):
```python
# En notebook
import some_db_library
conn = some_db_library.connect("data/database.db")
df = conn.execute("SELECT * FROM my_table").to_polars()
conn.close()
```

**✅ Correcto** (usar función de utils):
```python
# En notebook
from src.utils.db_utils import quick_query
df = quick_query("SELECT * FROM my_table")
```

#### ✅ Organización de Utilidades

```
src/utils/
├── db_utils.py              # Funciones para la base de datos
├── data_cleaning.py         # Limpieza de datos
├── data_quality.py          # Análisis de calidad
├── feature_engineering.py   # Creación de características
├── visualization.py         # Visualizaciones
└── secrets.py               # Manejo de secretos
```

**Regla**: Si una función se usa más de una vez, debe estar en `src/utils/`.

---

### 3. Programación Orientada a Objetos

#### ✅ Transformers de Scikit-Learn

Para feature engineering, SIEMPRE usar clases que hereden de `BaseEstimator` y `TransformerMixin`:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    '''
    A custom transformer for a specific feature engineering task.
    
    Parameters
    ----------
    param_name : str, default='default_value'
        Description of the parameter.
    '''
    
    def __init__(self, param_name='default_value'):
        self.param_name = param_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Apply transformation logic here
        return X
```

**Beneficios**:
- ✅ Reutilizable en pipelines de sklearn
- ✅ Compatible con `fit()` y `transform()`
- ✅ Fácil de testear

---

### 4. Pipelines de Scikit-Learn

#### ✅ SIEMPRE usar pipelines

**❌ Incorrecto**:
```python
# Código suelto
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model = SomeModel()
model.fit(X_scaled, y_train)
```

**✅ Correcto**:
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SomeModel())
])
pipeline.fit(X_train, y_train)
```

---

### 5. Manejo de Nombres de Columnas

#### ✅ Columnas con Caracteres Especiales

Si el dataset tiene columnas con caracteres especiales, asegurarse de que las funciones de utilidad los manejen correctamente o usar el método de escape apropiado para el motor de base de datos/dataframe.

**Para queries SQL personalizados**, usar comillas dobles o el carácter de escape adecuado:
```python
df = quick_query('''
    SELECT "column-with-special-chars" as clean_name
    FROM my_table
''')
```

---

### 6. Testing

#### ✅ Estructura de Tests

```
tests/
├── unit/              # Tests unitarios
├── integration/       # Tests de integración
└── e2e/              # Tests end-to-end
```

#### ✅ Convenciones de Tests

- **Nombres**: `test_*.py`
- **Clases**: `Test*`
- **Funciones**: `test_*`
- **Coverage mínimo**: >70%

**Ejemplo**:
```python
import pytest

class TestMyUtils:
    '''Tests for utility functions.'''
    
    def test_some_function(self):
        '''Test a specific behavior.'''
        result = some_function()
        assert result is not None
```

---

### 7. Versionado de Datos y Modelos

#### ✅ Usar DVC

**NUNCA** commitear archivos grandes a Git. Usar DVC:

```bash
# ✅ Correcto
dvc add data/processed/my_dataset.parquet
git add data/processed/my_dataset.parquet.dvc
git commit -m "data: add processed dataset"

# ❌ Incorrecto
git add data/processed/my_dataset.parquet
```

#### ✅ Archivos que van con DVC

- ✅ Datasets (CSV, Parquet, etc.)
- ✅ Modelos entrenados (pkl, pth, h5)
- ✅ Archivos >1MB

#### ✅ Archivos que van con Git

- ✅ Código fuente
- ✅ Configs (<1KB)
- ✅ Documentación
- ✅ Tests

---

### 8. MLflow Experiment Tracking

#### ✅ SIEMPRE loggear experimentos

```python
import mlflow

with mlflow.start_run(run_name="my_experiment_name"):
    # Log parameters
    mlflow.log_params(model.get_params())
    
    # Train
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

---

### 9. Convenciones de Código

#### ✅ Formateo

- **Formatter**: Black (line-length=100)
- **Linter**: Ruff
- **Type checker**: MyPy (opcional)

```bash
# Antes de commitear
poetry run black .
poetry run ruff check .
```

#### ✅ Docstrings

**Estilo Google** para todas las funciones:

```python
def my_function(param1: str, param2: int) -> dict:
    '''
    Brief description of the function.
    
    Parameters
    ----------
    param1 : str
        Description of the first parameter.
    param2 : int
        Description of the second parameter.
        
    Returns
    -------
    dict
        Description of the returned value.
        
    Examples
    --------
    >>> result = my_function('test', 123)
    >>> print(result)
    '''
```

#### ✅ Type Hints

SIEMPRE usar type hints:

```python
# ✅ Correcto
def process_data(df: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
    pass

# ❌ Incorrecto
def process_data(df, threshold=0.5):
    pass
```

---

### 10. Estructura de Notebooks

#### ✅ Orden Estándar

```markdown
# 1. Título y Descripción (en español)

## 2. Imports
import sys
sys.path.append('../..')
from src.utils.db_utils import setup_database

## 3. Configuración
conn = setup_database(...)

## 4. Análisis
### 4.1 Sección 1
### 4.2 Sección 2

## 5. Conclusiones (en español)

## 6. Limpieza
conn.close()
```

#### ✅ Usar Funciones de Utils

**SIEMPRE** preferir funciones de `src/utils/` sobre código inline.

---

### 11. Manejo de Errores

#### ✅ Logging en lugar de prints

```python
import logging

logger = logging.getLogger(__name__)

# ✅ Correcto
logger.info("Processing data...")
logger.error(f"Failed to load file: {e}")

# ❌ Incorrecto
print("Processing data...")
print(f"Error: {e}")
```

#### ✅ Manejo de Excepciones

```python
# ✅ Correcto
try:
    df = load_data(path)
except FileNotFoundError:
    logger.error(f"File not found: {path}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise

# ❌ Incorrecto
try:
    df = load_data(path)
except:
    pass
```

---

### 12. Git Commits

#### ✅ Conventional Commits

```bash
# Formato: <type>(<scope>): <description>

# Tipos:
feat:     # Nueva funcionalidad
fix:      # Corrección de bug
docs:     # Cambios en documentación
style:    # Formateo (no afecta código)
refactor: # Refactorización
test:     # Agregar/modificar tests
chore:    # Tareas de mantenimiento

# Ejemplos:
git commit -m "feat(database): add setup_database function"
git commit -m "fix(utils): handle special characters in column names"
git commit -m "docs: update AGENTS_general.md with best practices"
```

---

### 13. Dependencias

#### ✅ Gestión con Poetry (o el manejador de paquetes del proyecto)

```bash
# Agregar dependencia
poetry add some_package

# Agregar dependencia de desarrollo
poetry add --group dev pytest

# Actualizar dependencias
poetry update

# NUNCA editar pyproject.toml manualmente para dependencias
```

---

### 14. Configuración de Notebooks

#### ✅ Setup Inicial

**SIEMPRE** incluir al inicio de notebooks:

```python
# Imports
import sys
sys.path.append('../..')  # Para importar desde src/

# Configuración de visualización
import warnings
warnings.filterwarnings('ignore')

# Configuración de librería de dataframes (ej. Polars)
import polars as pl
pl.Config.set_tbl_rows(20)
pl.Config.set_fmt_str_lengths(100)
```

---

### 15. Funciones Reutilizables Disponibles

Consultar el directorio `src/utils/` para ver las funciones disponibles y su documentación.

---

## 🚫 Anti-Patrones (Evitar)

### ❌ Código Duplicado
- **Solución**: Crear funciones reutilizables en `src/utils/`.

### ❌ Hardcoded Paths
- **Solución**: Usar `pathlib` y rutas relativas al proyecto.

### ❌ Magic Numbers
- **Solución**: Definir constantes con nombres descriptivos.

### ❌ Commits de Archivos Grandes
- **Solución**: Usar DVC para datos y modelos.

---

## 📚 Referencias Rápidas

### Estructura del Proyecto

```
[nombre_del_proyecto]/
├── src/
│   ├── data/           # Scripts de procesamiento
│   ├── features/       # Feature engineering (POO)
│   ├── models/         # Training y pipelines
│   ├── api/            # API backend
│   └── utils/          # Funciones reutilizables ⭐
├── notebooks/
│   ├── exploratory/    # EDA (texto en español)
│   └── experimental/   # Experimentos
├── tests/
│   ├── unit/          # Tests unitarios
│   ├── integration/   # Tests de integración
│   └── e2e/           # Tests end-to-end
├── data/              # Datos (versionados con DVC)
├── models/            # Modelos (versionados con DVC)
└── docs/              # Documentación
```

### Comandos Útiles

```bash
# Tests
poetry run pytest tests/ -v
poetry run pytest --cov=src --cov-report=html

# Formateo
poetry run black .
poetry run ruff check .

# DVC
dvc add data/file.csv
dvc push
dvc pull

# MLflow
poetry run mlflow ui

# Git
git status
git add .
git commit -m "type(scope): description"
git push
```

---

## ✅ Checklist para Nuevas Funcionalidades

Antes de considerar una tarea completada:

- [ ] Código en inglés, documentación en español/inglés según convención.
- [ ] Funciones reutilizables en `src/utils/` si aplica.
- [ ] Docstrings estilo Google.
- [ ] Type hints en todas las funciones.
- [ ] Tests unitarios (coverage >70%).
- [ ] Formateado con Black.
- [ ] Sin warnings de Ruff.
- [ ] Datos/modelos versionados con DVC.
- [ ] Experimentos loggeados en MLflow.
- [ ] Commit con Conventional Commits.
- [ ] Documentación actualizada.

---

## 🎓 Principios de Diseño

1. **DRY (Don't Repeat Yourself)**: Si lo usas dos veces, crea una función.
2. **KISS (Keep It Simple, Stupid)**: Código simple es mejor que código complejo.
3. **YAGNI (You Aren't Gonna Need It)**: No agregues funcionalidad que no necesitas ahora.
4. **Separation of Concerns**: Cada módulo tiene una responsabilidad clara.
5. **Code Reusability**: Funciones en utils, no código duplicado en notebooks.

---

## 🚨 Código Natural - Evitar Marcadores de IA

### ❌ NO usar elementos que delatan código generado por IA

**Evitar estos patrones típicos de IA**:

```python
# ❌ NO usar emojis en comentarios
# 🔍 Load the data
# ✅ Process the results
# 🚀 Train the model

# ❌ NO usar separadores decorativos
print("="*70)
print("---"*20)

# ❌ NO usar comentarios excesivamente descriptivos
# Step 1: Initialize the connection
# Step 2: Load the data

# ❌ NO usar banners ASCII
# ============================================
# SECTION: DATA LOADING
# ============================================
```

**✅ Usar código natural y profesional**:

```python
# Load and validate data
df = pl.read_parquet("data/processed/dataset.parquet")

# Calculate summary statistics
summary = df.describe()

# Train model with cross-validation
model.fit(X_train, y_train)
```

### Reglas Generales

1. **Sin emojis** en código o comentarios.
2. **Sin separadores decorativos** (=, -, *).
3. **Comentarios concisos** y técnicos, no narrativos.
4. **Mensajes de log simples** y directos.
5. **Usar markdown cells** para estructura en notebooks, no prints decorativos.
6. **No crear archivos md cada que haces algo** generar solo un md al final cuando se concluyan todas las tareas de la US y guardarlo en docs/us-resolved/us-XXX.md don XXX es el numero de us resuelta 

---

**Versión**: 1.0
**Última actualización**: Noviembre 2025
**Mantenido por**: Equipo 24

---

## 📞 Contacto

Si tienes dudas sobre estas prácticas, consulta:
- `docs/` - Documentación del proyecto
- `README.md` - Guía de inicio
