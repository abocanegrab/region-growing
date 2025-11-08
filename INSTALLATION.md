# Installation Guide

## Quick Start

### Windows
```bash
.\setup.bat
```

### Linux/Mac
```bash
chmod +x setup.sh
./setup.sh
```

## Manual Installation

### 1. Install Poetry
```bash
# Linux/Mac
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### 2. Install Dependencies
```bash
# Install all dependencies including PyTorch with CUDA 12.9
poetry install
```

## PyTorch Installation

The project is configured to automatically install PyTorch with CUDA 12.9 support via Poetry:
- ✅ PyTorch 2.8.0+ with CUDA 12.9 from official PyTorch repository
- ✅ No manual installation needed - `poetry install` handles everything
- ✅ Configured via `[[tool.poetry.source]]` in pyproject.toml

**Note:** If you don't have a CUDA-compatible GPU, PyTorch will still install but will use CPU mode. To verify CUDA availability:

```bash
poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Verify Installation

```bash
# Activate poetry environment
poetry shell

# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check all imports
python -c "from src.utils.sentinel_download import download_sentinel2_bands; print('✅ src/ imports work!')"
```

## Configuration

1. Copy environment file:
```bash
cp backend/.env.example backend/.env
```

2. Edit `backend/.env` and add your Sentinel Hub credentials:
```env
SENTINEL_HUB_CLIENT_ID=your-client-id-here
SENTINEL_HUB_CLIENT_SECRET=your-client-secret-here
```

## Running the Project

### Backend (FastAPI)
```bash
poetry run python backend/app.py
# or
poetry run uvicorn backend.app.main:app --reload
```

### Jupyter Notebooks
```bash
poetry run jupyter notebook notebooks/
```

### Tests
```bash
# Run all tests
poetry run pytest tests/

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### CUDA Issues

If PyTorch doesn't detect your GPU:

1. Check NVIDIA driver (should be CUDA 12.9 compatible):
```bash
nvidia-smi
```

2. Verify PyTorch installation:
```bash
poetry run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

3. If needed, reinstall PyTorch:
```bash
poetry lock --no-update
poetry install --no-cache
```

### Import Errors

If you get import errors for `src/` modules:

```bash
# Reinstall project in editable mode
poetry install
```

### Poetry Lock Issues

If you encounter lock file issues:

```bash
poetry lock --no-update
poetry install
```

## System Requirements

- Python 3.11-3.13 (Python 3.14 not supported by PyTorch yet)
- Poetry 1.7+
- NVIDIA GPU with CUDA 12.9+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB recommended)
- 5GB disk space for dependencies

**Note:** If you have Python 3.14, you'll need to use Python 3.12 or 3.13:
```bash
poetry env use C:\Users\YOUR_USER\AppData\Local\Programs\Python\Python312\python.exe
```
