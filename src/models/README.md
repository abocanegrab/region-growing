# Prithvi Foundation Model

This module provides utilities for loading and using the Prithvi-EO-1.0-100M foundation model for extracting semantic embeddings from Sentinel-2 imagery.

## Overview

Prithvi is a 100M parameter Vision Transformer pre-trained on Harmonized Landsat Sentinel-2 (HLS) imagery by NASA and IBM. It can extract rich semantic features from multispectral satellite imagery without task-specific fine-tuning.

## Model Details

- **Name:** Prithvi-EO-1.0-100M
- **Architecture:** Vision Transformer (ViT)
- **Parameters:** ~100 million
- **Input:** 6-band HLS imagery (B02, B03, B04, B8A, B11, B12)
- **Output:** 256-dimensional embeddings per spatial location
- **Pre-training:** Masked autoencoding on HLS imagery

## Quick Start

### Installation

Dependencies are already in `pyproject.toml`. Just run:
```bash
poetry install
```

### Basic Usage

```python
from src.models.prithvi_loader import load_prithvi_model
import torch

# Option 1: Simplified model (fast, for development)
encoder = load_prithvi_model(use_simple_model=True)

# Option 2: Real Prithvi model (requires download, for production)
encoder = load_prithvi_model(use_simple_model=False)

# Prepare HLS image (6 bands: B02, B03, B04, B8A, B11, B12)
hls_image = torch.randn(1, 6, 224, 224)

# Extract embeddings
embeddings = encoder(hls_image)  # Output: (1, 256, 14, 14)

print(f"Input shape: {hls_image.shape}")
print(f"Output shape: {embeddings.shape}")
```

### Model Comparison

| Feature | Simple Model | Real Prithvi |
|---------|-------------|--------------|
| **Parameters** | ~86M | ~1.4M (encoder) |
| **Download** | No | Yes (432 MB) |
| **Pre-trained** | No | Yes (NASA/IBM) |
| **Use Case** | Development | Production |
| **Load Time** | <1s | ~5s |

**When to use each:**
- `use_simple_model=True`: Fast iteration, testing, development
- `use_simple_model=False`: Production, research, best results

### With Real Sentinel-2 Data

```python
from src.utils.sentinel_download import download_sentinel2_bands
from src.models.prithvi_loader import load_prithvi_model, normalize_hls_image
import torch
import numpy as np

# Download Sentinel-2 data (US-003)
bbox = {'min_lon': -100.0, 'min_lat': 40.0, 'max_lon': -99.9, 'max_lat': 40.1}
data = download_sentinel2_bands(bbox, config)

# Stack HLS bands in correct order
hls_bands = np.stack([
    data['bands']['B02'],  # Blue (10m)
    data['bands']['B03'],  # Green (10m)
    data['bands']['B04'],  # Red (10m)
    data['bands']['B8A'],  # NIR Narrow (20m, needs resampling)
    data['bands']['B11'],  # SWIR1 (20m, needs resampling)
    data['bands']['B12'],  # SWIR2 (20m, needs resampling)
], axis=0)

# Convert to tensor
hls_tensor = torch.from_numpy(hls_bands).unsqueeze(0).float()

# Normalize
hls_tensor = normalize_hls_image(hls_tensor, method='standardize')

# Load model and extract embeddings
encoder = load_prithvi_model(use_simple_model=True)
embeddings = encoder(hls_tensor)

# Convert to numpy
embeddings_np = embeddings.squeeze(0).permute(1, 2, 0).cpu().numpy()
print(f"Embeddings shape: {embeddings_np.shape}")  # (H, W, 256)
```

## HLS Band Order (CRITICAL)

Prithvi expects bands in this **exact** order:

1. **B02** - Blue (490 nm) - 10m
2. **B03** - Green (560 nm) - 10m
3. **B04** - Red (665 nm) - 10m
4. **B8A** - NIR Narrow (865 nm) - 20m ⚠️ (NOT B08!)
5. **B11** - SWIR1 (1610 nm) - 20m
6. **B12** - SWIR2 (2190 nm) - 20m

**Important:** Use B8A (20m), not B08 (10m). They are different bands!

## Resampling 20m Bands

Bands B8A, B11, B12 are at 20m resolution and must be resampled to 10m:

```python
from scipy.ndimage import zoom

# Resample 20m band to 10m (bilinear interpolation)
b8a_10m = zoom(b8a_20m, 2, order=1)
```

## Normalization

Always normalize input images:

```python
from src.models.prithvi_loader import normalize_hls_image

# Per-image normalization (recommended)
normalized = normalize_hls_image(image, method='standardize')
```

## Testing

### Run inference tests:
```bash
poetry run python scripts/test_prithvi_inference.py
```

### Run unit tests:
```bash
poetry run pytest tests/unit/test_prithvi_loader.py -v
```

### With coverage:
```bash
poetry run pytest tests/unit/test_prithvi_loader.py --cov=src.models --cov-report=html
```

## API Reference

### `load_prithvi_model(model_path, device, use_simple_model)`

Load Prithvi model for inference.

**Parameters:**
- `model_path` (str, optional): Path to model weights
- `device` (str, optional): 'cuda' or 'cpu' (auto-detects if None)
- `use_simple_model` (bool): Use simplified model for testing

**Returns:**
- `PrithviEncoder`: Model ready for inference

### `PrithviEncoder.forward(x)`

Extract embeddings from HLS imagery.

**Parameters:**
- `x` (torch.Tensor): Input tensor (B, 6, H, W)

**Returns:**
- `torch.Tensor`: Embeddings (B, 256, H', W')

### `normalize_hls_image(image, method)`

Normalize HLS image for inference.

**Methods:**
- `'standardize'`: Zero mean, unit variance (recommended)
- `'minmax'`: Scale to [0, 1]
- `'clip'`: Clip to [0, 1] then standardize

### `interpolate_embeddings(embeddings, target_size, mode)`

Interpolate embeddings to match original resolution.

**Parameters:**
- `embeddings` (torch.Tensor): Embeddings (B, C, H', W')
- `target_size` (tuple): Target size (H, W)
- `mode` (str): 'bilinear' or 'nearest'

**Returns:**
- `torch.Tensor`: Interpolated embeddings (B, C, H, W)

## Troubleshooting

### Model Not Found

```
FileNotFoundError: Prithvi model not found
```

**Solution:** For now, use the simple model for testing:
```python
encoder = load_prithvi_model(use_simple_model=True)
```

### Wrong Band Order

```
RuntimeError: Expected 6 channels
```

**Solution:** Ensure bands are stacked in correct HLS order (B02, B03, B04, B8A, B11, B12).

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Use CPU: `load_prithvi_model(device='cpu')`
2. Reduce image size
3. Clear cache: `torch.cuda.empty_cache()`

## Performance

### GPU Memory Usage

- Simple model: ~0.34 GB
- Full Prithvi model: ~0.4-1 GB (estimated)
- Fits comfortably in 8GB VRAM (RTX 4070)

### Inference Time (Simple Model on RTX 4070)

- 224x224 image: ~0.007s per image
- Batch of 4: ~0.000s per image (batching is efficient)

## References

### Papers

- **Jakubik et al. (2024).** Foundation models for generalist geospatial AI. arXiv:2310.18660
  - https://arxiv.org/abs/2310.18660
  - Official Prithvi paper

- **Claverie et al. (2018).** The Harmonized Landsat and Sentinel-2 surface reflectance data set.
  - https://doi.org/10.1016/j.rse.2018.09.002
  - HLS specification

### Resources

- **HuggingFace Model:** https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
- **Prithvi GitHub:** https://github.com/NASA-IMPACT/Prithvi-EO-1.0
- **HLS User Guide:** https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf

## License

Prithvi model is released under Apache 2.0 license by IBM and NASA.

---

**Project:** Sistema Híbrido de Detección de Estrés Vegetal  
**Team:** Equipo 24 - Region Growing  
**US:** US-005 - Prithvi Model Integration  
**Version:** 1.0.0
