"""Test loading model from notebook directory."""
import sys
import os

# Simulate running from notebooks/experimental
os.chdir('notebooks/experimental')
sys.path.append('../..')

from src.models.prithvi_loader import load_prithvi_model, get_model_info

print("Testing model loading from notebooks/experimental/...")
encoder = load_prithvi_model(use_simple_model=False)
info = get_model_info(encoder)

print(f"\nModelo cargado exitosamente desde notebooks/experimental")
print(f"   Parámetros: {info['total_parameters']:,}")
print(f"   Device: {info['device']}")
print(f"\nLas notebooks ahora funcionarán correctamente!")
