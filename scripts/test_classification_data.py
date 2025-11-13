"""
Test script to verify all data is available for classification notebook.
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def check_file(path, description):
    """Check if file exists and print info"""
    if path.exists():
        print(f"✓ {description}: {path}")
        if path.suffix == '.npy':
            data = np.load(path)
            print(f"  Shape: {data.shape}, dtype: {data.dtype}")
        elif path.suffix == '.npz':
            data = np.load(path)
            print(f"  Keys: {list(data.keys())}")
            for key in data.keys():
                print(f"    {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        return True
    else:
        print(f"✗ {description}: NOT FOUND - {path}")
        return False

def main():
    print("=" * 70)
    print("VERIFICACIÓN DE DATOS PARA US-010: Clasificación Semántica")
    print("=" * 70)
    
    all_ok = True
    
    # Check Mexicali data
    print("\n--- MEXICALI ---")
    mexicali_path = Path("data/processed/mexicali")
    all_ok &= check_file(mexicali_path / "ndvi.npy", "NDVI")
    all_ok &= check_file(mexicali_path / "mgrg_segmentation.npy", "MGRG Segmentation")
    
    embeddings_path = Path("img/sentinel2/embeddings")
    all_ok &= check_file(embeddings_path / "mexicali_prithvi.npz", "Prithvi Embeddings")
    
    dw_path = Path("data/dynamic_world")
    all_ok &= check_file(dw_path / "mexicali_dw.tif", "Dynamic World")
    
    # Check Bajio data
    print("\n--- BAJÍO ---")
    bajio_path = Path("data/processed/bajio")
    all_ok &= check_file(bajio_path / "ndvi.npy", "NDVI")
    all_ok &= check_file(bajio_path / "mgrg_segmentation.npy", "MGRG Segmentation")
    all_ok &= check_file(embeddings_path / "bajio_prithvi.npz", "Prithvi Embeddings")
    all_ok &= check_file(dw_path / "bajio_dw.tif", "Dynamic World")
    
    # Check Sinaloa data
    print("\n--- SINALOA ---")
    sinaloa_path = Path("data/processed/sinaloa")
    all_ok &= check_file(sinaloa_path / "ndvi.npy", "NDVI")
    all_ok &= check_file(sinaloa_path / "mgrg_segmentation.npy", "MGRG Segmentation")
    all_ok &= check_file(embeddings_path / "sinaloa_prithvi.npz", "Prithvi Embeddings")
    all_ok &= check_file(dw_path / "sinaloa_dw.tif", "Dynamic World")
    
    # Check classifier module
    print("\n--- MÓDULO DE CLASIFICACIÓN ---")
    try:
        from src.classification.zero_shot_classifier import (
            SemanticClassifier,
            LAND_COVER_CLASSES,
            CLASS_COLORS,
            cross_validate_with_dynamic_world
        )
        print("✓ Módulo zero_shot_classifier importado correctamente")
        print(f"  Clases disponibles: {list(LAND_COVER_CLASSES.values())}")
    except Exception as e:
        print(f"✗ Error importando módulo: {e}")
        all_ok = False
    
    # Check output directory
    print("\n--- DIRECTORIO DE SALIDA ---")
    output_path = Path("img/results/classification")
    if not output_path.exists():
        print(f"⚠ Creando directorio: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directorio de salida: {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("✓ TODOS LOS DATOS ESTÁN DISPONIBLES")
        print("✓ El notebook puede ejecutarse correctamente")
    else:
        print("✗ FALTAN ALGUNOS DATOS")
        print("✗ Revisa los archivos faltantes antes de ejecutar el notebook")
    print("=" * 70)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
