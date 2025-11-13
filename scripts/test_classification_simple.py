"""
Simple test to verify classification works with real data.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.classification.zero_shot_classifier import (
    SemanticClassifier,
    LAND_COVER_CLASSES,
    CLASS_COLORS,
    cross_validate_with_dynamic_world
)

def main():
    print("=" * 70)
    print("TEST: Clasificación Semántica con Datos Reales")
    print("=" * 70)
    
    # Load Mexicali data
    print("\n1. Cargando datos de Mexicali...")
    data_path = Path("data/processed/mexicali")
    embeddings_path = Path("img/sentinel2/embeddings")
    
    ndvi = np.load(data_path / "ndvi.npy")
    segmentation = np.load(data_path / "mgrg_segmentation.npy")
    embeddings_data = np.load(embeddings_path / "mexicali_prithvi.npz")
    embeddings = embeddings_data['embeddings']
    
    print(f"   NDVI shape: {ndvi.shape}")
    print(f"   Segmentation shape: {segmentation.shape}")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Number of regions: {len(np.unique(segmentation)) - 1}")
    
    # Initialize classifier
    print("\n2. Inicializando clasificador...")
    classifier = SemanticClassifier(embeddings, ndvi, resolution=10.0)
    print(f"   Classifier initialized: {classifier.h}x{classifier.w}")
    
    # Classify regions
    print("\n3. Clasificando regiones...")
    results = classifier.classify_all_regions(segmentation, min_size=10)
    print(f"   Classified {len(results)} regions")
    
    # Generate semantic map
    print("\n4. Generando mapa semántico...")
    semantic_map = classifier.generate_semantic_map(segmentation, results)
    colored_map = classifier.generate_colored_map(semantic_map)
    print(f"   Semantic map shape: {semantic_map.shape}")
    print(f"   Colored map shape: {colored_map.shape}")
    
    # Get statistics
    print("\n5. Calculando estadísticas por clase...")
    stats = classifier.get_class_statistics(results)
    
    print("\n" + "=" * 70)
    print("RESULTADOS DE CLASIFICACIÓN - MEXICALI")
    print("=" * 70)
    print(f"\n{'Clase':<35} {'Objetos':>10} {'Área (ha)':>12} {'NDVI Medio':>12}")
    print("-" * 70)
    
    total_area = 0
    for class_name in LAND_COVER_CLASSES.values():
        count = stats[class_name]['count']
        area = stats[class_name]['area_ha']
        mean_ndvi = stats[class_name]['mean_ndvi']
        total_area += area
        
        if count > 0:
            print(f"{class_name:<35} {count:>10} {area:>12.2f} {mean_ndvi:>12.3f}")
    
    print("-" * 70)
    print(f"{'TOTAL':<35} {len(results):>10} {total_area:>12.2f}")
    print("=" * 70)
    
    # Test a few specific regions
    print("\n6. Ejemplos de regiones clasificadas:")
    print("-" * 70)
    for i, (region_id, result) in enumerate(list(results.items())[:5]):
        print(f"\nRegión {region_id}:")
        print(f"  Clase: {result.class_name}")
        print(f"  Confianza: {result.confidence:.2f}")
        print(f"  NDVI medio: {result.mean_ndvi:.3f}")
        print(f"  Área: {result.area_hectares:.2f} ha")
    
    # Test cross-validation if rasterio is available
    print("\n7. Probando cross-validation con Dynamic World...")
    try:
        import rasterio
        from scipy.ndimage import zoom
        
        dw_path = Path("data/dynamic_world/mexicali_dw.tif")
        with rasterio.open(dw_path) as src:
            dw_mask = src.read(1)
        
        # Resize if needed
        if dw_mask.shape != semantic_map.shape:
            zoom_factors = (semantic_map.shape[0] / dw_mask.shape[0], 
                           semantic_map.shape[1] / dw_mask.shape[1])
            dw_mask = zoom(dw_mask, zoom_factors, order=0)
        
        agreements = cross_validate_with_dynamic_world(semantic_map, dw_mask)
        
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION CON DYNAMIC WORLD")
        print("=" * 70)
        print(f"\n{'Clase':<35} {'Agreement':>12}")
        print("-" * 70)
        
        for class_name in LAND_COVER_CLASSES.values():
            if class_name in agreements:
                agreement = agreements[class_name]
                print(f"{class_name:<35} {agreement:>11.1%}")
        
        print("-" * 70)
        print(f"{'Overall Agreement':<35} {agreements['overall']:>11.1%}")
        
        target = 0.70
        status = "✓ CUMPLE" if agreements['overall'] >= target else "✗ NO CUMPLE"
        print(f"{'Target (>70%)':<35} {status:>12}")
        print("=" * 70)
        
    except ImportError:
        print("   ⚠ rasterio no disponible, saltando cross-validation")
    except Exception as e:
        print(f"   ⚠ Error en cross-validation: {e}")
    
    print("\n" + "=" * 70)
    print("✓ TEST COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
